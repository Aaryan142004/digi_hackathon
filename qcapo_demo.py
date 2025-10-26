# qcapo_small.py — ~12 qubits; friendly to laptops, with mission constraints
# Works with:
#   qiskit==2.1.2
#   qiskit-optimization==0.7.0
#   qiskit-algorithms==0.4.0
#   qiskit-aer==0.17.1

import math
from typing import Dict, List, Optional

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import real_amplitudes
from qiskit_aer.primitives import SamplerV2

# -----------------------------  knobs --------------------------------
# Total qubits ≈ N*(|thr|+|pitch|+1) + |stage_nodes| + |launch_slots|
# Here: N=2, thr=2, pitch=2, coast=1  -> 2*(2+2+1)=10  + stage(1) + launch(1) = 12 qubits
N, dt = 2, 30.0
throttle_bins = ["t0", "t100"]     # 2 choices
pitch_bins    = ["p0", "p20"]      # 2 choices
launch_slots  = ["L1"]             # 1 choice (kept as a var for structure)
stage_nodes   = ["s_2"]            # 1 bit

# weights (fuel, time, safety)
w1, w2, w3 = 1.0, 0.01, 10.0

# Mission constraint toggles
REQUIRE_MIN_ONE_BURN = True     # enforce at least one t100 across the trajectory
NOT_ALL_COAST        = True     # forbid all nodes being coast simultaneously
MIN_TOTAL_FUEL: Optional[float] = None  # e.g., 100.0 to enforce minimum fuel use

# Aer / VQE settings
AER_METHOD = "matrix_product_state"   # memory-lean
SHOTS = 512
ANSATZ_REPS = 1
MAX_QUBITS = 20                       # guard for laptop-friendly runs

# -----------------------------  toy tables --------------------------------
fuel_rate = {"t0": 0, "t100": 100}
risk_dp_heat: Dict[int, Dict[str, float]] = {
    1: {"t0": 0, "t100": 8},
    2: {"t0": 0, "t100": 6},
}
debris_pitch_penalty = {(2, "p20"): 0.2}
coupling_penalty = 2.0  # penalize t100 with p0 at same node


def build_qp(w1: float, w2: float, w3: float) -> QuadraticProgram:
    qp = QuadraticProgram()

    # variables
    for n in range(1, N + 1):
        for t in throttle_bins:
            qp.binary_var(f"x_{n}_{t}")
        for p in pitch_bins:
            qp.binary_var(f"y_{n}_{p}")
        qp.binary_var(f"c_{n}")  # coast
    for s in stage_nodes:
        qp.binary_var(s)
    for L in launch_slots:
        qp.binary_var(L)

    # one-hot helpers
    def one_hot(names: List[str], name: str):
        qp.linear_constraint({v: 1 for v in names}, "==", 1, name=name)

    for n in range(1, N + 1):
        one_hot([f"x_{n}_{t}" for t in throttle_bins], f"onehot_throttle_{n}")
        one_hot([f"y_{n}_{p}" for p in pitch_bins],    f"onehot_pitch_{n}")
    if stage_nodes:
        one_hot(stage_nodes,  "onehot_stage_sep")
    if launch_slots:
        one_hot(launch_slots, "onehot_launch")

    # coast ⇒ throttle==t0 : c_n - x_{n,t0} == 0
    for n in range(1, N + 1):
        qp.linear_constraint({f"c_{n}": 1, f"x_{n}_t0": -1}, "==", 0, name=f"coast_implies_t0_{n}")

    # ---- mission constraints to avoid trivial all-coast ----
    if REQUIRE_MIN_ONE_BURN and "t100" in throttle_bins:
        qp.linear_constraint(
            {f"x_{n}_t100": 1 for n in range(1, N + 1)},
            ">=",
            1,
            name="min_one_burn",
        )

    if NOT_ALL_COAST:
        qp.linear_constraint(
            {f"c_{n}": 1 for n in range(1, N + 1)},
            "<=",
            N - 1,
            name="not_all_coast",
        )

    if MIN_TOTAL_FUEL is not None:
        # sum_n (fuel_rate * dt * x_{n,t100}) >= MIN_TOTAL_FUEL
        qp.linear_constraint(
            {f"x_{n}_t100": fuel_rate["t100"] * dt for n in range(1, N + 1)},
            ">=",
            float(MIN_TOTAL_FUEL),
            name="min_total_fuel",
        )

    # objective
    linear, quadratic = {}, {}

    # fuel
    for n in range(1, N + 1):
        for t in throttle_bins:
            linear[f"x_{n}_{t}"] = linear.get(f"x_{n}_{t}", 0.0) + w1 * (fuel_rate[t] * dt)

    # time
    for n in range(1, N + 1):
        linear[f"c_{n}"] = linear.get(f"c_{n}", 0.0) + w2 * 5.0
    for L, cost in {"L1": 10}.items():
        linear[L] = linear.get(L, 0.0) + w2 * cost

    # safety (linear + quadratic coupling)
    for n in range(1, N + 1):
        for t in throttle_bins:
            linear[f"x_{n}_{t}"] = linear.get(f"x_{n}_{t}", 0.0) + w3 * risk_dp_heat.get(n, {}).get(t, 0.0)
        if "t100" in throttle_bins and "p0" in pitch_bins:
            quadratic[(f"x_{n}_t100", f"y_{n}_p0")] = quadratic.get(
                (f"x_{n}_t100", f"y_{n}_p0"), 0.0
            ) + w3 * coupling_penalty

    for (n, p), pen in debris_pitch_penalty.items():
        linear[f"y_{n}_{p}"] = linear.get(f"y_{n}_{p}", 0.0) + w3 * pen

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def decode_solution(qp: QuadraticProgram, res) -> dict:
    sol = res.variables_dict
    sched = []
    for n in range(1, N + 1):
        t = [b for b in throttle_bins if sol.get(f"x_{n}_{b}", 0) > 0.5][0]
        p = [b for b in pitch_bins    if sol.get(f"y_{n}_{b}", 0) > 0.5][0]
        c = int(sol.get(f"c_{n}", 0)  > 0.5)
        sched.append({"node": n, "throttle": t, "pitch": p, "coast": c})

    # launch slot (robust to 0/1 length)
    if launch_slots:
        L_choices = [L for L in launch_slots if sol.get(L, 0) > 0.5]
        launch = L_choices[0] if L_choices else launch_slots[0]
    else:
        launch = None

    # stage sep node: parse the selected bit if present
    stage_val = None
    for s in stage_nodes:
        if sol.get(s, 0) > 0.5:
            try:
                stage_val = int(s.split("_")[1])
            except Exception:
                stage_val = s
            break

    return {"schedule": sched, "launch_slot": launch, "stage_sep_node": stage_val}


def toy_safety_metrics(decoded: dict) -> dict:
    rho0, H = 1.225, 8500.0
    q_score = heat_score = 0.0
    h = 1000.0; v = 0.0
    for step in decoded["schedule"]:
        thrust = {"t0": 0.0, "t100": 1.0}[step["throttle"]]
        pitch  = {"p0": 0.0, "p20": 20.0}[step["pitch"]]
        coast  = step["coast"]
        v += 200 * thrust - 10 * (1 if not coast else 0)
        h += 300 * math.sin(math.radians(pitch)) + 50 * coast
        rho = rho0 * math.exp(-h / H)
        q = 0.5 * rho * v**2
        heat = (rho**0.5) * (abs(v)**3) * 1e-10
        q_score += q; heat_score += heat
    return {"q_score": q_score, "heat_score": heat_score}


def main():
    global w1, w2, w3

    qp = build_qp(w1, w2, w3)
    num_qubits = qp.get_num_vars()
    print(f"Num vars/qubits: {num_qubits}  (guard ≤ {MAX_QUBITS})")
    if num_qubits > MAX_QUBITS:
        print(f"[guard] {num_qubits} qubits exceed guard {MAX_QUBITS}. "
              f"Reduce N/bins to continue.")
        return

    # ✅ Create sampler and set options AFTER instantiation (Aer 0.17.1)
    sampler = SamplerV2()
    sampler.options.shots = SHOTS
    sampler.options.method = AER_METHOD  # memory-lean backend

    # Build ansatz with explicit num_qubits
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=ANSATZ_REPS)

    vqe    = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA(maxiter=40))
    solver = MinimumEigenOptimizer(vqe)

    safety_q_limit, safety_heat_limit = 1e5, 1e3

    for it in range(4):
        print(f"\n=== Iteration {it+1} ===")
        qp = build_qp(w1, w2, w3)

        # If size changes, rebuild ansatz & solver
        if qp.get_num_vars() != num_qubits:
            num_qubits = qp.get_num_vars()
            print(f"Resizing ansatz → {num_qubits} qubits")
            ansatz = real_amplitudes(num_qubits=num_qubits, reps=ANSATZ_REPS)
            vqe    = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA(maxiter=40))
            solver = MinimumEigenOptimizer(vqe)

        res = solver.solve(qp)
        decoded = decode_solution(qp, res)
        metrics = toy_safety_metrics(decoded)

        print("Objective:", res.fval)
        print("Decoded plan:", decoded)
        print("Safety (toy):", metrics)

        if metrics["q_score"] > safety_q_limit or metrics["heat_score"] > safety_heat_limit:
            w3 *= 1.5
            print(f"Increasing safety weight → w3 = {w3}")
        else:
            print("Safety within limits ✓")
            break


if __name__ == "__main__":
    main()
