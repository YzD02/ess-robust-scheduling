"""Run grid-search stress tests for the ESS robust scheduling project.

This script is designed to work with the packaged project structure and the
baseline model / simulation modules already included in ``src``.

What this script does
---------------------
For each parameter combination in the grid, it performs the following steps:

1. Build one stress-test instance by scaling:
   - number of jobs
   - nominal processing-time level (mu_scale)
   - uncertainty level (sigma_scale)
   - robustness factor (k)

2. Solve the baseline Gurobi planning model:
   - assign jobs to daily bins
   - compute planned overtime if needed

3. Enforce the experiment rule discussed:
   - if the Gurobi run does *not* return a usable solution within 15 minutes,
     skip simulation for that case

4. If the case is accepted for simulation, evaluate the planned schedule with
   the cross-day breakdown simulator:
   - backlog-first execution
   - non-preemptive jobs
   - daily overtime cap
   - terminal backlog risk over the horizon

5. Save all results into a CSV file for later visualization

Why this file is separate
-------------------------
The goal is to make the full stress-test pipeline reproducible from a single
entry point, while keeping the modeling code inside ``src/models`` and the
execution validation inside ``src/simulation``.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

import pandas as pd
from gurobipy import GRB, Model, quicksum


# ============================================================
# 1. BASE PARAMETER PROFILES
# ============================================================
# These are the currently assumed baseline profiles in MINUTES.
# Later, once real shop-floor data is available, these can be replaced
# by calibrated empirical values without changing the rest of the script.

BASE_MU_A = {
    1: 88,  2: 80,  3: 96,  4: 84,  5: 92,
    6: 76,  7: 104, 8: 86,  9: 94, 10: 78,
    11: 100, 12: 87, 13: 90, 14: 79, 15: 102,
}

BASE_MU_B = {
    1: 62,  2: 54,  3: 68,  4: 58,  5: 60,
    6: 50,  7: 74,  8: 55,  9: 66, 10: 52,
    11: 72, 12: 57, 13: 59, 14: 51, 15: 70,
}

BASE_SIGMA_A = {
    1: 14, 2: 12, 3: 15, 4: 13, 5: 14,
    6: 11, 7: 17, 8: 12, 9: 15, 10: 11,
    11: 16, 12: 12, 13: 13, 14: 11, 15: 16,
}

BASE_SIGMA_B = {
    1: 18, 2: 16, 3: 20, 4: 17, 5: 18,
    6: 15, 7: 22, 8: 16, 9: 19, 10: 15,
    11: 21, 12: 16, 13: 17, 14: 15, 15: 21,
}


# ============================================================
# 2. GLOBAL EXPERIMENT SETTINGS
# ============================================================

N_DAYS = 20                   # 4-week horizon, 20 working days
C_STD = 480                   # 8-hour shift capacity, in minutes
COST_OT = 5.0
COST_FIX = 180.0
BIG_M = 2000.0

GUROBI_TIME_LIMIT_SEC = 900   # 15 minutes

# Cross-day simulation policy
REGULAR_SHIFT = 480
MAX_OVERTIME_PER_DAY = 120
DAILY_AVAILABLE_TIME = REGULAR_SHIFT + MAX_OVERTIME_PER_DAY

DEFAULT_REPLICATIONS = 200
BASE_RANDOM_SEED = 42


# ============================================================
# 3. INSTANCE GENERATION
# ============================================================

def build_instance(n_jobs: int, mu_scale: float, sigma_scale: float, k: float, seed: int = 42) -> dict:
    """
    Build one stress-test instance.

    Interpretation
    --------------
    - mu_scale scales the PHYSICAL workload level:
        higher average processing time at both stations
    - sigma_scale scales the PHYSICAL variability level:
        higher uncertainty at both stations
    - k scales the PLANNING robustness buffer only:
        it does not change the underlying real-world mu/sigma profile

    If n_jobs > len(base profile), the base pattern is repeated with small
    perturbations so that large-n experiments remain heterogeneous.
    """
    random.seed(seed)

    jobs = list(range(1, n_jobs + 1))
    days = list(range(1, N_DAYS + 1))

    base_keys = sorted(BASE_MU_A.keys())
    n_base = len(base_keys)

    mu_A = {}
    mu_B = {}
    sigma_A = {}
    sigma_B = {}

    for j in jobs:
        base_id = base_keys[(j - 1) % n_base]

        # Small perturbations keep repeated patterns from becoming identical.
        pert_mu_a = random.uniform(0.95, 1.05)
        pert_mu_b = random.uniform(0.95, 1.05)
        pert_sig_a = random.uniform(0.95, 1.05)
        pert_sig_b = random.uniform(0.95, 1.05)

        mu_A[j] = BASE_MU_A[base_id] * mu_scale * pert_mu_a
        mu_B[j] = BASE_MU_B[base_id] * mu_scale * pert_mu_b
        sigma_A[j] = BASE_SIGMA_A[base_id] * sigma_scale * pert_sig_a
        sigma_B[j] = BASE_SIGMA_B[base_id] * sigma_scale * pert_sig_b

    # Core job metrics
    mu_total = {j: mu_A[j] + mu_B[j] for j in jobs}
    sigma_total = {j: math.sqrt(sigma_A[j] ** 2 + sigma_B[j] ** 2) for j in jobs}
    k_buffer = {j: k * sigma_total[j] for j in jobs}
    p_robust = {j: mu_total[j] + k_buffer[j] for j in jobs}

    return {
        "jobs": jobs,
        "days": days,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
        "mu_total": mu_total,
        "sigma_total": sigma_total,
        "k_buffer": k_buffer,
        "p_robust": p_robust,
        "mu_scale": mu_scale,
        "sigma_scale": sigma_scale,
        "k": k,
    }


def summarize_instance(instance: dict) -> dict:
    """
    Produce parameter-summary statistics in REAL UNITS (minutes).

    These columns make the CSV much easier to interpret and support future
    k-calibration studies, because they clearly separate:
    - underlying workload / uncertainty
    - robustness buffer added by k
    """
    jobs = instance["jobs"]

    mu_A_vals = [instance["mu_A"][j] for j in jobs]
    mu_B_vals = [instance["mu_B"][j] for j in jobs]
    sigma_A_vals = [instance["sigma_A"][j] for j in jobs]
    sigma_B_vals = [instance["sigma_B"][j] for j in jobs]
    mu_total_vals = [instance["mu_total"][j] for j in jobs]
    sigma_total_vals = [instance["sigma_total"][j] for j in jobs]
    k_buffer_vals = [instance["k_buffer"][j] for j in jobs]
    p_robust_vals = [instance["p_robust"][j] for j in jobs]

    total_nominal_load = sum(mu_total_vals)
    total_robust_load = sum(p_robust_vals)

    return {
        # Station-specific means/stds
        "mu_A_mean": statistics.mean(mu_A_vals),
        "mu_B_mean": statistics.mean(mu_B_vals),
        "sigma_A_mean": statistics.mean(sigma_A_vals),
        "sigma_B_mean": statistics.mean(sigma_B_vals),

        # Combined nominal time
        "mu_total_mean": statistics.mean(mu_total_vals),
        "mu_total_min": min(mu_total_vals),
        "mu_total_max": max(mu_total_vals),

        # Combined uncertainty
        "sigma_total_mean": statistics.mean(sigma_total_vals),
        "sigma_total_min": min(sigma_total_vals),
        "sigma_total_max": max(sigma_total_vals),

        # Buffer induced by k
        "k_buffer_mean": statistics.mean(k_buffer_vals),
        "k_buffer_min": min(k_buffer_vals),
        "k_buffer_max": max(k_buffer_vals),

        # Robust processing time
        "p_robust_mean": statistics.mean(p_robust_vals),
        "p_robust_min": min(p_robust_vals),
        "p_robust_max": max(p_robust_vals),

        # Horizon and daily load summaries
        "total_nominal_load": total_nominal_load,
        "total_robust_load": total_robust_load,
        "avg_nominal_load_per_day": total_nominal_load / N_DAYS,
        "avg_robust_load_per_day": total_robust_load / N_DAYS,
        "daily_nominal_utilization_ratio": (total_nominal_load / N_DAYS) / C_STD,
        "daily_robust_utilization_ratio": (total_robust_load / N_DAYS) / C_STD,
    }


# ============================================================
# 4. GUROBI BASELINE MODEL
# ============================================================

def solve_gurobi_baseline(instance: dict) -> dict:
    """
    Baseline robust day-level scheduling model.

    Decision variables
    ------------------
    z[j,t] = 1 if job j is assigned to day t
    OT[t]  = overtime minutes used on day t
    u[t]   = 1 if overtime is activated on day t

    Objective
    ---------
    Minimize fixed overtime activation cost + variable overtime usage cost.

    Constraints
    -----------
    1) Each job must be assigned exactly once.
    2) Daily robust load cannot exceed regular capacity plus overtime.
    3) Overtime can only be used if the binary activation variable is on.
    """
    jobs = instance["jobs"]
    days = instance["days"]
    p_robust = instance["p_robust"]

    model = Model("ESS_Robust_BinPacking")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", GUROBI_TIME_LIMIT_SEC)

    z = model.addVars(jobs, days, vtype=GRB.BINARY, name="z")
    OT = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="OT")
    u = model.addVars(days, vtype=GRB.BINARY, name="u")

    model.setObjective(
        quicksum(COST_FIX * u[t] + COST_OT * OT[t] for t in days),
        GRB.MINIMIZE,
    )

    # Full assignment: every job must appear in exactly one day
    for j in jobs:
        model.addConstr(
            quicksum(z[j, t] for t in days) == 1,
            name=f"assign_{j}",
        )

    # Daily robust capacity
    for t in days:
        model.addConstr(
            quicksum(p_robust[j] * z[j, t] for j in jobs) <= C_STD + OT[t],
            name=f"capacity_{t}",
        )

    # Overtime activation logic
    for t in days:
        model.addConstr(
            OT[t] <= BIG_M * u[t],
            name=f"ot_activation_{t}",
        )

    start = time.time()
    model.optimize()
    solve_time = time.time() - start

    status = model.Status
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(status, f"STATUS_{status}")

    # We accept a case for simulation if:
    # - it finished within 15 minutes
    # - and the model has at least one feasible solution to extract
    has_solution = model.SolCount > 0
    accepted_for_simulation = (solve_time <= GUROBI_TIME_LIMIT_SEC) and has_solution

    result = {
        "gurobi_status": status,
        "gurobi_status_name": status_name,
        "solve_time_sec": solve_time,
        "has_solution": has_solution,
        "accepted_for_simulation": accepted_for_simulation,
        "objective_value": None,
        "mip_gap": None,
        "schedule": None,
        "planned_total_ot": None,
        "n_days_with_planned_ot": None,
        "planned_max_day_load": None,
        "planned_avg_day_load": None,
    }

    if has_solution:
        try:
            result["objective_value"] = model.objVal
        except Exception:
            pass

        try:
            result["mip_gap"] = model.MIPGap
        except Exception:
            result["mip_gap"] = None

        raw_schedule = {}
        day_loads = {}
        planned_total_ot = 0.0
        n_days_with_planned_ot = 0

        for t in days:
            assigned_jobs = [j for j in jobs if z[j, t].X > 0.5]
            raw_schedule[t] = assigned_jobs

            load_t = sum(p_robust[j] for j in assigned_jobs)
            day_loads[t] = load_t

            ot_val = OT[t].X
            planned_total_ot += ot_val
            if ot_val > 1e-6:
                n_days_with_planned_ot += 1

        # Post-processing rule: shortest robust job first within each day
        sorted_schedule = {
            t: sorted(raw_schedule[t], key=lambda j: p_robust[j])
            for t in days
        }

        result["schedule"] = sorted_schedule
        result["planned_total_ot"] = planned_total_ot
        result["n_days_with_planned_ot"] = n_days_with_planned_ot
        result["planned_max_day_load"] = max(day_loads.values()) if day_loads else None
        result["planned_avg_day_load"] = statistics.mean(day_loads.values()) if day_loads else None

    return result


# ============================================================
# 5. CROSS-DAY BREAKDOWN SIMULATION
# ============================================================

def sample_positive_normal(mu: float, sigma: float) -> float:
    """Draw a positive processing time from a truncated normal."""
    return max(0.1, random.gauss(mu, sigma))


def realize_total_job_time(job_id: int, mu_A: dict, mu_B: dict, sigma_A: dict, sigma_B: dict) -> float:
    """
    Lightweight simulation layer for grid search.

    This keeps runtime manageable:
    - A_real ~ Normal(mu_A, sigma_A)
    - B_real ~ Normal(mu_B, sigma_B)
    - total_real = A_real + B_real

    The cross-day backlog mechanism then tests whether the horizon can be
    cleared under stochastic realized durations.
    """
    a_real = sample_positive_normal(mu_A[job_id], sigma_A[job_id])
    b_real = sample_positive_normal(mu_B[job_id], sigma_B[job_id])
    return a_real + b_real


def execute_one_day_with_backlog(
    backlog_jobs: list,
    planned_jobs: list,
    time_budget: float,
    mu_A: dict,
    mu_B: dict,
    sigma_A: dict,
    sigma_B: dict,
) -> dict:
    """
    Backlog-first, non-preemptive day execution.

    If the next job does not fit in the remaining time, the job is not split.
    It is pushed entirely into the next-day backlog.
    """
    queue_today = list(backlog_jobs) + list(planned_jobs)
    remaining_time = time_budget
    used_time = 0.0

    executed_sequence = []
    unfinished_sequence = []

    for pos, job_id in enumerate(queue_today):
        realized_time = realize_total_job_time(job_id, mu_A, mu_B, sigma_A, sigma_B)

        if realized_time <= remaining_time:
            used_time += realized_time
            remaining_time -= realized_time
            executed_sequence.append(job_id)
        else:
            unfinished_sequence = queue_today[pos:]
            break

    overtime = max(0.0, used_time - REGULAR_SHIFT)

    return {
        "executed_sequence": executed_sequence,
        "unfinished_sequence": unfinished_sequence,
        "used_time": used_time,
        "overtime": overtime,
        "backlog_end_of_day": len(unfinished_sequence),
    }


def simulate_horizon_with_backlog(
    schedule_dict: dict,
    mu_A: dict,
    mu_B: dict,
    sigma_A: dict,
    sigma_B: dict,
    seed: int | None = None,
) -> dict:
    """
    Simulate the full 4-week horizon with backlog propagation across days.

    A system is considered cleared if terminal backlog is zero at the end.
    """
    if seed is not None:
        random.seed(seed)

    backlog = []
    total_overtime = 0.0
    max_backlog = 0
    outputs = {}

    for day in sorted(schedule_dict.keys()):
        day_result = execute_one_day_with_backlog(
            backlog_jobs=backlog,
            planned_jobs=schedule_dict[day],
            time_budget=DAILY_AVAILABLE_TIME,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
        )
        outputs[day] = day_result
        total_overtime += day_result["overtime"]

        backlog = day_result["unfinished_sequence"]
        max_backlog = max(max_backlog, len(backlog))

    terminal_backlog_count = len(backlog)
    cleared = terminal_backlog_count == 0

    return {
        "days": outputs,
        "total_overtime": total_overtime,
        "terminal_backlog_count": terminal_backlog_count,
        "max_backlog": max_backlog,
        "system_cleared_within_horizon": cleared,
    }


def monte_carlo_breakdown_analysis(
    schedule_dict: dict,
    mu_A: dict,
    mu_B: dict,
    sigma_A: dict,
    sigma_B: dict,
    n_replications: int = DEFAULT_REPLICATIONS,
    base_seed: int = 42,
) -> dict:
    """
    Repeat full-horizon simulation and estimate:
    - probability of clearing the horizon
    - probability of terminal backlog
    - average overtime
    """
    total_overtime_list = []
    terminal_backlog_list = []
    max_backlog_list = []
    cleared_flags = []

    for r in range(n_replications):
        out = simulate_horizon_with_backlog(
            schedule_dict=schedule_dict,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
            seed=base_seed + r,
        )
        total_overtime_list.append(out["total_overtime"])
        terminal_backlog_list.append(out["terminal_backlog_count"])
        max_backlog_list.append(out["max_backlog"])
        cleared_flags.append(out["system_cleared_within_horizon"])

    return {
        "avg_total_overtime": statistics.mean(total_overtime_list),
        "max_total_overtime": max(total_overtime_list),
        "prob_terminal_backlog": sum(1 for x in terminal_backlog_list if x > 0) / n_replications,
        "avg_terminal_backlog": statistics.mean(terminal_backlog_list),
        "avg_max_backlog": statistics.mean(max_backlog_list),
        "prob_cleared_within_horizon": sum(1 for x in cleared_flags if x) / n_replications,
    }


# ============================================================
# 6. SINGLE CASE EVALUATION
# ============================================================

def evaluate_one_case(n_jobs: int, mu_scale: float, sigma_scale: float, k: float,
                      replications: int, seed: int = 42) -> dict:
    """
    One complete case:
    1) build instance
    2) summarize parameters in real units
    3) solve planning model
    4) if accepted, run simulation
    5) return one CSV row
    """
    instance = build_instance(
        n_jobs=n_jobs,
        mu_scale=mu_scale,
        sigma_scale=sigma_scale,
        k=k,
        seed=seed,
    )

    param_summary = summarize_instance(instance)
    gurobi_result = solve_gurobi_baseline(instance)

    row = {
        "n_jobs": n_jobs,
        "n_days": N_DAYS,
        "daily_capacity_min": C_STD,
        "regular_shift_min": REGULAR_SHIFT,
        "max_overtime_per_day_min": MAX_OVERTIME_PER_DAY,

        # Control parameters
        "mu_scale": mu_scale,
        "sigma_scale": sigma_scale,
        "k": k,

        # Parameter summaries in minutes
        **param_summary,

        # Gurobi outputs
        "gurobi_status": gurobi_result["gurobi_status"],
        "gurobi_status_name": gurobi_result["gurobi_status_name"],
        "solve_time_sec": gurobi_result["solve_time_sec"],
        "has_solution": gurobi_result["has_solution"],
        "accepted_for_simulation": gurobi_result["accepted_for_simulation"],
        "objective_value": gurobi_result["objective_value"],
        "mip_gap": gurobi_result["mip_gap"],
        "planned_total_ot": gurobi_result["planned_total_ot"],
        "n_days_with_planned_ot": gurobi_result["n_days_with_planned_ot"],
        "planned_max_day_load": gurobi_result["planned_max_day_load"],
        "planned_avg_day_load": gurobi_result["planned_avg_day_load"],

        # Simulation outputs (filled only if simulation runs)
        "prob_cleared_within_horizon": None,
        "prob_terminal_backlog": None,
        "avg_terminal_backlog": None,
        "avg_max_backlog": None,
        "avg_total_overtime": None,
        "max_total_overtime": None,
        "meets_95_constraint": None,
        "system_ok": False,
    }

    if not gurobi_result["accepted_for_simulation"]:
        return row

    sim_summary = monte_carlo_breakdown_analysis(
        schedule_dict=gurobi_result["schedule"],
        mu_A=instance["mu_A"],
        mu_B=instance["mu_B"],
        sigma_A=instance["sigma_A"],
        sigma_B=instance["sigma_B"],
        n_replications=replications,
        base_seed=seed,
    )

    prob_cleared = sim_summary["prob_cleared_within_horizon"]
    meets_95 = prob_cleared >= 0.95

    row.update({
        "prob_cleared_within_horizon": prob_cleared,
        "prob_terminal_backlog": sim_summary["prob_terminal_backlog"],
        "avg_terminal_backlog": sim_summary["avg_terminal_backlog"],
        "avg_max_backlog": sim_summary["avg_max_backlog"],
        "avg_total_overtime": sim_summary["avg_total_overtime"],
        "max_total_overtime": sim_summary["max_total_overtime"],
        "meets_95_constraint": meets_95,
        "system_ok": bool(gurobi_result["accepted_for_simulation"] and meets_95),
    })

    return row


# ============================================================
# 7. GRID SEARCH DRIVER
# ============================================================

def run_grid_search(
    n_values: list[int],
    mu_scale_values: list[float],
    sigma_scale_values: list[float],
    k_values: list[float],
    replications: int,
    seed: int = BASE_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run the full parameter grid and collect a rich results table.
    """
    rows = []
    total_cases = len(n_values) * len(mu_scale_values) * len(sigma_scale_values) * len(k_values)
    case_id = 0

    for n_jobs in n_values:
        for mu_scale in mu_scale_values:
            for sigma_scale in sigma_scale_values:
                for k in k_values:
                    case_id += 1
                    print(
                        f"[{case_id}/{total_cases}] "
                        f"n={n_jobs}, mu_scale={mu_scale}, sigma_scale={sigma_scale}, k={k}"
                    )

                    row = evaluate_one_case(
                        n_jobs=n_jobs,
                        mu_scale=mu_scale,
                        sigma_scale=sigma_scale,
                        k=k,
                        replications=replications,
                        seed=seed + case_id,
                    )
                    rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. MAIN
# ============================================================

def parse_list_of_numbers(raw: str, cast_type=float):
    return [cast_type(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run a rich grid search for ESS robust scheduling."
    )
    parser.add_argument("--n-values", type=str, default="20,50,80,100",
                        help="Comma-separated job counts.")
    parser.add_argument("--mu-scales", type=str, default="1.0",
                        help="Comma-separated mu scaling factors.")
    parser.add_argument("--sigma-scales", type=str, default="1.0",
                        help="Comma-separated sigma scaling factors.")
    parser.add_argument("--k-values", type=str, default="0.5,1.0,1.5,2.0",
                        help="Comma-separated k values.")
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS,
                        help="Monte Carlo replications for accepted cases.")
    parser.add_argument("--out", type=str,
                        default="results/grid_search/grid_search_results.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    n_values = parse_list_of_numbers(args.n_values, int)
    mu_scale_values = parse_list_of_numbers(args.mu_scales, float)
    sigma_scale_values = parse_list_of_numbers(args.sigma_scales, float)
    k_values = parse_list_of_numbers(args.k_values, float)

    df_results = run_grid_search(
        n_values=n_values,
        mu_scale_values=mu_scale_values,
        sigma_scale_values=sigma_scale_values,
        k_values=k_values,
        replications=args.replications,
        seed=BASE_RANDOM_SEED,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)

    print("\nGrid search finished.")
    print(f"Saved results to: {out_path}")
    print("\nColumns in output CSV:")
    for c in df_results.columns:
        print(" -", c)


if __name__ == "__main__":
    main()
