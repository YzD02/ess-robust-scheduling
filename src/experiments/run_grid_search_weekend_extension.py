from __future__ import annotations

"""
run_grid_search_weekend_extension.py
====================================

Grid search for the ESS robust scheduling system under the new execution logic:

- 20 weekday bins planned by Gurobi
- 8 weekend extension bins available in simulation
- no same-day overtime completion
- jobs that do not fit are deferred entirely to the next day
- weekend extension usage is recorded as additional cost

This script is intentionally separate from the original grid search so that:
- the earlier baseline experiment is preserved
- the new weekend-extension logic can be studied independently

example usage
-------------
python -m src.experiments.run_grid_search_weekend_extension --n-values 20,50,80,100 --mu-scales 1.0 --sigma-scales 1.0 --k-values 0.5,1.0,1.5,2.0 --replications 100

"""

import argparse
from pathlib import Path
import random
import time
from typing import Any

import pandas as pd

from src.models.robust_processing import compute_robust_processing_times
from src.models.gurobi_baseline import solve_gurobi_baseline
from src.simulation.simpy_cross_day_breakdown import (
    SimulationPolicy,
    MachineStopConfig,
    monte_carlo_breakdown_analysis,
)


# =====================================================
# 1. DEFAULTS
# =====================================================

DEFAULT_REPLICATIONS = 100

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


# =====================================================
# 2. HELPERS
# =====================================================

def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_reusable_instance(
    *,
    n_jobs: int,
    n_days: int,
    mu_scale: float = 1.0,
    sigma_scale: float = 1.0,
    seed: int = 42,
    add_small_perturbation: bool = True,
) -> dict[str, Any]:
    """
    Build a reusable instance by extending the base job profiles.

    This matches the philosophy used in single-case experiments:
    - expand from a small set of base profiles
    - cycle if n_jobs exceeds the base count
    - optionally add small perturbations
    """
    random.seed(seed)

    jobs = list(range(1, n_jobs + 1))
    days = list(range(1, n_days + 1))

    base_keys = sorted(BASE_MU_A.keys())
    n_base = len(base_keys)

    mu_A = {}
    mu_B = {}
    sigma_A = {}
    sigma_B = {}

    for j in jobs:
        base_id = base_keys[(j - 1) % n_base]

        if add_small_perturbation:
            pert_mu_A = random.uniform(0.97, 1.03)
            pert_mu_B = random.uniform(0.97, 1.03)
            pert_sig_A = random.uniform(0.97, 1.03)
            pert_sig_B = random.uniform(0.97, 1.03)
        else:
            pert_mu_A = pert_mu_B = pert_sig_A = pert_sig_B = 1.0

        mu_A[j] = BASE_MU_A[base_id] * mu_scale * pert_mu_A
        mu_B[j] = BASE_MU_B[base_id] * mu_scale * pert_mu_B
        sigma_A[j] = BASE_SIGMA_A[base_id] * sigma_scale * pert_sig_A
        sigma_B[j] = BASE_SIGMA_B[base_id] * sigma_scale * pert_sig_B

    return {
        "jobs": jobs,
        "days": days,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
    }


def summarize_instance_parameters(
    *,
    mu_A: dict[int, float],
    mu_B: dict[int, float],
    sigma_A: dict[int, float],
    sigma_B: dict[int, float],
    p_nominal: dict[int, float],
    p_robust: dict[int, float],
    k: float,
    weekday_days: int,
    C_std: float,
) -> dict[str, float]:
    """
    Build parameter-interpretation fields so readers can understand
    the actual time scales and robust buffer size.
    """
    import math

    mu_total_vals = [mu_A[j] + mu_B[j] for j in mu_A]
    sigma_total_vals = [math.sqrt(sigma_A[j] ** 2 + sigma_B[j] ** 2) for j in sigma_A]
    k_buffer_vals = [k * s for s in sigma_total_vals]

    total_nominal_load = sum(p_nominal.values())
    total_robust_load = sum(p_robust.values())

    return {
        "mu_A_mean": sum(mu_A.values()) / len(mu_A),
        "mu_B_mean": sum(mu_B.values()) / len(mu_B),
        "sigma_A_mean": sum(sigma_A.values()) / len(sigma_A),
        "sigma_B_mean": sum(sigma_B.values()) / len(sigma_B),

        "mu_total_mean": sum(mu_total_vals) / len(mu_total_vals),
        "mu_total_min": min(mu_total_vals),
        "mu_total_max": max(mu_total_vals),

        "sigma_total_mean": sum(sigma_total_vals) / len(sigma_total_vals),
        "sigma_total_min": min(sigma_total_vals),
        "sigma_total_max": max(sigma_total_vals),

        "k_buffer_mean": sum(k_buffer_vals) / len(k_buffer_vals),
        "k_buffer_min": min(k_buffer_vals),
        "k_buffer_max": max(k_buffer_vals),

        "p_robust_mean": sum(p_robust.values()) / len(p_robust),
        "p_robust_min": min(p_robust.values()),
        "p_robust_max": max(p_robust.values()),

        "total_nominal_load": total_nominal_load,
        "total_robust_load": total_robust_load,
        "avg_nominal_load_per_day": total_nominal_load / weekday_days,
        "avg_robust_load_per_day": total_robust_load / weekday_days,
        "daily_nominal_utilization_ratio": (total_nominal_load / weekday_days) / C_std,
        "daily_robust_utilization_ratio": (total_robust_load / weekday_days) / C_std,
    }


def evaluate_one_case(
    *,
    n_jobs: int,
    mu_scale: float,
    sigma_scale: float,
    k: float,
    replications: int,
    random_seed: int,
    weekday_days: int,
    weekend_extension_days: int,
    C_std: float,
    Cost_OT: float,
    Cost_fix: float,
    M: float,
    gurobi_time_limit_sec: float,
    weekend_fixed_cost: float,
    weekend_variable_cost: float,
    output_flag: int = 0,
) -> dict[str, Any]:
    """
    Evaluate one grid-search combination.

    Steps
    -----
    1. build instance
    2. compute robust processing times
    3. solve weekday planning model
    4. if accepted, run weekend-extension Monte Carlo validation
    """
    # -------------------------------------
    # Instance construction
    # -------------------------------------
    instance = build_reusable_instance(
        n_jobs=n_jobs,
        n_days=weekday_days,
        mu_scale=mu_scale,
        sigma_scale=sigma_scale,
        seed=random_seed,
        add_small_perturbation=True,
    )

    jobs = instance["jobs"]
    days = instance["days"]
    mu_A = instance["mu_A"]
    mu_B = instance["mu_B"]
    sigma_A = instance["sigma_A"]
    sigma_B = instance["sigma_B"]

    p_nominal, p_robust = compute_robust_processing_times(
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        k=k,
    )

    # -------------------------------------
    # Planning layer
    # -------------------------------------
    t0 = time.time()
    gurobi_result = solve_gurobi_baseline(
        jobs=jobs,
        days=days,
        p_robust=p_robust,
        C_std=C_std,
        Cost_OT=Cost_OT,
        Cost_fix=Cost_fix,
        M=M,
        time_limit_sec=gurobi_time_limit_sec,
        output_flag=output_flag,
    )
    solve_wall_time = time.time() - t0

    accepted_for_simulation = (
        gurobi_result.sorted_schedule is not None
        and gurobi_result.solve_time_sec <= gurobi_time_limit_sec
    )

    row: dict[str, Any] = {
        "n_jobs": n_jobs,
        "weekday_days": weekday_days,
        "weekend_extension_days": weekend_extension_days,
        "mu_scale": mu_scale,
        "sigma_scale": sigma_scale,
        "k": k,

        "gurobi_status": gurobi_result.status,
        "solve_time_sec": gurobi_result.solve_time_sec,
        "solve_wall_time_sec": solve_wall_time,
        "accepted_for_simulation": accepted_for_simulation,

        "objective_value": gurobi_result.objective_value,
        "planned_total_ot": gurobi_result.planned_total_ot,
        "n_days_with_planned_ot": gurobi_result.n_days_with_planned_ot,
    }

    # Parameter interpretation layer
    row.update(
        summarize_instance_parameters(
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
            p_nominal=p_nominal,
            p_robust=p_robust,
            k=k,
            weekday_days=weekday_days,
            C_std=C_std,
        )
    )

    # -------------------------------------
    # Execution layer
    # -------------------------------------
    if not accepted_for_simulation:
        row.update({
            "prob_cleared_within_weekdays": None,
            "prob_cleared_within_extended_horizon": None,
            "prob_terminal_backlog_after_extension": None,
            "avg_terminal_backlog_after_extension": None,
            "avg_max_backlog": None,
            "avg_n_spillover_days": None,
            "prob_any_spillover": None,
            "avg_first_spillover_day": None,
            "avg_n_weekend_days_used": None,
            "prob_any_weekend_use": None,
            "avg_weekend_used_time": None,
            "avg_total_weekend_cost": None,
            "avg_final_completion_day": None,
            "system_ok": False,
        })
        return row

    policy = SimulationPolicy(
        regular_shift=480.0,
        weekday_horizon_days=weekday_days,
        weekend_extension_days=weekend_extension_days,
        weekend_fixed_cost=weekend_fixed_cost,
        weekend_variable_cost=weekend_variable_cost,
    )

    stop_cfg = MachineStopConfig(
        mean_uptime_between_stops=68.57,
        mean_stop_duration=8.0,
        stop_duration_cv=1.0,
    )

    summary = monte_carlo_breakdown_analysis(
        schedule_dict=gurobi_result.sorted_schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_B=sigma_B,
        policy=policy,
        stop_cfg=stop_cfg,
        n_replications=replications,
        base_seed=random_seed,
    )

    row.update({
        "prob_cleared_within_weekdays": summary["prob_cleared_within_weekdays"],
        "prob_cleared_within_extended_horizon": summary["prob_cleared_within_extended_horizon"],
        "prob_terminal_backlog_after_extension": summary["prob_terminal_backlog_after_extension"],
        "avg_terminal_backlog_after_extension": summary["avg_terminal_backlog_after_extension"],
        "avg_max_backlog": summary["avg_max_backlog"],
        "avg_n_spillover_days": summary["avg_n_spillover_days"],
        "prob_any_spillover": summary["prob_any_spillover"],
        "avg_first_spillover_day": summary["avg_first_spillover_day"],
        "avg_n_weekend_days_used": summary["avg_n_weekend_days_used"],
        "prob_any_weekend_use": summary["prob_any_weekend_use"],
        "avg_weekend_used_time": summary["avg_weekend_used_time"],
        "avg_total_weekend_cost": summary["avg_total_weekend_cost"],
        "avg_final_completion_day": summary["avg_final_completion_day"],
    })

    # Main success criterion for the new logic:
    # 95% of runs must clear within the extended horizon.
    row["system_ok"] = (
        accepted_for_simulation
        and summary["prob_cleared_within_extended_horizon"] is not None
        and summary["prob_cleared_within_extended_horizon"] >= 0.95
    )

    return row


# =====================================================
# 3. MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run weekend-extension grid search for ESS robust scheduling."
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="20,50,80,100",
        help="Comma-separated job counts."
    )
    parser.add_argument(
        "--mu-scales",
        type=str,
        default="1.0",
        help="Comma-separated mu scaling factors."
    )
    parser.add_argument(
        "--sigma-scales",
        type=str,
        default="1.0",
        help="Comma-separated sigma scaling factors."
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated k values."
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Monte Carlo replications for accepted cases."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/grid_search/grid_search_results_weekend_extension.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "--weekday-days",
        type=int,
        default=20,
        help="Number of weekday bins planned by Gurobi."
    )
    parser.add_argument(
        "--weekend-extension-days",
        type=int,
        default=8,
        help="Number of weekend extension bins available in simulation."
    )
    parser.add_argument(
        "--weekend-fixed-cost",
        type=float,
        default=300.0,
        help="Fixed cost for activating one weekend extension day."
    )
    parser.add_argument(
        "--weekend-variable-cost",
        type=float,
        default=8.0,
        help="Variable cost per minute used in a weekend extension day."
    )
    parser.add_argument(
        "--gurobi-time-limit-sec",
        type=float,
        default=900.0,
        help="Maximum solve time for Gurobi."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed."
    )
    parser.add_argument(
        "--output-flag",
        type=int,
        default=0,
        help="Gurobi OutputFlag (0 or 1)."
    )
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    mu_scales = parse_float_list(args.mu_scales)
    sigma_scales = parse_float_list(args.sigma_scales)
    k_values = parse_float_list(args.k_values)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_cases = len(n_values) * len(mu_scales) * len(sigma_scales) * len(k_values)
    case_idx = 0

    print("\n======================================")
    print("Weekend-Extension Grid Search Started")
    print("======================================")
    print(f"Total cases: {total_cases}")

    for n_jobs in n_values:
        for mu_scale in mu_scales:
            for sigma_scale in sigma_scales:
                for k in k_values:
                    case_idx += 1
                    print(
                        f"\n[{case_idx}/{total_cases}] "
                        f"n_jobs={n_jobs}, mu_scale={mu_scale}, sigma_scale={sigma_scale}, k={k}"
                    )

                    row = evaluate_one_case(
                        n_jobs=n_jobs,
                        mu_scale=mu_scale,
                        sigma_scale=sigma_scale,
                        k=k,
                        replications=args.replications,
                        random_seed=args.seed,
                        weekday_days=args.weekday_days,
                        weekend_extension_days=args.weekend_extension_days,
                        C_std=480.0,
                        Cost_OT=5.0,
                        Cost_fix=180.0,
                        M=2000.0,
                        gurobi_time_limit_sec=args.gurobi_time_limit_sec,
                        weekend_fixed_cost=args.weekend_fixed_cost,
                        weekend_variable_cost=args.weekend_variable_cost,
                        output_flag=args.output_flag,
                    )

                    rows.append(row)

                    # save progressively so long runs are not lost
                    pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print("\n======================================")
    print("Weekend-Extension Grid Search Finished")
    print("======================================")
    print(f"Saved results to: {out_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()