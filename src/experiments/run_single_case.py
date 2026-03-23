from __future__ import annotations

"""
run_single_case.py
==================

Run one end-to-end ESS scheduling experiment under the following logic:

1. Build one reusable instance (example: 50 jobs, 20 weekdays)
2. Compute robust processing times using the shared model utility
3. Solve the baseline Gurobi planning model on weekday bins only
4. Run one stochastic execution sample path with:
   - 20 weekday bins
   - 8 weekend extension bins
   - no same-day overtime completion
5. Export event logs for Gantt chart visualization
6. Run Monte Carlo summary validation

Design principle
----------------
This script should reuse existing library functions as much as possible
instead of re-implementing formulas or duplicated logic locally.
"""

from pathlib import Path
import random
import pandas as pd

from src.models.robust_processing import compute_robust_processing_times
from src.models.gurobi_baseline import solve_gurobi_baseline
from src.simulation.simpy_cross_day_breakdown import (
    SimulationPolicy,
    MachineStopConfig,
    simulate_horizon_with_backlog,
    monte_carlo_breakdown_analysis,
    print_single_run_result,
    print_monte_carlo_summary,
    event_log_to_dataframe,
)


# =====================================================
# 1. BASE JOB PROFILES
# =====================================================

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
# 2. REUSABLE INSTANCE BUILDER
# =====================================================

def build_reusable_instance(
    *,
    n_jobs: int,
    n_days: int,
    mu_scale: float = 1.0,
    sigma_scale: float = 1.0,
    seed: int = 42,
    add_small_perturbation: bool = True,
) -> dict:
    """
    Build an instance by expanding the base job profiles.

    Logic
    -----
    - If n_jobs <= number of base profiles:
        use the first n_jobs profiles
    - If n_jobs > number of base profiles:
        cycle through the base profiles repeatedly
    - Optional small perturbations make repeated jobs slightly different

    This keeps single-case experiments aligned with the grid-search philosophy.
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


# =====================================================
# 3. MAIN
# =====================================================

def main():
    # -------------------------------------------------
    # A. Single-case experiment settings
    # -------------------------------------------------
    # Change only these settings when you want to try another case.
    n_jobs = 50
    weekday_days = 20

    mu_scale = 1.0
    sigma_scale = 1.0
    k = 1.0

    random_seed = 42

    # -------------------------------------------------
    # B. Build one reusable instance
    # -------------------------------------------------
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

    # -------------------------------------------------
    # C. Planning-layer parameters
    # -------------------------------------------------
    # The planning model still uses weekday bins only.
    C_std = 480.0
    Cost_OT = 5.0
    Cost_fix = 180.0
    M = 2000.0
    time_limit_sec = 900.0

    p_nominal, p_robust = compute_robust_processing_times(
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        k=k,
    )

    # -------------------------------------------------
    # D. Execution-layer parameters
    # -------------------------------------------------
    # New policy:
    # - 20 weekdays
    # - 8 weekend extension bins
    # - no same-day overtime completion
    policy = SimulationPolicy(
        regular_shift=480.0,
        weekday_horizon_days=20,
        weekend_extension_days=8,
        weekend_fixed_cost=300.0,
        weekend_variable_cost=8.0,
    )

    stop_cfg = MachineStopConfig(
        mean_uptime_between_stops=68.57,
        mean_stop_duration=8.0,
        stop_duration_cv=1.0,
    )

    # -------------------------------------------------
    # E. Solve weekday planning model
    # -------------------------------------------------
    gurobi_result = solve_gurobi_baseline(
        jobs=jobs,
        days=days,
        p_robust=p_robust,
        C_std=C_std,
        Cost_OT=Cost_OT,
        Cost_fix=Cost_fix,
        M=M,
        time_limit_sec=time_limit_sec,
        output_flag=1,
    )

    print("\n==============================")
    print("Gurobi Result")
    print("==============================")
    print("Status               :", gurobi_result.status)
    print("Solve time (sec)     :", round(gurobi_result.solve_time_sec, 3))
    print("Objective value      :", gurobi_result.objective_value)
    print("Planned total OT     :", gurobi_result.planned_total_ot)
    print("Days with planned OT :", gurobi_result.n_days_with_planned_ot)

    if gurobi_result.sorted_schedule is None:
        print("\nNo usable schedule was produced by the planning model.")
        return

    schedule = gurobi_result.sorted_schedule

    print("\nPlanned weekday schedule (post-processed order):")
    for day in sorted(schedule.keys()):
        if schedule[day]:
            print(f"Day {day}: {schedule[day]}")

    # -------------------------------------------------
    # F. Run one stochastic sample path
    # -------------------------------------------------
    one_run = simulate_horizon_with_backlog(
        schedule_dict=schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_B=sigma_B,
        policy=policy,
        stop_cfg=stop_cfg,
        seed=42,
        simulation_run=1,
    )

    print_single_run_result(one_run)

    # -------------------------------------------------
    # G. Export Gantt event log
    # -------------------------------------------------
    out_dir = Path("results/simulation_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    gantt_df = event_log_to_dataframe(one_run.event_log)

    # Add a convenient lane column for plotting/debugging
    gantt_df["lane"] = gantt_df.apply(
        lambda r: f"Day {int(r['executed_day'])} - {r['day_type']} - Station {r['station']}",
        axis=1,
    )

    gantt_csv_path = out_dir / "gantt_events_single_run.csv"
    gantt_df.to_csv(gantt_csv_path, index=False)
    print(f"\nSaved Gantt event log to: {gantt_csv_path}")

    # -------------------------------------------------
    # H. Export day summary
    # -------------------------------------------------

    day_rows = []
    for day in sorted(one_run.days.keys()):
        d = one_run.days[day]

        planned_nominal_load = sum(p_nominal[j] for j in d.planned_jobs) if d.planned_jobs else 0.0
        planned_robust_load = sum(p_robust[j] for j in d.planned_jobs) if d.planned_jobs else 0.0

        backlog_nominal_load = sum(p_nominal[j] for j in d.backlog_jobs_in) if d.backlog_jobs_in else 0.0
        backlog_robust_load = sum(p_robust[j] for j in d.backlog_jobs_in) if d.backlog_jobs_in else 0.0

        realized_executed_load = sum(
            detail["total_time"] for detail in d.job_details.values()
        ) if d.job_details else 0.0

        capacity = policy.regular_shift
        slack = capacity - realized_executed_load
        overflow_flag = d.backlog_end_of_day > 0

        day_rows.append({
            "day": day,
            "day_type": d.day_type,

            "planned_jobs": str(d.planned_jobs),
            "backlog_jobs_in": str(d.backlog_jobs_in),
            "realized_queue": str(d.realized_queue),
            "executed_sequence": str(d.executed_sequence),
            "unfinished_sequence": str(d.unfinished_sequence),

            "planned_job_count": len(d.planned_jobs),
            "backlog_in_count": len(d.backlog_jobs_in),
            "executed_count": len(d.executed_sequence),
            "unfinished_count": len(d.unfinished_sequence),

            "planned_nominal_load": planned_nominal_load,
            "planned_robust_load": planned_robust_load,
            "backlog_nominal_load": backlog_nominal_load,
            "backlog_robust_load": backlog_robust_load,
            "realized_executed_load": realized_executed_load,

            "capacity": capacity,
            "slack": slack,
            "utilization_ratio": realized_executed_load / capacity if capacity > 0 else None,
            "overflow_flag": overflow_flag,
            "backlog_end_of_day": d.backlog_end_of_day,

            "weekend_day_used": d.weekend_day_used,
        })

    day_summary_df = pd.DataFrame(day_rows)
    day_summary_csv_path = out_dir / "single_run_day_summary.csv"
    day_summary_df.to_csv(day_summary_csv_path, index=False)
    print(f"Saved day summary to: {day_summary_csv_path}")

    # -------------------------------------------------
    # I. Monte Carlo summary
    # -------------------------------------------------
    n_replications = 100

    summary = monte_carlo_breakdown_analysis(
        schedule_dict=schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_B=sigma_B,
        policy=policy,
        stop_cfg=stop_cfg,
        n_replications=n_replications,
        base_seed=42,
    )

    print_monte_carlo_summary(summary, n_replications)

    summary_df = pd.DataFrame([{
        "n_jobs": len(jobs),
        "weekday_days": weekday_days,
        "weekend_extension_days": policy.weekend_extension_days,
        "mu_scale": mu_scale,
        "sigma_scale": sigma_scale,
        "k": k,
        "avg_nominal_job_time": sum(p_nominal.values()) / len(p_nominal),
        "avg_robust_job_time": sum(p_robust.values()) / len(p_robust),
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
    }])

    summary_csv_path = out_dir / "single_run_mc_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved Monte Carlo summary to: {summary_csv_path}")


if __name__ == "__main__":
    main()