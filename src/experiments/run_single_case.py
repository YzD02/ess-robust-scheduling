from __future__ import annotations

"""Run one complete end-to-end experiment with fixed parameters.

What this script does
---------------------
Executes the full pipeline — planning → simulation → output — for a single
parameter combination (50 jobs, k = 1.0 by default).

When to use this script
-----------------------
  1. First run after setup: verify that your environment and Gurobi license
     are working correctly before running the longer grid search.
  2. Deep inspection: examine exactly how one scenario plays out day by day,
     including backlog build-up, maintenance windows, and weekend recovery.
  3. Debugging: faster feedback loop than running the full grid search.

How to run
----------
Default parameters (50 jobs, k = 1.0):

    python -m src.experiments.run_single_case

Override any parameter on the command line:

    python -m src.experiments.run_single_case --n-jobs 60 --k 1.0
    python -m src.experiments.run_single_case --n-jobs 60 --k 0.5 --seed 99
    python -m src.experiments.run_single_case --help   # show all available options

Available options
-----------------
  --n-jobs        number of jobs to schedule          (default: 50)
  --k             robustness buffer factor            (default: 1.0)
  --mu-scale      multiplier for all mean job times   (default: 1.0)
  --sigma-scale   multiplier for all standard devs    (default: 1.0)
  --seed          random seed                         (default: 42)
  --replications  Monte Carlo replications            (default: 100)
  --out-dir       output directory for CSV files      (default: results/simulation_outputs)

Output files  (saved to results/simulation_outputs/)
-----------------------------------------------------
  gantt_events_single_run.csv   — start/end time for every job at every
                                  station; feed into plot_gantt.py to visualise
  single_run_day_summary.csv    — per-day statistics: jobs executed, backlog
                                  count, utilisation rate, overflow flag
  single_run_mc_summary.csv     — aggregated statistics over 100 Monte Carlo
                                  replications (clear probability, avg weekend
                                  days used, avg final completion day, etc.)
"""

from pathlib import Path
import argparse
import pandas as pd

from src.models.job_generation import JobGenerationConfig, generate_job_parameters
from src.models.robust_processing import compute_robust_processing_times
from src.models.gurobi_baseline import solve_gurobi_baseline
from src.simulation.simulation_engine import (
    SimulationPolicy,
    MachineStopConfig,
    simulate_horizon_with_backlog,
    monte_carlo_breakdown_analysis,
    print_single_run_result,
    print_monte_carlo_summary,
    event_log_to_dataframe,
)
from src.utils.maintenance import resolve_maintenance_map


def main():
    # ----------------------------------------------------------------
    # Command-line arguments — override any of these when running:
    #   python -m src.experiments.run_single_case --n-jobs 60 --k 1.0
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run one end-to-end ESS scheduling experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-jobs",       type=int,   default=50,   help="Number of jobs to schedule.")
    parser.add_argument("--k",            type=float, default=1.0,  help="Robustness buffer factor (0 = no buffer, 2 = very conservative).")
    parser.add_argument("--mu-scale",     type=float, default=1.0,  help="Multiplier for all mean processing times (>1 = heavier workload).")
    parser.add_argument("--sigma-scale",  type=float, default=1.0,  help="Multiplier for all standard deviations (>1 = more variability).")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed for reproducibility.")
    parser.add_argument("--replications", type=int,   default=100,  help="Number of Monte Carlo replications.")
    parser.add_argument("--out-dir",      type=str,   default="results/simulation_outputs",
                        help="Directory to save output CSV files.")
    parser.add_argument(
        "--maintenance",
        type=str,
        default=None,
        help=(
            "Fixed maintenance schedule as 'day:start_min:end_min,...'. "
            "Minutes are from shift start (0 = 08:00). "
            "Example — Week1 Wed 12:00-14:00, Week2 Thu 08:00-10:00: '3:240:360,9:0:120'. "
            "Pass 'random' to override DEFAULT_MAINTENANCE_SCHEDULE and use random windows. "
            "If omitted, DEFAULT_MAINTENANCE_SCHEDULE defined in this file is used."
        ),
    )
    args = parser.parse_args()

    n_jobs       = args.n_jobs
    weekday_days = 20           # planning horizon: 4 weeks x 5 days (not exposed as a flag — changing this requires reviewing the full model)
    mu_scale     = args.mu_scale
    sigma_scale  = args.sigma_scale
    k            = args.k
    random_seed  = args.seed

    # ---- Resolve maintenance schedule ----
    maintenance_map = resolve_maintenance_map(args.maintenance)
    if maintenance_map is not None:
        print(f"Maintenance schedule (fixed): {maintenance_map}")
    else:
        print("Maintenance schedule: random (generated per replication from seed).")

    # Job generation config: these are assumption-based baseline values.
    # Replace with calibrated values once real production data is available.
    gen_cfg = JobGenerationConfig(
        mu_A_mean=90.0,   # average Station A time per job (minutes)
        mu_A_std=6.0,     # job-to-job spread in Station A times
        mu_B_mean=60.0,   # average Station B time per job (minutes)
        mu_B_std=5.0,     # job-to-job spread in Station B times
        sigma_A_mean=14.0, # average within-job uncertainty at Station A
        sigma_A_std=2.0,
        sigma_B_mean=18.0, # average within-job uncertainty at Station B
        sigma_B_std=2.5,
    )

    generated = generate_job_parameters(
        n_jobs=n_jobs,
        mu_scale=mu_scale,
        sigma_scale=sigma_scale,
        seed=random_seed,
        config=gen_cfg,
    )
    jobs = generated['jobs']
    days = list(range(1, weekday_days + 1))
    mu_A = generated['mu_A']
    mu_B = generated['mu_B']
    sigma_A = generated['sigma_A']
    sigma_B = generated['sigma_B']

    # Gurobi model parameters
    C_std = 480.0        # regular shift capacity per day (8 hours × 60 minutes)
    Cost_OT = 5.0        # cost per overtime minute (used to penalise overtime usage)
    Cost_fix = 180.0     # fixed cost for activating overtime on any given day
    M = 2000.0           # big-M constant for the overtime activation constraint
    time_limit_sec = 900.0  # maximum solver time: 15 minutes

    p_nominal, p_robust = compute_robust_processing_times(
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        k=k,
    )

    # Simulation execution policy
    policy = SimulationPolicy(
        regular_shift=480.0,          # available minutes per working day
        weekday_horizon_days=20,      # number of planned weekdays (4 weeks)
        weekend_extension_days=8,     # extra weekend days available if backlog remains
        weekend_fixed_cost=300.0,     # cost to open one weekend recovery day
        weekend_variable_cost=8.0,    # cost per minute used on a weekend day
        maintenance_duration=120.0,   # planned maintenance window per week (minutes)
    )

    # Machine-stop model for Station A (automated equipment)
    # Micro-stops interrupt the machine at random intervals for random durations.
    # These values are assumption-based; replace with real maintenance logs when available.
    stop_cfg = MachineStopConfig(
        mean_uptime_between_stops=68.57,  # average running time before a stop (minutes)
        mean_stop_duration=8.0,           # average duration of each stop (minutes)
        stop_duration_cv=1.0,             # coefficient of variation for stop duration
                                          # (1.0 = exponential-like, high variability)
    )

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

    one_run = simulate_horizon_with_backlog(
        schedule_dict=schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_B=sigma_B,
        policy=policy,
        stop_cfg=stop_cfg,
        seed=42,
        simulation_run=1,
        maintenance_map=maintenance_map,
    )
    print_single_run_result(one_run)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gantt_df = event_log_to_dataframe(one_run.event_log)
    gantt_df['lane'] = gantt_df.apply(
        lambda r: f"Day {int(r['executed_day'])} - {r['day_type']} - Station {r['station']}",
        axis=1,
    )
    gantt_csv_path = out_dir / 'gantt_events_single_run.csv'
    gantt_df.to_csv(gantt_csv_path, index=False)
    print(f"\nSaved Gantt event log to: {gantt_csv_path}")

    day_rows = []
    first_overflow_day = None
    for day in sorted(one_run.days.keys()):
        d = one_run.days[day]
        planned_nominal_load = sum(p_nominal[j] for j in d.planned_jobs) if d.planned_jobs else 0.0
        planned_robust_load = sum(p_robust[j] for j in d.planned_jobs) if d.planned_jobs else 0.0
        backlog_nominal_load = sum(p_nominal[j] for j in d.backlog_jobs_in) if d.backlog_jobs_in else 0.0
        backlog_robust_load = sum(p_robust[j] for j in d.backlog_jobs_in) if d.backlog_jobs_in else 0.0
        realized_executed_load = sum(detail['total_time'] for detail in d.job_details.values()) if d.job_details else 0.0
        capacity = policy.regular_shift
        slack = capacity - realized_executed_load
        overflow_flag = d.backlog_end_of_day > 0
        if overflow_flag and first_overflow_day is None:
            first_overflow_day = day
        day_rows.append({
            'day': day,
            'day_type': d.day_type,
            'planned_jobs': str(d.planned_jobs),
            'backlog_jobs_in': str(d.backlog_jobs_in),
            'realized_queue': str(d.realized_queue),
            'executed_sequence': str(d.executed_sequence),
            'unfinished_sequence': str(d.unfinished_sequence),
            'planned_job_count': len(d.planned_jobs),
            'backlog_in_count': len(d.backlog_jobs_in),
            'executed_count': len(d.executed_sequence),
            'unfinished_count': len(d.unfinished_sequence),
            'planned_nominal_load': planned_nominal_load,
            'planned_robust_load': planned_robust_load,
            'backlog_nominal_load': backlog_nominal_load,
            'backlog_robust_load': backlog_robust_load,
            'realized_executed_load': realized_executed_load,
            'capacity': capacity,
            'slack': slack,
            'utilization_ratio': realized_executed_load / capacity if capacity > 0 else None,
            'overflow_flag': overflow_flag,
            'backlog_end_of_day': d.backlog_end_of_day,
            'weekend_day_used': d.weekend_day_used,
            'maintenance_start_day': d.maintenance_start_day,
            'maintenance_end_day': d.maintenance_end_day,
        })
    if first_overflow_day is not None:
        for row in day_rows:
            row['first_overflow_day_flag'] = (row['day'] == first_overflow_day)
    else:
        for row in day_rows:
            row['first_overflow_day_flag'] = False

    day_summary_df = pd.DataFrame(day_rows)
    day_summary_csv_path = out_dir / 'single_run_day_summary.csv'
    day_summary_df.to_csv(day_summary_csv_path, index=False)
    print(f"Saved day summary to: {day_summary_csv_path}")

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
        maintenance_map=maintenance_map,
    )
    print_monte_carlo_summary(summary, n_replications)

    summary_df = pd.DataFrame([{
        'n_jobs': len(jobs),
        'weekday_days': weekday_days,
        'weekend_extension_days': policy.weekend_extension_days,
        'maintenance_duration': policy.maintenance_duration,
        'mu_scale': mu_scale,
        'sigma_scale': sigma_scale,
        'k': k,
        'avg_nominal_job_time': sum(p_nominal.values()) / len(p_nominal),
        'avg_robust_job_time': sum(p_robust.values()) / len(p_robust),
        'prob_cleared_within_weekdays': summary['prob_cleared_within_weekdays'],
        'prob_cleared_within_extended_horizon': summary['prob_cleared_within_extended_horizon'],
        'prob_terminal_backlog_after_extension': summary['prob_terminal_backlog_after_extension'],
        'avg_terminal_backlog_after_extension': summary['avg_terminal_backlog_after_extension'],
        'avg_max_backlog': summary['avg_max_backlog'],
        'avg_n_spillover_days': summary['avg_n_spillover_days'],
        'prob_any_spillover': summary['prob_any_spillover'],
        'avg_first_spillover_day': summary['avg_first_spillover_day'],
        'avg_n_weekend_days_used': summary['avg_n_weekend_days_used'],
        'prob_any_weekend_use': summary['prob_any_weekend_use'],
        'avg_weekend_used_time': summary['avg_weekend_used_time'],
        'avg_total_weekend_cost': summary['avg_total_weekend_cost'],
        'avg_final_completion_day': summary['avg_final_completion_day'],
    }])
    summary_csv_path = out_dir / 'single_run_mc_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved Monte Carlo summary to: {summary_csv_path}")


if __name__ == '__main__':
    main()
