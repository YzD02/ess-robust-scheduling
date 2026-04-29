from __future__ import annotations

"""Grid search experiment: systematically sweep job counts and robustness factors.

What this script does
---------------------
Tests every combination of job count and k value in the parameter grid,
running the full planning → simulation pipeline for each one and recording
the results in a single CSV file.

For each parameter combination the script:
  1. Generates a set of jobs with the given mu_scale / sigma_scale
  2. Solves the Gurobi day-assignment model to produce a weekly schedule
  3. Runs Monte Carlo simulation to estimate how often the schedule
     survives real-world variability and maintenance disruptions
  4. Appends one result row to the output CSV immediately (so progress is
     not lost if the run is interrupted)

How to run
----------
Full grid (default settings, takes roughly 10–30 minutes):

    python -m src.experiments.run_grid_search

Quick validation run (~2–5 minutes):

    python -m src.experiments.run_grid_search --n-values 20,40 --k-values 0.5,1.0 --replications 20

Available command-line options
-------------------------------
  --n-values       comma-separated job counts to test  (default: 20,40,60,80,100)
  --k-values       comma-separated k values to test    (default: 0.5,1.0,1.5,2.0)
  --mu-scales      comma-separated mu scaling factors  (default: 1.0)
  --sigma-scales   comma-separated sigma scaling       (default: 1.0)
  --replications   Monte Carlo runs per case           (default: 100)
  --seed           base random seed                    (default: 42)
  --out            output CSV file path

Output
------
  results/grid_search/grid_search_results_weekend_extension.csv

This CSV can be passed directly to plot_heatmaps.py and plot_phase_diagram.py
to generate the visualisation figures.

Note on re-running
------------------
Re-running the script overwrites the existing CSV.  If you want to keep a
previous result set, rename or copy the file before running again.
"""

import argparse
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.models.job_generation import JobGenerationConfig, generate_job_parameters
from src.models.robust_processing import compute_robust_processing_times
from src.models.gurobi_baseline import solve_gurobi_baseline
from src.simulation.simulation_engine import (
    SimulationPolicy,
    MachineStopConfig,
    monte_carlo_breakdown_analysis,
)
from src.utils.maintenance import resolve_maintenance_map

DEFAULT_REPLICATIONS = 100


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def summarize_instance_parameters(*, mu_A, mu_B, sigma_A, sigma_B, p_nominal, p_robust, k, weekday_days, C_std):
    """Summarise planning-layer instance parameters for the results CSV.

    Note: sigma_B here is derived analytically from the triangular distribution
    parameters (low_B, high_B, mu_B) in job_generation.py, not sampled directly.
    It is used only in the planning buffer calculation, not in simulation.
    """
    import math
    import statistics
    mu_total_vals = [mu_A[j] + mu_B[j] for j in mu_A]
    sigma_total_vals = [math.sqrt(sigma_A[j]**2 + sigma_B[j]**2) for j in sigma_A]
    k_buffer_vals = [k * s for s in sigma_total_vals]
    total_nominal_load = sum(p_nominal.values())
    total_robust_load = sum(p_robust.values())
    return {
        'mu_A_mean': statistics.mean(mu_A.values()),
        'mu_B_mean': statistics.mean(mu_B.values()),
        'sigma_A_mean': statistics.mean(sigma_A.values()),
        'sigma_B_mean': statistics.mean(sigma_B.values()),
        'mu_total_mean': statistics.mean(mu_total_vals),
        'mu_total_min': min(mu_total_vals),
        'mu_total_max': max(mu_total_vals),
        'sigma_total_mean': statistics.mean(sigma_total_vals),
        'sigma_total_min': min(sigma_total_vals),
        'sigma_total_max': max(sigma_total_vals),
        'k_buffer_mean': statistics.mean(k_buffer_vals),
        'k_buffer_min': min(k_buffer_vals),
        'k_buffer_max': max(k_buffer_vals),
        'p_robust_mean': statistics.mean(p_robust.values()),
        'p_robust_min': min(p_robust.values()),
        'p_robust_max': max(p_robust.values()),
        'total_nominal_load': total_nominal_load,
        'total_robust_load': total_robust_load,
        'avg_nominal_load_per_day': total_nominal_load / weekday_days,
        'avg_robust_load_per_day': total_robust_load / weekday_days,
        'daily_nominal_utilization_ratio': (total_nominal_load / weekday_days) / C_std,
        'daily_robust_utilization_ratio': (total_robust_load / weekday_days) / C_std,
    }


def evaluate_one_case(*, n_jobs, mu_scale, sigma_scale, k, replications, random_seed,
                      weekday_days, weekend_extension_days, C_std, Cost_OT, Cost_fix, M,
                      gurobi_time_limit_sec, weekend_fixed_cost, weekend_variable_cost,
                      maintenance_map=None, candidate_days=None,
                      unscheduled_weeks_policy='random', output_flag=0):
    gen_cfg = JobGenerationConfig(
        mu_A_mean=90.0, mu_A_std=6.0,
        mu_B_mean=60.0, mu_B_std=5.0,
        sigma_A_mean=14.0, sigma_A_std=2.0,
        low_B_fraction=0.70,
        high_B_fraction=1.50,
    )
    generated = generate_job_parameters(
        n_jobs=n_jobs,
        mu_scale=mu_scale,
        sigma_scale=sigma_scale,
        seed=random_seed,
        config=gen_cfg,
    )
    jobs   = generated['jobs']
    days   = list(range(1, weekday_days + 1))
    mu_A   = generated['mu_A']
    mu_B   = generated['mu_B']
    low_B  = generated['low_B']
    high_B = generated['high_B']
    sigma_A = generated['sigma_A']
    sigma_B = generated['sigma_B']  # derived from triangular params; used in planning layer only

    p_nominal, p_robust = compute_robust_processing_times(
        mu_A=mu_A, mu_B=mu_B, sigma_A=sigma_A, sigma_B=sigma_B, k=k
    )

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
        gurobi_result.sorted_schedule is not None and gurobi_result.solve_time_sec <= gurobi_time_limit_sec
    )

    # Build policy here so maintenance_duration is available for the row dict
    # regardless of whether simulation runs.
    policy = SimulationPolicy(
        regular_shift=480.0,
        weekday_horizon_days=weekday_days,
        weekend_extension_days=weekend_extension_days,
        weekend_fixed_cost=weekend_fixed_cost,
        weekend_variable_cost=weekend_variable_cost,
        maintenance_duration=120.0,
    )
    stop_cfg = MachineStopConfig(
        mean_uptime_between_stops=68.57,
        mean_stop_duration=8.0,
        stop_duration_cv=1.0,
    )

    row: dict[str, Any] = {
        'n_jobs': n_jobs,
        'weekday_days': weekday_days,
        'weekend_extension_days': weekend_extension_days,
        'maintenance_duration': policy.maintenance_duration,
        'mu_scale': mu_scale,
        'sigma_scale': sigma_scale,
        'k': k,
        'gurobi_status': gurobi_result.status,
        'solve_time_sec': gurobi_result.solve_time_sec,
        'solve_wall_time_sec': solve_wall_time,
        'accepted_for_simulation': accepted_for_simulation,
        'objective_value': gurobi_result.objective_value,
        'planned_total_ot': gurobi_result.planned_total_ot,
        'n_days_with_planned_ot': gurobi_result.n_days_with_planned_ot,
    }
    row.update(summarize_instance_parameters(
        mu_A=mu_A, mu_B=mu_B, sigma_A=sigma_A, sigma_B=sigma_B,
        p_nominal=p_nominal, p_robust=p_robust, k=k,
        weekday_days=weekday_days, C_std=C_std,
    ))

    if not accepted_for_simulation:
        row.update({
            'prob_cleared_within_weekdays': None,
            'prob_cleared_within_extended_horizon': None,
            'prob_terminal_backlog_after_extension': None,
            'avg_terminal_backlog_after_extension': None,
            'avg_max_backlog': None,
            'avg_n_spillover_days': None,
            'prob_any_spillover': None,
            'avg_first_spillover_day': None,
            'avg_n_weekend_days_used': None,
            'prob_any_weekend_use': None,
            'avg_weekend_used_time': None,
            'avg_total_weekend_cost': None,
            'avg_final_completion_day': None,
            'system_ok': False,
        })
        return row

    summary = monte_carlo_breakdown_analysis(
        schedule_dict=gurobi_result.sorted_schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        low_B=low_B,
        high_B=high_B,
        policy=policy,
        stop_cfg=stop_cfg,
        n_replications=replications,
        base_seed=random_seed,
        maintenance_map=maintenance_map,
        candidate_days=candidate_days,
        unscheduled_weeks_policy=unscheduled_weeks_policy,
    )
    row.update({
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
    })
    row['system_ok'] = accepted_for_simulation and summary['prob_cleared_within_extended_horizon'] is not None and summary['prob_cleared_within_extended_horizon'] >= 0.95
    return row


def main():
    parser = argparse.ArgumentParser(description='Run weekend-extension grid search with maintenance-aware execution.')
    parser.add_argument('--n-values', type=str, default='20,40,60,80,100', help='Comma-separated job counts.')
    parser.add_argument('--mu-scales', type=str, default='1.0', help='Comma-separated mu scaling factors.')
    parser.add_argument('--sigma-scales', type=str, default='1.0', help='Comma-separated sigma scaling factors.')
    parser.add_argument('--k-values', type=str, default='0.5,1.0,1.5,2.0', help='Comma-separated k values.')
    parser.add_argument('--replications', type=int, default=DEFAULT_REPLICATIONS, help='Monte Carlo replications for accepted cases.')
    parser.add_argument('--out', type=str, default='results/grid_search/grid_search_results_weekend_extension.csv', help='Output CSV path.')
    parser.add_argument('--weekday-days', type=int, default=20, help='Number of weekday bins planned by Gurobi.')
    parser.add_argument('--weekend-extension-days', type=int, default=8, help='Number of weekend extension bins available in simulation.')
    parser.add_argument('--weekend-fixed-cost', type=float, default=300.0, help='Fixed cost for activating one weekend extension day.')
    parser.add_argument('--weekend-variable-cost', type=float, default=8.0, help='Variable cost per minute used in a weekend extension day.')
    parser.add_argument('--gurobi-time-limit-sec', type=float, default=900.0, help='Maximum solve time for Gurobi.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
    parser.add_argument('--output-flag', type=int, default=0, help='Gurobi OutputFlag (0 or 1).')
    parser.add_argument(
        '--maintenance', type=str, default=None,
        help=(
            "Maintenance schedule mode. Options: "
            "'random' = fully random; "
            "'candidates' = use DEFAULT_MAINTENANCE_CANDIDATE_DAYS; "
            "'day:s:e,...' = fixed schedule; "
            "'week:days;...' = constrained-random e.g. '1:4,5;2:1,2,3'. "
            "If omitted, uses defaults from src/utils/maintenance.py."
        ),
    )
    parser.add_argument(
        '--unscheduled', type=str, default=None,
        choices=['random', 'skip'],
        help=(
            "What to do with weeks not listed in a constrained-random schedule. "
            "'random' = schedule on a random day; 'skip' = no maintenance that week. "
            "If omitted, uses DEFAULT_UNSCHEDULED_WEEKS_POLICY from src/utils/maintenance.py."
        ),
    )
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    mu_scales = parse_float_list(args.mu_scales)
    sigma_scales = parse_float_list(args.sigma_scales)
    k_values = parse_float_list(args.k_values)

    maintenance_map, candidate_days, unscheduled_weeks_policy = resolve_maintenance_map(
        args.maintenance, unscheduled_override=args.unscheduled
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_cases = len(n_values) * len(mu_scales) * len(sigma_scales) * len(k_values)
    case_idx = 0
    print('\n======================================')
    print('Weekend-Extension Grid Search Started')
    print('======================================')
    print(f'Total cases: {total_cases}')
    if maintenance_map is not None:
        print(f'Maintenance schedule (fixed): {maintenance_map}')
    elif candidate_days is not None:
        print(f'Maintenance schedule (constrained-random, unscheduled={unscheduled_weeks_policy}): {candidate_days}')
    else:
        print('Maintenance schedule: fully random (generated per replication from seed).')

    for n_jobs in n_values:
        for mu_scale in mu_scales:
            for sigma_scale in sigma_scales:
                for k in k_values:
                    case_idx += 1
                    print(f"\n[{case_idx}/{total_cases}] n_jobs={n_jobs}, mu_scale={mu_scale}, sigma_scale={sigma_scale}, k={k}")
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
                        maintenance_map=maintenance_map,
                        candidate_days=candidate_days,
                        unscheduled_weeks_policy=unscheduled_weeks_policy,
                        output_flag=args.output_flag,
                    )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print('\n======================================')
    print('Weekend-Extension Grid Search Finished')
    print('======================================')
    print(f'Saved results to: {out_path}')
    print(f'Rows: {len(df)}')


if __name__ == '__main__':
    main()
