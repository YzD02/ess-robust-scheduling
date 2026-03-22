"""Run one fully documented end-to-end experiment.

This script is the quickest way to verify that the package works.
It performs four steps:

1. Define one stress-test instance.
2. Compute robust processing times.
3. Solve the baseline Gurobi planning model.
4. Validate the resulting schedule with cross-day breakdown simulation.

Run from the repository root:
    python -m src.experiments.run_single_case
"""

from __future__ import annotations

from src.models.gurobi_baseline import solve_gurobi_baseline
from src.models.robust_processing import compute_robust_processing_times
from src.simulation.simpy_cross_day_breakdown import (
    monte_carlo_breakdown_analysis,
    print_monte_carlo_summary,
    print_single_run_result,
    simulate_horizon_with_backlog,
)


def main() -> None:
    # -------------------------
    # Stress-test toy instance
    # -------------------------
    jobs = list(range(1, 16))
    days = list(range(1, 6))

    # Station A: Automated Busbar Assembly
    mu_A = {
        1: 88,  2: 80,  3: 96,  4: 84,  5: 92,
        6: 76,  7: 104, 8: 86,  9: 94, 10: 78,
        11: 100, 12: 87, 13: 90, 14: 79, 15: 102,
    }

    # Station B: Manual Harness Alignment and Riveting
    mu_B = {
        1: 62,  2: 54,  3: 68,  4: 58,  5: 60,
        6: 50,  7: 74,  8: 55,  9: 66, 10: 52,
        11: 72, 12: 57, 13: 59, 14: 51, 15: 70,
    }

    # sigma_A = machine-side variability at Station A
    sigma_A = {
        1: 14, 2: 12, 3: 15, 4: 13, 5: 14,
        6: 11, 7: 17, 8: 12, 9: 15, 10: 11,
        11: 16, 12: 12, 13: 13, 14: 11, 15: 16,
    }

    # sigma_B = human-side variability at Station B
    sigma_B = {
        1: 18, 2: 16, 3: 20, 4: 17, 5: 18,
        6: 15, 7: 22, 8: 16, 9: 19, 10: 15,
        11: 21, 12: 16, 13: 17, 14: 15, 15: 21,
    }

    # Planning and simulation parameters
    C_std = 480.0
    Cost_OT = 5.0
    Cost_fix = 180.0
    M = 1200.0
    k = 1.0

    regular_shift = 480.0
    max_overtime_per_day = 120.0

    # -------------------------
    # Robust processing times
    # -------------------------
    p_nominal, p_robust = compute_robust_processing_times(
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        k=k,
    )

    print("\nRobust processing times")
    print("=======================")
    for j in jobs:
        print(
            f"Job {j:2d}: nominal = {p_nominal[j]:7.2f}, "
            f"robust = {p_robust[j]:7.2f}"
        )

    # -------------------------
    # Gurobi planning layer
    # -------------------------
    result = solve_gurobi_baseline(
        jobs=jobs,
        days=days,
        p_robust=p_robust,
        C_std=C_std,
        Cost_OT=Cost_OT,
        Cost_fix=Cost_fix,
        M=M,
        output_flag=1,
    )

    print("\nGurobi planning result")
    print("======================")
    print(f"status          : {result.status}")
    print(f"solve time (s)  : {result.solve_time_sec:.2f}")
    print(f"objective value : {result.objective_value}")

    if result.sorted_schedule is None:
        print("No feasible solution was returned by Gurobi.")
        return

    print("\nRaw day assignment:")
    for t in days:
        day_jobs = result.raw_schedule[t]
        robust_load = sum(p_robust[j] for j in day_jobs)
        print(
            f"Day {t}: jobs = {day_jobs}, "
            f"robust load = {robust_load:.2f}, "
            f"OT = {result.overtime_by_day[t]:.2f}, "
            f"active = {result.active_by_day[t]}"
        )

    print("\nPost-processed within-day sequence (shortest robust job first):")
    for t in days:
        print(f"Day {t}: {result.sorted_schedule[t]}")

    # -------------------------
    # Cross-day simulation layer
    # -------------------------
    one_run = simulate_horizon_with_backlog(
        schedule_dict=result.sorted_schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        regular_shift=regular_shift,
        max_overtime_per_day=max_overtime_per_day,
        seed=42,
    )
    print_single_run_result(one_run)

    summary = monte_carlo_breakdown_analysis(
        schedule_dict=result.sorted_schedule,
        mu_A=mu_A,
        mu_B=mu_B,
        sigma_A=sigma_A,
        sigma_B=sigma_B,
        regular_shift=regular_shift,
        max_overtime_per_day=max_overtime_per_day,
        n_replications=100,
        base_seed=42,
    )
    print_monte_carlo_summary(summary, n_replications=100)


if __name__ == "__main__":
    main()
