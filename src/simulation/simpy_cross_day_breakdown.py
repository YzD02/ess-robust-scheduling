"""Cross-day execution validation with backlog propagation.

Why this simulation exists
--------------------------
The Gurobi model plans at the day level. It decides which jobs are assigned to
which working day, but it does not simulate the physical execution process.
This file provides the execution layer.

Key policy assumptions
----------------------
1. Each day has regular shift time plus an overtime cap.
2. Jobs are non-preemptive: if the next job cannot fit into the remaining time,
   that job is not partially processed. It rolls to the next day.
3. Backlog-first dispatching: unfinished jobs from previous days are executed
   before that day's originally planned jobs.
4. Station A and Station B are represented in an *aggregated* realized job time
   to keep this simulation aligned with the day-level robust bin-packing model.

Breakdown interpretation
------------------------
If the horizon ends and there are still unfinished jobs, terminal backlog is
positive. This is the natural signal of cross-day breakdown.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DayExecutionResult:
    day_index: int
    executed_sequence: List[int]
    unfinished_sequence: List[int]
    used_time: float
    overtime: float
    backlog_end_of_day: int
    job_details: Dict[int, Dict[str, float | int | bool]]


@dataclass
class HorizonExecutionResult:
    days: Dict[int, DayExecutionResult]
    terminal_backlog_jobs: List[int]
    terminal_backlog_count: int
    total_overtime: float
    max_backlog: int
    system_cleared_within_horizon: bool


def sample_positive_normal(mu: float, sigma: float) -> float:
    """Draw a positive processing time from a truncated normal distribution."""
    return max(0.1, random.gauss(mu, sigma))


def realize_total_job_time(
    *,
    job_id: int,
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_A: Dict[int, float],
    sigma_B: Dict[int, float],
) -> tuple[float, Dict[str, float]]:
    """Generate one realized processing time for a job.

    Current modeling choice
    -----------------------
    We use an aggregated realization:
        realized_job_time = realized_A + realized_B

    This keeps the simulation aligned with the planning abstraction, where each
    job consumes one aggregated daily workload amount.
    """
    actual_A = sample_positive_normal(mu_A[job_id], sigma_A[job_id])
    actual_B = sample_positive_normal(mu_B[job_id], sigma_B[job_id])
    total = actual_A + actual_B

    return total, {
        "actual_A_time": actual_A,
        "actual_B_time": actual_B,
        "total_time": total,
    }


def execute_one_day_with_backlog(
    *,
    day_index: int,
    backlog_jobs: List[int],
    planned_jobs: List[int],
    time_budget: float,
    regular_shift: float,
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_A: Dict[int, float],
    sigma_B: Dict[int, float],
) -> DayExecutionResult:
    """Execute one day under backlog-first, non-preemptive logic.

    Inputs
    ------
    backlog_jobs:
        Jobs carried over from previous days.
    planned_jobs:
        Jobs originally assigned to this day by Gurobi.
    time_budget:
        Maximum executable time this day = regular shift + overtime cap.
    regular_shift:
        Regular shift length used to compute overtime.

    Output
    ------
    DayExecutionResult with executed jobs, unfinished jobs, used time, and
    detailed realized processing information.
    """
    queue_today = list(backlog_jobs) + list(planned_jobs)
    remaining_time = float(time_budget)
    used_time = 0.0

    executed_sequence: List[int] = []
    unfinished_sequence: List[int] = []
    job_details: Dict[int, Dict[str, float | int | bool]] = {}

    for pos, job_id in enumerate(queue_today):
        realized_time, detail = realize_total_job_time(
            job_id=job_id,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
        )

        if realized_time <= remaining_time:
            start_time = used_time
            end_time = used_time + realized_time

            used_time += realized_time
            remaining_time -= realized_time

            detail.update(
                {
                    "day": day_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "from_backlog": pos < len(backlog_jobs),
                }
            )

            executed_sequence.append(job_id)
            job_details[job_id] = detail
        else:
            unfinished_sequence = queue_today[pos:]
            break

    overtime = max(0.0, used_time - regular_shift)

    return DayExecutionResult(
        day_index=day_index,
        executed_sequence=executed_sequence,
        unfinished_sequence=unfinished_sequence,
        used_time=used_time,
        overtime=overtime,
        backlog_end_of_day=len(unfinished_sequence),
        job_details=job_details,
    )


def simulate_horizon_with_backlog(
    *,
    schedule_dict: Dict[int, List[int]],
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_A: Dict[int, float],
    sigma_B: Dict[int, float],
    regular_shift: float,
    max_overtime_per_day: float,
    seed: int | None = None,
) -> HorizonExecutionResult:
    """Simulate the entire horizon with cross-day backlog propagation."""
    if seed is not None:
        random.seed(seed)

    all_days = sorted(schedule_dict.keys())
    backlog: List[int] = []
    outputs: Dict[int, DayExecutionResult] = {}
    total_overtime = 0.0
    max_backlog = 0

    daily_available_time = regular_shift + max_overtime_per_day

    for day in all_days:
        day_result = execute_one_day_with_backlog(
            day_index=day,
            backlog_jobs=backlog,
            planned_jobs=schedule_dict.get(day, []),
            time_budget=daily_available_time,
            regular_shift=regular_shift,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
        )
        outputs[day] = day_result
        total_overtime += day_result.overtime
        backlog = day_result.unfinished_sequence
        max_backlog = max(max_backlog, len(backlog))

    terminal_backlog_jobs = list(backlog)
    terminal_backlog_count = len(terminal_backlog_jobs)

    return HorizonExecutionResult(
        days=outputs,
        terminal_backlog_jobs=terminal_backlog_jobs,
        terminal_backlog_count=terminal_backlog_count,
        total_overtime=total_overtime,
        max_backlog=max_backlog,
        system_cleared_within_horizon=(terminal_backlog_count == 0),
    )


def monte_carlo_breakdown_analysis(
    *,
    schedule_dict: Dict[int, List[int]],
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_A: Dict[int, float],
    sigma_B: Dict[int, float],
    regular_shift: float,
    max_overtime_per_day: float,
    n_replications: int = 200,
    base_seed: int = 42,
) -> Dict[str, float | Dict[int, Dict[str, float]]]:
    """Repeat the cross-day simulation many times and summarize breakdown risk."""
    total_overtime_list: List[float] = []
    terminal_backlog_list: List[int] = []
    max_backlog_list: List[int] = []
    cleared_flags: List[bool] = []

    day_overtime = {day: [] for day in schedule_dict}
    day_backlog_end = {day: [] for day in schedule_dict}

    for r in range(n_replications):
        out = simulate_horizon_with_backlog(
            schedule_dict=schedule_dict,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_A=sigma_A,
            sigma_B=sigma_B,
            regular_shift=regular_shift,
            max_overtime_per_day=max_overtime_per_day,
            seed=base_seed + r,
        )

        total_overtime_list.append(out.total_overtime)
        terminal_backlog_list.append(out.terminal_backlog_count)
        max_backlog_list.append(out.max_backlog)
        cleared_flags.append(out.system_cleared_within_horizon)

        for day in schedule_dict:
            day_overtime[day].append(out.days[day].overtime)
            day_backlog_end[day].append(out.days[day].backlog_end_of_day)

    summary: Dict[str, float | Dict[int, Dict[str, float]]] = {
        "avg_total_overtime": statistics.mean(total_overtime_list),
        "max_total_overtime": max(total_overtime_list),
        "prob_terminal_backlog": sum(1 for x in terminal_backlog_list if x > 0) / n_replications,
        "avg_terminal_backlog": statistics.mean(terminal_backlog_list),
        "avg_max_backlog": statistics.mean(max_backlog_list),
        "prob_cleared_within_horizon": sum(1 for x in cleared_flags if x) / n_replications,
        "day_level": {},
    }

    day_level: Dict[int, Dict[str, float]] = {}
    for day in sorted(schedule_dict.keys()):
        day_level[day] = {
            "avg_overtime": statistics.mean(day_overtime[day]),
            "prob_overtime": sum(1 for x in day_overtime[day] if x > 0) / n_replications,
            "avg_end_backlog": statistics.mean(day_backlog_end[day]),
            "prob_end_backlog_positive": sum(1 for x in day_backlog_end[day] if x > 0) / n_replications,
        }
    summary["day_level"] = day_level

    return summary


def print_single_run_result(out: HorizonExecutionResult) -> None:
    """Pretty-print one sample path for debugging and reporting."""
    print("\n==============================")
    print("Single Sample-Path Result")
    print("==============================")

    for day in sorted(out.days.keys()):
        d = out.days[day]
        print(f"\nDay {day}")
        print(f"  Executed jobs      : {d.executed_sequence}")
        print(f"  Unfinished jobs    : {d.unfinished_sequence}")
        print(f"  Used time          : {d.used_time:.2f}")
        print(f"  Overtime           : {d.overtime:.2f}")
        print(f"  End-of-day backlog : {d.backlog_end_of_day}")
        for j in d.executed_sequence:
            info = d.job_details[j]
            print(
                f"    Job {j}: "
                f"start={info['start_time']:.2f}, "
                f"end={info['end_time']:.2f}, "
                f"A={info['actual_A_time']:.2f}, "
                f"B={info['actual_B_time']:.2f}, "
                f"from_backlog={info['from_backlog']}"
            )

    print("\nTerminal backlog jobs :", out.terminal_backlog_jobs)
    print("Terminal backlog count:", out.terminal_backlog_count)
    print("Total overtime        :", round(out.total_overtime, 2))
    print("Max backlog observed  :", out.max_backlog)
    print("Cleared within horizon:", out.system_cleared_within_horizon)


def print_monte_carlo_summary(summary: Dict[str, float | Dict[int, Dict[str, float]]], n_replications: int) -> None:
    """Pretty-print Monte Carlo summary results."""
    print("\n==============================")
    print("Monte Carlo Breakdown Summary")
    print("==============================")
    print(f"Replications: {n_replications}")
    print(f"Average total overtime          : {summary['avg_total_overtime']:.2f}")
    print(f"Max total overtime              : {summary['max_total_overtime']:.2f}")
    print(f"Probability of terminal backlog : {summary['prob_terminal_backlog']:.2%}")
    print(f"Average terminal backlog size   : {summary['avg_terminal_backlog']:.2f}")
    print(f"Average max backlog             : {summary['avg_max_backlog']:.2f}")
    print(f"Probability cleared in horizon  : {summary['prob_cleared_within_horizon']:.2%}")

    print("\nPer-day summary:")
    for day in sorted(summary["day_level"].keys()):
        d = summary["day_level"][day]
        print(
            f"Day {day}: "
            f"avg overtime = {d['avg_overtime']:.2f}, "
            f"overtime probability = {d['prob_overtime']:.2%}, "
            f"avg end backlog = {d['avg_end_backlog']:.2f}, "
            f"prob end backlog > 0 = {d['prob_end_backlog_positive']:.2%}"
        )
