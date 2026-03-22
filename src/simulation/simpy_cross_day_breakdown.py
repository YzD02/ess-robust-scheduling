"""
Cross-day execution validation with backlog propagation.

"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# =====================================================
# 1. CONFIGURATION DATACLASSES
# =====================================================

@dataclass(frozen=True)
class MachineStopConfig:
    """
    Configuration for machine-side disturbance generation at Station A.

    Parameters
    ----------
    mean_uptime_between_stops :
        Mean running time between two consecutive micro-stops.
    mean_stop_duration :
        Mean duration of one micro-stop.
    stop_duration_cv :
        Coefficient of variation of stop duration.
    """
    mean_uptime_between_stops: float
    mean_stop_duration: float
    stop_duration_cv: float


@dataclass(frozen=True)
class SimulationPolicy:
    """
    Simulation policy describing daily execution limits.

    Parameters
    ----------
    regular_shift :
        Regular daily available time.
    max_overtime_per_day :
        Maximum allowed overtime on top of regular shift.
    """
    regular_shift: float
    max_overtime_per_day: float

    @property
    def daily_available_time(self) -> float:
        return self.regular_shift + self.max_overtime_per_day


# =====================================================
# 2. OUTPUT DATACLASSES
# =====================================================

@dataclass
class DayExecutionResult:
    """
    Day-level execution result.

    Fields
    ------
    planned_jobs :
        Jobs originally assigned to this day by the planner.
    backlog_jobs_in :
        Jobs carried over from previous day.
    realized_queue :
        Actual execution queue = backlog + planned.
    executed_sequence :
        Jobs fully completed within the day.
    unfinished_sequence :
        Jobs rolled to the next day.
    job_details :
        Job-level realized details for this day.
    event_log :
        Flat event rows for Gantt export.
    """
    day_index: int
    planned_jobs: List[int]
    backlog_jobs_in: List[int]
    realized_queue: List[int]
    executed_sequence: List[int]
    unfinished_sequence: List[int]
    used_time: float
    overtime: float
    backlog_end_of_day: int
    job_details: Dict[int, Dict[str, Any]]
    event_log: List[Dict[str, Any]]


@dataclass
class HorizonExecutionResult:
    """
    Full-horizon execution result with both summary and detailed timeline data.
    """
    days: Dict[int, DayExecutionResult]
    planned_day_map: Dict[int, int]
    terminal_backlog_jobs: List[int]
    terminal_backlog_count: int
    total_overtime: float
    max_backlog: int
    system_cleared_within_horizon: bool
    spillover_days: List[int]
    n_spillover_days: int
    first_spillover_day: Optional[int]
    event_log: List[Dict[str, Any]]


# =====================================================
# 3. RANDOM HELPERS
# =====================================================

def sample_positive_normal(mu: float, sigma: float) -> float:
    """Draw a positive processing time from a truncated normal distribution."""
    return max(0.1, random.gauss(mu, sigma))


def sample_stop_duration(stop_cfg: MachineStopConfig) -> float:
    """
    Draw one stop duration from a Gamma distribution.

    Gamma is used because stop durations must be positive.
    """
    shape = 1.0 / (stop_cfg.stop_duration_cv ** 2)
    scale = stop_cfg.mean_stop_duration / shape
    return random.gammavariate(shape, scale)


def sample_time_to_next_stop(stop_cfg: MachineStopConfig) -> float:
    """Draw time until the next micro-stop."""
    return random.expovariate(1.0 / stop_cfg.mean_uptime_between_stops)


# =====================================================
# 4. JOB REALIZATION
# =====================================================

def realize_station_A_time(
    *,
    job_id: int,
    mu_A: Dict[int, float],
    stop_cfg: MachineStopConfig,
) -> tuple[float, float, int]:
    """
    Realize actual Station A processing time for one job.

    Logic
    -----
    Start with nominal processing time mu_A[job_id].
    While nominal work remains:
    - draw time to next stop
    - if stop occurs after completion -> finish
    - else process until stop, add stop delay, continue

    Returns
    -------
    actual_A_time :
        Realized total Station A time.
    total_stop_delay :
        Portion of A time caused by stop disruptions.
    stop_count :
        Number of stop events encountered.
    """
    nominal_remaining = mu_A[job_id]
    actual_A_time = 0.0
    total_stop_delay = 0.0
    stop_count = 0

    while nominal_remaining > 1e-8:
        time_to_stop = sample_time_to_next_stop(stop_cfg)

        if time_to_stop >= nominal_remaining:
            actual_A_time += nominal_remaining
            nominal_remaining = 0.0
        else:
            actual_A_time += time_to_stop
            nominal_remaining -= time_to_stop

            stop_duration = sample_stop_duration(stop_cfg)
            actual_A_time += stop_duration
            total_stop_delay += stop_duration
            stop_count += 1

    return actual_A_time, total_stop_delay, stop_count


def realize_station_B_time(
    *,
    job_id: int,
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
) -> float:
    """
    Realize actual Station B processing time.

    Station B is modeled as human-side variability using a truncated normal.
    """
    return sample_positive_normal(mu_B[job_id], sigma_B[job_id])


def realize_job_station_details(
    *,
    job_id: int,
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
    stop_cfg: MachineStopConfig,
) -> Dict[str, float | int]:
    """
    Realize station-level details for one job.

    Returns a dictionary that can later be used for:
    - summary reporting
    - event log export
    - Gantt chart visualization
    """
    actual_A_time, stop_delay, stop_count = realize_station_A_time(
        job_id=job_id,
        mu_A=mu_A,
        stop_cfg=stop_cfg,
    )
    actual_B_time = realize_station_B_time(
        job_id=job_id,
        mu_B=mu_B,
        sigma_B=sigma_B,
    )

    return {
        "actual_A_time": actual_A_time,
        "actual_B_time": actual_B_time,
        "total_time": actual_A_time + actual_B_time,
        "stop_delay": stop_delay,
        "stop_count": stop_count,
    }


# =====================================================
# 5. DAY EXECUTION WITH BACKLOG + EVENT LOG
# =====================================================

def execute_one_day_with_backlog(
    *,
    day_index: int,
    backlog_jobs: List[int],
    planned_jobs: List[int],
    policy: SimulationPolicy,
    planned_day_map: Dict[int, int],
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
    stop_cfg: MachineStopConfig,
    simulation_run: int = 1,
) -> DayExecutionResult:
    """
    Execute one day using backlog-first, non-preemptive logic.

    Policy
    ------
    1. Backlog jobs are executed before planned jobs.
    2. Jobs are non-preemptive.
    3. If the next job does not fit in the remaining daily time,
       it rolls entirely to the next day.

    This function records both:
    - compact summary outputs
    - detailed event log rows for Gantt charting
    """
    queue_today = list(backlog_jobs) + list(planned_jobs)
    remaining_time = float(policy.daily_available_time)
    used_time = 0.0

    executed_sequence: List[int] = []
    unfinished_sequence: List[int] = []
    job_details: Dict[int, Dict[str, Any]] = {}
    event_log: List[Dict[str, Any]] = []

    global_day_offset = (day_index - 1) * policy.daily_available_time

    for pos, job_id in enumerate(queue_today):
        detail = realize_job_station_details(
            job_id=job_id,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_B=sigma_B,
            stop_cfg=stop_cfg,
        )
        total_time = float(detail["total_time"])
        from_backlog = pos < len(backlog_jobs)

        if total_time <= remaining_time:
            # Local day timeline
            a_start_day = used_time
            a_end_day = a_start_day + float(detail["actual_A_time"])

            b_start_day = a_end_day
            b_end_day = b_start_day + float(detail["actual_B_time"])

            # Global horizon timeline
            a_start_global = global_day_offset + a_start_day
            a_end_global = global_day_offset + a_end_day
            b_start_global = global_day_offset + b_start_day
            b_end_global = global_day_offset + b_end_day

            used_time += total_time
            remaining_time -= total_time
            executed_sequence.append(job_id)

            planned_day = planned_day_map[job_id]
            executed_day = day_index
            was_delayed = executed_day > planned_day
            day_delay = executed_day - planned_day

            job_details[job_id] = {
                "planned_day": planned_day,
                "executed_day": executed_day,
                "from_backlog": from_backlog,
                "was_delayed": was_delayed,
                "day_delay": day_delay,
                "actual_A_time": float(detail["actual_A_time"]),
                "actual_B_time": float(detail["actual_B_time"]),
                "total_time": total_time,
                "stop_count": int(detail["stop_count"]),
                "stop_delay": float(detail["stop_delay"]),
                "a_start_day": a_start_day,
                "a_end_day": a_end_day,
                "b_start_day": b_start_day,
                "b_end_day": b_end_day,
            }

            # Event row for Station A
            event_log.append({
                "simulation_run": simulation_run,
                "planned_day": planned_day,
                "executed_day": executed_day,
                "job_id": job_id,
                "station": "A",
                "start_time_day": a_start_day,
                "end_time_day": a_end_day,
                "start_time_global": a_start_global,
                "end_time_global": a_end_global,
                "duration": float(detail["actual_A_time"]),
                "from_backlog": from_backlog,
                "was_delayed": was_delayed,
                "day_delay": day_delay,
                "stop_count": int(detail["stop_count"]),
                "stop_delay": float(detail["stop_delay"]),
            })

            # Event row for Station B
            event_log.append({
                "simulation_run": simulation_run,
                "planned_day": planned_day,
                "executed_day": executed_day,
                "job_id": job_id,
                "station": "B",
                "start_time_day": b_start_day,
                "end_time_day": b_end_day,
                "start_time_global": b_start_global,
                "end_time_global": b_end_global,
                "duration": float(detail["actual_B_time"]),
                "from_backlog": from_backlog,
                "was_delayed": was_delayed,
                "day_delay": day_delay,
                "stop_count": 0,
                "stop_delay": 0.0,
            })
        else:
            unfinished_sequence = queue_today[pos:]
            break

    overtime = max(0.0, used_time - policy.regular_shift)

    return DayExecutionResult(
        day_index=day_index,
        planned_jobs=list(planned_jobs),
        backlog_jobs_in=list(backlog_jobs),
        realized_queue=list(queue_today),
        executed_sequence=executed_sequence,
        unfinished_sequence=unfinished_sequence,
        used_time=used_time,
        overtime=overtime,
        backlog_end_of_day=len(unfinished_sequence),
        job_details=job_details,
        event_log=event_log,
    )


# =====================================================
# 6. HORIZON EXECUTION
# =====================================================

def simulate_horizon_with_backlog(
    *,
    schedule_dict: Dict[int, List[int]],
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
    policy: SimulationPolicy,
    stop_cfg: MachineStopConfig,
    seed: int | None = None,
    simulation_run: int = 1,
) -> HorizonExecutionResult:
    """
    Simulate the entire planning horizon with cross-day backlog propagation.

    Returns both:
    - compact summary information
    - full event log for downstream Gantt visualization
    """
    if seed is not None:
        random.seed(seed)

    all_days = sorted(schedule_dict.keys())

    planned_day_map: Dict[int, int] = {}
    for day, jobs in schedule_dict.items():
        for j in jobs:
            planned_day_map[j] = day

    backlog: List[int] = []
    outputs: Dict[int, DayExecutionResult] = {}
    full_event_log: List[Dict[str, Any]] = []

    total_overtime = 0.0
    max_backlog = 0
    spillover_days: List[int] = []

    for day in all_days:
        day_result = execute_one_day_with_backlog(
            day_index=day,
            backlog_jobs=backlog,
            planned_jobs=schedule_dict.get(day, []),
            policy=policy,
            planned_day_map=planned_day_map,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_B=sigma_B,
            stop_cfg=stop_cfg,
            simulation_run=simulation_run,
        )

        outputs[day] = day_result
        full_event_log.extend(day_result.event_log)

        total_overtime += day_result.overtime
        backlog = day_result.unfinished_sequence
        max_backlog = max(max_backlog, len(backlog))

        if day_result.backlog_end_of_day > 0:
            spillover_days.append(day)

    terminal_backlog_jobs = list(backlog)
    terminal_backlog_count = len(terminal_backlog_jobs)

    return HorizonExecutionResult(
        days=outputs,
        planned_day_map=planned_day_map,
        terminal_backlog_jobs=terminal_backlog_jobs,
        terminal_backlog_count=terminal_backlog_count,
        total_overtime=total_overtime,
        max_backlog=max_backlog,
        system_cleared_within_horizon=(terminal_backlog_count == 0),
        spillover_days=spillover_days,
        n_spillover_days=len(spillover_days),
        first_spillover_day=min(spillover_days) if spillover_days else None,
        event_log=full_event_log,
    )


# =====================================================
# 7. MONTE CARLO SUMMARY
# =====================================================

def monte_carlo_breakdown_analysis(
    *,
    schedule_dict: Dict[int, List[int]],
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
    policy: SimulationPolicy,
    stop_cfg: MachineStopConfig,
    n_replications: int = 200,
    base_seed: int = 42,
) -> Dict[str, float | Dict[int, Dict[str, float]]]:
    """
    Repeat the cross-day simulation many times and summarize breakdown risk.

    This function intentionally stores only summary metrics, not all event logs,
    so that large grid searches remain lightweight.
    """
    total_overtime_list: List[float] = []
    terminal_backlog_list: List[int] = []
    max_backlog_list: List[int] = []
    cleared_flags: List[bool] = []
    n_spillover_days_list: List[int] = []
    first_spillover_day_list: List[int] = []

    day_overtime = {day: [] for day in schedule_dict}
    day_backlog_end = {day: [] for day in schedule_dict}

    for r in range(n_replications):
        out = simulate_horizon_with_backlog(
            schedule_dict=schedule_dict,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_B=sigma_B,
            policy=policy,
            stop_cfg=stop_cfg,
            seed=base_seed + r,
            simulation_run=r + 1,
        )

        total_overtime_list.append(out.total_overtime)
        terminal_backlog_list.append(out.terminal_backlog_count)
        max_backlog_list.append(out.max_backlog)
        cleared_flags.append(out.system_cleared_within_horizon)
        n_spillover_days_list.append(out.n_spillover_days)
        if out.first_spillover_day is not None:
            first_spillover_day_list.append(out.first_spillover_day)

        for day in schedule_dict:
            day_overtime[day].append(out.days[day].overtime)
            day_backlog_end[day].append(out.days[day].backlog_end_of_day)

    summary: Dict[str, float | Dict[int, Dict[str, float]] | None] = {
        "avg_total_overtime": statistics.mean(total_overtime_list),
        "max_total_overtime": max(total_overtime_list),
        "prob_terminal_backlog": sum(1 for x in terminal_backlog_list if x > 0) / n_replications,
        "avg_terminal_backlog": statistics.mean(terminal_backlog_list),
        "avg_max_backlog": statistics.mean(max_backlog_list),
        "prob_cleared_within_horizon": sum(1 for x in cleared_flags if x) / n_replications,
        "avg_n_spillover_days": statistics.mean(n_spillover_days_list),
        "prob_any_spillover": sum(1 for x in n_spillover_days_list if x > 0) / n_replications,
        "avg_first_spillover_day": (
            statistics.mean(first_spillover_day_list) if first_spillover_day_list else None
        ),
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


# =====================================================
# 8. EXPORT / PRINT HELPERS
# =====================================================

def event_log_to_dataframe(event_log: List[Dict[str, Any]]):
    """
    Convert event log into a pandas DataFrame.

    Imported lazily so pandas is only required when export is needed.
    """
    import pandas as pd
    return pd.DataFrame(event_log)


def print_single_run_result(out: HorizonExecutionResult) -> None:
    """Pretty-print one sample path."""
    print("\n==============================")
    print("Single Sample-Path Result")
    print("==============================")

    for day in sorted(out.days.keys()):
        d = out.days[day]
        print(f"\nDay {day}")
        print(f"  Planned jobs       : {d.planned_jobs}")
        print(f"  Backlog input      : {d.backlog_jobs_in}")
        print(f"  Realized queue     : {d.realized_queue}")
        print(f"  Executed jobs      : {d.executed_sequence}")
        print(f"  Unfinished jobs    : {d.unfinished_sequence}")
        print(f"  Used time          : {d.used_time:.2f}")
        print(f"  Overtime           : {d.overtime:.2f}")
        print(f"  End-of-day backlog : {d.backlog_end_of_day}")

    print("\nSpillover days        :", out.spillover_days)
    print("Number of spillovers  :", out.n_spillover_days)
    print("First spillover day   :", out.first_spillover_day)
    print("Terminal backlog jobs :", out.terminal_backlog_jobs)
    print("Terminal backlog count:", out.terminal_backlog_count)
    print("Total overtime        :", round(out.total_overtime, 2))
    print("Max backlog observed  :", out.max_backlog)
    print("Cleared within horizon:", out.system_cleared_within_horizon)


def print_monte_carlo_summary(
    summary: Dict[str, float | Dict[int, Dict[str, float]]],
    n_replications: int,
) -> None:
    """Pretty-print Monte Carlo summary."""
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
    print(f"Average spillover days          : {summary['avg_n_spillover_days']:.2f}")
    print(f"Probability of any spillover    : {summary['prob_any_spillover']:.2%}")
    print(f"Average first spillover day     : {summary['avg_first_spillover_day']}")

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