from __future__ import annotations

"""
Cross-day execution simulation with weekday horizon + weekend extension bins.

Main logic
----------
1. The planned schedule covers weekday bins only (e.g., 20 days).
2. During execution:
   - each day can process at most `regular_shift` minutes
   - no same-day overtime completion is allowed
   - if the next job cannot fully fit in the remaining daily time, it is deferred
3. After the weekday horizon is exhausted, optional weekend extension bins
   are activated to absorb remaining backlog.
4. Weekend usage is tracked and converted into an additional extension cost.

This module preserves:
- compact summary outputs for grid-search / heatmaps
- detailed event logs for Gantt chart generation
"""

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# =====================================================
# 1. CONFIGURATION DATACLASSES
# =====================================================

@dataclass(frozen=True)
class MachineStopConfig:
    """Configuration for Station A micro-stop process."""
    mean_uptime_between_stops: float
    mean_stop_duration: float
    stop_duration_cv: float


@dataclass(frozen=True)
class SimulationPolicy:
    """
    Execution policy.

    Parameters
    ----------
    regular_shift :
        Available production time for one execution day.
    weekday_horizon_days :
        Number of planned weekday bins, e.g. 20.
    weekend_extension_days :
        Number of reserve weekend bins, e.g. 8.
    weekend_fixed_cost :
        Fixed cost for activating one weekend extension day.
    weekend_variable_cost :
        Variable cost per minute used in a weekend extension day.
    """
    regular_shift: float
    weekday_horizon_days: int
    weekend_extension_days: int
    weekend_fixed_cost: float = 300.0
    weekend_variable_cost: float = 8.0


# =====================================================
# 2. OUTPUT DATACLASSES
# =====================================================

@dataclass
class DayExecutionResult:
    """Result for one executed day."""
    day_index: int
    day_type: str                  # "weekday" or "weekend"
    planned_jobs: List[int]
    backlog_jobs_in: List[int]
    realized_queue: List[int]
    executed_sequence: List[int]
    unfinished_sequence: List[int]
    used_time: float
    backlog_end_of_day: int
    job_details: Dict[int, Dict[str, Any]]
    event_log: List[Dict[str, Any]]
    weekend_day_used: bool


@dataclass
class HorizonExecutionResult:
    """Full-horizon result."""
    days: Dict[int, DayExecutionResult]
    planned_day_map: Dict[int, int]
    terminal_backlog_jobs: List[int]
    terminal_backlog_count: int
    max_backlog: int
    cleared_within_weekdays: bool
    cleared_within_extended_horizon: bool
    spillover_days: List[int]
    n_spillover_days: int
    first_spillover_day: Optional[int]
    weekend_days_used: List[int]
    n_weekend_days_used: int
    weekend_used_time: float
    weekend_fixed_cost: float
    weekend_variable_cost: float
    total_weekend_cost: float
    final_completion_day: Optional[int]
    event_log: List[Dict[str, Any]]


# =====================================================
# 3. RANDOM HELPERS
# =====================================================

def sample_positive_normal(mu: float, sigma: float) -> float:
    return max(0.1, random.gauss(mu, sigma))


def sample_stop_duration(stop_cfg: MachineStopConfig) -> float:
    shape = 1.0 / (stop_cfg.stop_duration_cv ** 2)
    scale = stop_cfg.mean_stop_duration / shape
    return random.gammavariate(shape, scale)


def sample_time_to_next_stop(stop_cfg: MachineStopConfig) -> float:
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
    Realize actual Station A time via explicit stop process.
    Returns: (actual_A_time, total_stop_delay, stop_count)
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
    return sample_positive_normal(mu_B[job_id], sigma_B[job_id])


def realize_job_station_details(
    *,
    job_id: int,
    mu_A: Dict[int, float],
    mu_B: Dict[int, float],
    sigma_B: Dict[int, float],
    stop_cfg: MachineStopConfig,
) -> Dict[str, float | int]:
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
# 5. DAY EXECUTION
# =====================================================

def execute_one_day_with_backlog(
    *,
    day_index: int,
    day_type: str,
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
    Execute one day under strict no-overtime completion logic.

    Rule
    ----
    If the next job cannot be fully completed within the remaining `regular_shift`
    time of the current day, it is deferred entirely to the next day.
    """
    queue_today = list(backlog_jobs) + list(planned_jobs)
    remaining_time = float(policy.regular_shift)
    used_time = 0.0

    executed_sequence: List[int] = []
    unfinished_sequence: List[int] = []
    job_details: Dict[int, Dict[str, Any]] = {}
    event_log: List[Dict[str, Any]] = []

    global_day_offset = (day_index - 1) * policy.regular_shift

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

        # Strict no-overtime completion:
        # if this whole job does not fit into the remaining time, push it forward.
        if total_time <= remaining_time:
            a_start_day = used_time
            a_end_day = a_start_day + float(detail["actual_A_time"])

            b_start_day = a_end_day
            b_end_day = b_start_day + float(detail["actual_B_time"])

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
                "day_type": day_type,
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

            event_log.append({
                "simulation_run": simulation_run,
                "planned_day": planned_day,
                "executed_day": executed_day,
                "day_type": day_type,
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

            event_log.append({
                "simulation_run": simulation_run,
                "planned_day": planned_day,
                "executed_day": executed_day,
                "day_type": day_type,
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

    return DayExecutionResult(
        day_index=day_index,
        day_type=day_type,
        planned_jobs=list(planned_jobs),
        backlog_jobs_in=list(backlog_jobs),
        realized_queue=list(queue_today),
        executed_sequence=executed_sequence,
        unfinished_sequence=unfinished_sequence,
        used_time=used_time,
        backlog_end_of_day=len(unfinished_sequence),
        job_details=job_details,
        event_log=event_log,
        weekend_day_used=(day_type == "weekend" and used_time > 0),
    )


# =====================================================
# 6. HORIZON EXECUTION WITH WEEKEND EXTENSION
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
    Simulate:
    - weekday horizon first
    - then optional weekend extension bins

    Weekday days:
        1 ... policy.weekday_horizon_days
    Weekend extension days:
        weekday_horizon_days + 1 ... weekday_horizon_days + weekend_extension_days

    Weekend bins contain backlog only, no new planned jobs.
    """
    if seed is not None:
        random.seed(seed)

    all_weekdays = sorted(schedule_dict.keys())

    planned_day_map: Dict[int, int] = {}
    for day, jobs in schedule_dict.items():
        for j in jobs:
            planned_day_map[j] = day

    backlog: List[int] = []
    outputs: Dict[int, DayExecutionResult] = {}
    full_event_log: List[Dict[str, Any]] = []

    max_backlog = 0
    spillover_days: List[int] = []
    weekend_days_used: List[int] = []
    weekend_used_time = 0.0

    # ---------------------------
    # Phase 1: planned weekdays
    # ---------------------------
    for day in all_weekdays:
        day_result = execute_one_day_with_backlog(
            day_index=day,
            day_type="weekday",
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

        backlog = day_result.unfinished_sequence
        max_backlog = max(max_backlog, len(backlog))

        if day_result.backlog_end_of_day > 0:
            spillover_days.append(day)

    cleared_within_weekdays = (len(backlog) == 0)

    # ---------------------------
    # Phase 2: weekend extension
    # ---------------------------
    first_weekend_day = policy.weekday_horizon_days + 1
    last_weekend_day = policy.weekday_horizon_days + policy.weekend_extension_days

    for day in range(first_weekend_day, last_weekend_day + 1):
        if len(backlog) == 0:
            break

        day_result = execute_one_day_with_backlog(
            day_index=day,
            day_type="weekend",
            backlog_jobs=backlog,
            planned_jobs=[],   # weekend bins only process backlog
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

        backlog = day_result.unfinished_sequence
        max_backlog = max(max_backlog, len(backlog))

        if day_result.weekend_day_used:
            weekend_days_used.append(day)
            weekend_used_time += day_result.used_time

        if day_result.backlog_end_of_day > 0:
            spillover_days.append(day)

    terminal_backlog_jobs = list(backlog)
    terminal_backlog_count = len(terminal_backlog_jobs)

    cleared_within_extended_horizon = (terminal_backlog_count == 0)

    weekend_fixed_cost = len(weekend_days_used) * policy.weekend_fixed_cost
    weekend_variable_cost = weekend_used_time * policy.weekend_variable_cost
    total_weekend_cost = weekend_fixed_cost + weekend_variable_cost

    final_completion_day = None
    if full_event_log:
        final_completion_day = max(int(r["executed_day"]) for r in full_event_log)

    return HorizonExecutionResult(
        days=outputs,
        planned_day_map=planned_day_map,
        terminal_backlog_jobs=terminal_backlog_jobs,
        terminal_backlog_count=terminal_backlog_count,
        max_backlog=max_backlog,
        cleared_within_weekdays=cleared_within_weekdays,
        cleared_within_extended_horizon=cleared_within_extended_horizon,
        spillover_days=spillover_days,
        n_spillover_days=len(spillover_days),
        first_spillover_day=min(spillover_days) if spillover_days else None,
        weekend_days_used=weekend_days_used,
        n_weekend_days_used=len(weekend_days_used),
        weekend_used_time=weekend_used_time,
        weekend_fixed_cost=weekend_fixed_cost,
        weekend_variable_cost=weekend_variable_cost,
        total_weekend_cost=total_weekend_cost,
        final_completion_day=final_completion_day,
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
) -> Dict[str, Any]:
    """
    Monte Carlo summary for the weekday + weekend-extension policy.

    Keeps only compact summary statistics.
    """
    terminal_backlog_list: List[int] = []
    max_backlog_list: List[int] = []
    cleared_weekdays_flags: List[bool] = []
    cleared_extended_flags: List[bool] = []

    n_spillover_days_list: List[int] = []
    first_spillover_day_list: List[int] = []

    n_weekend_days_used_list: List[int] = []
    weekend_used_time_list: List[float] = []
    total_weekend_cost_list: List[float] = []
    final_completion_day_list: List[int] = []

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

        terminal_backlog_list.append(out.terminal_backlog_count)
        max_backlog_list.append(out.max_backlog)
        cleared_weekdays_flags.append(out.cleared_within_weekdays)
        cleared_extended_flags.append(out.cleared_within_extended_horizon)

        n_spillover_days_list.append(out.n_spillover_days)
        if out.first_spillover_day is not None:
            first_spillover_day_list.append(out.first_spillover_day)

        n_weekend_days_used_list.append(out.n_weekend_days_used)
        weekend_used_time_list.append(out.weekend_used_time)
        total_weekend_cost_list.append(out.total_weekend_cost)
        if out.final_completion_day is not None:
            final_completion_day_list.append(out.final_completion_day)

        for day in schedule_dict:
            if day in out.days:
                day_backlog_end[day].append(out.days[day].backlog_end_of_day)
            else:
                day_backlog_end[day].append(0)

    summary: Dict[str, Any] = {
        "prob_terminal_backlog_after_extension": sum(1 for x in terminal_backlog_list if x > 0) / n_replications,
        "avg_terminal_backlog_after_extension": statistics.mean(terminal_backlog_list),
        "avg_max_backlog": statistics.mean(max_backlog_list),
        "prob_cleared_within_weekdays": sum(1 for x in cleared_weekdays_flags if x) / n_replications,
        "prob_cleared_within_extended_horizon": sum(1 for x in cleared_extended_flags if x) / n_replications,
        "avg_n_spillover_days": statistics.mean(n_spillover_days_list),
        "prob_any_spillover": sum(1 for x in n_spillover_days_list if x > 0) / n_replications,
        "avg_first_spillover_day": (
            statistics.mean(first_spillover_day_list) if first_spillover_day_list else None
        ),
        "avg_n_weekend_days_used": statistics.mean(n_weekend_days_used_list),
        "prob_any_weekend_use": sum(1 for x in n_weekend_days_used_list if x > 0) / n_replications,
        "avg_weekend_used_time": statistics.mean(weekend_used_time_list),
        "avg_total_weekend_cost": statistics.mean(total_weekend_cost_list),
        "avg_final_completion_day": (
            statistics.mean(final_completion_day_list) if final_completion_day_list else None
        ),
        "day_level": {},
    }

    day_level: Dict[int, Dict[str, float]] = {}
    for day in sorted(schedule_dict.keys()):
        day_level[day] = {
            "avg_end_backlog": statistics.mean(day_backlog_end[day]),
            "prob_end_backlog_positive": sum(1 for x in day_backlog_end[day] if x > 0) / n_replications,
        }
    summary["day_level"] = day_level

    return summary


# =====================================================
# 8. EXPORT / PRINT HELPERS
# =====================================================

def event_log_to_dataframe(event_log: List[Dict[str, Any]]):
    import pandas as pd
    return pd.DataFrame(event_log)


def print_single_run_result(out: HorizonExecutionResult) -> None:
    print("\n==============================")
    print("Single Sample-Path Result")
    print("==============================")

    for day in sorted(out.days.keys()):
        d = out.days[day]
        print(f"\nDay {day} ({d.day_type})")
        print(f"  Planned jobs       : {d.planned_jobs}")
        print(f"  Backlog input      : {d.backlog_jobs_in}")
        print(f"  Realized queue     : {d.realized_queue}")
        print(f"  Executed jobs      : {d.executed_sequence}")
        print(f"  Unfinished jobs    : {d.unfinished_sequence}")
        print(f"  Used time          : {d.used_time:.2f}")
        print(f"  End-of-day backlog : {d.backlog_end_of_day}")

    print("\nSpillover days              :", out.spillover_days)
    print("Number of spillover days    :", out.n_spillover_days)
    print("First spillover day         :", out.first_spillover_day)
    print("Weekend days used           :", out.weekend_days_used)
    print("Number of weekend days used :", out.n_weekend_days_used)
    print("Weekend used time           :", round(out.weekend_used_time, 2))
    print("Weekend fixed cost          :", round(out.weekend_fixed_cost, 2))
    print("Weekend variable cost       :", round(out.weekend_variable_cost, 2))
    print("Total weekend cost          :", round(out.total_weekend_cost, 2))
    print("Final completion day        :", out.final_completion_day)
    print("Terminal backlog jobs       :", out.terminal_backlog_jobs)
    print("Terminal backlog count      :", out.terminal_backlog_count)
    print("Cleared within weekdays     :", out.cleared_within_weekdays)
    print("Cleared within extension    :", out.cleared_within_extended_horizon)


def print_monte_carlo_summary(summary: Dict[str, Any], n_replications: int) -> None:
    print("\n==============================")
    print("Monte Carlo Breakdown Summary")
    print("==============================")
    print(f"Replications                         : {n_replications}")
    print(f"Prob cleared within weekdays         : {summary['prob_cleared_within_weekdays']:.2%}")
    print(f"Prob cleared within extended horizon : {summary['prob_cleared_within_extended_horizon']:.2%}")
    print(f"Prob terminal backlog after extension: {summary['prob_terminal_backlog_after_extension']:.2%}")
    print(f"Avg terminal backlog after extension : {summary['avg_terminal_backlog_after_extension']:.2f}")
    print(f"Avg max backlog                      : {summary['avg_max_backlog']:.2f}")
    print(f"Avg spillover days                   : {summary['avg_n_spillover_days']:.2f}")
    print(f"Prob any spillover                   : {summary['prob_any_spillover']:.2%}")
    print(f"Avg first spillover day              : {summary['avg_first_spillover_day']}")
    print(f"Avg weekend days used                : {summary['avg_n_weekend_days_used']:.2f}")
    print(f"Prob any weekend use                 : {summary['prob_any_weekend_use']:.2%}")
    print(f"Avg weekend used time                : {summary['avg_weekend_used_time']:.2f}")
    print(f"Avg total weekend cost               : {summary['avg_total_weekend_cost']:.2f}")
    print(f"Avg final completion day             : {summary['avg_final_completion_day']}")

    print("\nPer-weekday summary:")
    for day in sorted(summary["day_level"].keys()):
        d = summary["day_level"][day]
        print(
            f"Day {day}: "
            f"avg end backlog = {d['avg_end_backlog']:.2f}, "
            f"prob end backlog > 0 = {d['prob_end_backlog_positive']:.2%}"
        )