from __future__ import annotations

"""Cross-day execution simulation with weekday horizon, weekend extension,
and maintenance-aware scheduling.

Module name note
----------------
This file was previously named ``simpy_cross_day_breakdown.py``.  It was
renamed to ``simulation_engine.py`` because the implementation does **not**
use the SimPy discrete-event library; it is a plain Python loop-based
simulator.  The old name was misleading to new contributors.

Station A vs Station B uncertainty — design decision
------------------------------------------------------
``sigma_A`` (machine-side uncertainty at Station A) is used in the **planning
layer** (``robust_processing.py``) to inflate the robust processing time.
It is intentionally **not** passed into this simulation layer.

In the simulation, Station A variability is instead modelled through a
**machine-stop process**: the automated busbar assembly machine experiences
random micro-stops whose inter-arrival times and durations follow exponential
and gamma distributions respectively (see ``MachineStopConfig``).  This is a
more realistic representation of automated-equipment behaviour than additive
Gaussian noise.

Station B (manual harness alignment / riveting) retains a Gaussian model
because human-side variability is well approximated by a normal distribution.

Main execution rules
--------------------
1. The planning model assigns jobs to weekday bins only.
2. In execution, each day has only ``regular_shift`` minutes available.
3. No same-day overtime is allowed; unfinished jobs roll to the next day.
4. A weekly 2-hour maintenance window is scheduled at a random time.
   If starting a job would overlap the maintenance window, the job is
   deferred until maintenance ends.
5. After all weekdays are exhausted, optional weekend extension bins absorb
   any remaining backlog.
"""

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MachineStopConfig:
    mean_uptime_between_stops: float
    mean_stop_duration: float
    stop_duration_cv: float


@dataclass(frozen=True)
class SimulationPolicy:
    regular_shift: float
    weekday_horizon_days: int
    weekend_extension_days: int
    weekend_fixed_cost: float = 300.0
    weekend_variable_cost: float = 8.0
    maintenance_duration: float = 120.0


@dataclass
class DayExecutionResult:
    day_index: int
    day_type: str
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
    maintenance_start_day: float | None = None
    maintenance_end_day: float | None = None


@dataclass
class HorizonExecutionResult:
    days: Dict[int, DayExecutionResult]
    planned_day_map: Dict[int, int]
    maintenance_map: Dict[int, tuple[float, float]]
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


def sample_positive_normal(mu: float, sigma: float) -> float:
    return max(0.1, random.gauss(mu, sigma))


def sample_stop_duration(stop_cfg: MachineStopConfig) -> float:
    shape = 1.0 / (stop_cfg.stop_duration_cv ** 2)
    scale = stop_cfg.mean_stop_duration / shape
    return random.gammavariate(shape, scale)


def sample_time_to_next_stop(stop_cfg: MachineStopConfig) -> float:
    return random.expovariate(1.0 / stop_cfg.mean_uptime_between_stops)


def generate_daily_maintenance_windows(
    *,
    weekday_horizon_days: int,
    weekend_extension_days: int,
    maintenance_duration: float = 120.0,
    shift_length: float = 480.0,
    seed: int | None = None,
) -> dict[int, tuple[float, float]]:
    """Generate one random maintenance window per 5-day work week."""
    if seed is not None:
        rng = random.Random(seed + 100_000)
    else:
        rng = random.Random()

    maintenance_map: dict[int, tuple[float, float]] = {}
    n_weeks = weekday_horizon_days // 5
    for w in range(n_weeks):
        week_days = list(range(w * 5 + 1, w * 5 + 6))
        chosen_day = rng.choice(week_days)
        start = rng.uniform(0.0, shift_length - maintenance_duration)
        end = start + maintenance_duration
        maintenance_map[chosen_day] = (start, end)
    return maintenance_map


def push_start_past_maintenance(
    *,
    current_time: float,
    total_job_time: float,
    maintenance_start: float | None,
    maintenance_end: float | None,
) -> float:
    """Delay the full job if it would overlap the maintenance window."""
    if maintenance_start is None or maintenance_end is None:
        return current_time
    if current_time >= maintenance_end:
        return current_time
    if current_time + total_job_time <= maintenance_start:
        return current_time
    return maintenance_end


def realize_station_A_time(*, job_id: int, mu_A: Dict[int, float], stop_cfg: MachineStopConfig) -> tuple[float, float, int]:
    """Simulate actual Station A processing time, including random machine stops.

    The automated busbar assembly machine (Station A) runs for a random
    amount of time, then experiences a micro-stop, then resumes — repeating
    until the nominal job work is done.  The actual wall-clock time is
    therefore longer than the nominal time by however long the stops lasted.

    Returns
    -------
    actual_A_time   : total elapsed time including all stop delays (minutes)
    total_stop_delay: cumulative time lost to stops (minutes)
    stop_count      : number of stops that occurred during this job
    """
    nominal_remaining = mu_A[job_id]  # work content still to be processed
    actual_A_time = 0.0
    total_stop_delay = 0.0
    stop_count = 0

    while nominal_remaining > 1e-8:
        # How long until the next stop?
        time_to_stop = sample_time_to_next_stop(stop_cfg)
        if time_to_stop >= nominal_remaining:
            # The machine completes the remaining work before its next stop.
            actual_A_time += nominal_remaining
            nominal_remaining = 0.0
        else:
            # A stop occurs before the job is finished — advance time through
            # the productive window, then add the stop duration.
            actual_A_time += time_to_stop
            nominal_remaining -= time_to_stop
            stop_duration = sample_stop_duration(stop_cfg)
            actual_A_time += stop_duration
            total_stop_delay += stop_duration
            stop_count += 1
    return actual_A_time, total_stop_delay, stop_count


def realize_station_B_time(*, job_id: int, mu_B: Dict[int, float], sigma_B: Dict[int, float]) -> float:
    return sample_positive_normal(mu_B[job_id], sigma_B[job_id])


def realize_job_station_details(*, job_id: int, mu_A: Dict[int, float], mu_B: Dict[int, float], sigma_B: Dict[int, float], stop_cfg: MachineStopConfig) -> Dict[str, float | int]:
    actual_A_time, stop_delay, stop_count = realize_station_A_time(job_id=job_id, mu_A=mu_A, stop_cfg=stop_cfg)
    actual_B_time = realize_station_B_time(job_id=job_id, mu_B=mu_B, sigma_B=sigma_B)
    return {
        'actual_A_time': actual_A_time,
        'actual_B_time': actual_B_time,
        'total_time': actual_A_time + actual_B_time,
        'stop_delay': stop_delay,
        'stop_count': stop_count,
    }


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
    maintenance_start_day: float | None = None,
    maintenance_end_day: float | None = None,
    simulation_run: int = 1,
) -> DayExecutionResult:
    """Simulate execution for a single day, respecting backlog priority and maintenance.

    Execution rules
    ---------------
    1. Backlog jobs (carried over from previous days) are processed first.
    2. Planned jobs follow in shortest-robust-time-first order.
    3. Jobs are non-preemptive: a job either completes in full during this
       shift or is pushed entirely to the next day — no splitting.
    4. If a job would overlap the maintenance window, its start is pushed
       to the end of maintenance.
    5. No overtime: the shift ends at ``policy.regular_shift`` minutes.
       Any jobs remaining at that point go into tomorrow's backlog.
    """
    # Backlog jobs always go first — they have been waiting longest.
    queue_today = list(backlog_jobs) + list(planned_jobs)
    used_time = 0.0
    remaining_time = float(policy.regular_shift)

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
        total_time = float(detail['total_time'])
        from_backlog = pos < len(backlog_jobs)

        candidate_start = push_start_past_maintenance(
            current_time=used_time,
            total_job_time=total_time,
            maintenance_start=maintenance_start_day,
            maintenance_end=maintenance_end_day,
        )
        candidate_finish = candidate_start + total_time

        if candidate_finish <= policy.regular_shift:
            a_start_day = candidate_start
            a_end_day = a_start_day + float(detail['actual_A_time'])
            b_start_day = a_end_day
            b_end_day = b_start_day + float(detail['actual_B_time'])

            a_start_global = global_day_offset + a_start_day
            a_end_global = global_day_offset + a_end_day
            b_start_global = global_day_offset + b_start_day
            b_end_global = global_day_offset + b_end_day

            used_time = b_end_day
            remaining_time = policy.regular_shift - used_time
            executed_sequence.append(job_id)

            planned_day = planned_day_map[job_id]
            executed_day = day_index
            was_delayed = executed_day > planned_day
            day_delay = executed_day - planned_day

            job_details[job_id] = {
                'planned_day': planned_day,
                'executed_day': executed_day,
                'day_type': day_type,
                'from_backlog': from_backlog,
                'was_delayed': was_delayed,
                'day_delay': day_delay,
                'actual_A_time': float(detail['actual_A_time']),
                'actual_B_time': float(detail['actual_B_time']),
                'total_time': total_time,
                'stop_count': int(detail['stop_count']),
                'stop_delay': float(detail['stop_delay']),
                'a_start_day': a_start_day,
                'a_end_day': a_end_day,
                'b_start_day': b_start_day,
                'b_end_day': b_end_day,
                'maintenance_start_day': maintenance_start_day,
                'maintenance_end_day': maintenance_end_day,
            }

            common = {
                'simulation_run': simulation_run,
                'planned_day': planned_day,
                'executed_day': executed_day,
                'day_type': day_type,
                'job_id': job_id,
                'from_backlog': from_backlog,
                'was_delayed': was_delayed,
                'day_delay': day_delay,
                'maintenance_start_day': maintenance_start_day,
                'maintenance_end_day': maintenance_end_day,
            }

            event_log.append({
                **common,
                'station': 'A',
                'start_time_day': a_start_day,
                'end_time_day': a_end_day,
                'start_time_global': a_start_global,
                'end_time_global': a_end_global,
                'duration': float(detail['actual_A_time']),
                'stop_count': int(detail['stop_count']),
                'stop_delay': float(detail['stop_delay']),
            })
            event_log.append({
                **common,
                'station': 'B',
                'start_time_day': b_start_day,
                'end_time_day': b_end_day,
                'start_time_global': b_start_global,
                'end_time_global': b_end_global,
                'duration': float(detail['actual_B_time']),
                'stop_count': 0,
                'stop_delay': 0.0,
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
        weekend_day_used=(day_type == 'weekend' and used_time > 0),
        maintenance_start_day=maintenance_start_day,
        maintenance_end_day=maintenance_end_day,
    )


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
    maintenance_map: dict | None = None,
) -> HorizonExecutionResult:
    """Simulate one complete sample path across the full planning horizon.

    Runs through all 20 weekdays in sequence, carrying any unfinished jobs
    forward as backlog.  If backlog remains after day 20, up to
    ``policy.weekend_extension_days`` extra days are used to clear it.

    Each call represents one possible future — the randomness comes from
    stochastic job times and machine stops.  To estimate probabilities, call
    this function many times with different seeds (see
    ``monte_carlo_breakdown_analysis``).

    Parameters
    ----------
    schedule_dict :
        Gurobi output — maps day index → list of job ids planned for that day.
    seed :
        Random seed for this replication.  Pass a different seed each call
        to get statistically independent sample paths.
    simulation_run :
        Run index (1-based).  Stored in the event log for identification.
    maintenance_map :
        Optional pre-built maintenance schedule mapping day index →
        (start_minute, end_minute).  If provided, this overrides random
        generation entirely — the same fixed windows are used in every
        replication.  If None (default), windows are generated randomly
        using the seed.

        Example (Week 1 Wednesday 12:00-14:00, Week 2 Thursday 08:00-10:00,
        assuming an 08:00 shift start):
            {3: (240, 360), 9: (0, 120)}
    """
    if seed is not None:
        random.seed(seed)

    all_weekdays = sorted(schedule_dict.keys())
    # Build a reverse map: job id → the day it was originally planned for.
    # Used later to measure how many days each job was delayed.
    planned_day_map: Dict[int, int] = {j: day for day, jobs in schedule_dict.items() for j in jobs}

    # Use the caller-supplied maintenance map if provided; otherwise generate
    # one random window per work week from the seed.
    if maintenance_map is not None:
        _maintenance_map = maintenance_map
    else:
        _maintenance_map = generate_daily_maintenance_windows(
            weekday_horizon_days=policy.weekday_horizon_days,
            weekend_extension_days=policy.weekend_extension_days,
            maintenance_duration=policy.maintenance_duration,
            shift_length=policy.regular_shift,
            seed=seed,
        )

    backlog: List[int] = []         # jobs not finished on their scheduled day
    outputs: Dict[int, DayExecutionResult] = {}
    full_event_log: List[Dict[str, Any]] = []
    max_backlog = 0
    spillover_days: List[int] = []  # days that ended with at least one job unfinished
    weekend_days_used: List[int] = []
    weekend_used_time = 0.0

    # ---- Phase 1: work through all planned weekdays ----
    for day in all_weekdays:
        m_window = _maintenance_map.get(day, (None, None))
        day_result = execute_one_day_with_backlog(
            day_index=day,
            day_type='weekday',
            backlog_jobs=backlog,
            planned_jobs=schedule_dict.get(day, []),
            policy=policy,
            planned_day_map=planned_day_map,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_B=sigma_B,
            stop_cfg=stop_cfg,
            maintenance_start_day=m_window[0],
            maintenance_end_day=m_window[1],
            simulation_run=simulation_run,
        )
        outputs[day] = day_result
        full_event_log.extend(day_result.event_log)
        backlog = day_result.unfinished_sequence
        max_backlog = max(max_backlog, len(backlog))
        if day_result.backlog_end_of_day > 0:
            spillover_days.append(day)

    cleared_within_weekdays = len(backlog) == 0

    # ---- Phase 2: weekend recovery (only if backlog remains) ----
    # Weekend days have no maintenance and no planned jobs — they exist purely
    # to absorb leftover backlog.  We stop as soon as backlog hits zero.
    first_weekend_day = policy.weekday_horizon_days + 1
    last_weekend_day = policy.weekday_horizon_days + policy.weekend_extension_days
    for day in range(first_weekend_day, last_weekend_day + 1):
        if len(backlog) == 0:
            break
        day_result = execute_one_day_with_backlog(
            day_index=day,
            day_type='weekend',
            backlog_jobs=backlog,
            planned_jobs=[],
            policy=policy,
            planned_day_map=planned_day_map,
            mu_A=mu_A,
            mu_B=mu_B,
            sigma_B=sigma_B,
            stop_cfg=stop_cfg,
            maintenance_start_day=None,
            maintenance_end_day=None,
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

    # ---- Summarise the full horizon outcome ----
    terminal_backlog_jobs = list(backlog)
    terminal_backlog_count = len(terminal_backlog_jobs)
    cleared_within_extended_horizon = terminal_backlog_count == 0
    weekend_fixed_cost = len(weekend_days_used) * policy.weekend_fixed_cost
    weekend_variable_cost = weekend_used_time * policy.weekend_variable_cost
    total_weekend_cost = weekend_fixed_cost + weekend_variable_cost
    final_completion_day = max((int(r['executed_day']) for r in full_event_log), default=None)

    return HorizonExecutionResult(
        days=outputs,
        planned_day_map=planned_day_map,
        maintenance_map=_maintenance_map,
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
    maintenance_map: dict | None = None,
) -> Dict[str, Any]:
    """Run many independent simulations and aggregate the results.

    Calls ``simulate_horizon_with_backlog`` n_replications times, each with
    a different random seed, then computes summary statistics across all runs.

    The key output metrics are:
      prob_cleared_within_weekdays          — fraction of runs where all jobs
                                              finished before the weekend
      prob_cleared_within_extended_horizon  — fraction of runs where all jobs
                                              finished within the full horizon
                                              (weekdays + weekend extension)
      avg_n_weekend_days_used               — average number of weekend days
                                              consumed across all runs
      avg_final_completion_day              — average day on which the last
                                              job was completed

    A schedule is considered reliable if
    ``prob_cleared_within_extended_horizon >= 0.95``.

    Parameters
    ----------
    maintenance_map :
        Optional fixed maintenance schedule (see ``simulate_horizon_with_backlog``
        for format).  If provided, the same windows are used in every replication.
        If None, each replication generates its own random windows from its seed.
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
            maintenance_map=maintenance_map,
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
            day_backlog_end[day].append(out.days[day].backlog_end_of_day if day in out.days else 0)

    summary: Dict[str, Any] = {
        'prob_terminal_backlog_after_extension': sum(1 for x in terminal_backlog_list if x > 0) / n_replications,
        'avg_terminal_backlog_after_extension': statistics.mean(terminal_backlog_list),
        'avg_max_backlog': statistics.mean(max_backlog_list),
        'prob_cleared_within_weekdays': sum(1 for x in cleared_weekdays_flags if x) / n_replications,
        'prob_cleared_within_extended_horizon': sum(1 for x in cleared_extended_flags if x) / n_replications,
        'avg_n_spillover_days': statistics.mean(n_spillover_days_list),
        'prob_any_spillover': sum(1 for x in n_spillover_days_list if x > 0) / n_replications,
        'avg_first_spillover_day': statistics.mean(first_spillover_day_list) if first_spillover_day_list else None,
        'avg_n_weekend_days_used': statistics.mean(n_weekend_days_used_list),
        'prob_any_weekend_use': sum(1 for x in n_weekend_days_used_list if x > 0) / n_replications,
        'avg_weekend_used_time': statistics.mean(weekend_used_time_list),
        'avg_total_weekend_cost': statistics.mean(total_weekend_cost_list),
        'avg_final_completion_day': statistics.mean(final_completion_day_list) if final_completion_day_list else None,
        'day_level': {},
    }
    day_level: Dict[int, Dict[str, float]] = {}
    for day in sorted(schedule_dict.keys()):
        day_level[day] = {
            'avg_end_backlog': statistics.mean(day_backlog_end[day]),
            'prob_end_backlog_positive': sum(1 for x in day_backlog_end[day] if x > 0) / n_replications,
        }
    summary['day_level'] = day_level
    return summary


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
        if d.maintenance_start_day is not None:
            print(f"  Maintenance window : [{d.maintenance_start_day:.2f}, {d.maintenance_end_day:.2f}]")
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
    for day in sorted(summary['day_level'].keys()):
        d = summary['day_level'][day]
        print(f"Day {day}: avg end backlog = {d['avg_end_backlog']:.2f}, prob end backlog > 0 = {d['prob_end_backlog_positive']:.2%}")
