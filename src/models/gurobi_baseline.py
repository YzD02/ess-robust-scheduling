"""Baseline robust bin-packing model solved with Gurobi.

What this module does
---------------------
Uses the Gurobi mathematical optimisation solver to assign all jobs to working
days, minimising overtime cost.

Think of it as an intelligent bin-packing problem:
  - Each job has a "planned processing time" (already inflated by the safety buffer)
  - Each working day is a "bin" with 960 minutes of regular capacity (two shifts)
  - If a day overflows, overtime can be added — but overtime costs money
  - Goal: assign every job to exactly one day while minimising total overtime cost

Output
------
  sorted_schedule   dict mapping day number → list of job ids assigned to that day,
                    ordered by shortest-robust-time first within each day (this order
                    is used by the simulation to sequence jobs within a day)
  planned_total_ot  total planned overtime minutes across the full horizon

Mathematical model
------------------
Decision variables:
  z[j, t]   binary — 1 if job j is assigned to day t, 0 otherwise
  OT[t]     continuous — overtime minutes used on day t
  u[t]      binary — 1 if overtime is activated on day t, 0 otherwise

Objective:
  minimise  sum_t ( Cost_fix * u[t]  +  Cost_OT * OT[t] )

Constraints:
  1) Full assignment:    every job appears in exactly one day
  2) Robust daily cap:  sum of robust job times on day t <= C_std + OT[t]
  3) OT activation:     OT[t] = 0 when u[t] = 0  (indicator constraint,
                        replaces the old big-M formulation)
  4) OT upper bound:    OT[t] <= M_tight * u[t]   (tight per-day ceiling)
  5) Global OT cut:     sum_t OT[t] >= max(0, total_robust_load - T*C_std)
                        (valid lower bound, tightens the LP relaxation)
  6) Symmetry breaking: non-increasing daily load ordering
                        (eliminates permutation-equivalent packings)

Solver settings applied
-----------------------
  Symmetry=2   — Gurobi's internal symmetry detection (aggressive)
  MIPGap       — configurable; default 0.01 (1%) as per Johnny's recommendation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gurobipy import GRB, Model, quicksum


@dataclass
class GurobiBaselineResult:
    """Container for optimization outputs."""

    status: int
    solve_time_sec: float
    objective_value: float | None
    raw_schedule: Dict[int, List[int]] | None
    sorted_schedule: Dict[int, List[int]] | None
    planned_total_ot: float | None
    n_days_with_planned_ot: int | None
    overtime_by_day: Dict[int, float] | None
    active_by_day: Dict[int, int] | None
    mip_gap: float | None = None
    model_name: str = "ESS_Robust_BinPacking"


def solve_gurobi_baseline(
    *,
    jobs: List[int],
    days: List[int],
    p_robust: Dict[int, float],
    C_std: float,
    Cost_OT: float,
    Cost_fix: float,
    M: float,                          # kept for backward compatibility; replaced internally
    time_limit_sec: float | None = None,
    output_flag: int = 1,
    mip_gap: float = 0.01,             # 1% gap — operationally indistinguishable from optimal
    use_indicator_constraints: bool = True,   # replaces big-M with indicator constraint
    add_symmetry_breaking: bool = True,       # non-increasing daily load ordering
    add_global_ot_cut: bool = True,           # valid lower bound on total overtime
) -> GurobiBaselineResult:
    """Solve the baseline day-assignment model with tightened formulation.

    Improvements over the original big-M model (per Johnny's recommendations):

    1. Tight M / indicator constraint
       The original M=50,000 made the LP relaxation of OT[t] <= M*u[t] nearly
       vacuous, starving branch-and-bound of a usable bound.  By default this is
       replaced with a Gurobi indicator constraint (u[t]=0 => OT[t]=0), which
       avoids the weak relaxation entirely.  A tight per-day OT ceiling is also
       added as a fallback upper bound.

    2. Global overtime lower-bound cut
       sum_t OT[t] >= max(0, total_robust_load - T*C_std) is a valid lower bound
       on total overtime.  It tightens the objective bound right where it matters
       for high-utilisation cases like n=125.

    3. Symmetry breaking
       All 20 planning days are interchangeable bins, so the solver wastes effort
       on permutations of the same schedule.  Adding a non-increasing daily load
       ordering constraint (load[t] >= load[t+1]) eliminates most symmetric
       branches.  Gurobi's built-in Symmetry=2 is also enabled.

    4. MIPGap = 0.01
       The schedule is fed into a Monte Carlo simulation, so a plan provably within
       1% of optimal is operationally indistinguishable from the true optimum.
       Reporting the achieved gap alongside the objective is honest and sufficient.

    Notes
    -----
    The returned `sorted_schedule` uses a shortest-robust-processing-time-first
    post-processing rule inside each day. This does not change the optimisation
    objective. It simply provides a feasible execution order for the simulation.

    The `M` parameter is kept in the signature for backward compatibility with
    existing callers (run_grid_search.py, run_single_case.py, etc.) but is no
    longer used when use_indicator_constraints=True (the default).
    """
    model = Model("ESS_Robust_BinPacking")
    model.setParam("OutputFlag", output_flag)

    if time_limit_sec is not None:
        model.setParam("TimeLimit", float(time_limit_sec))

    # MIPGap: 1% is operationally indistinguishable from optimal once the
    # schedule enters Monte Carlo simulation.
    model.setParam("MIPGap", mip_gap)

    # Aggressive symmetry detection — works in tandem with the explicit
    # ordering constraints below.
    model.setParam("Symmetry", 2)

    # -------------------------
    # Decision variables
    # -------------------------
    z = model.addVars(jobs, days, vtype=GRB.BINARY, name="z")
    OT = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="OT")
    u = model.addVars(days, vtype=GRB.BINARY, name="u")

    # Daily load auxiliary variable (needed for symmetry-breaking constraint).
    load = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="load")

    # -------------------------
    # Objective
    # -------------------------
    model.setObjective(
        quicksum(Cost_fix * u[t] + Cost_OT * OT[t] for t in days),
        GRB.MINIMIZE,
    )

    # -------------------------
    # Constraint 1: Full assignment
    # Every job must appear in exactly one day.
    # -------------------------
    for j in jobs:
        model.addConstr(
            quicksum(z[j, t] for t in days) == 1,
            name=f"assign_{j}",
        )

    # -------------------------
    # Constraint 2: Robust daily capacity
    # -------------------------
    for t in days:
        model.addConstr(
            quicksum(p_robust[j] * z[j, t] for j in jobs) <= C_std + OT[t],
            name=f"capacity_{t}",
        )

    # -------------------------
    # Constraint 2b: Load definition (for symmetry-breaking)
    # -------------------------
    for t in days:
        model.addConstr(
            load[t] == quicksum(p_robust[j] * z[j, t] for j in jobs),
            name=f"load_def_{t}",
        )

    # -------------------------
    # Constraint 3: Overtime activation logic
    #
    # Two approaches — indicator constraint (preferred) or tight big-M.
    # The indicator constraint avoids the weak LP relaxation of the original
    # M=50,000 formulation entirely.
    # -------------------------
    # Tight per-day OT ceiling: a single job can never exceed max(p_robust),
    # so the worst-case single-day overtime is bounded by:
    #   M_tight = max(p_robust) * n_jobs_per_day_max — C_std
    # In practice, well under 1,000 min for our problem.
    p_max = max(p_robust.values())
    M_tight = max(p_max * len(jobs), C_std)   # conservative upper bound, << 50,000

    for t in days:
        # Hard upper bound on per-day overtime (always added).
        model.addConstr(OT[t] <= M_tight * u[t], name=f"ot_ub_{t}")

        if use_indicator_constraints:
            # Indicator: if u[t] == 0 then OT[t] == 0.
            # This gives Gurobi a tight bound without a numerical big-M.
            model.addGenConstrIndicator(
                u[t], False, OT[t], GRB.EQUAL, 0.0,
                name=f"ot_indicator_{t}",
            )

    # -------------------------
    # Constraint 4: Global overtime lower-bound cut  (Johnny item 1b)
    # sum_t OT[t] >= max(0, total_robust_load - T * C_std)
    # This is always valid and tightens the LP relaxation at the objective level.
    # -------------------------
    if add_global_ot_cut:
        total_robust_load = sum(p_robust.values())
        T = len(days)
        min_total_ot = max(0.0, total_robust_load - T * C_std)
        if min_total_ot > 0:
            model.addConstr(
                quicksum(OT[t] for t in days) >= min_total_ot,
                name="global_ot_lower_bound",
            )

    # -------------------------
    # Constraint 5: Symmetry breaking — non-increasing daily load  (Johnny item 2)
    # load[t] >= load[t+1] for all consecutive day pairs.
    # This eliminates permutation-equivalent packings (e.g. day 1 heavy / day 2
    # light is treated as identical to day 1 light / day 2 heavy).
    # -------------------------
    if add_symmetry_breaking:
        sorted_days = sorted(days)
        for i in range(len(sorted_days) - 1):
            t_curr = sorted_days[i]
            t_next = sorted_days[i + 1]
            model.addConstr(
                load[t_curr] >= load[t_next],
                name=f"sym_break_{t_curr}_{t_next}",
            )

    model.optimize()

    status = int(model.status)
    solve_time_sec = float(model.Runtime)

    objective_value = None
    raw_schedule = None
    sorted_schedule = None
    planned_total_ot = None
    n_days_with_planned_ot = None
    overtime_by_day = None
    active_by_day = None
    achieved_gap = None

    if model.SolCount > 0:
        objective_value = float(model.objVal)
        # MIPGap attribute only available when a feasible solution exists.
        try:
            achieved_gap = float(model.MIPGap)
        except Exception:
            achieved_gap = None

        raw_schedule = {t: [j for j in jobs if z[j, t].X > 0.5] for t in days}
        sorted_schedule = {
            t: sorted(raw_schedule[t], key=lambda j: p_robust[j])
            for t in days
        }
        overtime_by_day = {t: float(OT[t].X) for t in days}
        active_by_day = {t: int(round(u[t].X)) for t in days}
        planned_total_ot = sum(overtime_by_day.values())
        n_days_with_planned_ot = sum(1 for t in days if overtime_by_day[t] > 1e-6)

    return GurobiBaselineResult(
        status=status,
        solve_time_sec=solve_time_sec,
        objective_value=objective_value,
        raw_schedule=raw_schedule,
        sorted_schedule=sorted_schedule,
        planned_total_ot=planned_total_ot,
        n_days_with_planned_ot=n_days_with_planned_ot,
        overtime_by_day=overtime_by_day,
        active_by_day=active_by_day,
        mip_gap=achieved_gap,
    )
