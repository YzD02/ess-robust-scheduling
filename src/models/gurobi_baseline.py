"""Baseline robust bin-packing model solved with Gurobi.

What this module does
---------------------
Uses the Gurobi mathematical optimisation solver to assign all jobs to working
days, minimising overtime cost.

Think of it as an intelligent bin-packing problem:
  - Each job has a "planned processing time" (already inflated by the safety buffer)
  - Each working day is a "bin" with 480 minutes of regular capacity
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
  1) Full assignment:  every job appears in exactly one day
  2) Robust daily capacity:  sum of robust job times on day t <= C_std + OT[t]
  3) Overtime activation:  OT[t] <= M * u[t]
     (overtime can only be nonzero if the activation variable is switched on)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
    model_name: str = "ESS_Robust_BinPacking"


def solve_gurobi_baseline(
    *,
    jobs: List[int],
    days: List[int],
    p_robust: Dict[int, float],
    C_std: float,
    Cost_OT: float,
    Cost_fix: float,
    M: float,
    time_limit_sec: float | None = None,
    output_flag: int = 1,
) -> GurobiBaselineResult:
    """Solve the baseline day-assignment model.

    Notes
    -----
    The returned `sorted_schedule` uses a shortest-robust-processing-time-first
    post-processing rule inside each day. This does not change the optimization
    objective. It simply provides a feasible execution order for the simulation.
    """
    model = Model("ESS_Robust_BinPacking")
    model.setParam("OutputFlag", output_flag)
    if time_limit_sec is not None:
        model.setParam("TimeLimit", float(time_limit_sec))

    # -------------------------
    # Decision variables
    # -------------------------
    z = model.addVars(jobs, days, vtype=GRB.BINARY, name="z")
    OT = model.addVars(days, vtype=GRB.CONTINUOUS, lb=0.0, name="OT")
    u = model.addVars(days, vtype=GRB.BINARY, name="u")

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
    # The robust job load assigned to one day cannot exceed regular capacity
    # plus overtime.
    # -------------------------
    for t in days:
        model.addConstr(
            quicksum(p_robust[j] * z[j, t] for j in jobs) <= C_std + OT[t],
            name=f"capacity_{t}",
        )

    # -------------------------
    # Constraint 3: Overtime activation logic
    # Overtime can only be positive if the activation variable is turned on.
    # -------------------------
    for t in days:
        model.addConstr(
            OT[t] <= M * u[t],
            name=f"ot_activation_{t}",
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

    if model.SolCount > 0:
        objective_value = float(model.objVal)
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
    )
