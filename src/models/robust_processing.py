"""Utilities for robust processing-time construction.

What this module does
---------------------
Converts the raw statistical job parameters (mu, sigma) into a single
*robust processing time* that the Gurobi planning model can use.

The core formula
----------------
For each job j:

    p_robust[j] = (mu_A[j] + mu_B[j])  +  k * sqrt(sigma_A[j]^2 + sigma_B[j]^2)

In plain language:

    planned time = average total time  +  (k × combined uncertainty margin)

Why add a buffer?
-----------------
If we schedule based on average times alone, roughly half of all days will
overrun — because on bad days the actual times exceed the average.  Those
overruns accumulate as backlog and spill into the following days.

By inflating each job's planned time by k standard deviations, we give the
schedule a built-in cushion.  The larger k is, the safer the plan, but also
the more conservative (fewer jobs fit in one day).

What k controls
---------------
  k = 0.0  →  no buffer; plan uses average times only (highest risk)
  k = 0.5  →  moderate buffer; good starting point for low-variability lines
  k = 1.0  →  one standard deviation of margin (commonly used default)
  k = 2.0  →  very conservative; appropriate for high-variability operations

This module returns both the nominal time (no buffer) and the robust time
(with buffer) so that downstream analysis can compare them side by side.
"""

from __future__ import annotations

import math
from typing import Dict


NumericDict = Dict[int, float]


def compute_robust_processing_times(
    mu_A: NumericDict,
    mu_B: NumericDict,
    sigma_A: NumericDict,
    sigma_B: NumericDict,
    k: float,
) -> tuple[NumericDict, NumericDict]:
    """Compute nominal and robust processing times for every job.

    Parameters
    ----------
    mu_A, mu_B :
        Mean processing times (minutes) at Stations A and B respectively.
        Keyed by job id.
    sigma_A, sigma_B :
        Standard deviations of processing time at each station.
        sigma_A captures machine-side uncertainty; sigma_B captures
        human-side uncertainty.
    k :
        Robustness factor — see module docstring for guidance on choosing k.

    Returns
    -------
    p_nominal :
        Simple sum mu_A + mu_B.  Used for reporting and comparison only;
        it is NOT passed to Gurobi.
    p_robust :
        Buffered time = p_nominal + k * sqrt(sigma_A^2 + sigma_B^2).
        This is what the Gurobi model uses to pack jobs into daily bins.
    """
    jobs = sorted(mu_A.keys())
    p_nominal: NumericDict = {}
    p_robust: NumericDict = {}

    for j in jobs:
        nominal = float(mu_A[j]) + float(mu_B[j])
        combined_sigma = math.sqrt(float(sigma_A[j]) ** 2 + float(sigma_B[j]) ** 2)
        robust = nominal + float(k) * combined_sigma

        p_nominal[j] = nominal
        p_robust[j] = robust

    return p_nominal, p_robust
