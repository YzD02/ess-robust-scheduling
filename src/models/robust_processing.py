"""Utilities for robust processing-time construction.

This module is intentionally small and explicit because the robust processing time
is the key bridge between the physical system and the planning model.

Modeling idea
-------------
For each job j, the planning model does not use a deterministic nominal time only.
Instead, it inflates the nominal time with a safety buffer:

    p'_j = (mu_A_j + mu_B_j) + k * sqrt(sigma_A_j^2 + sigma_B_j^2)

where:
- mu_A_j = mean processing time at Station A
- mu_B_j = mean processing time at Station B
- sigma_A_j = machine-side uncertainty at Station A
- sigma_B_j = human-side uncertainty at Station B
- k = robustness factor

The purpose of this transformation is to convert a stochastic job into a
single robust item that can be packed into a daily capacity bin.
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
    """Compute nominal and robust processing times for all jobs.

    Parameters
    ----------
    mu_A, mu_B:
        Mean processing times for Stations A and B.
    sigma_A, sigma_B:
        Standard deviation terms representing uncertainty at Stations A and B.
    k:
        Robustness tuning factor.

    Returns
    -------
    p_nominal:
        Nominal total job duration mu_A + mu_B.
    p_robust:
        Robust job duration after adding the uncertainty buffer.
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
