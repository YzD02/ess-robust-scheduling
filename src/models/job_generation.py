from __future__ import annotations

"""Job parameter generation utilities.

What this module does
---------------------
Randomly generates processing-time parameters for each job (work order) on
the ESS assembly line.  In real production, no two jobs take exactly the same
time to complete:

  - Station A (automated busbar assembly):  variability comes mainly from the
    machine cycle and micro-stops.
  - Station B (manual harness alignment / riveting):  variability comes from
    worker speed and skill differences.

This module represents that variability statistically using two values per job:

  mu    (mean)   – the expected processing time under normal conditions
  sigma (std dev) – how much the actual time is likely to deviate from the mean

Where the output is used
------------------------
  robust_processing.py  – uses mu and sigma to add a safety buffer to the
                          planned processing time before handing it to Gurobi
  simulation_engine.py  – uses mu and sigma to draw random actual processing
                          times during each simulation replication

Key parameters
--------------
  n_jobs      : how many jobs to generate
  mu_scale    : scales all mean times uniformly (1.0 = use default baseline values)
  sigma_scale : scales all standard deviations uniformly (1.0 = use default baseline)
  seed        : random seed — fix this to get the same job set every run,
                which is essential for reproducible experiments
"""

from dataclasses import dataclass
import math
import random
from typing import Dict


NumericDict = Dict[int, float]


@dataclass(frozen=True)
class JobGenerationConfig:
    """System-level parameters that control how job times are sampled.

    All times are in minutes.  The defaults reflect the current assumption-based
    baseline for the ESS line; replace them with calibrated values once real
    production data becomes available.

    Station A parameters
    --------------------
    mu_A_mean, mu_A_std   : mean and spread of Station A processing times
    sigma_A_mean/std      : mean and spread of per-job uncertainty at Station A
                            (used in the planning layer only; Station A variability
                            in simulation is modelled via machine-stop events)

    Station B parameters  (manual operation — triangular distribution)
    --------------------
    mu_B_mean, mu_B_std        : mean and spread of the per-job mode (most likely time)
    low_B_fraction             : low  = mode × low_B_fraction  (best-case multiplier, < 1.0)
    high_B_fraction            : high = mode × high_B_fraction (worst-case multiplier, > 1.0)

    The triangular distribution is parameterised as (low, high, mode) where:
        low  = shortest realistic processing time for a job
        high = longest  realistic processing time for a job
        mode = most frequently observed time (the peak of the triangle)

    Example with defaults:
        mode ≈ 60 min  →  low ≈ 60 × 0.70 = 42 min,  high ≈ 60 × 1.50 = 90 min

    Once real factory data is available, estimate these from observations:
        low  ← fastest job time recorded
        high ← slowest job time recorded
        mode ← most common / median job time
    """
    # Station A
    mu_A_mean: float = 90.0
    mu_A_std: float = 6.0
    sigma_A_mean: float = 14.0
    sigma_A_std: float = 2.0

    # Station B — triangular distribution
    mu_B_mean: float = 60.0       # mean of the per-job mode
    mu_B_std: float = 5.0         # job-to-job spread in the mode
    low_B_fraction: float = 0.70  # low  = mode × 0.70  (30% below mode)
    high_B_fraction: float = 1.50 # high = mode × 1.50  (50% above mode)

    # Shared
    min_processing: float = 1.0
    min_sigma: float = 1.0


def _positive_normal(rng: random.Random, mu: float, sigma: float, lower: float) -> float:
    """Sample from a normal distribution, rejecting values below `lower`.

    Processing times and standard deviations must be strictly positive, so we
    resample until a valid value is drawn.  After 1000 failed attempts (extremely
    unlikely in practice) we fall back to the mean to avoid an infinite loop.
    """
    for _ in range(1000):
        x = rng.gauss(mu, sigma)
        if x >= lower:
            return x
    return max(lower, mu)


def generate_job_parameters(
    *,
    n_jobs: int,
    mu_scale: float = 1.0,
    sigma_scale: float = 1.0,
    seed: int = 42,
    config: JobGenerationConfig | None = None,
) -> dict[str, NumericDict]:
    """Generate per-job planning parameters for one experiment instance.

    Each job receives its own mu and sigma values, drawn independently from
    normal distributions whose means and spreads are set in ``config``.
    The ``mu_scale`` and ``sigma_scale`` arguments let experiments uniformly
    inflate or deflate the entire job set without changing the config object —
    useful for stress-testing the system at different load levels.

    Parameters
    ----------
    n_jobs :
        Number of jobs to create (e.g. 20, 40, 60 …).
    mu_scale :
        Multiplier applied to all mean processing times.
        1.0 = baseline; >1.0 = heavier workload per job.
    sigma_scale :
        Multiplier applied to all standard deviations.
        1.0 = baseline; >1.0 = more variability per job.
    seed :
        Random seed.  Use the same seed across experiments to ensure that
        differences in results come from the parameters, not from luck.
    config :
        Generation config.  Defaults to ``JobGenerationConfig()`` if omitted.

    Returns
    -------
    dict with keys ``jobs``, ``mu_A``, ``mu_B``, ``low_B``, ``high_B``, ``sigma_A``.
    Each value is a dict mapping job id (int) → time in minutes (float).

    Station B keys:
        mu_B   — mode (most likely time) of the triangular distribution
        low_B  — lower bound (best-case time) = mode × low_B_fraction
        high_B — upper bound (worst-case time) = mode × high_B_fraction
    """
    cfg = config or JobGenerationConfig()
    rng = random.Random(seed)

    jobs = list(range(1, n_jobs + 1))
    mu_A: NumericDict = {}
    mu_B: NumericDict = {}       # mode of the triangular distribution for Station B
    low_B: NumericDict = {}      # lower bound (best-case time)
    high_B: NumericDict = {}     # upper bound (worst-case time)
    sigma_A: NumericDict = {}

    for j in jobs:
        mu_A[j] = _positive_normal(
            rng,
            cfg.mu_A_mean * mu_scale,
            cfg.mu_A_std,
            cfg.min_processing,
        )
        mode_B = _positive_normal(
            rng,
            cfg.mu_B_mean * mu_scale,
            cfg.mu_B_std,
            cfg.min_processing,
        )
        mu_B[j] = mode_B
        low_B[j]  = max(cfg.min_processing, mode_B * cfg.low_B_fraction)
        high_B[j] = mode_B * cfg.high_B_fraction
        sigma_A[j] = _positive_normal(
            rng,
            cfg.sigma_A_mean * sigma_scale,
            cfg.sigma_A_std,
            cfg.min_sigma,
        )

    # Derive sigma_B analytically from the triangular distribution parameters.
    # For a triangular(low, high, mode), the standard deviation is:
    #   std = sqrt((low² + high² + mode² - low×high - low×mode - high×mode) / 18)
    # This is used in the planning layer (robust_processing.py) to size the
    # safety buffer.  It is NOT used in simulation — actual times are drawn
    # directly from the triangular distribution there.
    sigma_B: NumericDict = {}
    for j in jobs:
        a, b, c = low_B[j], high_B[j], mu_B[j]
        sigma_B[j] = math.sqrt((a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18.0)

    return {
        "jobs":    jobs,
        "mu_A":    mu_A,
        "mu_B":    mu_B,    # mode of triangular distribution
        "low_B":   low_B,   # lower bound of triangular distribution
        "high_B":  high_B,  # upper bound of triangular distribution
        "sigma_A": sigma_A,
        "sigma_B": sigma_B, # derived from triangular params; used in planning layer only
    }
