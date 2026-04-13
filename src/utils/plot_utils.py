"""Shared utilities for visualization scripts.

This module centralises helper functions that are used by both
plot_heatmaps.py and plot_phase_diagram.py, eliminating code duplication
and providing a single place to maintain these utilities.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# Columns that are treated as hyperparameters in every experiment grid.
HYPERPARAM_COLS = ["n_jobs", "mu_scale", "sigma_scale", "k"]


# ------------------------------------------------------------------
# CSV loading
# ------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    """Load a grid-search CSV and normalise boolean-like columns.

    Columns such as ``accepted_for_simulation`` and ``system_ok`` are
    written as the strings ``"True"`` / ``"False"`` by pandas when a
    DataFrame is round-tripped through CSV. This function converts them
    back to proper Python booleans so downstream comparisons work
    correctly.
    """
    df = pd.read_csv(csv_path)

    bool_cols = ["accepted_for_simulation", "system_ok"]
    for col in bool_cols:
        if col in df.columns and df[col].dtype != bool:
            mapped = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
            )
            if mapped.notna().any():
                df[col] = mapped

    return df


# ------------------------------------------------------------------
# Hyperparameter introspection
# ------------------------------------------------------------------

def get_varying_hyperparams(df: pd.DataFrame) -> list[str]:
    """Return hyperparameter columns that actually vary in the CSV.

    Only columns listed in ``HYPERPARAM_COLS`` are considered. Columns
    with a single unique value are treated as fixed and excluded, since
    they cannot form a meaningful axis in a 2-D plot.
    """
    return [
        col for col in HYPERPARAM_COLS
        if col in df.columns and df[col].nunique(dropna=True) > 1
    ]


def choose_axes(varying_params: list[str]) -> tuple[str, str | None, list[str]]:
    """Choose x-axis, y-axis, and facet columns automatically.

    Priority
    --------
    - x-axis: ``n_jobs`` if available, otherwise the first varying param
    - y-axis: ``k`` if available among the remainder, otherwise the next one
    - facet columns: everything that is neither x nor y
    """
    if not varying_params:
        return "n_jobs", None, []

    x_col = "n_jobs" if "n_jobs" in varying_params else varying_params[0]
    remaining = [c for c in varying_params if c != x_col]

    if not remaining:
        return x_col, None, []

    y_col = "k" if "k" in remaining else remaining[0]
    facet_cols = [c for c in remaining if c != y_col]

    return x_col, y_col, facet_cols


# ------------------------------------------------------------------
# Facet slicing
# ------------------------------------------------------------------

def unique_sorted(series: pd.Series) -> list:
    """Return sorted unique non-null values from a Series."""
    return sorted(series.dropna().unique().tolist())


def iter_facet_slices(df: pd.DataFrame, facet_cols: list[str]):
    """Yield (slice_name, sub_dataframe) for every facet combination.

    If ``facet_cols`` is empty, a single slice named ``"all"`` containing
    the full dataframe is yielded.

    Example
    -------
    With ``facet_cols = ["mu_scale", "sigma_scale"]`` the function yields
    one sub-dataframe for every combination such as
    ``mu_scale=1.0, sigma_scale=1.0``.
    """
    if not facet_cols:
        yield "all", df.copy()
        return

    value_lists = [unique_sorted(df[c]) for c in facet_cols]
    for combo in product(*value_lists):
        sub = df.copy()
        parts = []
        for col, val in zip(facet_cols, combo):
            sub = sub[np.isclose(sub[col], val)]
            parts.append(f"{col}-{val}")
        yield "__".join(parts), sub


# ------------------------------------------------------------------
# String helpers
# ------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Convert an arbitrary string into a filesystem-safe filename part."""
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("=", "-")
        .replace(",", "_")
        .replace(".", "p")
    )
