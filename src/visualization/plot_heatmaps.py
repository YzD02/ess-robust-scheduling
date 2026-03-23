from __future__ import annotations

"""
plot_heatmaps.py
================

Heatmap dashboard for the weekend-extension grid search results.

This script is tailored for the CSV produced by:

    run_grid_search_weekend_extension.py

Main goals
----------
1. Show the operational performance of the new execution logic:
   - 20 weekday bins
   - 8 weekend extension bins
   - no same-day overtime completion
   - backlog-first execution

2. Make the result interpretable by plotting both:
   - outcome metrics
   - parameter-scale metrics

Typical usage
-------------
python -m src.visualization.plot_heatmaps --csv results/grid_search/grid_search_results_weekend_extension.csv
"""

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HYPERPARAM_COLS = ["n_jobs", "mu_scale", "sigma_scale", "k"]


# =====================================================
# 1. HELPERS
# =====================================================

def load_results(csv_path: str) -> pd.DataFrame:
    """Load CSV and normalize boolean-like columns."""
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


def get_varying_hyperparams(df: pd.DataFrame) -> list[str]:
    """Return hyperparameter columns that actually vary in the CSV."""
    varying = []
    for col in HYPERPARAM_COLS:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            varying.append(col)
    return varying


def choose_axes(varying_params: list[str]) -> tuple[str, str | None, list[str]]:
    """
    Choose x-axis, y-axis, and facet columns automatically.

    Priority
    --------
    - x-axis: n_jobs if available
    - y-axis: k if available among remaining
    - otherwise use the first remaining varying parameter
    - the rest become facet/slice conditions
    """
    if not varying_params:
        return "n_jobs", None, []

    x_col = "n_jobs" if "n_jobs" in varying_params else varying_params[0]
    remaining = [c for c in varying_params if c != x_col]

    if not remaining:
        return x_col, None, []

    if "k" in remaining:
        y_col = "k"
    else:
        y_col = remaining[0]

    facet_cols = [c for c in remaining if c != y_col]
    return x_col, y_col, facet_cols


def unique_sorted(series: pd.Series):
    return sorted(series.dropna().unique().tolist())


def iter_facet_slices(df: pd.DataFrame, facet_cols: list[str]):
    """
    Yield all facet slices.

    Example:
    if facet_cols = [mu_scale, sigma_scale], then one slice is
    mu_scale=1.0, sigma_scale=1.2
    """
    if not facet_cols:
        yield "all", df.copy()
        return

    value_lists = [unique_sorted(df[c]) for c in facet_cols]
    for combo in product(*value_lists):
        sub = df.copy()
        parts = []
        for c, v in zip(facet_cols, combo):
            sub = sub[np.isclose(sub[c], v)]
            parts.append(f"{c}-{v}")
        yield "__".join(parts), sub


def make_pivot(df: pd.DataFrame, value_col: str, x_col: str, y_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc="first",
    )
    return pivot.sort_index().sort_index(axis=1)


def format_cell(val, mode="float", decimals=1) -> str:
    if pd.isna(val):
        return "NA"
    if mode == "percent":
        return f"{100 * val:.0f}%"
    if mode == "int":
        return f"{int(round(val))}"
    return f"{val:.{decimals}f}"


def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("=", "-")
        .replace(",", "_")
        .replace(".", "p")
    )


# =====================================================
# 2. PLOTTING
# =====================================================

def plot_one_heatmap(ax, pivot_df, title, mode="float", decimals=1, vmin=None, vmax=None):
    values = pivot_df.values.astype(float)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e6e6e6")
    masked = np.ma.masked_invalid(values)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels([f"{y:.2f}" if isinstance(y, float) else str(y) for y in pivot_df.index])

    ax.set_xlabel(pivot_df.columns.name)
    ax.set_ylabel(pivot_df.index.name)
    ax.set_title(title, fontsize=11, pad=8)

    ax.set_xticks(np.arange(-0.5, len(pivot_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = format_cell(values[i, j], mode=mode, decimals=decimals)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    return im


def build_dashboard_for_slice(
    df_slice: pd.DataFrame,
    x_col: str,
    y_col: str,
    slice_name: str,
    out_dir: Path,
    csv_stem: str,
):
    """
    Build one multi-panel dashboard for one hyperparameter slice.
    """
    metric_specs = [
        # Outcome layer
        ("prob_cleared_within_extended_horizon", "Prob. Cleared in Extended Horizon", "percent", 0, 0.0, 1.0),
        ("avg_total_weekend_cost", "Avg. Weekend Extension Cost", "float", 1, None, None),
        ("avg_n_weekend_days_used", "Avg. Weekend Days Used", "float", 1, None, None),
        ("avg_final_completion_day", "Avg. Final Completion Day", "float", 1, None, None),

        # Diagnostics / planning layer
        ("solve_time_sec", "Gurobi Solve Time (sec)", "float", 0, None, None),
        ("avg_robust_load_per_day", "Avg. Robust Load / Weekday", "float", 1, None, None),
        ("k_buffer_mean", "Avg. Robustness Buffer / Job", "float", 1, None, None),
        ("prob_any_weekend_use", "Prob. Any Weekend Use", "percent", 0, 0.0, 1.0),
    ]

    existing_specs = [m for m in metric_specs if m[0] in df_slice.columns]
    if not existing_specs:
        return

    n_panels = len(existing_specs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (col, title, mode, decimals, vmin, vmax) in enumerate(existing_specs):
        pivot = make_pivot(df_slice, col, x_col, y_col)
        if pivot.empty:
            axes[idx].axis("off")
            continue

        plot_one_heatmap(
            axes[idx],
            pivot_df=pivot,
            title=title,
            mode=mode,
            decimals=decimals,
            vmin=vmin,
            vmax=vmax,
        )

    for idx in range(len(existing_specs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Weekend-Extension Heatmap Dashboard | {csv_stem} | slice: {slice_name}",
        fontsize=15,
        y=1.02,
    )

    plt.tight_layout()

    out_path = out_dir / f"heatmap_{sanitize_name(csv_stem)}__{sanitize_name(slice_name)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# =====================================================
# 3. MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmap dashboards for weekend-extension grid search results."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/grid_search/grid_search_results_weekend_extension.csv",
        help="Path to weekend-extension grid search CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Directory to save output heatmaps.",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="Optional manual x-axis hyperparameter.",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Optional manual y-axis hyperparameter.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(str(csv_path))
    varying = get_varying_hyperparams(df)

    auto_x, auto_y, _ = choose_axes(varying)
    x_col = args.x_col if args.x_col else auto_x
    y_col = args.y_col if args.y_col else auto_y

    if y_col is None:
        raise ValueError(
            "Not enough varying hyperparameters to make a 2D heatmap. "
            "At least two hyperparameters must vary."
        )

    facet_cols = [c for c in varying if c not in [x_col, y_col]]

    print("Detected varying hyperparameters:", varying)
    print(f"Using x = {x_col}, y = {y_col}, facets = {facet_cols}")

    csv_stem = csv_path.stem

    for slice_name, df_slice in iter_facet_slices(df, facet_cols):
        if df_slice.empty:
            continue
        build_dashboard_for_slice(
            df_slice=df_slice,
            x_col=x_col,
            y_col=y_col,
            slice_name=slice_name,
            out_dir=out_dir,
            csv_stem=csv_stem,
        )


if __name__ == "__main__":
    main()