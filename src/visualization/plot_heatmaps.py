from __future__ import annotations

"""
plot_heatmaps.py
================

Heatmap dashboard for the grid search results.

What this script does
---------------------
Reads the grid search CSV produced by ``run_grid_search.py`` and generates
a multi-panel heatmap dashboard showing how key performance metrics change
across the (n_jobs × k) parameter space.

Each panel in the dashboard shows one metric — for example, the probability
of clearing all jobs within the extended horizon, or the average number of
weekend days consumed.  Cells are colour-coded from low (dark) to high
(bright), and the numeric value is printed inside each cell so the chart
is self-contained.

How to run
----------
    python -m src.visualization.plot_heatmaps

To point at a different CSV file:

    python -m src.visualization.plot_heatmaps --csv path/to/results.csv

Output
------
PNG figures saved to ``results/figures/``.

Typical usage
-------------
    python -m src.visualization.plot_heatmaps --csv results/grid_search/grid_search_results_weekend_extension.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot_utils import (
    load_results,
    get_varying_hyperparams,
    choose_axes,
    iter_facet_slices,
    sanitize_name,
    unique_sorted,
)


HYPERPARAM_COLS = ["n_jobs", "mu_scale", "sigma_scale", "k"]


# =====================================================
# 1. HEATMAP-SPECIFIC HELPERS
# =====================================================

def make_pivot(df: pd.DataFrame, value_col: str, x_col: str, y_col: str) -> pd.DataFrame:
    """Pivot a results DataFrame into a 2-D matrix for heatmap plotting."""
    pivot = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc="first",
    )
    return pivot.sort_index().sort_index(axis=1)


def format_cell(val, mode: str = "float", decimals: int = 1) -> str:
    """Format a single heatmap cell value as a display string."""
    if pd.isna(val):
        return "NA"
    if mode == "percent":
        return f"{100 * val:.0f}%"
    if mode == "int":
        return f"{int(round(val))}"
    return f"{val:.{decimals}f}"


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