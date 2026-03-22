"""
plot_heatmaps.py
================

This script automatically generates heatmap dashboards from the grid search
results of the ESS robust scheduling experiments.

The goal of the visualization is to show how system performance changes across
different hyperparameter combinations, including:

    - n_jobs        : number of jobs scheduled within the planning horizon
    - mu_scale      : scaling factor applied to nominal processing times
    - sigma_scale   : scaling factor applied to processing time variability
    - k             : robustness parameter used in the robust processing time
                      formulation p'_j = μ_j + k σ_j

Key Features
------------
1. Automatic hyperparameter detection
   The script inspects the CSV file and detects which hyperparameters vary.
   This allows the visualization to adapt automatically to different
   experiments without rewriting plotting code.

2. Automatic axis selection
   - x-axis: n_jobs (preferred)
   - y-axis: first remaining varying hyperparameter
   - remaining hyperparameters become slice conditions (facets)

3. Automatic slicing
   If multiple hyperparameters vary, the script generates a separate heatmap
   dashboard for each hyperparameter combination.

4. Multi-metric dashboard
   Each generated figure contains several heatmaps, including:

       • Probability of clearing horizon
       • Gurobi solve time
       • Average overtime
       • Nominal workload
       • Robust workload
       • Robustness buffer size

5. Automatic file naming
   Output figures include hyperparameter values in the filename so that
   multiple experiments can coexist in the same results folder.

Input
-----
CSV produced by run_grid_search.py

Expected columns include:

    n_jobs
    mu_scale
    sigma_scale
    k
    prob_cleared_within_horizon
    solve_time_sec
    avg_total_overtime
    avg_nominal_load_per_day
    avg_robust_load_per_day
    k_buffer_mean

Output
------
PNG heatmap dashboards saved in:

    results/figures/

Example
-------
python -m src.visualization.plot_heatmaps \
    --csv results/grid_search/grid_search_results.csv
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HYPERPARAM_COLS = ["n_jobs", "mu_scale", "sigma_scale", "k"]


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Load the grid search result CSV.

    This function also normalizes boolean columns that may have been saved
    as strings (e.g., "true"/"false") during CSV export.

    Parameters
    ----------
    csv_path : str
        Path to the grid search results CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe ready for visualization.
    """
    df = pd.read_csv(csv_path)

    # normalize booleans if saved as strings
    bool_cols = ["accepted_for_simulation", "meets_95_constraint", "system_ok"]
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
    """
    Detect which hyperparameters vary in the dataset.

    A hyperparameter is considered "varying" if it has more than one
    unique value in the CSV.

    This detection allows the plotting script to automatically adapt
    to different experiment configurations.

    Returns
    -------
    list[str]
        List of varying hyperparameter names.
    """
    varying = []
    for col in HYPERPARAM_COLS:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            varying.append(col)
    return varying


def choose_axes(varying_params: list[str]) -> tuple[str, str | None, list[str]]:
    """
    Determine which hyperparameters should be used as heatmap axes.

    Strategy
    --------
    1. Prefer 'n_jobs' as the x-axis when available
    2. Use the next varying hyperparameter as the y-axis
    3. Remaining hyperparameters become slicing dimensions

    Example
    -------
    If varying parameters are:

        [n_jobs, sigma_scale, k]

    Then:

        x-axis = n_jobs
        y-axis = sigma_scale
        facets = [k]

    Returns
    -------
    x_col : str
    y_col : str
    facet_cols : list[str]
    """
    if not varying_params:
        return "n_jobs", None, []

    x_col = "n_jobs" if "n_jobs" in varying_params else varying_params[0]
    remaining = [c for c in varying_params if c != x_col]

    if not remaining:
        return x_col, None, []

    y_col = remaining[0]
    facet_cols = remaining[1:]

    return x_col, y_col, facet_cols


def unique_sorted(series: pd.Series):
    vals = sorted(series.dropna().unique().tolist())
    return vals


def iter_facet_slices(df: pd.DataFrame, facet_cols: list[str]):
    """
    Iterate over all hyperparameter slice combinations.

    If multiple hyperparameters remain after axis selection, the
    dataset is split into smaller slices. Each slice corresponds
    to one fixed hyperparameter combination.

    Example
    -------
    If facet columns are:

        sigma_scale = [1.0, 1.2]
        k = [1.0, 1.5]

    The function will generate four slices:

        sigma_scale=1.0, k=1.0
        sigma_scale=1.0, k=1.5
        sigma_scale=1.2, k=1.0
        sigma_scale=1.2, k=1.5
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
        slice_name = "__".join(parts)
        yield slice_name, sub


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


def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("=", "-")
        .replace(",", "_")
        .replace(".", "p")
    )


def build_dashboard_for_slice(
    df_slice: pd.DataFrame,
    x_col: str,
    y_col: str,
    slice_name: str,
    out_dir: Path,
    csv_stem: str,
):
    """
    Generate a multi-panel heatmap dashboard for one hyperparameter slice.

    Each dashboard contains several heatmaps summarizing system performance
    metrics across the selected hyperparameter grid.

    Panels typically include:

        • Probability of clearing the horizon
        • Gurobi solve time
        • Average total overtime
        • Average nominal load per day
        • Average robust load per day
        • Average robustness buffer per job

    The figure is automatically saved to the output directory with
    hyperparameter values embedded in the filename.
    """
    metric_specs = [
        ("prob_cleared_within_horizon", "Probability of Clearing Horizon", "percent", 1, 0.0, 1.0),
        ("solve_time_sec", "Gurobi Solve Time (sec)", "float", 0, None, None),
        ("avg_total_overtime", "Average Total Overtime", "float", 1, None, None),
        ("avg_nominal_load_per_day", "Average Nominal Load / Day", "float", 1, None, None),
        ("avg_robust_load_per_day", "Average Robust Load / Day", "float", 1, None, None),
        ("k_buffer_mean", "Average Robustness Buffer / Job", "float", 1, None, None),
    ]

    existing_specs = [m for m in metric_specs if m[0] in df_slice.columns]
    if not existing_specs:
        return

    n_panels = len(existing_specs)
    n_rows = 2
    n_cols = int(np.ceil(n_panels / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    ims = []
    for idx, (col, title, mode, decimals, vmin, vmax) in enumerate(existing_specs):
        pivot = make_pivot(df_slice, col, x_col, y_col)
        if pivot.empty:
            axes[idx].axis("off")
            continue
        im = plot_one_heatmap(
            axes[idx],
            pivot,
            title=title,
            mode=mode,
            decimals=decimals,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append((im, axes[idx], title))

    for idx in range(len(existing_specs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Heatmap Dashboard | {csv_stem} | slice: {slice_name}",
        fontsize=14,
        y=1.01
    )

    plt.tight_layout()

    out_path = out_dir / f"heatmap_{sanitize_name(csv_stem)}__{sanitize_name(slice_name)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate heatmap dashboards for all hyperparameter slices."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/grid_search/grid_search_results.csv",
        help="Path to grid search CSV."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures."
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="Optional manual x-axis hyperparameter."
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Optional manual y-axis hyperparameter."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(str(csv_path))
    varying = get_varying_hyperparams(df)

    auto_x, auto_y, auto_facets = choose_axes(varying)
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