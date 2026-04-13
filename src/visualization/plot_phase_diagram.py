from __future__ import annotations

"""
plot_phase_diagram.py
=====================

Phase diagram for the grid search results.

What this script does
---------------------
Reads the grid search CSV and produces a colour-coded grid where each cell
represents one (n_jobs, k) combination and its colour shows which operational
zone that combination falls into:

  Green  — feasible within weekdays
           The schedule clears with ≥ 95 % probability using regular shifts only.

  Orange — feasible only with weekend extension
           Weekdays alone are not enough, but the extra weekend recovery days
           bring the success rate back up to ≥ 95 %.

  Red-orange — infeasible even after extension
               Even with full weekend support the schedule fails more than
               5 % of the time.  The cell shows the actual success probability.

  Red    — computationally infeasible
           Gurobi could not produce a valid schedule within the time limit.

  Grey   — missing / not yet simulated

The cell label shows the most useful diagnostic value for each zone:
  green       → "WD OK"
  orange      → average weekend days used
  red-orange  → success probability (e.g. "72%")
  red         → Gurobi solve time

How to run
----------
    python -m src.visualization.plot_phase_diagram

To point at a different CSV:

    python -m src.visualization.plot_phase_diagram --csv path/to/results.csv

Output
------
PNG figures saved to ``results/figures/``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

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
# 2. PHASE CLASSIFICATION
# =====================================================

def classify_case(row: pd.Series) -> int:
    """
    New weekend-extension phase logic.

    Phase meanings
    --------------
    0 = computationally infeasible
        Gurobi did not produce an accepted schedule in time

    1 = infeasible even after weekend extension
        accepted_for_simulation = True
        prob_cleared_within_extended_horizon < 0.95

    2 = feasible only with weekend extension
        extended horizon feasible, but weekday horizon not feasible

    3 = feasible within weekdays
        weekday horizon itself already feasible

    4 = missing / unknown
    """
    accepted = row.get("accepted_for_simulation", None)
    p_weekday = row.get("prob_cleared_within_weekdays", None)
    p_extended = row.get("prob_cleared_within_extended_horizon", None)

    if pd.isna(accepted):
        return 4

    if accepted is False:
        return 0

    if accepted is True:
        if pd.isna(p_extended):
            return 4

        if p_extended < 0.95:
            return 1

        # extended feasible
        if pd.notna(p_weekday) and p_weekday >= 0.95:
            return 3
        else:
            return 2

    return 4


def make_label(row: pd.Series) -> str:
    """
    Cell label logic:
    - computational infeasible -> solve time
    - infeasible after extension -> extended clear probability
    - feasible only via weekend -> weekend days used
    - feasible within weekdays -> 'WD OK'
    """
    accepted = row.get("accepted_for_simulation", None)
    p_weekday = row.get("prob_cleared_within_weekdays", None)
    p_extended = row.get("prob_cleared_within_extended_horizon", None)
    solve_time = row.get("solve_time_sec", None)
    avg_weekend_days = row.get("avg_n_weekend_days_used", None)

    if pd.isna(accepted):
        return "NA"

    if accepted is False:
        return f"{solve_time:.0f}s" if pd.notna(solve_time) else "Comp"

    if accepted is True:
        if pd.isna(p_extended):
            return "NA"

        if p_extended < 0.95:
            return f"{100 * p_extended:.0f}%"

        if pd.notna(p_weekday) and p_weekday >= 0.95:
            return "WD OK"

        return f"Wknd {avg_weekend_days:.1f}" if pd.notna(avg_weekend_days) else "Ext OK"

    return "NA"


def build_phase_tables(df: pd.DataFrame, x_col: str, y_col: str):
    tmp = df.copy()
    tmp["phase_class"] = tmp.apply(classify_case, axis=1)
    tmp["phase_label"] = tmp.apply(make_label, axis=1)

    phase_df = tmp.pivot_table(
        index=y_col,
        columns=x_col,
        values="phase_class",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)

    label_df = tmp.pivot_table(
        index=y_col,
        columns=x_col,
        values="phase_label",
        aggfunc="first",
    ).sort_index().sort_index(axis=1)

    return phase_df, label_df


# =====================================================
# 3. SUMMARY PANEL
# =====================================================

def build_summary_text(df_slice: pd.DataFrame) -> str:
    lines = []

    def add_line(label, col, fmt="{:.1f}"):
        if col in df_slice.columns:
            vals = df_slice[col].dropna()
            if len(vals) > 0:
                lines.append(f"{label}: {fmt.format(vals.mean())}")

    add_line("avg nominal job time", "mu_total_mean")
    add_line("avg combined sigma", "sigma_total_mean")
    add_line("avg k-buffer / job", "k_buffer_mean")
    add_line("avg robust load / day", "avg_robust_load_per_day")
    add_line("avg weekday clear prob", "prob_cleared_within_weekdays", "{:.2f}")
    add_line("avg extended clear prob", "prob_cleared_within_extended_horizon", "{:.2f}")
    add_line("avg weekend days used", "avg_n_weekend_days_used", "{:.2f}")
    add_line("avg weekend cost", "avg_total_weekend_cost", "{:.1f}")

    return "\n".join(lines)


# =====================================================
# 4. PLOTTING
# =====================================================

def plot_phase_for_slice(
    df_slice: pd.DataFrame,
    x_col: str,
    y_col: str,
    slice_name: str,
    csv_stem: str,
    out_dir: Path,
):
    phase_df, label_df = build_phase_tables(df_slice, x_col, y_col)
    if phase_df.empty:
        return

    values = phase_df.values

    # 0 computational infeasible -> red
    # 1 infeasible even after extension -> dark orange
    # 2 feasible only with weekend extension -> light orange
    # 3 feasible within weekdays -> green
    # 4 missing -> gray
    cmap = ListedColormap([
        "#d73027",  # red
        "#f46d43",  # orange-red
        "#fdae61",  # orange
        "#1a9850",  # green
        "#d9d9d9",  # gray
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.7, 1.4], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(phase_df.columns)))
    ax.set_yticks(np.arange(len(phase_df.index)))
    ax.set_xticklabels(phase_df.columns)
    ax.set_yticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in phase_df.index])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Weekend-Extension Phase Diagram | slice: {slice_name}", fontsize=13, pad=10)

    ax.set_xticks(np.arange(-0.5, len(phase_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(phase_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = label_df.iloc[i, j]
            if pd.notna(txt):
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    legend_handles = [
        mpatches.Patch(color="#1a9850", label="Feasible within weekdays"),
        mpatches.Patch(color="#fdae61", label="Feasible only via weekend extension"),
        mpatches.Patch(color="#f46d43", label="Infeasible even after extension"),
        mpatches.Patch(color="#d73027", label="Computationally infeasible"),
        mpatches.Patch(color="#d9d9d9", label="Missing / not simulated"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Phase",
    )

    summary_text = build_summary_text(df_slice)
    ax_info.text(
        -0.35, 0.6,
        f"CSV: {csv_stem}\n\nSlice: {slice_name}\n\n{summary_text}",
        va="top",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("Weekend-Extension System Phase Diagram", fontsize=15, y=1.02)
    plt.tight_layout()

    out_path = out_dir / f"phase_diagram_{sanitize_name(csv_stem)}__{sanitize_name(slice_name)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# =====================================================
# 5. MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate phase diagrams for weekend-extension grid search results."
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
        help="Directory to save output phase diagrams.",
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
            "Not enough varying hyperparameters to make a 2D phase diagram. "
            "At least two hyperparameters must vary."
        )

    facet_cols = [c for c in varying if c not in [x_col, y_col]]

    print("Detected varying hyperparameters:", varying)
    print(f"Using x = {x_col}, y = {y_col}, facets = {facet_cols}")

    csv_stem = csv_path.stem

    for slice_name, df_slice in iter_facet_slices(df, facet_cols):
        if df_slice.empty:
            continue
        plot_phase_for_slice(
            df_slice=df_slice,
            x_col=x_col,
            y_col=y_col,
            slice_name=slice_name,
            csv_stem=csv_stem,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()