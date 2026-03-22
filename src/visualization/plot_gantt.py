from __future__ import annotations

"""
plot_gantt.py
=============

Plot Gantt charts from the simulation event-log CSV.

Supported modes
---------------
1. detailed
   - Separate lane for each day-station pair
   - Example lanes:
       Day 1 - Station A
       Day 1 - Station B
       Day 2 - Station A
       Day 2 - Station B
   - Best for inspecting station-level timing

2. merged
   - One lane per day
   - Station A and B are merged into one job block
   - Best for showing actual within-day execution in a simpler way
"""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def load_event_log(csv_path: str) -> pd.DataFrame:
    """Load and sort event log CSV."""
    df = pd.read_csv(csv_path)

    required_cols = {
        "executed_day",
        "station",
        "job_id",
        "start_time_day",
        "end_time_day",
        "start_time_global",
        "end_time_global",
        "duration",
        "from_backlog",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in event log CSV: {sorted(missing)}")

    df = df.sort_values(
        ["executed_day", "station", "start_time_day", "job_id"]
    ).reset_index(drop=True)
    return df


# =====================================================
# MODE 1: DETAILED GANTT
# =====================================================

def lane_name_detailed(executed_day: int, station: str) -> str:
    return f"Day {executed_day} - Station {station}"


def build_lane_order_detailed(df: pd.DataFrame) -> list[str]:
    lanes = []
    for d in sorted(df["executed_day"].unique()):
        lanes.append(f"Day {int(d)} - Station A")
        lanes.append(f"Day {int(d)} - Station B")
    return lanes


def plot_detailed_gantt(df: pd.DataFrame, out_path: str | None = None):
    """
    Plot station-level Gantt chart.

    - Y-axis: Day + station
    - X-axis: global time
    """
    df = df.copy()
    df["lane"] = df.apply(
        lambda r: lane_name_detailed(int(r["executed_day"]), str(r["station"])),
        axis=1
    )

    lanes = build_lane_order_detailed(df)
    lane_to_y = {lane: i for i, lane in enumerate(lanes)}

    unique_jobs = sorted(df["job_id"].unique())
    cmap = cm.get_cmap("tab20", max(20, len(unique_jobs)))
    job_to_color = {job: cmap(i % cmap.N) for i, job in enumerate(unique_jobs)}

    fig_height = max(6, len(lanes) * 0.55)
    fig, ax = plt.subplots(figsize=(40, fig_height))

    bar_height = 0.35

    for _, row in df.iterrows():
        y = lane_to_y[row["lane"]]
        start = row["start_time_global"]
        duration = row["duration"]
        job_id = int(row["job_id"])

        hatch = "///" if bool(row["from_backlog"]) else None

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=bar_height,
            color=job_to_color[job_id],
            edgecolor="black",
            hatch=hatch,
            alpha=0.9,
        )

        ax.text(
            start + duration / 2,
            y,
            f"J{job_id}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_yticks(list(lane_to_y.values()))
    ax.set_yticklabels(list(lane_to_y.keys()))
    ax.set_xlabel("Global Time")
    ax.set_ylabel("Execution Lane")
    ax.set_title("Detailed Realized Gantt Chart (Station-Level)", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Executed in current-day queue"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Executed from backlog"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Legend",
    )

    plt.tight_layout()

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved detailed Gantt chart to: {out}")

    plt.show()


# =====================================================
# MODE 2: MERGED DAILY GANTT
# =====================================================

def build_merged_job_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Station A and Station B into one job block per day.

    For each (executed_day, job_id), compute:
    - start_time_day   = min of A/B starts
    - end_time_day     = max of A/B ends
    - duration         = merged duration
    - from_backlog     = any(from_backlog)
    - was_delayed      = any(was_delayed)
    """
    agg_df = (
        df.groupby(["executed_day", "job_id", "planned_day"], as_index=False)
        .agg(
            start_time_day=("start_time_day", "min"),
            end_time_day=("end_time_day", "max"),
            start_time_global=("start_time_global", "min"),
            end_time_global=("end_time_global", "max"),
            from_backlog=("from_backlog", "max"),
            was_delayed=("was_delayed", "max"),
            day_delay=("day_delay", "max"),
        )
    )

    agg_df["duration"] = agg_df["end_time_day"] - agg_df["start_time_day"]
    return agg_df.sort_values(["executed_day", "start_time_day", "job_id"]).reset_index(drop=True)


def plot_merged_daily_gantt(df: pd.DataFrame, out_path: str | None = None):
    """
    Plot a simplified day-level Gantt chart.

    - One lane per day
    - Station A and B merged into one continuous job block
    - X-axis uses within-day time for clarity
    """
    merged = build_merged_job_blocks(df)

    days = sorted(merged["executed_day"].unique())
    lane_to_y = {f"Day {int(d)}": i for i, d in enumerate(days)}

    unique_jobs = sorted(merged["job_id"].unique())
    cmap = cm.get_cmap("tab20", max(20, len(unique_jobs)))
    job_to_color = {job: cmap(i % cmap.N) for i, job in enumerate(unique_jobs)}

    fig_height = max(5, len(days) * 0.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    bar_height = 0.45

    for _, row in merged.iterrows():
        lane = f"Day {int(row['executed_day'])}"
        y = lane_to_y[lane]

        start = row["start_time_day"]
        duration = row["duration"]
        job_id = int(row["job_id"])

        hatch = "///" if bool(row["from_backlog"]) else None

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=bar_height,
            color=job_to_color[job_id],
            edgecolor="black",
            hatch=hatch,
            alpha=0.95,
        )

        ax.text(
            start + duration / 2,
            y,
            f"J{job_id}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_yticks(list(lane_to_y.values()))
    ax.set_yticklabels(list(lane_to_y.keys()))
    ax.set_xlabel("Within-Day Time")
    ax.set_ylabel("Day")
    ax.set_title("Merged Daily Gantt Chart (A+B Combined)", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # highlight regular shift and overtime area
    ax.axvline(480, color="red", linestyle="--", linewidth=1.5, label="Regular shift limit (480 min)")
    ax.axvline(600, color="gray", linestyle=":", linewidth=1.2, label="Max daily time incl. OT (600 min)")

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Executed in current-day queue"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Executed from backlog"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Execution Pattern",
    )

    info_text = (
        "Notes:\n"
        "- Each bar represents one full job.\n"
        "- Station A and B are merged into one block.\n"
        "- Red dashed line = regular shift limit.\n"
        "- Gray dotted line = shift + overtime cap.\n"
    )
    ax.text(
        1.02, 0.70,
        info_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    plt.tight_layout()

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved merged daily Gantt chart to: {out}")

    plt.show()


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Gantt charts from simulation event-log CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/simulation_outputs/gantt_events_single_run.csv",
        help="Path to the Gantt event-log CSV.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detailed", "merged", "both"],
        default="both",
        help="Gantt chart mode.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Directory to save output figures.",
    )
    args = parser.parse_args()

    df = load_event_log(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("detailed", "both"):
        plot_detailed_gantt(
            df,
            out_path=str(out_dir / "gantt_detailed_single_run.png"),
        )

    if args.mode in ("merged", "both"):
        plot_merged_daily_gantt(
            df,
            out_path=str(out_dir / "gantt_merged_single_run.png"),
        )


if __name__ == "__main__":
    main()