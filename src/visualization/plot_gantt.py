from __future__ import annotations

"""
plot_gantt.py
=============

Plot Gantt charts from simulation event-log CSV.

Modes
-----
1. detailed_global
   - Separate lanes by day and station
   - X-axis uses global time
   - Best for inspecting station-level congestion and propagation

2. merged_shift
   - One lane per executed day
   - Station A and B merged into one full-job block
   - X-axis uses within-day time (0-480)
   - Best for reading one-day execution clearly

3. merged_calendar
   - One lane per calendar day in 4-week structure
   - Weekends are interleaved after every 5 weekdays
   - Station A and B merged
   - X-axis uses within-day time (0-480)
   - Best for presentation / PPT readability
"""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def load_event_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "executed_day",
        "day_type",
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

    return df.sort_values(
        ["executed_day", "station", "start_time_day", "job_id"]
    ).reset_index(drop=True)


def build_merged_job_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Station A and Station B into one block per (executed_day, job_id).
    """
    agg_df = (
        df.groupby(["executed_day", "day_type", "job_id", "planned_day"], as_index=False)
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


def get_job_colors(job_ids):
    unique_jobs = sorted(set(job_ids))
    cmap = cm.get_cmap("tab20", max(20, len(unique_jobs)))
    return {job: cmap(i % cmap.N) for i, job in enumerate(unique_jobs)}


# =====================================================
# Mode 1: detailed_global
# =====================================================

def build_lane_order_detailed(df: pd.DataFrame) -> list[str]:
    lanes = []
    for d in sorted(df["executed_day"].unique()):
        lanes.append(f"Day {int(d)} - Station A")
        lanes.append(f"Day {int(d)} - Station B")
    return lanes


def plot_detailed_global(df: pd.DataFrame, out_path: str | None = None):
    df = df.copy()
    df["lane"] = df.apply(
        lambda r: f"Day {int(r['executed_day'])} - Station {r['station']}",
        axis=1
    )

    lanes = build_lane_order_detailed(df)
    lane_to_y = {lane: i for i, lane in enumerate(lanes)}
    job_to_color = get_job_colors(df["job_id"].tolist())

    fig_height = max(6, len(lanes) * 0.55)
    fig, ax = plt.subplots(figsize=(40, fig_height))

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
            height=0.35,
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
    ax.set_title("Detailed Gantt Chart (Global Timeline)", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Current-day queue"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="From backlog"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    plt.tight_layout()
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved detailed global Gantt to: {out}")
    plt.show()


# =====================================================
# Mode 2: merged_shift
# =====================================================

def plot_merged_shift(df: pd.DataFrame, out_path: str | None = None):
    merged = build_merged_job_blocks(df)
    merged["lane"] = merged.apply(
        lambda r: f"Day {int(r['executed_day'])} ({r['day_type']})",
        axis=1
    )

    lanes = merged["lane"].drop_duplicates().tolist()
    lane_to_y = {lane: i for i, lane in enumerate(lanes)}
    job_to_color = get_job_colors(merged["job_id"].tolist())

    fig_height = max(6, len(lanes) * 0.55)
    fig, ax = plt.subplots(figsize=(22, fig_height))

    for _, row in merged.iterrows():
        y = lane_to_y[row["lane"]]
        start = row["start_time_day"]
        duration = row["duration"]
        job_id = int(row["job_id"])
        hatch = "///" if bool(row["from_backlog"]) else None
        alpha = 0.85 if row["day_type"] == "weekend" else 0.95

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.45,
            color=job_to_color[job_id],
            edgecolor="black",
            hatch=hatch,
            alpha=alpha,
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
    ax.set_ylabel("Executed Day")
    ax.set_title("Merged Daily Gantt (Shift View)", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.axvline(480, color="red", linestyle="--", linewidth=1.5, label="Shift limit (480 min)")

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Current-day queue"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="From backlog"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    plt.tight_layout()
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved merged shift Gantt to: {out}")
    plt.show()


# =====================================================
# Mode 3: merged_calendar
# =====================================================

def calendar_label(executed_day: int, day_type: str, weekday_horizon_days: int = 20) -> str:
    """
    Interleave weekends after every 5 weekdays.

    Examples
    --------
    executed_day 1  -> W1D1
    executed_day 5  -> W1D5
    executed_day 21 -> W1-Sat
    executed_day 22 -> W1-Sun
    executed_day 23 -> W2-Sat
    executed_day 24 -> W2-Sun
    """
    if day_type == "weekday":
        week_idx = (executed_day - 1) // 5 + 1
        day_in_week = (executed_day - 1) % 5 + 1
        return f"Week {week_idx} - WD{day_in_week}"

    weekend_idx = executed_day - weekday_horizon_days
    week_idx = (weekend_idx - 1) // 2 + 1
    weekend_pos = (weekend_idx - 1) % 2
    weekend_name = "Sat" if weekend_pos == 0 else "Sun"
    return f"Week {week_idx} - {weekend_name}"


def build_calendar_lane_order(merged: pd.DataFrame, weekday_horizon_days: int = 20) -> list[str]:
    """
    Build a human-readable lane order:
    Week 1 - WD1 ... WD5, Week 1 - Sat, Week 1 - Sun,
    Week 2 - WD1 ... WD5, Week 2 - Sat, Week 2 - Sun, ...
    """
    used_labels = set(
        merged.apply(
            lambda r: calendar_label(int(r["executed_day"]), str(r["day_type"]), weekday_horizon_days),
            axis=1
        ).tolist()
    )

    lanes = []
    n_weeks = weekday_horizon_days // 5
    for w in range(1, n_weeks + 1):
        for wd in range(1, 6):
            label = f"Week {w} - WD{wd}"
            if label in used_labels:
                lanes.append(label)

        sat = f"Week {w} - Sat"
        sun = f"Week {w} - Sun"
        if sat in used_labels:
            lanes.append(sat)
        if sun in used_labels:
            lanes.append(sun)

    # In case extra labels exist unexpectedly
    for label in sorted(used_labels):
        if label not in lanes:
            lanes.append(label)

    return lanes


def plot_merged_calendar(df: pd.DataFrame, out_path: str | None = None, weekday_horizon_days: int = 20):
    merged = build_merged_job_blocks(df)
    merged["lane"] = merged.apply(
        lambda r: calendar_label(int(r["executed_day"]), str(r["day_type"]), weekday_horizon_days),
        axis=1
    )

    lanes = build_calendar_lane_order(merged, weekday_horizon_days=weekday_horizon_days)
    lane_to_y = {lane: i for i, lane in enumerate(lanes)}
    job_to_color = get_job_colors(merged["job_id"].tolist())

    fig_height = max(7, len(lanes) * 0.55)
    fig, ax = plt.subplots(figsize=(22, fig_height))

    for _, row in merged.iterrows():
        y = lane_to_y[row["lane"]]
        start = row["start_time_day"]
        duration = row["duration"]
        job_id = int(row["job_id"])
        hatch = "///" if bool(row["from_backlog"]) else None
        alpha = 0.85 if row["day_type"] == "weekend" else 0.95

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.45,
            color=job_to_color[job_id],
            edgecolor="black",
            hatch=hatch,
            alpha=alpha,
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
    ax.set_ylabel("Calendar Day")
    ax.set_title("Merged Gantt Chart (Calendar View with Interleaved Weekends)", fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.axvline(480, color="red", linestyle="--", linewidth=1.5, label="Shift limit (480 min)")

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Current-day queue"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="From backlog"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    info_text = (
        "Notes:\n"
        "- Each bar is one full job (A+B merged).\n"
        "- Weekends are interleaved after every 5 weekdays.\n"
        "- Hatched bars indicate backlog jobs.\n"
        "- Red dashed line marks the shift limit.\n"
    )
    ax.text(
        1.02, 0.72,
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
        print(f"Saved merged calendar Gantt to: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Gantt charts from simulation event-log CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/simulation_outputs/gantt_events_single_run.csv",
        help="Path to event-log CSV.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detailed_global", "merged_shift", "merged_calendar", "all"],
        default="all",
        help="Which Gantt plot(s) to generate.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--weekday-horizon-days",
        type=int,
        default=20,
        help="Number of planned weekday days used to build calendar labels.",
    )
    args = parser.parse_args()

    df = load_event_log(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("detailed_global", "all"):
        plot_detailed_global(
            df,
            out_path=str(out_dir / "gantt_detailed_global.png"),
        )

    if args.mode in ("merged_shift", "all"):
        plot_merged_shift(
            df,
            out_path=str(out_dir / "gantt_merged_shift.png"),
        )

    if args.mode in ("merged_calendar", "all"):
        plot_merged_calendar(
            df,
            out_path=str(out_dir / "gantt_merged_calendar.png"),
            weekday_horizon_days=args.weekday_horizon_days,
        )


if __name__ == "__main__":
    main()