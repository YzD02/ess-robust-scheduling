from __future__ import annotations

"""Exploratory analysis of Samjin factory data (E5SP TOP Tape line, Jan–Apr).

What this script does
---------------------
Performs an ETL (Extract → Transform → Load/Report) pipeline on the raw
Excel file received from the company.  It extracts three data sources from
the workbook, cleans and reshapes each one, prints summary statistics to the
console, and saves analysis-ready CSV files for downstream use.

Data sources extracted
----------------------
1. **Production records** (sheets: Jan / Feb / Mar / Apr)
   Daily plan vs actual output, achievement rate, and output-per-person,
   split by day shift / night shift / daily total.

2. **Machine trouble log** (sheet: Automation Results)
   Per-operating-day: stoppage count (day + night shift), total lost time
   (minutes), and long-duration loss (minutes).

3. **Under-achievement reasons** (sheet: Under-Achievement Reasons)
   Free-text log of root causes on days where output fell short of plan.

How to run
----------
From the project root:

    python -m src.experiments.explore_factory_data \\
        --input  data/E5SP_TOP_Tape_Production_Records_Jan-Apr.xlsx \\
        --out-dir results/factory_eda

Outputs
-------
All files are saved to ``--out-dir`` (default: ``results/factory_eda``):

    production_daily.csv        Daily output records (all months combined)
    machine_trouble_daily.csv   Daily stoppage counts and lost-time minutes
    underachievement_log.csv    Cleaned under-achievement reason log
    eda_summary.txt             Key statistics printed to a text file
    figures/
      fig1_achievement_rate_distribution.png
      fig2_achievement_rate_timeline.png
      fig3_stoppage_two_layer_structure.png
      fig4_lost_time_composition.png
      fig5_failure_category_breakdown.png
      fig6_machinestopcfg_calibration.png

Key statistics reported
-----------------------
Production section
  - Daily actual output: mean, median, std, min, max, p10, p90
  - Daily achievement rate: same set of stats
  - Output-per-person: same set of stats
  - Monthly breakdown of the above
  - Count and fraction of days below plan (achievement rate < 1.0)

Machine trouble section
  - Daily stoppage count: mean, median, std, min, max
  - Daily total lost time (min): same stats
  - Implied mean uptime between stops (approximation for MachineStopConfig)
  - Implied mean stop duration (approximation for MachineStopConfig)
  - Monthly breakdown

Under-achievement section
  - Counts of each broad failure category (equipment fault, injection-
    machine issue, parts shortage, quality defect, manual/safety)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all terminals
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


# ---------------------------------------------------------------------------
# Sheet and column constants
# ---------------------------------------------------------------------------

MONTHLY_SHEETS = [
    "E5SP TOP (Tape) (Jan)",
    "E5SP TOP (Tape) (Feb)",
    "E5SP TOP (Tape) (Mar)",
    "E5SP TOP (Tape) (Apr)",
]

AUTOMATION_SHEET = "Automation Results (Jan-Apr)"
REASONS_SHEET    = "Under-Achievement Reasons"

# Row indices (0-based) in the Automation Results sheet where data lives.
# These are fixed by the workbook layout confirmed during EDA.
TROUBLE_HEADER_ROW  = 50   # row with date column headers
TROUBLE_DAY_STOP    = 51   # day-shift stoppage count row
TROUBLE_NIGHT_STOP  = 52   # night-shift stoppage count row
TROUBLE_TOTAL_STOP  = 53   # combined stoppage count row
TROUBLE_DAY_LOST    = 54   # day-shift lost time (minutes) row
TROUBLE_NIGHT_LOST  = 55   # night-shift lost time (minutes) row
TROUBLE_LONG_LOSS   = 56   # long-duration loss (minutes) row
TROUBLE_TOTAL_LOST  = 57   # total lost time (minutes) row

# Row indices in the Automation Results sheet for the production metrics block.
PROD_HEADER_ROW  = 15  # row with date column headers (same dates as trouble log)
PROD_INPUT_ROW   = 16  # total input qty row
PROD_GOOD_ROW    = 17  # good qty row
PROD_INPROC_ROW  = 18  # in-process defect count row
PROD_FINAL_ROW   = 19  # final defect count row

# Assumed shift length used to compute per-unit cycle time proxies.
SHIFT_MINUTES   = 480.0   # one 8-hour shift
WORKERS_PER_SHIFT = 3     # confirmed constant across all operating days


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _percentile(series: pd.Series, q: float) -> float:
    """Return the q-th percentile (0–100 scale) of a numeric series."""
    return float(np.nanpercentile(series.dropna().values, q))


def _summary_stats(series: pd.Series, label: str) -> dict:
    """Return a dict of key statistics for a numeric series."""
    s = series.dropna()
    return {
        "metric":   label,
        "n":        int(s.count()),
        "mean":     round(float(s.mean()),   2),
        "median":   round(float(s.median()), 2),
        "std":      round(float(s.std()),    2),
        "min":      round(float(s.min()),    2),
        "p10":      round(_percentile(s, 10), 2),
        "p25":      round(_percentile(s, 25), 2),
        "p75":      round(_percentile(s, 75), 2),
        "p90":      round(_percentile(s, 90), 2),
        "max":      round(float(s.max()),    2),
    }


def _print_stats(stats: dict, out_lines: list[str]) -> None:
    """Format a stats dict and append to out_lines for console + file output."""
    line = (
        f"  {stats['metric']:<40s}  "
        f"n={stats['n']:3d}  "
        f"mean={stats['mean']:8.2f}  "
        f"median={stats['median']:8.2f}  "
        f"std={stats['std']:7.2f}  "
        f"min={stats['min']:7.2f}  "
        f"p10={stats['p10']:7.2f}  "
        f"p90={stats['p90']:7.2f}  "
        f"max={stats['max']:7.2f}"
    )
    out_lines.append(line)
    print(line)


# ---------------------------------------------------------------------------
# ETL — Production records (monthly sheets)
# ---------------------------------------------------------------------------

def extract_production_records(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse and combine the four monthly production sheets into one DataFrame.

    Each sheet has a 3-row merged header (rows 1–3) followed by daily data
    rows.  Week-subtotal rows and the final monthly-total row are identified
    by a non-date value in the date column and are excluded.

    Returns a tidy DataFrame with one row per operating day and columns:
        date, month, shift,
        net_workers, assigned_workers,
        plan_qty, actual_qty,
        achievement_rate, output_per_person
    """
    records = []

    for sheet in MONTHLY_SHEETS:
        # Read without a header so we can handle the multi-row layout manually.
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)

        # The usable data starts at row index 4 (0-based).
        # Column layout (confirmed by inspection):
        #   col 0  : (empty)
        #   col 1  : date / week-label
        #   col 2  : day-shift net workers
        #   col 3  : day-shift assigned workers
        #   col 4  : day-shift plan qty
        #   col 5  : day-shift actual qty
        #   col 6  : day-shift achievement rate
        #   col 7  : day-shift output per person
        #   col 8  : night-shift net workers
        #   col 9  : night-shift assigned workers
        #   col 10 : night-shift plan qty
        #   col 11 : night-shift actual qty
        #   col 12 : night-shift achievement rate
        #   col 13 : night-shift output per person
        #   col 14 : daily-total net workers
        #   col 15 : daily-total assigned workers
        #   col 16 : daily-total plan qty
        #   col 17 : daily-total actual qty
        #   col 18 : daily-total achievement rate
        #   col 19 : daily-total output per person

        data = raw.iloc[4:].copy()
        data.columns = range(data.shape[1])

        # Keep only rows where col 1 is a datetime (operating days).
        # Week-subtotal rows have string labels like "Week 1"; monthly-total
        # rows have "Total"; both are dropped here.
        data = data[pd.to_datetime(data[1], errors="coerce").notna()].copy()
        data[1] = pd.to_datetime(data[1])

        month_label = sheet.split("(")[-1].rstrip(")")  # "Jan", "Feb", …

        for _, row in data.iterrows():
            date = row[1]

            # Day shift — only include if actual qty is a valid number.
            if pd.notna(row[5]):
                records.append({
                    "date":              date,
                    "month":             month_label,
                    "shift":             "day",
                    "net_workers":       row[2],
                    "assigned_workers":  row[3],
                    "plan_qty":          row[4],
                    "actual_qty":        row[5],
                    "achievement_rate":  row[6],
                    "output_per_person": row[7],
                })

            # Night shift
            if pd.notna(row[11]):
                records.append({
                    "date":              date,
                    "month":             month_label,
                    "shift":             "night",
                    "net_workers":       row[8],
                    "assigned_workers":  row[9],
                    "plan_qty":          row[10],
                    "actual_qty":        row[11],
                    "achievement_rate":  row[12],
                    "output_per_person": row[13],
                })

            # Daily total
            if pd.notna(row[17]):
                records.append({
                    "date":              date,
                    "month":             month_label,
                    "shift":             "total",
                    "net_workers":       row[14],
                    "assigned_workers":  row[15],
                    "plan_qty":          row[16],
                    "actual_qty":        row[17],
                    "achievement_rate":  row[18],
                    "output_per_person": row[19],
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "shift"]).reset_index(drop=True)

    # Coerce numeric columns (they may arrive as object dtype due to mixed rows).
    for col in ["plan_qty", "actual_qty", "achievement_rate", "output_per_person",
                "net_workers", "assigned_workers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# ETL — Machine trouble log (Automation Results sheet)
# ---------------------------------------------------------------------------

def extract_machine_trouble(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse the Tape Auto-Applicator Trouble Log from the Automation Results sheet.

    The trouble log is a wide table where each column is one operating day.
    Rows of interest (0-based):
        Row 50 : date headers
        Row 51 : day-shift stoppage count
        Row 52 : night-shift stoppage count
        Row 53 : combined stoppage count
        Row 54 : day-shift lost time (minutes)   [micro-stops]
        Row 55 : night-shift lost time (minutes) [micro-stops]
        Row 56 : long-duration loss (minutes)    [single incidents ≥ threshold]
        Row 57 : total lost time (minutes)

    Returns a tidy DataFrame with one row per operating day and columns:
        date,
        stoppage_count_day, stoppage_count_night, stoppage_count_total,
        lost_time_day_min, lost_time_night_min,
        long_duration_loss_min, total_lost_time_min
    """
    raw = pd.read_excel(xl, sheet_name=AUTOMATION_SHEET, header=None)

    # Extract the header row to get dates.
    header_row = raw.iloc[TROUBLE_HEADER_ROW]

    # Columns 2 onwards are operating-day dates (column 0 is empty,
    # column 1 holds the row labels "Category", "Day Shift", etc.).
    date_cols = {}
    for col_idx in range(2, raw.shape[1]):
        cell = header_row.iloc[col_idx]
        parsed = pd.to_datetime(cell, errors="coerce")
        if pd.notna(parsed):
            date_cols[col_idx] = parsed

    if not date_cols:
        raise ValueError(
            "Could not find any date columns in the trouble log header row. "
            "Check that TROUBLE_HEADER_ROW is still correct for this workbook."
        )

    def _extract_row(row_idx: int) -> dict[int, float]:
        """Return {col_index: numeric_value} for a given row."""
        row = raw.iloc[row_idx]
        result = {}
        for col_idx in date_cols:
            val = pd.to_numeric(row.iloc[col_idx], errors="coerce")
            result[col_idx] = float(val) if pd.notna(val) else np.nan
        return result

    day_stops   = _extract_row(TROUBLE_DAY_STOP)
    night_stops = _extract_row(TROUBLE_NIGHT_STOP)
    total_stops = _extract_row(TROUBLE_TOTAL_STOP)
    day_lost    = _extract_row(TROUBLE_DAY_LOST)
    night_lost  = _extract_row(TROUBLE_NIGHT_LOST)
    long_loss   = _extract_row(TROUBLE_LONG_LOSS)
    total_lost  = _extract_row(TROUBLE_TOTAL_LOST)

    records = []
    for col_idx, date in date_cols.items():
        records.append({
            "date":                   date,
            "stoppage_count_day":     day_stops[col_idx],
            "stoppage_count_night":   night_stops[col_idx],
            "stoppage_count_total":   total_stops[col_idx],
            "lost_time_day_min":      day_lost[col_idx],
            "lost_time_night_min":    night_lost[col_idx],
            "long_duration_loss_min": long_loss[col_idx],
            "total_lost_time_min":    total_lost[col_idx],
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.strftime("%b")

    # Drop rows where all numeric columns are zero or NaN — these correspond
    # to non-operating days that appear in the header date range.
    numeric_cols = [c for c in df.columns if c not in ("date", "month")]
    df = df[df[numeric_cols].sum(axis=1) > 0].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# ETL — Under-achievement reasons
# ---------------------------------------------------------------------------

def extract_underachievement_reasons(xl: pd.ExcelFile) -> pd.DataFrame:
    """Parse the Under-Achievement Reasons sheet into a tidy DataFrame.

    Columns in the output:
        date, product, achievement_rate, reason, countermeasure,
        failure_category (derived by keyword matching on reason text)
    """
    raw = pd.read_excel(xl, sheet_name=REASONS_SHEET, header=None)

    # The data table starts at row 2 (0-based).
    # Col layout: 0=empty, 1=product(merged), 2=date, 3=achievement_rate,
    #             4=reason, 5=countermeasure
    data = raw.iloc[2:].copy()
    data = data.rename(columns={
        0: "_empty",
        1: "product",
        2: "date_raw",
        3: "achievement_rate",
        4: "reason",
        5: "countermeasure",
    })

    # Forward-fill the product column (it is merged across groups of rows).
    data["product"] = data["product"].ffill()

    # Keep only rows with a valid date.
    data["date"] = pd.to_datetime(data["date_raw"], errors="coerce")
    data = data[data["date"].notna()].copy()

    # Coerce achievement rate; rows with "Non-operating" in reason have no rate.
    data["achievement_rate"] = pd.to_numeric(data["achievement_rate"], errors="coerce")

    # Derive a broad failure category from keyword matching on the reason text.
    def _categorise(reason: str) -> str:
        if not isinstance(reason, str):
            return "no_issue"
        r = reason.lower()
        if "non-operating" in r or r.strip() == "":
            return "non_operating"
        if "no notable issues" in r:
            return "no_issue"
        if "injection" in r or "mold" in r or "ejector" in r or "molded parts" in r:
            return "injection_machine"
        if "parts shortage" in r or "part shortage" in r:
            return "parts_shortage"
        if "safety" in r or "refused" in r:
            return "safety_incident"
        if "defect" in r or "dislodge" in r or "tape" in r and "fault" in r:
            return "quality_defect"
        if "vacuum" in r or "suction" in r or "axis" in r or "robot" in r \
                or "wire" in r or "belt" in r or "chuck" in r or "jig" in r \
                or "valve" in r or "sensor" in r or "alarm" in r:
            return "equipment_fault"
        return "other"

    data["failure_category"] = data["reason"].apply(_categorise)

    result = data[
        ["date", "product", "achievement_rate", "reason", "countermeasure", "failure_category"]
    ].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyse_production(df: pd.DataFrame, out_lines: list[str]) -> None:
    """Print summary statistics for the production records DataFrame."""
    header = "\n" + "=" * 80 + "\n  PRODUCTION RECORDS — KEY STATISTICS\n" + "=" * 80
    print(header)
    out_lines.append(header)

    # ------------------------------------------------------------------
    # Overall stats across all operating days
    # ------------------------------------------------------------------
    section = "\n--- Overall (all months, daily-total rows only) ---"
    print(section)
    out_lines.append(section)

    total_df = df[df["shift"] == "total"].copy()

    for label, col in [
        ("Daily actual output (units)",   "actual_qty"),
        ("Daily plan qty (units)",         "plan_qty"),
        ("Achievement rate",               "achievement_rate"),
        ("Output per person (units/shift)","output_per_person"),
    ]:
        _print_stats(_summary_stats(total_df[col], label), out_lines)

    # Fraction of days below plan
    below_plan = (total_df["achievement_rate"] < 1.0).sum()
    total_days  = total_df["achievement_rate"].notna().sum()
    pct = 100 * below_plan / total_days if total_days else 0
    line = f"\n  Days below plan (rate < 1.0): {below_plan} / {total_days}  ({pct:.1f}%)"
    print(line)
    out_lines.append(line)

    # ------------------------------------------------------------------
    # Cycle-time proxy from output-per-person
    # Each worker processes output_per_person units in a SHIFT_MINUTES shift.
    # Implied minutes-per-unit = SHIFT_MINUTES / output_per_person.
    # This is a LINE-level C/T proxy — not station-level — but it is the best
    # estimate available from aggregate data.
    # ------------------------------------------------------------------
    ct_section = "\n--- Implied cycle time proxy (min/unit = shift_minutes / output_per_person) ---"
    print(ct_section)
    out_lines.append(ct_section)

    ct_note = (
        "  NOTE: Individual station C/T data is not available (company confirmed\n"
        "  they track only line-level daily output, not per-worker or per-station\n"
        "  cycle times).  The proxy below is computed as:\n"
        "      implied_ct_min = SHIFT_MINUTES / output_per_person\n"
        "  where SHIFT_MINUTES = 480 and output_per_person is the units produced\n"
        "  per worker per shift.  This reflects total processing time per unit\n"
        "  (both stations combined), not the individual station A or B C/T."
    )
    print(ct_note)
    out_lines.append(ct_note)

    # Compute CT proxy on per-shift rows (day and night separately)
    shift_df = df[df["shift"].isin(["day", "night"])].copy()
    shift_df = shift_df[shift_df["output_per_person"] > 0].copy()
    shift_df["implied_ct_min"] = SHIFT_MINUTES / shift_df["output_per_person"]

    for label, subset in [
        ("Implied C/T — all shifts",   shift_df),
        ("Implied C/T — day shift",    shift_df[shift_df["shift"] == "day"]),
        ("Implied C/T — night shift",  shift_df[shift_df["shift"] == "night"]),
    ]:
        _print_stats(_summary_stats(subset["implied_ct_min"], label), out_lines)

    # ------------------------------------------------------------------
    # Monthly breakdown
    # ------------------------------------------------------------------
    monthly_section = "\n--- Monthly breakdown (daily-total rows) ---"
    print(monthly_section)
    out_lines.append(monthly_section)

    for month in ["Jan", "Feb", "Mar", "Apr"]:
        sub = total_df[total_df["month"] == month]
        if sub.empty:
            continue
        month_header = f"\n  Month: {month}  (n={len(sub)} operating days)"
        print(month_header)
        out_lines.append(month_header)
        for label, col in [
            ("  Actual output",      "actual_qty"),
            ("  Achievement rate",   "achievement_rate"),
            ("  Output per person",  "output_per_person"),
        ]:
            _print_stats(_summary_stats(sub[col], label), out_lines)


def analyse_machine_trouble(df: pd.DataFrame, out_lines: list[str]) -> None:
    """Print summary statistics for the machine trouble log DataFrame.

    Also derives approximate MachineStopConfig parameters:
      - mean_uptime_between_stops : estimated from stoppage count and shift time
      - mean_stop_duration        : estimated from total lost time / stoppage count
    """
    header = "\n" + "=" * 80 + "\n  MACHINE TROUBLE LOG — KEY STATISTICS\n" + "=" * 80
    print(header)
    out_lines.append(header)

    # Overall stats
    section = "\n--- Overall (all operating days) ---"
    print(section)
    out_lines.append(section)

    for label, col in [
        ("Daily stoppage count (total)",         "stoppage_count_total"),
        ("Daily stoppage count (day shift)",      "stoppage_count_day"),
        ("Daily stoppage count (night shift)",    "stoppage_count_night"),
        ("Total lost time per day (min)",         "total_lost_time_min"),
        ("Long-duration loss per day (min)",      "long_duration_loss_min"),
    ]:
        _print_stats(_summary_stats(df[col], label), out_lines)

    # ------------------------------------------------------------------
    # MachineStopConfig parameter estimation
    # ------------------------------------------------------------------
    # The trouble log records:
    #   - stoppage_count_total : number of micro-stop events in one operating day
    #   - total_lost_time_min  : total minutes lost to all stops
    #
    # Current simulation model uses:
    #   inter-arrival time ~ Exponential(mean = mean_uptime_between_stops)
    #   stop duration      ~ Gamma(shape, scale) with mean = mean_stop_duration
    #
    # Approximation approach:
    #   mean_stop_duration        ≈ total_lost_time_min / stoppage_count_total
    #   effective_run_time        ≈ SHIFT_MINUTES * 2 - total_lost_time_min
    #                                (two shifts per day)
    #   mean_uptime_between_stops ≈ effective_run_time / stoppage_count_total
    # ------------------------------------------------------------------
    param_section = (
        "\n--- MachineStopConfig parameter estimates ---\n"
        "  (These are approximations from aggregated daily data.\n"
        "   Per-incident timestamps are not available in this dataset.)"
    )
    print(param_section)
    out_lines.append(param_section)

    valid = df[
        (df["stoppage_count_total"] > 0) &
        (df["total_lost_time_min"]  > 0)
    ].copy()

    # Estimated mean stop duration per incident (minutes)
    valid["est_mean_stop_dur"] = valid["total_lost_time_min"] / valid["stoppage_count_total"]

    # Estimated run time per day (both shifts minus lost time)
    valid["est_run_time"] = (SHIFT_MINUTES * 2) - valid["total_lost_time_min"]
    valid["est_mean_uptime"] = valid["est_run_time"] / valid["stoppage_count_total"]

    overall_mean_stop_dur  = valid["est_mean_stop_dur"].mean()
    overall_median_stop_dur = valid["est_mean_stop_dur"].median()
    overall_mean_uptime    = valid["est_mean_uptime"].mean()
    overall_median_uptime  = valid["est_mean_uptime"].median()

    param_lines = [
        f"  Estimated mean stop duration:          mean={overall_mean_stop_dur:.2f} min  "
        f"median={overall_median_stop_dur:.2f} min  (n={len(valid)} days)",
        f"  Estimated mean uptime between stops:   mean={overall_mean_uptime:.2f} min  "
        f"median={overall_median_uptime:.2f} min",
        f"\n  Suggested MachineStopConfig values (use with caution — aggregate data only):",
        f"    mean_stop_duration        = {overall_mean_stop_dur:.1f}",
        f"    mean_uptime_between_stops = {overall_mean_uptime:.1f}",
        f"  (Current defaults: mean_stop_duration=8.0, mean_uptime_between_stops=68.57)",
    ]
    for line in param_lines:
        print(line)
        out_lines.append(line)

    _print_stats(_summary_stats(valid["est_mean_stop_dur"],  "Per-day est. mean stop duration (min)"), out_lines)
    _print_stats(_summary_stats(valid["est_mean_uptime"],    "Per-day est. mean uptime between stops (min)"), out_lines)

    # Monthly breakdown
    monthly_section = "\n--- Monthly breakdown ---"
    print(monthly_section)
    out_lines.append(monthly_section)

    for month in ["Jan", "Feb", "Mar", "Apr"]:
        sub = df[df["month"] == month]
        if sub.empty:
            continue
        month_header = f"\n  Month: {month}  (n={len(sub)} operating days)"
        print(month_header)
        out_lines.append(month_header)
        for label, col in [
            ("  Stoppage count (total)",   "stoppage_count_total"),
            ("  Total lost time (min)",    "total_lost_time_min"),
        ]:
            _print_stats(_summary_stats(sub[col], label), out_lines)


def analyse_underachievement(df: pd.DataFrame, out_lines: list[str]) -> None:
    """Print failure category counts from the under-achievement reasons log."""
    header = "\n" + "=" * 80 + "\n  UNDER-ACHIEVEMENT REASONS — FAILURE CATEGORY COUNTS\n" + "=" * 80
    print(header)
    out_lines.append(header)

    # Exclude non-operating days from counts
    operating = df[df["failure_category"] != "non_operating"].copy()
    n_operating = len(operating)

    section = f"\n--- Total operating days in log: {n_operating} ---"
    print(section)
    out_lines.append(section)

    counts = operating["failure_category"].value_counts()
    for category, count in counts.items():
        pct = 100 * count / n_operating
        line = f"  {category:<25s}  {count:3d}  ({pct:.1f}%)"
        print(line)
        out_lines.append(line)

    # Days where achievement rate was below plan AND reason is recorded
    below_plan = operating[
        operating["achievement_rate"].notna() &
        (operating["achievement_rate"] < 1.0)
    ]
    below_section = (
        f"\n--- Failure categories on days below plan "
        f"(achievement rate < 1.0, n={len(below_plan)}) ---"
    )
    print(below_section)
    out_lines.append(below_section)

    below_counts = below_plan["failure_category"].value_counts()
    for category, count in below_counts.items():
        pct = 100 * count / len(below_plan)
        line = f"  {category:<25s}  {count:3d}  ({pct:.1f}%)"
        print(line)
        out_lines.append(line)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Shared style applied to every figure.
PALETTE   = ["#2E4057", "#E84855", "#F4A261", "#6DB65B", "#9B89C4"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr"]
MONTH_COLORS = {
    "Jan": "#2E4057",
    "Feb": "#6DB65B",
    "Mar": "#F4A261",
    "Apr": "#E84855",
}


def _apply_base_style() -> None:
    """Set a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F8F8",
        "axes.edgecolor":    "#CCCCCC",
        "axes.grid":         True,
        "grid.color":        "white",
        "grid.linewidth":    1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
    })


def _save(fig: plt.Figure, path: Path, label: str) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {path.name}  [{label}]")


# ---- Figure 1 : Achievement Rate Distribution ----------------------------

def plot_achievement_distribution(
    production_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Histogram + KDE of daily achievement rate, coloured by month.

    Shows the bimodal shape: a tight cluster of normal/above-plan days
    (≥1.0) and a heavier left tail of disrupted days (<1.0).
    """
    _apply_base_style()
    total = production_df[production_df["shift"] == "total"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Fig 1 — Daily Achievement Rate Distribution  (Jan–Apr)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: overall histogram + KDE
    ax = axes[0]
    ax.axvline(1.0, color="#E84855", linewidth=1.5, linestyle="--",
               label="Plan = 1.0", zorder=3)
    bins = np.arange(0.4, 1.45, 0.05)
    ax.hist(total["achievement_rate"].dropna(), bins=bins,
            color="#2E4057", alpha=0.75, edgecolor="white", linewidth=0.6, zorder=2)
    # Overlay KDE
    from scipy.stats import gaussian_kde
    vals = total["achievement_rate"].dropna().values
    kde  = gaussian_kde(vals, bw_method=0.25)
    xs   = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 300)
    ax2  = ax.twinx()
    ax2.plot(xs, kde(xs), color="#E84855", linewidth=2, zorder=4)
    ax2.set_ylabel("Density", color="#E84855")
    ax2.tick_params(axis="y", labelcolor="#E84855")
    ax2.set_ylim(bottom=0)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Achievement Rate  (actual / plan)")
    ax.set_ylabel("Number of Days")
    ax.set_title("Overall Distribution  (n=81 operating days)")
    ax.legend(loc="upper left")

    # Annotate below-plan fraction
    below = (total["achievement_rate"] < 1.0).sum()
    total_n = total["achievement_rate"].notna().sum()
    ax.text(0.44, ax.get_ylim()[1] * 0.85,
            f"Below plan:\n{below}/{total_n} days\n({100*below/total_n:.0f}%)",
            fontsize=9, color="#E84855",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E84855", alpha=0.8))

    # Right: per-month box plot
    ax = axes[1]
    month_data = [
        total.loc[total["month"] == m, "achievement_rate"].dropna().values
        for m in MONTH_ORDER
    ]
    bp = ax.boxplot(
        month_data, patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="o", markersize=4,
                        markerfacecolor="#E84855", alpha=0.6),
    )
    for patch, month in zip(bp["boxes"], MONTH_ORDER):
        patch.set_facecolor(MONTH_COLORS[month])
        patch.set_alpha(0.8)
    ax.axhline(1.0, color="#E84855", linewidth=1.5, linestyle="--",
               label="Plan = 1.0")
    ax.set_xticklabels(MONTH_ORDER)
    ax.set_xlabel("Month")
    ax.set_ylabel("Achievement Rate")
    ax.set_title("Monthly Breakdown")
    ax.legend(loc="lower right")

    # Add median labels above each box
    for i, m in enumerate(MONTH_ORDER):
        med = np.median(month_data[i]) if len(month_data[i]) else np.nan
        if not np.isnan(med):
            ax.text(i + 1, med + 0.02, f"{med:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, fig_dir / "fig1_achievement_rate_distribution.png",
          "Achievement rate distribution")


# ---- Figure 2 : Achievement Rate Timeline --------------------------------

def plot_achievement_timeline(
    production_df: pd.DataFrame,
    reasons_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Time-series of daily achievement rate with failure events annotated.

    Disrupted days are colour-coded by failure category so the reader can
    see when and why production fell below plan.
    """
    _apply_base_style()
    total = production_df[production_df["shift"] == "total"].copy()
    total = total.sort_values("date")

    # Merge failure category onto production rows
    reasons_clean = reasons_df[["date", "failure_category"]].copy()
    reasons_clean["date"] = pd.to_datetime(reasons_clean["date"])
    total["date"] = pd.to_datetime(total["date"])
    merged = total.merge(reasons_clean, on="date", how="left")
    merged["failure_category"] = merged["failure_category"].fillna("no_issue")

    cat_colors = {
        "no_issue":         "#AAAAAA",
        "injection_machine":"#E84855",
        "equipment_fault":  "#F4A261",
        "quality_defect":   "#9B89C4",
        "parts_shortage":   "#E84855",
        "safety_incident":  "#2E4057",
        "other":            "#6DB65B",
    }

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.suptitle(
        "Fig 2 — Daily Achievement Rate Timeline  (Jan–Apr)",
        fontsize=13, fontweight="bold",
    )

    # Draw the plan line
    ax.axhline(1.0, color="#333333", linewidth=1.2, linestyle="--",
               alpha=0.6, label="Plan = 1.0", zorder=2)

    # Shade months alternately
    month_ranges = merged.groupby("month")["date"].agg(["min", "max"])
    for i, (m, row_m) in enumerate(month_ranges.iterrows()):
        ax.axvspan(row_m["min"], row_m["max"],
                   alpha=0.06 if i % 2 == 0 else 0.0,
                   color="#2E4057", zorder=0)
        ax.text(row_m["min"] + (row_m["max"] - row_m["min"]) / 2,
                1.38, m, ha="center", va="center",
                fontsize=9, color="#555555", fontweight="bold")

    # Plot each point, coloured by failure category
    for _, row in merged.iterrows():
        if pd.isna(row["achievement_rate"]):
            continue
        cat   = row["failure_category"]
        color = cat_colors.get(cat, "#AAAAAA")
        size  = 60 if row["achievement_rate"] < 1.0 else 30
        zord  = 4 if row["achievement_rate"] < 1.0 else 3
        ax.scatter(row["date"], row["achievement_rate"],
                   color=color, s=size, zorder=zord,
                   edgecolors="white", linewidths=0.4, alpha=0.9)

    # Connect points with a light line
    ax.plot(merged["date"], merged["achievement_rate"],
            color="#CCCCCC", linewidth=0.8, zorder=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Achievement Rate")
    ax.set_ylim(0.35, 1.45)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Legend
    legend_items = [
        mpatches.Patch(color=cat_colors["injection_machine"], label="Injection machine"),
        mpatches.Patch(color=cat_colors["equipment_fault"],   label="Equipment fault"),
        mpatches.Patch(color=cat_colors["quality_defect"],    label="Quality defect"),
        mpatches.Patch(color=cat_colors["safety_incident"],   label="Safety incident"),
        mpatches.Patch(color=cat_colors["no_issue"],          label="No notable issue"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Plan = 1.0"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              ncol=3, framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save(fig, fig_dir / "fig2_achievement_rate_timeline.png",
          "Achievement rate timeline")


# ---- Figure 3 : Two-Layer Stoppage Structure -----------------------------

def plot_two_layer_stoppage(
    trouble_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Illustrates that total lost time is driven by two independent mechanisms.

    Panel A: scatter of stoppage count vs total lost time, coloured by
             presence of a long-duration event.  Shows that high lost-time
             days are caused by long-duration events, NOT by more micro-stops.
    Panel B: distribution of micro-stop-only lost time per shift — the
             stable background noise that should drive MachineStopConfig.
    """
    _apply_base_style()
    df = trouble_df.copy()
    df["micro_stop_lost"] = df["total_lost_time_min"] - df["long_duration_loss_min"]
    df["has_long"]        = df["long_duration_loss_min"] > 0
    df["micro_per_shift"] = df["micro_stop_lost"] / 2   # two shifts per day

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Fig 3 — Two-Layer Stoppage Structure: Micro-stops vs Long-duration Events",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Panel A: scatter
    ax = axes[0]
    colors = df["has_long"].map({False: "#2E4057", True: "#E84855"})
    ax.scatter(df["stoppage_count_total"], df["total_lost_time_min"],
               c=colors, alpha=0.75, s=55, edgecolors="white", linewidths=0.5)

    # Correlation annotation
    corr = df["stoppage_count_total"].corr(df["total_lost_time_min"])
    ax.text(0.97, 0.96, f"r = {corr:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.8))

    legend_items = [
        mpatches.Patch(color="#2E4057", alpha=0.8, label="Micro-stops only"),
        mpatches.Patch(color="#E84855", alpha=0.8, label="Long-duration event present"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8)
    ax.set_xlabel("Daily Stoppage Count  (micro-stops)")
    ax.set_ylabel("Total Lost Time  (min/day)")
    ax.set_title(
        "Panel A: Stoppage Count vs Total Lost Time\n"
        "(low r shows stoppage count ≠ lost time driver)"
    )

    # Panel B: histogram of micro-stop lost time per shift
    ax = axes[1]
    vals = df["micro_per_shift"].dropna()
    ax.hist(vals, bins=20, color="#2E4057", alpha=0.8,
            edgecolor="white", linewidth=0.6)
    ax.axvline(vals.mean(),   color="#E84855",  linewidth=2,
               linestyle="-",  label=f"Mean = {vals.mean():.0f} min")
    ax.axvline(vals.median(), color="#F4A261",  linewidth=2,
               linestyle="--", label=f"Median = {vals.median():.0f} min")

    # Mark the 6% threshold line
    pct_line = 480 * 0.0625
    ax.axvline(pct_line, color="#6DB65B", linewidth=1.5, linestyle=":",
               label=f"6.25% of shift = {pct_line:.0f} min")

    ax.set_xlabel("Micro-stop Lost Time per Shift  (min)")
    ax.set_ylabel("Number of Days")
    ax.set_title(
        "Panel B: Micro-stop Lost Time per Shift\n"
        "(stable background noise for MachineStopConfig)"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, fig_dir / "fig3_stoppage_two_layer_structure.png",
          "Two-layer stoppage structure")


# ---- Figure 4 : Lost-Time Composition Over Time --------------------------

def plot_lost_time_composition(
    trouble_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Stacked bar chart: micro-stop vs long-duration lost time per operating day.

    Sorted chronologically so the reader can see whether the long-duration
    events cluster in any particular period (they do — January and February).
    """
    _apply_base_style()
    df = trouble_df.copy().sort_values("date")
    df["micro"] = (df["total_lost_time_min"] - df["long_duration_loss_min"]).clip(lower=0)
    df["long"]  = df["long_duration_loss_min"].fillna(0)
    df["label"] = df["date"].dt.strftime("%b %d")

    fig, ax = plt.subplots(figsize=(16, 4.5))
    fig.suptitle(
        "Fig 4 — Daily Lost-Time Composition  (micro-stops vs long-duration events)",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(len(df))
    ax.bar(x, df["micro"], color="#2E4057", alpha=0.85,
           label="Micro-stop lost time", zorder=2)
    ax.bar(x, df["long"],  bottom=df["micro"], color="#E84855", alpha=0.85,
           label="Long-duration event", zorder=2)

    # Month dividers
    prev_month = None
    for i, row in df.reset_index(drop=True).iterrows():
        m = row["month"]
        if m != prev_month:
            if prev_month is not None:
                ax.axvline(i - 0.5, color="#AAAAAA",
                           linewidth=1.0, linestyle="--", zorder=3)
            ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 580,
                    m, fontsize=9, color="#555555",
                    fontweight="bold", va="bottom")
            prev_month = m

    ax.set_xticks(x[::3])
    ax.set_xticklabels(df["label"].iloc[::3], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Lost Time  (min/day)")
    ax.set_xlabel("Operating Day")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, df[["micro", "long"]].sum(axis=1).max() * 1.18)

    fig.tight_layout()
    _save(fig, fig_dir / "fig4_lost_time_composition.png",
          "Lost-time composition")


# ---- Figure 5 : Failure Category Breakdown -------------------------------

def plot_failure_categories(
    reasons_df: pd.DataFrame,
    production_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Two panels: (A) failure category frequency; (B) impact on achievement rate.

    Designed to make the injection-machine dominance immediately visible and
    to show the quantitative production hit for each category.
    """
    _apply_base_style()
    operating = reasons_df[reasons_df["failure_category"] != "non_operating"].copy()

    # Compute mean achievement rate per category (merged with production totals)
    total = production_df[production_df["shift"] == "total"].copy()
    total["date"] = pd.to_datetime(total["date"])
    operating["date"] = pd.to_datetime(operating["date"])
    merged = operating.merge(
        total[["date", "achievement_rate"]], on="date", how="left",
        suffixes=("_reason", "_prod"),
    )
    # Use production achievement rate where available
    if "achievement_rate_prod" in merged.columns:
        merged["ar"] = merged["achievement_rate_prod"].combine_first(
            merged.get("achievement_rate_reason")
        )
    else:
        merged["ar"] = merged["achievement_rate"]

    cat_order = (
        operating["failure_category"].value_counts().index.tolist()
    )
    cat_colors_map = {
        "no_issue":         "#AAAAAA",
        "injection_machine":"#E84855",
        "equipment_fault":  "#F4A261",
        "quality_defect":   "#9B89C4",
        "parts_shortage":   "#E84855",
        "safety_incident":  "#2E4057",
        "other":            "#6DB65B",
    }
    colors = [cat_colors_map.get(c, "#AAAAAA") for c in cat_order]

    counts    = operating["failure_category"].value_counts().reindex(cat_order)
    mean_ar   = merged.groupby("failure_category")["ar"].mean().reindex(cat_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Fig 5 — Failure Category Analysis  (Jan–Apr)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Panel A: frequency
    ax = axes[0]
    bars = ax.barh(cat_order[::-1], counts.values[::-1],
                   color=colors[::-1], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({100*val/len(operating):.0f}%)",
                va="center", fontsize=8)
    ax.set_xlabel("Number of Operating Days")
    ax.set_title("Panel A: Frequency of Each Category")
    ax.set_xlim(0, counts.max() * 1.35)

    # Panel B: mean achievement rate per category
    ax = axes[1]
    bars = ax.barh(cat_order[::-1], mean_ar.values[::-1],
                   color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="#333333", linewidth=1.5, linestyle="--",
               label="Plan = 1.0", zorder=3)
    for bar, val in zip(bars, mean_ar.values[::-1]):
        if not np.isnan(val):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Achievement Rate on Affected Days")
    ax.set_title("Panel B: Average Production Impact")
    ax.set_xlim(0, 1.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    _save(fig, fig_dir / "fig5_failure_category_breakdown.png",
          "Failure category breakdown")


# ---- Figure 6 : MachineStopConfig Calibration ----------------------------

def plot_machinestop_calibration(
    trouble_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Compare current MachineStopConfig defaults against data-derived estimates.

    Panel A: distribution of estimated mean stop duration per day.
    Panel B: distribution of estimated mean uptime between stops per day.
    Both panels show the current default as a vertical line so the gap is
    immediately obvious.
    """
    _apply_base_style()
    df = trouble_df.copy()
    df["micro_stop_lost"]  = (df["total_lost_time_min"] - df["long_duration_loss_min"]).clip(lower=0)
    valid = df[(df["stoppage_count_total"] > 0) & (df["micro_stop_lost"] > 0)].copy()

    valid["est_stop_dur"]  = valid["micro_stop_lost"] / valid["stoppage_count_total"]
    valid["est_run_time"]  = (SHIFT_MINUTES * 2) - valid["micro_stop_lost"]
    valid["est_uptime"]    = valid["est_run_time"] / valid["stoppage_count_total"]

    # Current defaults
    DEFAULT_STOP_DUR  = 8.0
    DEFAULT_UPTIME    = 68.57

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Fig 6 — MachineStopConfig Calibration: Data Estimates vs Current Defaults",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, col, default, xlabel, title_suffix in [
        (axes[0], "est_stop_dur",  DEFAULT_STOP_DUR,
         "Estimated Mean Stop Duration  (min/stop)",
         "Mean Stop Duration"),
        (axes[1], "est_uptime",    DEFAULT_UPTIME,
         "Estimated Mean Uptime Between Stops  (min)",
         "Mean Uptime Between Stops"),
    ]:
        vals = valid[col].dropna()
        ax.hist(vals, bins=20, color="#2E4057", alpha=0.8,
                edgecolor="white", linewidth=0.6, label="Data estimate")
        ax.axvline(vals.mean(),   color="#6DB65B", linewidth=2,
                   linestyle="-",
                   label=f"Data mean = {vals.mean():.2f} min")
        ax.axvline(vals.median(), color="#F4A261", linewidth=2,
                   linestyle="--",
                   label=f"Data median = {vals.median():.2f} min")
        ax.axvline(default,       color="#E84855", linewidth=2.5,
                   linestyle="-.",
                   label=f"Current default = {default} min")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Days")
        ax.set_title(f"Panel: {title_suffix}\n(n={len(vals)} operating days)")
        ax.legend(fontsize=8)

    # Add a note about the data limitation
    fig.text(
        0.5, -0.04,
        "Note: estimates are derived from aggregated daily totals (no per-incident timestamps available). "
        "Confirm stoppage-count definition with factory before updating parameters.",
        ha="center", fontsize=8, color="#666666", style="italic",
    )

    fig.tight_layout()
    _save(fig, fig_dir / "fig6_machinestop_calibration.png",
          "MachineStopConfig calibration")


def generate_all_figures(
    production_df: pd.DataFrame,
    trouble_df: pd.DataFrame,
    reasons_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate and save all six EDA figures to ``out_dir/figures/``."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Generating figures ...")
    plot_achievement_distribution(production_df, fig_dir)
    plot_achievement_timeline(production_df, reasons_df, fig_dir)
    plot_two_layer_stoppage(trouble_df, fig_dir)
    plot_lost_time_composition(trouble_df, fig_dir)
    plot_failure_categories(reasons_df, production_df, fig_dir)
    plot_machinestop_calibration(trouble_df, fig_dir)
    print(f"  All figures saved to: {fig_dir}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of Samjin factory production data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/E5SP_TOP_Tape_Production_Records_Jan-Apr.xlsx",
        help="Path to the factory Excel workbook.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/factory_eda",
        help="Directory to write output CSV files and summary text.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        default=False,
        help="Skip figure generation (useful when running headless or in CI).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading workbook: {input_path}")
    xl = pd.ExcelFile(input_path)

    # ---- Extract ----
    print("  Extracting production records ...")
    production_df = extract_production_records(xl)

    print("  Extracting machine trouble log ...")
    trouble_df = extract_machine_trouble(xl)

    print("  Extracting under-achievement reasons ...")
    reasons_df = extract_underachievement_reasons(xl)

    # ---- Save cleaned CSVs ----
    prod_csv    = out_dir / "production_daily.csv"
    trouble_csv = out_dir / "machine_trouble_daily.csv"
    reasons_csv = out_dir / "underachievement_log.csv"

    production_df.to_csv(prod_csv,    index=False)
    trouble_df.to_csv(trouble_csv,    index=False)
    reasons_df.to_csv(reasons_csv,    index=False)

    print(f"\n  Saved: {prod_csv}    ({len(production_df)} rows)")
    print(f"  Saved: {trouble_csv} ({len(trouble_df)} rows)")
    print(f"  Saved: {reasons_csv} ({len(reasons_df)} rows)")

    # ---- Analyse and print ----
    out_lines: list[str] = []

    intro = (
        f"\n{'=' * 80}\n"
        f"  ESS Robust Scheduling — Factory Data Exploratory Analysis\n"
        f"  Source: {input_path.name}\n"
        f"{'=' * 80}"
    )
    print(intro)
    out_lines.append(intro)

    analyse_production(production_df, out_lines)
    analyse_machine_trouble(trouble_df, out_lines)
    analyse_underachievement(reasons_df, out_lines)

    # ---- Save summary text ----
    summary_path = out_dir / "eda_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"\n  Saved summary: {summary_path}")

    # ---- Generate figures ----
    if not args.no_figures:
        generate_all_figures(production_df, trouble_df, reasons_df, out_dir)
    else:
        print("\n  Figures skipped (--no-figures flag set).")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
