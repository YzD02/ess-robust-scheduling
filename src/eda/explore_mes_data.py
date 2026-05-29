from __future__ import annotations
"""
explore_mes_data.py
===================

Exploratory data analysis for the MES Daily Production History of the
upstream Parts Injection Machine No.033 (ASSY COVER TOP, E5SP TOP line).

What this script does
---------------------
This is the upstream supply analysis for the ESS robust scheduling model.
The injection machine produces the ASSY COVER TOP parts that feed into the
TOP Tape assembly line.  If the injection machine under-produces, the TOP
Tape line cannot meet its daily target regardless of how well the scheduling
model is set up.

The script analyses:

1. Production Achievement
   - Daily achievement rate distribution (overall and monthly)
   - Zero-production days (full stoppages)
   - Below-plan frequency and severity

2. Mold Generation Effect
   - The dataset covers two mold generations (2nd Gen and 3rd Gen)
   - 2nd Gen shows dramatically worse performance (mean ach = 0.55 vs 0.88)
   - This is a critical confounding factor in the data

3. Defect and Quality Analysis
   - Daily defect rate distribution
   - Days with high defect spikes (>5%)
   - Total defective volume and good product rate

4. Upstream Supply Risk for the TOP Tape Line
   - Both the injection machine and TOP Tape line plan 1,200 units/day
   - On 52.7% of days the injection machine produced fewer than 1,200 units
   - Median shortfall on those days = 372 units
   - This quantifies how often upstream starvation could cause TOP Tape
     under-achievement independently of scheduling decisions

5. Cross-reference with TOP Tape EDA
   - Injection machine below-plan rate: 50.5% (vs TOP Tape 43.2%)
   - The injection machine is less reliable than the assembly line itself
   - This explains a large share of TOP Tape under-achievement that
     cannot be fixed by better scheduling alone

Design decisions
----------------
- Mold generation is separated throughout because 2nd Gen vs 3rd Gen
  produce fundamentally different achievement distributions.  Pooling
  them would give a misleading picture of the machine's baseline behaviour.
- Production Time is all zeros in the MES (not recorded), so no cycle
  time analysis is possible from this dataset.
- Zero-production days (4 days) are kept in the dataset for achievement
  rate stats but flagged explicitly.
- The supply risk section uses 1,200 as the threshold because both lines
  share the same daily target — this makes the cross-line comparison direct.

How to run
----------
    python -m src.eda.explore_mes_data

Or with explicit paths:

    python -m src.eda.explore_mes_data \
        --input data/MES_Daily_Production_History_E5SP_TOP.xlsx \
        --out-dir results/mes_eda
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
SHIFT_MINUTES   = 480.0
LINE_PLAN_QTY   = 1200          # shared daily target for both lines
TOP_TAPE_BELOW_PLAN_RATE = 0.432   # from TOP Tape EDA for cross-reference
TOP_TAPE_MEAN_ACH        = 0.965


# ── Helpers ───────────────────────────────────────────────────────────────────
def _summary_stats(s: pd.Series, label: str) -> str:
    s = s.dropna()
    return (
        f"  {label:<55}"
        f"n={len(s):4d}  "
        f"mean={s.mean():8.3f}  "
        f"median={s.median():8.3f}  "
        f"std={s.std():7.3f}  "
        f"min={s.min():8.3f}  "
        f"p10={s.quantile(0.10):8.3f}  "
        f"p90={s.quantile(0.90):8.3f}  "
        f"max={s.max():8.3f}"
    )


def _print(lines: list, msg: str = "") -> None:
    print(msg)
    lines.append(msg)


# ── ETL ───────────────────────────────────────────────────────────────────────
def load_mes_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df['Production Date'] = pd.to_datetime(df['Production Date'], errors='coerce', format='mixed')
    df['month'] = df['Production Date'].dt.month
    df['month_name'] = df['Production Date'].dt.strftime('%b')
    df['ach'] = df['Achievement Rate (%)'] / 100
    df['is_3rd_gen'] = df['Mold Generation'] == '3rd Gen'
    df['below_plan'] = df['ach'] < 1.0
    df['zero_production'] = df['Actual Output'] == 0
    # Supply risk: days where injection output < TOP Tape daily target
    df['starves_top_tape'] = df['Actual Output'] < LINE_PLAN_QTY
    df['supply_shortfall'] = (LINE_PLAN_QTY - df['Actual Output']).clip(lower=0)
    return df


# ── Analysis sections ─────────────────────────────────────────────────────────
def analyse_achievement(df: pd.DataFrame, out_lines: list) -> None:
    _print(out_lines, "\n" + "="*70)
    _print(out_lines, "  1. PRODUCTION ACHIEVEMENT ANALYSIS")
    _print(out_lines, "="*70)

    _print(out_lines, "\n--- Overall (all 93 operating days) ---")
    _print(out_lines, _summary_stats(df['ach'], "Achievement rate"))
    _print(out_lines, _summary_stats(df['Actual Output'].astype(float), "Actual output (units/day)"))
    _print(out_lines, _summary_stats(df['Planned Qty'].astype(float), "Planned qty (units/day)"))

    below = df['below_plan']
    _print(out_lines, f"\n  Days below plan (ach < 1.0): {below.sum()} / {len(df)} = {below.mean()*100:.1f}%")

    zeros = df['zero_production']
    _print(out_lines, f"  Zero-production days (complete stoppage): {zeros.sum()}")
    _print(out_lines, "  Dates: " + ", ".join(df[zeros]['Production Date'].dt.strftime('%Y-%m-%d').tolist()))

    _print(out_lines, "\n--- Monthly breakdown ---")
    for m, grp in df.groupby('month'):
        name = grp['month_name'].iloc[0]
        _print(out_lines, f"\n  {name} (n={len(grp)}):")
        _print(out_lines, _summary_stats(grp['ach'], "  Achievement rate"))
        _print(out_lines, f"  Below plan: {grp['below_plan'].sum()} / {len(grp)} = {grp['below_plan'].mean()*100:.1f}%")

    _print(out_lines, "\n--- Mold generation effect ---")
    for gen, grp in df.groupby('Mold Generation'):
        _print(out_lines, f"\n  {gen} (n={len(grp)}):")
        _print(out_lines, _summary_stats(grp['ach'], "  Achievement rate"))
        _print(out_lines, f"  Below plan: {grp['below_plan'].sum()} / {len(grp)} = {grp['below_plan'].mean()*100:.1f}%")
        _print(out_lines, f"  Mean actual output: {grp['Actual Output'].mean():.0f} units/day")

    _print(out_lines,
        "\n  NOTE: 2nd Gen mold (n=11, Jan) shows dramatically lower achievement "
        "(mean=0.55, median=0.39)\n"
        "  vs 3rd Gen (mean=0.88, median=1.01). The 2nd Gen period should be\n"
        "  treated as a transition/burn-in phase, not representative of steady-state."
    )


def analyse_quality(df: pd.DataFrame, out_lines: list) -> None:
    _print(out_lines, "\n" + "="*70)
    _print(out_lines, "  2. QUALITY AND DEFECT ANALYSIS")
    _print(out_lines, "="*70)

    _print(out_lines, "\n--- Overall defect statistics ---")
    _print(out_lines, _summary_stats(df['Defect Rate (%)'], "Defect rate (%)"))
    _print(out_lines, _summary_stats(df['Good Product Rate (%)'], "Good product rate (%)"))
    _print(out_lines, _summary_stats(df['Defective Qty'].astype(float), "Defective qty (units/day)"))

    total_actual = df['Actual Output'].sum()
    total_defect = df['Defective Qty'].sum()
    _print(out_lines, f"\n  Total defective over period: {total_defect} / {total_actual} = "
                      f"{total_defect/total_actual*100:.2f}%")

    high_defect = df[df['Defect Rate (%)'] > 5]
    _print(out_lines, f"  Days with defect rate > 5%: {len(high_defect)} / {len(df)}")
    if len(high_defect) > 0:
        _print(out_lines, "  Dates and rates:")
        for _, row in high_defect.iterrows():
            _print(out_lines, f"    {row['Production Date'].strftime('%Y-%m-%d')}: "
                              f"{row['Defect Rate (%)']:.1f}% ({int(row['Defective Qty'])} defective)")

    _print(out_lines, "\n--- Mold generation effect on quality ---")
    for gen, grp in df.groupby('Mold Generation'):
        _print(out_lines, f"\n  {gen}: mean defect rate = {grp['Defect Rate (%)'].mean():.2f}%, "
                          f"median = {grp['Defect Rate (%)'].median():.2f}%")


def analyse_supply_risk(df: pd.DataFrame, out_lines: list) -> None:
    _print(out_lines, "\n" + "="*70)
    _print(out_lines, "  3. UPSTREAM SUPPLY RISK FOR THE TOP TAPE LINE")
    _print(out_lines, "="*70)

    _print(out_lines,
        f"\n  Both the injection machine and the TOP Tape line target {LINE_PLAN_QTY} units/day.\n"
        f"  When the injection machine produces fewer than {LINE_PLAN_QTY} units, the TOP Tape\n"
        f"  line may not have enough parts to meet its own daily plan."
    )

    starve = df['starves_top_tape']
    shortfall = df[starve]['supply_shortfall']

    _print(out_lines, f"\n  Days where injection output < {LINE_PLAN_QTY} (supply risk): "
                      f"{starve.sum()} / {len(df)} = {starve.mean()*100:.1f}%")
    _print(out_lines, _summary_stats(shortfall, "  Supply shortfall on those days (units)"))

    _print(out_lines, f"\n  Median daily shortfall when at risk: {shortfall.median():.0f} units")
    _print(out_lines, f"  = {shortfall.median()/LINE_PLAN_QTY*100:.1f}% of the daily plan")
    _print(out_lines, f"  = {shortfall.median()/192:.1f} model jobs worth of missing parts")

    _print(out_lines, "\n--- Cross-reference with TOP Tape EDA ---")
    _print(out_lines, f"\n  Injection machine below-plan rate:  {df['below_plan'].mean()*100:.1f}%")
    _print(out_lines, f"  TOP Tape line below-plan rate:      {TOP_TAPE_BELOW_PLAN_RATE*100:.1f}%")
    _print(out_lines, f"\n  The injection machine is LESS reliable than the TOP Tape line itself.")
    _print(out_lines,
        f"\n  3rd Gen only (steady-state, n=82):\n"
        f"  Injection below-plan rate: "
        f"{df[df['is_3rd_gen']]['below_plan'].mean()*100:.1f}% (vs 50.5% overall)\n"
        f"  This is still higher than TOP Tape's 43.2%, confirming the machine\n"
        f"  is a genuine upstream bottleneck even under normal operating conditions."
    )

    _print(out_lines,
        "\n  Modelling implication:\n"
        "  The current scheduling model captures PM windows, machine micro-stops,\n"
        "  and processing variability on the TOP Tape line itself. It does NOT yet\n"
        "  model upstream parts starvation. On 52.7% of days, the injection machine\n"
        "  produced fewer than 1,200 parts — meaning the TOP Tape line would face\n"
        "  a material shortage independent of any scheduling decision. Adding an\n"
        "  upstream disruption layer based on this machine's achievement distribution\n"
        "  would improve the model's realism significantly."
    )


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_achievement(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fig 1 — Injection Machine Achievement Rate (Jan–Apr 2026)", fontsize=13, fontweight='bold')

    # Panel A: overall distribution
    ax = axes[0]
    ax.hist(df['ach'], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Plan = 1.0')
    ax.axvline(df['ach'].median(), color='orange', linestyle='--', linewidth=1.5,
               label=f"Median = {df['ach'].median():.2f}")
    below_pct = (df['ach'] < 1.0).mean() * 100
    ax.text(0.05, 0.95, f"{below_pct:.0f}% below plan\n({(df['ach']<1.0).sum()}/{len(df)} days)",
            transform=ax.transAxes, va='top', fontsize=10, color='darkred',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlabel("Achievement Rate (actual / plan)")
    ax.set_ylabel("Number of Days")
    ax.set_title("Overall Distribution (n=93)")
    ax.legend(fontsize=9)

    # Panel B: monthly boxplot
    ax = axes[1]
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    data = [df[df['month_name'] == m]['ach'].values for m in months]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#4393c3', '#92c5de', '#f4a582', '#d6604d']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for i, (m, d) in enumerate(zip(months, data)):
        if len(d) > 0:
            ax.text(i+1, np.median(d), f"{np.median(d):.2f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Plan = 1.0')
    ax.set_xticklabels(months)
    ax.set_ylabel("Achievement Rate")
    ax.set_title("Monthly Breakdown")
    ax.legend(fontsize=9)

    # Panel C: mold generation comparison
    ax = axes[2]
    gen2 = df[~df['is_3rd_gen']]['ach']
    gen3 = df[df['is_3rd_gen']]['ach']
    ax.hist(gen2, bins=12, alpha=0.7, color='tomato', edgecolor='white',
            label=f"2nd Gen (n={len(gen2)}, mean={gen2.mean():.2f})")
    ax.hist(gen3, bins=12, alpha=0.7, color='steelblue', edgecolor='white',
            label=f"3rd Gen (n={len(gen3)}, mean={gen3.mean():.2f})")
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Achievement Rate")
    ax.set_ylabel("Number of Days")
    ax.set_title("Mold Generation Comparison")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig1_injection_achievement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig1_injection_achievement.png")


def plot_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle("Fig 2 — Daily Production Timeline (Jan–Apr 2026)", fontsize=13, fontweight='bold')

    # Panel A: actual vs planned
    ax = axes[0]
    colors = ['tomato' if v < LINE_PLAN_QTY else 'steelblue' for v in df['Actual Output']]
    ax.bar(df['Production Date'], df['Actual Output'], color=colors, alpha=0.8, width=0.8)
    ax.axhline(LINE_PLAN_QTY, color='black', linestyle='--', linewidth=1.5, label=f'Plan = {LINE_PLAN_QTY}')
    ax.set_ylabel("Units Produced")
    ax.set_title("Daily Actual Output  (red = below plan)")
    ax.legend(fontsize=9)

    # Add month separators
    for month in [2, 3, 4]:
        d = pd.Timestamp(f'2026-{month:02d}-01')
        ax.axvline(d, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(d, ax.get_ylim()[1]*0.95, ['Feb','Mar','Apr'][month-2],
                fontsize=9, color='gray', ha='left')

    # Panel B: defect rate
    ax = axes[1]
    ax.bar(df['Production Date'], df['Defect Rate (%)'], color='darkorange', alpha=0.8, width=0.8)
    ax.axhline(5.0, color='red', linestyle='--', linewidth=1.5, label='5% threshold')
    ax.set_ylabel("Defect Rate (%)")
    ax.set_title("Daily Defect Rate")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig2_injection_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig2_injection_timeline.png")


def plot_supply_risk(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig 3 — Upstream Supply Risk for TOP Tape Line", fontsize=13, fontweight='bold')

    # Panel A: supply shortfall distribution
    ax = axes[0]
    shortfall = df['supply_shortfall']
    at_risk = shortfall[shortfall > 0]
    ax.hist(at_risk, bins=15, color='tomato', edgecolor='white', alpha=0.85)
    ax.axvline(at_risk.median(), color='black', linestyle='--', linewidth=1.5,
               label=f"Median shortfall = {at_risk.median():.0f} units")
    ax.set_xlabel("Units Short of 1,200 Target")
    ax.set_ylabel("Number of Days")
    ax.set_title(f"Supply Shortfall Distribution\n({len(at_risk)}/{len(df)} days = {len(at_risk)/len(df)*100:.0f}% at risk)")
    ax.legend(fontsize=9)

    # Panel B: injection machine vs TOP Tape reference stats (real values from EDA)
    # TOP Tape EDA (Jan-Apr, 81 days): mean=0.965, median=1.025, std=0.182, min=0.45, max=1.35
    ax = axes[1]
    inj_3rd = df[df['is_3rd_gen']]['ach']

    bp = ax.boxplot(
        [inj_3rd.values],
        positions=[1],
        patch_artist=True,
        boxprops=dict(facecolor='#f4a582'),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        widths=0.5,
    )
    ax.text(1, inj_3rd.median() + 0.03,
            f"median={inj_3rd.median():.2f}",
            ha='center', fontsize=9, fontweight='bold')
    ax.text(1, inj_3rd.quantile(0.25) - 0.08,
            f"below plan: {(inj_3rd<1.0).mean()*100:.0f}%",
            ha='center', fontsize=8, color='darkred')

    # TOP Tape reference lines (real stats from explore_factory_data.py EDA)
    ax.axhline(1.0,   color='red',       linestyle='--', linewidth=1.5, label='Plan = 1.0')
    ax.axhline(1.025, color='steelblue', linestyle='-',  linewidth=1.5,
               label=f'TOP Tape median ach = 1.025')
    ax.axhline(0.965, color='steelblue', linestyle='-.', linewidth=1.2,
               label=f'TOP Tape mean ach = 0.965')

    ax.set_xticks([1])
    ax.set_xticklabels(['Injection Machine (3rd Gen, n=82)'])
    ax.set_ylabel("Achievement Rate")
    ax.set_title("Injection Machine vs TOP Tape Reference (TOP Tape: median=1.025, below plan=43.2%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig3_supply_risk.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig3_supply_risk.png")


def plot_quality(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig 4 — Quality Analysis (Defect Rate)", fontsize=13, fontweight='bold')

    # Panel A: defect rate distribution
    ax = axes[0]
    ax.hist(df['Defect Rate (%)'], bins=20, color='darkorange', edgecolor='white', alpha=0.85)
    ax.axvline(df['Defect Rate (%)'].median(), color='black', linestyle='--', linewidth=1.5,
               label=f"Median = {df['Defect Rate (%)'].median():.2f}%")
    ax.axvline(5.0, color='red', linestyle='--', linewidth=1.5, label='5% threshold')
    ax.set_xlabel("Defect Rate (%)")
    ax.set_ylabel("Number of Days")
    ax.set_title(f"Overall Defect Rate Distribution\n(mean={df['Defect Rate (%)'].mean():.2f}%, total defective={df['Defective Qty'].sum():,})")
    ax.legend(fontsize=9)

    # Panel B: mold gen quality comparison
    ax = axes[1]
    gen2 = df[~df['is_3rd_gen']]['Defect Rate (%)']
    gen3 = df[df['is_3rd_gen']]['Defect Rate (%)']
    ax.boxplot([gen2.values, gen3.values],
               labels=[f'2nd Gen\n(n={len(gen2)})', f'3rd Gen\n(n={len(gen3)})'],
               patch_artist=True,
               boxprops=dict(facecolor='lightyellow'),
               medianprops=dict(color='black', linewidth=2))
    ax.set_ylabel("Defect Rate (%)")
    ax.set_title("Defect Rate by Mold Generation")
    for i, (gen, d) in enumerate(zip(['2nd', '3rd'], [gen2, gen3]), 1):
        ax.text(i, d.median()+0.2, f"median={d.median():.2f}%", ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig4_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig4_quality_analysis.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EDA for MES injection machine data.")
    parser.add_argument('--input', type=str,
                        default='data/MES_Daily_Production_History_E5SP_TOP.xlsx',
                        help='Path to the MES Excel file.')
    parser.add_argument('--out-dir', type=str, default='results/mes_eda',
                        help='Output directory for figures and summary.')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation (text output only).')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    df = load_mes_data(args.input)
    print(f"Loaded {len(df)} rows, {df['Production Date'].min().date()} to {df['Production Date'].max().date()}")

    out_lines = [
        "MES Daily Production History EDA — Parts Injection Machine No.033",
        f"Item: ASSY COVER TOP  |  Period: Jan–Apr 2026  |  n={len(df)} operating days",
        "="*70,
    ]

    analyse_achievement(df, out_lines)
    analyse_quality(df, out_lines)
    analyse_supply_risk(df, out_lines)

    # Save summary text
    summary_path = out_dir / 'mes_eda_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f"\nSaved summary: {summary_path}")

    if not args.no_figures:
        print("\nGenerating figures...")
        plot_achievement(df, fig_dir)
        plot_timeline(df, fig_dir)
        plot_supply_risk(df, fig_dir)
        plot_quality(df, fig_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
