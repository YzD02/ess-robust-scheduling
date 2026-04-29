# ESS Robust Scheduling

Research prototype for robust production scheduling of an ESS (Energy Storage
System) assembly line under real-world uncertainty.

---

## What this project does

Planning a production schedule on paper is straightforward.  The harder
question is: *how often does that plan actually survive execution?*

This project answers that question through two layers:

1. **Planning layer (Gurobi)**
   Builds a day-level schedule that assigns each job to one of 20 working days.
   Rather than using average job times, the model inflates each job's planned
   duration by a safety buffer (`k × sigma`) so the schedule has built-in
   protection against overruns.

2. **Simulation layer (Monte Carlo)**
   Takes that plan and runs it through hundreds of randomised scenarios —
   varying job times, injecting machine micro-stops, and scheduling maintenance
   windows — to estimate how reliably the line can clear all jobs within the
   planning horizon.

The result is a map of the system's behaviour: at what job load does the line
stay stable?  When does it start needing weekend recovery?  When does it
become overloaded even with weekends?

---

## Assembly line scope

The model covers Stage 2 of the ESS line — two sequential stations:

| Station | Operation | Uncertainty source |
|---|---|---|
| **A** | Automated Busbar Assembly | Machine micro-stops (random frequency and duration) |
| **B** | Manual Harness Alignment & Riveting | Worker speed variability (triangular distribution: min, mode, max) |

Each working day has **480 minutes** of regular capacity (one 8-hour shift).
No same-day overtime is allowed.  Unfinished jobs carry over to the next day
as backlog.  Up to **8 weekend recovery days** are available after the 20-day
weekday horizon.

---

## Repository structure

```
src/
├── models/
│   ├── job_generation.py        generates per-job mu and sigma values
│   ├── robust_processing.py     computes buffered planning times (p = mu + k·sigma)
│   └── gurobi_baseline.py       Gurobi day-assignment MIP model
│
├── simulation/
│   └── simulation_engine.py     stochastic cross-day execution simulator
│
├── experiments/
│   ├── run_single_case.py       one end-to-end run with fixed parameters
│   └── run_grid_search.py       sweeps job counts × k values; saves results CSV
│
├── visualization/
│   ├── plot_gantt.py            Gantt charts from simulation event log
│   ├── plot_heatmaps.py         heatmap dashboard from grid-search CSV
│   └── plot_phase_diagram.py    phase diagram (stable / recoverable / overloaded)
│
└── utils/
    ├── plot_utils.py            shared helpers used by all visualization scripts
    └── maintenance.py           maintenance schedule config and CLI parser
```

---

## Setup

### Requirements

- Python 3.11 or later
- A valid **Gurobi licence** (free academic licences available at gurobi.com)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to run

### Step 1 — Verify your setup with a single case

This runs one complete experiment (50 jobs, k = 1.0) and saves three output
files.  Use this first to confirm everything is installed correctly.

```bash
python -m src.experiments.run_single_case
```

Expected output files in `results/simulation_outputs/`:

| File | Contents |
|---|---|
| `gantt_events_single_run.csv` | Per-job start/end times at each station — input for Gantt chart |
| `single_run_day_summary.csv` | Per-day statistics: backlog count, utilisation, overflow flag |
| `single_run_mc_summary.csv` | Aggregated Monte Carlo statistics (100 replications) |

To change the parameters (number of jobs, k value, random seed), open
`src/experiments/run_single_case.py` and edit the variables at the top of
the `main()` function.  Or pass them directly on the command line:

```bash
python -m src.experiments.run_single_case --n-jobs 60 --k 1.0 --seed 42
```

**Available CLI flags for `run_single_case`:**

| Flag | Default | Description |
|---|---|---|
| `--n-jobs` | 50 | Number of jobs to schedule |
| `--k` | 1.0 | Robustness buffer factor (0 = no buffer, 2 = very conservative) |
| `--mu-scale` | 1.0 | Multiplier for all mean job times |
| `--sigma-scale` | 1.0 | Multiplier for all standard deviations |
| `--seed` | 42 | Random seed |
| `--replications` | 100 | Number of Monte Carlo replications |
| `--maintenance` | — | Maintenance schedule (see section below) |
| `--unscheduled` | — | Policy for unlisted weeks: `skip` or `random` |

---

### Step 2 — Run the full grid search

Tests all combinations of job count and k value.  Results are written to
a CSV file that you can then visualise.

```bash
python -m src.experiments.run_grid_search
```

**Quick test run** (finishes in a few minutes):

```bash
python -m src.experiments.run_grid_search --n-values 20,40 --k-values 0.5,1.0 --replications 20
```

**Full parameter grid** (default, takes 10–30 minutes):

```bash
python -m src.experiments.run_grid_search --n-values 20,40,60,80,100 --k-values 0.5,1.0,1.5,2.0
```

The script writes results incrementally — if it is interrupted, the rows
already completed are saved.  Re-running overwrites the file, so back it
up first if you want to keep the previous results.

---

### Controlling the maintenance schedule

Both experiment scripts support the same `--maintenance` and `--unscheduled`
flags, and share the same defaults defined in `src/utils/maintenance.py`.
There are three modes:

**Mode 1 — Fixed schedule** (exact day and time predetermined)

```bash
# Week 1 Wednesday 12:00-14:00  and  Week 2 Thursday 08:00-10:00
python -m src.experiments.run_single_case --maintenance "3:240:360,9:0:120"
python -m src.experiments.run_grid_search --maintenance "3:240:360,9:0:120"
```

The same windows are used in every Monte Carlo replication.

**Mode 2 — Constrained-random schedule** (day constrained per week, time random)

Specify which days each week's maintenance is allowed to fall on.  The exact
day within those candidates and the start time are drawn randomly each
replication.  Use `--unscheduled` to control what happens to weeks not listed:

```bash
# Week 1 only Thu/Fri, Week 2 only Mon-Wed; unscheduled weeks get no maintenance
python -m src.experiments.run_single_case --maintenance "1:4,5;2:1,2,3" --unscheduled skip

# Same candidates, but unscheduled weeks fall back to a random day
python -m src.experiments.run_single_case --maintenance "1:4,5;2:1,2,3" --unscheduled random
```

If `--unscheduled` is omitted, `DEFAULT_UNSCHEDULED_WEEKS_POLICY` in
`src/utils/maintenance.py` is used (currently `'skip'`).

To use the default candidate days without typing them out:

```bash
python -m src.experiments.run_single_case --maintenance candidates
python -m src.experiments.run_single_case --maintenance candidates --unscheduled random
```

**Mode 3 — Fully random** (original behaviour)

```bash
python -m src.experiments.run_single_case --maintenance random
```

**Changing the project-wide defaults**

Edit `src/utils/maintenance.py` — one change applies to both scripts:

| Constant | Controls |
|---|---|
| `DEFAULT_MAINTENANCE_SCHEDULE` | Fixed schedule (set to `None` to disable) |
| `DEFAULT_MAINTENANCE_CANDIDATE_DAYS` | Constrained-random candidate days |
| `DEFAULT_UNSCHEDULED_WEEKS_POLICY` | Fallback for unlisted weeks (`'skip'` or `'random'`) |

**Time conversion reference** (08:00 shift start):

| Clock time | Minutes from shift start |
|---|---|
| 08:00 – 10:00 | 0 – 120 |
| 10:00 – 12:00 | 120 – 240 |
| 12:00 – 14:00 | 240 – 360 |
| 14:00 – 16:00 | 360 – 480 |

**Day number reference** (20-day horizon):

| | Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|---|
| Week 1 | 1 | 2 | 3 | 4 | 5 |
| Week 2 | 6 | 7 | 8 | 9 | 10 |
| Week 3 | 11 | 12 | 13 | 14 | 15 |
| Week 4 | 16 | 17 | 18 | 19 | 20 |

---

### Step 3 — Generate visualisations

All three commands read from `results/grid_search/` and write PNG figures
to `results/figures/`.

```bash
# Gantt chart (shows one simulation run day by day)
python -m src.visualization.plot_gantt

# Heatmap dashboard (shows metrics across all parameter combinations)
python -m src.visualization.plot_heatmaps

# Phase diagram (classifies each case as stable / recoverable / overloaded)
python -m src.visualization.plot_phase_diagram
```

---

## Understanding the outputs

### Phase diagram — the big picture

Each cell in the phase diagram represents one (n_jobs, k) combination:

| Colour | Meaning |
|---|---|
| 🟢 Green | All jobs finish within regular weekdays in ≥ 95 % of runs |
| 🟠 Orange | Weekdays alone are not enough, but weekend recovery brings success to ≥ 95 % |
| 🔴 Red-orange | Even with weekends the schedule fails > 5 % of the time |
| 🔴 Red | Gurobi could not build a feasible schedule |

### Key metrics in the heatmap

| Metric | What it means |
|---|---|
| `prob_cleared_within_extended_horizon` | Fraction of simulation runs where all jobs completed on time — higher is better; ≥ 0.95 is the target |
| `avg_n_weekend_days_used` | Average number of weekend days consumed across all replications |
| `avg_final_completion_day` | Average day on which the last job was finished (day 20 = last weekday) |
| `avg_total_weekend_cost` | Average cost of weekend recovery (fixed + variable components) |

---

## Key design decisions

### Why k (the robustness factor)?

If you schedule jobs using their average times, roughly half of all days will
overrun, because real job times vary.  The buffer `k × sigma` gives each job
extra planned time so the schedule can absorb moderate variability without
creating backlog.  Larger k = safer but fewer jobs fit per day.

### Why are Station A and Station B modelled differently?

- **Station A** (automated):  variability comes from random machine micro-stops.
  The simulation models these explicitly — inter-stop times are exponentially
  distributed, stop durations are gamma-distributed.
- **Station B** (manual):  variability comes from human factors.  The actual
  processing time is drawn from a **triangular distribution** parameterised by
  three values per job:

  | Parameter | Meaning | Default |
  |---|---|---|
  | `mode` (mu_B) | most likely processing time | ~60 min |
  | `low_B` | best-case time = mode × `low_B_fraction` | mode × 0.70 |
  | `high_B` | worst-case time = mode × `high_B_fraction` | mode × 1.50 |

  The triangular distribution is preferred over Gaussian because it is bounded
  (no negative times), directly interpretable from factory observations (record
  the fastest, slowest, and most common job times), and naturally supports
  asymmetric variability.

  Once real factory data is available, calibrate by setting `low_B_fraction`,
  `high_B_fraction`, and `mu_B_mean` in the `JobGenerationConfig` block inside
  `run_single_case.py` and `run_grid_search.py`.

`sigma_A` is used only in the planning layer (to size the k-buffer) and is not
passed into the simulation, where Station A randomness is represented by the
machine-stop process.  `sigma_B` is derived analytically from the triangular
parameters and is also used only in the planning layer.

### Why is the maintenance window fixed across all Monte Carlo replications?

When using constrained-random maintenance mode, the maintenance windows are
drawn **once before the simulation loop** using the base seed, and the same
windows are reused in all 100 replications.  This ensures that the only source
of variability between replications is job processing times and machine stops —
not when maintenance happens.  This makes results across different parameter
combinations (e.g. k = 0.5 vs k = 1.0) directly comparable.

If you want each replication to draw its own random maintenance windows, use
`--maintenance random`.

### Why keyword-only arguments?

Core functions use Python's `*` convention so all arguments must be named
explicitly.  This prevents hard-to-find bugs from passing arguments in the
wrong order when a function has many numeric parameters.

---

## Updating baseline values

All processing-time parameters in this project are currently assumption-based.
Once real production data is available:

**Station B (triangular distribution)** — open `src/experiments/run_single_case.py`
and update the `JobGenerationConfig` block:

```python
gen_cfg = JobGenerationConfig(
    mu_B_mean=60.0,        # ← update with observed average job time
    mu_B_std=5.0,          # ← update with observed job-to-job spread
    low_B_fraction=0.70,   # ← fastest observed time / average mode
    high_B_fraction=1.50,  # ← slowest observed time / average mode
)
```

**Station A (micro-stop model)** — open `src/experiments/run_single_case.py`
and update the `MachineStopConfig` block:

```python
stop_cfg = MachineStopConfig(
    mean_uptime_between_stops=68.57,  # ← avg running minutes between stops
    mean_stop_duration=8.0,           # ← avg stop duration in minutes
    stop_duration_cv=1.0,             # ← coefficient of variation for stop duration
)
```

After updating, re-run the grid search to see how results change:

```bash
python -m src.experiments.run_grid_search
```
