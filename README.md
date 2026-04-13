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
| **B** | Manual Harness Alignment & Riveting | Worker speed variability (modelled as Gaussian) |

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
the `main()` function.

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

Both experiment scripts support the same `--maintenance` flag and share the
same default schedule defined in `src/utils/maintenance.py`.

**Default behaviour (no flag needed)**

Both scripts automatically use the fixed schedule in `DEFAULT_MAINTENANCE_SCHEDULE`
inside `src/utils/maintenance.py`.  Edit that file to change the project-wide
default — one edit applies to both scripts at once.

**Override from the command line**

Pass a custom schedule as `day:start_minute:end_minute` entries separated by
commas.  Minutes are measured from the start of the shift (minute 0 = 08:00
if the shift starts at 08:00):

```bash
# Week 1 Wednesday 12:00-14:00  and  Week 2 Thursday 08:00-10:00
python -m src.experiments.run_single_case --n-jobs 60 --maintenance "3:240:360,9:0:120"
python -m src.experiments.run_grid_search --maintenance "3:240:360,9:0:120"
```

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

**Revert to random windows**

Pass `random` to bypass the default and let each replication generate its
own random maintenance windows from its seed:

```bash
python -m src.experiments.run_single_case --maintenance random
python -m src.experiments.run_grid_search --maintenance random
```

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
- **Station B** (manual):  variability comes from human factors, which are
  well approximated by a normal distribution centred on the mean job time.

`sigma_A` is therefore used only in the planning layer (to size the buffer)
and is not passed into the simulation, where Station A randomness is already
represented by the stop process.

### Why keyword-only arguments?

Core functions use Python's `*` convention so all arguments must be named
explicitly.  This prevents hard-to-find bugs from passing arguments in the
wrong order when a function has many numeric parameters.

---

## Updating baseline values

All processing-time parameters in this project are currently assumption-based.
Once real production data is available:

1. Open `src/experiments/run_single_case.py`
2. Update the `JobGenerationConfig` block with calibrated values
3. Re-run the grid search to see how the results change
