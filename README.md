# ESS Robust Scheduling

This repository contains a research prototype for robust production scheduling of an ESS assembly line under human-machine uncertainty.

## System scope

Stage 2 of the ESS line contains two sequential stations:

- **Station A**: Automated Busbar Assembly
- **Station B**: Manual Harness Alignment and Riveting

The planning model treats each working day as one time bin with standard capacity 480 minutes.

## Modeling philosophy

The project has two layers:

1. **Gurobi planning layer**
   - builds a robust day-level schedule
   - assigns jobs to daily bins
   - minimizes overtime cost
2. **Simulation validation layer**
   - executes the planned schedule under stochastic disruptions
   - propagates unfinished jobs across days
   - estimates breakdown probability and 95% service reliability

## Repository structure

```text
src/
├── models/
│   ├── robust_processing.py
│   └── gurobi_baseline.py
├── simulation/
│   └── simpy_cross_day_breakdown.py
├── experiments/
│   └── run_single_case.py
├── visualization/
│   └── plot_heatmaps.py
└── utils/
```

## Run one experiment

From the repository root:

```bash
python -m src.experiments.run_single_case
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run one experiment

From the repository root:

```bash
python -m src.experiments.run_grid_search
```


Note: Gurobi requires a valid license.
