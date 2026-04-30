# Data-informed parameter update notes

Updated for the Jan-Apr Samjin E5SP TOP Tape factory-data EDA.

## Main modelling decision

The model keeps the batch-level job abstraction:

- one model job is not one physical unit;
- one model job represents a batch-level production workload;
- station-level C/T is not directly available in the provided data.

The EDA C/T proxy is used only to interpret scale:
median implied C/T ≈ 2.34 min/unit, so the existing 150 min/job assumption
roughly corresponds to a batch of about 60-65 units.

## Parameters changed

1. MachineStopConfig
   - mean_uptime_between_stops: 68.57 -> 9.21 min
   - mean_stop_duration: 8.0 -> 0.55 min
   - stop_duration_cv: kept at 1.0

   Reason: the Automation Results data indicates frequent, short micro-stops.
   Median values are used to reduce sensitivity to outlier days.

2. Runtime-friendly experiment defaults
   - run_single_case default n_jobs: 50 -> 40
   - run_single_case default replications: 100 -> 20
   - Gurobi time limit: 900 sec -> 120 sec
   - grid default n_values: 20,30,40,50
   - grid default k_values: 0.5,1.0,1.5
   - grid default replications: 20

   Reason: 50+ jobs can be slow with the current MIP formulation. These defaults
   are intended for quick presentation-ready runs. Heavier cases can still be
   run by overriding command-line arguments.

## Parameters intentionally not changed

- Station A mean processing time: 90 min/job
- Station B mode processing time: 60 min/job
- Station B triangular range: 0.70 to 1.50
- Maintenance duration: 120 min

Reason: the company data is aggregated at line/shift/day level. It does not
provide station-level C/T, worker-level C/T, or true planned PM duration.
