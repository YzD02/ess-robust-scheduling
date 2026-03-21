# ESS Robust Scheduling

A robust scheduling and simulation framework for ESS assembly lines under human-machine uncertainty.

## Overview

This project studies production scheduling for an ESS assembly line over a 4-week planning horizon. The system is modeled as a robust day-level scheduling problem, where each 8-hour shift is treated as a time bin. The objective is to assign jobs to working days while accounting for uncertainty from both machine and human operations.

The project combines:

- a robust optimization model solved by Gurobi
- a simulation-based validation layer using SimPy
- stress testing and grid search experiments to identify system breakdown regions

---

## System Description

The scope is limited to **Stage 2** of the ESS production process, which contains two sequential stations:

- **Station A**: Automated Busbar Assembly
- **Station B**: Manual Harness Alignment and Riveting

The main uncertainty sources are:

- **machine-side variability** at Station A due to micro-stops
- **human-side variability** at Station B due to worker performance differences

---

## Research Goal

This project aims to answer the following questions:

1. How should jobs be assigned to daily bins under uncertain processing times?
2. Under what workload and variability levels does the system begin to break down?
3. Can the schedule satisfy a target reliability level, such as clearing the full horizon in at least 95% of simulation runs?
4. Can Gurobi solve the planning problem within an operationally acceptable time limit?

---

## Mathematical Modeling Idea

The baseline planning model is a **robust bin-packing formulation**.

- each job is treated as an item
- each working day is treated as a bin
- the standard daily capacity is 480 minutes (8 Hours)
- overtime is allowed with a fixed and variable cost

For job `j`, the robust processing time is defined as:

```text
p'_j = (mu_A_j + mu_B_j) + k * sqrt(sigma_A_j^2 + sigma_B_j^2)
