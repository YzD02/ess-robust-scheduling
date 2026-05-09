#!/bin/bash
#SBATCH --account=def-cglee
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=ess_grid_search_two_shift
#SBATCH --output=logs/slurm-%j.log

module purge
module load StdEnv/2023 python/3.11
module load gurobi/12.0.0

source $SCRATCH/GUROBI_ENV/bin/activate

python -m src.experiments.run_grid_search \
    --n-values 75,100,125,150 \
    --k-values 0.5,1.0,1.5 \
    --replications 100 \
    --maintenance "1:4,5;2:4,5;3:4,5;4:4,5" \
    --unscheduled skip \
    --gurobi-time-limit-sec 7200 \
    --out results/grid_search/grid_search_results_two_shift.csv

python -m src.visualization.plot_heatmaps \
    --csv results/grid_search/grid_search_results_two_shift.csv \
    --out-dir results/figures

python -m src.visualization.plot_phase_diagram \
    --csv results/grid_search/grid_search_results_two_shift.csv \
    --out-dir results/figures
