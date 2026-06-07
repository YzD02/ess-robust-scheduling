#!/bin/bash
#SBATCH --account=def-cglee
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ess_k0_n125
#SBATCH --output=logs/slurm-%j.log

module purge
module load StdEnv/2023 python/3.11
module load gurobi/12.0.0

source $SCRATCH/GUROBI_ENV/bin/activate

python -m src.experiments.run_grid_search \
    --n-values 125 \
    --k-values 0.0 \
    --replications 100 \
    --maintenance "1:4,5;2:4,5;3:4,5;4:4,5" \
    --unscheduled skip \
    --gurobi-time-limit-sec 7200 \
    --out results/grid_search/grid_search_results_k0_n125.csv
