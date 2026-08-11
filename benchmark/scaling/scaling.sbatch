#!/bin/bash -l
#SBATCH --job-name=wtm-scaling
#SBATCH --partition=msilong          # single-node; add --exclusive for a clean final measurement
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # >= the max rank count in RANKS below
#SBATCH --mem=32gb
#SBATCH --time=1:30:00
#SBATCH --output=wtm-scaling-%j.out
#
# Unattended MPI strong-scaling sweep of the CURRENT build. Submit it and
# disconnect; results land in this job's .out (the table) and in
# benchmark/scaling/results.csv (persistent). Reconnect any time to collect.
#
#     cd ~/models/WTM/benchmark/scaling
#     sbatch scaling.sbatch                              # 2000^2 grid, ranks 32..1, 2 reps
#     GRID=4000 RANKS="32 16 8 4 2 1" REPS=3 sbatch scaling.sbatch
#
# For the publication-grade run, uncomment --exclusive above (whole node, no
# co-tenant competing for memory bandwidth) and expect a longer queue wait.
set -euo pipefail

WTM_ROOT="${WTM_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/models/WTM}}"
GRID="${GRID:-2000}"
RANKS="${RANKS:-32 16 8 4 2 1}"
REPS="${REPS:-2}"

# Modules (-> mpiexec) + wtmtest conda env (-> rasterio for the .tif I/O).
# shellcheck source=../../msi_env.sh
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1              # pure MPI: one rank per core, no OpenMP threads

cd "$WTM_ROOT/benchmark/scaling"
echo "grid=$GRID ranks=[$RANKS] reps=$REPS  (job $SLURM_JOB_ID on $(hostname))"
python3 scaling_study.py --strong "$GRID" --ranks $RANKS --builds after --reps "$REPS"
