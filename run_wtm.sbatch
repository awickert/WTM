#!/bin/bash -l
#SBATCH --job-name=wtm
#SBATCH --partition=msilong          # long single-node (<=32 cores / 128 GB); or msismall / msilarge
#SBATCH --nodes=1
#SBATCH --ntasks=32                  # MPI ranks; a bandwidth-bound stencil may plateau below the node max
#SBATCH --mem=120gb
#SBATCH --time=1-00:00:00            # D-HH:MM:SS  (msilong allows up to 37 days)
#SBATCH --output=wtm-%j.out
#
# Run a WTM simulation on MSI (Agate). Submit from the repo root, or set WTM_ROOT:
#     sbatch run_wtm.sbatch /path/to/config.cfg
#     WTM_ROOT=$HOME/models/WTM sbatch run_wtm.sbatch /path/to/config.cfg
#
# Loads the SAME toolchain used to build (via msi_env.sh), so the runtime libs
# match the ones the binary was linked against. Edit the #SBATCH headers above
# (partition, ntasks, mem, time) for your run; see BUILD_HPC.md for partitions.
set -euo pipefail

# Repo location: the dir you ran `sbatch` from, unless WTM_ROOT is set (or edit this).
WTM_ROOT="${WTM_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/models/WTM}}"
CONFIG="${1:?usage: sbatch run_wtm.sbatch <config.cfg>}"

# shellcheck source=msi_env.sh
source "$WTM_ROOT/msi_env.sh"          # same petsc/gdal/gcc used at build time

# Pure MPI: one rank per core, no OpenMP threads. OpenMP x MPI on the same cores
# oversubscribes and can hang at >=4 ranks; parallelism comes from --ntasks above.
export OMP_NUM_THREADS=1

cd "$WTM_ROOT/build"
echo "Running wtm.x on ${SLURM_NTASKS:-?} ranks: $CONFIG"
# MPICH (bundled with PETSc) under Slurm needs the pmi2 PMI.
srun --mpi=pmi2 ./wtm.x "$CONFIG"
