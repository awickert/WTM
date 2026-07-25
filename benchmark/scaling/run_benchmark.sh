#!/usr/bin/env bash
# Full clean -O3 scaling benchmark on MSI: build all three builds with PETSc's MPI
# wrappers, verify they are Release/-O3, then run the grid x rank sweep.
#
# SOURCE this (module load / conda activate must affect your shell), from an MSI
# compute session -- e.g.:
#
#     srun -N 1 --ntasks-per-node=32 --mem=64gb -t 6:00:00 -p msilong --pty bash
#     source ~/models/WTM/benchmark/scaling/run_benchmark.sh
#
# Builds (siblings of this repo, the layout scaling_study.py expects):
#   after       ~/models/WTM             (this repo / current tip -- distributed build)
#   before      ~/models/WTM-before      (a worktree at e5aab70 = pre-flip proxy)
#   kcallaghan  ~/models/WTM-kcallaghan  (published v2.0.1 baseline)
#
# Why the flags matter (learned the hard way):
#   * CXX=mpicxx CC=mpicc  -- the flip added direct MPI calls (MPI_Bcast) in
#     initialise(); plain g++ fails to link ("undefined reference to MPI_Bcast /
#     libmpi.so: DSO missing from command line"). kcallaghan (no direct MPI) slips
#     through with g++, which is misleading. Use PETSc's wrappers for all three.
#   * -DCMAKE_BUILD_TYPE=Release (-O3) for ALL builds -- a -O0 build silently
#     inflated the original study's per-core "regression". Changing the compiler
#     needs a FRESH build dir (cmake won't switch compilers on an existing cache).
#   * BUILD with conda OFF (module GDAL for linking); RUN with conda ON
#     (wtmtest -> rasterio for the .tif I/O). See BUILD_HPC.md, msi_env.sh.
#
# Edit GRIDS / RANKS to trim the sweep (the 8000^2 rows are long).

GRIDS="1000 2000 4000 8000"
RANKS="1 2 4 8 16 32"
MODELS=~/models

# ---------- BUILD (conda OFF; PETSc mpicxx/mpicc; fresh build dirs at -O3) ----------
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null   # leave wtmtest for the build
cd "$MODELS/WTM" && source msi_env.sh
which mpicxx mpicc                                            # sanity: under $PETSC_DIR/bin

# create the pre-flip 'before' worktree if it's not there yet
[ -e "$MODELS/WTM-before/.git" ] || git -C "$MODELS/WTM" worktree add "$MODELS/WTM-before" e5aab70

for d in WTM WTM-before WTM-kcallaghan; do
  echo "=========== building $MODELS/$d ==========="
  cd "$MODELS/$d" && git submodule update --init --recursive
  rm -rf build && mkdir build && cd build
  CXX=mpicxx CC=mpicc cmake -DCMAKE_BUILD_TYPE=Release -DUSE_GDAL=ON .. && make -j8
  printf ">>> %-14s " "$d:"; [ -x wtm.x ] && echo "wtm.x OK" || echo "!!! BUILD FAILED"
done

echo "=========== build types (all three must read Release) ==========="
for d in WTM WTM-before WTM-kcallaghan; do
  printf "  %-16s " "$d:"
  grep CMAKE_BUILD_TYPE "$MODELS/$d/build/CMakeCache.txt" 2>/dev/null || echo "(no build)"
done

# ---------- RUN (conda ON for rasterio's .tif I/O) ----------
source "$MODELS/WTM/msi_env.sh" test
conda activate wtmtest
cd "$MODELS/WTM/benchmark/scaling"
python3 scaling_study.py --grids $GRIDS --ranks $RANKS
