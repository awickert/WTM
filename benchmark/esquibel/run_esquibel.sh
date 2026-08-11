#!/usr/bin/env bash
# Run the full-Esquibel (384,703-cell) equilibrium test for one model, at a given MPI
# rank count. Same structure as benchmark/island/run_island.sh -- the larger domain
# where parallel scaling has room to show and v2.0.1's parallel SEGV is decisive.
#
#   run_esquibel.sh <target> <binary> [nranks]
#     <target>  kcallaghan | awickert   (selects the cfg + the solver flags)
#     <binary>  path to that model's wtm.x
#     [nranks]  MPI ranks (default 1). Runs via `mpirun -n <nranks>`.
#
# Stage the domain first:  ./make_esquibel.py   (populates ./domain from the source data)

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
# WTM parallelism is MPI, not OpenMP: single-threaded ranks, or FormFunctionLocal's
# OpenMP region deadlocks against PETSc's collectives under oversubscription.
export OMP_NUM_THREADS=1

TARGET=${1:?usage: run_esquibel.sh <kcallaghan|awickert> <binary> [nranks]}
BIN=${2:?path to wtm.x}
NR=${3:-1}

# Both models solve each step to the SAME inner tolerance (snes_stol 1e-6) -- required
# for a fair speed comparison (ours' PETSc default is 1e-8 = 100x tighter).
case "$TARGET" in
  kcallaghan) FLAGS="-snes_mf -snes_type anderson -snes_stol 1e-6" ;;
  awickert)   FLAGS="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6" ;;
  *) echo "unknown target '$TARGET' (want kcallaghan|awickert)"; exit 2 ;;
esac

[ -f "$HERE/domain/Esquibel_010000_topography.tif" ] || { echo "domain not staged -- run ./make_esquibel.py first"; exit 1; }
cd "$HERE"
mkdir -p results
CFG=eq_$TARGET.cfg
LOG=results/eq_${TARGET}_n${NR}.log

echo "### Esquibel EQ | target=$TARGET | nranks=$NR | $BIN"
echo "### flags: $FLAGS"
t0=$(date +%s%N)
mpirun -n "$NR" "$BIN" "$CFG" $FLAGS > "$LOG" 2>&1
rc=$?
t1=$(date +%s%N)
echo "WALL_MS $(( (t1-t0)/1000000 ))  RC $rc  (log: $LOG)"

TXT=results/eq_$TARGET.txt
[ -f "$TXT" ] && awk 'NF>=5 && $1~/^[0-9]+$/{c=$1; d=$5} END{if(c!="") print "  last cycle="c"  Δ(col5)="d}' "$TXT"
