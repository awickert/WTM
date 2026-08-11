#!/usr/bin/env bash
# Run the island equilibrium test for one model, at a given MPI rank count.
#
#   run_island.sh <target> <binary> [nranks]
#     <target>  kcallaghan | awickert   (selects the cfg + the solver flags)
#     <binary>  path to that model's wtm.x
#     [nranks]  MPI ranks (default 1). Runs via `mpirun -n <nranks>`.
#
# The two models solve the SAME domain/physics (eq_<target>.cfg differ only in
# output paths); they differ ONLY in how the solver is invoked:
#   kcallaghan (v2.0.1): no built-in Anderson -> PETSc matrix-free Anderson on the CLI.
#   awickert  (fork):    Anderson, 2nd-order-in-time (BDF2-on-V), capillary taper length.
#
# KCallaghan v2.0.1 is expected to SEGV at nranks>=4 (the MPI ghost-cell bug the
# fork fixes) -- that parallel capability is itself an axis of the comparison.

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
# WTM's parallelism is MPI, not OpenMP. Each rank must run single-threaded, or the
# OpenMP region in FormFunctionLocal deadlocks against PETSc's collective reductions
# under oversubscription (hangs at >=4 ranks). The repo's MPI tests pin this too.
export OMP_NUM_THREADS=1

TARGET=${1:?usage: run_island.sh <kcallaghan|awickert> <binary> [nranks]}
BIN=${2:?path to wtm.x}
NR=${3:-1}

case "$TARGET" in
  kcallaghan) FLAGS="-snes_mf -snes_type anderson -snes_stol 1e-6" ;;
  awickert)   FLAGS="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6" ;;
              # snes_stol 1e-6 MATCHES kcallaghan's inner tolerance -- required for a fair
              # speed comparison (ours' PETSc default is 1e-8 = 100x tighter = more iters/solve).
              # 1st-order in time: this is a cold-start equilibrium spin-up, where
              # 2nd-order (-wtm_bdf2_on_V) rings in a limit cycle instead of settling.
              # BDF2-on-V belongs to the warm transient, not here.
              # OPEN: add -wtm_Tbar (log-mean T)? Champion combo in prior work; left off
              # pending a call, since it is not part of the stated spec.
  *) echo "unknown target '$TARGET' (want kcallaghan|awickert)"; exit 2 ;;
esac

cd "$HERE"
mkdir -p results
CFG=eq_$TARGET.cfg
LOG=results/eq_${TARGET}_n${NR}.log

echo "### island EQ | target=$TARGET | nranks=$NR | $BIN"
echo "### flags: $FLAGS"
t0=$(date +%s%N)
timeout 1200 mpirun -n "$NR" "$BIN" "$CFG" $FLAGS > "$LOG" 2>&1
rc=$?
t1=$(date +%s%N)
echo "WALL_MS $(( (t1-t0)/1000000 ))  RC $rc  (log: $LOG)"

# convergence trace: cycle (col1) and |wtd change| (col5) from the text log
TXT=results/eq_$TARGET.txt
[ -f "$TXT" ] && awk 'NF>=5 && $1~/^[0-9]+$/{c=$1; d=$5} END{if(c!="") print "  last cycle="c"  Δ(col5)="d}' "$TXT"
