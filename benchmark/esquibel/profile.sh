#!/usr/bin/env bash
# Profile the Esquibel groundwater solve BEFORE committing to a long run: per-phase
# timers (WTM's GW/FSM/recharge) + PETSc -log_view + a scaling curve, over a few cycles.
# Answers "where does per-cycle time go, and why does it scale as it does" -- without the
# full equilibrium run. maxiter stays 50 (unchanged).
#
# Usage: profile.sh [ranks...]        (default "1 2 4 8");  NCYC=3 overridable
# Findings 2026-08-10 (this laptop): FSM ~= 2e-6 s/cyc (a no-op here); the cost is SNESSolve
# (matrix-free residual evals); it is MEMORY-BANDWIDTH-BOUND (Mflop/s ~600/core, tiny comm,
# load balance ~1.0) so it saturates at ~2x on one shared memory bus. Cluster (MSI, multi-
# socket/node) should scale better -- re-profile there before quoting scaling.

set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
OURS=${OURS:-/home/awickert/models/WTM/build/wtm.x}
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6"
RANKS=${RANKS:-${*:-1 2 4 8}}
NCYC=${NCYC:-3}
[ -f domain/Esquibel_010000_topography.tif ] || { echo "domain not staged -- run ./make_esquibel.py"; exit 1; }
P=results/prof; mkdir -p "$P"
sed "s/^total_cycles.*/total_cycles $NCYC/;s/^cycles_to_save.*/cycles_to_save $NCYC/;s#results/eq_awickert#$P/p#g" eq_awickert.cfg > "$P/p.cfg"

for n in $RANKS; do
  : > "$P/p.txt"
  mpirun -n $n "$OURS" "$P/p.cfg" $OF -log_view > "$P/logview_n$n.txt" 2>&1
  gw=$(awk -F'= ' '/t GW time =/{g+=$2;c++} END{if(c)printf"%.2f",g/c;else printf"?"}' "$P/logview_n$n.txt")
  fsm=$(awk -F'= ' '/t FSM time =/{f+=$2;c++} END{if(c)printf"%.2e",f/c;else printf"?"}' "$P/logview_n$n.txt")
  echo "n=$n: GW~${gw}s/cyc  FSM~${fsm}s/cyc"
done

echo "--- SNESSolve scaling + Mflop/s (low => memory-bandwidth-bound) ---"
base=""
for n in $RANKS; do
  t=$(grep -E "^SNESSolve " "$P/logview_n$n.txt" | awk '{print $4}')
  mf=$(grep -E "^SNESSolve " "$P/logview_n$n.txt" | awk '{print $NF}')
  pc=$(awk "BEGIN{printf \"%.2f\", $t/$NCYC}")
  [ -z "$base" ] && base=$pc
  printf "  n=%-2s %6ss/cyc  %5.2fx  %s Mflop/s\n" "$n" "$pc" "$(awk "BEGIN{printf \"%.2f\", $base/$pc}")" "$mf"
done
