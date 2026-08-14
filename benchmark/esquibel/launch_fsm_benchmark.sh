#!/bin/bash
# Launch the FSM-ON transient dt-sweep benchmark (production "route as fill-spill-merge lakes" mode).
#
# Companion to the FSM-OFF (discard) sweeps that live in results/algo/transient/{bench1,bigdt}. This runs the
# SAME schemes with fsm_on 1 (on top of the clamp) to check whether the cross-scheme ranking (TR-BDF2 dominant
# on SNES iterations) still holds when FSM routes the exfiltrated surface water each cycle. FSM sits OUTSIDE
# the GW solve, so SNES-iteration cost stays a clean solver metric.
#
# Schemes: cc (backward-Euler baseline), tr (TR-BDF2, the FSM-off winner), bdf2v (BDF2-on-V, the other
# 2nd-order). Tbar variants are omitted: Tbar is a cold-start tool and was shown redundant on these WARM
# perturbations (see memory finding_transient_speed_benchmark). Dry -20% P-ET step (the telling direction).
#
# Each sweep goes to its OWN dir (bench_fsm, bigdt_fsm) so it carries its own FSM-on tr@finest reference,
# which the analysis (dtsweep_speed.py) uses as the dt->0 truth. Analyze afterward with, e.g.:
#   WTM_METHODS="cc tr bdf2v" python3 dtsweep_speed.py results/algo/transient/bench_fsm dry 2
#   WTM_DTS="0.5 1 2 4 8 16 32" WTM_METHODS="cc tr bdf2v" python3 dtsweep_speed.py results/algo/transient/bigdt_fsm dry 32 tr 0.5
#
# Run from benchmark/esquibel on MSI (Agate), conda env active (source ../../msi_env.sh test).
set -eu
cd "$(dirname "$0")"

export FACTOR=0.8 FSM=1 CLAMP=1 EXTRA="-wtm_ghost_boundary" N=16   # dry step, FSM on, clamp on, corrected domain

for M in cc tr bdf2v; do
  METHOD=$M TEND_WK=2  DTS="0.0625 0.125 0.25 0.5 1 2" TAG=bench_fsm sbatch transient_diag_msi.sbatch
  METHOD=$M TEND_WK=32 DTS="0.5 1 2 4 8 16 32"         TAG=bigdt_fsm sbatch transient_diag_msi.sbatch
done
echo "submitted 6 FSM-on jobs (cc/tr/bdf2v x {bench_fsm 2wk, bigdt_fsm 32wk})"
