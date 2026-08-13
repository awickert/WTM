#!/bin/bash
# THE baseline: corrected-Callaghan (cc), COLD from t=0 (wtd=0), dt=1 week, fsm OFF.
# cc_1 is Andy's baseline for BOTH time (cycles/wall to converge) and convergence (its
# converged wtd = the reference w_eq). No warm-start anywhere -- maximally solid.
#
# Runs to convergence NO MATTER HOW MANY STEPS: huge cycle backstop + NO wall-clock
# timeout; the model's eq_tol auto-stop (per-cycle max|Δwtd| < 0.01 m for 2 consecutive
# cycles) ends it exactly at convergence. Saves every 100 cycles (progress + restart pts).
set -uo pipefail
cd "$(dirname "$0")"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
BIN=../../build/wtm.x; N=${1:-4}
stem=results/algo/reference/cc1_baseline
sed "s#^fsm_on.*#fsm_on 0#;s#^supplied_wt.*#supplied_wt 0#;s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 100000#;s#^cycles_to_save.*#cycles_to_save 100#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > ${stem}.cfg
rm -f ${stem}.txt ${stem}_*.tif
echo "baseline start: $(date)" > ${stem}.log
t0=$(date +%s)
# NO timeout: run until eq_tol fires (or the 100000-cycle backstop, ~unbounded).
mpirun -n "$N" "$BIN" ${stem}.cfg \
  -wtm_anderson -snes_anderson_restart_type none -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_eq_tol 0.01 \
  >> ${stem}.log 2>&1
echo "rc=$? wall=$(( $(date +%s)-t0 ))s  end: $(date)" >> ${stem}.log
