#!/bin/bash -l
#SBATCH --job-name=cc1-baseline
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=cc1_baseline-%j.out
#
# THE clean-wall-time equilibrium baseline: corrected-Callaghan (matrix-free Anderson,
# 1st-order backward Euler, restart OFF, no Tbar, no TR-BDF2), COLD from t=0, dt=1 week,
# fsm OFF, on a DEDICATED (--exclusive) node so the wall time is uncontended. cc_1 is the
# baseline for BOTH equilibrium time (cycles + clean wall) and the reference w_eq (its
# converged wtd = the "correct answer" other methods are checked against).
#
# Runs to convergence NO MATTER HOW MANY STEPS: 100000-cycle backstop + no per-run timeout;
# the model's eq_tol auto-stop (per-cycle max|Δwtd| < 0.01 m for 2 consecutive cycles) ends
# it exactly at convergence. Saves every 100 cycles (progress + restart points).
#
#   cd ~/models/WTM/benchmark/esquibel && sbatch run_baseline_msi.sbatch     # N=16 default
#   N=32 sbatch run_baseline_msi.sbatch                                      # override cores
set -uo pipefail
N="${N:-16}"                          # FIXED ranks for the whole algorithmic comparison
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
mkdir -p results/algo/reference
stem="results/algo/reference/cc1_baseline_msi"
sed "s#^fsm_on.*#fsm_on 0#;s#^supplied_wt.*#supplied_wt 0#;s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 100000#;s#^cycles_to_save.*#cycles_to_save 100#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > ${stem}.cfg
rm -f ${stem}.txt ${stem}_*.tif

echo "cc1 baseline: job ${SLURM_JOB_ID:-?} on $(hostname); N=$N ranks; exclusive node; cold from t=0"
echo "placement probe:"; mpiexec -ppn "$N" -n "$N" hostname 2>/dev/null | sort | uniq -c | sed 's/^/  /'
t0=$(date +%s)
mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg \
  -wtm_anderson -snes_anderson_restart_type none -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_eq_tol 0.01 \
  > ${stem}.log 2>&1
rc=$?; t1=$(date +%s)
echo "rc=$rc  CLEAN_WALL_S=$((t1-t0))  cycles=$(grep -c 'per-cycle max' ${stem}.log 2>/dev/null)"
grep 'equilibrium reached' ${stem}.log | tail -1 || echo "did NOT auto-stop (check backstop/walltime)"
