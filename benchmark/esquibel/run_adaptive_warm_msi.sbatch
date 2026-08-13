#!/bin/bash -l
#SBATCH --job-name=wtm-adwarm
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=6:00:00
#SBATCH --output=adwarm-%j.out
#
# WARM adaptive-dt test of the limit-cycle hypothesis: start from the cc_1 LIMIT-CYCLE state
# (cc1_bulk_eq.tif, cycle 3200 -- bulk-converged with ~65 cells oscillating 2-6 m at fixed
# dt=1wk) and turn on adaptive dt. Hypothesis (Andy): error-controlled dt SHRINKING breaks the
# overshoot cycle in those cells and reaches TRUE convergence (eq_tol 0.01 fires) where fixed
# dt=1wk cannot. Fast, targeted: only ~65 cells left to settle. If it does NOT converge, the
# limit cycle survives adaptive dt -> honest negative.
set -uo pipefail
N="${N:-16}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
# warm domain: symlink real inputs + seed the limit-cycle state as the starting water table.
# equilibrium + supplied_wt=1 reads {region}_{time}_starting_wt.tif (src/irf.cpp:139).
WDOM=results/algo/reference/warmdom_adaptive
mkdir -p "$WDOM"
for f in domain/Esquibel_*.tif; do ln -sf "$(readlink -f "$f")" "$WDOM/$(basename "$f")"; done
cp -f results/algo/reference/cc1_bulk_eq.tif "$WDOM/Esquibel_010000_starting_wt.tif"
stem=results/algo/reference/adaptive_warm_msi
sed "s#^run_type.*#run_type equilibrium#;s#^fsm_on.*#fsm_on 0#;s#^supplied_wt.*#supplied_wt 1#;s#^surfdatadir.*#surfdatadir ${WDOM}/#;s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 300#;s#^cycles_to_save.*#cycles_to_save 50#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > ${stem}.cfg
rm -f ${stem}.txt ${stem}_*.tif
echo "warm adaptive: job ${SLURM_JOB_ID:-?} on $(hostname); seed=cc1_bulk_eq.tif (limit-cycle state); N=$N"
t0=$(date +%s)
mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg \
  -wtm_anderson -snes_anderson_restart_type none -wtm_dt_adaptive -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_eq_tol 0.01 \
  > ${stem}.log 2>&1
rc=$?; t1=$(date +%s)
echo "rc=$rc WALL_S=$((t1-t0)) cycles=$(grep -c 'per-cycle max' ${stem}.log 2>/dev/null)"
grep 'equilibrium reached' ${stem}.log | tail -1 || echo "did NOT reach true eq -- limit cycle survived adaptive dt"
