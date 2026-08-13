#!/bin/bash -l
#SBATCH --job-name=wtm-eq
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=eq-%j.out
#
# Equilibrium algorithmic-comparison runner (ONE config per job), cold from t=0, dt=1 week,
# fsm OFF, with -wtm_surface_exfiltration_to_runoff = the CORRECT free-surface treatment:
# water mounds during the solve, then POST-SOLVE the above-surface excess is routed to runoff
# (->FSM when on) and the table is CLAMPED to the surface (wtd=0). This is the true seepage
# face. (NOT -wtm_direct_to_runoff, the in-residual max(0,w)/dt kink -- that leaves a
# dt-dependent RESIDUAL POND above the surface (14k cells, up to 2.26m) and drives the bounce.)
# eq_tol 0.01 m: with the post-solve clamp the bounce is gone, so TRUE convergence is reached
# (verified: ~20 cycles, monotone, wtd pinned at 0). Distinct stem per CONFIG.
#   CONFIG=tbar sbatch algo_eq_msi.sbatch
set -uo pipefail
CONFIG="${CONFIG:-cc}"                 # cc | tbar | adaptive | tbar_adaptive
MULT="${MULT:-1}"                      # dt multiplier: deltat = MULT * 1 week (dt sweep)
N="${N:-16}"
BASE="-wtm_anderson -snes_anderson_restart_type none"   # all configs are Anderson, restart off
case "$CONFIG" in
  cc)            CFGFLAGS="$BASE" ;;
  tbar)          CFGFLAGS="$BASE -wtm_Tbar" ;;
  adaptive)      CFGFLAGS="$BASE -wtm_dt_adaptive" ;;                 # Anderson + adaptive dt (verified: SNES=anderson)
  tbar_adaptive) CFGFLAGS="$BASE -wtm_Tbar -wtm_dt_adaptive" ;;
  *) echo "unknown CONFIG '$CONFIG' (want cc|tbar|adaptive|tbar_adaptive)"; exit 2 ;;
esac
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
mkdir -p results/algo/reference
stem="results/algo/reference/${CONFIG}_dt${MULT}_eq_msi"
DT=$((604800*MULT))
sed "s#^fsm_on.*#fsm_on 0#;s#^supplied_wt.*#supplied_wt 0#;s#^deltat.*#deltat $DT#;s#^total_cycles.*#total_cycles 400#;s#^cycles_to_save.*#cycles_to_save 400#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > ${stem}.cfg
rm -f ${stem}.txt ${stem}_*.tif
echo "eq CONFIG=$CONFIG dt=${MULT}wk: job ${SLURM_JOB_ID:-?} on $(hostname); N=$N ranks; cold from t=0; flags: $CFGFLAGS"
t0=$(date +%s)
mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg $CFGFLAGS -wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff -snes_stol 1e-6 -wtm_eq_tol 0.01 > ${stem}.log 2>&1
rc=$?; t1=$(date +%s)
echo "rc=$rc CONFIG=$CONFIG WALL_S=$((t1-t0)) cycles=$(grep -c 'per-cycle max' ${stem}.log 2>/dev/null)"
grep 'equilibrium reached' ${stem}.log | tail -1 || echo "did NOT auto-stop (backstop/walltime)"
