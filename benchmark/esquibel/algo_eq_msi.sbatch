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
# fsm OFF but WITH -wtm_direct_to_runoff = the Fan & Miguez-Macho config (exfiltration/ponding
# collected into arp.runoff and removed from the water table, NOT ponded above surface, NOT
# FSM-routed). This is the physically-correct fsm-off equilibrium; without it, closed basins
# pond unboundedly (found: 110k cells, up to 758 m) and drive a multi-metre limit cycle.
# 16-core shared slice (first-look timing). eq_tol 0.1 m auto-stop: runoff cuts the residual
# oscillation ~100x to ~5 cm / ~18 cells (acceptable margin), so 0.1 m fires cleanly where
# 0.01 m cannot. Distinct stem per CONFIG so it never clobbers other runs.
#   CONFIG=tbar sbatch algo_eq_msi.sbatch
set -uo pipefail
CONFIG="${CONFIG:-cc}"                 # cc | tbar | adaptive | tbar_adaptive
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
stem="results/algo/reference/${CONFIG}_eq_msi"
sed "s#^fsm_on.*#fsm_on 0#;s#^supplied_wt.*#supplied_wt 0#;s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 100000#;s#^cycles_to_save.*#cycles_to_save 100#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > ${stem}.cfg
rm -f ${stem}.txt ${stem}_*.tif
echo "eq CONFIG=$CONFIG: job ${SLURM_JOB_ID:-?} on $(hostname); N=$N ranks; cold from t=0; flags: $CFGFLAGS"
t0=$(date +%s)
mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg $CFGFLAGS -wtm_fringe_source ksat -wtm_direct_to_runoff -snes_stol 1e-6 -wtm_eq_tol 0.1 > ${stem}.log 2>&1
rc=$?; t1=$(date +%s)
echo "rc=$rc CONFIG=$CONFIG WALL_S=$((t1-t0)) cycles=$(grep -c 'per-cycle max' ${stem}.log 2>/dev/null)"
grep 'equilibrium reached' ${stem}.log | tail -1 || echo "did NOT auto-stop (backstop/walltime)"
