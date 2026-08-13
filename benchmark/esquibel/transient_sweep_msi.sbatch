#!/bin/bash -l
#SBATCH --job-name=wtm-tr-sweep
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=trsweep-%j.out
#
# Transient dt-sweep (task #93): corrected-Callaghan (cc, backward-Euler, 1st-order) vs
# TR-BDF2 (2nd-order), warm from the corrected equilibrium, under a +/-20% P-ET step.
# Fixed physical horizon T_end = 16 weeks; error-vs-dt (accuracy) + wall-vs-dt (cost)
# => "wall time at fixed accuracy" = how much TR-BDF2 buys over cc.
# Both Anderson + -wtm_surface_exfiltration_to_runoff (post-solve clamp), fsm OFF.
#   FACTOR=0.8 METHOD=cc sbatch transient_sweep_msi.sbatch   # dry, backward-Euler
#   FACTOR=0.8 METHOD=tr sbatch transient_sweep_msi.sbatch   # dry, TR-BDF2 (emits 0.25wk reference)
#   FACTOR=1.2 METHOD=cc ... ; FACTOR=1.2 METHOD=tr ...      # wet control
set -uo pipefail
FACTOR="${FACTOR:-0.8}"
METHOD="${METHOD:-cc}"      # cc | tr
N="${N:-16}"
TEND_WK=16                  # fixed physical horizon (weeks)
if awk "BEGIN{exit !($FACTOR < 1.0)}"; then FTAG=dry; else FTAG=wet; fi

# dt list (weeks): tr also carries the 0.25wk reference trajectory
if [ "$METHOD" = tr ]; then
  DTS="0.25 0.5 1 2 4 8 16"
  METHFLAGS="-wtm_tr_bdf2"
else
  DTS="0.5 1 2 4 8 16"
  METHFLAGS=""
fi
BASE="-wtm_anderson -snes_anderson_restart_type none"

WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"

TR="results/algo/transient/sweep"
mkdir -p "$TR"
PERT="results/algo/transient/pert_${FTAG}_${METHOD}"   # per-job dir: cc/tr of a forcing must not race the same perturbed domain
IC="results/algo/transient/w_eq_correct.tif"

# (re)build the perturbed domain (idempotent; drops IC as _wtd.tif)
python3 perturb_pet.py domain "$PERT" "$IC" "$FACTOR"

CSV="$TR/runs_${FTAG}_${METHOD}.csv"
echo "forcing,method,dt_wk,steps,rc,wall_s,final_tif" > "$CSV"
echo "sweep FACTOR=$FACTOR ($FTAG) METHOD=$METHOD job ${SLURM_JOB_ID:-?} on $(hostname); dts: $DTS; T_end=${TEND_WK}wk"

for DTWK in $DTS; do
  DT=$(awk "BEGIN{printf \"%d\", $DTWK*604800}")
  CYC=$(awk "BEGIN{printf \"%d\", $TEND_WK/$DTWK + 0.5}")
  stem="$TR/${FTAG}_${METHOD}_dt${DTWK}wk"
  cat > ${stem}.cfg <<CFG
run_type transient
fsm_on 0
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat $DT
total_cycles $CYC
cycles_to_save $CYC
maxiter            50
fdepth_a           100
fdepth_b           150
fdepth_fmin        2.5
time_start         010000
time_end           010000
surfdatadir $PERT/
region             Esquibel
supplied_wt 0
textfilename ${stem}.txt
outfile_prefix ${stem}_
CFG
  rm -f ${stem}.txt ${stem}_*.tif
  finaltif=$(printf "%s_%09d.tif" "$stem" "$CYC")
  t0=$(date +%s)
  mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg $BASE $METHFLAGS \
    -wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff \
    -snes_stol 1e-6 > ${stem}.log 2>&1
  rc=$?; t1=$(date +%s)
  [ -f "$finaltif" ] || finaltif="MISSING"
  echo "$FTAG,$METHOD,$DTWK,$CYC,$rc,$((t1-t0)),$finaltif" | tee -a "$CSV"
done
echo "DONE FACTOR=$FACTOR METHOD=$METHOD"
