#!/bin/bash -l
#SBATCH --job-name=wtm-tr-diag
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=8:00:00
#SBATCH --output=trdiag-%j.out
#
# Transient diagnostic (#93 follow-up). Two uses, keyed by env:
#  (B) per-method self-convergence at an intermediate horizon (time-order, isolated from
#      the cc-vs-tr method difference): TEND_WK=2, DTS="0.0625 0.125 0.25 0.5 1 2", SAVEEVERY=0
#  (A) cell trajectory: fine dt, SAVEEVERY=1 -> one raster per cycle to read wtd(t) at a cell.
# Each job gets its OWN perturbed-domain dir (no cross-job race).
#   METHOD=cc FACTOR=0.8 TEND_WK=2 DTS="0.0625 0.125 0.25 0.5 1 2" SAVEEVERY=0 TAG=selfconv sbatch transient_diag_msi.sbatch
#   METHOD=tr FACTOR=0.8 TEND_WK=16 DTS="0.5" SAVEEVERY=1 TAG=traj sbatch transient_diag_msi.sbatch
set -uo pipefail
FACTOR="${FACTOR:-0.8}"; METHOD="${METHOD:-cc}"; N="${N:-16}"
TEND_WK="${TEND_WK:-2}"; DTS="${DTS:-0.0625 0.125 0.25 0.5 1 2}"
SAVEEVERY="${SAVEEVERY:-0}"; TAG="${TAG:-selfconv}"
CLAMP="${CLAMP:-1}"   # 1 = -wtm_surface_exfiltration_to_runoff (default); 0 = pure GW, no clamp (consistency test)
if awk "BEGIN{exit !($FACTOR < 1.0)}"; then FTAG=dry; else FTAG=wet; fi
[ "$CLAMP" = 1 ] && CLAMPFLAG="-wtm_surface_exfiltration_to_runoff" || CLAMPFLAG=""
case "$METHOD" in
  cc)     BASE="-wtm_anderson -snes_anderson_restart_type none"; METHFLAGS="" ;;
  tr)     BASE="-wtm_anderson -snes_anderson_restart_type none"; METHFLAGS="-wtm_tr_bdf2" ;;
  picard) BASE=""; METHFLAGS="-wtm_picard" ;;                    # independent volume-form BE-in-time
  bdf2v)  BASE="-wtm_anderson -snes_anderson_restart_type none"; METHFLAGS="-wtm_bdf2_on_V" ;;
  cctbar) BASE="-wtm_anderson -snes_anderson_restart_type none"; METHFLAGS="-wtm_Tbar" ;;       # BE + time-averaged T
  trtbar) BASE="-wtm_anderson -snes_anderson_restart_type none"; METHFLAGS="-wtm_tr_bdf2 -wtm_Tbar" ;;
  *) echo "unknown METHOD $METHOD"; exit 2 ;;
esac

WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
OUT="results/algo/transient/${TAG}"; mkdir -p "$OUT"
PERT="results/algo/transient/pert_${TAG}_${FTAG}_${METHOD}"
IC="results/algo/transient/w_eq_correct.tif"
python3 perturb_pet.py domain "$PERT" "$IC" "$FACTOR"

CSV="$OUT/runs_${FTAG}_${METHOD}.csv"; echo "forcing,method,dt_wk,steps,rc,wall_s,final_tif" > "$CSV"
echo "diag TAG=$TAG FACTOR=$FACTOR($FTAG) METHOD=$METHOD job ${SLURM_JOB_ID:-?} on $(hostname); T_end=${TEND_WK}wk; dts:$DTS saveevery=$SAVEEVERY"
for DTWK in $DTS; do
  DT=$(awk "BEGIN{printf \"%d\", $DTWK*604800}")
  CYC=$(awk "BEGIN{printf \"%d\", $TEND_WK/$DTWK + 0.5}")
  [ "$SAVEEVERY" = 1 ] && SAVE=1 || SAVE=$CYC
  stem="$OUT/${FTAG}_${METHOD}_dt${DTWK}wk"
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
cycles_to_save $SAVE
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
    -wtm_fringe_source ksat $CLAMPFLAG -snes_stol 1e-6 > ${stem}.log 2>&1
  rc=$?; t1=$(date +%s)
  [ -f "$finaltif" ] || finaltif="MISSING"
  echo "$FTAG,$METHOD,$DTWK,$CYC,$rc,$((t1-t0)),$finaltif" | tee -a "$CSV"
done
echo "DONE TAG=$TAG METHOD=$METHOD"
