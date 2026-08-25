#!/bin/bash -l
#SBATCH --job-name=wtm-tr-smoke
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=8:00:00
#SBATCH --output=trsmoke-%j.out
#
# Transient relaxation-timescale SMOKE (task #93 pre-step).
# Warm-start from the corrected equilibrium (w_eq_correct.tif, wtd pinned at surface),
# apply a FACTOR-f step change to net forcing P-ET (perturb_pet.py: precip'=fP+(1-f)E),
# and watch the per-cycle |Delta wtd| relax. Goal: pick T_end inside the ACTIVE-relaxation
# window before the cc-vs-TR-BDF2 dt-sweep. cc config (Anderson, backward-Euler, restart off),
# fsm OFF, -wtm_surface_exfiltration_to_runoff (post-solve clamp = the true exfiltration constraint).
#   FACTOR=0.8 sbatch transient_smoke_msi.sbatch   # -20% dry (primary; deepens toward steep exp-T)
#   FACTOR=1.2 sbatch transient_smoke_msi.sbatch   # +20% wet (control)
set -uo pipefail
FACTOR="${FACTOR:-0.8}"
N="${N:-16}"
CYCLES="${CYCLES:-104}"        # 2 yr at dt=1 wk; long enough to see the decay flatten
SAVE="${SAVE:-13}"             # snapshot every quarter-year
# tag: dry (f<1) or wet (f>=1)
if awk "BEGIN{exit !($FACTOR < 1.0)}"; then FTAG=dry; else FTAG=wet; fi

WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"

TR="results/algo/transient"
PERT="$TR/pert_${FTAG}"
IC="$TR/w_eq_correct.tif"
stem="$TR/smoke_${FTAG}"

# 1. build the perturbed domain (mirrors inputs, overwrites precip, drops IC as _wtd.tif)
python3 perturb_pet.py domain "$PERT" "$IC" "$FACTOR"

# 2. transient cfg (self-contained; matches Esquibel domain geometry)
cat > ${stem}.cfg <<CFG
run_type transient
fsm_on 0
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat 604800
total_cycles $CYCLES
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
echo "tr-smoke FACTOR=$FACTOR ($FTAG) job ${SLURM_JOB_ID:-?} on $(hostname); N=$N; cycles=$CYCLES dt=1wk"
t0=$(date +%s)
mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg \
  -wtm_anderson -snes_anderson_restart_type none \
  -wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff \
  -snes_stol 1e-6 > ${stem}.log 2>&1
rc=$?; t1=$(date +%s)
echo "rc=$rc FACTOR=$FACTOR WALL_S=$((t1-t0)) cycles_logged=$(grep -c '^[0-9]' ${stem}.txt 2>/dev/null)"
echo "== per-cycle |Delta wtd| (col 5 abs_total) decay =="
awk 'NR>1 && /^[0-9]/{print $1, $5}' ${stem}.txt | tail -n +1
