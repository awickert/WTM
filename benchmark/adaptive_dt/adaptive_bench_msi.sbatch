#!/bin/bash -l
#SBATCH --job-name=wtm-adapt-bench
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=adaptbench-%j.out
#
# Adaptive-dt vs constant-dt benchmark: SNES iterations + fractional wall + accuracy, for TRANSIENT and
# SPIN-UP. Self-contained sbatch (survives login disconnect). See README.md in this directory.
#   REGIME=transient sbatch adaptive_bench_msi.sbatch   (warm from equilibrium, -20% dry, 8-week horizon)
#   REGIME=spinup    sbatch adaptive_bench_msi.sbatch   (cold from wtd=0 at surface, normal forcing, ~5 yr)
# Schemes: const tr (dt sweep) vs adaptive tr MAX-norm (tol sweep) vs adaptive tr RMS-norm (tol sweep).
# Inputs come from ../esquibel (domain/, w_eq_correct.tif, perturb_pet.py); ALL outputs land under
# benchmark/adaptive_dt/results/ (this benchmark's own results home). Analyze with analyze_adapt_bench.py.
set -uo pipefail
N=16; REGIME="${REGIME:-transient}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"                 # Esquibel inputs live here
OUT="$WTM_ROOT/benchmark/adaptive_dt/results"      # this benchmark's results home
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"                                           # cwd = Esquibel (for domain/, perturb_pet.py)
mkdir -p "$OUT"
CSV="$OUT/adapt_bench_${REGIME}.csv"; echo "regime,run,param,rc,wall_s,snes_its,nsteps,final_tif" > "$CSV"
COMMON="-wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff -wtm_ghost_boundary -snes_stol 1e-6"
BASE="-wtm_anderson -snes_anderson_restart_type none -wtm_tr_bdf2"

do_run() {  # reg lbl param surfdatadir deltat_s maxiter total_cycles extra-flags
  local reg="$1" lbl="$2" par="$3" sdd="$4" dt="$5" mi="$6" tc="$7" extra="$8"
  local stem="$OUT/${reg}_${lbl}_${par}"
  cat > ${stem}.cfg <<CFG
run_type transient
fsm_on 0
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat $dt
total_cycles $tc
cycles_to_save $tc
maxiter            $mi
fdepth_a           100
fdepth_b           150
fdepth_fmin        2.5
time_start         010000
time_end           010000
surfdatadir $sdd
region             Esquibel
supplied_wt 0
textfilename ${stem}.txt
outfile_prefix ${stem}_
CFG
  rm -f ${stem}.txt ${stem}_*.tif
  local t0=$(date +%s.%N)
  mpiexec -ppn "$N" -n "$N" "$BIN" ${stem}.cfg $BASE $extra $COMMON > ${stem}.log 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -ao "Number of nonlinear iterations = [0-9]*" ${stem}.log | awk '{s+=$6} END{print s+0}')
  local nsteps=$(grep -ao "adaptive dt: [0-9]* steps" ${stem}.log | grep -o "[0-9]*" | head -1)
  [ -z "$nsteps" ] && nsteps=$((mi*tc))
  local ftif="${stem}_$(printf %09d $tc).tif"; [ -f "$ftif" ] || ftif=MISSING
  echo "$reg,$lbl,$par,$rc,$wall,$its,$nsteps,$ftif" | tee -a "$CSV"
}

if [ "$REGIME" = transient ]; then
  # warm from equilibrium, -20% dry step, 8-week horizon (maxiter*deltat = 8wk, tc=1).
  IC="results/algo/transient/w_eq_correct.tif"; PT="$OUT/pert_adaptbench_dry"
  python3 perturb_pet.py domain "$PT" "$IC" 0.8
  do_run transient const    0.25 "$PT/" 151200  32 1 ""
  do_run transient const    0.5  "$PT/" 302400  16 1 ""
  do_run transient const    1    "$PT/" 604800   8 1 ""
  do_run transient const    2    "$PT/" 1209600  4 1 ""
  do_run transient adaptmax 5    "$PT/" 302400  16 1 "-wtm_dt_adaptive -wtm_dt_tol 5"
  do_run transient adaptmax 20   "$PT/" 302400  16 1 "-wtm_dt_adaptive -wtm_dt_tol 20"
  do_run transient adaptrms 0.5  "$PT/" 302400  16 1 "-wtm_dt_adaptive -wtm_dt_norm_rms -wtm_dt_tol 0.5"
  do_run transient adaptrms 2    "$PT/" 302400  16 1 "-wtm_dt_adaptive -wtm_dt_norm_rms -wtm_dt_tol 2"
else
  # SPIN-UP: cold start = water table at the surface (wtd = 0). The base domain/ ships no wtd tif, so build
  # a cold_domain/ = symlinked domain inputs + a zero wtd tif. Normal forcing, horizon ~5 yr. Cold start
  # spans a huge dynamic range (violent initial drainage -> slow settling): where adaptive should beat a
  # fixed dt stuck at the cold-start ceiling.
  COLD="$OUT/cold_domain"; mkdir -p "$COLD"
  for f in domain/Esquibel_*.tif; do ln -sf "$PWD/$f" "$COLD/$(basename "$f")"; done
  python3 - "$COLD" <<'PYEOF'
import rasterio, numpy as np, sys
cold = sys.argv[1]
s = rasterio.open('domain/Esquibel_010000_topography.tif')
p = s.profile.copy()
z = np.zeros((s.height, s.width), dtype=p['dtype'])
with rasterio.open(cold + '/Esquibel_010000_wtd.tif', 'w', **p) as w:
    w.write(z, 1)
print('cold_domain: wtd=0 IC written')
PYEOF
  do_run spinup const    0.5 "$COLD/" 302400  520 1 ""
  do_run spinup const    1   "$COLD/" 604800  260 1 ""
  do_run spinup const    2   "$COLD/" 1209600 130 1 ""
  do_run spinup adaptmax 20  "$COLD/" 302400  520 1 "-wtm_dt_adaptive -wtm_dt_tol 20"
  do_run spinup adaptrms 2   "$COLD/" 302400  520 1 "-wtm_dt_adaptive -wtm_dt_norm_rms -wtm_dt_tol 2"
fi
echo "ADAPT BENCH DONE ($REGIME)"
