#!/bin/bash -l
#SBATCH --job-name=kc-adapt-scaling
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=120gb
#SBATCH --time=10:00:00
#SBATCH --output=scaling-%j.out
#
# SCALING of the cc-vs-adaptive spin-up with DOMAIN SIZE. Tile the real Esquibel stack 1x1/2x2/3x3/4x4
# (384k -> 6.2M cells), run cc (fixed 1-wk) AND one-knob adaptive to equilibrium at each, FIXED N=16 ranks
# so wall shows the O(N) trend. Record SNES iters (rank-independent), wall, stop cycle. Tiled = replicated
# real topo (valid compute-cost domain; not a science domain -- seams + repeated relief).
set -uo pipefail
N="${N:-16}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
R=results/scaling; mkdir -p "$R"
CSV="$R/scaling.csv"; echo "tiles,cells,method,rc,wall_s,snes_its,stop_cycle" > "$CSV"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

run() { # tiles domdir cells method extra_flags...
  local tiles="$1" dom="$2" cells="$3" meth="$4"; shift 4
  local stem="$R/${tiles}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save 200#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${dom}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== ${tiles} (${cells} cells) ${meth} ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$N" -n "$N" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  echo "${tiles},${cells},${meth},${rc},${wall},${its},${stop}" | tee -a "$CSV"
}

for t in 1 2 3 4; do
  tiles="${t}x${t}"; ny=$((t*451)); nx=$((t*853)); cells=$((ny*nx))
  if [ "$t" -eq 1 ]; then dom="domain"; else
    dom="$R/tiles/${ny}x${nx}"
    if [ ! -f "${dom}/Esquibel_010000_topography.tif" ]; then
      echo "### tiling ${t}x${t} -> ${ny}x${nx} (${cells} cells) ###"
      python3 make_tiled.py "$t" "$t" domain "$R/tiles" >/dev/null 2>&1 || { echo "tiling ${t}x${t} FAILED"; continue; }
    fi
  fi
  run "$tiles" "$dom" "$cells" cc
  run "$tiles" "$dom" "$cells" adapt -wtm_tr_bdf2 -wtm_dt_adaptive
done
echo "### SCALING SUMMARY ###"; cat "$CSV"
