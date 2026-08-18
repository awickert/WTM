#!/bin/bash -l
#SBATCH --job-name=kc-water
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=60gb
#SBATCH --time=0:40:00
#SBATCH --output=water-%j.out
#
# PURE-WATER-DEPTH convergence probe (non-exclusive, quick). Runs cc / fixed_tr / adapt on the base 384k
# Esquibel (single node) with the new -wtm_eq_metric metric so the per-cycle log carries |S·Δwtd| (mm-water)
# for EVERY cycle. Convergence (cycles/iters to a threshold) is mesh-independent, so 384k gives the same
# water-depth trajectory as the 13.85M exclusive runs at a fraction of the cost; wall here is throwaway
# (shared node) -- the numbers we want are per-cycle water max/rms + cumulative iterations. Stops on a tight
# water-rms so each method runs well past 1 mm-water; a 60-cycle cap backstops it.
set -uo pipefail
METHODS="${METHODS:-cc fixed_tr adapt}"
TY="${TY:-1}"; TX="${TX:-1}"   # tile the base domain TY(N-S) x TX(W-E); 1x1 = base 384k (no internal seam)
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "mm-water" "$BIN" || { echo "FATAL: binary lacks the water-depth metric -- rebuild."; exit 1; }
R=results/scaling; mkdir -p "$R"
if [ "$TY" -eq 1 ] && [ "$TX" -eq 1 ]; then DOM="domain"; TAG="1x1"; else
  ny=$((TY*451)); nx=$((TX*853)); DOM="$R/tiles/${ny}x${nx}"; TAG="${TY}x${TX}"
  [ -f "${DOM}/Esquibel_010000_topography.tif" ] || python3 make_tiled.py "$TY" "$TX" domain "$R/tiles" >/dev/null 2>&1
fi
echo "domain=$DOM tag=$TAG methods=$METHODS"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_metric water-rms -wtm_eq_tol 0.0002"
run() { # method flags...
  local meth="$1"; shift
  local stem="$R/water_${TAG}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 60#;s#^cycles_to_save.*#cycles_to_save 60#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${DOM}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== ${meth} (384k Esquibel, water-depth metric) ==="
  mpiexec -n 16 "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  echo "  rc=$? -> ${stem}.log"
}
case " $METHODS " in *" cc "*)       run cc ;; esac
case " $METHODS " in *" fixed_tr "*) run fixed_tr -wtm_tr_bdf2 ;; esac
case " $METHODS " in *" adapt "*)    run adapt -wtm_tr_bdf2 -wtm_dt_adaptive ;; esac
echo "### DONE ###"
