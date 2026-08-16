#!/bin/bash -l
#SBATCH --job-name=kc-adapt-ncore
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120gb
#SBATCH --time=16:00:00
#SBATCH --output=ncore-%j.out
#
# SINGLE-NODE MULTI-CORE scaling of cc-vs-adaptive spin-up: same four grid sizes as the fixed-N=16
# scaling (384k/1.5M/3.5M/6.2M), swept over cores (default 8,4,2). Seeds the existing N=16 rows so the
# master CSV holds cores = {2,4,8,16}. Idempotent (skips rows already recorded) so it can be resumed.
# This is the 1-NODE part of the scaling set and the DRY RUN for the eventual --exclusive sweep (which
# reuses this harness with CORES="32 64 128" and #SBATCH --exclusive). Reuses the tiled domains in
# results/scaling/tiles/. NOTE: shared agsmall node -> wall is noise-limited (iterations are the clean
# signal); the --exclusive pass provides publication-grade wall.
set -uo pipefail
CORES="${CORES:-8 4 2}"
METHODS="${METHODS:-cc fixed_tr adapt}"   # subset with e.g. METHODS="fixed_tr" to avoid colliding with a
                                          # concurrent job doing the other methods
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
R=results/scaling; mkdir -p "$R"
CSV="$R/scaling_ncore.csv"
if [ ! -f "$CSV" ]; then
  echo "cores,tiles,cells,method,rc,wall_s,snes_its,stop_cycle" > "$CSV"
  # seed the fixed-N=16 rows from the earlier scaling run (if present)
  [ -f "$R/scaling.csv" ] && awk -F, 'NR>1{print "16,"$1","$2","$3","$4","$5","$6","$7}' "$R/scaling.csv" >> "$CSV"
fi
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

have() { awk -F, -v c="$1" -v t="$2" -v m="$3" 'NR>1 && $1==c && $2==t && $4==m{f=1} END{exit !f}' "$CSV"; }
run() { # cores tiles dom cells method flags...
  local nc="$1" tiles="$2" dom="$3" cells="$4" meth="$5"; shift 5
  if have "$nc" "$tiles" "$meth"; then echo "skip nc=$nc $tiles $meth (already recorded)"; return; fi
  local stem="$R/nc${nc}_${tiles}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save 200#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${dom}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== nc=$nc ${tiles} (${cells}) ${meth} ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$nc" -n "$nc" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  echo "${nc},${tiles},${cells},${meth},${rc},${wall},${its},${stop}" | tee -a "$CSV"
}

for nc in $CORES; do
  for t in 1 2 3 4; do
    tiles="${t}x${t}"; ny=$((t*451)); nx=$((t*853)); cells=$((ny*nx))
    if [ "$t" -eq 1 ]; then dom="domain"; else
      dom="$R/tiles/${ny}x${nx}"
      [ -f "${dom}/Esquibel_010000_topography.tif" ] || python3 make_tiled.py "$t" "$t" domain "$R/tiles" >/dev/null 2>&1
    fi
    case " $METHODS " in *" cc "*)       run "$nc" "$tiles" "$dom" "$cells" cc ;; esac
    case " $METHODS " in *" fixed_tr "*) run "$nc" "$tiles" "$dom" "$cells" fixed_tr -wtm_tr_bdf2 ;; esac
    case " $METHODS " in *" adapt "*)    run "$nc" "$tiles" "$dom" "$cells" adapt -wtm_tr_bdf2 -wtm_dt_adaptive ;; esac
  done
done
echo "### DONE ###"; sort -t, -k2,2 -k1,1n "$CSV"
