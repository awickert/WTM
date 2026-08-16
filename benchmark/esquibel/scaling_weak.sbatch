#!/bin/bash -l
#SBATCH --job-name=kc-weak
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=120gb
#SBATCH --time=16:00:00
#SBATCH --output=weak-%j.out
#
# WEAK-SCALING dry run: fixed cells/rank (one Esquibel tile per rank), growing domain and ranks together.
# For a t x t tiling we use ranks = t*t, so EVERY run carries exactly 384,703 cells/rank -- ideal weak
# scaling shows FLAT wall as t grows. Complements scaling_ncore.sbatch (strong scaling: fixed size, more
# cores). Single-node dry run on shared agsmall -> wall noise-limited (iterations are the clean signal;
# at fixed cells/rank they should also be ~flat). The --exclusive pass reuses this with the larger t
# spanning nodes for the publication weak-scaling curve. Reuses the tiled domains in results/scaling/tiles/.
#
# Default ladder: t = 1..6 -> ranks 1,4,9,16,25,36 (all fit one 128-core node). adapt is OFF by default:
# its controller is non-deterministic under MPI (see SCALING.md), so it does not give a reproducible weak
# curve; add it with METHODS="cc fixed_tr adapt" only to bracket the spread.
set -uo pipefail
TVALS="${TVALS:-1 2 3 4 5 6}"          # t -> t x t tiling, ranks = t*t, cells/rank held at one tile
METHODS="${METHODS:-cc fixed_tr}"      # subset e.g. METHODS="fixed_tr"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
R=results/scaling; mkdir -p "$R"
CSV="$R/scaling_weak.csv"
[ -f "$CSV" ] || echo "tiles,cells,ranks,cells_per_rank,method,rc,wall_s,snes_its,stop_cycle" > "$CSV"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

have() { awk -F, -v t="$1" -v m="$2" 'NR>1 && $1==t && $5==m{f=1} END{exit !f}' "$CSV"; }
run() { # tiles dom cells ranks method flags...
  local tiles="$1" dom="$2" cells="$3" ranks="$4" meth="$5"; shift 5
  if have "$tiles" "$meth"; then echo "skip $tiles $meth (recorded)"; return; fi
  local cpr=$((cells/ranks))
  local stem="$R/wk${tiles}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save 200#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${dom}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== ${tiles} (${cells}) ranks=${ranks} (~${cpr}/rank) ${meth} ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$ranks" -n "$ranks" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  echo "${tiles},${cells},${ranks},${cpr},${meth},${rc},${wall},${its},${stop}" | tee -a "$CSV"
}

for t in $TVALS; do
  tiles="${t}x${t}"; ny=$((t*451)); nx=$((t*853)); cells=$((ny*nx)); ranks=$((t*t))
  if [ "$t" -eq 1 ]; then dom="domain"; else
    dom="$R/tiles/${ny}x${nx}"
    [ -f "${dom}/Esquibel_010000_topography.tif" ] || python3 make_tiled.py "$t" "$t" domain "$R/tiles" >/dev/null 2>&1
  fi
  case " $METHODS " in *" cc "*)       run "$tiles" "$dom" "$cells" "$ranks" cc ;; esac
  case " $METHODS " in *" fixed_tr "*) run "$tiles" "$dom" "$cells" "$ranks" fixed_tr -wtm_tr_bdf2 ;; esac
  case " $METHODS " in *" adapt "*)    run "$tiles" "$dom" "$cells" "$ranks" adapt -wtm_tr_bdf2 -wtm_dt_adaptive ;; esac
done
echo "### DONE ###"; sort -t, -k2,2n "$CSV"
