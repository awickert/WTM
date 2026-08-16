#!/bin/bash -l
#SBATCH --job-name=kc-weak-mnode
#SBATCH --partition=msilarge
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=weakmn-%j.out
#
# MULTI-NODE WEAK SCALING -- the geometry that actually predicts production (220M): 16 ranks/node (one rank
# per memory channel), growing domain AND nodes together so each rank keeps EXACTLY 384,703 cells and each
# node brings its own bandwidth pool. Ideal weak scaling = FLAT wall as nodes grow. (The single-node
# scaling_weak.sbatch instead piles all ranks on one node's fixed bandwidth -> it measures saturation, not
# weak scaling; see SCALING.md.) For the --exclusive pass add `#SBATCH --exclusive`.
#
# Ladder entries are "nodes:ny_tiles:nx_tiles" with ny*nx = 16*nodes tiles, so cells/rank == 384,703 exactly:
#   1:4:4  -> 16 ranks,  6.16M   2:4:8 -> 32 ranks, 12.3M   4:8:8 -> 64 ranks, 24.6M   8:8:16 -> 128 ranks, 49.3M
# Extend toward the 220M production point with e.g. 16:16:16 (256 ranks, 98.5M).
set -uo pipefail
PPN="${PPN:-16}"                                       # ranks/node = memory channels/node (bandwidth-matched)
LADDER="${LADDER:-1:4:4 2:4:8 4:8:8 8:8:16}"
METHODS="${METHODS:-cc fixed_tr}"                      # adapt off: no reproducible weak curve under MPI
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
echo "=== placement probe (mpiexec must span $SLURM_JOB_NUM_NODES nodes) ==="
mpiexec -ppn "$PPN" -n $((PPN*SLURM_JOB_NUM_NODES)) hostname 2>/dev/null | sort | uniq -c | sed 's/^/  /' || echo "  PROBE FAILED"
R=results/scaling; mkdir -p "$R"
CSV="$R/scaling_weak_multinode.csv"
[ -f "$CSV" ] || echo "nodes,ranks,ny,nx,cells,cells_per_rank,method,rc,wall_s,snes_its,stop_cycle" > "$CSV"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

have() { awk -F, -v nd="$1" -v m="$2" 'NR>1 && $1==nd && $7==m{f=1} END{exit !f}' "$CSV"; }
run() { # nodes ranks ny nx dom cells method flags...
  local nd="$1" ranks="$2" ny="$3" nx="$4" dom="$5" cells="$6" meth="$7"; shift 7
  if have "$nd" "$meth"; then echo "skip nodes=$nd $meth (recorded)"; return; fi
  local cpr=$((cells/ranks))
  local stem="$R/wkmn${nd}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save 200#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${dom}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== nodes=$nd ranks=$ranks ${ny}x${nx}tiles (${cells}, ~${cpr}/rank) ${meth} ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$PPN" -n "$ranks" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  echo "${nd},${ranks},${ny},${nx},${cells},${cpr},${meth},${rc},${wall},${its},${stop}" | tee -a "$CSV"
}

for entry in $LADDER; do
  nd="${entry%%:*}"; rest="${entry#*:}"; ny="${rest%%:*}"; nx="${rest##*:}"
  ranks=$((nd*PPN))
  [ $((ny*nx)) -eq "$ranks" ] || echo "WARN nodes=$nd: ${ny}x${nx}=$((ny*nx)) tiles != $ranks ranks (cells/rank drifts)"
  gy=$((ny*451)); gx=$((nx*853)); cells=$((gy*gx))
  dom="$R/tiles/${gy}x${gx}"
  [ -f "${dom}/Esquibel_010000_topography.tif" ] || python3 make_tiled.py "$ny" "$nx" domain "$R/tiles" >/dev/null 2>&1
  case " $METHODS " in *" cc "*)       run "$nd" "$ranks" "$ny" "$nx" "$dom" "$cells" cc ;; esac
  case " $METHODS " in *" fixed_tr "*) run "$nd" "$ranks" "$ny" "$nx" "$dom" "$cells" fixed_tr -wtm_tr_bdf2 ;; esac
  case " $METHODS " in *" adapt "*)    run "$nd" "$ranks" "$ny" "$nx" "$dom" "$cells" adapt -wtm_tr_bdf2 -wtm_dt_adaptive ;; esac
done
echo "### DONE ###"; sort -t, -k1,1n "$CSV"
