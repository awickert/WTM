#!/bin/bash -l
#SBATCH --job-name=kc-adapt-mnode
#SBATCH --partition=msilarge
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64gb
#SBATCH --time=3:00:00
#SBATCH --output=multinode-%j.out
#
# MULTI-NODE DRY RUN of cc-vs-adaptive spin-up (item #2 of the pre-dedicated set): validate the multi-node
# harness, DE-RISK the rank-0 FSM gather at scale, and get multi-node iteration data. NOT clean wall (shared
# msilarge + cross-node -> noisy; that's the --exclusive pass). Allocates 4 nodes, sweeps 2x16=32 and
# 4x16=64 ranks via mpiexec -ppn (Hydra spans nodes; srun is broken for PETSc's bundled MPICH). Sizes
# 2x2 + 4x4 (the 4x4/4-node point is the real gather de-risk). Reuses the tiled domains.
set -uo pipefail
PPN="${PPN:-16}"   # ranks per node; sweep layouts as (nodes x PPN), e.g. PPN=8 NODES_SWEEP="2 4"
NODES_SWEEP="${NODES_SWEEP:-2 4}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"
grep -qa "derived from eq_tol" "$BIN" || { echo "FATAL: stale binary."; exit 1; }
echo "=== placement probe (mpiexec must span $SLURM_JOB_NUM_NODES nodes) ==="
mpiexec -ppn "$PPN" -n $((PPN*SLURM_JOB_NUM_NODES)) hostname 2>/dev/null | sort | uniq -c | sed 's/^/  /' || echo "  PROBE FAILED"
R=results/scaling; mkdir -p "$R"
CSV="$R/scaling_multinode.csv"
[ -f "$CSV" ] || echo "nodes,ranks,tiles,cells,method,rc,wall_s,snes_its,stop_cycle" > "$CSV"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

have() { awk -F, -v nd="$1" -v rk="$2" -v t="$3" -v m="$4" 'NR>1 && $1==nd && $2==rk && $3==t && $5==m{f=1} END{exit !f}' "$CSV"; }
run() { # nodes ranks tiles dom cells method flags...
  local nd="$1" ranks="$2" tiles="$3" dom="$4" cells="$5" meth="$6"; shift 6
  if have "$nd" "$ranks" "$tiles" "$meth"; then echo "skip nodes=$nd ranks=$ranks $tiles $meth (recorded)"; return; fi
  local stem="$R/mn${nd}n${ranks}r_${tiles}_${meth}"
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 200#;s#^cycles_to_save.*#cycles_to_save 200#;s#^supplied_wt.*#supplied_wt 0#;s#^surfdatadir.*#surfdatadir ${dom}/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== nodes=$nd ranks=$ranks ${tiles} (${cells}) ${meth} ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$PPN" -n "$ranks" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  echo "${nd},${ranks},${tiles},${cells},${meth},${rc},${wall},${its},${stop}" | tee -a "$CSV"
}

for nd in $NODES_SWEEP; do
  ranks=$((nd*PPN))
  for t in 2 4; do
    tiles="${t}x${t}"; ny=$((t*451)); nx=$((t*853)); cells=$((ny*nx))
    dom="$R/tiles/${ny}x${nx}"
    [ -f "${dom}/Esquibel_010000_topography.tif" ] || python3 make_tiled.py "$t" "$t" domain "$R/tiles" >/dev/null 2>&1
    run "$nd" "$ranks" "$tiles" "$dom" "$cells" cc
    run "$nd" "$ranks" "$tiles" "$dom" "$cells" adapt -wtm_tr_bdf2 -wtm_dt_adaptive
  done
done
echo "### DONE ###"; cat "$CSV"
