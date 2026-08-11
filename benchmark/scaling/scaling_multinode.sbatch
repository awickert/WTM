#!/bin/bash -l
#SBATCH --job-name=wtm-multinode
#SBATCH --partition=msilarge          # multi-node partition (worked: job 15405885 got acn[129-132]).
                                      # Verify with `sinfo -s` if it changes.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16           # 16 (not RPN) so the correctness check can put 16 ranks on ONE node
#SBATCH --mem=32gb                     # per node
#SBATCH --time=0:40:00
#SBATCH --output=wtm-multinode-%j.out
#
# Multi-node correctness check + node sweep for WTM's groundwater solve (FSM off).
#
# LAUNCHER = mpiexec (MPICH's own Hydra), NOT `srun --mpi=pmi2`. The first attempt
# (job 15405885) had every MULTI-RANK run fail under srun -- 16 ranks/2 nodes aborted
# rc=134 (SIGABRT at startup), 8 ranks/1 node hung to the time limit -- while the
# single-rank run was fine. Same rank counts run fine single-node via mpiexec, and
# BUILD_HPC.md warns PETSc's bundled MPICH does not wire up under Slurm's srun/PMI.
# So this is a launcher issue, not a cross-node code bug. mpiexec reads the Slurm
# nodelist; the PLACEMENT PROBE below confirms whether it actually spans nodes -- if
# it prints only ONE host, mpiexec isn't spanning and we need `-bootstrap slurm`,
# `-hosts $(scontrol show hostnames | paste -sd,)`, or MSI's documented launch.
# Failing runs dump their log tail into this .out so errors survive the temp cleanup.
#
#   sbatch benchmark/scaling/scaling_multinode.sbatch
#   GRID=4000 RPN=8 NODELIST="1 2 4" sbatch scaling_multinode.sbatch
set -uo pipefail

WTM_ROOT="${WTM_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/models/WTM}}"
WTM="$WTM_ROOT/build/wtm.x"
GRID="${GRID:-2000}"; RPN="${RPN:-8}"; NODELIST="${NODELIST:-1 2 4}"
SOLVER="-wtm_anderson -snes_stol 1e-6"

source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
# WORK must be on a SHARED filesystem ($HOME, mounted on all nodes) -- NOT node-local
# /tmp: every node's ranks read the config + synthetic grid, and /tmp is per-node, so
# the first cross-node attempt died "Failed to read config file!" on the other nodes.
WORK="$HOME/.wtm_multinode_${SLURM_JOB_ID:-$$}"; mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
SDIR="$WORK/synth"; mkdir -p "$SDIR"
echo "host $(hostname)  nodes_alloc=${SLURM_JOB_NUM_NODES:-?}  grid=${GRID}^2  ranks/node=$RPN  launcher=mpiexec"

python3 - "$SDIR" "$GRID" "$WTM_ROOT" <<'PY'
import sys, os
sdir, grid, root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.join(root, "benchmark", "scaling"))
import make_synthetic as ms
topo = os.path.join(sdir, "synth_topography.tif")
if not os.path.exists(topo):
    t = ms.build_topography(grid, grid); s = ms.slope_from_topo(t)
    ms.write_tif(topo, t); ms.write_tif(os.path.join(sdir, "synth_slope.tif"), s)
print("grid ready:", topo)
PY

mkcfg() { local cpd; cpd="$(python3 -c "print($GRID/120.0)")"; cat <<EOF
run_type           test
fsm_on             0
evap_mode          0
infiltration_on    0
runoff_ratio_on    0
cells_per_degree   $cpd
southern_edge      -45
deltat             31536000
total_cycles       1
maxiter            5
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        $SDIR
region             synth
supplied_wt        0
textfilename       $1.txt
outfile_prefix     ${1}_
cycles_to_save     9999
EOF
}
gwt() { python3 -c '
import re, sys
try: t = open(sys.argv[1]).read()
except Exception: print(""); sys.exit()
# bounded (require whitespace/end after the number) so an interleaved timestamp
# ("... 2026-08-11 ...") cannot be misparsed as a GW time; sanity-cap at 1e5 s.
v = [float(x) for x in re.findall(r"GW time =\s*(\d+(?:\.\d+)?)(?=\s|$)", t)]
v = [x for x in v if x < 1e5]
print(max(v) if v else "")
' "$1"; }
runwtm() {  # label ppn ntot stem   -> run; on failure dump log tail to stdout(.out)
  local label="$1" ppn="$2" ntot="$3" stem="$4"
  mkcfg "$stem" > "$stem.cfg"
  mpiexec -ppn "$ppn" -n "$ntot" "$WTM" "$stem.cfg" $SOLVER > "$stem.log" 2>&1
  local rc=$?
  [ "$rc" -ne 0 ] && { echo "  $label rc=$rc -- log tail:"; tail -20 "$stem.log" | sed 's/^/    | /'; }
  return "$rc"
}

echo
echo "=== mpiexec placement probe (want $RPN ranks on each of 2 DISTINCT hosts) ==="
mpiexec -ppn "$RPN" -n $((2*RPN)) hostname 2>/dev/null | sort | uniq -c | sed 's/^/  /' || echo "  probe failed -- mpiexec cannot launch"

echo
echo "==================== CORRECTNESS: isolate node-spanning from rank-count noise ===================="
# Matrix-free Anderson is NOT bit-reproducible across rank counts (non-associative
# reductions), so |16-rank - 1-rank| is a NOISE FLOOR, not a bug. The clean test is
# 16-ranks-1-node vs 16-ranks-2-nodes: SAME DMDA decomposition, only node placement
# (and thus which MPI msgs cross the network) differs -> a real cross-node bug shows
# as |2 nodes - 1 node| >> the noise floor.
echo "  [1 rank  / 1 node ]"; runwtm "ref"  1  1  "$WORK/ref"  || true
echo "  [16 ranks / 1 node]"; runwtm "sn16" 16 16 "$WORK/sn16" || true
echo "  [16 ranks / 2 nodes]"; runwtm "mn16" 8  16 "$WORK/mn16" || true
python3 - "$WORK" <<'PY'
import sys, glob
w = sys.argv[1]
def fld(stem):
    fs = sorted(glob.glob(stem + "_*.tif"))
    if not fs: return None
    import rasterio; return rasterio.open(fs[-1]).read(1)
import numpy as np
ref, sn, mn = fld(w+"/ref"), fld(w+"/sn16"), fld(w+"/mn16")
def cmp(a, b): return None if a is None or b is None else float(np.nanmax(np.abs(a-b)))
d_sn, d_mn, d_node = cmp(ref, sn), cmp(ref, mn), cmp(sn, mn)
print(f"  |16-rank(1 node)  - 1-rank|  = {d_sn}  m   (Anderson cross-rank noise floor)")
print(f"  |16-rank(2 nodes) - 1-rank|  = {d_mn}  m")
print(f"  |2 nodes - 1 node| (SAME 16 ranks -- isolates node-spanning) = {d_node}  m")
if d_node is None:
    print("  -> CHECK: a run failed (see log tails above)")
elif d_sn is not None and d_node <= max(1e-6, 2.0*d_sn):
    print("  -> PASS: node-spanning adds no error beyond the single-node cross-rank noise")
else:
    print("  -> CHECK: multi-node differs from single-node BEYOND the rank-count noise -- possible cross-node bug")
PY

echo
echo "==================== NODE SWEEP (fixed $RPN ranks/node, FSM off) ===================="
printf "%6s %7s %10s\n" "nodes" "ranks" "gw_s"
for M in $NODELIST; do
  runwtm "s$M" "$RPN" $((M*RPN)) "$WORK/s$M" || true
  printf "%6s %7s %10s\n" "$M" "$((M*RPN))" "$(gwt "$WORK/s$M.log")"
done
echo
echo "gw_s dropping with node count = added memory channels help; flat/blank = the"
echo "per-cycle gather-to-rank-0 over Infiniband, or a launcher issue (see tails above)."
