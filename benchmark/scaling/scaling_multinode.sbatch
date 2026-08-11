#!/bin/bash -l
#SBATCH --job-name=wtm-multinode
#SBATCH --partition=msilarge          # !! EDIT: a MULTI-NODE partition. Run `sinfo -s` first;
                                      #    Agate multi-node is NOT msismall/msilong. Job won't submit if wrong.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32gb                     # per node
#SBATCH --time=0:40:00
#SBATCH --output=wtm-multinode-%j.out
#
# Multi-node correctness check + node sweep for WTM's groundwater solve (FSM off).
#
# CONTEXT: WTM was DESIGNED single-node (benchmark/DISTRIBUTED_ARP_DESIGN.md lists
# cross-node as a NON-GOAL), but the solve + gather/scatter use PETSc COLLECTIVES
# (node-agnostic), so it SHOULD run cross-node with no code change -- only the
# launcher differs. This is UNTESTED, so we (1) confirm a multi-node run matches a
# single-rank run, then (2) sweep node count at fixed ranks/node to see whether the
# added memory channels speed the GW solve, or whether the per-cycle full-grid
# gather-to-rank-0 over Infiniband is the ceiling.
#
#   sbatch benchmark/scaling/scaling_multinode.sbatch                 # 2000^2, 8 ranks/node, 1/2/4 nodes
#   GRID=4000 RPN=8 NODELIST="1 2 4" sbatch scaling_multinode.sbatch
set -uo pipefail                       # NOT -e: continue past a failed run to collect what we can

WTM_ROOT="${WTM_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/models/WTM}}"
WTM="$WTM_ROOT/build/wtm.x"
GRID="${GRID:-2000}"
RPN="${RPN:-8}"                        # ranks per node (efficient regime)
NODELIST="${NODELIST:-1 2 4}"          # node counts to sweep (each <= #SBATCH --nodes)
LAUNCH="srun --mpi=pmi2"               # MPICH under Slurm, spans nodes
SOLVER="-wtm_anderson -snes_stol 1e-6"

source "$WTM_ROOT/msi_env.sh" test     # modules (srun/mpi) + wtmtest (rasterio)
export OMP_NUM_THREADS=1

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SDIR="$WORK/synth"; mkdir -p "$SDIR"
echo "host $(hostname)  nodes_alloc=${SLURM_JOB_NUM_NODES:-?}  grid=${GRID}^2  ranks/node=$RPN"

# --- synthetic grid (reuse the study's generator so filenames/config match) ------
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

mkcfg() {  # $1 = prefix stem -> writes config to stdout (matches scaling_study.write_cfg)
  local cpd; cpd="$(python3 -c "print($GRID/120.0)")"
  cat <<EOF
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
gwt() { grep -oE "GW time =\s*[0-9.]+" "$1" 2>/dev/null | grep -oE "[0-9.]+" | sort -g | tail -1; }

echo
echo "==================== CORRECTNESS: multi-node vs single-rank ===================="
mkcfg "$WORK/ref" > "$WORK/ref.cfg"
echo "  [1 rank / 1 node]"
$LAUNCH -N 1 -n 1 "$WTM" "$WORK/ref.cfg" $SOLVER > "$WORK/ref.log" 2>&1 || echo "  ref run rc=$?"
mkcfg "$WORK/mn" > "$WORK/mn.cfg"
echo "  [$((2*RPN)) ranks / 2 nodes]"
$LAUNCH -N 2 --ntasks-per-node="$RPN" "$WTM" "$WORK/mn.cfg" $SOLVER > "$WORK/mn.log" 2>&1 || echo "  multi-node run rc=$?"
python3 - "$WORK" <<'PY'
import sys, glob, re
w = sys.argv[1]
def final_tif(stem):
    fs = sorted(glob.glob(stem + "_*.tif")); return fs[-1] if fs else None
rt, mt = final_tif(w+"/ref"), final_tif(w+"/mn")
if rt and mt:
    import rasterio, numpy as np
    a = rasterio.open(rt).read(1); b = rasterio.open(mt).read(1)
    d = float(np.nanmax(np.abs(a-b)))
    print(f"  field compare: max|wtd(multi) - wtd(1-rank)| = {d:.3e} m  ->",
          "PASS (multi-node matches single-node)" if d < 1e-4 else "FAIL -- multi-node DIVERGES")
else:
    def dw(log):
        m = re.findall(r"per-cycle max[^=]*=\s*([0-9.eE+-]+)", open(log).read())
        return float(m[-1]) if m else None
    r, m = dw(w+"/ref.log"), dw(w+"/mn.log")
    ok = (r is not None and m is not None and abs(r-m) < 1e-6*max(1.0, abs(r)))
    print(f"  no output rasters; per-cycle max|dwtd|: 1-rank={r} multi={m}  ->",
          "PASS" if ok else "CHECK ref.log/mn.log (hang or crash cross-node?)")
PY

echo
echo "==================== NODE SWEEP (fixed $RPN ranks/node, FSM off) ===================="
printf "%6s %7s %10s\n" "nodes" "ranks" "gw_s"
for M in $NODELIST; do
  mkcfg "$WORK/s$M" > "$WORK/s$M.cfg"
  $LAUNCH -N "$M" --ntasks-per-node="$RPN" "$WTM" "$WORK/s$M.cfg" $SOLVER > "$WORK/s$M.log" 2>&1 || true
  printf "%6s %7s %10s\n" "$M" "$((M*RPN))" "$(gwt "$WORK/s$M.log")"
done
echo
echo "Read: gw_s should DROP roughly with node count if added memory channels help;"
echo "if it flattens, the per-cycle gather-to-rank-0 over Infiniband is the ceiling."
