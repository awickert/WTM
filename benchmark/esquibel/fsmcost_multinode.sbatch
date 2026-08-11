#!/bin/bash -l
#SBATCH --job-name=wtm-gather
#SBATCH --partition=msilarge         # multi-node (no msilong 32-CPU/user cap; prior study ran here)
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8          # 8 ranks/node (matches the validated GW-scaling study)
#SBATCH --mem=48gb                   # per node
#SBATCH --time=1:30:00
#SBATCH --output=gather-%j.out
#
# Task B: measure the per-cycle GATHER cost (all-to-one to rank 0) as a function of NODE COUNT,
# coupled (fsm_on 1), at a FIXED large grid. This is the Amdahl term the single-node A pass cannot
# see: as nodes rise the GW solve shrinks but the gather to rank 0 (an Infiniband transfer of the
# full wtd every cycle) does NOT -- so it, plus the serial rank-0 FSM, caps the coupled model.
# Requires the build with the "t gather time =" timer (WTM.cpp). Launcher: mpiexec -ppn (Hydra),
# which spans nodes; srun --mpi=pmi2 is the known-broken path for this MPICH.
#
#   cd ~/models/WTM/benchmark/esquibel && sbatch fsmcost_multinode.sbatch
#   TILE=12x12 PPN=8 NODES_SWEEP="1 2 4 8" NCYC=3 sbatch fsmcost_multinode.sbatch
set -uo pipefail

WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
TILE="${TILE:-6x6}"                  # fixed grid for the node sweep: 6x6=13.8M (gather ~112MB/cyc)
PPN="${PPN:-8}"
NODES_SWEEP="${NODES_SWEEP:-1 2 4 8}"
NCYC="${NCYC:-3}"

source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"

# --- generate the fixed tiled grid once (idempotent) ---
ty="${TILE%x*}"; tx="${TILE#*x}"
python3 make_tiled.py "$ty" "$tx" || { echo "grid gen failed"; exit 1; }
ny=$((ty*451)); nx=$((tx*853)); D="$ESQ/tiled/${ny}x${nx}"; CFG="$D/eq_tiled.cfg"
sed -i "s/^total_cycles.*/total_cycles $NCYC/;s/^cycles_to_save.*/cycles_to_save $NCYC/" "$CFG"
CELLS=$((ny*nx))
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6"

echo "job $SLURM_JOB_ID on $(hostname); grid=$TILE ($CELLS cells) ppn=$PPN sweep=[$NODES_SWEEP] ncyc=$NCYC"
echo "=== placement probe (want PPN ranks on each of 2 distinct hosts) ==="
mpiexec -ppn "$PPN" -n $((2*PPN)) hostname 2>/dev/null | sort | uniq -c | sed 's/^/  /' || echo "  probe FAILED"

CSV="$ESQ/fsmcost_multinode_results.csv"
echo "cells,nodes,ppn,ranks,gw_med_s,gather_med_s,fsm_med_s,total_med_s,job_id" > "$CSV"

for nodes in $NODES_SWEEP; do
  ntot=$((nodes*PPN))
  LOG="$D/gather_N${nodes}_n${ntot}.log"
  echo "=== nodes=$nodes ranks=$ntot ==="
  t0=$(date +%s)
  mpiexec -ppn "$PPN" -n "$ntot" "$BIN" "$CFG" $OF > "$LOG" 2>&1
  rc=$?; t1=$(date +%s)
  echo "  rc=$rc wall=$((t1-t0))s"
  read gw ga fsm tot < <(python3 - "$LOG" <<'PY'
import re, sys, statistics as st
txt = open(sys.argv[1], errors="ignore").read()
def med(label):
    xs=[]
    for m in re.findall(rf"t {label} time = ([0-9.eE+-]+)", txt):
        try:
            v=float(m)
            if v < 1000: xs.append(v)   # strip interleaved year timestamps
        except ValueError: pass
    return st.median(xs) if xs else float("nan")
print(f"{med('GW'):.4f} {med('gather'):.5f} {med('FSM'):.3e} {med('WTM update'):.4f}")
PY
)
  echo "  GW=${gw}s  gather=${ga}s  FSM=${fsm}s  total=${tot}s"
  echo "${CELLS},${nodes},${PPN},${ntot},${gw},${ga},${fsm},${tot},${SLURM_JOB_ID}" >> "$CSV"
done

echo "=== DONE ==="; column -t -s, "$CSV"
echo "Read: gather should stay ~flat (or rise) as nodes climb while GW falls -> the crossover = the"
echo "coupled Amdahl ceiling the parallelize-FSM work (#80) must remove."
