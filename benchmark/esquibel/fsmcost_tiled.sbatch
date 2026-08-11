#!/bin/bash -l
#SBATCH --job-name=wtm-fsmcost
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=2:30:00
#SBATCH --output=fsmcost-%j.out
#
# Single-node coupled (fsm_on 1) FSM+DH cost pass at increasing grid sizes, to extend the
# FSM/DH-vs-N fraction curve toward the ~141M single-node ceiling (#78). Real topography via
# TILING the 384,703-cell Esquibel stack (make_tiled.py) -- real depressions in every tile.
# Extracts, per size: GW time/cyc (SNESSolve), FSM time/cyc (rank-0 FillSpillMerge), and the
# one-time DH build ("Outlets found in"). The point: FSM+DH stay a small fraction while DH-build
# grows O(N log N) but is one-time/amortized (GLOBAL_SCALING_DESIGN.md: global is a MEMORY wall,
# not a compute one). fsm compute is not the parallelize-FSM driver -- see FSM_COST.md.
#
#   cd ~/models/WTM/benchmark/esquibel && sbatch fsmcost_tiled.sbatch
#   TILES="2x2 6x6 12x12" NRANKS=64 NCYC=3 sbatch fsmcost_tiled.sbatch   # override
set -uo pipefail

WTM_ROOT="${WTM_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/models/WTM}/../..}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
# tile specs "ty,tx" -- 384703 cells * ty*tx: 2x2=1.5M 6x6=13.8M 12x12=55M 19x19=139M
TILES="${TILES:-2x2 6x6 12x12 19x19}"
NRANKS="${NRANKS:-32}"
NCYC="${NCYC:-3}"

source "$WTM_ROOT/msi_env.sh" test     # modules (-> mpiexec) + wtmtest conda (-> rasterio)
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"

cd "$ESQ"
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6"
CSV="$ESQ/fsmcost_tiled_results.csv"
echo "grid_cells,ncells_x,ncells_y,nranks,ncyc,dh_build_s,fsm_med_s,gw_med_s,total_med_s,fsm_frac_pct,job_id" > "$CSV"
echo "job $SLURM_JOB_ID on $(hostname); tiles=[$TILES] nranks=$NRANKS ncyc=$NCYC"

for spec in $TILES; do
  ty="${spec%x*}"; tx="${spec#*x}"
  echo "=================== TILE $spec (${ty}x${tx}) ==================="
  # generate the tiled stack (idempotent: make_tiled writes tiled/<ny>x<nx>/)
  python3 make_tiled.py "$ty" "$tx" || { echo "gen failed $spec"; continue; }
  ny=$((ty*451)); nx=$((tx*853)); D="$ESQ/tiled/${ny}x${nx}"
  CFG="$D/eq_tiled.cfg"
  sed -i "s/^total_cycles.*/total_cycles $NCYC/;s/^cycles_to_save.*/cycles_to_save $NCYC/" "$CFG"
  LOG="$D/run_n${NRANKS}.log"
  echo "run: mpiexec -n $NRANKS wtm.x ($((ny*nx)) cells, $NCYC cyc)"
  t0=$(date +%s)
  mpiexec -n "$NRANKS" "$BIN" "$CFG" $OF -log_view > "$LOG" 2>&1
  rc=$?; t1=$(date +%s)
  echo "  rc=$rc wall=$((t1-t0))s"
  # --- parse (median over cycles; drop year-artifact values >= 1000) ---
  read dh fsm gw tot frac < <(python3 - "$LOG" "$((ny*nx))" <<'PY'
import re, sys, statistics as st
log, cells = sys.argv[1], int(sys.argv[2])
txt = open(log, errors="ignore").read()
def vals(label):
    out=[]
    for m in re.findall(rf"t {label} time = ([0-9.eE+-]+)", txt):
        try:
            v=float(m)
            if v < 1000: out.append(v)   # strip interleaved year timestamps
        except ValueError: pass
    return out
def med(xs): return st.median(xs) if xs else float("nan")
gw, fsm, tot = med(vals("GW")), med(vals("FSM")), med(vals("WTM update"))
dhm = re.search(r"Outlets found in = ([0-9.]+) s", txt)
dh = float(dhm.group(1)) if dhm else float("nan")
frac = 100*fsm/tot if tot and tot==tot and tot>0 else float("nan")
print(f"{dh:.3f} {fsm:.3e} {gw:.4f} {tot:.4f} {frac:.5f}")
PY
)
  echo "  DH_build=${dh}s (once)  FSM_med=${fsm}s  GW_med=${gw}s  total=${tot}s  FSM_frac=${frac}%"
  echo "${ny}*${nx},${nx},${ny},${NRANKS},${NCYC},${dh},${fsm},${gw},${tot},${frac},${SLURM_JOB_ID}" >> "$CSV"
done

echo "=================== DONE ==================="
column -t -s, "$CSV"
