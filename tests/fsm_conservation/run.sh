#!/usr/bin/env bash
# FSM water-conservation + lake-persistence quality gate.
#
# On a fixture with a PERSISTENT LAKE and evaporation (fsm_test, mode: lakes), with FSM on every step and the
# lake-aware active-set skim, this asserts the two properties any correct FSM path must have -- and that the
# FSM-acceleration work (#122, pruning the overflow walk) must preserve:
#   CONSERVATION : the per-cycle water balance closes. With the evap term in the budget (irf.cpp), the
#                  cumulative budget_residual (col 16) is a CONSTANT startup offset, so its change between
#                  consecutive cycles is ~0 -- i.e. no water is created or destroyed per cycle. Asserted as
#                  |Δ budget_residual| / recharge < TOL over the last several cycles.
#   LAKE PERSISTS : the lake keeps its head (max wtd > 0), i.e. the skim did not flatten it to the ground.
#
# NOTE (adequacy): conservation catches water CREATED/DESTROYED but not MISPLACED. When the FSM prune lands,
# its correctness gate is additionally accelerated-FSM == full-FSM (the water-table field), added there.
#
# Usage:  tests/fsm_conservation/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/fscons_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-4}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

../emit_config.sh > "$WORK/c.yaml" <<EOF
run_type equilibrium
total_time 24yr
supplied_wt 1
deltat 31536000
report_interval 2
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector implicit
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/c.txt
outfile_prefix $WORK/c_
EOF
"$WTM" "$WORK/c.yaml" -wtm_anderson -wtm_active_set > "$WORK/c.log" 2>&1 \
  || { echo "RUN FAILED"; tail -5 "$WORK/c.log"; exit 2; }

TIF=$(ls "$WORK"/c_*.tif | tail -1)
TOL="$TOL" "$PY" - "$WORK/c.txt" "$TIF" <<'PY'
import sys, os, numpy as np, rasterio
txt, tif = sys.argv[1], sys.argv[2]
tol = float(os.environ["TOL"])
rows = [l.split() for l in open(txt) if l and l[0].isdigit()]
# cols (1-indexed): 9 recharge(cum), 16 budget_residual(cum)
R    = np.array([float(r[8])  for r in rows])
resid= np.array([float(r[15]) for r in rows])
# per-cycle conservation = change in the cumulative residual, normalised by the cycle's recharge increment
dresid = np.abs(np.diff(resid))
dR     = np.abs(np.diff(R))
rel = dresid[-5:] / np.where(dR[-5:] > 0, dR[-5:], 1.0)
worst = float(rel.max())
lake = float(rasterio.open(tif).read(1).astype(float)[1:-1, 1:-1].max())
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("CONSERVATION (per-cycle balance closes)", worst < tol,
      f"max |Δbudget_residual|/Δrecharge over last 5 cycles = {worst:.3e} (< {tol})")
check("LAKE PERSISTS (head kept)", lake > 1.0,
      f"max wtd = {lake:.4f} m")
print("PASS: FSM path conserves water per cycle and keeps the lake" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
