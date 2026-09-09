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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work fscons
TOL="${TOL:-1e-4}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

../emit_config.sh > "$WORK/c.yaml" <<EOF
solver_method anderson
run_type equilibrium
total_time 24yr
supplied_wt 1
deltat 31536000
report_interval 2
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector active_set  # was: implicit + the retired -wtm_active_set flag
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
eq_tol 0
# STATED, not inherited. These four are DERIVED by the model from solver.method and the collector, so
# leaving them out meant this suite's configs described a run they did not choose: `auto` resolves them
# and the config recorded whatever came back. They are pinned here at the values this arm has always
# run, so the measurement is unchanged and the choice is now visible where the arm is read.
time_integration tr-bdf2
time_step_mode adaptive
dt_tol 0.5
textfilename $WORK/c.txt
outfile_prefix $WORK/c_
EOF
# FIRST ARM pinned to `impulse`. It used to get impulse for free, as the default; `continuous` is the
# default now, so WITHOUT this pin both arms would run continuous and every comparison below would be
# vacuous -- NON-VACUOUS measured exactly 0.000e+00 the moment the default flipped. Pin it rather than
# lean on the default: what this test compares is the two couplings, and that has to stay true whichever
# one the model happens to ship.
sed -i -e "s|^  mode: routed|  mode: routed\n  fsm_coupling: impulse|" "$WORK/c.yaml"
"$WTM" "$WORK/c.yaml" > "$WORK/c.log" 2>&1 \
  || { echo "RUN FAILED"; tail -5 "$WORK/c.log"; exit 2; }

# SECOND ARM: the same physical problem under the OTHER FSM coupling. `impulse` overwrites the water
# table with FSM's result between steps; `continuous` instead feeds FSM's per-cell volume change into
# the NEXT step's source term. The two integrate differently and reach different states -- which is the
# point, and what makes the comparison below non-vacuous.
sed -e "s|$WORK/c.txt|$WORK/s.txt|" -e "s|$WORK/c_|$WORK/s_|" \
    -e "s|^  fsm_coupling: impulse|  fsm_coupling: continuous|" "$WORK/c.yaml" > "$WORK/s.yaml"
"$WTM" "$WORK/s.yaml" > "$WORK/s.log" 2>&1 \
  || { echo "SOURCE-COUPLING RUN FAILED"; tail -5 "$WORK/s.log"; exit 2; }

TIF=$(ls "$WORK"/c_*.tif | tail -1)
TOL="$TOL" TESTS="$(readlink -f ..)" "$PY" - "$WORK/c.txt" "$TIF" "$WORK/s.txt" <<'PY'
import sys, os, numpy as np, rasterio
txt, tif, txt_src = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(os.environ["TOL"])
sys.path.insert(0, os.environ["TESTS"])
import wtm_log as LOG               # columns BY NAME; see tests/log_schema
I    = LOG.index_map(txt)
rows = [l.split() for l in open(txt) if l and l[0].isdigit()]
# cols (1-indexed): 9 recharge(cum), 16 budget_residual(cum)
R    = np.array([float(r[I["total_recharge_added"]])  for r in rows])
resid= np.array([float(r[I["budget_residual"]]) for r in rows])
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

# EXTERNAL INPUT IS COUPLING-INDEPENDENT.
#
# Column 19 (recharge_direct, summed into column 9) is DEFINED in benchmark/WATER_BUDGET.md as the
# EXTERNAL water entering the domain. External means external: how FSM's water is handed back to the
# groundwater -- overwritten between steps, or fed in as a source term -- is an INTERNAL redistribution
# and cannot change how much water crossed the domain boundary. So the two arms must agree on column 19
# even though they disagree about almost everything else.
#
# THE DEFECT THIS CATCHES. fsm_coupling: continuous used to fold FSM's per-cell delta into rech_dist, the
# same array set_starting_values books as total_recharge_direct. The external columns then reported
# external input PLUS internal redistribution: on tests/fsm_consistency at 120 yr, cumulative column 9
# ran to -6.34e10 by cycle 1 -- a negative cumulative external input -- and everything derived from it
# (ocean_loss_closing, column 16) was wrong with it. Fixed by giving the delta its own carrier.
#
# TOLERANCE, and it is a choice worth naming: 1e-6 relative. The defect drives a relative difference of
# order 1 (sign flip), so this is ~6 orders of magnitude clear of it, while leaving room for a fixture
# whose recharge is genuinely state-dependent -- the open-water-evaporation branch means a different
# water table can draw a different P-ET. On THIS fixture the two arms agree EXACTLY (0.0e+00 at every
# cycle), so the tolerance is headroom, not slack being consumed.
rows_s = [l.split() for l in open(txt_src) if l and l[0].isdigit()]
R19    = np.array([float(r[I["recharge_direct"]]) for r in rows])
R19s   = np.array([float(r[I["recharge_direct"]]) for r in rows_s])
S14    = np.array([float(r[13]) for r in rows])
S14s   = np.array([float(r[13]) for r in rows_s])
n = min(len(R19), len(R19s))
rel19 = float(np.max(np.abs(R19s[:n] - R19[:n]) / np.where(np.abs(R19[:n]) > 0, np.abs(R19[:n]), 1.0)))
check("EXTERNAL INPUT coupling-independent (col 19)", rel19 < 1e-6,
      f"max relative difference impulse vs continuous = {rel19:.3e} (< 1e-6)")

# NON-VACUITY. The check above is only meaningful if the two arms are actually different runs. If a
# future change made the couplings converge to the same trajectory, column 19 would match trivially and
# the assertion would pass while testing nothing. Require the STATES to differ materially.
state_gap = float(np.max(np.abs(S14s[:n] - S14[:n])) / max(np.max(np.abs(S14[:n])), 1.0))
check("NON-VACUOUS (the two couplings really differ)", state_gap > 1e-3,
      f"max |d stored_volume| / |impulse| = {state_gap:.3e} (> 1e-3)")

# ABSOLUTE CLOSURE, and the reason a per-cycle check could not stand in for it.
#
# The CONSERVATION check above differences the cumulative residual between consecutive cycles. That is
# deliberately blind to a CONSTANT offset -- this file's own header calls the offset a startup constant
# and normalises it away -- so an error that enters once, at the start, and then simply sits there is
# invisible to it. One did. The budget baseline (stored_volume_initial) was captured on the first
# PrintValues call, i.e. at the END of cycle 0, while every flux accumulator starts AT cycle 0, so the
# first cycle's storage change was missing from d_stored. On tests/fsm_consistency at 120 yr that was
# 34.84% of recharge under the impulse coupling, and the per-cycle check passed throughout.
#
# Hence this: the residual must be small in ABSOLUTE terms at the end of the run, not merely steady.
#
# TOLERANCE, named because it is a choice: 1e-2 of cumulative recharge, on the OVERWRITE arm at the end
# of the run. Measured here at 2.08e-3, so ~5x headroom; with the baseline defect present it exceeds
# this by orders of magnitude. WATER_BUDGET.md section 4 puts the spun-up expectation near 4e-4, which
# this 24 yr run is too short to reach.
#
# ONLY THE OVERWRITE ARM IS GATED, and that is a deliberate limit rather than an oversight.
# fsm_coupling: continuous hands FSM's volume change to the NEXT step's source term, so at any report
# boundary there is water FSM has already moved -- it is in stored_volume -- whose source term has not
# yet been applied. That in-flight lag makes the source arm's closure large early and decay with run
# length: 7.37e-01 at the end of THIS 24 yr run against 1.4e-04 at 120 yr on the same fixture.
# HYPOTHESIS, NOT VERIFIED: that the lag is exactly one step and therefore O(dt). Until someone
# measures its dt-scaling, gating the source arm here would pin a number nobody has explained.
R9    = np.array([float(r[8])  for r in rows])
resid = np.array([float(r[15]) for r in rows])
closure = abs(float(resid[-1])) / float(R9[-1]) if R9[-1] > 0 else float("inf")
check("ABSOLUTE CLOSURE (impulse arm, end of run)", closure < 1e-2,
      f"|budget_residual|/recharge = {closure:.3e} (< 1e-2)")

print("PASS: FSM path conserves water per cycle and keeps the lake" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
