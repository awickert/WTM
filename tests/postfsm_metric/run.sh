#!/usr/bin/env bash
# THE PER-CYCLE EQUILIBRIUM METRIC MUST BE COMPUTED FROM POST-FSM STATE, ON EVERY PATH.
#
# Andy's rule (#103): "The only valid states are immediately after FSM is run; within-cycle motion is
# computational but not physical." The per-cycle metric is not a diagnostic -- it TERMINATES runs, so
# it does not describe the answer, it SELECTS which state becomes the answer. Computed from a pre-FSM
# state it stops the run on a state that does not physically exist.
#
# WHY THIS SUITE EXISTS RATHER THAN A COMMENT. This has now gone wrong twice, and the second time a
# comment asserting it was right is what hid it:
#   #106 (2026-09-16) found the SERIAL path measuring pre-FSM, fixed that path, and left a note saying
#        the distributed path was already post-FSM.
#   2026-09-17  the note was FALSE, and the distributed path is the PRODUCTION one --
#        distribute_recharge = !fsm_on || !infiltration_on, so the serial branch is the RARE case
#        (FSM *and* infiltration on) and the shipped default took the unfixed path. Measured:
#            cycle 0   logged 7.738868e-02   pre-FSM pair 7.738868e-02   post-FSM pair 5.216603e+00
#        Digit-for-digit with pre-FSM; 67x from the physical state. Fixed by DELETING the branch.
#
# So the property is now asserted by measurement on BOTH paths, which is the check #106 never ran on
# the path it did not touch. The arms differ in ONE key -- infiltration_during_flow -- because that is
# exactly what selects the path, and routing is held at `impulse` so nothing else moves.
#
# HOW IT IS MEASURED, without trusting any model-side label: the run writes BOTH raster series
# (output.extra_rasters.post_groundwater), so the two states are visible from outside. The logged
# abs_change_volume_max is then compared against max|S*dwtd| recomputed from consecutive ORDINARY
# (post-FSM) snapshots and from consecutive postgw (pre-FSM) ones. It must match the first.
#
# AND THE COMPARISON MUST NOT BE VACUOUS. Late in a converging run the two series coincide, so a run
# where they never differ would pass this test while asserting nothing -- the #34 failure mode. The
# suite therefore REQUIRES at least one cycle where the two pairs differ by more than DIFF_MIN, and
# fails if it cannot find one.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
PY="${PY:-python3}"
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_vertical_ksat.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work postfsm
export OMP_NUM_THREADS=1

# TOL: the logged metric and the recomputed one are the SAME arithmetic on the same doubles, so they
# agree to rounding. This is a reproduction tolerance, not a physics one.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, SEPARATING, and the assertion line carries both edges itself: the logged
#   metric matches the POST-FSM recomputation to 4.445e-12 m (distributed) and 3.458e-12 m
#   (serial), while against the PRE-FSM state the same comparison gives 3.585e+00 and 3.615e+00 m.
#   Twelve orders separate reading the right state from reading the wrong one. The bound sits
#   inside that gap with 225x of headroom above the matching pair -- and the pre-FSM number is
#   exactly the defect #106 fixed, so the broken edge is measured history.
TOL="${TOL:-1e-9}"
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DIFF_MIN: how far apart the two STATES must get on at least one cycle for the comparison to be able
# to discriminate. Measured on this fixture: the two differ by O(1) m early in the run.
DIFF_MIN="${DIFF_MIN:-1e-3}"

emit() { # $1 stem, $2 infiltration_during_flow
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  infiltration_during_flow: true|  infiltration_during_flow: $2|" \
      config.yaml > "$WORK/$1.yaml"
}
run() { # stem  infiltration
  emit "$1" "$2"
  "$WTM" "$WORK/$1.yaml" > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -5 "$WORK/$1.log"; exit 2; }
}
echo "=== per-cycle metric must be POST-FSM on both paths ==="
run distributed false   # distribute_recharge = true  -- the PRODUCTION path
run serial      true    # distribute_recharge = false -- FSM and infiltration together
echo

TESTS="$(readlink -f ..)" TOL="$TOL" DIFF_MIN="$DIFF_MIN" PHI="$INP/fsm_test_porosity.tif" \
  MASK="$INP/fsm_test_t0_mask.tif" "$PY" - "$WORK" <<'PYEOF'
import glob, os, re, sys
import numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL
import wtm_log as L

WORK = sys.argv[1]
TOL, DIFF_MIN = float(os.environ["TOL"]), float(os.environ["DIFF_MIN"])
phi  = VOL.read_band(os.environ["PHI"])
mask = rasterio.open(os.environ["MASK"]).read(1) > 0

def series(stem, pat):
    fs = sorted(glob.glob(os.path.join(WORK, pat)),
                key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
    return [rasterio.open(p).read(1).astype(float) for p in fs]

def dvol(a, b):
    return float(np.nanmax(np.where(mask, VOL.volume_diff(a, b, phi), np.nan)))

fail = 0
def check(name, ok, detail):
    global fail
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok: fail = 1

for stem in ("distributed", "serial"):
    post = series(stem, f"{stem}_0*.tif")
    pre  = series(stem, f"{stem}_postgw_*.tif")
    log  = L.read_log(os.path.join(WORK, f"{stem}.txt"))
    logged = log.col("abs_change_volume_max")
    n = min(len(post), len(pre), len(logged) + 1)
    worst_post, worst_pre, spread = 0.0, 0.0, 0.0
    for c in range(n - 1):
        lv = float(logged[c])
        dpost, dpre = dvol(post[c+1], post[c]), dvol(pre[c+1], pre[c])
        worst_post = max(worst_post, abs(dpost - lv))
        worst_pre  = max(worst_pre,  abs(dpre  - lv))
        spread     = max(spread, abs(dpost - dpre))
    check(f"{stem}: DISCRIMINATING (the two states actually differ somewhere)",
          spread > DIFF_MIN,
          f"max |post-FSM - pre-FSM| over the run = {spread:.4e} m (needs > {DIFF_MIN:g};"
          f" below it this comparison asserts nothing)")
    check(f"{stem}: metric matches POST-FSM",
          worst_post <= TOL,
          f"max |logged - recomputed(post-FSM)| = {worst_post:.3e} m (tol TOL={TOL:g});"
          f" against pre-FSM it is {worst_pre:.3e}")

print()
print("POST-FSM METRIC: ALL PASSED" if not fail else "POST-FSM METRIC: FAILED")
sys.exit(fail)
PYEOF
rc=$?
exit $rc
