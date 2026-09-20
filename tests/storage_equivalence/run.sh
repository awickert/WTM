#!/usr/bin/env bash
# SECANT ≡ VOLUME backward-Euler storage equivalence (unit/regression test).
#
# WTM's default backward Euler forms the storage term as S·Δh with the EXACT secant effective storativity
# S = (V(wⁿ⁺¹) − V(wⁿ))/(wⁿ⁺¹ − wⁿ); -wtm_volume_storage uses the stored-volume change ΔV directly. Since S
# is the exact secant, S·Δh ≡ ΔV identically (even across the surface where dV/dh jumps porosity→~1). So on
# a well-behaved (non-oscillating) domain the two must agree to MACHINE PRECISION.
#
# THIS SUITE DOES NOT CURRENTLY PROVE THAT, AND SAYING SO IS THE POINT (#34, 2026-09-11). It used to end
# "-- there is none, and this proves it". It proved nothing: its fixture carried the placeholder
# geotransform, cells were ~111 km, the plateau saturated, and BOTH compared fields were identically zero.
# tests/nonvacuous.py caught it. max|dV| = 0.000e+00 against a 2.5e-07 target, PASS, for as long as it has
# existed.
#
# WHAT THE LIVE MEASUREMENTS SAY. Sweeping cell size (the only knob that moves the field relative to the
# surface), max|dV| in water and how much of the domain is at wtd = 0:
#     cpd     1 / 4 / 16  ->  0.000e+00   88/88 saturated -- vacuous, the old state
#     cpd    64           ->  2.386e-03   56/88 at the surface  <- the S != Sy regime this test is FOR
#     cpd   256           ->  6.121e-05    0/88, wtd -93.6 .. -57.9
#     cpd  1000           ->  5.689e-07    0/88, wtd -99.6 .. -97.0
# The disagreement is largest exactly where the specific yield jumps, and falls off monotonically as the
# field drains away from the surface. It does NOT scale with the solver tolerance -- 2.39e-03, 4.46e-03,
# 2.47e-03 at tol 1e-8, 1e-10, 1e-12 -- so it is not convergence noise. It sits in the 32 UNCLAMPED
# drawdown cells (wtd -62.9 .. -2.8), not the 56 clamped ones, so it is not the collector's clamp either.
#
# WHY IT IS AN XFAIL RATHER THAN A DECLARED DEFECT. The claim above excludes one case: a surface limit
# cycle. At cpd 64 the domain IS in one, so the precondition fails and the identity is UNVERIFIED, not
# disproven.
#
# ANDY, 2026-09-11: "For the #102 tests: note that they are allowed to fail." STANDING, and it means
# what it says -- a red from this suite is NOT a release blocker and is not to be chased on its own
# account. The configuration it needs (`secant`, a table crossing wtd = 0, and no flicker) is one the
# model cannot currently supply, and the collector that flickers is `explicit`, which is not the
# production method. Do not spend time here to turn it green, and do not let it gate anything.
#
# The suite already exits 0 on the UNVERIFIED outcome. The two exit-1 paths below are RATCHETS, not
# failures
# of the identity: one fires if the defect disappears, the other if its size moves. They exist so a
# change announces itself rather than passing quietly, and they stay -- but under the standing above,
# either one is a prompt to read and re-record, not a defect to fix.
#
# BUT THE LIMIT CYCLE IS ITSELF A DEFECT, NOT MERELY AN EXCUSE FOR THIS XFAIL -- see task #103. It does
# not decay: over cycles 5-30 the within-cycle max|dw| wanders between 4e-04 and 1.3e-02 m with no trend,
# and the count of cells moving >1mm SNAPS between 56, 48 and 0, so the 56 surface cells switch
# collectively between free and clamped. It is the COLLECTOR: on this identical fixture at cycle 30,
#     explicit    0.0081 .. 0.0912 m across 56/88 cells, every cycle
#     active_set  4.6e-08 .. 4.4e-07, ZERO cells       <- cured
# Five orders of magnitude. And with the shipped equilibrium stop the run declares convergence INSIDE the
# oscillation -- "equilibrium reached (frac metric) ... stopping at cycle 4 of 30" while that very cycle
# carries 0.0606 m of within-cycle motion across 56 of 88 cells.
#
# Do not read this as "the flicker is an acceptable background condition". Read it as: two defects
# are stacked here, and the outer one (#103) is why the inner one (#102) cannot be measured. The fixture's docstring asks for "coastal cells cross wtd=0 ...
# but the table does not flicker", and measurement says those two are not simultaneously reachable here:
#     collection.method: explicit    crosses the surface, flickers, does not settle in 460 yr
#     collection.method: active_set  REFUSED BY THE MODEL -- dev.storage_form: secant cannot be used with
#                                    it (the constraint needs a b=0 residual path), so the known flicker
#                                    cure is unavailable to the arm that needs it
#     collection.method: off         mounds to +65.7 m and is still filling at the end of the run
#
# THIS MATTERS BEYOND THE TEST. The model's own refusal message for secant x active_set says "NOTE the two
# forms are mathematically identical BY CONSTRUCTION -- S is DEFINED as the secant, so S*dh == dV is an
# identity". It no longer cites this suite as authority: the algebra is exact, the numerics across the
# surface jump are what this suite has never verified.
# A runtime message cites this suite as its authority for a claim this suite has never checked in the
# regime that matters.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/storeq_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work storeq
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head: the model conserves water
# and judges every stopping criterion in water volume (#61/#65). Uniform phi = 0.25 on this fixture, so this
# is the old 1e-6 head bound x0.25 exactly -- the same strictness, correctly labelled.
TOL="${TOL:-2.5e-7}"      # machine-precision agreement expected (observed ~1e-12 head)
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# Both pins below are load-bearing, and neither was here before 26ad949 -- which made this test VACUOUS.
#   time_integration: with auto now resolving anderson -> tr-bdf2, the storage branch of the residual is
#     never reached at all (tr_stage and bdf2v both take precedence), so neither arm would exercise the
#     thing under test.
#   runoff_collector: the default is active_set, which REQUIRES the b=0 volume path, so
#     dev.storage_form: secant with it now aborts by name. explicit is the collector this identity can
#     be measured under.
# THE CONFIG IS A FILE NOW (#83): tests/storage_equivalence/config.yaml, read and edited directly
# rather than translated from legacy key/value lines. Every setting the run resolves to is stated
# there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# The two pins described above are now WRITTEN DOWN in that file rather than passed each run --
# solver.time_integration: backward-euler and surface_water.collection.method: explicit -- which is
# what stops either of them going missing again and taking the test's subject with it.
emit() { # $1 stem  [env: STORAGE=volume|secant]
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  storage_form: secant|  storage_form: ${STORAGE:-secant}|" config.yaml > "$WORK/$1.yaml"
}

# mask-aware ghost boundary (now the default) so edge land cells are not forced against a
# hard-draining topo-0 ocean pad. Under the old padding, boundary-adjacent SURFACE cells drain so hard that
# the ÷S (secant) vs ÷Sy (tangent) residual scaling leaves a ~1e-4 convergence-region difference there --
# a scaling/conditioning artifact, NOT the identity failing. The ghost boundary removes that edge stress so
# the S·Δh ≡ ΔV identity shows at machine precision (observed ~1e-15) and the test is a clean invariant check.
# STORAGE=secant is now EXPLICIT. It used to be left unset, relying on secant being the default -- and
# when dev.storage_form's default became volume (879a188), that made this test compare volume against
# volume, i.e. a config against ITSELF. It still reported max|dwtd| = 0.000e+00 and still PASSED, which
# is exactly what a vacuous arm looks like.
STORAGE=secant emit secant; STORAGE=volume emit volume
"$WTM" "$WORK/secant.yaml"                     > "$WORK/secant.log" 2>&1 \
  || { echo "RUN FAILED: secant"; tail -3 "$WORK/secant.log"; exit 2; }
"$WTM" "$WORK/volume.yaml" > "$WORK/volume.log" 2>&1 \
  || { echo "RUN FAILED: volume"; tail -3 "$WORK/volume.log"; exit 2; }

# NON-VACUITY GATE. This test compares two runs and asserts they AGREE, so it passes trivially if the
# two arms are the same run -- and that is exactly what happened when dev.storage_form's default became
# volume (879a188) while the secant arm relied on the default: it compared volume against volume,
# reported max|dwtd| = 0.000e+00, and PASSED. An agreement test must prove its arms differ before its
# agreement means anything. Checked on the emitted CONFIGS, which is where the failure actually was.
SEC_FORM=$(grep -oE "storage_form: [a-z]+" "$WORK/secant.yaml" | head -1)
VOL_FORM=$(grep -oE "storage_form: [a-z]+" "$WORK/volume.yaml" | head -1)
echo "  arms: secant -> ${SEC_FORM:-<unset, i.e. the DEFAULT>}   volume -> ${VOL_FORM:-<unset, i.e. the DEFAULT>}"
if [ "$SEC_FORM" != "storage_form: secant" ] || [ "$VOL_FORM" != "storage_form: volume" ]; then
    echo "FAIL: the two arms are not the two forms -- this test would be VACUOUS." >&2
    echo "      Each arm must set dev.storage_form EXPLICITLY; relying on the default is what made it" >&2
    echo "      compare a configuration against itself once that default changed." >&2
    exit 1
fi

SEC=$(ls "$WORK"/secant_*.tif | tail -1)
VOL=$(ls "$WORK"/volume_*.tif | tail -1)
TOL="$TOL" PHI="$INP/storeq_porosity.tif" TESTS="$(readlink -f ..)" "$PY" - "$SEC" "$VOL" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

sec, vol = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(sec, bool); m[:, 0] = False   # interior + land edges (exclude ocean column)
phi = VOL.read_band(os.environ["PHI"])
d = float(VOL.volume_diff(sec, vol, phi)[m].max())
tol = float(os.environ["TOL"])
land = sec[:, 1:]
print(f"  secant-BE vs volume-BE (S·Δh ≡ ΔV): max|ΔV| = {d:.3e} m water volume  (target {tol})")
print(f"  field: wtd {land.min():.4f} .. {land.max():.4f} m, {int((land == 0.0).sum())}/{land.size} cells at the surface")

# XFAIL, WITH A GUARD. See the long note above for the measurements; the short version is that this
# fixture CANNOT currently satisfy the two preconditions its own docstring sets -- cells crossing wtd=0
# AND a non-flickering table -- so the identity is NOT verified in the regime where S != Sy.
MOVED_FLOOR = 1.0e-4   # my choice: an order below the smallest crossing-regime value measured (2.39e-03)
if d <= tol:
    print("UNEXPECTED PASS: the identity now holds in the surface-crossing regime.")
    print("  This is the outcome this suite is waiting for -- but do not just delete the guard. Re-read the")
    print("  note above, confirm the table is genuinely non-flickering (within-cycle max|dw| at the surface")
    print("  cells, not just per-cycle), and record what changed. Failing so it cannot pass unnoticed.")
    sys.exit(1)
if d < MOVED_FLOOR:
    print(f"FAIL: {d:.3e} m is below the ratchet floor {MOVED_FLOOR:g} but above the target {tol}.")
    print("  The defect has MOVED. Re-measure rather than re-tune the floor.")
    sys.exit(1)
print(f"  unverified  NOT ASKED: {d:.3e} m vs target {tol} m -- this suite does not exercise S·Δh vs ΔV\n              in the surface-crossing regime at all, so the identity is neither confirmed nor denied")
print("  The identity is UNVERIFIED here, not disproven: the table flickers, which is the documented")
print("  exception. Quiet cell sizes put every cell far from the surface, where S != Sy is not exercised.")
sys.exit(0)
PY
