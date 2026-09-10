#!/usr/bin/env bash
# runoff_collector selector: the input-file choice of how the wtd<=0 exfiltration constraint is enforced. One boundary
# condition, three enforcements (see benchmark/SURFACE_WATER_ROUTING.md):
#   implicit : in-residual exfiltration (direct_to_runoff) -- pins wtd=0, dt-independent, exact (Anderson today).
#   explicit : post-solve clamp -- robust on every solver, a dt-lagged form of the same face.
#   off      : no collection -- above-surface water piles up (NONPHYSICAL; warns).
#   extended_soil : also piles, but continues the AQUIFER above the surface (storativity stays porosity,
#              T never clamps), so it removes the wtd=0 free boundary instead of leaving it unenforced.
#              NONPHYSICAL/[WIP]. `-wtm_extended_soil` is the legacy alias that selects it.
# On a partial-exfiltration fixture (interior driven to the surface) these are distinguishable by the peak water
# table. Asserts, on the matrix-free Anderson path unless noted:
#   IMPLICIT : table pinned at the surface (0 <= max wtd < 0.5 m: a exfiltration constraint, not a pile) with exfiltrating cells.
#   EXPLICIT : table clamped to exactly the surface (|max wtd| < 1e-4 m) with exfiltrating cells.
#   OFF      : water piles far above the surface (max wtd > 5 m) AND the nonphysical warning is printed.
#   UNSET    : no collector set -> the DEFAULT applies. That default is now active_set (was implicit),
#              so UNSET is compared against the ACTIVE_SET run -- this is the check that catches a flip.
#              (The legacy band sink is no longer the default; it is covered as an explicit mode in taper / dt_sensitivity.)
#   AGREE    : implicit and explicit land within a few cm (same face, converging as dt->0).
#   SOLVER   : explicit also converges on the default Picard path (it needs no tangent).
# This test asserts the modes via the config KEY, so it also fences off the -wtm_<flag> 0 CLI mis-parse hazard.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work + expect_resolved: source BEFORE first use
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/rcoll_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work rc
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/runoff_collector/config.yaml. solver.time_step.mode: fixed is
# pinned there for EVERY arm -- the collector is the subject, so every other mechanism is held
# identical, and a like-for-like adaptive comparison is impossible anyway (adaptive x implicit is
# refused outright).
#
# THE `unset` ARM CANNOT BE FULLY DECLARED, BY CONSTRUCTION. Its whole point is that an ABSENT
# collection.method resolves to active_set, so declaring the key would destroy the thing it tests.
# That arm passes "" and the method line is DELETED from its config. It is why this suite is NOT on
# WTM_DECLARED_SUITES -- see the task filed alongside this commit.
emit() { # $1 stem, $2 collection.method ("" = OMIT the key, which is the `unset` arm's subject)
  local m="${2-}"
  if [ -z "$m" ]; then
      sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
          -e "/^    method: @METHOD@/d" config.yaml > "$WORK/$1.yaml"
  else
      sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
          -e "s|@METHOD@|$m|g" config.yaml > "$WORK/$1.yaml"
  fi
}
# ASSERT THE RUN USED THE COLLECTOR THE ARM ASKED FOR. This suite is the one whose arms went vacuous:
# every arm ran the DEFAULT because the tests said `collection_method` while the shim only knows
# `runoff_collector`, and they all passed while proving nothing. The model itself writes what it
# actually resolved to, AFTER overrides and the solver-dependent downgrade, so it is the only witness
# that cannot be fooled by a config that looks right. A wrong resolution invalidates the arm outright,
# so it is fatal rather than a recorded failure.
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"   # keep the suite's log if it set one
run() { # stem  collector-line  extra-flags  [expected resolved collector]
  emit "$1" "$2"
  WTM_COVERAGE_TAG="runoff_collector/$1" "$WTM" "$WORK/$1.yaml" $3 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
  [ -n "${4:-}" ] && { expect_resolved "$WTM_COVERAGE_LOG" "collector=$4" || exit 3; }
  return 0
}
run implicit implicit   "" implicit
run explicit explicit   "" explicit
run off      off        "" off
run aset     active_set "" active_set
run unset    ""         "" active_set
# extended_soil, reachable only as a MODE. The legacy alias -wtm_extended_soil and its supersession
# warning were retired 2026-09-01 with the rest of the alias flags; the RETIRED arm below replaces the
# two arms that covered them, asserting the flag now aborts rather than silently doing nothing.
run xsoil_mode extended_soil ""
# explicit on the DEFAULT Picard path (no): must converge (no tangent needed)
emit picard explicit
"$WTM" "$WORK/picard.yaml" > "$WORK/picard.log" 2>&1 \
  || { echo "RUN FAILED: explicit on Picard"; tail -3 "$WORK/picard.log"; exit 2; }
# grep -F: the banner names the CONFIG KEY (surface_water.collection.method), whose dots would
# otherwise be regex wildcards. Fixed-string matching keeps the assertion on the exact text a user sees.
OFFWARN=$(grep -cF "WARNING [surface_water.collection.method=off]" "$WORK/off.log" || true)
# The extended-soil mode must announce itself, and the superseded run must say so rather than silently
# dropping the request. Both are asserted: a silent override is the exact defect these arms exist for.
XSBANNER=$(grep -cF "surface_water.collection.method=extended_soil]: NONPHYSICAL" "$WORK/xsoil_mode.log" || true)

IM=$(ls "$WORK"/implicit_*.tif | tail -1); EX=$(ls "$WORK"/explicit_*.tif | tail -1)
OF=$(ls "$WORK"/off_*.tif | tail -1);      UN=$(ls "$WORK"/unset_*.tif | tail -1)
AS=$(ls "$WORK"/aset_*.tif | tail -1)
XS=$(ls "$WORK"/xsoil_mode_*.tif | tail -1)
OFFWARN="$OFFWARN" XSBANNER="$XSBANNER" \
  TESTS="$(readlink -f ..)" PHI="$INP/rcoll_porosity.tif" "$PY" - "$IM" "$EX" "$OF" "$UN" "$AS" "$XS" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # ONE verified V(wtd); see tests/verify_wtm_volume.sh
im, ex, of, un, aset, xs = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:7]]
def interior(a): return a[1:-1, 1:-1]
im_mx, ex_mx, of_mx, un_mx, as_mx = (float(interior(a).max()) for a in (im, ex, of, un, aset))
im_seep = int((interior(im) > -1e-3).sum()); ex_seep = int((interior(ex) > -1e-3).sum())
phi = VOL.read_band(os.environ["PHI"])
agree = float(VOL.volume_diff(im, ex, phi).max())
offwarn = int(os.environ["OFFWARN"]) > 0
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}")
    ok = ok and cond
check("IMPLICIT (exfiltration constraint, not piled)", (0.0 - 1e-3 <= im_mx < 0.5) and im_seep > 0,
      f"max wtd = {im_mx:.4f} m, exfiltrating cells = {im_seep}")
check("EXPLICIT (clamped to surface)",      abs(ex_mx) < 1e-4 and ex_seep > 0,
      f"max wtd = {ex_mx:.4e} m, exfiltrating cells = {ex_seep}")
check("OFF (piles + warns)",                of_mx > 5.0 and offwarn,
      f"max wtd = {of_mx:.2f} m, warning printed = {offwarn}")
# UNSET must track the CURRENT default, which is active_set (it was implicit until 2026-08-25). This
# check is the one that catches a default flip, so it compares against the active_set run rather than
# hard-coding a number.
check("UNSET (defaults to active_set)", abs(un_mx - as_mx) < 1e-6,
      f"max wtd = {un_mx:.4f} m (== active_set {as_mx:.4f} m; implicit would be {im_mx:.4f} m)")
# 0.1 m OF WATER VOLUME, kept at the old numeric bound rather than scaled by phi. MEASURED: the
# governing cell sits at wtd = +0.038, AT THE SURFACE, where dV/dwtd -> 1, so head 3.8356e-02 and
# volume 3.5087e-02 differ by a factor of 0.915, not 0.25. A blind x0.25 set the bound to 0.025 and
# failed a test that had not regressed. Holding 0.1 keeps the original margin (2.6x -> 2.85x) and is
# 4x STRICTER below ground, so it loosens nothing.
check("AGREE implicit vs explicit",         agree < 0.1,
      f"max|ΔV(implicit) - ΔV(explicit)| = {agree:.3e} m water volume")

# --- extended_soil: mode, alias, supersession -----------------------------------------------------
xs_mx = float(interior(xs).max())
xs_banner = int(os.environ["XSBANNER"]) > 0
# It piles like `off` -- but NOT identically to it, and that difference is the point. Both leave wtd<=0
# unenforced; extended soil additionally continues the aquifer upward, so above-surface water fills
# pore space at porosity instead of standing as free surface water, and the same water makes a
# DIFFERENT pile. Asserting "differs from off" keeps this arm from passing on a build where
# extended_soil silently degrades to plain `off`.
check("EXT_SOIL mode (piles, announces, and is NOT `off`)",
      xs_mx > 5.0 and xs_banner and abs(xs_mx - of_mx) > 1e-6,
      f"max wtd = {xs_mx:.2f} m (off = {of_mx:.2f} m), banner printed = {xs_banner}")
print("PASS: runoff_collector modes behave as specified" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
checks_rc=$?
echo "  SOLVER: explicit converged on the default Picard path (no tangent needed)"

# RETIRED ALIASES. -wtm_extended_soil and -wtm_dev_active_set were the two ALIAS-class flags: older
# spellings that reached a mode the config now names directly. They are gone (2026-09-01), and this arm
# replaces the two that used to test the alias route. It asserts the flags ABORT rather than being
# silently ignored -- the same property the RETIRED arm of tests/config_schema pins for a removed YAML
# key, and the reason retiring a flag is safe: a script still passing one stops instead of drifting.
emit ret active_set
# -wtm_definitely_not_a_flag is not a retired alias -- it is the NAMESPACE lock (#86). Every WTM setting
# is a config key now and nothing reads a -wtm_ option, so ANY -wtm_ must land in the unconsumed-flag
# guard. If this one is ever accepted, something has started reading the namespace again and the command
# line is a second route into a setting.
for flag in -wtm_extended_soil -wtm_dev_active_set -wtm_definitely_not_a_flag; do
    OUT=$(sh -c '"$0" "$1" "$2" 2>&1' "$WTM" "$WORK/ret.yaml" "$flag" 2>/dev/null)
    if echo "$OUT" | command grep -q "nothing read"; then
        echo "  PASS  RETIRED  $flag aborts as an unconsumed flag"
    else
        echo "  FAIL  RETIRED  $flag was ACCEPTED. A retired alias that is silently ignored is the"
        echo "        defect retirement exists to remove -- the run reports success having done nothing."
        checks_rc=1
    fi
done
# Propagate the checks' status. This was previously dropped: the trailing echo returned 0, so a FAILING
# assertion still exited 0 and run_all.sh reported the suite PASS while printing "FAIL" in its output.
exit $checks_rc
