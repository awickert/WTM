#!/usr/bin/env bash
# Adaptive-dt + pure-water-depth-metric regression. On a small coastal wedge, a cold-start equilibrium must
# reach the SAME steady water table regardless of time-integration scheme or equilibrium-stop metric. Guards:
#   (1) -wtm_tr_bdf2 -wtm_dt_adaptive converges AND matches backward-Euler (cc) -> adaptive reaches the
#       correct equilibrium (not just "runs").
#   (2) -wtm_eq_metric water-rms stops on the pure-water-depth metric (|S*Δwtd|) AND reaches the same table.
# Bites if the adaptive controller or the water-depth metric ever produces a wrong field or fails to stop.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/adwater_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work adw
# metres OF WATER VOLUME (tests/wtm_volume.py), not head -- see #61/#65. Uniform phi = 0.25 here, so this is
# the old 0.05 m head bound x0.25 exactly.
TOL="${TOL:-0.0125}"     # cross-scheme steady-state agreement, in water
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/adaptive_water/config.yaml, read and edited directly rather
# than translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it (this suite is on WTM_DECLARED_SUITES).
#
# THE HISTORY THAT MAKES THIS MATTER. These keys were once emitted only when the caller set them
# (`${INTEG:+...}`), so an unset key was ABSENT -- and absent meant `auto`, which resolved to tr-bdf2
# on the Anderson path and to adaptive under any non-implicit collector. All three arms therefore ran
# tr_bdf2 + adaptive + active_set: the arm named as the backward-Euler fixed-step CONTROL was a second
# copy of `adapt`, and guard (1) -- "adaptive matches backward-Euler" -- compared adaptive with
# itself (#24, #37). Every one of those keys now lives in the file, for every arm.
#
# config.yaml IS THE `adapt` ARM. The others are made from it by DELETING the controller dials, which
# is the only direction that works: under mode: fixed the model records no dials at all, so building
# the adaptive arm by inserting them is how one goes missing unnoticed.
emit() { # $1 stem, $2 time_integration, $3 time_step.mode, $4 equilibrium_stop.tol   (ALL REQUIRED)
  local ti="${2:?emit needs a time_integration: name the value for this arm, do not inherit it}"
  local sm="${3:?emit needs a time_step.mode: naming it is what stopped all three arms being one}"
  local et="${4:?emit needs an equilibrium_stop.tol}"
  local dials=()
  if [ "$sm" = fixed ]; then
      dials=(-e "/^    grow:/d" -e "/^    shrink:/d" -e "/^    grow_if_niter_leq:/d"
             -e "/^    max_retries:/d" -e "/^    norm:/d"
             # error_tol is still RECORDED under fixed -- at a different value -- so it is SET, not
             # deleted. Only the controller dials disappear.
             -e "s|^    error_tol: .*|    error_tol: 0.1   # the value resolved under mode: fixed|")
  fi
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  time_integration: tr-bdf2|  time_integration: $ti|" \
      -e "s|^    mode: adaptive|    mode: $sm|" \
      -e "s|^    tol: 0.001|    tol: $et|" \
      "${dials[@]}" config.yaml > "$WORK/$1.yaml"
}

BB=""
emit cc    backward-euler fixed    0.001
emit adapt tr-bdf2       adaptive 0.001
emit water backward-euler fixed    0.0005
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
# The whole point of this suite is that DIFFERENT schemes reach the SAME equilibrium, so each arm has to
# prove it ran the scheme it names. Checked against the fingerprint the model writes, not the config we
# think we wrote -- that is what caught all three arms silently sharing one configuration.
go() { # $1 stem  $2.. expected resolutions
    local stem="$1"; shift
    WTM_COVERAGE_TAG="adaptive_water/$stem" "$WTM" "$WORK/$stem.yaml" $BB > "$WORK/$stem.log" 2>&1 \
      || { echo "RUN FAILED: $stem"; tail -3 "$WORK/$stem.log"; exit 2; }
    expect_resolved "$WTM_COVERAGE_LOG" "$@" >/dev/null || exit 3
}
go cc    integrator=be_volume dtctl=fixed
go adapt integrator=tr_bdf2   dtctl=adaptive
go water integrator=be_volume dtctl=fixed

# (1) adaptive must have actually reached equilibrium (not hit the total_time cap)
grep -q "equilibrium reached" "$WORK/adapt.log" || { echo "FAIL: adaptive did not reach equilibrium"; exit 1; }
# (2) the water arm must stop on the (water-based) rms metric at the tighter 0.5 mm tolerance -- every metric
#     judges water moved |S*Δwtd| now, so this exercises a tighter pure-water-depth stop than cc's 1 mm.
grep -q "equilibrium reached (rms metric)" "$WORK/water.log" \
  || { echo "FAIL: the rms (water-depth) metric did not drive the stop"; grep -i "equilibrium reached" "$WORK/water.log"; exit 1; }

CC=$(ls "$WORK"/cc_*.tif | tail -1); AD=$(ls "$WORK"/adapt_*.tif | tail -1); WA=$(ls "$WORK"/water_*.tif | tail -1)
TOL="$TOL" PHI="$(readlink -f inputs/adwater_porosity.tif)" TESTS="$(readlink -f ..)" \
  "$PY" - "$CC" "$AD" "$WA" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh
cc, ad, wa = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
phi = VOL.read_band(os.environ["PHI"])
m = np.ones_like(cc, bool); m[:, 0] = False   # exclude the ocean column
d_ad = float(VOL.volume_diff(ad, cc, phi)[m].max()); d_wa = float(VOL.volume_diff(wa, cc, phi)[m].max())
tol = float(os.environ["TOL"])
print(f"  adaptive (tr-bdf2+dt_adaptive) vs cc: max|ΔV| = {d_ad:.4f} m water")
print(f"  water-depth metric vs cc:            max|ΔV| = {d_wa:.4f} m water volume  (tol {tol})")
if d_ad <= tol and d_wa <= tol:
    print("PASS: adaptive dt and the pure-water-depth stop metric both reach cc's equilibrium")
    sys.exit(0)
print(f"FAIL: adaptive={d_ad:.4f}, water={d_wa:.4f} m water volume exceed tol {tol} m water")
sys.exit(1)
PY
