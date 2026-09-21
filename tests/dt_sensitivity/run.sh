#!/usr/bin/env bash
# dt-SENSITIVITY: the equilibrium water table must NOT depend on the time step. The active-set semismooth
# exfiltration constraint (-wtm_active_set) pins wtd=0 INSIDE the solve, so the free-surface equilibrium is
# dt-INDEPENDENT to machine precision. The `implicit` in-residual siphon is NOT (it removes at rate
# 2*qmax*dt, so a table sitting in the band equilibrates at a dt-dependent depth). (The default `implicit`
# in-residual siphon is also dt-DEPENDENT at the face -- its finite 1/dt conductance leaves a dt*excess head
# above the surface -- which is exactly why the active-set face exists.) This test runs one equilibrium
# problem at two time steps (4x apart), holding report_interval fixed so only dt changes, and asserts:
#   DT-INDEPENDENT : under active-set, max|Δwtd| between the two dt is below DT_TOL (measured ~1e-14).
#   BITES          : under runoff_collector=implicit, the SAME comparison is dt-DEPENDENT
#                    (max|Δwtd| above BITE_MIN) -- proving the fixture exercises the effect and the active-set
#                    face is what removes it (a regression test that fails without it).
# Total simulated time is matched across the two dt (total_time is identical; the cycle counts scale inversely), so both reach the same
# equilibrium; report_interval is fixed so the FSM/coupling frequency is not a variable here (that is a separate axis).
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/dtsens_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work dts
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head -- the model conserves water
# and judges its stopping criteria in it (#61). Uniform phi = 0.25 here, so this is the old 1e-3 m
# head bound x0.25 exactly. BITE_MIN below is derived from it, so it follows automatically.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
DT_TOL="${DT_TOL:-2.5e-4}"   # the active-set equilibrium must match across the 4x dt change (it is ~1e-14)
# The POSITIVE CONTROL was the taper-1 band sink under runoff_collector=legacy, whose band width scaled
# as 2*qmax*dt; both were retired 2026-09-01 (fork issue #7). `implicit` replaces it, and is the better
# control anyway: it is the enforcement active_set was chosen OVER, and its retained head is ~linear in
# dt. Without a control this test could pass while measuring nothing.
#
# THE THRESHOLD IS NOW RELATIVE, NOT ABSOLUTE. It was 0.3 m, a number calibrated for the band sink's
# 2*qmax*dt spread; carrying that over to a different mechanism would be arbitrary, and lowering it until
# `implicit` passed would be worse -- fitting the threshold to the answer. What the test actually needs is
# that the control's dt-sensitivity sits far ABOVE the tolerance the test polices, so a broken comparison
# cannot slip through. 100x DT_TOL is that statement. Measured here: active_set 6.04e-14 m, implicit
# 2.25e-01 m -- a separation of twelve orders of magnitude, and 225x DT_TOL.
#
# BITE_MIN IS NO LONGER DERIVED FROM DT_TOL, and that is a consequence of the move to water (#65)
# rather than a change of mind. The two quantities now live in DIFFERENT REGIMES of V(wtd): the
# active-set arm it polices is subsurface, where dV/dwtd = phi, so its bound scaled by 0.25 with the
# units change; the implicit-siphon control it bounds is water held AT THE SURFACE, where dV/dwtd -> 1,
# so its value did NOT scale (measured: 2.250e-01 m head -> 2.248e-01 m water). Keeping the 100x
# coupling would therefore have quietly weakened the control from a 2.25x margin to a 9x one -- a
# looser assertion arriving as a side effect of a units fix, which is exactly the kind of silent
# slackening this conversion exists to prevent. Set independently, it keeps the ORIGINAL strictness.
PY="${PY:-python3}"
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED bite guard: measured implicit-siphon max|dV(1yr) - dV(quarter-yr)|
#   = 2.248e-01 m of water, 2.25x above the floor. This is the assertion that the implicit
#   collector IS dt-dependent; if it ever stops moving with dt the comparison it guards is empty,
#   so the degenerate value is 0 and the floor sits an order above it.
BITE_MIN="${BITE_MIN:-0.1}"   # metres OF WATER VOLUME; the control sits at 2.248e-01, a 2.25x margin (400x DT_TOL)
export OMP_NUM_THREADS=1

# The band sink's dt-dependence scales with ABSOLUTE dt (band = 2*qmax*dt), so use YEAR-scale steps to make it
# sharp: coarse = 1 yr x 100 cycles; fine = 0.25 yr x 400 cycles (same total simulated time, report_interval fixed).
# THE CONFIG IS A FILE NOW (#83): tests/dt_sensitivity/config.yaml. Every setting the run resolves to
# is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# solver.time_step.mode: fixed IS PINNED in that file, and it is the pin that makes this test possible:
# this suite runs the SAME problem at two time steps 4x apart and asserts only dt changes. An adaptive
# controller would resize dt away from both starting values and erase the very separation under test,
# including the `implicit` CONTROL arm (measured 2.25e-01 m) that proves the test can detect
# dt-dependence at all.
emit() { # $1 stem, $2 time_step.dt, $3 save_every_n_reports, $4 collection.method  (ALL REQUIRED)
  local dt="${2:?emit needs a dt -- it is the subject, never inherit it}"
  local sv="${3:?emit needs a save_every_n_reports: it scales inversely with dt}"
  local cm="${4:?emit needs a collection.method: active_set or the implicit control}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|@DT@|$dt|g" -e "s|@SAVE@|$sv|g" \
      -e "s|^    method: active_set|    method: $cm|" config.yaml > "$WORK/$1.yaml"
}
run() { # stem deltat cycles collector extra_flags
  emit "$1" "$2" "$3" "$4"
  "$WTM" "$WORK/$1.yaml" $5 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
COARSE=31536000; FINE=7884000   # 1 yr, 0.25 yr
run as_c  $COARSE 100 active_set ""   # tolerance is a config key now (snes_stol in emit)
run as_f  $FINE   400 active_set ""
run leg_c $COARSE 100 implicit ""
run leg_f $FINE   400 implicit ""

AC=$(ls "$WORK"/as_c_*.tif|tail -1); AF=$(ls "$WORK"/as_f_*.tif|tail -1)
LC=$(ls "$WORK"/leg_c_*.tif|tail -1); LF=$(ls "$WORK"/leg_f_*.tif|tail -1)
DT_TOL="$DT_TOL" BITE_MIN="$BITE_MIN" PHI="$(readlink -f inputs/dtsens_porosity.tif)" \
  TESTS="$(readlink -f ..)" "$PY" - "$AC" "$AF" "$LC" "$LF" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh
ac, af, lc, lf = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
phi = VOL.read_band(os.environ["PHI"])
dt_tol = float(os.environ["DT_TOL"]); bite = float(os.environ["BITE_MIN"])
act = float(VOL.volume_diff(ac, af, phi).max())   # active-set: dt sensitivity (should be ~0)
leg = float(VOL.volume_diff(lc, lf, phi).max())   # implicit siphon: dt sensitivity (should be large)
print(f"  DT-INDEPENDENT : active-set        max|ΔV(1yr) - ΔV(quarter-yr)| = {act:.3e} m water  (tol DT_TOL={dt_tol})")
print(f"  BITES          : implicit siphon   max|ΔV(1yr) - ΔV(quarter-yr)| = {leg:.3e} m water  (min BITE_MIN={bite})")
ok = act <= dt_tol and leg >= bite
if ok:
    print("PASS: the active-set exfiltration constraint gives a dt-independent equilibrium; implicit does not (test bites)")
else:
    # NAME WHAT FAILED. The bare word "FAIL" left a reader to work out which of the two clauses broke,
    # and left any tool unable to tell a failed assertion from a crash. Same fix as limit_cycle.
    if act > dt_tol:
        print(f"  FAIL  DT-INDEPENDENT: active-set moved across the dt change, max|ΔV| = {act:.3e} m water (tol DT_TOL={dt_tol})")
    if leg < bite:
        print(f"  FAIL  BITES: the implicit siphon did NOT move across the dt change, max|ΔV| = {leg:.3e} m water (min BITE_MIN={bite})")
        print("        Without that contrast the dt-independence claim above proves nothing.")
    print("FAIL")
sys.exit(0 if ok else 1)
PY
