#!/usr/bin/env bash
# Lake-aware active-set / semismooth exfiltration regression (-wtm_active_set).
#
# The active-set pin enforces the exfiltration complementarity INSIDE the matrix-free Anderson residual, pinned to
# the FSM FREE SURFACE (wtd <= d_pond, d_pond = lagged ponded depth; 0 off lakes) via the min-NCP
# f = max(w_c - d_pond, f). It supersedes the runoff_collector enforcement, so the FSM-on equilibrium is
# INDEPENDENT of the collector choice -- the collector x FSM coupling ambiguity is dissolved -- WHILE keeping
# lakes: a ponded cell holds water up to its stage (its head is felt during the solve), and only the overflow
# above the stage is skimmed to runoff. (See benchmark/FSM_EVERY_STEP_DESIGN.md, project_lake_head_boundary_design.)
#
# On the fsm_test fixture (a plateau with an off-centre depression, surface water supplied), on the Anderson
# path with FSM on, this test asserts:
#   LAKE PERSISTS         : with active-set the lake keeps its head (max wtd well above 0) -- it is NOT
#                           flattened to the land surface (the pre-lake-aware pin gave max wtd = 0).
#   COLLECTOR-INDEPENDENT  : with active-set, implicit == explicit == off to machine zero (< 1e-9 m spread).
#   BITE                   : WITHOUT active-set the collector choice moves the equilibrium (implicit vs
#                            explicit spread > 0.05 m) -- proving the independence is the pin doing work.
#
# active-set is EXPERIMENTAL and OFF BY DEFAULT (-wtm_active_set). Anderson residual only for now.
#
# Usage:  tests/active_set/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

# Reuse the fsm_consistency fixture (the fsm_test region), as the golden suite does.
FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work as
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# ARM ASYMMETRY, NOW VISIBLE. The three arms do NOT differ only in the collector: `explicit` runs
# routing: impulse while the other two run continuous. That is not a change -- it is what has always
# happened, because all three left fsm_coupling ABSENT and the model resolves an absent coupling to
# impulse under the explicit collector (continuous x explicit is refused outright). Writing the values
# down is what made it visible. Whether a collector-independence claim survives one arm also changing
# its coupling is a real question, recorded rather than papered over.
#
# THE CONFIG IS A FILE NOW (#83): tests/active_set/config.yaml, read and edited directly rather than
# translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it (this suite is on WTM_DECLARED_SUITES).
#
# Two preconditions are now WRITTEN DOWN in that file rather than inherited: surface_water.routing
# must be ON (the pin is defined against the FSM free surface, so with no lakes there is nothing to
# pin against) and solver.method must be anderson (the pin lives in the matrix-free residual). If
# either drifted, every arm would agree trivially and the suite would pass while testing nothing.
# EACH ARM NAMES ITS STEP MODE AS WELL AS ITS COLLECTOR, because the two are COUPLED, not independent:
# `adaptive` with `implicit` is REFUSED by name -- the implicit siphon removes above-surface water at
# rate max(0,wtd)/dt, so its per-step error GROWS as the controller shrinks dt and no step is ever
# accepted. Before #83 the mode was simply absent and the model resolved it per collector; writing the
# collector down means writing the mode down too, or the arm aborts.
emit() { # $1 stem, $2 collection.method, $3 time_step.mode, $4 routing  (ALL REQUIRED)
  local m="${2:?emit needs a collection.method: name the value for this arm, do not inherit it}"
  local sm="${3:?emit needs a time_step.mode: adaptive is refused with the implicit collector}"
  local rt="${4:?emit needs a routing: continuous is refused with the explicit collector}"
  # THE STEP-MODE ARMS DIFFER STRUCTURALLY, not just in values. Under `fixed` the model records NO
  # controller dials at all and resolves a different error_tol, so declaring them would be EXTRA keys
  # that full_config does not carry. The fixed arm therefore drops those lines rather than setting them.
  local dials=()
  if [ "$sm" = fixed ]; then
      dials=(-e "/^    grow:/d" -e "/^    shrink:/d" -e "/^    grow_if_niter_leq:/d"
             -e "/^    max_retries:/d" -e "/^    norm:/d"
             -e "s|^    error_tol: .*|    error_tol: 0.1   # the value resolved under mode: fixed|")
  fi
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^    method: active_set|    method: $m|" \
      -e "s|^    mode: adaptive|    mode: $sm|" \
      -e "s|^  routing: continuous|  routing: $rt|" \
      "${dials[@]}" config.yaml > "$WORK/$1.yaml"
}
run() { # stem  collector  step-mode  routing  [extra-flags]
  emit "$1" "$2" "$3" "$4"
  "$WTM" "$WORK/$1.yaml" $5 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
# Without active-set: the collector choice is a live variable (the BITE).
run imp_plain implicit fixed continuous ""
run exp_plain explicit adaptive impulse ""
# Lake-aware active-set, now selected as a MODE (collection.method: active_set) rather than by a flag
# that superseded whatever collector was configured.
#
# THE COLLECTOR-INDEPENDENCE ARM IS GONE, and deliberately, not by oversight. It ran implicit/explicit/off
# each with -wtm_active_set on top and asserted the three agreed to 1e-9: the flag was an ORTHOGONAL
# switch, so "which collector did you ask for" was a live variable that active-set had to dissolve. As a
# member of the collection.method enumeration, active_set is mutually exclusive with the other five -- the
# three configs would now be textually identical and the assertion could not fail. That is a genuine loss
# of a property, not a rename: the supersession it tested no longer exists to be tested.
run as active_set adaptive continuous ""

IP=$(ls "$WORK"/imp_plain_*.tif | tail -1); EP=$(ls "$WORK"/exp_plain_*.tif | tail -1)
IA=$(ls "$WORK"/as_*.tif | tail -1)
TESTS="$(readlink -f ..)" PHI="$INP/fsm_test_porosity.tif" "$PY" - "$IP" "$EP" "$IA" <<'PY'
import sys, numpy as np, rasterio, os
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # ONE verified V(wtd); see tests/verify_wtm_volume.sh
ip, ep, ia = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
def interior(a): return a[1:-1, 1:-1]
ip, ep, ia = map(interior, (ip, ep, ia))
lake_head = float(ia.max())
phi_i = interior(VOL.read_band(os.environ["PHI"]))
bite      = float(VOL.volume_diff(ip, ep, phi_i).max())
# active_set must also DIFFER from both plain collectors -- otherwise this arm is measuring nothing.
differs   = min(float(np.max(np.abs(ia - ip))), float(np.max(np.abs(ia - ep))))
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("LAKE PERSISTS (head kept, not flattened)", lake_head > 1.0,
      f"max wtd with active-set = {lake_head:.4f} m (lake stage; the pre-lake-aware pin gave 0)")
check("DISTINCT (active-set is not either plain collector)", differs > 1e-6,
      f"min|active_set - {{implicit,explicit}}| = {differs:.3e} m")
# 0.0125 m OF WATER VOLUME = the old 0.05 head floor x0.25, and here that IS correct: MEASURED
# head 1.7992 vs volume 0.4498, ratio exactly 0.250, so this comparison is purely subsurface and
# the 36x margin is preserved exactly. Checked rather than assumed -- the same scaling was WRONG
# on runoff_collector and newton_solver, where the governing cell sits at the surface.
check("BITE (collectors diverge without active-set)", bite > 0.0125,
      f"max|ΔV(implicit) - ΔV(explicit)| (no active-set) = {bite:.4f} m water volume")
print("PASS: lake-aware active-set keeps the lake's head and differs from both plain collectors"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
