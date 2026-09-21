#!/usr/bin/env bash
# Lake-aware active-set / semismooth exfiltration regression
# (surface_water.collection.method: active_set -- THE DEFAULT).
#
# The active-set pin enforces the exfiltration complementarity INSIDE the matrix-free Anderson residual, pinned to
# the FSM FREE SURFACE (wtd <= d_pond, d_pond = lagged ponded depth; 0 off lakes) via the min-NCP
# f = max(w_c - d_pond, f). It supersedes the runoff_collector enforcement, so the FSM-on equilibrium is
# INDEPENDENT of the collector choice -- the collector x FSM coupling ambiguity is dissolved -- WHILE keeping
# lakes: a ponded cell holds water up to its stage (its head is felt during the solve), and only the overflow
# above the stage is skimmed to runoff. (See benchmark/FSM_EVERY_STEP_DESIGN.md, project_lake_head_boundary_design.)
#
# On the fsm_test fixture (a plateau with an off-centre depression, surface water supplied), on the Anderson
# path with FSM on, this test asserts THREE things -- and the list below is the whole list:
#   LAKE PERSISTS : with active-set the lake keeps its head (max wtd well above 0) -- it is NOT
#                   flattened to the land surface (the pre-lake-aware pin gave max wtd = 0).
#   DISTINCT      : active_set differs from BOTH plain collectors. Without this the arm measures nothing.
#   BITE          : WITHOUT active-set the collector choice moves the equilibrium (implicit vs explicit
#                   spread > 0.0125 m of water volume) -- so the pin is doing real work.
#
# WHAT THIS HEADER USED TO CLAIM, and why it is gone rather than merely deleted. It listed a fourth
# assertion, COLLECTOR-INDEPENDENT: "with active-set, implicit == explicit == off to machine zero
# (< 1e-9 m spread)". That arm HAS NOT EXISTED since active_set became a member of the
# collection.method enumeration -- see the note further down, which records the deletion and the
# reason: the three configs would be textually identical, so the assertion could not fail. The header
# went on advertising it, which is worse than having lost it, because a reader takes the header for
# the claim and this was the strongest of the four. (#90)
#
# STATUS, corrected (#90): active-set is THE DEFAULT collector, not experimental and not off by
# default -- src/resolve_defaults.cpp resolves an absent surface_water.collection.method to
# active_set on every solver except Picard. The flag this header used to name, -wtm_active_set, is
# RETIRED: the whole -wtm_ namespace is closed (#86) and passing any of it aborts the run by name.
# Anderson residual only, still true.
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
# BITE GUARDS -- the floors that prove this suite is not passing on nothing. Named and made
# overridable (#121) so assertion_probe can RAISE them and confirm each one still fails when
# it should. A literal buried in a condition cannot be reached from outside, so its liveness
# was simply unknown -- and a dead bite guard means the whole suite reports success while
# comparing nothing (#34/#91/#96). These are chosen to sit unmistakably above noise, NOT
# tuned: raise one only with a measurement, never to make a run pass.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
DISTINCT_MIN="${DISTINCT_MIN:-1e-6}"   # active_set must differ from BOTH plain collectors
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED bite guard: measured max|dV(implicit) - dV(explicit)| WITHOUT
#   active-set = 0.4498 m of water volume. The guard exists because the comparison it protects is
#   empty if the two plain collectors agree; the degenerate case is therefore exactly 0. Floor
#   0.0125 sits 36x below the measurement.
BITE_MIN="${BITE_MIN:-0.0125}"   # the two plain collectors must diverge, else the comparison is empty
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, SEPARATING: measured max wtd with active-set = 9.9212 m. The broken value is
#   MEASURED history, not a guess: the pre-lake-aware pin flattened this to exactly 0, which is what
#   the assertion line still records. The floor at 1.0 m sits 9.9x below the lake stage.
LAKE_MIN="${LAKE_MIN:-1.0}"   # a lake must survive the active-set pin, not be flattened to zero
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED, unlike the three above: the sibling BITE measured 0.1590 m, and this bar is that / 36.
LF_BITE_MIN="${LF_BITE_MIN:-0.0044}"   # like-for-like: the collectors must still diverge in isolation
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED, and BLUNT by design: measured min|active_set - {explicit,implicit}|
#   = 2.672e-01 m with routing and step mode HELD, so headroom is 2.7e5. Like XS_DIFF_MIN in
#   runoff_collector, this floor asks only whether active_set has collapsed onto one of the plain
#   collectors -- a yes/no question, not a size question. Nothing here bounds how far apart they
#   ought to be.
LF_DISTINCT_MIN="${LF_DISTINCT_MIN:-1e-6}"   # like-for-like: active_set must differ from both, in isolation


# ARM ASYMMETRY, NOW VISIBLE. The three arms do NOT differ only in the collector: `explicit` runs
# routing: impulse while the other two run continuous. That is not a change -- it is what has always
# happened, because all three left fsm_coupling ABSENT and the model resolves an absent coupling to
# impulse under the explicit collector (continuous x explicit is refused outright). Writing the values
# down is what made it visible. Whether a collector-independence claim survives one arm also changing
# its coupling is a real question, recorded rather than papered over.
#
# THE CONFIG IS A FILE NOW (#83): tests/active_set/config.yaml, read and edited directly rather than
# translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it for every suite (#79 Phase 5).
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
             # dt_min is a CONTROLLER key too: under `fixed` neither the adaptive branch nor the
             # ramp runs, so nothing resolves it and declaring it would be EXTRA. The key spans
             # THREE lines in config.yaml (value + two comment lines), hence the range delete.
             -e "/^    dt_min:/,+2d"
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

# LIKE-FOR-LIKE ARMS: the collector varied ALONE, with routing and step mode HELD (#90).
#
# The three arms above differ in more than the collector -- imp_plain is continuous/fixed, exp_plain is
# impulse/adaptive, `as` is continuous/adaptive -- and that is FORCED, not careless: continuous x
# explicit is refused, and adaptive x implicit is refused. A comparison across them therefore has three
# variables in it, so "the collectors differ" cannot be attributed to the collector.
#
# It is fixable, and the reason is structural: THE TWO REFUSALS BITE ON DIFFERENT AXES. `explicit` is
# refused only against `continuous`; `implicit` only against `adaptive`. So routing: impulse with
# mode: fixed is the ONE pairing under which all three collectors run, and it is therefore the only
# place a single-variable collector comparison can be made. The full table is in config.yaml.
#
# The production arm above is KEPT rather than converted: it is the combination real runs use
# (continuous + active_set + adaptive), and dropping it to gain comparability would trade a property
# for a property. These three are additional.
run lf_as  active_set fixed impulse ""
run lf_exp explicit   fixed impulse ""
run lf_imp implicit   fixed impulse ""

IP=$(ls "$WORK"/imp_plain_*.tif | tail -1); EP=$(ls "$WORK"/exp_plain_*.tif | tail -1)
IA=$(ls "$WORK"/as_*.tif | tail -1)
LFA=$(ls "$WORK"/lf_as_*.tif | tail -1); LFE=$(ls "$WORK"/lf_exp_*.tif | tail -1)
LFI=$(ls "$WORK"/lf_imp_*.tif | tail -1)
TESTS="$(readlink -f ..)" PHI="$INP/fsm_test_porosity.tif" \
  DISTINCT_MIN="$DISTINCT_MIN" BITE_MIN="$BITE_MIN" LAKE_MIN="$LAKE_MIN" LF_BITE_MIN="$LF_BITE_MIN" LF_DISTINCT_MIN="$LF_DISTINCT_MIN" \
  "$PY" - "$IP" "$EP" "$IA" "$LFA" "$LFE" "$LFI" <<'PY'
import sys, numpy as np, rasterio, os
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # ONE verified V(wtd); see tests/verify_wtm_volume.sh
ip, ep, ia, lfa, lfe, lfi = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:7]]
def interior(a): return a[1:-1, 1:-1]
ip, ep, ia, lfa, lfe, lfi = map(interior, (ip, ep, ia, lfa, lfe, lfi))
distinct_min = float(os.environ["DISTINCT_MIN"]); bite_min = float(os.environ["BITE_MIN"])
lake_min = float(os.environ["LAKE_MIN"]); lf_bite_min = float(os.environ["LF_BITE_MIN"])
lf_distinct_min = float(os.environ["LF_DISTINCT_MIN"])
lake_head = float(ia.max())
phi_i = interior(VOL.read_band(os.environ["PHI"]))
bite      = float(VOL.volume_diff(ip, ep, phi_i).max())
# active_set must also DIFFER from both plain collectors -- otherwise this arm is measuring nothing.
differs   = min(float(np.max(np.abs(ia - ip))), float(np.max(np.abs(ia - ep))))
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("LAKE PERSISTS (head kept, not flattened)", lake_head > lake_min,
      f"max wtd with active-set = {lake_head:.4f} m (min LAKE_MIN={lake_min}) -- lake stage; the pre-lake-aware pin gave 0")
check("DISTINCT (active-set is not either plain collector)", differs > distinct_min,
      f"min|active_set - {{implicit,explicit}}| = {differs:.3e} m (min DISTINCT_MIN={distinct_min})")
# 0.0125 m OF WATER VOLUME = the old 0.05 head floor x0.25, and here that IS correct: MEASURED
# head 1.7992 vs volume 0.4498, ratio exactly 0.250, so this comparison is purely subsurface and
# the 36x margin is preserved exactly. Checked rather than assumed -- the same scaling was WRONG
# on runoff_collector and newton_solver, where the governing cell sits at the surface.
check("BITE (collectors diverge without active-set)", bite > bite_min,
      f"max|ΔV(implicit) - ΔV(explicit)| without active-set = {bite:.4f} m water volume (min BITE_MIN={bite_min})")

# LIKE-FOR-LIKE (#90): the same two claims, with the collector as the ONLY variable. The three arms
# above are forced to differ in routing and step mode as well (continuous x explicit and adaptive x
# implicit are both refused), so an attribution to the collector is not available from them. These
# three run at routing: impulse and mode: fixed -- the one pairing all three collectors can take.
lf_distinct = min(float(np.max(np.abs(lfa - lfe))), float(np.max(np.abs(lfa - lfi))))
lf_bite     = float(VOL.volume_diff(lfe, lfi, phi_i).max())
check("LIKE-FOR-LIKE DISTINCT (collector is the ONLY variable)", lf_distinct > lf_distinct_min,
      f"min|active_set - {{explicit,implicit}}| = {lf_distinct:.3e} m, routing and step mode HELD")
# THE LIKE-FOR-LIKE BITE BAR, 0.0044 m, AND WHERE IT COMES FROM. It is NOT the 0.0125 m above: that
# was measured on the continuous/adaptive arms and does not transfer, and carrying it over would have
# been a number that looks derived and is not (#84). MEASURED here, first run, 2026-09-17:
#     max|ΔV(explicit) - ΔV(implicit)| = 0.1590 m of water volume, at routing: impulse / mode: fixed
# The bar is that measurement divided by 36, which is the SAME RELATIVE MARGIN the sibling BITE check
# carries (0.4498 measured against a 0.0125 bar) -- the only precedent in this suite, so the two
# checks fail at the same fraction of their own signal rather than at two unrelated round numbers.
check("LIKE-FOR-LIKE BITE (collectors diverge, collector the ONLY variable)", lf_bite > lf_bite_min,
      f"max|ΔV(explicit) - ΔV(implicit)| at impulse/fixed = {lf_bite:.4f} m water volume"
      f" (min LF_BITE_MIN={lf_bite_min}) -- the bar is the sibling check's measured signal / 36")
print("PASS: lake-aware active-set keeps the lake's head and differs from both plain collectors"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
