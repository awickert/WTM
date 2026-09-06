#!/usr/bin/env bash
# ROUTE EQUALITY: a config key and the -wtm_ flag it abstracts must produce the SAME RUN.
#
# WHY THIS EXISTS. WTM has two ways to say most things -- a nested-YAML key and a `-wtm_` flag -- and
# benchmark/CONFIG_FLAG_COVERAGE.md classifies eight flags as ABSTRACTED, meaning "the config expresses
# this; the flag is the primitive underneath". That is a CLAIM about behaviour, and until this file
# nothing tested it. The suite covered each mechanism, never the equivalence of the two routes to it.
#
# The claim is not idle. Two channels to one setting is where this repo's config bugs live:
#   - dev.active_set silently OVERRODE an explicit surface_water.collection.method. A config asking for
#     `explicit` ran active_set instead -- 54 of 256 cells, max 0.127 m, no log line. (Removed; the
#     RETIRED arm of tests/config_schema pins it.)
#   - -wtm_extended_soil and the post-solve truncation keyed off the same global, each masking the other.
# Both were invisible because every individual mechanism worked. Only comparing ROUTES exposes them.
#
# WHAT IS ASSERTED. For each pair: run the flag route and the config route from an IDENTICAL base config
# that omits the setting entirely, and require the output water table to agree to the BYTE. Not "close":
# these are the same computation reached two ways, so anything but 0.000e+00 is a defect, and a
# tolerance would hide exactly the small-but-real divergence dev.active_set produced.
#
# SUBSUMES the old proposal to stop triplicating the water-budget closure check across three
# active-set arms in tests/budget_closure (task #23). Those three arms reach one configuration by three
# routes -- CLI override, explicit key, default resolution -- and each independently re-checked CLOSURE,
# which the first arm had already established. The property worth pinning was that the three ROUTES
# agree, and that is asserted here instead.
#
# THE NEWTON ROW IS DELIBERATELY ASYMMETRIC. `solver.method: newton` maps to
# `solver.method: newton` + `solver.newton.dt_continuation`, NOT to plain Newton: Newton does not converge from a cold
# start without continuation (measured here: DIVERGED_LINE_SEARCH after 4 iterations, rc 134), so a
# config value that meant plain Newton would be a documented setting that crashes. The bare flag keeps
# its primitive meaning because three things depend on it -- tests/newton_solver's CONTRACT arm pins
# that plain -wtm_newton does NOT converge, benchmark/scheme_bench measures a "Newton (plain)" arm, and
# EQUILIBRIUM_ROBUSTNESS.md documents plain Newton as the thing that needs the recipe. So the config key
# is an abstraction OVER the flags, the same relation `collection.method: legacy` has to the -wtm_
# surface flags, and the equality asserted is the one that is actually true.
#
# Usage:  tests/route_equality/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work routeeq
export OMP_NUM_THREADS=1
fail=0

# Base config. Deliberately omits solver.method / time_integration / storage / collection.method /
# boundaries.land, so each arm supplies exactly one of them by exactly one route.
# The step lives at solver.time_step.dt, and BOTH arms below append their own `solver:` block, so the
# base config must not open one too -- a duplicate mapping key would silently drop one copy. Each arm
# therefore carries dt inside its own solver block.
mk() { # $1 stem, $2 extra yaml (may be empty)
cat > "$WORK/$1.yaml" <<EOF
run:
  type: equilibrium
  initial_water_table: supplied
  equilibrium_stop: { tol: 0 }
time:
  total: "3yr"
  report_interval: 1
  save_every_n_reports: 9999
transmissivity:
  fdepth: { a: 200, b: 150, fmin: 2 }
surface_water:
  mode: routed
  runoff_ratio: 0.3
  infiltration_during_flow: false
io:
  source: '$INP'
  region: 'fsm_test'
  time_start: 't0'
  time_end: 't0'
output:
  outfile_prefix: '$WORK/$1_'
  run_log: '$WORK/$1.txt'
EOF
[ -n "${2:-}" ] && printf '%s\n' "$2" >> "$WORK/$1.yaml"
return 0
}

arm() { # $1 label, $2 cli flags, $3 yaml lines, [$4 yaml lines for the FLAG side]
    # $4 exists because flags are being retired one at a time: a setting the flag route can no longer
    # express has to come from the config on BOTH sides, or the comparison stops being expressible at all.
    local tag rc_f rc_y a b
    tag=$(echo "$1" | tr -c 'a-zA-Z0-9' '_')
    mk "${tag}_f" "${4:-}"   ; "$WTM" "$WORK/${tag}_f.yaml" $2 > "$WORK/${tag}_f.log" 2>&1; rc_f=$?
    mk "${tag}_y" "$3" ; "$WTM" "$WORK/${tag}_y.yaml"    > "$WORK/${tag}_y.log" 2>&1; rc_y=$?
    if [ $rc_f -ne 0 ] || [ $rc_y -ne 0 ]; then
        echo "  FAIL  $1 -- a route did not complete (flag rc=$rc_f, config rc=$rc_y)"
        echo "        flag route: $(command grep -m1 -oE 'what\(\):.*|DIVERGED[A-Z_]*' "$WORK/${tag}_f.log" | cut -c1-80)"
        echo "        cfg  route: $(command grep -m1 -oE 'what\(\):.*|DIVERGED[A-Z_]*' "$WORK/${tag}_y.log" | cut -c1-80)"
        fail=1; return
    fi
    a=$(ls "$WORK/${tag}_f"_*3yr.tif 2>/dev/null | head -1)
    b=$(ls "$WORK/${tag}_y"_*3yr.tif 2>/dev/null | head -1)
    if [ -z "$a" ] || [ -z "$b" ]; then
        echo "  FAIL  $1 -- a route produced no raster"; fail=1; return
    fi
    local out; out=$(python3 - "$a" "$b" <<'PY'
import sys
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
a=gdal.Open(sys.argv[1]).ReadAsArray().astype(float)
b=gdal.Open(sys.argv[2]).ReadAsArray().astype(float)
d=np.abs(b-a)
print(f"{np.nanmax(d):.3e} {int((d>0).sum())}")
PY
)
    local mx n; read -r mx n <<< "$out"
    if [ "$n" = "0" ]; then
        echo "  PASS  $1 -- config route == flag route, byte-identical"
    else
        echo "  FAIL  $1 -- the two routes DISAGREE: max|Δ| = $mx m over $n cells."
        echo "        These are meant to be the same computation reached two ways. A difference means one"
        echo "        route silently reaches a different configuration -- see the dev.active_set defect."
        fail=1
    fi
}

echo "=== route equality: does the config key reach the same run as the flag it abstracts? ==="
echo "WTM binary: $WTM"
echo

# See the header: newton's config value abstracts BOTH flags, because the bare path does not converge.

# NEWTON COLD-START CONTRACT, from the config side. `solver.method: newton` must be USABLE FROM YAML
# ALONE. Before dt-continuation was wired into the abstraction this aborted with DIVERGED_LINE_SEARCH
# after 4 iterations -- a documented config value that crashed. This arm is the positive control for
# that fix and fails loudly if the coupling is ever unpicked.
echo
mk newt_alone "$(printf 'solver:\n  method: newton\n  time_step:\n    dt: 31536000')"
if "$WTM" "$WORK/newt_alone.yaml" > "$WORK/newt_alone.log" 2>&1; then
    echo "  PASS  NEWTON-YAML   solver.method: newton converges from YAML alone (no flags)"
else
    echo "  FAIL  NEWTON-YAML   solver.method: newton does NOT converge from YAML alone."
    echo "        $(command grep -m1 -oE 'DIVERGED[A-Z_]*|what\(\):.*' "$WORK/newt_alone.log" | cut -c1-80)"
    echo "        Newton needs dt-continuation from a cold start; the config value is supposed to imply it."
    fail=1
fi

# ... and the documented escape hatch must still give PLAIN Newton, with a warning rather than silence.
mk newt_off "$(printf 'solver:\n  method: newton\n  time_step:\n    dt: 31536000\n  newton:\n    dt_continuation: false')"
sh -c '"$0" "$1" > "$2" 2>&1' "$WTM" "$WORK/newt_off.yaml" "$WORK/newt_off.log" 2>/dev/null
if command grep -q "WARNING \[solver.method: newton + solver.newton.dt_continuation: false\]" "$WORK/newt_off.log"; then
    echo "  PASS  NEWTON-OPTOUT dt_continuation: false gives plain Newton and WARNS that it will"
else
    echo "  FAIL  NEWTON-OPTOUT dt_continuation: false did not warn. Opting out of continuation is"
    echo "        legitimate for a warm finish but diverges from a cold start; it must not be silent."
    fail=1
fi

echo
if [[ $fail -eq 0 ]]; then echo "ROUTE EQUALITY: ALL PASSED"; else echo "ROUTE EQUALITY: FAILED" >&2; fi
exit $fail
