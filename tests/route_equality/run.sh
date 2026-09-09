#!/usr/bin/env bash
# ROUTE EQUALITY: a config key and the -wtm_ flag it abstracts must produce the SAME RUN.
#
# WHY THIS EXISTS. WTM has two ways to say some things -- a nested-YAML key and a `-wtm_` flag -- and
# a flag classified ABSTRACTED means "the config expresses this; the flag is the primitive underneath".
# That is a CLAIM about behaviour, and nothing else tests it. Every other suite covers each mechanism;
# only this one covers the EQUIVALENCE of the two routes to it.
#
# The claim is not idle. Two channels to one setting is where this repo's config bugs live:
#   - dev.active_set silently OVERRODE an explicit surface_water.collection.method. A config asking for
#     `explicit` ran active_set instead -- 54 of 256 cells, max 0.127 m, no log line. (Removed; the
#     RETIRED arm of tests/config_schema pins it.)
#   - -wtm_extended_soil and the post-solve truncation keyed off the same global, each masking the other.
# Both were invisible because every individual mechanism worked. Only comparing ROUTES exposes them.
#
# WHAT IS ASSERTED, per setting, over its two values:
#   1. config(v1) == flag(v1)   byte-identical water table
#   2. config(v2) == flag(v2)   byte-identical water table
#   3. config(v1) != config(v2) THE CONTROL -- the setting must actually change the answer
# Equality is byte-exact, not "close": these are the same computation reached two ways, so anything but
# 0.000e+00 is a defect, and a tolerance would hide exactly the small-but-real divergence dev.active_set
# produced. Assertion 3 is what makes 1 and 2 mean anything. WITHOUT IT, A SETTING THAT REACHED THE
# MODEL BY NEITHER ROUTE WOULD SATISFY BOTH EQUALITIES AND PASS -- two no-ops agree perfectly. That is
# the vacuous-arm failure (#24) in the form this particular suite is prone to, and the reason the
# measured deltas are recorded next to each arm below: an arm whose control goes quiet has stopped
# testing, and says so.
#
# WHY THIS FILE WAS REBUILT (2026-09-06). It had FOUR arms when written, all for solver-path flags --
# -wtm_newton, -wtm_anderson, -wtm_tr_bdf2, -wtm_bdf2_on_V. The flag-retirement work (#30) removed every
# one of them: `grep -c '"-wtm_newton"' src/` is 0, and so are the other three. Each retirement correctly
# deleted its arm, and when the last one went the suite was left with a helper nothing called (`arm()`),
# a header still describing eight ABSTRACTED flags, and a banner still printing "does the config key
# reach the same run as the flag it abstracts?" above two Newton checks that answer no such question.
# It passed, in the suite, claiming coverage it no longer had. That is the same shape as the defect this
# file exists to catch, one level up: the MECHANISM (flag retirement) worked every time, and the loss
# only shows when you ask what the suite as a whole still asserts.
#
# The property is not dead -- 29 flags survive and the bridge in WTM.cpp still translates config keys
# into them. The three arms below are the ANSWER-CHANGING ones, each measured to move this fixture
# before being written (that measurement is assertion 3, now permanent).
#
# THE NEWTON ARMS ARE NOT ROUTE-EQUALITY and never were: -wtm_newton is retired, so there is no second
# route to compare. They are CONFIG CONTRACT checks -- `solver.method: newton` must be usable from YAML
# alone, because Newton does not converge from a cold start without dt-continuation (measured:
# DIVERGED_LINE_SEARCH after 4 iterations, rc 134), so a config value that meant plain Newton would be a
# documented setting that crashes. They stay here because this is where the abstraction is tested; they
# are labelled NEWTON- rather than ROUTE- so the distinction survives in the output.
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
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
fail=0

# Base config, with TWO NAMED INSERTION SLOTS rather than an appended block. An arm that appends its own
# `surface_water:` or `solver:` mapping to a config that already opens one creates a DUPLICATE MAPPING
# KEY, and yaml-cpp silently keeps one copy -- the setting vanishes and the arm compares two identical
# default runs. That is not hypothetical: it happened while measuring the deltas quoted above, and read
# exactly like a model defect (a config asking for `fsm_coupling: impulse` reporting `coupling=continuous`)
# until the config file itself was looked at. Slots make it unrepresentable.
mk() { # $1 stem ; $2 lines inside surface_water: ; $3 lines inside solver: ; $4 replaces the time_step line
    { echo "run: { type: equilibrium, initial_water_table: supplied, equilibrium_stop: { tol: 0 } }"
      echo 'time: { total: "3yr", report_interval: 1, save_every_n_reports: 9999 }'
      echo "transmissivity: { fdepth: { a: 200, b: 150, fmin: 2 } }"
      echo "surface_water:"
      echo "  mode: routed"
      echo "  runoff_ratio: 0.3"
      echo "  infiltration_during_flow: false"
      [ -n "${2:-}" ] && printf '%s\n' "$2"
      echo "solver:"
      # $4 REPLACES this line when an arm needs another key inside time_step -- a second `time_step:`
      # mapping would be a duplicate YAML key and one of the two is dropped without a word.
      # Held in a variable: a `}` inside a ${x:-default} closes the expansion early and leaks a literal
      # brace into the YAML (measured: `mode: fixed }}`, which yaml-cpp then took as a flow-map end).
      local ts_line="  time_step: { dt: 31536000 }"
      printf '%s\n' "${4:-$ts_line}"
      [ -n "${3:-}" ] && printf '%s\n' "$3"
      echo "io: { source: '$INP', region: 'fsm_test', time_start: 't0', time_end: 't0' }"
      echo "output: { outfile_prefix: '$WORK/$1_', run_log: '$WORK/$1.txt' }"
    } > "$WORK/$1.yaml"
    return 0
}

go() { # $1 stem ; $2.. flags -- returns rc, leaves the log at $WORK/$1.log
    local stem="$1"; shift
    WTM_COVERAGE_TAG="route_equality/$stem" "$WTM" "$WORK/$stem.yaml" "$@" > "$WORK/$stem.log" 2>&1
}

# max|delta| and differing-cell count between two arms' final water tables, or "MISSING" if either
# output is absent. Reads the 3yr raster each arm writes.
delta() { # $1 stem_a  $2 stem_b  ->  "<max> <ncells>"
    local a b
    a=$(ls "$WORK/$1"_*3yr.tif 2>/dev/null | head -1)
    b=$(ls "$WORK/$2"_*3yr.tif 2>/dev/null | head -1)
    if [ -z "$a" ] || [ -z "$b" ]; then echo "MISSING 0"; return; fi
    python3 - "$a" "$b" <<'PY'
import sys
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
a = gdal.Open(sys.argv[1]).ReadAsArray().astype(float)
b = gdal.Open(sys.argv[2]).ReadAsArray().astype(float)
d = np.abs(b - a)
print(f"{np.nanmax(d):.3e} {int((d > 0).sum())}")
PY
}

# One setting, two values, four runs: config and flag route for each value.
pair() { # $1 label ; $2 slot(sw|solver) ; $3 v1-yaml ; $4 v1-flags ; $5 v2-yaml ; $6 v2-flags
    local label="$1" slot="$2" tag rc
    tag=$(echo "$label" | tr -c 'a-zA-Z0-9' '_')
    local i sy fy
    for i in 1 2; do
        if [ "$i" = 1 ]; then sy="$3"; fy="$4"; else sy="$5"; fy="$6"; fi
        if [ "$slot" = sw ]; then mk "${tag}_v${i}_c" "$sy" ""; mk "${tag}_v${i}_f" "" ""
        else                     mk "${tag}_v${i}_c" "" "$sy"; mk "${tag}_v${i}_f" "" ""
        fi
        go "${tag}_v${i}_c"        || { echo "  FAIL  $label -- config route v$i did not complete"; sed -n 's/.*what():/        /p' "$WORK/${tag}_v${i}_c.log" | head -1; fail=1; return; }
        go "${tag}_v${i}_f" $fy    || { echo "  FAIL  $label -- flag route v$i did not complete";   sed -n 's/.*what():/        /p' "$WORK/${tag}_v${i}_f.log" | head -1; fail=1; return; }
    done

    local mx n arm_fail=0
    for i in 1 2; do
        read -r mx n <<< "$(delta "${tag}_v${i}_c" "${tag}_v${i}_f")"
        if [ "$n" != "0" ]; then
            arm_fail=1
            echo "  FAIL  ROUTE-$label value $i -- the two routes DISAGREE: max|d| = $mx m over $n cells."
            echo "        These are meant to be the same computation reached two ways. A difference means one"
            echo "        route silently reaches a different configuration -- see the dev.active_set defect."
            fail=1
        fi
    done

    # THE CONTROL. If the setting is inert, both equalities above are between identical default runs.
    read -r mx n <<< "$(delta "${tag}_v1_c" "${tag}_v2_c")"
    if [ "$n" = "0" ]; then
        echo "  FAIL  CONTROL-$label -- the two VALUES of this setting give an identical water table, so the"
        echo "        equality above compares two default runs and asserts nothing. Either the config key"
        echo "        stopped reaching the model, or this fixture no longer discriminates the setting."
        fail=1
    elif [ "$arm_fail" = 0 ]; then
        # PASS only if the equalities ALSO held. Printing it unconditionally here reported FAIL and PASS
        # for the same arm, and run_all.sh counts pass LINES -- so a broken arm would still have been
        # counted as covered. Caught by the probe that proves this arm bites.
        echo "  PASS  ROUTE-$label -- config route == flag route, byte-identical, for both values"
        echo "        control: the two values differ by max|d| = $mx m over $n cells, so the equality bites"
    fi
}

echo "=== route equality: does the config key reach the same run as the flag it abstracts? ==="
echo "WTM binary: $WTM"
echo

# solver.convergence.metric -> -wtm_snes_head_conv / -wtm_snes_volume_conv_govern. The per-solve
# stopping test. Answer-changing (#61 made volume the default after finding head let 88% of solves
# exit early on a stagnation test), and both values are bridged, so both are checked.
pair "convergence-metric" solver \
     "$(printf '  convergence:\n    metric: head')"   "-wtm_snes_head_conv true" \
     "$(printf '  convergence:\n    metric: volume')" "-wtm_snes_volume_conv_govern true"

# solver.convergence.water_volume_tol -> -wtm_snes_vol_tol. The tolerance that test is applied at.
pair "volume-tol" solver \
     "$(printf '  convergence:\n    water_volume_tol: 1e-6')"  "-wtm_snes_vol_tol 1e-6" \
     "$(printf '  convergence:\n    water_volume_tol: 1e-11')" "-wtm_snes_vol_tol 1e-11"

# surface_water.fsm_coupling -> -wtm_fsm_continuous. How FillSpillMerge's result reaches the
# groundwater. BOTH values are bridged deliberately: the C++ default is continuous, so bridging only
# `continuous` would leave `fsm_coupling: impulse` silently doing nothing -- a config key that reads as
# a choice and is not one. This arm is what holds that open.
pair "fsm-coupling" sw \
     "$(printf '  fsm_coupling: impulse')"    "-wtm_fsm_continuous false" \
     "$(printf '  fsm_coupling: continuous')" "-wtm_fsm_continuous true"

# NEWTON COLD-START CONTRACT, from the config side (see the header: not a route equality). `solver.method:
# newton` must be USABLE FROM YAML ALONE. Before dt-continuation was wired into the abstraction this
# aborted with DIVERGED_LINE_SEARCH after 4 iterations -- a documented config value that crashed. This arm
# is the positive control for that fix and fails loudly if the coupling is ever unpicked.
echo
mk newt_alone "" "$(printf '  method: newton')"
if go newt_alone; then
    echo "  PASS  NEWTON-YAML   solver.method: newton converges from YAML alone (no flags)"
else
    echo "  FAIL  NEWTON-YAML   solver.method: newton does NOT converge from YAML alone."
    echo "        $(command grep -m1 -oE 'DIVERGED[A-Z_]*|what\(\):.*' "$WORK/newt_alone.log" | cut -c1-80)"
    echo "        Newton needs dt-continuation from a cold start; the config value is supposed to imply it."
    fail=1
fi

# ... and the documented escape hatch must still give PLAIN Newton, with a warning rather than silence.
# NOTE the flat form: mk already emits `time_step: { dt: ... }`, and a SECOND `time_step:` key here
# would be a duplicate mapping key -- yaml-cpp keeps one of them silently, which is how this arm first
# came back reporting `absent -> ramp` with `mode: fixed` sitting in the file. Extend mk's own line.
mk newt_off "" "  method: newton" "  time_step: { dt: 31536000, mode: fixed }"
sh -c 'WTM_COVERAGE_TAG=route_equality/newt_off "$0" "$1" > "$2" 2>&1' "$WTM" "$WORK/newt_off.yaml" "$WORK/newt_off.log" 2>/dev/null
if command grep -q "WARNING \[solver.method: newton + solver.time_step.mode: fixed\]" "$WORK/newt_off.log"; then
    echo "  PASS  NEWTON-OPTOUT time_step.mode: fixed gives plain Newton and WARNS that it will"
else
    echo "  FAIL  NEWTON-OPTOUT time_step.mode: fixed did not warn. Opting out of continuation is"
    echo "        legitimate for a warm finish but diverges from a cold start; it must not be silent."
    fail=1
fi

echo
if [[ $fail -eq 0 ]]; then echo "ROUTE EQUALITY: ALL PASSED"; else echo "ROUTE EQUALITY: FAILED" >&2; fi
exit $fail
