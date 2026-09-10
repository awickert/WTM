#!/usr/bin/env bash
# CONFIG ARRIVAL: a config key must REACH the model and change what it is supposed to change.
#
# WHAT THIS FILE USED TO BE, and why it changed. It was ROUTE EQUALITY: WTM had two ways to say some
# things -- a nested-YAML key and a `-wtm_` flag -- and this suite asserted that the two produced the
# SAME RUN, byte for byte. That claim mattered because two channels to one setting is where this repo's
# config bugs lived:
#   - dev.active_set silently OVERRODE an explicit surface_water.collection.method. A config asking for
#     `explicit` ran active_set instead -- 54 of 256 cells, max 0.127 m, no log line.
#   - -wtm_extended_soil and the post-solve truncation keyed off the same global, each masking the other.
#   - a -wtm_ flag on the command line beat the config key it duplicated, because the bridge used
#     set_opt_if_unset and took the FIRST setter (#86).
#
# THE SECOND ROUTE IS GONE (#86, 2026-09-10). Nothing in the model reads a -wtm_ option; the namespace
# is retired and any -wtm_ aborts. So route equality has no second route to compare against -- the
# suite's subject ceased to exist, which is NOT the same as the suite being wrong.
#
# WHAT SURVIVES IS THE HALF THAT MADE FLAG REMOVAL SAFE. A single-route interface has exactly one new
# failure mode: a key that PARSES and never ARRIVES. Nothing else checks that directly. So each arm
# below runs ONE setting at TWO values, both from the config, and requires the water tables to DIFFER.
# If they do not, either the key stopped reaching the model or this fixture stopped discriminating it --
# and both make every other assertion here meaningless.
#
# WHY "DIFFER" IS THE RIGHT ASSERTION AND NOT A WEAKER ONE. It is the old suite's CONTROL, promoted to
# be the whole test. That control was always the part that gave the equalities meaning: WITHOUT IT, A
# SETTING THAT REACHED THE MODEL BY NEITHER ROUTE WOULD SATISFY BOTH EQUALITIES AND PASS -- two no-ops
# agree perfectly. The measured deltas are printed next to each arm so an arm whose discrimination goes
# quiet says so rather than passing.
#
# THIS FILE HAS BEEN EMPTIED ONCE BEFORE, AND THAT IS THE REASON FOR THE PARAGRAPHS ABOVE. It had four
# arms for solver-path flags -- -wtm_newton, -wtm_anderson, -wtm_tr_bdf2, -wtm_bdf2_on_V. Flag
# retirement (#30) removed every one, each retirement correctly deleting its arm, and when the last one
# went the suite was left with a helper nothing called, a header describing eight flags, and a banner
# asking a question no remaining check answered. It PASSED, claiming coverage it no longer had. The
# mechanism worked every time; what no step owned was the question of what the suite still asserted.
# So: when the last instance of a category is removed, ask what the test covering that category tests.
#
# THE NEWTON ARMS ARE NOT ARRIVAL CHECKS and never were route-equality either. They are CONFIG CONTRACT
# checks -- `solver.method: newton` must be usable from YAML alone, because Newton does not converge
# from a cold start without its continuation ramp (measured: DIVERGED_LINE_SEARCH after 4 iterations,
# rc 134), so a config value that meant plain Newton would be a documented setting that crashes. They
# are labelled NEWTON- so the distinction survives in the output.

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

# One setting, two values, two runs -- both from the CONFIG, because there is no longer a second route.
arrives() { # $1 label ; $2 slot(sw|solver) ; $3 v1-yaml ; $4 v2-yaml
    local label="$1" slot="$2" tag
    tag=$(echo "$label" | tr -c 'a-zA-Z0-9' '_')
    local i sy
    for i in 1 2; do
        if [ "$i" = 1 ]; then sy="$3"; else sy="$4"; fi
        if [ "$slot" = sw ]; then mk "${tag}_v${i}" "$sy" ""; else mk "${tag}_v${i}" "" "$sy"; fi
        go "${tag}_v${i}" || { echo "  FAIL  $label -- value $i did not complete"; sed -n 's/.*what():/        /p' "$WORK/${tag}_v${i}.log" | head -1; fail=1; return; }
    done

    local mx n
    read -r mx n <<< "$(delta "${tag}_v1" "${tag}_v2")"
    if [ "$n" = "0" ]; then
        echo "  FAIL  ARRIVES-$label -- the two VALUES give an IDENTICAL water table. Either the config key"
        echo "        stopped reaching the model -- which is the failure mode a single-route interface is"
        echo "        exposed to, and what this suite now exists to catch -- or this fixture no longer"
        echo "        discriminates the setting. Both make every other arm here meaningless."
        fail=1
    else
        echo "  PASS  ARRIVES-$label -- the key reaches the model: its two values differ by max|d| = $mx m"
        echo "        over $n cells"
    fi
}

echo "=== config arrival: does a config key reach the model and change what it should? ==="
echo "WTM binary: $WTM"
echo

# solver.convergence.metric -> -wtm_snes_head_conv / -wtm_snes_volume_conv_govern. The per-solve
# stopping test. Answer-changing (#61 made volume the default after finding head let 88% of solves
# exit early on a stagnation test), and both values are bridged, so both are checked.
arrives "convergence-metric" solver \
     "$(printf '  convergence:\n    metric: head')" \
     "$(printf '  convergence:\n    metric: volume')"

# solver.convergence.water_volume_tol -> -wtm_snes_vol_tol. The tolerance that test is applied at.
arrives "volume-tol" solver \
     "$(printf '  convergence:\n    water_volume_tol: 1e-6')" \
     "$(printf '  convergence:\n    water_volume_tol: 1e-11')"

# surface_water.fsm_coupling -> -wtm_fsm_continuous. How FillSpillMerge's result reaches the
# groundwater. BOTH values are bridged deliberately: the C++ default is continuous, so bridging only
# `continuous` would leave `fsm_coupling: impulse` silently doing nothing -- a config key that reads as
# a choice and is not one. This arm is what holds that open.
arrives "fsm-coupling" sw \
     "$(printf '  fsm_coupling: impulse')" \
     "$(printf '  fsm_coupling: continuous')"

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
