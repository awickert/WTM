#!/usr/bin/env bash
# ESTIMATOR ORDER: does the adaptive controller's local-error estimate actually respond to dt?
#
# WHY THIS EXISTS. -wtm_dt_adaptive sizes every step from a local-error estimate `est`. That estimate
# is the single quantity steering the integration -- and until -wtm_dt_trace was added it was computed
# on every step and reported NOWHERE. Nothing in the suite could see it, so nothing could notice that
# on the generic (non-TR) path it did not respond to dt AT ALL. A controller reading a constant cannot
# control anything: it shrinks dt against an error that never shrinks, rejects, and grinds toward the
# step floor. That is not a tuning problem, it is a broken instrument, and it was invisible because it
# was unobservable.
#
# WHAT IT ASSERTS -- an ORDER, not a value. Every arm refines dt from an IDENTICAL initial state and
# fits the observed order p in est ~ dt^p between successive rungs. Asserting "est is small" would be
# worthless (a constant is small too); asserting the ORDER is what distinguishes a working estimator
# from a number that merely looks reasonable.
#
# THE CONTROLLER IS FROZEN so that dt is exactly what we set:
#     -wtm_dtc_grow 1 -wtm_dtc_shrink 1   dt cannot change
#     -wtm_dt_tol 1e9                     nothing is ever rejected
# and we read only the FIRST trace line of each run, where dt is the configured deltat by construction
# and every arm starts from the same state. (Those dtc_* flags were themselves parsed ONLY on the
# Newton-continuation path until this was written -- on a plain adaptive run they were accepted and
# silently ignored, so this freeze would have been a no-op and this test a fiction. Fixed alongside.)
#
# snes_stol IS TIGHT (1e-12) ON PURPOSE. An under-converged solve leaves x nearer h^n, and the
# predictor extrapolates from h^n too, so a loose solve makes the estimate look SMALLER than the true
# truncation error -- measured: at -wtm_dt_tol 0.001 the step count doubles (90 -> 178) on snes_stol
# alone. Order must be measured where the algebraic error cannot masquerade as truncation error.
#
# THE KNOWN HOLE IS CARRIED AS AN xfail, NOT HIDDEN. On the generic path with FSM ON the observed order
# is 0.00 -- est is constant in dt to 3 significant figures across a 64x refinement. The cause is
# structural: WTM's step is an operator SPLIT (solve maps w^n -> x, then FillSpillMerge maps
# x -> w^{n+1}), so EVERY pair of states a history-based estimator can difference straddles a handoff,
# and the FSM jump -- which does not shrink with dt -- lands in the estimate. Phase-aligning WHICH pair
# is differenced does not remove it (tried: extrapolating the solve-only increment x^{n-1} - w^{n-1}
# instead of w^n - w^{n-1}; still p = 0.00, because w^n = FSM(x^{n-1}) either way). TR-BDF2 is immune
# structurally rather than by luck: its estimate is EMBEDDED WITHIN one step, built from the internal
# stage Y_gamma, and never differences across a handoff at all -- it measures p = 2.00 with FSM on.
# The general rule this pins: A HISTORY-BASED LOCAL-ERROR ESTIMATOR IS INVALID ACROSS AN
# OPERATOR-SPLIT HANDOFF. The real repair is an embedded within-step estimator on every integrator
# that offers adaptive dt (or refusing adaptive dt on those that cannot supply one); until then this
# arm holds the hole VISIBLE and fails loudly the day it changes in either direction.
#
# Usage:  tests/estimator_order/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/estorder_XXXX); trap 'rm -rf "$WORK"' EXIT
PTOL="${PTOL:-0.2}"        # how far the observed order may sit from its expected value
export OMP_NUM_THREADS=1

# dt ladder: 1 yr down by 4x, a 64x span. Wide enough that a genuine power law is unmistakable and a
# constant is equally unmistakable.
LADDER="31536000 7884000 1971000 492750"

mkcfg() { # $1 stem, $2 deltat, $3 fsm_on
    cat > "$WORK/$1.yaml.in" <<EOF
run_type equilibrium
total_time 20yr
supplied_wt 1
deltat $2
report_interval 1
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on $3
runoff_ratio 0.3
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
textfilename $WORK/$1.txt
outfile_prefix $WORK/$1_
runoff_collector active_set
EOF
    ../emit_config.sh < "$WORK/$1.yaml.in" > "$WORK/$1.yaml"
}

# One frozen-controller run; echoes "dt est" from the FIRST traced step, or nothing on failure.
probe() { # $1 stem, $2 deltat, $3 fsm_on, $4 integrator flag
    mkcfg "$1" "$2" "$3"
    "$WTM" "$WORK/$1.yaml" -wtm_anderson $4 -wtm_dt_adaptive -wtm_dt_trace \
        -wtm_dt_tol 1e9 -wtm_dtc_grow 1.0 -wtm_dtc_shrink 1.0 \
        -snes_stol 1e-12 -wtm_eq_tol 0 > "$WORK/$1.log" 2>&1
    grep -m1 DTTRACE "$WORK/$1.log" | sed -E 's/.*dt=([-0-9.e+]+) est=([-0-9.e+]+).*/\1 \2/'
}

echo "=== adaptive local-error estimate: observed order in dt ==="
echo "WTM binary: $WTM"
echo
fail=0

arm() { # $1 label, $2 integrator flag, $3 fsm_on, $4 expected p, $5 mode (check|xfail)
    local label="$1" ig="$2" fsm="$3" want="$4" mode="$5"
    local stem tag pdt="" pe="" line="" p="" n=0
    tag=$(echo "$label" | tr -c 'a-zA-Z0-9' '_')
    for d in $LADDER; do
        stem="${tag}_${d}"
        read -r dt e <<< "$(probe "$stem" "$d" "$fsm" "$ig")"
        if [ -z "${e:-}" ]; then
            echo "  FAIL  $label -- no DTTRACE at deltat=$d (is -wtm_dt_trace wired?)"
            tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1; return
        fi
        if [ -n "$pe" ]; then
            p=$(python3 -c "import math;print(f'{math.log($pe/$e)/math.log($pdt/$dt):.2f}')")
            line="$line $p"; n=$((n+1))
        fi
        pdt="$dt"; pe="$e"
    done
    # Judge on the FINEST pair: the asymptotic regime is where an order claim actually lives.
    local pfin; pfin=$(echo "$line" | awk '{print $NF}')
    local ok; ok=$(python3 -c "print(1 if abs($pfin-($want))<=$PTOL else 0)")
    if [ "$mode" = xfail ]; then
        if [ "$ok" = 1 ]; then
            echo "  xfail   $label: p =$line  (finest $pfin, expected ~$want) -- KNOWN HOLE, still broken"
        else
            echo "  FAIL  $label: p =$line  (finest $pfin) -- expected the KNOWN HOLE at ~$want."
            echo "        If the estimator has been repaired this is GOOD NEWS: promote this arm to"
            echo "        check() and update the note at the top of this file. If it has moved some"
            echo "        other way, the estimator has changed character and needs re-diagnosing."
            fail=1
        fi
    else
        if [ "$ok" = 1 ]; then
            echo "  PASS  $label: p =$line  (finest $pfin, expected ~$want +/- $PTOL)"
        else
            echo "  FAIL  $label: p =$line  (finest $pfin, expected ~$want +/- $PTOL)"
            fail=1
        fi
    fi
}

# PRECONDITION: the trace must exist at all, and est must genuinely MOVE across the ladder -- otherwise
# every order below is fitted to noise and this whole test is decoration.
read -r d0 e0 <<< "$(probe pre_coarse 31536000 1 "-wtm_tr_bdf2")"
read -r d1 e1 <<< "$(probe pre_fine     492750 1 "-wtm_tr_bdf2")"
if [ -n "${e0:-}" ] && [ -n "${e1:-}" ] && \
   [ "$(python3 -c "print(1 if $e0/$e1 > 10 else 0)")" = 1 ]; then
    echo "  PASS  PRECONDITION  est moves over the ladder (${e0} -> ${e1}, $(python3 -c "print(f'{$e0/$e1:.0f}x')")):"
    echo "                      the fits below have something to fit"
else
    echo "  FAIL  PRECONDITION  est did not move across a 64x dt refinement (${e0:-none} -> ${e1:-none})."
    echo "        Either -wtm_dt_trace is not reporting or the controller freeze"
    echo "        (-wtm_dtc_grow/-wtm_dtc_shrink) is not being honoured -- both make this test vacuous."
    fail=1
fi
echo

# TR-BDF2 carries an EMBEDDED within-step estimate (internal stage Y_gamma): second order, and immune
# to the operator split because it never differences across a handoff.
arm "TR-BDF2   fsm on " "-wtm_tr_bdf2"   1 2.0 check
arm "TR-BDF2   fsm off" "-wtm_tr_bdf2"   0 2.0 check
# The generic linear-history predictor. With FSM OFF it converges -- at FIRST order, not the O(dt^2)
# its own source comment claims, which is a second and separate discrepancy worth keeping in view
# (candidate: active-set switching leaves the trajectory only C^1 in time). Pinned at what it MEASURES.
arm "BDF2-on-V fsm off" "-wtm_bdf2_on_V" 0 1.0 check
# ... and with FSM ON it does not respond to dt at all. See the KNOWN HOLE note at the top.
arm "BDF2-on-V fsm on " "-wtm_bdf2_on_V" 1 0.0 xfail

echo
if [[ $fail -eq 0 ]]; then echo "ESTIMATOR ORDER: ALL PASSED"; else echo "ESTIMATOR ORDER: FAILED" >&2; fi
exit $fail
