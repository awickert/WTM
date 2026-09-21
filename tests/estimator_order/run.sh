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
#     time_step.grow 1, time_step.shrink 1   dt cannot change
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
# THE KNOWN HOLE IS CARRIED AS AN xfail, NOT HIDDEN. On the generic path with FSM ON the estimate
# carries NO ORDER: measured p = 0.80, -0.37, 0.49 over a 64x refinement, one of them NEGATIVE, so est
# does not even move monotonically with dt. (This note previously said "order 0.00, constant to 3
# significant figures". That was never measured: the probe read the FIRST traced step, which for this
# one arm is a 3-level scheme's startup step with no history -- nest=0, est=0 -- so the arm differenced
# 0/0 at every rung and asserted an empty string. Fixed 2026-09-06; the numbers above are the first real
# measurement this arm has produced.) The cause is
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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work estorder
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, ONE-SIDED, and it bounds a CONVERGENCE ORDER, not a physical quantity:
#   measured |p - expected| = 0.0000 at the finest rung for all three arms (ladders 1.99 2.00 2.00
#   and 1.98 1.99 2.00 against an expected 2.0). The bound is 0.2 because an observed order is a
#   ratio of differences and is noisy away from the asymptotic regime -- the COARSE arm right below
#   shows exactly that, p = 1.87 2.87 5.95 2.26, and is deliberately NOT asserted. 0.2 is the width
#   that admits a genuine second-order scheme while excluding first order (|1.0 - 2.0| = 1.0).
PTOL="${PTOL:-0.2}"        # how far the observed order may sit from its expected value
export OMP_NUM_THREADS=1

# dt ladder: 1 yr down by 4x, a 64x span. Wide enough that a genuine power law is unmistakable and a
# constant is equally unmistakable.
LADDER="31536000 7884000 1971000 492750"

# THE CONFIG IS A FILE NOW (#83): tests/estimator_order/config.yaml. Every setting the run resolves to
# is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# THREE SETTINGS IN THAT FILE ARE LOAD-BEARING, and it says so: mode: adaptive (no estimate exists
# without it), grow/shrink 1.0 (the controller must NOT resize between steps, or the dt whose order is
# measured is not the dt that ran), and error_tol 1e9 (deliberately unreachable, so no step is ever
# rejected and every arm completes at the dt it was given).
mkcfg() { # $1 stem, $2 time_step.dt, $3 routing, $4 time_integration, $5 time.total  (ALL REQUIRED)
    local dt="${2:?mkcfg needs a dt}"
    local rt="${3:?mkcfg needs a routing: off or continuous}"
    local ti="${4:?mkcfg needs a time_integration -- an ABSENT one resolves to tr-bdf2, whose order is
                   also 2.0, so it would agree with the expectation for the wrong reason}"
    local tt="${5:?mkcfg needs a time.total}"
    # The controller floor tracks this arm's dt (1e-5 x). NOT because a fixed floor would bind here --
    # that claim was made, and measured false: 315.36 s is 1562x below the FINEST rung of LADDER, the
    # frozen grow/shrink = 1.0 stop dt_next from ever falling, and error_tol 1e9 kills the reject path.
    # The scaling is for consistency with every other config in the tree. See config.yaml at dt_min.
    local dtmin; dtmin=$(awk -v d="$dt" 'BEGIN{printf "%g", d*1e-5}')
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
        -e "s|@DT@|$dt|g" -e "s|@DTMIN@|${dtmin}s|g" -e "s|@ROUTING@|$rt|g" \
        -e "s|@INTEG@|$ti|g" -e "s|@TOTAL@|$tt|g" \
        config.yaml > "$WORK/$1.yaml"
}

# One frozen-controller run; echoes "dt est" from the FIRST traced step, or nothing on failure.
probe() { # $1 stem, $2 deltat, $3 fsm_on (0|1), $4 extra CLI flags
    # fsm_on 0/1 became surface_water.routing off/continuous when the two keys merged (#89).
    local _rt=off; [ "$3" = 1 ] && _rt=continuous
    mkcfg "$1" "$2" "$_rt" "${INTEG:?probe needs INTEG: name the integrator, never inherit it}" "${TT:-20}yr"
    WTM_COVERAGE_TAG="estimator_order/$1" "$WTM" "$WORK/$1.yaml" $4 > "$WORK/$1.log" 2>&1
    # An observed ORDER is only attributable to a scheme if the run used that scheme. mkcfg emits
    # time_integration only when INTEG is set, and an absent key resolves to `auto` -> tr-bdf2 on the
    # Anderson path -- so a BDF2 arm that lost its config value would silently measure TR-BDF2's order
    # and, being 2.0 as well, would agree with its expectation for the wrong reason (#24, #37).
    if [ -n "${WANT_INTEG:-}" ]; then
        expect_resolved "$WTM_COVERAGE_LOG" "integrator=$WANT_INTEG" >/dev/null || return 1
    fi
    # The first traced step is NOT always a measurement. A 3-level scheme has no history on its first
    # step, so it reports nest=0 (no cell informed the estimate) and est=0 by definition. `grep -m1` took
    # that step, and for BDF2-on-V with FSM ON it is the ONLY arm where step 1 is a startup step -- so
    # that arm differenced 0/0 at every rung of the ladder, raised ZeroDivisionError four times, and
    # produced an empty order that the xfail below then accepted. Take the first step that actually
    # CARRIES an estimate. Verified not to move the other three arms: their step 1 already has nest=196.
    awk '/DTTRACE/ {
             n = 0; dt = ""; e = ""
             for (i = 1; i <= NF; i++) {
                 split($i, kv, "=")
                 if (kv[1] == "nest") n  = kv[2] + 0
                 if (kv[1] == "dt")   dt = kv[2]
                 if (kv[1] == "est")  e  = kv[2]
             }
             if (n > 0) { print dt, e; exit }
         }' "$WORK/$1.log"
}

export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
echo "=== adaptive local-error estimate: observed order in dt ==="
echo "WTM binary: $WTM"
echo
fail=0

arm() { # $1 label, $2 integrator FLAG, $3 fsm_on, $4 expected p, $5 mode, [$6 integrator CONFIG value]
    # $6 exists because the integrators are being moved from flags to solver.time_integration one at a
    # time; an arm names its integrator by whichever channel that one still uses.
    local label="$1" ig="$2" fsm="$3" want="$4" mode="$5" integ="${6:-}"
    local stem tag pdt="" pe="" line="" p="" n=0
    tag=$(echo "$label" | tr -c 'a-zA-Z0-9' '_')
    for d in $LADDER; do
        stem="${tag}_${d}"
        local wi; case "$integ" in tr-bdf2) wi=tr_bdf2 ;; bdf2) wi=bdf2 ;; *) wi="" ;; esac
        read -r dt e <<< "$(INTEG="$integ" WANT_INTEG="$wi" probe "$stem" "$d" "$fsm" "$ig")"
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
    # NO ARM MAY PASS ON ABSENT DATA. With pfin empty the comparison below becomes abs(-(want)), which
    # is 0 for want=0 -- so the xfail arm reported "KNOWN HOLE, still broken" while having measured
    # NOTHING, for as long as this file has existed. An expected failure that cannot tell a hole from a
    # missing measurement is not holding anything visible.
    # Tested as "is a number", not "is non-empty": awk '{print $NF}' on a whitespace-only line has NF=0
    # and $NF then refers to field 0, i.e. the whole record -- so a line of failed ratios comes back as
    # SPACES, and [ -z ] is false. The guard silently did not fire the first time for exactly that reason.
    if ! printf '%s' "$pfin" | grep -qE '^-?[0-9]+(\.[0-9]+)?$'; then
        echo "  FAIL  $label -- no observed order could be computed (p =$line)."
        echo "        Every ladder pair failed to produce a ratio. Check the DTTRACE lines in"
        echo "        $WORK/${tag}_*.log: an estimate of exactly 0 at two rungs gives log(0/0)."
        fail=1; return
    fi
    # The DEVIATION is what PTOL bounds, so compute it once and print it. Both come from one
    # python call rather than two: this runs per arm, per ladder.
    local ok dev
    read -r ok dev <<<"$(python3 -c "d=abs($pfin-($want)); print(1 if d<=$PTOL else 0, '%.4f'%d)")"
    if [ "$mode" = xfail ]; then
        # For an xfail arm $want is an UPPER BOUND on |p|, not a target. The hole is that the estimate
        # carries NO order at all -- it is not "order 0" in the sense of a clean constant. Measured:
        # p = 0.80 -0.37 0.49 over the ladder, one of them negative, so the estimate does not even move
        # monotonically with dt. Asserting a bound is the falsifiable form of that: the arm fails the day
        # the estimator reaches its scheme's design order, which is the news worth interrupting for.
        # (The previous target-form assertion, ~0.00 +/- 0.2, was never actually tested -- see the probe.)
        local under; under=$(python3 -c "print(1 if abs($pfin) < $want else 0)")
        if [ "$under" = 1 ]; then
            # |p| FIRST, ladder LAST. The ladder values are bare decimals and would outrank $pfin
            # under the parser magnitude rule if they came before the marker.
            echo "  xfail   $label: |p| = $pfin (tol $want) -- KNOWN HOLE, still no order; ladder p =$line"
        else
            echo "  FAIL  $label: p =$line  (finest $pfin) -- the estimate now carries an ORDER (|p| >= $want)."
            echo "        If the estimator has been repaired this is GOOD NEWS: promote this arm to"
            echo "        check() and update the note at the top of this file. If it has moved some"
            echo "        other way, the estimator has changed character and needs re-diagnosing."
            fail=1
        fi
    else
        if [ "$ok" = 1 ]; then
            echo "  PASS  $label: |p - expected| = $dev (tol PTOL=$PTOL) -- ladder p =$line, finest $pfin, expected ~$want"
        else
            echo "  FAIL  $label: |p - expected| = $dev (tol PTOL=$PTOL) -- ladder p =$line, finest $pfin, expected ~$want"
            fail=1
        fi
    fi
}

# COARSE RANGE: is the estimate still a local-error measure WHERE THE CONTROLLER ACTUALLY OPERATES?
#
# The ladder above tops out at 1 yr. The controller does not. Measured on tests/golden transient_test
# with 8-yr cycles, it accepted steps of 3.926 and 3.850 yr -- nearly 4x above the coarsest rung whose
# order had ever been checked. So the one quantity steering the integration was verified only BELOW the
# range it is used in, which is the same blind spot this file was written to close, one octave up.
#
# What the measurement found (fsm_test, TR-BDF2, frozen controller, first step, snes_stol 1e-12):
#   dt (yr)   8      6      4      3      2      1.5    1      0.5    0.25
#   p              1.87   2.87   5.95   2.26   1.97   1.98   1.99   1.99
# Clean 2nd order up to ~2 yr, then it stops being a power law at all: est falls 5.5x between dt 4 and
# 3, a 1.33x change. That is a REGIME CHANGE, not an order degrading, so asserting an order up here
# would be asserting a fiction.
#
# ASSERT WHAT IS ACTUALLY TRUE AND ACTUALLY NEEDED: strict MONOTONICITY. A controller can only find a
# step if a bigger step yields a bigger estimate; that property does hold across the whole range
# (1.34e-04 at 0.25 yr rising to 6.36e-01 at 8 yr) and it is what makes reject-and-shrink terminate.
# The observed orders are PRINTED, not asserted, so a future change of character is visible without
# pinning a number nobody has justified. See task #58.
COARSE_LADDER_YR="8 6 4 3 2"

coarse_arm() { # $1 label, $2 fsm_on
    local label="$1" fsm="$2" tag pdt="" pe="" line="" mono=1
    tag=$(echo "$label" | tr -c 'a-zA-Z0-9' '_')
    for m in $COARSE_LADDER_YR; do
        local d; d=$(python3 -c "print(int(31536000*$m))")
        # total_time = one step: only the FIRST traced step is read, and a coarse rung must divide evenly.
        read -r dt e <<< "$(TT="$m" INTEG=tr-bdf2 probe "coarse_${tag}_${m}" "$d" "$fsm" "")"
        if [ -z "${e:-}" ]; then
            echo "  FAIL  $label -- no DTTRACE at deltat=$d yr"; fail=1; return
        fi
        if [ -n "$pe" ]; then
            local p; p=$(python3 -c "import math;print(f'{math.log($pe/$e)/math.log($pdt/$dt):.2f}')")
            line="$line $p"
            # ladder descends, so est must DECREASE as dt decreases
            [ "$(python3 -c "print(1 if $e < $pe else 0)")" = 1 ] || mono=0
        fi
        pdt="$dt"; pe="$e"
    done
    if [ "$mono" = 1 ]; then
        echo "  PASS  $label: est strictly monotone in dt over 8..2 yr (observed p =$line, NOT asserted)"
    else
        echo "  FAIL  $label: est is NOT monotone in dt over 8..2 yr (observed p =$line)."
        echo "        A controller cannot size a step from a non-monotone estimate: shrinking would not"
        echo "        reduce it, so reject-and-retry need not terminate. Re-diagnose before trusting"
        echo "        adaptive stepping at these step sizes."
        fail=1
    fi
}

# PRECONDITION: the trace must exist at all, and est must genuinely MOVE across the ladder -- otherwise
# every order below is fitted to noise and this whole test is decoration.
read -r d0 e0 <<< "$(INTEG=tr-bdf2 probe pre_coarse 31536000 1 "")"
read -r d1 e1 <<< "$(INTEG=tr-bdf2 probe pre_fine     492750 1 "")"
if [ -n "${e0:-}" ] && [ -n "${e1:-}" ] && \
   [ "$(python3 -c "print(1 if $e0/$e1 > 10 else 0)")" = 1 ]; then
    echo "  PASS  PRECONDITION  est moves over the ladder (${e0} -> ${e1}, $(python3 -c "print(f'{$e0/$e1:.0f}x')")):"
    echo "                      the fits below have something to fit"
else
    echo "  FAIL  PRECONDITION  est did not move across a 64x dt refinement (${e0:-none} -> ${e1:-none})."
    echo "        Either -wtm_dt_trace is not reporting or the controller freeze"
    echo "        (solver.time_step.grow / .shrink) is not being honoured -- both make this test vacuous."
    fail=1
fi
echo

# TR-BDF2 carries an EMBEDDED within-step estimate (internal stage Y_gamma): second order, and immune
# to the operator split because it never differences across a handoff.
arm "TR-BDF2   fsm on " ""   1 2.0 check tr-bdf2
arm "TR-BDF2   fsm off" ""   0 2.0 check tr-bdf2
coarse_arm "TR-BDF2   fsm on, COARSE (the range the controller uses)" 1
# The generic linear-history predictor. With FSM OFF it now achieves the O(dt^2) its source claims.
#
# It did NOT until 879a188, where it measured 1.56 1.08 1.00 -- degrading toward first order as dt
# shrank, which is what task #22 recorded. The cause was not the estimator: with
# collection.method: active_set and time_integration: bdf2, the BOOTSTRAP step ran on RHS b = h^n, so
# the semismooth pin (which needs a b=0 residual) was not enforced on it. The old auto-enable skipped
# bdf2_on_V paths on the grounds that BDF2-on-V is already b=0 -- true from step 2, but bdf2v requires
# bdf2_have_history, which step 1 does not have. dev.storage_form defaulting to volume closed that hole.
#
# MEASURED, this fixture, same binary, isolating the cause:
#   collector=explicit    storage=volume   p = 0.16 0.03 0.01
#   collector=explicit    storage=secant   p = 0.16 0.03 0.01   <- storage form alone changes NOTHING
#   collector=active_set  storage=volume   p = 1.98 1.99 2.00
# so the order is set by whether the pin is enforced from the first step, not by the storage assembly.
arm "BDF2-on-V fsm off" "" 0 2.0 check bdf2
# ... and with FSM ON it does not respond to dt at all. See the KNOWN HOLE note at the top.
arm "BDF2-on-V fsm on " "" 1 1.0 xfail bdf2   # $4 = |p| BOUND for an xfail arm, not a target

echo
if [[ $fail -eq 0 ]]; then echo "ESTIMATOR ORDER: ALL PASSED"; else echo "ESTIMATOR ORDER: FAILED" >&2; fi
exit $fail
