#!/usr/bin/env bash
# Run the full WTM correctness test suite. Each sub-suite is independent; this
# runner reports a summary and exits non-zero if any fails.
#
#   1. DMDA gather/scatter unit tests        (tests/run_unit_tests.sh)
#   2. Ghost-cell MPI validation             (tests/ghost_cell/run_test.sh)
#   3. Mass-balance MPI consistency          (benchmark/mass_balance_test.sh)
#   4. General MPI consistency (config matrix) (tests/mpi_consistency/run.sh)
#   5. FillSpillMerge MPI consistency        (tests/fsm_consistency/run.sh)
#
# Two tiers:
#   tests/run_all.sh              STANDARD (fast pre-push gate): each test's core assertion at n=1 vs n=4.
#   tests/run_all.sh --extended   EXTENDED (nightly / pre-release): full MPI rank sweeps + the at-scale
#                                 mass-balance fixture. Every test still runs in both tiers -- only the rank
#                                 breadth and fixture scale change, so the fast gate keeps full coverage of
#                                 the assertions while dropping the belt-and-suspenders decomposition sweeps.
# Requires wtm.x and test_dmda.x built in ../build.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(readlink -f ..)
WTM="$ROOT/build/wtm.x"
TDMDA="$ROOT/build/test_dmda.x"

TIER=standard
[[ "${1:-}" == "--extended" ]] && { TIER=extended; shift; }
if [[ "$TIER" == extended ]]; then
    MPI_RANKS="2 4 6 8"; GOLDEN_RANKS="1 2 4 6 8"; TAPER_RANKS="4 8"; MASSBAL_N=8
else
    MPI_RANKS="4";       GOLDEN_RANKS="1 4";       TAPER_RANKS="4";   MASSBAL_N=4
fi
echo "WTM test suite -- tier: $TIER  (MPI ranks: n=1 vs {$MPI_RANKS})"

# SELF-INTEGRITY. On 2026-09-05 this script was edited WHILE IT WAS RUNNING. bash reads a script by
# BYTE OFFSET, so the edit shifted everything after it: one sub-suite ran twice, another never ran at
# all, and the run still printed ALL SUITES PASSED. A green report from a corrupted run is the worst
# thing a test suite can do, so record the hash now and re-check it before the summary is believed.
# WHICH BINARY IS THIS? Announced once, up front, and the run REFUSES to start if src/ is newer than
# the binary -- testing code you did not build makes every number below meaningless. A dirty tree is
# recorded but does not block; see the reasoning in lib.sh.
. ./lib.sh
wtm_provenance "$WTM" || exit 2

SELF_SHA=$(sha256sum "$0" | cut -d" " -f1)
EXPECTED_SUITES=$(grep -c '^run "' "$0")   # every run call is top-level and unconditional

# ONE RUN AT A TIME, and a way to stop it that is not `pkill -f`. That pattern twice matched the
# caller's own command line and killed the calling shell. tests/stop.sh reads the PID from here.
LOCK="$(dirname "$0")/.run_all.lock"
if [ -f "$LOCK" ] && kill -0 "$(head -1 "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a suite run is already live (PID $(head -1 "$LOCK")); stop it with tests/stop.sh" >&2
    exit 2
fi
printf '%s\n%s\n' "$$" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# COVERAGE FINGERPRINTS. Every WTM run appends one line describing what it actually resolved to; the
# tag names the test it belongs to. Accumulated across the whole suite, then turned into
# tests/COVERAGE.md at the end. Off for anyone running a test directly (the variable is unset), so
# this changes nothing about how the tests behave -- it only records.
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$(mktemp /tmp/wtm_coverage_XXXX)}"
: > "$WTM_COVERAGE_LOG"

# ASSERTION HEALTH. Every suite's output is teed to a file so assertion_health.py can read it at the end.
# It reports headroom (tol/value) AND whether each bound carries a derivation -- see ASSERTION_HEALTH.md,
# which defines the vocabulary and states why headroom alone is not a verdict.
# Same shape as the coverage log above, and for the same reason: the information is already being
# printed, and the only thing missing was somewhere to put it. Its predecessor tol_margin.py existed since
# #84 was opened and NOTHING CALLED IT -- not run_all.sh, not any suite -- so the ranking it produces
# had never been seen. A tolerance decides PASS/FAIL, so one set too loose is a vacuous test reporting
# success; this is the instrument that finds those, and it was sitting unused.
export WTM_TOLSCAN_DIR="${WTM_TOLSCAN_DIR:-$(mktemp -d /tmp/wtm_tolscan_XXXX)}"

declare -a NAMES RESULTS
run() { # name  command...
    local name="$1" rc=0; shift
    echo; echo "########## $name ##########"
    export WTM_COVERAGE_TAG="$name"
    # TEE, not redirect: a suite's output must still appear as it runs. stderr is merged so the log
    # keeps the true interleaving, and the suite's own exit code is taken from PIPESTATUS -- the pipe
    # would otherwise report tee's success as the suite's.
    local slug; slug=$(printf '%s' "$name" | tr -cs 'A-Za-z0-9' '_')
    "$@" 2>&1 | tee "$WTM_TOLSCAN_DIR/$slug.out"; rc=${PIPESTATUS[0]}
    NAMES+=("$name"); RESULTS+=($([ $rc -eq 0 ] && echo PASS || echo FAIL))
    # EXIT 3 = a suite on the declared-config ratchet (tests/lib.sh; unconditional since #79 Phase 5) stopped saying
    # everything its run resolved to. BREAK OUT rather than carry on: unlike an ordinary assertion
    # failure, this one says the CONFIGURATION the rest of the run is about to test is not the
    # configuration anyone wrote down, so every result after it is of unknown provenance. Stopping here
    # keeps the failure attached to the change that caused it instead of burying it 30 suites later.
    if [ $rc -eq 3 ]; then
        echo >&2
        echo "ABORTING THE RUN: $name broke the declared-config rule (exit 3)." >&2
        echo "  A test config must already state every setting the run resolves to, so that what was" >&2
        echo "  tested is exactly what was written down. Fix the config, or -- if the key genuinely" >&2
        echo "  should not be stated -- add the suite to WTM_DECLARED_EXEMPT in tests/lib.sh, with the structural reason, and" >&2
        echo "  say why in the commit." >&2
        exit 3
    fi
}

# FIRST, because everything downstream that compares water tables trusts it. tests/wtm_volume.py
# is the suite's one V(wtd); if it drifts from the C++ it puts a confident, wrongly-scaled number
# in front of every assertion that uses it. A verified helper that nothing verifies on every run
# is exactly the "dead control" failure -- the knob turns and nothing is checked.
run "unit: volume helper == C++ storedVolume" ./verify_wtm_volume.sh
# Pins the run-log column NAMES and order. Four budget suites read that log; until wtm_log.py they
# read it positionally, and nothing asserted the mapping -- so a column inserted mid-header would
# silently reindex every budget assertion, and several would still report PASS. Loud, in one place.
run "unit: run-log header + trace parsing" ./log_schema/run.sh "$WTM"
# Refuses a NEW water-table comparison written in head. Cheap, and it is the only thing standing
# between the suite and a slow drift back to head norms once the conversion stops being recent.
run "unit: lint (head norms, dead continuations)" ./lint_norms.sh
run "unit: DMDA gather + storage + geometry" ./run_unit_tests.sh "$TDMDA"
# THE TOOLS THAT JUDGE THE OTHER TESTS, checked before any of them run. Seconds, no model: these
# are string functions. Five bugs were found in them by USE on the day they were written, one of
# which failed silently and accused the non-vacuity guards of being disconnected.
run "unit: assertion tools (assertion_health / assertion_probe)" ./test_assertion_tools.py
run "ghost-cell MPI"           ./ghost_cell/run_test.sh "$WTM"
run "mass-balance MPI"         "$ROOT/benchmark/mass_balance_test.sh" "$WTM" "$MASSBAL_N"
run "MPI consistency matrix"   ./mpi_consistency/run.sh "$WTM" $MPI_RANKS
run "FSM MPI consistency"      ./fsm_consistency/run.sh "$WTM" $MPI_RANKS
run "golden (expected results)" ./golden/run.sh "$WTM" $GOLDEN_RANKS
run "taper determinism+smooth"  ./taper/run.sh "$WTM" $TAPER_RANKS
run "ghost-boundary (#96)"      ./ghost_boundary/run.sh "$WTM" 4
run "storage secant≡volume"     ./storage_equivalence/run.sh "$WTM"
run "recharge consistency (#93)" ./recharge_consistency/run.sh "$WTM"
run "adaptive dt + water metric" ./adaptive_water/run.sh "$WTM"
run "adaptive estimator order"   ./estimator_order/run.sh "$WTM"
# A CONVERGED ANSWER MUST NOT DEPEND ON THE TOLERANCE YOU STOPPED AT. Runs each dt twice, differing only
# in solver.convergence.water_volume_tol, and requires the two water tables to agree. Needs no stored
# reference, which is the point: a golden can be regenerated under the shipped settings until the
# reference IS the defect, and this cannot. Currently an xfail on the banded dt (#104) with a guard that
# fails on an UNEXPECTED PASS, plus live clean arms that prove the comparison is not dead.
run "tolerance independence (#104)" ./tolerance_independence/run.sh "$WTM"
run "config schema (unknown keys)"  ./config_schema/run.sh "$WTM"
run "config/flag route equality"   ./route_equality/run.sh "$WTM"
run "snapshot name + restart"    ./snapshot_restart/run.sh "$WTM"
run "solver consistency (A≡P≡N)" ./solver_consistency/run.sh "$WTM"
# The ONLY fixture with spatially varying porosity, and the only place a head norm and a volume
# norm can rank cells differently -- everywhere else phi is uniform 0.25 and the distinction is
# invisible by construction. Its DISCRIMINATES arm asserts exactly that, so the fixture cannot
# quietly decay into uniform-equivalent behaviour while its other arms keep passing.
run "variable porosity (head != volume)" ./variable_porosity/run.sh "$WTM"
run "boundary: dirichlet≡padding" ./boundary_consistency/run.sh "$WTM"
run "boundary: analytic parabola" ./boundary_analytic/run.sh "$WTM"
run "adaptive-restart robustness" ./adaptive_restart/run.sh "$WTM"
run "flicker 1: storativity jump" ./limit_cycle/run.sh "$WTM"
run "flicker 2: evap discontinuity" ./flicker_evap/run.sh "$WTM"
run "runoff gathering (wtd=0)"     ./direct_to_runoff/run.sh "$WTM"
run "runoff_collector selector"    ./runoff_collector/run.sh "$WTM"
run "dt-sensitivity (active-set)"  ./dt_sensitivity/run.sh "$WTM"
run "active-set collector-indep"   ./active_set/run.sh "$WTM"
run "FSM conservation + lake"       ./fsm_conservation/run.sh "$WTM"
run "lake evap == ET (transition inert)" ./lake_evap_equals_et/run.sh "$WTM"
run "cross-rank drift regime"     ./xrank_growth/run.sh "$WTM"
run "cross-rank adaptive determinism" ./xrank_adaptive/run.sh "$WTM" $MPI_RANKS
run "water-budget closure (schemes)" ./budget_closure/run.sh "$WTM"
run "multi-lake stages vs dt"       ./multilake/run.sh "$WTM"
run "solve-count invariance"        ./dt_invariance/run.sh "$WTM"
run "coupling convergence"          ./coupling_convergence/run.sh "$WTM"
run "per-step water ledger"         ./budget_step_ledger/run.sh "$WTM"
run "serial rank-0 recharge path"   ./serial_recharge/run.sh "$WTM"
run "local-in-space water ledger"   ./local_ledger/run.sh "$WTM"
run "Newton Jacobian + contract"    ./newton_solver/run.sh "$WTM"
run "combination sweep"             ./combination_sweep/run.sh "$WTM"
run "nested DH + skim spill-accuracy" ./fsm_fullness/run.sh "$WTM"
run "cascade A->B->ocean (skim)"    ./fsm_cascade/run.sh "$WTM"

echo; echo "==================== SUMMARY ===================="
fail=0
# Was this script edited under us, and did every declared suite actually report?
if [ "$(sha256sum "$0" | cut -d" " -f1)" != "$SELF_SHA" ]; then
    echo "  THIS SCRIPT WAS EDITED WHILE RUNNING -- THIS RUN IS VOID (bash reads by byte offset," >&2
    echo "  so sub-suites may have been repeated or skipped). Re-run it." >&2
    fail=1
fi
if [ "${#NAMES[@]}" -ne "$EXPECTED_SUITES" ]; then
    echo "  SUITE COUNT MISMATCH: $EXPECTED_SUITES declared, ${#NAMES[@]} reported -- a sub-suite was" >&2
    echo "  skipped or duplicated, so this run does not cover what it claims." >&2
    fail=1
fi
DUPES=$(printf '%s\n' "${NAMES[@]}" | sort | uniq -d)
if [ -n "$DUPES" ]; then
    echo "  DUPLICATE SUITE NAMES (one ran twice, and something else probably did not):" >&2
    printf '    %s\n' $DUPES >&2
    fail=1
fi
for i in "${!NAMES[@]}"; do
    printf "  %-4s  %s\n" "${RESULTS[$i]}" "${NAMES[$i]}"
    [[ "${RESULTS[$i]}" == "FAIL" ]] && fail=1
done
echo "================================================="

# Regenerate the coverage matrix from what actually ran. Never fails the suite -- it is a map, not a
# gate; if a crossing in it matters, give it an arm.
# Report the REAL failure if this breaks. A blanket "skipped (no fingerprints)" once hid a crash in
# the aggregator behind a plausible-sounding reason, and the matrix silently went stale.
python3 ./coverage_matrix.py "$WTM_COVERAGE_LOG" -o ./COVERAGE.md --readme "$ROOT/README.md" \
    || echo "coverage matrix: FAILED to regenerate (see the error above); COVERAGE.md/README are STALE"

# HOW CLOSE DID EACH ASSERTION RUN TO ITS OWN TOLERANCE? (#84) Never fails the suite -- it carries no
# threshold of its own, which would just be another invented number one level up. It sorts by
# margin = tol/value and prints the thin end, because the fix for a thin margin is never "loosen it":
# it is to ask what the bound should have been DERIVED from.
echo
echo "===== tolerance margins ====="
python3 ./assertion_health.py "$WTM_TOLSCAN_DIR" \
    || echo "assertion_health: FAILED to scan (see the error above)"

[[ $fail -eq 0 ]] && echo "ALL SUITES PASSED" || { echo "SOME SUITES FAILED" >&2; }
# MACHINE-READABLE TERMINATOR. Whoever is watching this run needs to know it ENDED, and telling that
# from the outside is unreliable: matching the process by name catches the watcher's own command line
# (a `pgrep -f run_all.sh` inside a shell whose arguments contain that string matches itself, and did,
# repeatedly -- once reporting a suite as still running eight hours after it finished). Backgrounding
# the script and waiting on the wrapper is no better: the wrapper exits immediately and the suite lives
# on detached, so its exit code is the WRAPPER's, not the suite's.
#
# So the run says so itself, last line, greppable, with its own status. Watch a run with:
#     until grep -q '^SUITE_COMPLETE' log; do sleep 20; done
# and read the code off that line rather than from any process.
echo "SUITE_COMPLETE rc=$fail suites=${#NAMES[@]} failed=$(printf '%s\n' "${RESULTS[@]}" | grep -c FAIL)"
exit $fail
