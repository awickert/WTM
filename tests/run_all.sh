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

declare -a NAMES RESULTS
run() { # name  command...
    local name="$1"; shift
    echo; echo "########## $name ##########"
    if "$@"; then NAMES+=("$name"); RESULTS+=("PASS"); else NAMES+=("$name"); RESULTS+=("FAIL"); fi
}

run "unit: DMDA gather + storage + geometry" ./run_unit_tests.sh "$TDMDA"
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
run "snapshot name + restart"    ./snapshot_restart/run.sh "$WTM"
run "solver consistency (A≡P≡N)" ./solver_consistency/run.sh "$WTM"
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
run "water-budget closure (schemes)" ./budget_closure/run.sh "$WTM"
run "nested DH + skim spill-accuracy" ./fsm_fullness/run.sh "$WTM"
run "cascade A->B->ocean (skim)"    ./fsm_cascade/run.sh "$WTM"

echo; echo "==================== SUMMARY ===================="
fail=0
for i in "${!NAMES[@]}"; do
    printf "  %-4s  %s\n" "${RESULTS[$i]}" "${NAMES[$i]}"
    [[ "${RESULTS[$i]}" == "FAIL" ]] && fail=1
done
echo "================================================="
[[ $fail -eq 0 ]] && echo "ALL SUITES PASSED" || { echo "SOME SUITES FAILED" >&2; }
exit $fail
