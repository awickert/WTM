#!/usr/bin/env bash
# Refuse a NEW water-table comparison written in head.
#
# The model conserves WATER VOLUME, and every stopping criterion is judged in it (#61). A test that
# differences two rasters with np.abs is comparing water tables in HEAD unless it goes through
# tests/wtm_volume.py -- and that is not cosmetic: below the surface dV/dwtd is the porosity, so on a
# phi = 0.25 fixture a head norm over-weights deep cells 4x. Converting the suite (#65) found FOUR
# tests where the difference was not a rescale at all: in two of them the move to volume CHANGED WHICH
# CELL GOVERNS, and a mechanical x0.25 failed a test that had not regressed.
#
# The rule: if a file reads rasters AND differences them with np.abs, it must import wtm_volume, or
# appear below with a reason. Keyed on RASTER READS rather than on np.abs alone, because several
# np.abs sites in the suite operate on run-log budget columns that are ALREADY volumes -- converting
# those would be wrong, and a lint that demanded it would be teaching the wrong lesson.
set -uo pipefail
cd "$(dirname "$0")"

# EXEMPT, each for a reason. An entry here is a claim that the comparison is NOT a water-table
# accuracy claim in head; adding one without such a reason defeats the check.
exempt_reason() {
    case "$1" in
        boundary_analytic/run.sh) echo "compares the table against a closed-form PARABOLA, which is a head solution" ;;
        fsm_conservation/run.sh)  echo "differences run-log budget values, which are already volumes" ;;
        fsm_cascade/run.sh)       echo "MPI identity: 1 rank vs 4 (wk vs wk4), unit-agnostic" ;;
        fsm_fullness/run.sh)      echo "MPI identity: 1 rank vs 4 (wk vs wk4), unit-agnostic" ;;
        xrank_growth/run.sh)      echo "cross-rank drift: an identity comparison, unit-agnostic" ;;
        ghost_cell/check_results.py) echo "MPI identity across ranks, unit-agnostic" ;;
        mpi_consistency/compare.py)  echo "MPI identity across ranks, unit-agnostic" ;;
        taper/taper_test.py)         echo "MPI identity across ranks, unit-agnostic" ;;
        *) return 1 ;;
    esac
}

fail=0
for f in */run*.sh */*.py; do
    [ -f "$f" ] || continue
    case "$f" in */make_inputs.py) continue ;; esac          # fixture generators write rasters, they do not compare
    grep -q "rasterio.open\|read_band(" "$f" || continue
    grep -qE "np\.abs\(" "$f" || continue
    # An actual IMPORT, not the string anywhere: a first version of this grepped for "wtm_volume"
    # and was satisfied by a COMMENT mentioning it, so stripping the import did not trip the lint.
    grep -qE "^[[:space:]]*import wtm_volume" "$f" && continue
    if reason=$(exempt_reason "$f"); then
        printf "  exempt  %-30s %s\n" "$f" "$reason"
    else
        printf "  FAIL    %-30s differences rasters without wtm_volume -- that is a HEAD norm\n" "$f"
        echo   "                                         use volume_diff(), or add an exemption WITH a reason"
        fail=1
    fi
done
# ---- A LINE CONTINUATION THAT CONTINUES INTO NOTHING -------------------------------------------
# A trailing `\` followed by a blank or whitespace-only line. bash PARSES it -- `bash -n` is silent,
# the suite runs, and the arguments that used to be on the continuation are simply gone.
#
# Both shapes were live in this tree. In tests/budget_closure two arms ended in `\` over an empty
# line because the flags they once carried (-wtm_anderson -wtm_tr_bdf2 -wtm_active_set) had been
# retired and only the backslash was left. In tests/estimator_order a scripted edit removed
# `-snes_stol 1e-12` from a continuation and left the line above it ending in `\`, producing
#     ... -wtm_dtc_shrink 1.0 \ > "$WORK/$1.log" 2>&1
# -- an escaped space, silently passed to the model as an extra argument.
#
# Neither was caught by anything. The second was found only because the first had just been noticed,
# which is luck, not a process. This lint is the process.
for f in */run.sh *.sh; do
    [ -f "$f" ] || continue
    # SKIP THIS FILE. It necessarily CONTAINS the patterns it searches for -- the grep expression and
    # the failure message both spell them out -- so scanning itself is a guaranteed false positive.
    # Third self-match of this session, after a pkill and a pgrep that each matched their own command
    # line; a checker that reads source is always a candidate for its own rule.
    [ "$f" = "lint_norms.sh" ] && continue
    bad=$(awk '/\\$/ { prev = NR; line = $0 }
               NR == prev + 1 && /^[[:space:]]*$/ { printf "%d: %s\n", NR - 1, line }' "$f")
    if [ -n "$bad" ]; then
        echo "  FAIL  $f: line continuation into a BLANK line -- the continued arguments are lost:"
        printf '%s\n' "$bad" | sed 's/^/          /'
        fail=1
    fi
    if command grep -qn '\\ >' "$f"; then
        echo "  FAIL  $f: '\\ >' -- a stranded backslash before a redirect passes a literal space:"
        command grep -n '\\ >' "$f" | sed 's/^/          /'
        fail=1
    fi
done


# #112 ROLLBACK COMPLETENESS. ../src/coupling_snapshot.hpp must carry EVERY `double total_*` accumulator
# declared in ../src/ArrayPack.hpp. Adding a tenth to ArrayPack and forgetting the snapshot is the
# SILENT MASS ERROR the design names: the coupling iteration would restore eight of nine and lose
# water without a word. The C++ round-trip test cannot see this -- it only checks the fields it knows
# about -- so the two SETS are compared here instead.
ap=$(grep -oE '^\s*double\s+(total_[a-z_]+)' ../src/ArrayPack.hpp | grep -oE 'total_[a-z_]+' | sort -u)
cs=$(grep -oE '^\s*double\s+(total_[a-z_]+)' ../src/coupling_snapshot.hpp | grep -oE 'total_[a-z_]+' | sort -u)
nap=$(printf '%s\n' "$ap" | grep -c .)
# NON-VACUOUS: two EMPTY sets compare equal, so a wrong path would report OK having checked
# nothing. That is exactly how this check first passed while reading a file that did not exist.
if [ "$nap" -eq 0 ]; then
    echo "  FAIL  #112 rollback check found ZERO accumulators in ../src/ArrayPack.hpp -- it is" >&2
    echo "        reading the wrong file, so it is checking nothing." >&2
    fail=1
elif [ "$ap" != "$cs" ]; then
    echo "  FAIL  #112 rollback is INCOMPLETE -- ArrayPack and coupling_snapshot disagree:" >&2
    diff <(printf '%s\n' "$ap") <(printf '%s\n' "$cs") | sed 's/^/        /' >&2
    echo "        Add the field to ../src/coupling_snapshot.hpp (capture AND restore) and to the" >&2
    echo "        round-trip test in src/test_coupling_snapshot.cpp, then raise kAccumulators." >&2
    fail=1
else
    n=$(printf '%s\n' "$ap" | grep -c .)
    echo "  OK   #112 ROLLBACK  coupling_snapshot carries all $n ArrayPack total_* accumulators"
    k=$(grep -oE 'kAccumulators = [0-9]+' ../src/coupling_snapshot.hpp | grep -oE '[0-9]+')
    if [ "$k" != "$n" ]; then
        echo "  FAIL  #112 kAccumulators says $k but there are $n accumulators" >&2; fail=1
    fi
fi

[ "$fail" -eq 0 ] && echo "LINT: no head-norm comparison, no continuation that continues into nothing" \
                  || echo "LINT: FAILED"
exit $fail
