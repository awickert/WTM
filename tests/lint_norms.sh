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
[ "$fail" -eq 0 ] && echo "NORM LINT: no water-table comparison is written in head" || echo "NORM LINT: FAILED"
exit $fail
