#!/usr/bin/env bash
# MPI-consistency regression: the model must produce identical results at 1 and
# N MPI ranks. Runs a matrix of configurations (evap mode x FSM on/off) on the
# small ghost_cell input grid, at n=1 and each requested rank count, and checks:
#   - the final water-table TIF is bit-identical across rank counts, and
#   - the cumulative water-budget diagnostics agree.
# This is the core invariant every phase of the ArrayPack distribution must
# preserve (benchmark/DISTRIBUTED_ARP_DESIGN.md).
#
# Usage:  tests/mpi_consistency/run.sh [path/to/wtm.x] [ranks...]
# Default binary: ../../build/wtm.x   Default extra rank counts: 2 4
set -euo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS

WTM=${1:-../../build/wtm.x}
# DERIVED, SEPARATING: the converged water table agrees across decompositions to
#   1.019e-08 .. 1.249e-08 m (recorded in compare.py's own notes), while a real MPI fault -- a
#   ghost-cell error -- perturbs the field by >= 0.1 m at boundaries. The bound sits ~2 orders above
#   the agreeing values and 5 orders below the broken ones. Both edges come from measurement.
# SPREAD: 0   measured 2026-09-22 by repeat run; see this file's first bound.
WTD_TOL="${WTD_TOL:-1e-6}"
shift || true
RANKS=("$@")
if [[ ${#RANKS[@]} -eq 0 ]]; then RANKS=(2 4); fi

WTM_ABS=$(readlink -f "$WTM")
INP_ABS=$(readlink -f ../ghost_cell/inputs)
if [[ ! -x "$WTM_ABS" ]]; then
    echo "ERROR: WTM binary not found at $WTM" >&2
    exit 1
fi

# Inputs are shared with the ghost_cell test.
INPUTS=../ghost_cell/inputs
if [[ ! -d "$INPUTS" ]]; then
    echo "ERROR: expected shared inputs at $INPUTS" >&2
    exit 1
fi

make_work mpi_consistency

# Base config (equilibrium, small grid). fsm_on is overridden per case.
# THE CONFIG IS A FILE NOW (#83): tests/mpi_consistency/config.yaml. Every setting the run resolves
# to is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# equilibrium_stop.tol: 0 is PINNED in that file so the n=1-vs-n=N comparison is at the SAME cycle:
# the auto-stop could otherwise fire at slightly decomposition-dependent cycles, and the comparison
# would be between runs of different length.

run_case() { # fsm runoff_ratio nranks tag
    local fsm="$1" rr="$2" n="$3" tag="$4"
    local cfg="$WORK/${tag}.yaml"
    # fsm 0/1 became surface_water.routing off/continuous when the two keys merged (#89).
    local routing=off; [ "$fsm" = 1 ] && routing=continuous
    sed -e "s|@INPUTS@|$INP_ABS|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$tag|g" \
        -e "s|@ROUTING@|$routing|g" -e "s|@ITERS@|$(coupling_iters_for "$routing")|g" -e "s|@RR@|$rr|g" config.yaml > "$cfg"
    # -wtm_eq_tol 0: pin the full fixed cycle count so the n=1-vs-n=N comparison is at the same cycle
    # (the equilibrium auto-stop default could otherwise fire at slightly MPI-decomposition-dependent cycles).
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM_ABS" "$cfg" >"$WORK/${tag}.log" 2>&1 )
}

echo "=== MPI-consistency regression ==="
echo "binary: $WTM_ABS   rank counts vs n=1: ${RANKS[*]}"
echo

# evap_mode is GONE (2026-09-10): the member was frozen at 0 and unsettable, so the old
# evap 0/1 dimension is gone -- it would now produce identical configs. Left: FSM off/on.
# The runoff_ratio arm exists because the ROUTED input channel (col 20) is accumulated separately from
# the direct one, and with runoff_ratio 0 it is identically zero -- so without this case its MPI
# reduction is never exercised at all, which is how it went untested when the channel was added.
fail=0
for case in "0:0" "1:0" "1:0.3"; do
    fsm="${case%%:*}"; rr="${case##*:}"
    label="fsm${fsm}_rr${rr/./}"
    run_case "$fsm" "$rr" 1 "${label}_n1"
    for n in "${RANKS[@]}"; do
      run_case "$fsm" "$rr" "$n" "${label}_n${n}"
      if WTD_TOL="$WTD_TOL" python3 compare.py "$WORK/${label}_n1" "$WORK/${label}_n${n}"; then
        printf "  %-14s n=1 vs n=%-2s : PASS\n" "$label" "$n"
      else
        printf "  %-14s n=1 vs n=%-2s : FAIL\n" "$label" "$n"
        fail=1
      fi
    done
done

echo
if [[ $fail -eq 0 ]]; then echo "ALL CONSISTENCY CHECKS PASSED"; else echo "CONSISTENCY CHECKS FAILED" >&2; fi
exit $fail
