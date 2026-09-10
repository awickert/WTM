#!/usr/bin/env bash
# FillSpillMerge MPI-consistency regression.
#
# The ghost_cell / mpi_consistency fixtures keep the water table below ground,
# so FSM is a no-op there and cannot catch bugs in the rank-0 FSM path. This
# fixture supplies surface water over a plateau with an off-centre depression,
# so FSM genuinely redistributes wtd. The final water table must be identical
# (to FP-reduction tolerance) whether run on 1 or N ranks -- verified to BITE:
# removing the post-FSM wtd broadcast makes n=1 vs n=4 diverge by ~4e-2 m.
#
# Usage:  tests/fsm_consistency/run.sh [path/to/wtm.x] [ranks...]
# Default binary: ../../build/wtm.x   Default extra rank counts: 2 4
set -euo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS

WTM=$(readlink -f "${1:-../../build/wtm.x}")
shift || true
RANKS=("$@"); [[ ${#RANKS[@]} -eq 0 ]] && RANKS=(2 4)

if [[ ! -x "$WTM" ]]; then echo "ERROR: WTM binary not found at $WTM" >&2; exit 1; fi

SD=$(readlink -f inputs)
if [[ ! -f "$SD/fsm_test_t0_topography.tif" ]]; then
    echo "Generating FSM test inputs..."
    python3 make_inputs.py >/dev/null
fi

make_work fsm_consistency

# THE CONFIG IS A FILE NOW (#83): tests/fsm_consistency/config.yaml, read and edited directly rather
# than translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it (this suite is on WTM_DECLARED_SUITES).
#
# ONE FILE, EVERY RANK COUNT -- which is the point: the comparison is only meaningful if both sides
# solve the identical problem, and a single shared file is what guarantees that.
#
# surface_water.routing: continuous is stated there now. It had been ABSENT and resolving to the
# default, so this suite -- whose entire subject is the rank-0 FSM path -- never actually said it was
# running FSM at all. If the default had moved to off, it would have kept passing while measuring
# nothing, the way the ghost_cell fixture cannot catch FSM bugs because its table stays below ground.
mkcfg() { # nranks -> writes $WORK/n<nranks>.yaml
    local n="$1"
    sed -e "s|@INPUTS@|$SD|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|n${n}|g" config.yaml > "$WORK/n${n}.yaml"
}

# -wtm_eq_tol 0: pin the full fixed cycle count so the cross-rank comparison is at the same cycle (the
# equilibrium auto-stop default could otherwise fire at slightly MPI-decomposition-dependent cycles).
run() { local n="$1"; mkcfg "$n"; ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "n${n}.yaml" >"$WORK/n${n}.log" 2>&1 ); }

echo "=== FillSpillMerge MPI-consistency regression ==="
echo "binary: $WTM   rank counts vs n=1: ${RANKS[*]}"
run 1
fail=0
for n in "${RANKS[@]}"; do
    run "$n"
    if python3 ../mpi_consistency/compare.py "$WORK/n1_" "$WORK/n${n}_"; then
        printf "  fsm n=1 vs n=%-2s : PASS\n" "$n"
    else
        printf "  fsm n=1 vs n=%-2s : FAIL\n" "$n"; fail=1
    fi
done
echo
if [[ $fail -eq 0 ]]; then echo "FSM CONSISTENCY PASSED"; else echo "FSM CONSISTENCY FAILED" >&2; fi
exit $fail
