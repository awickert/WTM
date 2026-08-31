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

WTM=$(readlink -f "${1:-../../build/wtm.x}")
shift || true
RANKS=("$@"); [[ ${#RANKS[@]} -eq 0 ]] && RANKS=(2 4)

if [[ ! -x "$WTM" ]]; then echo "ERROR: WTM binary not found at $WTM" >&2; exit 1; fi

SD=$(readlink -f inputs)
if [[ ! -f "$SD/fsm_test_t0_topography.tif" ]]; then
    echo "Generating FSM test inputs..."
    python3 make_inputs.py >/dev/null
fi

WORK=$(mktemp -d /tmp/fsm_consistency_XXXX)
trap 'rm -rf "$WORK"' EXIT

mkcfg() { # nranks -> writes $WORK/n<nranks>.yaml, echoes prefix
    local n="$1"
    sed "s|__TXT__|$WORK/n${n}.txt|; s|__OUT__|$WORK/n${n}_|; s|__SD__|$SD|" <<EOF | ../emit_config.sh > "$WORK/n${n}.yaml"
run_type           equilibrium
fsm_on             1
evap_mode          0
infiltration_on    0
runoff_ratio_on    0
cells_per_degree   10
southern_edge      -45
deltat             31536000
total_time       6yr
report_interval            2
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        __SD__
region             fsm_test
supplied_wt        1
eq_tol 0
textfilename       __TXT__
outfile_prefix     __OUT__
save_nreport_interval     9999
EOF
}

# -wtm_eq_tol 0: pin the full fixed cycle count so the cross-rank comparison is at the same cycle (the
# equilibrium auto-stop default could otherwise fire at slightly MPI-decomposition-dependent cycles).
run() { local n="$1"; mkcfg "$n"; ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "n${n}.yaml" -snes_stol 1e-8 >"$WORK/n${n}.log" 2>&1 ); }

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
