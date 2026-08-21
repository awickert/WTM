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

WTM=${1:-../../build/wtm.x}
shift || true
RANKS=("$@")
if [[ ${#RANKS[@]} -eq 0 ]]; then RANKS=(2 4); fi

WTM_ABS=$(readlink -f "$WTM")
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

WORK=$(mktemp -d /tmp/mpi_consistency_XXXX)
trap 'rm -rf "$WORK"' EXIT

# Base config (equilibrium, small grid). evap_mode and fsm_on are overridden per case.
base_cfg() {
    cat <<EOF
run_type           equilibrium
fsm_on             __FSM__
evap_mode          __EVAP__
infiltration_on    0
runoff_ratio_on    0
cells_per_degree   10
southern_edge      -45
deltat             31536000
total_time       8yr
report_interval            2
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        $(readlink -f "$INPUTS")
region             ghost_cell_test
supplied_wt        0
textfilename       __TXT__
outfile_prefix     __OUT__
save_nreport_interval     9999
EOF
}

run_case() { # evap fsm nranks tag
    local evap="$1" fsm="$2" n="$3" tag="$4"
    local cfg="$WORK/${tag}.cfg"
    base_cfg | sed "s|__EVAP__|$evap|; s|__FSM__|$fsm|; s|__TXT__|$WORK/${tag}.txt|; s|__OUT__|$WORK/${tag}_|" > "$cfg"
    # -wtm_eq_tol 0: pin the full fixed cycle count so the n=1-vs-n=N comparison is at the same cycle
    # (the equilibrium auto-stop default could otherwise fire at slightly MPI-decomposition-dependent cycles).
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM_ABS" "$cfg" -snes_stol 1e-8 -wtm_eq_tol 0 >"$WORK/${tag}.log" 2>&1 )
}

echo "=== MPI-consistency regression ==="
echo "binary: $WTM_ABS   rank counts vs n=1: ${RANKS[*]}"
echo

fail=0
for evap in 0 1; do
  for fsm in 0 1; do
    label="evap${evap}_fsm${fsm}"
    run_case "$evap" "$fsm" 1 "${label}_n1"
    for n in "${RANKS[@]}"; do
      run_case "$evap" "$fsm" "$n" "${label}_n${n}"
      if python3 compare.py "$WORK/${label}_n1" "$WORK/${label}_n${n}"; then
        printf "  %-14s n=1 vs n=%-2s : PASS\n" "$label" "$n"
      else
        printf "  %-14s n=1 vs n=%-2s : FAIL\n" "$label" "$n"
        fail=1
      fi
    done
  done
done

echo
if [[ $fail -eq 0 ]]; then echo "ALL CONSISTENCY CHECKS PASSED"; else echo "CONSISTENCY CHECKS FAILED" >&2; fi
exit $fail
