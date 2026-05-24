#!/usr/bin/env bash
set -euo pipefail

WTM=../build/wtm.x
cd "$(dirname "$0")"

OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_NUM_THREADS

echo "=== WTM solver benchmark (1000x1000 grid, 3 cycles, OMP_NUM_THREADS=$OMP_NUM_THREADS) ==="
echo

run_and_measure() {
    local label="$1"; shift
    local cfg="$1";  shift
    local snes_args=("$@")

    echo "--- $label ---"
    # /usr/bin/time -v reports peak RSS; redirect its stderr to stdout for capture
    local tmpout
    tmpout=$(mktemp)
    /usr/bin/time -v \
        "$WTM" "$cfg" "${snes_args[@]}" \
        2>&1 | tee "$tmpout"

    echo
    echo "  Peak RSS (KB): $(grep 'Maximum resident' "$tmpout" | awk '{print $NF}')"
    echo "  Wall clock:    $(grep 'Elapsed (wall' "$tmpout" | awk '{print $NF}')"
    rm -f "$tmpout"
    echo
}

run_and_measure \
    "Anderson (matrix-free)" \
    config_anderson.cfg \
    -snes_mf -snes_type anderson -snes_stol 1e-6

run_and_measure \
    "Newton-Krylov (FD Jacobian + GAMG)" \
    config_newton.cfg \
    -snes_type newtonls -snes_rtol 1e-6 -snes_atol 1e-8
