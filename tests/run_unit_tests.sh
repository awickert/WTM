#!/usr/bin/env bash
# Run the DMDA gather/scatter unit tests (test_dmda.x) across a sweep of MPI
# rank counts. The row-major layout the primitives guarantee must hold
# independent of how PETSc decomposes the grid, so running at n>1 is the whole
# point. Exit code (not just stdout) is checked on every rank count, so a
# failure on any rank -- including non-root -- is caught.
#
# Usage:  tests/run_unit_tests.sh [path/to/test_dmda.x] [ranks...]
# Default binary: ../build/test_dmda.x   Default ranks: 1 2 3 4 8
set -euo pipefail
cd "$(dirname "$0")"

BIN=${1:-../build/test_dmda.x}
shift || true
RANKS=("$@")
if [[ ${#RANKS[@]} -eq 0 ]]; then RANKS=(1 2 3 4 8); fi

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: test binary not found at $BIN (build test_dmda.x first)" >&2
    exit 1
fi

echo "=== DMDA gather/scatter unit tests ==="
echo "binary: $BIN"
fail=0
for n in "${RANKS[@]}"; do
    printf "  n=%-2s ... " "$n"
    if OMP_NUM_THREADS=1 mpirun -n "$n" "$BIN" >/tmp/unit_n${n}.log 2>&1; then
        echo "PASS  ($(grep -oE '[0-9]+ passed' /tmp/unit_n${n}.log | head -1))"
    else
        echo "FAIL"
        echo "    --- output (n=$n) ---"
        sed 's/^/    /' /tmp/unit_n${n}.log | tail -20
        fail=1
    fi
done

if [[ $fail -eq 0 ]]; then
    echo "ALL UNIT TESTS PASSED"
else
    echo "UNIT TESTS FAILED" >&2
fi
exit $fail
