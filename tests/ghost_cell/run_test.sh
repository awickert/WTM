#!/usr/bin/env bash
# Ghost-cell MPI validation test.
#
# Runs the WTM on a 102x3 heterogeneous-ksat domain with 1 and then 2 MPI
# processes, then compares the output TIFs.  With the ghost-cell fix the two
# runs agree; without it they diverge at the MPI processor boundary.
#
# Usage:
#   cd tests/ghost_cell
#   ./run_test.sh [path/to/wtm.x]
#
# Default binary: ../../build/wtm.x

set -euo pipefail
cd "$(dirname "$0")"

# Pin OpenMP to one thread per rank (as the other test runners do). Without this,
# each MPI rank spawns a thread per core; on a many-core node the thread
# spawn/sync overhead for this tiny grid dominates, and at n>1 with a busy-wait
# MPI (e.g. MPICH on MSI) the oversubscription makes the run crawl.
export OMP_NUM_THREADS=1

WTM=${1:-../../build/wtm.x}

if [[ ! -x "$WTM" ]]; then
    echo "ERROR: WTM binary not found at $WTM" >&2
    echo "Build the project first (cmake --build build) or supply the path as \$1." >&2
    exit 1
fi

echo "=== Ghost-cell MPI validation test ==="
echo "WTM binary: $WTM"
echo

# 1. Generate synthetic inputs
echo "--- Generating inputs ---"
python3 make_inputs.py

# 2. One-process reference run
echo
echo "--- 1-process reference run ---"
rm -rf out_1p run_1p.txt
mkdir -p out_1p
# outfile_prefix and textfilename are relative to CWD when wtm.x is called.
# Override them with sed-generated temp configs so both runs share the same
# base config without modifying it.
CFG_1P=$(mktemp /tmp/ghost_cell_1p_XXXXXX.yaml)
sed 's|^  outfile_prefix:.*|  outfile_prefix: out_1p/out_|;
     s|^  run_log:.*|  run_log: run_1p.txt|' ghost_cell.yaml > "$CFG_1P"
# A CRASHED RUN MUST NOT REACH THE COMPARISON. This used to be a pipeline ending in `|| true`,
# which threw mpirun's exit status away twice over -- once through the pipe, and again because
# running `true` REPLACES PIPESTATUS. A run that aborted after writing only its 0 yr output still
# left a TIF behind, check_results.py picked that one, and the suite reported PASS on two copies of
# the INITIAL CONDITION. A vacuous pass is worse than a failure.
#
# Redirect to a file rather than piping: under `set -euo pipefail` a pipeline whose grep matches
# NOTHING exits 1, which would abort a perfectly good run. Separating the two keeps mpirun's status
# authoritative and leaves the grep purely cosmetic.
mpirun -n 1 "$WTM" "$CFG_1P" -snes_stol 1e-8 > run_1p.out 2>&1 \
  || { echo "FAIL: the 1-process run did not complete (exit $?)"; tail -20 run_1p.out; exit 2; }
grep -E 'SNES|converged|norm|Error|error' run_1p.out || true
echo "1-process run complete."
rm -f "$CFG_1P"

# 3. Two-process run (processor boundary at the ksat discontinuity)
echo
echo "--- 2-process run (split at ksat boundary) ---"
rm -rf out_2p run_2p.txt
mkdir -p out_2p
CFG_2P=$(mktemp /tmp/ghost_cell_2p_XXXXXX.yaml)
sed 's|^  outfile_prefix:.*|  outfile_prefix: out_2p/out_|;
     s|^  run_log:.*|  run_log: run_2p.txt|' ghost_cell.yaml > "$CFG_2P"
mpirun -n 2 "$WTM" "$CFG_2P" -snes_stol 1e-8 \
    -da_processors_x 2 -da_processors_y 1 > run_2p.out 2>&1 \
  || { echo "FAIL: the 2-process run did not complete (exit $?)"; tail -20 run_2p.out; exit 2; }
grep -E 'SNES|converged|norm|Error|error' run_2p.out || true
echo "2-process run complete."
rm -f "$CFG_2P"

# 4. Compare outputs
echo
echo "--- Comparing outputs ---"
python3 check_results.py
