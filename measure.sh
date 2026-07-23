#!/usr/bin/env bash
# measure.sh -- time + peak-memory sweep for a WTM binary, for before/after comparisons.
#
# Usage:  source msi_env.sh          # load the toolchain first
#         ./measure.sh <label> <path/to/wtm.x> <config.cfg> [ranks ...]
#   e.g.  ./measure.sh kcallaghan ../WTM-kcallaghan/build/wtm.x prod.cfg 1
#         ./measure.sh after      build/wtm.x                   prod.cfg 1 2 4 8 16 32
#
# For each rank count it runs the model with PETSc -memory_view and prints one
# row: wall time (s) and process-memory total/max/min (bytes, summed / largest /
# smallest rank). Memory is allocation-driven, so a config with total_cycles=1,
# maxiter=1 gives the true footprint in one solve. KCallaghan is only meaningful
# at n=1 (its ghost-cell bug mis-runs at >1 rank) -- run it at 1 for the baseline.
set -u

LABEL=${1:?need a label}; WTM=$(readlink -f "${2:?need wtm.x}"); CFG=$(readlink -f "${3:?need config}")
shift 3
RANKS=("$@"); [ ${#RANKS[@]} -eq 0 ] && RANKS=(1 2 4 8)

printf '%-12s %-4s %-9s %-6s   %s\n' LABEL n wall_s rc "process-memory (total / max / min)"
for n in "${RANKS[@]}"; do
    log=$(mktemp)
    t0=$(date +%s.%N)
    mpiexec -n "$n" "$WTM" "$CFG" -memory_view >"$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
    # PETSc prints e.g.: "Maximum (over computational time) process memory: total 1.23e+09 max 3.1e+08 min 2.9e+08"
    mem=$(grep -iE 'process memory' "$log" | grep -iE 'total' | head -1 | sed -E 's/.*(total.*)/\1/')
    printf '%-12s %-4s %-9s %-6s   %s\n' "$LABEL" "$n" "$wall" "$rc" "${mem:-<no -memory_view line; see log>}"
    # keep failed logs for inspection
    [ "$rc" -eq 0 ] && rm -f "$log" || echo "    (rc=$rc; log kept: $log)"
done
