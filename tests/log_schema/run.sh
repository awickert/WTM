#!/usr/bin/env bash
# PIN THE RUN-LOG HEADER, and prove the trace parser cannot be fooled by a substring.
#
# WHY. The run log's column names are emitted once, in src/WTM.cpp. Four budget suites read that log,
# and until tests/wtm_log.py they read it POSITIONALLY -- r[8], r[16]. Nothing asserted the mapping,
# so inserting a column would silently reindex every budget assertion in the suite, and several would
# still report PASS: a residual column read as a recharge column is still a number. The failure would
# not announce itself. This arm makes a column change fail LOUDLY, in ONE place, and say so.
#
# The pin is deliberately the FULL ORDERED LIST, not a subset. A subset pin would let a column be
# inserted in the middle without complaint, which is the exact case that breaks positional readers.
# When you DO add a column: append it at the END (positional readers survive), then update the list
# here in the same commit. The diff is then the record that the change was intended.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
PY="${PY:-python3}"
GHOST=$(readlink -f ../ghost_cell/inputs)
[[ -f "$GHOST/ghost_cell_test_t0_topography.tif" ]] || ( cd ../ghost_cell && python3 make_inputs.py >/dev/null )
make_work logschema
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/log_schema/config.yaml, read and edited directly rather than
# translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# Two of those settings are LOAD-BEARING for what this suite asserts, and were previously implicit:
# solver.time_step.mode: adaptive (no adaptive controller, no DTTRACE lines, and the substring-proof
# check below has nothing to parse) and output.trace: [dt] (the trace itself).
sed -e "s|@INPUTS@|$GHOST|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|pin|g" config.yaml > "$WORK/pin.yaml"
"$WTM" "$WORK/pin.yaml" > "$WORK/pin.log" 2>&1 \
  || { echo "FAIL: the pin run did not complete"; tail -5 "$WORK/pin.log"; exit 2; }

TESTS="$(readlink -f ..)" "$PY" - "$WORK/pin.txt" "$WORK/pin.log" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["TESTS"])
import wtm_log as L

EXPECTED = [
    "Cycles_done", "Total_wtd_change", "Change_in_GW_only", "Change_in_SW_only",
    "absolute_value_total_wtd_change", "abs_change_in_GW", "abs_change_in_SW",
    "change_in_infiltration", "total_recharge_added", "total_loss_to_ocean",
    "sum_of_water_tables", "total_surface_removed", "total_ocean_outflow",
    "stored_volume", "ocean_loss_closing", "budget_residual", "exact_budget_residual",
    "total_evap_removed", "recharge_direct", "runoff_to_surface",
    "elapsed_time_s", "solves_done", "rejects_done",
    # Appended for #65: the per-cycle change in WATER VOLUME (|S*dwtd|), the units the equilibrium
    # stop, the adaptive step target and the budget all use. Column 5 stays the HEAD change.
    "abs_change_volume_max", "abs_change_volume_rms",
    # Appended for #105: the off-map ghost flux under boundaries.land: neumann_toposlope, signed
    # + for INFLOW. It is a mass SOURCE the model books, and until this column existed it was
    # visible only inside exact_budget_residual -- a sum it shares with five other terms, where
    # two errors can cancel and read as a closed budget.
    "boundary_inflow_gw",
]
fail = 0
def check(name, ok, detail):
    global fail
    print(f"  {'OK  ' if ok else 'FAIL'} {name}  {detail}")
    if not ok: fail = 1

got = L.header_names(sys.argv[1])
check("HEADER PINNED", got == EXPECTED,
      f"{len(got)} columns" if got == EXPECTED else
      f"header changed.\n       expected: {' '.join(EXPECTED)}\n       got:      {' '.join(got)}")

# The reader must reach columns BY NAME and refuse a name that is not there.
log = L.read_log(sys.argv[1])
check("BY NAME", isinstance(log.last("exact_budget_residual"), float),
      f"exact_budget_residual = {log.last('exact_budget_residual'):.3e}")
try:
    log.col("no_such_column"); check("REFUSES UNKNOWN", False, "read a column that does not exist")
except KeyError:
    check("REFUSES UNKNOWN", True, "an absent column raises instead of returning a neighbour")

# POSITIVE CONTROL for the substring trap: est= must not be read out of nest=.
rows = L.read_trace(sys.argv[2], "DTTRACE")
if not rows:
    check("TRACE SUBSTRING-PROOF", False, "no DTTRACE lines to check (did the trace flag stop working?)")
else:
    r = rows[0]
    has_both = "est" in r and "nest" in r
    check("TRACE SUBSTRING-PROOF", has_both and r["est"] != r["nest"],
          f"est={r.get('est')} parsed independently of nest={r.get('nest')}")
sys.exit(fail)
PYEOF
rc=$?
[ "$rc" -eq 0 ] && echo "LOG SCHEMA: ALL PASSED" || echo "LOG SCHEMA: FAILED"
exit $rc
