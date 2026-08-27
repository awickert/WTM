#!/usr/bin/env bash
# SOLVE-COUNT INVARIANCE: the same physical problem, integrated with different numbers of steps, must
# produce the same cumulative water budget.
#
# THE INVARIANT THIS PINS:
#
#     No cumulative quantity may be proportional to the SOLVE COUNT.
#     Every one must be proportional to ELAPSED TIME, or be a difference of states.
#
# WHY IT EXISTS. Three separate defects violated exactly that, and none of them was caught by anything
# in this suite, because every existing budget test runs at a single fixed dt where the distinction is
# invisible (rech_dt_scale is exactly 1, and one solve == one nominal step):
#   * the adaptive controller wrote the NEXT step's dt into user_context.deltat before the current
#     step's accounting had consumed it, so five accumulators read the wrong dt (f84126b);
#   * column 9 booked the UNSCALED rech_dist, so it tracked the solve count rather than elapsed time
#     (d42d844);
#   * the runoff-ratio channel was DELIVERED to FillSpillMerge at nominal-step size on every accepted
#     sub-step, so the model routed water in proportion to solve count -- a MASS error, not a reporting
#     one (task #15). Carried here as an `xfail` with a guard until fixed; the guard is what reported
#     the fix. Both runoff_ratio blocks are now full invariance checks.
# Each showed up as a column tracking solves instead of time. This test is the general gate: it does
# not know about any particular bug, only about the invariant.
#
# HOW IT DISCRIMINATES. Adaptive dt covers the same cycle duration as fixed dt but with a different
# number of sub-steps, so elapsed time is identical while the solve count is not. A correct cumulative
# column is then invariant; a solve-count-proportional one moves by the ratio of the counts. Two
# tolerances, both measured rather than invented (see the tables below): pure INPUT channels are
# exactly invariant, while trajectory-dependent columns carry ordinary truncation error.
#
# Usage:  tests/dt_invariance/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/dtinv_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

mkcfg() { # $1 = stem, $2 = runoff_ratio
    # dt_tol travels in the CONFIG now (solver.water_volume_timestep_error_tol); DT_TOL= per arm.
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
total_time 20yr
supplied_wt 1
deltat 31536000
report_interval 5
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_ratio $2
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
${DT_TOL:+dt_tol $DT_TOL}
${ADAPT:+adaptive_dt true}
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

run() { # $1 = stem, $2 = runoff_ratio, $3.. = extra flags
    local stem="$1" rr="$2"; shift 2
    mkcfg "$stem" "$rr"; rm -f "$WORK/$stem.txt"
    if ! "$WTM" "$WORK/$stem.yaml" -wtm_anderson -wtm_tr_bdf2 "$@" -snes_stol 1e-8 -wtm_eq_tol 0 \
            > "$WORK/$stem.log" 2>&1; then
        echo "  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; return 1
    fi
}

echo "=== solve-count invariance of the cumulative water budget ==="
echo "WTM binary: $WTM"
echo
fail=0
# Three solve counts per block: fixed dt, and adaptive at a loose and a tight step tolerance.
for rr_tag in "0:z" "0.3:r"; do
    rr="${rr_tag%%:*}"; p="${rr_tag##*:}"
    run "${p}fx"    "$rr"                                    || fail=1
    ADAPT=1 DT_TOL=0.5  run "${p}ad_lo" "$rr" || fail=1
    ADAPT=1 DT_TOL=0.02 run "${p}ad_hi" "$rr" || fail=1
done
[[ $fail -eq 0 ]] || { echo "DT INVARIANCE: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" "$PY" - <<'PY'
import os, sys
W = os.environ["WORK"]
YEAR = 31536000.0
# (index, label). Columns are 1-indexed in the header; these are the 0-indexed positions.
INPUTS = [(18, "19 recharge_direct"), (19, "20 runoff_to_surface"), (8, "9 total_recharge_added")]
TRAJ   = [(9, "10 total_loss_to_ocean"), (11, "12 total_surface_removed"),
          (12, "13 total_ocean_outflow"), (13, "14 stored_volume"), (17, "18 total_evap_removed")]
# Tolerances MEASURED on this fixture with the routed channel off, where every spread is pure
# truncation: max observed 3.5e-03 (col 12), so 1e-2 carries ~3x margin. The input channels are
# exactly invariant here because this fixture's recharge does not depend on the water table, which is
# what makes them a clean probe -- asserted below so the test cannot silently stop discriminating.
TOL_INPUT, TOL_TRAJ = 1e-9, 1e-2
EXPECT_YEARS = 20.0  # must match `total_time` in mkcfg above

def last(stem):
    rows = [[float(x) for x in l.split()] for l in open(f"{W}/{stem}.txt")
            if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
    return rows[-1] if rows else None

def spread(vals):
    ref = abs(vals[0]) or 1.0
    return max(abs(v - vals[0]) for v in vals) / ref

fail = 0
for p, rr, label in (("z", "0", "routed channel OFF (runoff_ratio 0)"),
                     ("r", "0.3", "routed channel ON  (runoff_ratio 0.3)")):
    stems = [f"{p}fx", f"{p}ad_lo", f"{p}ad_hi"]
    rows = [last(s) for s in stems]
    if any(r is None for r in rows):
        print(f"  FAIL  {label} -- missing output"); fail = 1; continue
    elapsed = [r[20] / YEAR for r in rows]
    solves  = [int(r[21]) for r in rows]
    print(f"-- {label} --")
    print(f"        elapsed_yr {[round(e,4) for e in elapsed]}   solves {solves}")

    # PRECONDITIONS. Without these the comparison proves nothing.
    # ABSOLUTE elapsed time, not just agreement between arms: the config asks for 20 yr, so every arm
    # must report 20 yr. Agreement alone would pass happily if all three were off by the same
    # cycles_done off-by-one, and every rate a reader derives from this file divides by this number.
    ok = max(abs(e - EXPECT_YEARS) for e in elapsed) < 1e-9
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  elapsed time is the configured "
          f"{EXPECT_YEARS:g} yr in every arm {[round(e, 6) for e in elapsed]}")
    ok = max(abs(e - elapsed[0]) for e in elapsed) < 1e-9
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  every arm covers the same elapsed time")
    ok = len(set(solves)) > 1
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  PRECONDITION  solve counts actually differ {solves} "
          f"(if these ever match, this test proves nothing)")

    # INPUT channels: driven by the forcing and elapsed time, so exactly invariant on this fixture.
    for idx, name in INPUTS:
        s = spread([r[idx] for r in rows])
        ok = s < TOL_INPUT
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  INPUT   {name:<26} spread {s:.3e}  (tol {TOL_INPUT:.0e})")

    # col 9 must be exactly the sum of the two channels, in every arm.
    ok = all(abs(r[8] - (r[18] + r[19])) <= 1e-11 * max(1.0, abs(r[8])) for r in rows)
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CONSISTENT  col 9 == col 19 + col 20 in every arm")

    for idx, name in TRAJ:
        s = spread([r[idx] for r in rows])
        ok = s < TOL_TRAJ
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  TRAJ    {name:<26} spread {s:.3e}  (tol {TOL_TRAJ:.0e})")
    print()

print("DT INVARIANCE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
