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
    # dt_tol travels in the CONFIG now (solver.time_step.error_tol); DT_TOL= per arm.
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
solver_method anderson
run_type equilibrium
# CONVERGE TIGHTER THAN YOU COMPARE. The FATES CANCEL assertion below is threshold-free by design: it
# says the total moves LESS than its largest single part. That only means anything once the per-arm
# solver noise is smaller than the fate differences being compared. At the default water tolerance
# (1e-8, #61) the routed-ON block gave 0.9x -- no cancellation visible -- because the noise WAS the
# signal. Measured (cancellation factor, routed-off / routed-on):
#     vol_tol 1e-8   9.4x / 0.9x       <- fails
#     vol_tol 1e-10  19.4x / 1.3x
#     vol_tol 1e-12  19.4x / 1.3x      <- IDENTICAL to 1e-10
# 1e-10 and 1e-12 agree to every digit printed, which is the proof that 1e-10 is already converged:
# what is left is the documented BDF2-startup gap the comment below describes, not solver noise.
# NOTE the routed-ON margin is only 1.3x even converged. That arm is thin, and it is thin about a REAL
# residual gap, not about tolerance -- see task #48.
convergence_water_volume_tol 1e-10
time_integration tr-bdf2
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
fsm_coupling impulse
runoff_ratio $2
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
${DT_TOL:+dt_tol $DT_TOL}
adaptive_dt $([ -n "${ADAPT:-}" ] && echo true || echo false)
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

run() { # $1 = stem, $2 = runoff_ratio, $3.. = extra flags
    local stem="$1" rr="$2"; shift 2
    mkcfg "$stem" "$rr"; rm -f "$WORK/$stem.txt"
    if ! "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 \
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

WORK="$WORK" TESTS="$(readlink -f ..)" "$PY" - <<'PY'
import os, sys, glob
W = os.environ["WORK"]
sys.path.insert(0, os.environ["TESTS"])
import wtm_log as LOG                 # columns BY NAME; the header is pinned by tests/log_schema
YEAR = 31536000.0
# COLUMNS BY NAME, not by literal index. These lists used to carry hand-written 0-based positions with
# the column number baked into the label ("19 recharge_direct"), so a column inserted upstream would
# have shifted every one of them onto its neighbour AND left the printed label confidently wrong.
# Derived from the header the run actually wrote, the label maintains itself.
I = LOG.index_map(sorted(glob.glob(f"{W}/*.txt"))[0])
def col(name): return (I[name], f"{I[name] + 1} {name}")
INPUTS = [col("recharge_direct"), col("runoff_to_surface"), col("total_recharge_added")]
TRAJ   = [col("total_loss_to_ocean"), col("total_surface_removed"),
          col("total_ocean_outflow"), col("stored_volume"), col("total_evap_removed")]
# Tolerances MEASURED on this fixture. The input channels are exactly invariant here because this
# fixture's recharge does not depend on the water table, which is what makes them a clean probe --
# asserted below so the test cannot silently stop discriminating.
TOL_INPUT, TOL_TRAJ = 1e-9, 1e-2
# Conservation floor: measured max |col17|/recharge over all arms and both couplings was 3.5e-07.
TOL_CONSERVE = 1e-6
EXPECT_YEARS = 20.0  # must match `total_time` in mkcfg above

def last(stem):
    rows = [[float(x) for x in l.split()] for l in open(f"{W}/{stem}.txt")
            if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
    return rows[-1] if rows else None

# NORMALISE BY CUMULATIVE RECHARGE, not by one arm's own value.
#
# This used to be  max|v - v[0]| / |v[0]|  -- the spread divided by the FIRST ARM'S OWN VALUE. That
# conflates "the arms disagree more" with "the quantity got smaller", and it lied badly once a flux
# started heading toward zero. Measured on the routed-channel-on block, col 12, refining dt:
#     dt        arm0 (the divisor)   max abs difference   what it REPORTED
#     1 yr           9.980e+07            3.389e+07             0.340
#     0.5 yr         3.702e+07            2.957e+07             0.799
#     0.25 yr        1.036e+07            1.361e+07             1.314
# The arms' actual disagreement IMPROVED 2.5x while the reported number got 3.9x WORSE, purely because
# the divisor collapsed 9.6x. The test was announcing a catastrophic divergence while the physics
# converged. On this scale the same series reads 8.9e-04 -> 7.8e-04 -> 3.6e-04.
#
# Cumulative recharge is the right divisor: it is the water that ENTERED the domain, it does not
# collapse, and this test PROVES it is a valid common scale a few lines below -- the INPUT assertions
# require col 9 to be identical across arms to 1e-9. Every TRAJ number is then "what fraction of the
# water that came in did the arms disagree about", which is the question worth asking.
#
# max-min rather than max|v - v[0]|: no arm is privileged as the reference.
def spread(vals, scale):
    return (max(vals) - min(vals)) / (abs(scale) or 1.0)

fail = 0
for p, rr, label in (("z", "0", "routed channel OFF (runoff_ratio 0)"),
                     ("r", "0.3", "routed channel ON  (runoff_ratio 0.3)")):
    stems = [f"{p}fx", f"{p}ad_lo", f"{p}ad_hi"]
    rows = [last(s) for s in stems]
    if any(r is None for r in rows):
        print(f"  FAIL  {label} -- missing output"); fail = 1; continue
    elapsed = [r[I["elapsed_time_s"]] / YEAR for r in rows]
    solves  = [int(r[I["solves_done"]]) for r in rows]
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
    # Every spread is normalised by CUMULATIVE RECHARGE (col 9), which the INPUT block just below
    # proves is identical across arms -- so the divisor is one number for the whole comparison.
    RECH = abs(rows[0][I["total_recharge_added"]]) or 1.0
    for idx, name in INPUTS:
        s = spread([r[idx] for r in rows], RECH)
        ok = s < TOL_INPUT
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  INPUT   {name:<26} spread/rech {s:.3e}  (tol {TOL_INPUT:.0e})")

    # col 9 must be exactly the sum of the two channels, in every arm.
    ok = all(abs(r[I["total_recharge_added"]] - (r[I["recharge_direct"]] + r[I["runoff_to_surface"]]))
             <= 1e-11 * max(1.0, abs(r[I["total_recharge_added"]])) for r in rows)
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CONSISTENT  col 9 == col 19 + col 20 in every arm")

    # CONSERVATION, per arm. What the model actually OWES: the water that came in must equal the
    # water accounted for. Col 17 is the solver's own discrete identity (storage change = recharge -
    # ocean_outflow - surface_removed), so its departure from zero is unaccounted vertical flux. This
    # is asserted SEPARATELY from the fate columns below, because the two are different claims and
    # conflating them cost a real investigation: when the fate columns disagreed it was not obvious
    # whether water was being LOST or merely partitioned differently. It was partitioned.
    for st, r in zip(stems, rows):
        c = abs(r[I["exact_budget_residual"]]) / RECH
        ok = c < TOL_CONSERVE
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  CONSERVATION  {st:<8} |exact residual|/recharge "
              f"{c:.3e}  (tol {TOL_CONSERVE:.0e})")

    # ...and the arms must agree on the TOTAL even where they disagree on the split. Differences in
    # the individual fates have to cancel: same water in, same water accounted for. Measured on the
    # routed-off block at dt=1yr, the two most-separated arms differ by -1.000e-02 in stored_volume
    # and +1.253e-02 in evap_removed -- and the fates sum to -6.227e-08. surface_removed is NOT in
    # this sum: it is an internal transfer to FillSpillMerge, which routes the water onward, so
    # counting it here would double-book.
    # Stated WITHOUT a threshold, deliberately. Summing the PHYSICAL fates carries the documented
    # BDF2-startup gap (col 16, ~2e-3 of recharge), and that gap itself varies a little with step
    # count -- so the sum does not go to machine zero and any absolute tolerance here would be a
    # number invented to fit. What IS true, and is the whole point, is that the differences CANCEL:
    # the total moves LESS than its largest single part does. Assert exactly that.
    FATES = [I[n] for n in ("stored_volume", "total_evap_removed",
                            "total_ocean_outflow", "total_loss_to_ocean")]
    base = rows[0]
    worst = max(abs(sum(r[i] - base[i] for i in FATES)) / RECH for r in rows)
    biggest_part = max(spread([r[i] for r in rows], RECH) for i in FATES)
    ok = worst < biggest_part
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  FATES CANCEL  arms differ on the SPLIT, not the TOTAL: "
          f"total moves {worst:.3e} vs largest part {biggest_part:.3e} "
          f"({biggest_part / max(worst, 1e-30):.1f}x cancellation)")

    # FATE INVARIANCE. Scoped to fsm_coupling: impulse, which is what mkcfg pins. FillSpillMerge
    # re-equilibrates the state every step under impulse, which pins the partition, so each fate is
    # individually invariant to the solve count. That is NOT true of `continuous`, where the water
    # arrives over the interval and how much of it evaporates depends on the interval -- a first-order
    # effect that vanishes under refinement (measured 1.253e-02 -> 5.539e-03 -> 1.114e-03 on col 18)
    # and is the physics that coupling was chosen for. Asserting fate invariance of `continuous` would
    # be asserting something the model does not promise; that belongs in tests/coupling_convergence.
    for idx, name in TRAJ:
        s = spread([r[idx] for r in rows], RECH)
        ok = s < TOL_TRAJ
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  FATE    {name:<26} spread/rech {s:.3e}  (tol {TOL_TRAJ:.0e})")
    print()

print("DT INVARIANCE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
