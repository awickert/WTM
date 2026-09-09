#!/bin/bash
# COUPLING CONVERGENCE: impulse and continuous must agree in the dt -> 0 limit.
#
# WHY THIS EXISTS. surface_water.fsm_coupling picks HOW FillSpillMerge's result reaches the
# groundwater solve: `impulse` overwrites the step baseline with the post-FSM table, `continuous`
# carries FSM's per-cell volume change into the next step's recharge. These are two DISCRETISATIONS
# OF THE SAME PHYSICS -- instantaneous routing -- so they must converge to the same answer as the
# step shrinks. Nothing asserted that. It was assumed.
#
# It is worth asserting because the assumption was WRONG for a while and nobody noticed. The comment
# that justified making `continuous` the default recorded the symptom without recognising it:
#     "the gap GROWS with refinement instead of vanishing, because it is a difference in the physics
#      encoded, not a timing artifact"
# The growth was real. The reading was not: it was a DEFECT, not encoded physics. Under `continuous`
# the active-set obstacle was destroying water that the FSM delta was separately moving -- a double
# removal, fixed in 19ee097.
#
# SHOWN TO BITE, which is the only reason to trust it. Reverting 19ee097 and re-running, the gap
# GROWS on every column instead of shrinking:
#     stored_volume     3.771e-01 -> 4.711e-01 -> 5.168e-01   (x0.80, x0.91)
#     evap_removed      2.835e-02 -> 4.692e-02 -> 5.623e-02   (x0.60, x0.83)
#     surface_removed   2.456e+00 -> 2.656e+00 -> 2.756e+00   (x0.92, x0.96)
#     ocean_outflow     8.813e-02 -> 1.393e-01 -> 1.670e-01   (x0.63, x0.83)
# i.e. this test reproduces the exact signature the old comment recorded, and fails on it.
#
# What it does NOT catch, so nobody assumes otherwise: the rech_dt_scale double-scaling of the FSM
# delta (fixed in 69a0d0c) is INERT here by construction. Every arm is fixed-dt, so rech_dt_scale is
# exactly 1 and the bug cannot express itself. Re-introducing it changes nothing in this test --
# verified, not assumed. Catching that one needs a VARIABLE-dt arm; tests/dt_invariance and
# tests/budget_closure cover it.
#
# WHAT IT ASSERTS, and what it deliberately does not:
#   1. CONSERVATION  -- every run closes its own exact budget identity. This is the claim the model
#                       OWES, and it is asserted for BOTH couplings here because tests/dt_invariance
#                       is now scoped to impulse.
#   2. NON-VACUOUS   -- the couplings must actually DIFFER at the coarsest step, or convergence is
#                       trivially satisfied by two identical runs and this test proves nothing.
#   3. CONVERGES     -- the gap must shrink MONOTONICALLY under refinement, on every column. Stated
#                       without a rate, because a rate would be a number invented to fit.
#   4. FIRST ORDER   -- on the two LARGEST gaps (stored_volume, evap_removed) the gap must shrink by
#                       at least 1.5x per halving. Measured 2.58/2.58 and 2.15/1.77, so 1.5 carries
#                       margin. Not applied to the small-gap columns: ocean_outflow's first ratio is
#                       1.33, which is convergence but not a clean first order, and asserting 1.5
#                       there would be asserting more than the data shows.
#
# Usage:  tests/coupling_convergence/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
MLDIR=$(readlink -f ../multilake)
CASDIR=$(readlink -f ../fsm_cascade)
for d in "$FSMDIR" "$MLDIR" "$CASDIR"; do
    [[ -d "$d/inputs" ]] || ( cd "$d" && python3 make_inputs.py >/dev/null )
done
make_work cconv
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

mkcfg() { # $1 = stem, $2 = coupling, $3 = deltat seconds, $4 = report_interval, $5 = inputs, $6 = region
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
snes_stol 1e-10
solver_method anderson
run_type equilibrium
time_integration tr-bdf2
total_time 8yr
supplied_wt 1
deltat $3
report_interval $4
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
fsm_coupling $2
runoff_ratio 0
adaptive_dt false
surfdatadir $5
region $6
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

echo "=== coupling convergence: impulse and continuous must agree as dt -> 0 ==="
echo "WTM binary: $WTM"
echo

# FIXED dt on every arm -- adaptive_dt is pinned false above. This is a refinement study, so the step
# has to be the thing being varied, not something the controller chooses. (An omitted adaptive_dt
# resolves to `auto` -> TRUE, which is how tests/dt_invariance lost its fixed-dt control arm.)
# THREE FIXTURES, because one was not enough. The original evidence for this default came entirely
# from `lake`, which is built around a filling depression -- the most favourable case for the coupling
# to matter, as task #43 itself flagged. Measured at a matched 8 yr, the coupling gap is 40-55x SMALLER
# on the other two, even though fsm_cascade moves NINE TIMES more surface water than lake does:
#     fixture       evap gap    stored gap   surface water moved (/recharge)
#     lake          5.428e-02   4.477e-02    9.760e-02
#     multilake     9.935e-04   1.008e-03    8.088e-02
#     cascade       1.345e-03   0.000e+00    8.505e-01
# So the size of the effect is NOT set by how much water FSM routes. It is specific to starting with a
# water table above the surface over a plateau, which is what `lake` does and the others do not.
fail=0
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
for fixture in "lake:$FSMDIR/inputs:fsm_test" "multilake:$MLDIR/inputs:multilake" "cascade:$CASDIR/inputs:fsm_cascade"; do
IFS=: read -r fxname fxinp fxregion <<<"$fixture"
for spec in "1yr:31536000:2" "05yr:15768000:4" "025yr:7884000:8"; do
    IFS=: read -r tag dt ri <<<"$spec"
    for cp in impulse continuous; do
        stem="${fxname}_${tag}_${cp}"
        mkcfg "$stem" "$cp" "$dt" "$ri" "$fxinp" "$fxregion"; rm -f "$WORK/$stem.txt"
        if ! WTM_COVERAGE_TAG="coupling_convergence/$stem" "$WTM" "$WORK/$stem.yaml" \
                > "$WORK/$stem.log" 2>&1; then
            echo "  FAIL  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1
        # This suite's ENTIRE subject is the difference between the two couplings. If one silently
        # resolved to the other -- which the model does do, and announces, when the collector is
        # `explicit` -- every pair below would compare a run with itself and the convergence orders
        # would agree perfectly while measuring nothing (#24). Check against the fingerprint the model
        # writes, not against the config we think we wrote.
        elif ! expect_resolved "$WTM_COVERAGE_LOG" "coupling=$cp" >/dev/null 2>&1; then   # its own message is redundant with the one below
            echo "  FAIL  $stem asked for coupling=$cp but the run resolved otherwise:"
            command grep '^coverage ' "$WTM_COVERAGE_LOG" | tail -1 | sed 's/^/        /'
            fail=1
        fi
    done
done; done
[[ $fail -eq 0 ]] || { echo "COUPLING CONVERGENCE: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" TESTS="$(readlink -f ..)" "$PY" - <<'PYX' || fail=1
import os, sys
W = os.environ["WORK"]
TAGS = ["1yr", "05yr", "025yr"]                       # each half the previous
# COLUMNS BY NAME. This map used to be hand-written 0-based indices with the column number baked into
# each label, so a column inserted upstream would have shifted every entry onto its neighbour AND left
# the label confidently wrong. Built from the header the run wrote; tests/log_schema pins that header.
import glob
sys.path.insert(0, os.environ["TESTS"])
import wtm_log as LOG
I = LOG.index_map(sorted(glob.glob(f"{W}/*.txt"))[0])
NAME = {I[n]: f"{I[n] + 1} {lbl}" for n, lbl in
        (("total_loss_to_ocean", "loss_to_ocean"), ("total_surface_removed", "surface_removed"),
         ("total_ocean_outflow", "ocean_outflow"), ("stored_volume", "stored_volume"),
         ("total_evap_removed", "evap_removed"))}
RECH_COL, RESID_COL = I["total_recharge_added"], I["exact_budget_residual"]
TOL_CONSERVE, MIN_GAP, MIN_RATE = 1e-6, 1e-2, 1.5
# The exact budget identity does NOT close on multilake at the coarsest step: 1.802e-05 of recharge,
# BIT-IDENTICAL under both couplings, collapsing to 4.8e-10 when dt halves and 8.2e-11 at dt/4. It is
# therefore a property of that fixture at that step size, NOT of the coupling -- which is why this
# test found it and tests/budget_closure (a different fixture) never did. Held, not hidden. Task #52.
XFAIL_CONSERVE = {("multilake", "1yr"): 1e-6}

# WHAT EACH FIXTURE IS ASKED TO SHOW. Deliberately different, because the fixtures behave
# differently and asserting one policy on all three would either be vacuous on two of them or
# claim a trend the data does not show on the third. Every number below is measured.
#
#   lake       the coupling MATTERS here (gap ~5e-2) and converges first order. Full policy.
#   cascade    spill-dominated. evap/surface_removed converge cleanly (x2.00, x1.97); storage and
#              ocean outflow are EXACTLY equal under both couplings, because the sills set them.
#   multilake  the gap is SMALL (~1e-3) and does NOT converge over this range (x0.91, x1.24). It is
#              held to a BOUND, not a trend. That non-convergence is an open observation, not a
#              known-good behaviour -- see task #48.
POLICY = {
    "lake":      {"nonvacuous": [13, 17], "converge": [13, 17, 11, 12], "rate": [13, 17]},
    "cascade":   {"identical":  [13, 12], "converge": [17, 11],         "rate": [17, 11]},
    "multilake": {"bounded":    ([13, 17, 11, 12], 5e-3)},
}

def last(stem):
    rows = [[float(x) for x in l.split()] for l in open(f"{W}/{stem}.txt")
            if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
    return rows[-1] if rows else None

fail = 0
for fx, pol in POLICY.items():
    runs = {(t, c): last(f"{fx}_{t}_{c}") for t in TAGS for c in ("impulse", "continuous")}
    if any(v is None for v in runs.values()):
        print(f"  FAIL  {fx}: a run produced no data rows"); fail = 1; continue
    print(f"-- fixture: {fx} --")

    # CONSERVATION -- both couplings, every step size, every fixture. The claim the model owes.
    # Reported PER STEP SIZE, not as a single worst-case, because on multilake it is the COARSEST
    # step that misbehaves and a single number would hide which.
    for t in TAGS:
        w = max(abs(runs[(t, c)][RESID_COL]) / (abs(runs[(t, c)][RECH_COL]) or 1.0) for c in ("impulse", "continuous"))
        xf = XFAIL_CONSERVE.get((fx, t))
        if xf is None:
            ok = w < TOL_CONSERVE
            fail |= not ok
            print(f"  {'PASS' if ok else 'FAIL'}  CONSERVATION  {t:<6} worst |exact residual|/recharge "
                  f"{w:.3e}  (tol {TOL_CONSERVE:.0e})")
        else:
            # Held as an EXPECTED failure with a floor, so it keeps a regression test rather than
            # being tuned away -- and so the suite FAILS the day it starts closing, which means the
            # defect is fixed and this arm must be promoted.
            still = w > xf
            fail |= not still
            print(f"  {'xfail' if still else 'FAIL '}   CONSERVATION  {t:<6} |exact residual|/recharge "
                  f"{w:.3e} -- KNOWN, task #52" +
                  ("" if still else f"  <-- NOW CLOSES (below {xf:.0e}): promote this arm"))

    gap = {c: [abs(runs[(t, 'impulse')][c] - runs[(t, 'continuous')][c]) /
               (abs(runs[(t, 'impulse')][RECH_COL]) or 1.0) for t in TAGS] for c in NAME}

    for c in pol.get("nonvacuous", []):
        ok = gap[c][0] > MIN_GAP
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  NON-VACUOUS   {NAME[c]:<20} gap at the coarsest dt "
              f"{gap[c][0]:.3e}  (need > {MIN_GAP:.0e}, else the couplings stopped differing here)")

    for c in pol.get("identical", []):
        ok = max(gap[c]) == 0.0
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  IDENTICAL     {NAME[c]:<20} both couplings agree EXACTLY "
              f"at every dt (worst {max(gap[c]):.3e}) -- the sills set this, not the coupling")

    for c in pol.get("converge", []):
        g = gap[c]
        ok = g[0] > g[1] > g[2]
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  CONVERGES     {NAME[c]:<20} "
              f"{g[0]:.3e} -> {g[1]:.3e} -> {g[2]:.3e}  (x{g[0]/g[1]:.2f}, x{g[1]/g[2]:.2f})")

    for c in pol.get("rate", []):
        g = gap[c]
        r1, r2 = g[0] / g[1], g[1] / g[2]
        ok = r1 >= MIN_RATE and r2 >= MIN_RATE
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  FIRST ORDER   {NAME[c]:<20} shrinks >= {MIN_RATE}x per "
              f"halving (x{r1:.2f}, x{r2:.2f})")

    if "bounded" in pol:
        cols, bound = pol["bounded"]
        for c in cols:
            g = gap[c]
            ok = max(g) < bound
            fail |= not ok
            print(f"  {'PASS' if ok else 'FAIL'}  BOUNDED       {NAME[c]:<20} "
                  f"{g[0]:.3e} -> {g[1]:.3e} -> {g[2]:.3e}  (worst < {bound:.0e}; NOT asserted to "
                  f"converge -- it does not, over this range)")
    print()

print("COUPLING CONVERGENCE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PYX
exit $?
