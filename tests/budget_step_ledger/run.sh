#!/bin/bash
# PER-STEP WATER LEDGER, swept across the configuration matrix.
#
# WHY THIS EXISTS. The exact water identity is reported once per CYCLE (col 17), and a cycle is many
# steps. A defect confined to ONE step is diluted by every other step in the sum, and one that only
# appears at a coarse step vanishes under refinement before anyone sees it. Task #52 is exactly that
# shape: the ledger over-books removal by up to 2.8% of a step's recharge on the single step where
# surface water first appears, and the CYCLE-level number for the same run is 1.8e-05. Nothing in the
# suite looked at a single step, so it went unseen until a fixture was added that happened to cross the
# land surface at a coarse step. This test removes the luck.
#
# WHAT IT CHECKS, per accepted step, from output.trace: [budget]:
#
#   1. CROSSING EXISTS  the sweep must actually contain a step where surface water FIRST appears
#                       (d_surf transitions 0 -> nonzero). That is the condition the whole test exists
#                       for; if a fixture stops producing it, this test silently stops discriminating.
#   2. STATE == ACC     two INDEPENDENT computations of the step's storage change must agree:
#                         d_stor_acc    what the solver ACCUMULATED
#                         d_stor_state  recomputed from the STATE
#                       This is the sharpest available check that the model's story matches what it did.
#                       It is what established #52 as a FLUX-REPORTING defect and not a state error --
#                       the state is exact to 1.8e-14 across the very step where the ledger fails.
#   3. LEDGER CLOSES    |d_resid| / d_rech under tolerance, every step, every combination.
#
# NORMALISE BY THE STEP'S RECHARGE, never by either operand. Dividing by a quantity that can approach
# zero has produced a confident wrong reading TWICE in this codebase: tests/dt_invariance's `spread`
# (fixed in ad31b40) and my own analysis of this very sweep, which reported an absolute 0.078 against
# terms of 1e+13 as "7.8e-02" and invented a defect that was not there. d_rech is stable and nonzero.
#
# bdf2 IS EXCLUDED FROM CHECK 2, and this is not a fudge. transient_groundwater.cpp accumulates
#     storage = a_c*V(w^{n+1}) - b_c*V(w^n) + c_c*V(w^{n-1})
# a THREE-LEVEL difference, while the state comparison is the two-level V_after - V_before. They are
# different quantities and SHOULD differ -- measured 8.6e-02 to 1.9e+01 on every bdf2 run. Check 3 still
# applies to bdf2; only the storage cross-check does not. See task #53.
#
# Usage:  tests/budget_step_ledger/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

LAKE=$(readlink -f ../fsm_consistency); MULTI=$(readlink -f ../multilake); CASC=$(readlink -f ../fsm_cascade)
for d in "$LAKE" "$MULTI" "$CASC"; do
    [[ -d "$d/inputs" ]] || ( cd "$d" && python3 make_inputs.py >/dev/null )
done
make_work bledger
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

echo "=== per-step water ledger, swept across fixtures x collectors x integrators x couplings ==="
echo "WTM binary: $WTM"
echo

fail=0
for fx in "lake:$LAKE/inputs:fsm_test" "multilake:$MULTI/inputs:multilake" "cascade:$CASC/inputs:fsm_cascade"; do
IFS=: read -r fn inp reg <<<"$fx"
for coupling in impulse continuous; do
for coll in active_set off implicit explicit; do
for integ in tr-bdf2 backward-euler bdf2; do
    # continuous x explicit is REFUSED by the model (it does not converge, task #44). Skip rather than
    # expect a failure: the refusal has its own coverage, and running it here would only test the guard.
    [[ "$coupling" == "continuous" && "$coll" == "explicit" ]] && continue
    # SEPARATOR IS "__": collector names contain single underscores (active_set), so splitting
    # a single-underscore stem shreds the fields -- it silently defeated both the bdf2
    # exclusion and the xfail lookup when this test was first written.
    stem="${fn}__${coupling}__${coll}__${integ}"
    # THE CONFIG IS A FILE NOW (#83): tests/budget_step_ledger/config.yaml, rendered per sweep cell.
    # Every setting each run resolves to is stated there, and tests/config_identity.py enforces it
    # (this suite is on unconditional since #79 Phase 5). output.trace: [budget] and time_step.mode: fixed are
    # marked LOAD-BEARING in that file: the per-step identity IS the trace, and the sweep must compare
    # cells at a KNOWN step rather than one a controller resized per cell.
    sed -e "s|@INPUTS@|$inp|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" \
        -e "s|@ROUTING@|$coupling|g" -e "s|@COLLECTOR@|$coll|g" -e "s|@INTEG@|$integ|g" \
        -e "s|@REGION@|$reg|g" config.yaml > "$WORK/$stem.yaml"
    if ! "$WTM" "$WORK/$stem.yaml" > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1
    fi
done; done; done; done
[[ $fail -eq 0 ]] || { echo "BUDGET STEP LEDGER: FAILED (a run did not complete)"; exit 1; }

# PROMOTED 2026-09-22: was a Python local inside the heredoc, invisible to #121's sweep.
# SPREAD: 0   measured 2026-09-22 by repeat run after promotion.
# DERIVED, ONE-SIDED: worst |resid|/rech measured 4.59e-07 over 504 step-checks, ~2.2x. The suite's
#   own header records the same quantity at step 1 as 3.1e-09, a margin of 319x -- so the BINDING
#   value is the accumulated worst, not the per-step one, and the bound is sized on it.
RESID_TOL="${RESID_TOL:-1e-6}"
# SPREAD: 0   measured 2026-09-22 by repeat run after promotion.
# DERIVED, ONE-SIDED, in units of MACHINE EPSILON rather than metres -- see the note in the heredoc.
#   STATE == ACC compares two independent summations of the SAME quantity, equal in exact arithmetic,
#   so the only honest bound is accumulated rounding. A previous absolute 1e-12 was ~4500 eps and
#   failed at 10400 eps with nothing in the model changed; stating the budget in eps is what fixed it.
STATE_EPS_MAX="${STATE_EPS_MAX:-1048576}"   # 2**20 eps, written out: a shell default must parse as a float
RESID_TOL="$RESID_TOL" STATE_EPS_MAX="$STATE_EPS_MAX" WORK="$WORK" "$PY" - <<'PYX' || fail=1
import os, glob, sys, re
W = os.environ["WORK"]
TOL_RESID = float(os.environ["RESID_TOL"])   # derivation beside the shell default
# STATE == ACC IS A FLOATING-POINT IDENTITY, SO ITS BUDGET IS STATED IN UNITS OF MACHINE EPSILON (#84).
# It compares two independent summations of the SAME storage change: equal in exact arithmetic, so they
# differ only by summation order and the only honest bound is accumulated rounding.
#
# IT WAS `|acc - state| / recharge < 1e-12`, and the NORMALISER IS RIGHT -- recharge is the water moving
# through the step, so it is what makes a disagreement matter or not. What was wrong is 1e-12: that is
# ~4500 eps, which is what a few hundred differently-ordered sums drift by. It failed at 2.29e-12, about
# 10400 eps, and nothing in the model had changed.
#
# A FIRST ATTEMPT AT THIS NORMALISED BY THE SUMS THEMSELVES and was worse: where both sums are near zero
# their relative disagreement is meaningless, and the same steps reported 4.5e+15 ulps. Keeping recharge
# as the scale and moving only the threshold into eps units is the change that is actually justified.
#
# STATE_EPS_BUDGET is a count of eps, not a tolerance: how far two sums over this domain and this many
# steps may drift. Summation error over n terms grows as at worst n*eps, and these are sums over cells
# and steps, so ~1e4 eps is arithmetic. 2**20 leaves two orders of headroom for a larger domain, sits at
# 2.3e-10 of recharge, and is still FIVE orders below the booking error this arm exists to catch -- #52
# shows at 1e-5 of recharge, which is ~4.5e+10 eps.
STATE_EPS_BUDGET = float(os.environ["STATE_EPS_MAX"])   # derivation beside the shell default
EPS = sys.float_info.epsilon
# The ONE known defect (#52): active_set over-books removal on the step where surface water first
# appears. Held with a FLOOR so it keeps a regression test and this suite FAILS the day it closes --
# which means the defect is fixed and the arm must be promoted to a plain check.
# WAS {("multilake", "active_set"): 1e-5} -- #52's arm, held as an expected failure. IT CLOSES NOW.
# Measured 2026-09-11: worst |resid|/rech 3.135e-09 at multilake__continuous__active_set__backward-euler
# step 1, against the plain TOL_RESID of 1e-6 -- a margin of 319x, not a value scraping under a floor.
# The xfail guard is what reported it: it fails the suite on an UNEXPECTED PASS precisely so a defect
# closing cannot be absorbed in silence. Promoted to a plain check, which is what that guard asks for.
XFAIL_RESID = {}

runs = {}
for f in sorted(glob.glob(f"{W}/*.log")):
    stem = os.path.basename(f)[:-4]
    steps = []
    for l in open(f):
        if not l.startswith("BUDGETTRACE"): continue
        d = dict(kv.split("=") for kv in l.split()[1:])
        steps.append({k: float(v) for k, v in d.items()})
    if steps: runs[stem] = steps

fail = 0
if not runs:
    print("  FAIL  no BUDGETTRACE output -- output.trace: [budget] is not reaching the model"); sys.exit(1)

# 1. CROSSING EXISTS -- the condition this test is built around.
crossings = []
for stem, steps in runs.items():
    for k in range(1, len(steps)):
        if steps[k-1]["d_surf"] == 0.0 and steps[k]["d_surf"] > 0.0:
            crossings.append((stem, int(steps[k]["step"]))); break
ok = len(crossings) > 0
fail |= not ok
print(f"  {'PASS' if ok else 'FAIL'}  CROSSING EXISTS  {len(crossings)} runs contain a step where surface "
      f"water FIRST appears (d_surf 0 -> nonzero); this test exists for that step")

# 2 + 3, per run.
worst_state = worst_resid = 0.0
n_state = n_resid = 0
for stem, steps in sorted(runs.items()):
    fn, coupling, coll, integ = stem.split("__")
    xf = XFAIL_RESID.get((fn, coll))
    for s in steps:
        R = abs(s["d_rech"]) or 1.0
        # 2. STATE == ACC. Skipped for bdf2 -- three-level storage, see the header. Task #53.
        if integ != "bdf2":
            # disagreement as a multiple of eps, measured against the water moving through the step
            v = abs(s["d_stor_acc"] - s["d_stor_state"]) / R / EPS
            worst_state = max(worst_state, v); n_state += 1
            if v >= STATE_EPS_BUDGET:
                print(f"  FAIL  STATE==ACC   {stem} step {int(s['step'])}: two storage sums differ by "
                      f"{v:.3g} eps of recharge (budget {STATE_EPS_BUDGET:.3g})")
                fail = 1
        # 3. LEDGER CLOSES.
        v = abs(s["d_resid"]) / R
        if xf is None:
            worst_resid = max(worst_resid, v); n_resid += 1
            if v >= TOL_RESID:
                print(f"  FAIL  CLOSES      {stem} step {int(s['step'])}: |resid|/rech {v:.3e}")
                fail = 1

print(f"  {'PASS' if not fail else 'FAIL'}  STATE == ACC     two independent storage computations agree over "
      f"{n_state} step-checks; worst disagreement {worst_state:.3g} eps of recharge  (tol STATE_EPS_MAX={STATE_EPS_BUDGET:.3g})")
print(f"  {'PASS' if not fail else 'FAIL'}  LEDGER CLOSES    over {n_resid} step-checks; worst |resid|/rech "
      f"{worst_resid:.2e}  (tol RESID_TOL={TOL_RESID:.0e})")

# The held defect, reported with its measured size so drift is visible rather than absorbed.
for (fn, coll), floor in XFAIL_RESID.items():
    worst = 0.0; where = ""
    for stem, steps in runs.items():
        p = stem.split("__")
        if p[0] != fn or p[2] != coll: continue
        for s in steps:
            v = abs(s["d_resid"]) / (abs(s["d_rech"]) or 1.0)
            if v > worst: worst, where = v, f"{stem} step {int(s['step'])}"
    still = worst > floor
    fail |= not still
    print(f"  {'xfail' if still else 'FAIL '}   KNOWN #52       {fn} x {coll}: worst |resid|/rech {worst:.3e} "
          f"at {where}" + ("" if still else f"  <-- NOW CLOSES (below {floor:.0e}): promote to a check"))

print()
print("BUDGET STEP LEDGER: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PYX
exit $?
