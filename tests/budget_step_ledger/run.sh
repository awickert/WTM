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
    ../emit_config.sh > "$WORK/$stem.yaml" <<EOF
snes_stol 1e-10
solver_method anderson
adaptive_dt false
time_integration $integ
run_type equilibrium
total_time 8yr
supplied_wt 1
deltat 31536000
report_interval 2
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
fsm_coupling $coupling
runoff_collector $coll
trace budget
surfdatadir $inp
region $reg
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$stem.txt
outfile_prefix $WORK/${stem}_
EOF
    if ! "$WTM" "$WORK/$stem.yaml" > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1
    fi
done; done; done; done
[[ $fail -eq 0 ]] || { echo "BUDGET STEP LEDGER: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" "$PY" - <<'PYX' || fail=1
import os, glob, sys, re
W = os.environ["WORK"]
TOL_STATE, TOL_RESID = 1e-12, 1e-6
# The ONE known defect (#52): active_set over-books removal on the step where surface water first
# appears. Held with a FLOOR so it keeps a regression test and this suite FAILS the day it closes --
# which means the defect is fixed and the arm must be promoted to a plain check.
XFAIL_RESID = {("multilake", "active_set"): 1e-5}

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
            v = abs(s["d_stor_acc"] - s["d_stor_state"]) / R
            worst_state = max(worst_state, v); n_state += 1
            if v >= TOL_STATE:
                print(f"  FAIL  STATE==ACC   {stem} step {int(s['step'])}: |acc-state|/rech {v:.3e}")
                fail = 1
        # 3. LEDGER CLOSES.
        v = abs(s["d_resid"]) / R
        if xf is None:
            worst_resid = max(worst_resid, v); n_resid += 1
            if v >= TOL_RESID:
                print(f"  FAIL  CLOSES      {stem} step {int(s['step'])}: |resid|/rech {v:.3e}")
                fail = 1

print(f"  {'PASS' if not fail else 'FAIL'}  STATE == ACC     two independent storage computations agree over "
      f"{n_state} step-checks; worst |acc-state|/rech {worst_state:.2e}  (tol {TOL_STATE:.0e})")
print(f"  {'PASS' if not fail else 'FAIL'}  LEDGER CLOSES    over {n_resid} step-checks; worst |resid|/rech "
      f"{worst_resid:.2e}  (tol {TOL_RESID:.0e})")

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
