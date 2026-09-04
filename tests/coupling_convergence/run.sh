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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/cconv_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

mkcfg() { # $1 = stem, $2 = coupling, $3 = deltat seconds, $4 = report_interval
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
solver_method anderson
run_type equilibrium
time_integration tr-bdf2
total_time 8yr
supplied_wt 1
deltat $3
report_interval $4
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
fsm_coupling $2
runoff_ratio 0
adaptive_dt false
surfdatadir $INP
region fsm_test
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
fail=0
for spec in "1yr:31536000:2" "05yr:15768000:4" "025yr:7884000:8"; do
    IFS=: read -r tag dt ri <<<"$spec"
    for cp in impulse continuous; do
        stem="${tag}_${cp}"
        mkcfg "$stem" "$cp" "$dt" "$ri"; rm -f "$WORK/$stem.txt"
        if ! "$WTM" "$WORK/$stem.yaml" -snes_stol 1e-10 > "$WORK/$stem.log" 2>&1; then
            echo "  FAIL  RUN FAILED: $stem"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1
        fi
    done
done
[[ $fail -eq 0 ]] || { echo "COUPLING CONVERGENCE: FAILED (a run did not complete)"; exit 1; }

WORK="$WORK" "$PY" - <<'PY' || fail=1
import os, sys
W = os.environ["WORK"]
TAGS = ["1yr", "05yr", "025yr"]           # each half the previous
COLS = [(13, "14 stored_volume", True), (17, "18 total_evap_removed", True),
        (11, "12 total_surface_removed", False), (12, "13 total_ocean_outflow", False)]
TOL_CONSERVE, MIN_GAP, MIN_RATE = 1e-6, 1e-2, 1.5

def last(stem):
    rows = [[float(x) for x in l.split()] for l in open(f"{W}/{stem}.txt")
            if l.split() and l.split()[0].isdigit() and len(l.split()) >= 23]
    return rows[-1] if rows else None

fail = 0
runs = {(t, c): last(f"{t}_{c}") for t in TAGS for c in ("impulse", "continuous")}
if any(v is None for v in runs.values()):
    print("  FAIL  a run produced no data rows"); sys.exit(1)

# 1. CONSERVATION -- both couplings, every step size. The claim the model owes.
for (t, c), r in runs.items():
    v = abs(r[16]) / (abs(r[8]) or 1.0)
    ok = v < TOL_CONSERVE
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  CONSERVATION  {t:<6} {c:<11} |exact residual|/recharge "
          f"{v:.3e}  (tol {TOL_CONSERVE:.0e})")
print()

gaps = {}
for idx, name, strict in COLS:
    gaps[name] = []
    for t in TAGS:
        i, c = runs[(t, "impulse")], runs[(t, "continuous")]
        gaps[name].append(abs(i[idx] - c[idx]) / (abs(i[8]) or 1.0))

# 2. NON-VACUOUS -- if the couplings agree at the coarsest step there is nothing to converge.
for idx, name, strict in COLS:
    if not strict:
        continue
    g0 = gaps[name][0]
    ok = g0 > MIN_GAP
    fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  NON-VACUOUS   {name:<24} gap at the coarsest dt "
          f"{g0:.3e}  (need > {MIN_GAP:.0e}; if this ever fails the couplings stopped differing "
          f"and the test proves nothing)")
print()

# 3. CONVERGES -- monotone, every column. 4. FIRST ORDER -- rate, on the two largest gaps only.
for idx, name, strict in COLS:
    g = gaps[name]
    mono = g[0] > g[1] > g[2]
    fail |= not mono
    r1, r2 = g[0] / g[1], g[1] / g[2]
    print(f"  {'PASS' if mono else 'FAIL'}  CONVERGES     {name:<24} "
          f"{g[0]:.3e} -> {g[1]:.3e} -> {g[2]:.3e}  (x{r1:.2f}, x{r2:.2f})")
    if strict:
        ok = r1 >= MIN_RATE and r2 >= MIN_RATE
        fail |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  FIRST ORDER   {name:<24} shrinks >= {MIN_RATE}x per "
              f"halving (x{r1:.2f}, x{r2:.2f})")

print()
print("COUPLING CONVERGENCE: " + ("ALL PASSED" if not fail else "FAILED"))
sys.exit(1 if fail else 0)
PY
exit $?
