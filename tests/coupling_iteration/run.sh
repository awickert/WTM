#!/usr/bin/env bash
# COUPLING ITERATION (#112): does a step really solve against its OWN surface water, and is the
# state rolled back COMPLETELY between passes?
#
# Under surface_water.routing: continuous, FillSpillMerge's per-cell volume change is fed into the
# NEXT step's recharge source, so a step never sees its own runoff -- the coupling is lagged by one
# step. surface_water.coupling.iterations: k re-solves the step against its own output, k times.
#
# WHY THIS SUITE EXISTS, stated plainly: the feature shipped with every suite declaring
# `iterations: 1`, so it was green only because nothing ran it. That is the vacuous-coverage shape
# this repo has hit six ways (tests/ASSERTION_HEALTH.md), and it is what the RAN assertion below is
# for -- it fails if the passes never happen, which no answer-comparison can detect on its own.
#
# THE TWO REGIMES, on one fixture and both needed:
#   FILLING     the lake rises. The lag is the only place the coupling can be wrong, and the extra
#               passes cost real solves -- which is what RAN measures.
#   AT THE SILL the lake sits at 96 m. Here w_{n+1} = w_n, so FSM(w_n) and FSM(w_{n+1}) are the SAME
#               array and the lag is identically zero BY CONSTRUCTION. Iterating must therefore not
#               move the converged answer at all. INVARIANT pins that.
#
# WHAT INVARIANT IS NOT, MEASURED RATHER THAN ASSUMED: it is NOT a rollback-completeness test, which
# is what it was written to be. Dropping starting_wtd from WTM_COUPLING_ROLLBACK_LIST left it
# passing at 0.000e+00 -- while the run visibly changed, 393 -> 378 solver calls at k=3. EQUILIBRIUM
# IS AN ATTRACTOR: the lake refills to its sill whatever happened on the way, so the same property
# that makes the physics argument work is what makes this assertion blunt. Completeness is pinned
# instead in src/test_coupling_snapshot.cpp, where every member of the set is round-tripped and
# dropping any one name fails three cases by construction. Both are kept; neither substitutes.
#
# Asserts, Anderson + TR-BDF2 + active-set skim + continuous routing:
#   RAN         : k=2 costs measurably more solver calls than k=1 (the passes are real).
#   LAKE        : the fixture actually ponds water (without a lake there is nothing to couple).
#   INVARIANT   : the converged answer at k=2 and k=3 equals the k=1 answer.
#   CONSERVES   : the exact water budget still closes at k>1.
#   REFUSES     : iterations: 0, and k>1 under routing: impulse and routing: off.
#
# Usage:  tests/coupling_iteration/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/coupling_iteration_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work coupit
PY="${PY:-python3}"; export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE (#83): tests/coupling_iteration/config.yaml. The arms differ in exactly ONE
# key, surface_water.coupling.iterations, which is the subject.
emit() { # $1 stem, $2 iterations
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" -e "s|@ITER@|$2|g" config.yaml > "$WORK/$1.yaml"
}
run_arm() { # $1 stem, $2 iterations
  emit "$1" "$2"
  "$WTM" "$WORK/$1.yaml" > "$WORK/$1.err" 2>&1 \
    || { echo "RUN FAILED: $1 (iterations=$2)"; tail -3 "$WORK/$1.err"; exit 2; }
}
echo "=== coupling iteration: does the step re-solve against its own surface water? ==="
run_arm k1 1
run_arm k2 2
run_arm k3 3

# SOLVER CALLS PER ARM. Counted from the solver's own per-solve line, not from a timer: wall time on
# a 24x24 fixture rounds to 0.0 s and could not tell a doubled cost from a dead loop.
n1=$(grep -a -c 'Number of nonlinear iterations' "$WORK/k1.err")
n2=$(grep -a -c 'Number of nonlinear iterations' "$WORK/k2.err")
n3=$(grep -a -c 'Number of nonlinear iterations' "$WORK/k3.err")

# SPREAD: 0   measured 2026-09-23 -- a repeat run of the k=2 arm reproduced both the final water
#             table (max|Δwtd| = 0.000e+00 m) and the solver-call count (252) exactly. Headroom here
#             therefore measures SENSITIVITY, never flake risk; see tests/ASSERTION_HEALTH.md sec 3.
# DERIVED 2026-09-23, SEPARATING, and this is the one bound in the suite that cannot be argued from
#   physics -- it is the non-vacuity guard. A DEAD loop gives a ratio of exactly 1.000, because k=2
#   would run one pass like k=1. Measured n2/n1 = 252/131 = 1.924. The bound sits in that gap: 1.5 is
#   50% above the dead value and 22% below the measured one, so neither edge is close.
RAN_MIN="${RAN_MIN:-1.5}"   # solver calls at k=2 relative to k=1
# SPREAD: 0   measured 2026-09-23; see the note at this file's first bound.
# DERIVED 2026-09-23, SEPARATING: a fixture that stopped ponding gives 0 cells, and this one gives
#   168 (at a max depth of 4.000 m, which is exactly sill 96 - floor 92, so the lake is full). 50 is
#   3.4x below the measured count and well above the 0 that would make INVARIANT vacuous.
LAKE_MIN="${LAKE_MIN:-50}"   # cells holding standing water at k=1
# SPREAD: 0   measured 2026-09-23; see the note at this file's first bound.
# DERIVED 2026-09-23, ONE-SIDED, and THE MULTIPLIER IS A CONVENTION because one edge of the gap is
#   zero and there is no second edge to cite: measured max|Δwtd| = 0.000e+00 m at BOTH k=2 and k=3
#   against k=1 -- bit-identical, 0 of 576 cells moved. The bound is 1e-6 m, which is 2.5e-7 of the
#   4.000 m lake this fixture produces, and 100x the solver's own volume tolerance (1e-8), below
#   which the answer is not defined anyway. Stated as a tolerance rather than `== 0` so the
#   assertion does not rest on float equality.
#   SENSITIVITY, MEASURED: this bound has enormous headroom and it is NOT sharp -- an incomplete
#   rollback (starting_wtd removed from the set) still measured 0.000e+00. Read it as pinning the
#   physical claim, not as a guard on the machinery; see the header note.
INVAR_TOL="${INVAR_TOL:-1e-6}"   # |wtd(k) - wtd(1)| at the converged answer
# SPREAD: 0   measured 2026-09-23; see the note at this file's first bound.
# DERIVED 2026-09-23, ONE-SIDED, multiplier by CONVENTION: measured |exact_budget_residual|/recharge
#   = 2.20e-14 (k=1), 1.53e-14 (k=2), 1.93e-13 (k=3) -- machine level on every arm. 1e-10 is ~500x
#   above the worst measured, which leaves room for the arithmetic to reassociate under a different
#   pass count without leaving room for a leak: losing even 1e-8 of the recharge would fail.
CONS_TOL="${CONS_TOL:-1e-10}"   # |exact budget residual| / recharge, at every arm
K1=$(ls "$WORK"/k1_*.tif | tail -1); K2=$(ls "$WORK"/k2_*.tif | tail -1); K3=$(ls "$WORK"/k3_*.tif | tail -1)

RAN_MIN="$RAN_MIN" LAKE_MIN="$LAKE_MIN" INVAR_TOL="$INVAR_TOL" CONS_TOL="$CONS_TOL" \
N1="$n1" N2="$n2" N3="$n3" "$PY" - "$K1" "$K2" "$K3" "$WORK/k1.txt" "$WORK/k2.txt" "$WORK/k3.txt" <<'PY'
import sys, os, numpy as np, rasterio
ran_min = float(os.environ["RAN_MIN"]); lake_min = float(os.environ["LAKE_MIN"])
invar_tol = float(os.environ["INVAR_TOL"]); cons_tol = float(os.environ["CONS_TOL"])
n1, n2, n3 = (int(os.environ[k]) for k in ("N1", "N2", "N3"))
w1, w2, w3 = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]

def resid(txt):
    # Columns are read POSITIONALLY and the two used here are named in src/irf.cpp's writer:
    # 9 = global_added_recharge, 17 = exact_budget_residual. 0-indexed 8 and 16.
    rows = [l.split() for l in open(txt) if l and l[0].isdigit()]
    R, X = float(rows[-1][8]), float(rows[-1][16])
    return abs(X) / abs(R) if R else float("nan")

ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond

# UNITS: |Δwtd| is a WATER DEPTH in metres and is compared against a depth bound; the lake count is
# dimensionless; the budget ratio is dimensionless. Nothing here needs converting to volume (#65).
d2, d3 = float(np.abs(w2 - w1).max()), float(np.abs(w3 - w1).max())
lake = int((w1 > 1e-6).sum())
ratio = n2 / n1 if n1 else 0.0
r1, r2, r3 = resid(sys.argv[4]), resid(sys.argv[5]), resid(sys.argv[6])

check("RAN (k=2 costs more solves than k=1)", ratio > ran_min,
      f"solver calls {n1} -> {n2} -> {n3}, ratio n2/n1 = {ratio:.3f} (tol RAN_MIN={ran_min})"
      f" -- a DEAD iteration would give exactly 1.000")
check("LAKE (the fixture ponds water)", lake >= lake_min,
      f"{lake} cells with standing water, max depth {w1.max():.3f} m (tol LAKE_MIN={lake_min})")
check("INVARIANT (iterating does not move the converged answer)", max(d2, d3) < invar_tol,
      f"max|Δwtd| = {max(d2, d3):.3e} m (k=2: {d2:.3e}, k=3: {d3:.3e}) (tol INVAR_TOL={invar_tol})")
check("CONSERVES (exact budget closes at every k)", max(r1, r2, r3) < cons_tol,
      f"max |residual|/recharge = {max(r1, r2, r3):.3e}"
      f" (k=1: {r1:.2e}, k=2: {r2:.2e}, k=3: {r3:.2e}) (tol CONS_TOL={cons_tol})")
print("PASS: the passes run, and the converged answer and the budget survive them" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
asserts=$?

# REFUSALS. A request to iterate that quietly does not iterate is the #27/#35 defect class, and this
# key is expensive enough that finding out afterwards would waste a production run. Each case must
# exit NON-ZERO and name the key -- exiting nonzero for some other reason would pass a weaker test,
# so the message is matched too.
echo "  -- refusals --"
refuse() { # $1 label, $2 sed expression applied to the k2 config, $3 required substring
  sed -e "$2" "$WORK/k2.yaml" > "$WORK/refuse.yaml"
  out=$("$WTM" "$WORK/refuse.yaml" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
      echo "  FAIL REFUSES ($1): the run SUCCEEDED; the request was silently accepted"; return 1
  elif ! printf '%s' "$out" | grep -q "$3"; then
      echo "  FAIL REFUSES ($1): exited $rc but did not say why (no '$3' in the output)"; return 1
  fi
  echo "  OK   REFUSES ($1): exits $rc naming surface_water.coupling.iterations"; return 0
}
rf=0
refuse "iterations: 0"      's|iterations: 2|iterations: 0|'          "coupling.iterations must be >= 1" || rf=1
refuse "k>1 with impulse"   's|routing: continuous|routing: impulse|' "coupling.iterations > 1 needs"    || rf=1
refuse "k>1 with off"       's|routing: continuous|routing: off|'     "coupling.iterations > 1 needs"    || rf=1

[ "$asserts" -eq 0 ] && [ "$rf" -eq 0 ] && { echo "COUPLING ITERATION TESTS PASSED"; exit 0; }
echo "COUPLING ITERATION TESTS FAILED"; exit 1
