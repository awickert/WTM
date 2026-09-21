#!/usr/bin/env bash
# CROSS-RANK DRIFT: does it stay put, or compound?
#
# Every other MPI check in this suite compares the FINAL state at n=1 against n=N and asserts a
# threshold. That answers "how far apart did they end up" and cannot distinguish two very different
# situations: a fixed offset injected once, and an error that compounds every step. Over a short
# fixture both look like a small number. Over a production run they do not.
#
# WHAT THIS PINS. surface_water.fsm_coupling changes which of the two you get, and the reason is a
# single line (WTM.cpp, the `impulse` branch of couple_surface_and_recharge):
#
#     scatter_into_owned(user_context, arp.wtd.data(), dmdapack.starting_wtd);
#
# It overwrites EVERY rank's starting_wtd from RANK 0's array, for EVERY cell -- not only the ones
# FSM touched. So `impulse` resets cross-rank drift once per step. `continuous` never runs that line,
# so its drift compounds. Measured on this fixture, max|dwtd| between n=1 and n=6 per report:
#
#     continuous   8.212e-10 -> 1.284e-09 -> 7.137e-09      grows
#     impulse      8.981e-12 -> 8.995e-12 -> 1.576e-11      flat
#
# It does NOT track snes_stol -- it floors at 2.243e-08 for both 1e-10 and 1e-12 -- which is the
# signature of accumulation punctuated by a reset, not of round-off that a tighter solve resolves away.
# The excess lands in a few DEEP cells (wtd -14.5 m, -24.7 m) where low storativity turns a small
# volume difference into a large head difference; the MEDIAN land cell is ~2e-12 under both.
#
# WHY IT IS WORTH A TEST OF ITS OWN. The corollary is the uncomfortable part: `impulse`'s excellent
# cross-rank agreement is partly an ARTEFACT of that broadcast rather than evidence that the parallel
# solve agrees. Every cross-rank tolerance in this suite was calibrated under `impulse`, i.e. on a
# resynchronised system. This test measures the drift's BEHAVIOUR instead of its size, so it keeps
# working when the size changes and it says which regime we are in. See task #39.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f ../golden/inputs_runoff/runoff_test_t0_topography.tif ]] || ( cd ../golden && python3 make_runoff_inputs.py >/dev/null )
INP=$(readlink -f ../golden/inputs_runoff)
make_work xrg
NRANK="${NRANK:-6}"; PY="${PY:-python3}"
# PROMOTED FROM LITERALS (#121). The shell form is what makes them reachable: assertion_probe
# reads run.sh for `NAME="${NAME:-VALUE}"`, so an os.environ.get default inside the python block
# would be invisible to it -- the bound would exist and still be unprobeable.
# The last two are BITE GUARDS: without them "impulse drift is flat" would pass just as well if
# BOTH regimes were flat, i.e. if the test had stopped distinguishing anything.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, SEPARATING: impulse drift measures last/first = 2.05 while continuous drift,
#   the arm right below, measures 103.23. The ceiling at 10 sits inside that measured gap -- 4.9x
#   above flat, 10x below compounding -- so the pair of bounds cannot both pass if the two regimes
#   ever collapse into one.
FLAT_MAX="${FLAT_MAX:-10.0}"          # impulse drift must stay flat: last/first below this
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, SEPARATING, the same measured pair from the other side: continuous drift
#   compounds to 103.23 where impulse stays at 2.05. The floor at 3.0 sits just above the flat
#   value, 34x below the compounding one.
COMPOUND_MIN="${COMPOUND_MIN:-3.0}"   # BITE GUARD: continuous drift must actually compound
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED bite guard, BLUNT at 2747x: measured continuous/impulse = 27474.6
#   at the final report. Like the other collapse guards in this tree it asks a yes/no question --
#   have the two regimes become the same run -- and the degenerate value is 1.0, not something
#   near the bound.
DISTINCT_MIN="${DISTINCT_MIN:-10.0}"  # BITE GUARD: the two regimes must be far apart at the end
export OMP_NUM_THREADS=1

for coup in continuous impulse; do for n in 1 "$NRANK"; do
  t="${coup}_n$n"
  # THE CONFIG IS A FILE NOW (#83): tests/xrank_growth/config.yaml. Every setting the run resolves to
  # is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
  # surface_water.routing is THE SUBJECT and the only thing that varies between arms; the rank count
  # is not a config setting.
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$t|g" \
      -e "s|^  routing: continuous|  routing: $coup|" config.yaml > "$WORK/$t.yaml"
  mpirun -n "$n" "$WTM" "$WORK/$t.yaml" > "$WORK/$t.log" 2>&1 \
    || { echo "RUN FAILED: $t"; tail -5 "$WORK/$t.log"; exit 2; }
done; done

NRANK="$NRANK" FLAT_MAX="$FLAT_MAX" COMPOUND_MIN="$COMPOUND_MIN" DISTINCT_MIN="$DISTINCT_MIN" \
  "$PY" - "$WORK" <<'PY'
import sys, os, glob, numpy as np, rasterio
work = sys.argv[1]; N = os.environ["NRANK"]
def series(t):
    return [rasterio.open(f).read(1).astype(np.float64) for f in sorted(glob.glob(f"{work}/{t}_0*.tif"))]
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond

# UNITS: NOT CONVERTED TO WATER VOLUME (#65). This suite asserts a RATIO -- last/first drift, and
# continuous/impulse at the final report -- and a ratio of two quantities in the same unit is
# unit-agnostic: multiplying both by porosity cancels. The subject is the drift's BEHAVIOUR (flat vs
# compounding), not its size, which is the whole reason this test exists. The printed values stay in
# metres of head so they can be read against the numbers recorded in #39.
drift = {}
for c in ("continuous", "impulse"):
    a, b = series(f"{c}_n1"), series(f"{c}_n{N}")
    d = [float(np.abs(y - x).max()) for x, y in zip(a, b)]
    drift[c] = [v for v in d if v > 0.0]        # report 0 is the identical initial condition
    print(f"  {c:>11}: " + "  ".join(f"{v:.3e}" for v in drift[c]))

check("PRECONDITION both couplings ran and drifted", all(len(v) >= 2 for v in drift.values()),
      f"continuous {len(drift['continuous'])} reports, impulse {len(drift['impulse'])}")

# THRESHOLDS ARE CHOICES, so they are named. Measured: impulse grows 1.8x over the run, continuous
# 8.7x, and continuous ends 453x above impulse. The bounds below sit well clear of all three, so they
# assert the REGIME (flat vs compounding) rather than pinning any measured value.
flat_max = float(os.environ["FLAT_MAX"]); compound_min = float(os.environ["COMPOUND_MIN"])
distinct_min = float(os.environ["DISTINCT_MIN"])
gi = drift["impulse"][-1] / drift["impulse"][0]
gc = drift["continuous"][-1] / drift["continuous"][0]
check("impulse drift is FLAT (reset each step from rank 0)", gi < flat_max,
      f"last/first = {gi:.2f} (tol FLAT_MAX={flat_max})")
check("continuous drift COMPOUNDS (no reset)", gc > compound_min,
      f"last/first = {gc:.2f} (min COMPOUND_MIN={compound_min})")
check("the two regimes are distinguishable", drift["continuous"][-1] / drift["impulse"][-1] > distinct_min,
      f"continuous/impulse at the final report = {drift['continuous'][-1] / drift['impulse'][-1]:.1f} (min DISTINCT_MIN={distinct_min})")

print("PASS: cross-rank drift is flat under impulse and compounding under continuous, as the "
      "rank-0 rescatter predicts" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
