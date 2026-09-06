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
export OMP_NUM_THREADS=1

for coup in continuous impulse; do for n in 1 "$NRANK"; do
  t="${coup}_n$n"
  cat <<EOF | ../emit_config.sh > "$WORK/$t.yaml"
solver_method anderson
run_type equilibrium
fsm_on 1
evap_mode 1
infiltration_on 0
runoff_ratio_on 1
cells_per_degree 10
southern_edge -45
deltat 31536000
total_time 6yr
report_interval 2
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start t0
time_end t0
surfdatadir $INP
region runoff_test
supplied_wt 1
save_nreport_interval 1
fsm_coupling $coup
textfilename $WORK/$t.txt
outfile_prefix $WORK/${t}_
EOF
  mpirun -n "$n" "$WTM" "$WORK/$t.yaml" > "$WORK/$t.log" 2>&1 \
    || { echo "RUN FAILED: $t"; tail -5 "$WORK/$t.log"; exit 2; }
done; done

NRANK="$NRANK" "$PY" - "$WORK" <<'PY'
import sys, os, glob, numpy as np, rasterio
work = sys.argv[1]; N = os.environ["NRANK"]
def series(t):
    return [rasterio.open(f).read(1).astype(np.float64) for f in sorted(glob.glob(f"{work}/{t}_0*.tif"))]
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond

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
gi = drift["impulse"][-1] / drift["impulse"][0]
gc = drift["continuous"][-1] / drift["continuous"][0]
check("impulse drift is FLAT (reset each step from rank 0)", gi < 10.0,
      f"last/first = {gi:.2f} (< 10)")
check("continuous drift COMPOUNDS (no reset)", gc > 3.0,
      f"last/first = {gc:.2f} (> 3)")
check("the two regimes are distinguishable", drift["continuous"][-1] / drift["impulse"][-1] > 10.0,
      f"continuous/impulse at the final report = {drift['continuous'][-1] / drift['impulse'][-1]:.0f}x (> 10)")

print("PASS: cross-rank drift is flat under impulse and compounding under continuous, as the "
      "rank-0 rescatter predicts" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
