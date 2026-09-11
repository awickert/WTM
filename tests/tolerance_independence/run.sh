#!/usr/bin/env bash
# A CONVERGED ANSWER CANNOT DEPEND ON THE TOLERANCE YOU STOPPED AT.
#
# Each dt is run twice, differing in exactly one key -- solver.convergence.water_volume_tol -- and the
# two water tables must agree. If they do not, the looser run had not converged, whatever reason code it
# printed. That is the entire assertion, and it needs no stored reference: there is nothing here to
# regold, so the check cannot be quietly regenerated into agreement. (A golden CAN be: tests/golden had
# its reference captured from a run that never finished, and passed for its whole life as a result.)
#
# WHAT IT CATCHES, and why it exists (#104). On the shipped `active_set` path, the per-solve criterion
# is a RELATIVE WATER STEP -- it asks "has the iterate stopped moving?", not "is the equation
# satisfied?". Those coincide when the solve has arrived, and ALSO when it has stalled. On the first
# step of a saturated cold start it stalls: measured at dt = 2.5 wk the residual sits flat at 1.31e+02
# for iterations 5-7 -- HIGHER than the 6.71e+01 it had already reached at iteration 1 -- and the run
# exits CONVERGED_SNORM_RELATIVE at 7 iterations. Let it continue and it escapes the plateau at
# iteration 12 and converges properly at 35 on the RESIDUAL test, at 6.21e-07. The answer moves 18 m.
#
# THE ARMS SPAN dt ON PURPOSE. The defect appears in BANDS, not as a trend -- clean at 4 wk, bad at 2.5,
# clean at 2.25, bad at 1.0, clean at 0.25 -- so a single-dt test would sit in a clean window and never
# bite. The clean arms are not filler: they are what proves the test is not simply failing everywhere.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/tolind_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
PY="${PY:-python3}"
make_work tolind
export OMP_NUM_THREADS=1

# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head (#61/#65). Porosity varies
# 0.05-0.40 on this fixture, so this is a real water measure and not a uniform rescale of head.
#
# WHERE 1e-4 COMES FROM. It is not a precision claim; it is a floor sitting in an empty gap. MEASURED,
# shipped-vs-tight water disagreement per arm:
#     clean   4.0000 wk 8.5575e-08   2.2500 wk 2.3214e-08   0.2500 wk 0.0000e+00
#     band    2.5000 wk 6.6836e+00   1.1875 wk 5.0426e+00   1.0000 wk 1.5850e+00   0.5000 wk 1.4518e+00
# Seven orders of magnitude of empty space between the two groups: 1e-4 clears the worst clean arm by
# ~1200x and sits ~14000x under the mildest failing one. There is no grey zone to calibrate into, which
# is the only reason a round number is defensible here.
TOL="${TOL:-1e-4}"

# The tight tolerance the answer is judged against. 1e-12 is where the answer STOPS MOVING: measured,
# 1e-12 and 1e-14 give identical iteration counts (35/27/27) and identical fields, and both exit on
# CONVERGED_FNORM_RELATIVE -- the RESIDUAL test -- rather than on the step test or on max_iterations.
# That is what makes this side of the comparison an oracle rather than just a second opinion.
TIGHT="${TIGHT:-1e-12}"
SHIPPED="${SHIPPED:-1e-08}"

emit() { # $1 stem, $2 dt (s), $3 water_volume_tol   (ALL REQUIRED)
  local dt="${2:?emit needs a dt: the defect is banded in dt, so the arm must name its step}"
  local wv="${3:?emit needs a water_volume_tol -- it is the subject; inheriting it deletes the test}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|@DT@|$dt|g" -e "s|@WVTOL@|$wv|g" config.yaml > "$WORK/$1.yaml"
  grep -q "@[A-Z_]*@" "$WORK/$1.yaml" && { echo "ERROR: unfilled slot in $1.yaml"; exit 1; }
  return 0
}

# dt in WEEKS -> seconds. band = where #104 bit when this was characterised; clean = where it did not.
#   4.0000 clean   2.5000 BAND (worst of band A)   2.2500 clean
#   1.1875 BAND (worst of band B)   1.0000 BAND   0.5000 BAND   0.2500 clean
WK=604800
declare -a ARMS=( "w4000 2419200 clean" "w2500 1512000 band" "w2250 1360800 clean" \
                  "w1188 718200 band"  "w1000 604800 band"  "w0500 302400 band" "w0250 151200 clean" )

for a in "${ARMS[@]}"; do
  read -r stem dt kind <<<"$a"
  for tag in ship tight; do
    [ "$tag" = ship ] && wv="$SHIPPED" || wv="$TIGHT"
    emit "${stem}_${tag}" "$dt" "$wv"
    "$WTM" "$WORK/${stem}_${tag}.yaml" > "$WORK/${stem}_${tag}.log" 2>&1 \
      || { echo "RUN FAILED: $stem $tag"; tail -3 "$WORK/${stem}_${tag}.log"; exit 2; }
  done
done

# THE TIGHT SIDE MUST HAVE CONVERGED ON THE RESIDUAL, not merely finished. Without this the oracle could
# itself be a stalled solve and the comparison would be two wrongs agreeing -- the exact shape of the
# vacuity this suite exists to refuse.
for a in "${ARMS[@]}"; do
  read -r stem dt kind <<<"$a"
  if ! grep -q "CONVERGED_FNORM" "$WORK/${stem}_tight.log"; then
    echo "  FAIL  ORACLE  ${stem} tight arm did not converge on the RESIDUAL test:" >&2
    grep -oE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" \
         "$WORK/${stem}_tight.log" | tail -1 >&2
    echo "      A reference that stopped on the STEP test cannot judge a run that stopped on the step" >&2
    echo "      test. Raise max_iterations or tighten TIGHT until this converges on FNORM." >&2
    exit 3
  fi
done

TOL="$TOL" SHIPPED="$SHIPPED" TIGHT="$TIGHT" PHI="$INP/tolind_porosity.tif" \
  TESTS="$(readlink -f ..)" WORK="$WORK" ARMS="${ARMS[*]}" "$PY" - <<'PY'
import sys, os, glob, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

work, tol = os.environ["WORK"], float(os.environ["TOL"])
phi  = VOL.read_band(os.environ["PHI"])
arms = os.environ["ARMS"].split()
arms = [tuple(arms[i:i+3]) for i in range(0, len(arms), 3)]

def final(stem):
    fs = sorted(glob.glob(os.path.join(work, f"{stem}_0*.tif")))
    if len(fs) < 2:
        raise SystemExit(f"  FAIL  {stem}: fewer than two snapshots -- the step did not report")
    return rasterio.open(fs[-1]).read(1).astype(float)

print(f"  a converged answer must not depend on the tolerance: {os.environ['SHIPPED']} vs {os.environ['TIGHT']}")
print(f"  {'dt (wk)':>8s}  {'expect':>6s}   max|dV| (water)   verdict")
bad_clean, bad_band, ok_band = [], [], []
for stem, dt, kind in arms:
    a, b = final(f"{stem}_ship"), final(f"{stem}_tight")
    m = np.ones_like(a, bool); m[:, 0] = False          # land only; the ocean column is pinned
    d = float(VOL.volume_diff(a, b, phi)[m].max())
    agree = d <= tol
    if kind == "clean" and not agree: bad_clean.append((dt, d))
    if kind == "band"  and not agree: bad_band.append((dt, d))
    if kind == "band"  and agree:     ok_band.append((dt, d))
    print(f"  {int(dt)/604800:8.4f}  {kind:>6s}   {d:.4e}        {'agree' if agree else 'DISAGREE'}")

# A CLEAN ARM THAT DISAGREES IS A REAL FAILURE -- the defect has spread, or something else broke.
if bad_clean:
    print("\nFAIL: arms that have always agreed no longer do -- this is NOT the known #104 banding:")
    for dt, d in bad_clean: print(f"         dt = {int(dt)/604800:.4f} wk   max|dV| = {d:.4e} m water")
    sys.exit(1)

# THE BANDED ARMS ARE A KNOWN, OPEN DEFECT (#104). Held as an xfail WITH A GUARD, so that a fix is
# detected rather than silently absorbed, and so the suite cannot go green while the defect is live.
if ok_band:
    print("\nUNEXPECTED PASS: banded arms now agree across tolerances -- #104 may be FIXED:")
    for dt, d in ok_band: print(f"         dt = {int(dt)/604800:.4f} wk   max|dV| = {d:.4e} m water")
    print("  Do not just delete the xfail. Confirm the shipped run now exits on CONVERGED_FNORM (the")
    print("  residual test) rather than after 4-7 iterations on the step test, then convert these arms")
    print("  to a live assertion. Failing so this cannot pass unnoticed.")
    sys.exit(1)
print(f"\n  xfail   KNOWN #104   {len(bad_band)} of {len(bad_band)} banded arms disagree across tolerances")
print("  The shipped tolerance stops the semismooth active_set solve on a STALL and calls it converged;")
print("  the tight arm is the same solve allowed to reach its residual. See #104 and the note at")
print("  src/transient_groundwater.cpp (VolumeStepConverged).")
print(f"  PASS: the {len(arms) - len(bad_band)} clean arms agree (see the column above), so the comparison is LIVE and not simply failing everywhere.")
PY
