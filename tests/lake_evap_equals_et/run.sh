#!/usr/bin/env bash
# LAKE EVAPORATION EQUALS ET: the ET->open-water transition must vanish when there is
# nothing to transition between.
#
# WTM has no switch between "soil ET" and "lake evaporation". It BLENDS them as a logistic
# in water-table depth (evaporation.et_sigmoid), so a cell evaporates at the ET rate when
# deep, at the open-water rate when ponded, and in between across the transition:
#
#     E_eff(wtd) = ET + (owe - ET) * sigma((wtd - wtd_center) / logistic_width)
#
# Set owe == ET and the (owe - ET) factor is identically zero. E_eff collapses to ET at
# every depth, and the transition parameters cannot affect the answer AT ALL -- not the
# residual, and not the Jacobian, whose tangent (owe-ET)*sigma*(1-sigma)/s carries the same
# factor (src/transient_groundwater.cpp, evapTaper / evapTaperTangentRaw). The invariant is
# EXACT, so this asserts bit-identity rather than a tolerance.
#
# Note this holds because the evaporation taper is ON, which is the default: recharge is
# then just precipitation and evaporation enters ONLY through E_eff. With the taper off,
# the recharge branch picks (precip - owe) above the surface against max(0, precip - ET)
# below it, and the max() clamp means owe == ET does NOT make those two branches agree.
#
# TWO ARMS, and the second is what keeps the first honest:
#   eq   owe == ET   -> changing wtd_center/logistic_width must change NOTHING
#   neq  owe != ET   -> the same change must move the answer, else the eq arm is vacuous
# The regions are identical in every other field, written from the same arrays by
# make_inputs.py, so this is a same-method comparison.
set -u
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${WTM:-../../build/wtm.x}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/eq_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP="$(readlink -f inputs)"
make_work lakeevap
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# Two transition settings, deliberately far apart: the shipped default, and one whose
# half-rate depth and width are 20x larger.
# THE CONFIG IS A FILE NOW (#83): tests/lake_evap_equals_et/config.yaml. Every setting the run
# resolves to is stated there, and tests/config_identity.py enforces it (this suite is on
# unconditional since #79 Phase 5). evaporation.et_sigmoid is THE SUBJECT and is tokenised per arm, so the file
# says outright that the sigmoid is being varied rather than inherited.
emit () { # $1 region, $2 tag, $3 et_sigmoid.wtd_center, $4 et_sigmoid.logistic_width  (ALL REQUIRED)
  local rg="${1:?emit needs a region: eq (ET == owe) or neq}"
  local tg="${2:?emit needs a tag}"
  local wc="${3:?emit needs a wtd_center -- it is the subject, never inherit it}"
  local lw="${4:?emit needs a logistic_width -- it is the subject, never inherit it}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|${rg}_${tg}|g" \
      -e "s|@REGION@|$rg|g" -e "s|@WTDC@|$wc|g" -e "s|@WIDTH@|$lw|g" \
      config.yaml > "$WORK/${rg}_${tg}.yaml"
  "$WTM" "$WORK/${rg}_${tg}.yaml" > "$WORK/${rg}_${tg}.log" 2>&1 \
    || { echo "RUN FAILED: $rg $tg"; tail -5 "$WORK/${rg}_${tg}.log"; exit 2; }
}

for region in eq neq; do
  emit "$region" a 0.05 0.1     # shipped default
  emit "$region" b 1.0  2.0     # 20x deeper centre, 20x wider transition
done

# BITE GUARD, promoted from a literal (#121) so assertion_probe can RAISE it and confirm the
# check still fails when it should. A dead guard here means the suite passes on nothing.
# Chosen to sit clear of noise, NOT tuned: raise it only with a measurement.
CONTROL_MIN="${CONTROL_MIN:-1e-3}"   # the sigmoid parameters MUST matter where owe != ET

CONTROL_MIN="$CONTROL_MIN" TESTS="$(readlink -f ..)" "$PY" - "$WORK" <<'PY'
import sys, os, glob, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # latest_output: refuses a match from a DIFFERENT stem
work = sys.argv[1]
def wtd(region, tag):
    f = VOL.latest_output(f"{work}/{region}_{tag}_")
    return rasterio.open(f).read(1).astype(np.float64)[1:-1, 1:-1]

ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond

d_eq  = float(np.abs(wtd("eq",  "a") - wtd("eq",  "b")).max())
d_neq = float(np.abs(wtd("neq", "a") - wtd("neq", "b")).max())

# EXACT: with owe == ET the (owe - ET) factor is zero, so the sigmoid parameters enter
# nowhere. Any nonzero difference means evaporation is reading the transition somewhere it
# should not, so this is asserted at zero rather than at a tolerance.
check("TRANSITION IS INERT when lake evap == ET", d_eq == 0.0,
      f"max |wtd(default sigmoid) - wtd(20x sigmoid)| = {d_eq:.6e} m (must be exactly 0)")

# CONTROL. Without this the check above would pass just as well if the sigmoid parameters
# were ignored outright, or if both runs had silently failed into the same state.
control_min = float(os.environ["CONTROL_MIN"])
check("CONTROL: the parameters DO matter when owe != ET", d_neq > control_min,
      f"same comparison on the unequal region = {d_neq:.6e} m (min {control_min})")

print("PASS: the ET/open-water transition is inert exactly when there is nothing to transition"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
