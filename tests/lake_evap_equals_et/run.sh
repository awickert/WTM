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
WTM="${WTM:-../../build/wtm.x}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/eq_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP="$(readlink -f inputs)"
WORK=$(mktemp -d /tmp/lakeevap_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# Two transition settings, deliberately far apart: the shipped default, and one whose
# half-rate depth and width are 20x larger.
emit () { # $1 = region, $2 = tag, $3 = wtd_center, $4 = logistic_width
  ../emit_config.sh > "$WORK/$1_$2.yaml" <<EOF
solver_method anderson
run_type equilibrium
total_time 40yr
supplied_wt 1
deltat 31536000
report_interval 2
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_collector active_set
et_sigmoid_wtd_center $3
et_sigmoid_width $4
surfdatadir $INP
region $1
time_start t0
time_end t0
eq_tol 0
textfilename $WORK/$1_$2.txt
outfile_prefix $WORK/$1_$2_
EOF
  "$WTM" "$WORK/$1_$2.yaml" > "$WORK/$1_$2.log" 2>&1 \
    || { echo "RUN FAILED: $1 $2"; tail -5 "$WORK/$1_$2.log"; exit 2; }
}

for region in eq neq; do
  emit "$region" a 0.05 0.1     # shipped default
  emit "$region" b 1.0  2.0     # 20x deeper centre, 20x wider transition
done

"$PY" - "$WORK" <<'PY'
import sys, glob, numpy as np, rasterio
work = sys.argv[1]
def wtd(region, tag):
    f = sorted(glob.glob(f"{work}/{region}_{tag}_*.tif"))[-1]
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
check("CONTROL: the parameters DO matter when owe != ET", d_neq > 1e-3,
      f"same comparison on the unequal region = {d_neq:.6e} m (> 1e-3)")

print("PASS: the ET/open-water transition is inert exactly when there is nothing to transition"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
