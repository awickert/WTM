#!/usr/bin/env bash
# WHERE does the FillSpillMerge time-stepping error live? It is NOT spread over the domain, and it is
# NOT "coastal cells are hard": on tests/golden's transient fixture the MEDIAN error across land is
# exactly 0.0 while the max is ~1 m, and every cell above 10 cm lies between the depression and the
# NEAREST ocean. The rest of the perimeter -- ~50 land cells with an ocean neighbour -- is exact.
#
# This suite exists because that was found on a fixture whose pit sits off-centre, so the finding and
# the fixture's lopsidedness were confounded. Three geometries separate them; see make_inputs.py.
#
# THE METRIC, stated once (and see tests/wtm_volume.py for why it is water and not head):
#   error = |V(run) - V(reference)| per cell, in METRES OF WATER, over LAND cells only (ocean cells are
#   pinned, contribute exactly 0, and only dilute a norm). Reported as MEDIAN and MAX together -- max
#   alone is what made a three-cell artefact look like a model-wide inaccuracy.
#   The reference is the dt = 1/1000 yr run OF THE SAME GEOMETRY AND ROUTING. Never the previous
#   refinement: under adaptive stepping two runs at different dt0 are different step PATTERNS, not a
#   refinement of one, so differencing them measures nothing.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
PY="${PY:-python3}"
[[ -d inputs_centre ]] || "$PY" make_inputs.py >/dev/null
make_work exitpath
export OMP_NUM_THREADS=1
YR=31536000
fail=0

emit() {  # $1 stem  $2 geometry  $3 mode  $4 dt  $5 nstep  $6 routing
  sed -e "s|@INPUTS@|$(readlink -f inputs_$2)|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|@MODE@|$3|g" -e "s|@DT@|$4|g" -e "s|@NSTEP@|$5|g" -e "s|@ROUTING@|$6|g" config.yaml > "$WORK/$1.yaml"
  grep -q '@' "$WORK/$1.yaml" && { echo "  FAIL  $1: unsubstituted token in config" >&2; fail=1; return 1; }
  "$WTM" "$WORK/$1.yaml" > "$WORK/$1.log" 2>&1 \
    || { echo "  FAIL  $1: run did not complete"; tail -3 "$WORK/$1.log" | sed 's/^/        /'; fail=1; return 1; }
}

# Four pairs. Each pair is a coarse run and its OWN dt = 1/1000 yr reference, same geometry, same routing.
for g in exit_right exit_left; do
  emit "${g}_ref"   "$g" fixed $((YR/1000)) 8000 continuous || continue
  emit "${g}_run"   "$g" fixed $((YR/16))    128  continuous || continue
done
# CONTROL: the same geometry with the routing OFF. If the concentration is an FSM factor it must vanish.
emit nofsm_ref exit_right fixed $((YR/1000)) 8000 off
emit nofsm_run exit_right fixed $((YR/16))    128 off
# SYMMETRY pair: the same symmetric geometry with the routing on and off. These need NO reference --
# the fixture is its own control, because a mirror-symmetric problem must give a mirror-symmetric answer.
emit sym_centre_off        centre fixed $((YR/16)) 128 off
emit sym_centre_continuous centre fixed $((YR/16)) 128 continuous
# EQUATOR-CENTRED: cos(lat) is even in latitude, so rows mirrored about the equator have identical cell
# widths and UP-DOWN becomes a fair test too. (A uniform-cellsize projected grid is not an option --
# src/grid_geometry.cpp supports geographic grids only.)
emit sym_centre_eq_off        centre_eq fixed $((YR/16)) 128 off
emit sym_centre_eq_continuous centre_eq fixed $((YR/16)) 128 continuous
# TIE-BROKEN MIRROR PAIR: each has ONE strictly-lowest outlet, and the two are exact mirrors of each
# other. A model with no directional preference must answer one as the mirror of the other.
emit tilt_e_run tilt_e fixed $((YR/16)) 128 continuous
emit tilt_w_run tilt_w fixed $((YR/16)) 128 continuous

TESTS="$(readlink -f ..)" WORK="$WORK" "$PY" - <<'PYEOF'
import os, sys, glob, re
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as V, rasterio, numpy as np
WORK=os.environ["WORK"]; HERE=os.path.dirname(os.path.abspath("run.sh"))
IDX=re.compile(r'_(\d{9})_')
fail=0
def check(ok, name, detail):
    global fail
    print(f"  {'OK  ' if ok else 'FAIL'} {name}  {detail}")
    if not ok: fail=1
def last(stem):
    fs=[f for f in glob.glob(f"{WORK}/{stem}_0*.tif") if IDX.search(f)]
    fs.sort(key=lambda p:int(IDX.search(p).group(1)))
    assert fs, f"{stem}: no output rasters"
    return rasterio.open(fs[-1]).read(1).astype(float)
def err(geom, run, ref):
    d=os.path.join("inputs_"+geom)
    phi=V.read_band(os.path.join(d,"exitpath_porosity.tif"))
    land=rasterio.open(os.path.join(d,"exitpath_ta_mask.tif")).read(1)!=0
    e=np.abs(V.volume_diff(last(run), last(ref), phi))
    return e, land

def sym_ud(geom, routing):
    w=last(f"sym_{geom}_{routing}")
    return float(np.abs(w - w[::-1, :]).max())

def cross_mirror():
    """|answer(tilt_e) - mirror(answer(tilt_w))|. Works where self-symmetry CANNOT: both runs have a
    unique lowest outlet, so nothing is decided by a tie-break, yet the pair is still an exact mirror."""
    a=last("tilt_e_run"); b=last("tilt_w_run")
    return float(np.abs(a - b[:, ::-1]).max())

def sym(geom, routing):
    """max|w - mirror_LR(w)| on the final field. Needs no reference run at all: the fixture is its own."""
    w=last(f"sym_{geom}_{routing}")
    return float(np.abs(w - w[:, ::-1]).max())

THRESH=0.10   # metres of water. The split between "carrying the lake's discharge" and "exact": on the
              # golden fixture 3 cells exceeded it and 193 did not, with nothing in between near it.
sides={}
for g in ("exit_right","exit_left"):
    e,land=err(g,f"{g}_run",f"{g}_ref")
    el=e[land]
    bad=(e>THRESH)&land
    nx=e.shape[1]; mid=nx//2
    L=int(bad[:, :mid].sum()); R=int(bad[:, mid:].sum())
    sides[g]=(L,R)
    print(f"  {g:<11} median {np.median(el):.4e} m   max {el.max():.4e} m   cells>{THRESH:g}m: {int(bad.sum())} (left {L}, right {R})")
print()
# 1. NON-VACUITY FIRST. If nothing exceeds the threshold there is no error to locate and every
#    assertion below would pass by emptiness -- the failure mode this whole suite is guarding against.
tot=sum(L+R for L,R in sides.values())
check(tot>0, "NON-VACUOUS", f"{tot} cells above {THRESH} m across the three geometries")
# 2. THE ERROR FOLLOWS THE WATER. Mirror the geometry, mirror the error.
L,R=sides["exit_right"]; check(R>L, "exit_right -> RIGHT", f"{R} right vs {L} left")
L,R=sides["exit_left"];  check(L>R, "exit_left  -> LEFT ", f"{L} left vs {R} right")
# 3. THE DIRECTIONAL-BUG CATCHER, and it CAUGHT ONE. `centre` is symmetric to machine precision in
#    every input field, on both axes (make_inputs writes one array and its mirror is itself). Left-right
#    is the axis that must be EXACT: mirroring swaps two cells at the SAME latitude, so their areas and
#    fluxes are identical. Up-down is NOT a fair test and is deliberately not asserted -- the grid spans
#    latitudes, so cell size varies north to south and the domain is not physically symmetric that way.
#
#    MEASURED on the answer itself, max|w - mirror_LR(w)|, `centre` geometry, 128 fixed steps:
#        surface_water.routing: off          0.0000e+00 m     <- EXACT. The groundwater solve is clean.
#        surface_water.routing: continuous   7.2081e+00 m     <- FillSpillMerge breaks it.
#    And it is not a time-stepping artefact: the dt = 1/1000 yr reference is asymmetric by 7.9827e+00 m,
#    slightly WORSE than the coarse run. Refining does not help because refining is not the problem.
#
#    The likely mechanism, UNTESTED and stated as a hypothesis: the plateau outside the pit is perfectly
#    flat, so the flow routing has ties everywhere and must break them somehow; a deterministic
#    index-ordered tie-break is a directional bias. If that is it, the asymmetry is a property of
#    degenerate flat terrain rather than a defect in the physics -- but it is not currently known, and a
#    real coastline is not flat, so nothing here says what happens on one.
lr_off = sym("centre", "off")
check(lr_off == 0.0, "POSITIVE CONTROL: solve is L-R exact",
      f"routing off -> max|w - mirror(w)| = {lr_off:.4e} m (must be exactly 0)")
lr_on = sym("centre", "continuous")
XFAIL_FLOOR = 1.0    # an order below the 7.2 m measured, so a real repair cannot hide under it
if lr_on == 0.0:
    check(False, "XFAIL centre L-R symmetry", "UNEXPECTED PASS -- FillSpillMerge is now L-R symmetric. "
          "This is the outcome the xfail waits for: record what changed and promote it to a real check.")
elif lr_on < XFAIL_FLOOR:
    check(False, "XFAIL centre L-R symmetry", f"{lr_on:.4e} m is below the {XFAIL_FLOOR:g} m floor but not "
          "zero -- the defect MOVED. Re-measure rather than re-tune the floor.")
else:
    print(f"  xfail  KNOWN: FillSpillMerge breaks L-R symmetry by {lr_on:.4e} m on symmetric terrain "
          f"(solve alone: {lr_off:.4e} m)")

# 3b. THE EQUATOR-CENTRED PAIR: both axes are now fair, so assert both.
for ax,fn in (("L-R", sym), ("U-D", sym_ud)):
    off = fn("centre_eq", "off")
    check(off == 0.0, f"centre_eq {ax}: solve exact", f"routing off -> {off:.4e} m (must be exactly 0)")
on_lr, on_ud = sym("centre_eq","continuous"), sym_ud("centre_eq","continuous")
print(f"  xfail  KNOWN: routing on, centre_eq -- L-R {on_lr:.4e} m, U-D {on_ud:.4e} m")

# 3c. THE DISCRIMINATOR. This is what separates "the depression hierarchy picks arbitrarily among TIED
#     outlets", which src/dephier.hpp documents and which no symmetric fixture can ever avoid, from "a
#     direction is preferred even when one outlet is strictly lowest". tilt_e and tilt_w each have ONE
#     lowest outlet and are exact mirrors of each other, so a tie-break decides nothing here.
#       cross-mirror ~ 0  -> the flat-case asymmetry IS the documented tie-break. Not a defect.
#       cross-mirror >> 0 -> something prefers a direction regardless. That WOULD be a defect.
xm = cross_mirror()
check(xm < 1.0e-6, "TIE-BROKEN cross-mirror", f"|tilt_e - mirror(tilt_w)| = {xm:.4e} m"
      + ("  -> the flat-case break is the documented arbitrary tie-choice, not a directional bias"
         if xm < 1.0e-6 else
         "  <- a direction is preferred even with a UNIQUE lowest outlet: NOT explained by tie-breaking"))

# 4. CONTROL: routing off. Same geometry; the concentration must vanish.
e,land=err("exit_right","nofsm_run","nofsm_ref"); el=e[land]
nbad=int(((e>THRESH)&land).sum())
print(f"\n  routing OFF  median {np.median(el):.4e} m   max {el.max():.4e} m   cells>{THRESH:g}m: {nbad}")
check(nbad==0, "CONTROL routing off", f"{nbad} cells above {THRESH} m (must be 0 -- the concentration is an FSM factor)")
sys.exit(fail)
PYEOF
rc=$?
[ $rc -eq 0 ] && [ $fail -eq 0 ] && echo "FSM EXIT-PATH: ALL PASSED" || { echo "FSM EXIT-PATH: FAILED"; exit 1; }
