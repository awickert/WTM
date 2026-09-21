#!/usr/bin/env bash
# UNITS: NOT CONVERTED TO WATER VOLUME, AND MUST NOT BE (#65). This suite compares the computed water
# table against a CLOSED-FORM ANALYTIC SOLUTION -- a parabola in HEAD. The reference exists only in
# head; converting the measurement to volume would compare it against a reference in the wrong unit.
# ANALYTICAL boundary-condition tests: validate the ocean-Dirichlet and land-Neumann BCs against CLOSED-FORM
# solutions, not snapshot goldens. On a flat sea-level domain with uniform recharge and the water table mounded
# above the surface (constant transmissivity T = ksat*(1.5+fdepth)), steady groundwater solves T h'' = -R, whose
# solution is a parabola:
#   DIRICHLET (ocean h=0 at both x-ends): symmetric parabola  h(x) = A x (L-x),  h = 0 at both ends.
#   NEUMANN   (ocean-left, land no-flow right): half-parabola with ZERO GRADIENT (vertex) exactly at the
#             no-flow boundary face -- the analytical signature of a no-flow edge.
# We assert the water table matches these forms to solver tolerance (observed ~3e-10 m). Ocean-Dirichlet is thus
# analytically anchored here; land-Dirichlet is tied to it by tests/boundary_consistency (dirichlet == padding ==
# ocean-Dirichlet); land-Neumann is analytically anchored here.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/anbcD_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work anbc
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, ONE-SIDED, and the binding arm is NEUMANN SLOPE: three arms use this bound
#   and their residuals span two orders -- dirichlet 3.548e-10 m, neumann flat 3.427e-10 m,
#   neumann slope 3.479e-08 m. The bound is sized by the worst of them (28.7x headroom), not by the
#   best (2800x), because a single tolerance must clear the hardest case. The slope arm is larger
#   because the terrain gradient (0.05/cell) makes the closed-form parabola an approximation there
#   rather than an identity.
FIT_TOL="${FIT_TOL:-1e-6}"   # metres; max deviation of the water table from the closed-form parabola
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/boundary_analytic/config.yaml. Every setting the run resolves
# to is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# solver.time_step.error_tol is 1e-08 in that file and that is deliberate: this suite compares against
# ANALYTIC solutions, so the discretisation must not contribute to the error being judged.
emit() { # $1 stem, $2 io.region (REQUIRED -- the region IS the arm here)
  local rg="${2:?emit needs an io.region: each arm names its own analytic fixture}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" -e "s|@REGION@|$rg|g" \
      config.yaml > "$WORK/$1.yaml"
}
# constant-T regime: flat sea-level topo + uniform recharge mounded above the surface -> only constant-T
# diffusion + uniform source -> exact parabola.
#
# WHAT PERMITS THE MOUND. Two settings together, and neither alone is enough:
#   surface_water.collection.method: off        -- active_set would pin wtd <= 0, deleting the mound
#   evaporation.tapers.surface_transition: true -- with the taper OFF the model removes ALL surface
#                                                  water every step (irf.cpp: `if (wtd > 0) wtd = 0`)
# An earlier version of this comment claimed `collection.method: off` was "the physical successor to
# -wtm_dev_allow_aboveground_water_columns". It is not, and that error is what emptied this suite: the
# collector was off, the taper was off too, and the clamp in irf.cpp held every arm at wtd == 0 for the
# whole run while the parabola fit reported residual 0.000e+00 and passed. See the taper note in config.yaml.
FL=""   # no CLI flags: every setting this suite depends on is in config.yaml (#83). Taper-1 sink retired (#7)

emit dir anbcD; "$WTM" "$WORK/dir.yaml" $FL > "$WORK/dir.log" 2>&1 || { echo "RUN FAILED: dirichlet"; tail -3 "$WORK/dir.log"; exit 2; }
emit neu anbcN; "$WTM" "$WORK/neu.yaml" $FL  > "$WORK/neu.log" 2>&1 || { echo "RUN FAILED: neumann"; tail -3 "$WORK/neu.log"; exit 2; }
emit slp anbcS; "$WTM" "$WORK/slp.yaml" $FL  > "$WORK/slp.log" 2>&1 || { echo "RUN FAILED: sloped neumann"; tail -3 "$WORK/slp.log"; exit 2; }

DIR=$(ls "$WORK"/dir_*.tif | tail -1); NEU=$(ls "$WORK"/neu_*.tif | tail -1); SLP=$(ls "$WORK"/slp_*.tif | tail -1)
FIT_TOL="$FIT_TOL" SLOPE="0.05" "$PY" - "$DIR" "$NEU" "$SLP" <<'PY'
import sys, os, numpy as np, rasterio
tol = float(os.environ["FIT_TOL"]); slope = float(os.environ["SLOPE"])

# A NAN COMPARISON FAILS OPEN, so every measured quantity is checked for finiteness before it is judged.
# This is not hypothetical: while this suite was comparing an identically-zero field, the Neumann vertex
# was -c[1]/(2*c[0]) = 0/0 = nan, the script PRINTED "vertex at x = nan", and `abs(nan - 22.0) > 0.1` is
# False -- so the arm passed on a quantity that did not exist. Route every assertion through here.
def bad(name, value, ok_expr):
    if not np.isfinite(value):
        print(f"  NOT-FINITE: {name} = {value} -- a nan/inf assertion would pass silently; failing instead")
        return True
    return not ok_expr
dirf = rasterio.open(sys.argv[1]).read(1).astype(float)[1]   # middle row (uniform in y); output is depth-to-wt (wtd)
neuf = rasterio.open(sys.argv[2]).read(1).astype(float)[1]
slpf = rasterio.open(sys.argv[3]).read(1).astype(float)[1]
NX = len(dirf); x = np.arange(NX) + 0.5
noflow_face = float(x[-1] + 0.5)               # the no-flow boundary face is half a cell beyond the edge centre
ok = True

# --- DIRICHLET: symmetric parabola h = A (x-x0)(x1-x), h=0 at the ocean cell centres (cols 0 and NX-1) ---
xi, hi = x[1:-1], dirf[1:-1]; x0, x1 = x[0], x[-1]
basis = (xi - x0) * (x1 - xi); A = float(np.sum(basis * hi) / np.sum(basis * basis))
d_resid = float(np.max(np.abs(hi - A * basis)))
d_edges = float(max(abs(dirf[0]), abs(dirf[-1])))
print(f"  DIRICHLET (ocean both ends): parabola residual = {d_resid:.3e} m (tol FIT_TOL={tol})")
print(f"  DIRICHLET (ocean both ends): |h| at ocean ends = {d_edges:.3e} m (tol FIT_TOL={tol})")
if bad('dirichlet residual', d_resid, d_resid <= tol) or bad('|h| at ocean ends', d_edges, d_edges <= tol): ok = False

# --- NEUMANN (flat): half-parabola over the land cells; zero-gradient vertex at the no-flow face ---
xn, hn = x[1:], neuf[1:]
c = np.polyfit(xn, hn, 2); n_resid = float(np.max(np.abs(hn - np.polyval(c, xn)))); vertex = float(-c[1]/(2*c[0]))
print(f"  NEUMANN flat  (ocean-left, land no-flow right): parabola residual = {n_resid:.3e} m (tol FIT_TOL={tol})")
# The DEVIATION is what is compared, so the deviation is what is printed -- the reader should not have
# to subtract two numbers to see whether an assertion is close to its bound.
print(f"  NEUMANN flat  vertex vs no-flow face (x = {noflow_face:.1f}): |offset| = {abs(vertex - noflow_face):.3f} (tol 0.1)")
if bad('neumann residual', n_resid, n_resid <= tol) or bad('neumann vertex', vertex, abs(vertex - noflow_face) <= 0.1): ok = False

# --- NEUMANN (sloped): terrain-following. The WATER-TABLE DEPTH wtd (the output) is the half-parabola whose
#     zero-gradient vertex is on the no-flow face -> d(wtd)/dx = 0 there = CONSTANT DEPTH (parallel to terrain).
#     Consequently the head gradient at the edge is the terrain slope, not zero -- the topo-following signature. ---
xs, ws = x[1:], slpf[1:]
cs = np.polyfit(xs, ws, 2); s_resid = float(np.max(np.abs(ws - np.polyval(cs, xs)))); s_vertex = float(-cs[1]/(2*cs[0]))
head = slpf + slope * np.arange(NX)            # h = wtd + topo, topo = slope * col
h_grad_edge = float((head[-1] - head[-2]))     # head gradient at the no-flow edge (should be ~ slope, not 0)
# TERRAIN SLOPE AFTER THE MARKER. As "(topo={slope}/cell)" it is a bare decimal sitting before the
# bound, and 0.05 outranks a 3.5e-08 residual under the magnitude rule -- the assertion was being
# compared against its own label. Introduced here today; caught by the probe reporting DID NOT BITE.
print(f"  NEUMANN slope: wtd parabola residual = {s_resid:.3e} m (tol FIT_TOL={tol}) -- terrain {slope}/cell")
print(f"  NEUMANN slope vertex vs no-flow face (x = {noflow_face:.1f}): |offset| = {abs(s_vertex - noflow_face):.3f} (tol 0.1)")
print(f"  NEUMANN slope edge head gradient vs terrain slope: |offset| = {abs(h_grad_edge - slope):.4f} (tol 0.02)")
if (bad('sloped residual', s_resid, s_resid <= tol) or bad('sloped vertex', s_vertex, abs(s_vertex - noflow_face) <= 0.1)
        or bad('sloped edge head gradient', h_grad_edge, abs(h_grad_edge - slope) <= 0.02)): ok = False

if ok:
    print("PASS: ocean-Dirichlet and land-Neumann (flat + terrain-following) match their closed-form solutions")
    sys.exit(0)
print(f"FAIL: a closed-form match exceeded tol {tol} m, a Neumann vertex is off the no-flow face, "
      f"or the sloped head gradient is not the terrain slope")
sys.exit(1)
PY
