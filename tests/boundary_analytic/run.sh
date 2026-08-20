#!/usr/bin/env bash
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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/anbcD_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/anbc_XXXX); trap 'rm -rf "$WORK"' EXIT
FIT_TOL="${FIT_TOL:-1e-6}"   # metres; max deviation of the water table from the closed-form parabola
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem region
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 1
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1000
southern_edge 0
deltat 2419200
total_cycles 20000
cycles_to_save 20000
maxiter 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region $2
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
# constant-T regime: flat sea-level topo + uniform recharge mounded above the surface (ponding allowed, ALL
# wtd-dependent removals off) -> the only physics is constant-T diffusion + uniform source -> exact parabola.
FL="-wtm_anderson -wtm_allow_surface_ponding -wtm_evap_taper 0 -wtm_surface_sink 0 -wtm_extinction 0 -wtm_eq_metric rms -wtm_eq_tol 1e-8"

emit dir anbcD; "$WTM" "$WORK/dir.cfg" $FL > "$WORK/dir.log" 2>&1 || { echo "RUN FAILED: dirichlet"; tail -3 "$WORK/dir.log"; exit 2; }
emit neu anbcN; "$WTM" "$WORK/neu.cfg" $FL -wtm_land_boundary neumann_toposlope > "$WORK/neu.log" 2>&1 || { echo "RUN FAILED: neumann"; tail -3 "$WORK/neu.log"; exit 2; }

DIR=$(ls "$WORK"/dir_*.tif | tail -1); NEU=$(ls "$WORK"/neu_*.tif | tail -1)
FIT_TOL="$FIT_TOL" "$PY" - "$DIR" "$NEU" <<'PY'
import sys, os, numpy as np, rasterio
tol = float(os.environ["FIT_TOL"])
dirf = rasterio.open(sys.argv[1]).read(1).astype(float)[1]   # middle row (uniform in y)
neuf = rasterio.open(sys.argv[2]).read(1).astype(float)[1]
NX = len(dirf); x = np.arange(NX) + 0.5
ok = True

# --- DIRICHLET: symmetric parabola h = A (x-x0)(x1-x), h=0 at the ocean cell centres (cols 0 and NX-1) ---
xi, hi = x[1:-1], dirf[1:-1]; x0, x1 = x[0], x[-1]
basis = (xi - x0) * (x1 - xi); A = float(np.sum(basis * hi) / np.sum(basis * basis))
d_resid = float(np.max(np.abs(hi - A * basis)))
d_edges = float(max(abs(dirf[0]), abs(dirf[-1])))
print(f"  DIRICHLET (ocean both ends): parabola residual = {d_resid:.3e} m; |h| at ocean ends = {d_edges:.3e} m")
if d_resid > tol or d_edges > tol: ok = False

# --- NEUMANN: half-parabola over the land cells (col 0 is ocean h=0); zero-gradient vertex at the no-flow face ---
xn, hn = x[1:], neuf[1:]                       # land cells, incl. the no-flow right edge
c = np.polyfit(xn, hn, 2)                      # general parabola a x^2 + b x + c
n_resid = float(np.max(np.abs(hn - np.polyval(c, xn))))
vertex = float(-c[1] / (2 * c[0]))             # x where dh/dx = 0
noflow_face = float(x[-1] + 0.5)               # the no-flow boundary face is half a cell beyond the edge centre
print(f"  NEUMANN (ocean-left, land no-flow right): parabola residual = {n_resid:.3e} m; "
      f"zero-gradient vertex at x = {vertex:.3f} (no-flow face x = {noflow_face:.1f})")
if n_resid > tol or abs(vertex - noflow_face) > 0.1: ok = False

if ok:
    print("PASS: ocean-Dirichlet and land-Neumann match their closed-form parabola solutions")
    sys.exit(0)
print(f"FAIL: closed-form match exceeded tol {tol} m, or the Neumann vertex is off the no-flow face")
sys.exit(1)
PY
