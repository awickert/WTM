#!/usr/bin/env bash
# Boundary-consistency regression: the new land-edge ghost-node Dirichlet reproduces the OLD sea-level
# padding method, and is distinct from the default terrain-slope Neumann.
#
#   DIRICHLET   : bcons (land edges) run with -wtm_land_boundary dirichlet  -> land edges become Dirichlet h=0
#                 via ghost nodes (ghost = an ocean neighbour: head 0, surface T).
#   OLD PADDING : bconspad (same interior, +1-cell ocean ring at sea level) run with the DEFAULT boundary
#                 -> the ocean ring imposes Dirichlet h=0 all around. This IS the legacy sea-level padding
#                 (on an ocean-ringed grid the legacy setEdges path and the default coincide).
#   NEUMANN     : bcons run with the DEFAULT (-wtm_land_boundary neumann_toposlope) -> terrain-following no-flow.
#
# Asserts: (1) DIRICHLET == OLD PADDING interior to ~machine precision (the two are the same BC), and
#          (2) DIRICHLET differs from NEUMANN by a real margin (the selector actually changes the physics).
# The padded grid's southern_edge is shifted one cell south so the shared interior cells sit at identical
# latitudes (identical geometry) -- without that, cos-lat cell-size drift would blur the machine-precision match.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/bcons_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work bcons
# metres OF WATER VOLUME (tests/wtm_volume.py), not head (#61/#65).
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, SEPARATING, and the gap here is ELEVEN ORDERS: two implementations of the
#   SAME boundary condition (dirichlet ghost, old padding) agree to 5.446e-12 m of water volume,
#   while a genuinely DIFFERENT condition (neumann_toposlope, the DIFF_MIN arm below) differs by
#   4.645e-01 m. Both edges are measured in this run. The bound sits inside that gap with 459x of
#   headroom above the agreeing pair.
MATCH_TOL="${MATCH_TOL:-2.5e-9}" # dirichlet-vs-padding agreement, in water (was 1e-8 head)
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, SEPARATING, same measured gap read from the other side: dirichlet vs
#   neumann_toposlope differ by 4.645e-01 m of water where two spellings of one BC agree to
#   5.446e-12 m. The floor sits 4.6x below the real difference and 10 orders above the agreement,
#   so it fires exactly when the selector stops selecting.
DIFF_MIN="${DIFF_MIN:-0.1}"      # metres OF WATER VOLUME; dirichlet-vs-neumann must differ by at least this.
                                 # DELIBERATELY LEFT AT 0.1 rather than scaled to 0.025. This is a floor the
                                 # separation must EXCEED, so keeping the number while the measured value
                                 # scales by phi makes the check HARDER, not weaker: the separation went
                                 # 1.858 m head -> 4.645e-01 m water volume, so the margin tightens from 18.6x to
                                 # 4.6x. Scaling it would have preserved the ratio and bought nothing; a
                                 # bites-floor is worth keeping strict.
PY="${PY:-python3}"
CPD=100
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/boundary_consistency/config.yaml. Every setting the run
# resolves to is stated there, and tests/config_identity.py enforces it (this suite is on
# unconditional since #79 Phase 5).
#
# The newton arm needs mode: ramp, which in turn makes the model record solver.newton.dt0 and
# solver.time_step.dt_max and NO step norm -- so that arm adds two keys and drops one, via the
# #@RAMP@ / #@RAMPNEWTON@ placeholders that sit at the right nesting level.
emit() { # $1 stem, $2 io.region, $3 boundaries.land, $4 solver.method, $5 time_integration,
         # $6 time_step.mode   (ALL REQUIRED)
  local rg="${2:?emit needs an io.region: bcons or bconspad}"
  local bc="${3:?emit needs a boundaries.land}"
  local m="${4:?emit needs a solver.method}"
  local ti="${5:?emit needs a time_integration: it follows the method}"
  local sm="${6:?emit needs a time_step.mode: newton requires ramp}"
  local ramp=(-e "/^#@RAMP@/d" -e "/^#@RAMPNEWTON@/d")
  if [ "$sm" = ramp ]; then
      ramp=(-e "/^    norm: rms/d"
            -e 's|^#@RAMP@|    dt_max: "2419200000s"   # ramp only|'
            -e "s|^    error_tol: 1e-08.*|    error_tol: 0.1   # the value resolved under mode: ramp|"
            -e "s|^#@RAMPNEWTON@|  newton:\n    dt0: \"12096s\"   # ramp only; the `s` is required, a bare number is YEARS|")
  fi
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|@REGION@|$rg|g" -e "s|@LANDBC@|$bc|g" -e "s|@METHOD@|$m|g" \
      -e "s|@INTEG@|$ti|g" -e "s|@STEPMODE@|$sm|g" \
      "${ramp[@]}" config.yaml > "$WORK/$1.yaml"
}
BB=""   # solver.method: anderson now travels in the config (METHOD=)
SE_PAD=$("$PY" -c "print(-1.0/$CPD)")   # padded grid one cell further south

emit dir bcons    dirichlet_sea_level anderson tr-bdf2        adaptive ; "$WTM" "$WORK/dir.yaml" $BB         > "$WORK/dir.log" 2>&1 || { echo "RUN FAILED: dirichlet(anderson)"; tail -3 "$WORK/dir.log"; exit 2; }
emit pad bconspad neumann_toposlope   anderson tr-bdf2        adaptive ; "$WTM" "$WORK/pad.yaml" $BB                                     > "$WORK/pad.log" 2>&1 || { echo "RUN FAILED: padding";   tail -3 "$WORK/pad.log"; exit 2; }
emit neu bcons    neumann_toposlope   anderson tr-bdf2        adaptive ; "$WTM" "$WORK/neu.yaml" $BB > "$WORK/neu.log" 2>&1 || { echo "RUN FAILED: neumann";   tail -3 "$WORK/neu.log"; exit 2; }
# Newton (analytic Jacobian) must reach the SAME land-Dirichlet water table -> its off-map Dirichlet Jacobian
# tangent is consistent with the residual (FD-verified separately in tests/ghost_boundary).
#
# This arm used PLAIN Newton (MODE=fixed) on the grounds that "this small well-posed problem converges
# directly (cycle ~3), so dt-continuation is unnecessary", and that continuation would grind to the
# total_time cap at eq_tol 1e-8. BOTH premises died with dev.storage_form defaulting to volume (879a188):
# the volume form folds the storage into f with RHS b=0 and scales by Sy, which changes the line search,
# and plain Newton now DIVERGED_LINE_SEARCH after 3 iterations on this fixture. The arm runs the WORKING
# recipe instead -- solver.method: newton implies dt_continuation -- so it also tests what a user gets.
# Measured after the change: equilibrium at cycle 3, 1.0 s. The feared grind does not happen.
emit nwt bcons    dirichlet_sea_level newton   backward-euler ramp     ; "$WTM" "$WORK/nwt.yaml" $BB > "$WORK/nwt.log" 2>&1 || { echo "RUN FAILED: dirichlet(newton)"; tail -3 "$WORK/nwt.log"; exit 2; }

DIR=$(ls "$WORK"/dir_*.tif | tail -1); PAD=$(ls "$WORK"/pad_*.tif | tail -1); NEU=$(ls "$WORK"/neu_*.tif | tail -1); NWT=$(ls "$WORK"/nwt_*.tif | tail -1)
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# PROMOTED FROM A LITERAL (#121): reachable from outside so assertion_probe can tighten
# it and confirm the assertion still fails when it should.
# DERIVED 2026-09-22, SEPARATING but BLUNT: Newton and Anderson on the same Dirichlet problem agree
#   to 5.365e-12 m, so headroom is 1.9e5. Left as is deliberately -- the failure this guards
#   against is a wrong off-map Dirichlet tangent in the Jacobian, which does not produce a slightly
#   larger number: it moves the answer to the 1e-1 scale of the DIFF_MIN arm. A bound anywhere in
#   the eleven-order gap catches it, so the headroom is not the property that matters here.
NEWTON_TOL="${NEWTON_TOL:-1e-6}"   # newton vs the analytic boundary solution
MATCH_TOL="$MATCH_TOL" DIFF_MIN="$DIFF_MIN" PHI="$INP/bcons_porosity.tif" TESTS="$(readlink -f ..)" \
  NEWTON_TOL="$NEWTON_TOL" "$PY" - "$DIR" "$PAD" "$NEU" "$NWT" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
newton_tol = float(os.environ["NEWTON_TOL"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

dir_, pad, neu, nwt = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
padi = pad[1:-1, 1:-1]                       # padded interior == the land-edge domain
mtol = float(os.environ["MATCH_TOL"]); dmin = float(os.environ["DIFF_MIN"])
phi = VOL.read_band(os.environ["PHI"])
match  = float(VOL.volume_diff(dir_, padi, phi).max())   # anderson dirichlet vs old padding, IN WATER VOLUME VOLUME
diff   = float(VOL.volume_diff(dir_, neu, phi).max())    # dirichlet vs neumann, IN WATER VOLUME VOLUME
newton = float(VOL.volume_diff(nwt, dir_, phi).max())    # newton vs anderson dirichlet, IN WATER VOLUME VOLUME
print(f"  dirichlet ghost vs old padding:  max|ΔV| = {match:.3e} m water volume  (tol MATCH_TOL={mtol})")
print(f"  dirichlet vs neumann_toposlope:  max|ΔV| = {diff:.3e} m water (min DIFF_MIN={dmin})")
print(f"  newton vs anderson (dirichlet):  max|ΔV| = {newton:.3e} m water volume (tol NEWTON_TOL={newton_tol})")
ok = match <= mtol and diff >= dmin and newton <= newton_tol
if ok:
    print("PASS: land-edge ghost Dirichlet == old sea-level padding, distinct from Neumann, and Newton agrees")
    sys.exit(0)
if match > mtol:  print(f"FAIL: dirichlet vs padding max|ΔV| = {match:.3e} m water volume (tol MATCH_TOL={mtol}) -- the two should be the same BC")
if diff < dmin:   print(f"FAIL: dirichlet vs neumann max|ΔV| = {diff:.3e} m (min DIFF_MIN={dmin}) -- selector had no effect?")
if newton > newton_tol: print(f"FAIL: newton vs anderson max|ΔV| = {newton:.3e} m (tol NEWTON_TOL={newton_tol}) -- Jacobian off-map Dirichlet tangent inconsistent?")
sys.exit(1)
PY
