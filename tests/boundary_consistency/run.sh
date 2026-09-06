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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/bcons_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/bcons_XXXX); trap 'rm -rf "$WORK"' EXIT
# metres OF WATER (tests/wtm_water.py), not head (#61/#65).
MATCH_TOL="${MATCH_TOL:-2.5e-9}" # dirichlet-vs-padding agreement, in water (was 1e-8 head)
DIFF_MIN="${DIFF_MIN:-0.1}"      # metres OF WATER; dirichlet-vs-neumann must differ by at least this.
                                 # DELIBERATELY LEFT AT 0.1 rather than scaled to 0.025. This is a floor the
                                 # separation must EXCEED, so keeping the number while the measured value
                                 # scales by phi makes the check HARDER, not weaker: the separation went
                                 # 1.858 m head -> 4.645e-01 m water, so the margin tightens from 18.6x to
                                 # 4.6x. Scaling it would have preserved the ratio and bought nothing; a
                                 # bites-floor is worth keeping strict.
PY="${PY:-python3}"
CPD=100
export OMP_NUM_THREADS=1

emit() { # stem region surfdir southern_edge   [env: LAND_BC=dirichlet for the Dirichlet arms]
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
land_boundary ${LAND_BC:-neumann_toposlope}
# CONVERGE TIGHTER THAN YOU COMPARE. Assertion (1) below says two spellings of the SAME boundary
# condition land on the same interior, so what actually bounds their agreement is how far each solve
# was driven -- not the boundary condition, which is identical by construction. At the default water
# tolerance (1e-8, #61) they agree to 3.6e-08 m, which is looser than the 1e-8 m the test compares at.
# Measured here: vol_tol 1e-8 -> 3.606e-08 m, 1e-10 -> 6.038e-09 m, 1e-12 -> 2.179e-11 m -- a clean
# convergence-level artifact, so drive the solves three decades past the comparison instead of
# loosening the comparison. (Under the OLD head-judged default these happened to agree to ~7e-12,
# which is why no such setting was needed before.)
convergence_water_volume_tol 1e-12
${METHOD:+solver_method $METHOD}
${DTC:+dt_continuation $DTC}
fsm_on 0
# Pinned to 'explicit' on purpose. This test's subject is the land-edge BOUNDARY CONDITION, not the
# exfiltration enforcement. Its Newton arm uses PLAIN Newton deliberately (the comment below explains
# why dt-continuation is avoided here), and plain Newton + active_set diverges in the line search on
# this fixture -- the semismooth kink, which dt-continuation cures but which would make this test slow.
# Holding the enforcement fixed keeps the boundary comparison clean. (Newton + active_set IS supported;
# it converges on tests/multilake with or without continuation. See the README solution-mode table.)
runoff_collector explicit
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree $CPD
southern_edge $4
deltat 2419200
total_time 96768000000s
save_nreport_interval 800
report_interval 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $3
region $2
supplied_wt 0
eq_tol 1e-8
eq_metric rms
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
BB=""   # solver.method: anderson now travels in the config (METHOD=)
SE_PAD=$("$PY" -c "print(-1.0/$CPD)")   # padded grid one cell further south

LAND_BC=dirichlet METHOD=anderson emit dir bcons    "$INP" 0       ; "$WTM" "$WORK/dir.yaml" $BB         > "$WORK/dir.log" 2>&1 || { echo "RUN FAILED: dirichlet(anderson)"; tail -3 "$WORK/dir.log"; exit 2; }
emit pad bconspad "$INP" "$SE_PAD"; "$WTM" "$WORK/pad.yaml" $BB                                     > "$WORK/pad.log" 2>&1 || { echo "RUN FAILED: padding";   tail -3 "$WORK/pad.log"; exit 2; }
METHOD=anderson emit neu bcons    "$INP" 0       ; "$WTM" "$WORK/neu.yaml" $BB > "$WORK/neu.log" 2>&1 || { echo "RUN FAILED: neumann";   tail -3 "$WORK/neu.log"; exit 2; }
# Newton (analytic Jacobian) must reach the SAME land-Dirichlet water table -> its off-map Dirichlet Jacobian
# tangent is consistent with the residual (FD-verified separately in tests/ghost_boundary).
#
# This arm used PLAIN Newton (DTC=false) on the grounds that "this small well-posed problem converges
# directly (cycle ~3), so dt-continuation is unnecessary", and that continuation would grind to the
# total_time cap at eq_tol 1e-8. BOTH premises died with dev.storage_form defaulting to volume (879a188):
# the volume form folds the storage into f with RHS b=0 and scales by Sy, which changes the line search,
# and plain Newton now DIVERGED_LINE_SEARCH after 3 iterations on this fixture. The arm runs the WORKING
# recipe instead -- solver.method: newton implies dt_continuation -- so it also tests what a user gets.
# Measured after the change: equilibrium at cycle 3, 1.0 s. The feared grind does not happen.
LAND_BC=dirichlet METHOD=newton emit nwt bcons    "$INP" 0       ; "$WTM" "$WORK/nwt.yaml" $BB > "$WORK/nwt.log" 2>&1 || { echo "RUN FAILED: dirichlet(newton)"; tail -3 "$WORK/nwt.log"; exit 2; }

DIR=$(ls "$WORK"/dir_*.tif | tail -1); PAD=$(ls "$WORK"/pad_*.tif | tail -1); NEU=$(ls "$WORK"/neu_*.tif | tail -1); NWT=$(ls "$WORK"/nwt_*.tif | tail -1)
MATCH_TOL="$MATCH_TOL" DIFF_MIN="$DIFF_MIN" PHI="$INP/bcons_porosity.tif" TESTS="$(readlink -f ..)" \
  "$PY" - "$DIR" "$PAD" "$NEU" "$NWT" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_water as W                      # ONE verified V(wtd); see tests/verify_wtm_water.sh

dir_, pad, neu, nwt = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
padi = pad[1:-1, 1:-1]                       # padded interior == the land-edge domain
mtol = float(os.environ["MATCH_TOL"]); dmin = float(os.environ["DIFF_MIN"])
phi = W.read_band(os.environ["PHI"])
match  = float(W.water_diff(dir_, padi, phi).max())   # anderson dirichlet vs old padding, IN WATER
diff   = float(W.water_diff(dir_, neu, phi).max())    # dirichlet vs neumann, IN WATER
newton = float(W.water_diff(nwt, dir_, phi).max())    # newton vs anderson dirichlet, IN WATER
print(f"  dirichlet ghost vs old padding:  max|ΔV| = {match:.3e} m water  (tol {mtol})")
print(f"  dirichlet vs neumann_toposlope:  max|ΔV| = {diff:.3e} m water  (must exceed {dmin})")
print(f"  newton vs anderson (dirichlet):  max|ΔV| = {newton:.3e} m water  (tol 1e-6)")
ok = match <= mtol and diff >= dmin and newton <= 1e-6
if ok:
    print("PASS: land-edge ghost Dirichlet == old sea-level padding, distinct from Neumann, and Newton agrees")
    sys.exit(0)
if match > mtol:  print(f"FAIL: dirichlet vs padding {match:.3e} > tol {mtol} m water (the two should be the same BC)")
if diff < dmin:   print(f"FAIL: dirichlet vs neumann {diff:.3e} m < {dmin} m (selector had no effect?)")
if newton > 1e-6: print(f"FAIL: newton vs anderson {newton:.3e} m > 1e-6 m (Jacobian off-map Dirichlet tangent inconsistent?)")
sys.exit(1)
PY
