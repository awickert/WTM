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
MATCH_TOL="${MATCH_TOL:-1e-8}"   # metres; dirichlet-vs-padding agreement (observed ~7e-12)
DIFF_MIN="${DIFF_MIN:-0.1}"      # metres; dirichlet-vs-neumann must differ by at least this
PY="${PY:-python3}"
CPD=100
export OMP_NUM_THREADS=1

emit() { # stem region surfdir southern_edge
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
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
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
BB="-wtm_anderson -wtm_eq_metric rms -wtm_eq_tol 1e-8"
SE_PAD=$("$PY" -c "print(-1.0/$CPD)")   # padded grid one cell further south

emit dir bcons    "$INP" 0       ; "$WTM" "$WORK/dir.cfg" -wtm_anderson $BB -wtm_land_boundary dirichlet         > "$WORK/dir.log" 2>&1 || { echo "RUN FAILED: dirichlet(anderson)"; tail -3 "$WORK/dir.log"; exit 2; }
emit pad bconspad "$INP" "$SE_PAD"; "$WTM" "$WORK/pad.cfg" -wtm_anderson $BB                                     > "$WORK/pad.log" 2>&1 || { echo "RUN FAILED: padding";   tail -3 "$WORK/pad.log"; exit 2; }
emit neu bcons    "$INP" 0       ; "$WTM" "$WORK/neu.cfg" -wtm_anderson $BB -wtm_land_boundary neumann_toposlope > "$WORK/neu.log" 2>&1 || { echo "RUN FAILED: neumann";   tail -3 "$WORK/neu.log"; exit 2; }
# Newton (analytic Jacobian, dt-continuation) must reach the SAME land-Dirichlet water table -> its off-map
# Dirichlet Jacobian tangent is consistent with the residual (FD-verified separately in tests/ghost_boundary).
emit nwt bcons    "$INP" 0       ; "$WTM" "$WORK/nwt.cfg" -wtm_newton -wtm_dt_continuation $BB -wtm_land_boundary dirichlet > "$WORK/nwt.log" 2>&1 || { echo "RUN FAILED: dirichlet(newton)"; tail -3 "$WORK/nwt.log"; exit 2; }

DIR=$(ls "$WORK"/dir_*.tif | tail -1); PAD=$(ls "$WORK"/pad_*.tif | tail -1); NEU=$(ls "$WORK"/neu_*.tif | tail -1); NWT=$(ls "$WORK"/nwt_*.tif | tail -1)
MATCH_TOL="$MATCH_TOL" DIFF_MIN="$DIFF_MIN" "$PY" - "$DIR" "$PAD" "$NEU" "$NWT" <<'PY'
import sys, os, numpy as np, rasterio
dir_, pad, neu, nwt = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
padi = pad[1:-1, 1:-1]                       # padded interior == the land-edge domain
mtol = float(os.environ["MATCH_TOL"]); dmin = float(os.environ["DIFF_MIN"])
match  = float(np.max(np.abs(dir_ - padi)))  # anderson dirichlet vs old padding
diff   = float(np.max(np.abs(dir_ - neu)))   # dirichlet vs neumann
newton = float(np.max(np.abs(nwt - dir_)))   # newton dirichlet vs anderson dirichlet
print(f"  dirichlet ghost vs old padding:  max|Δwtd| = {match:.3e} m  (tol {mtol})")
print(f"  dirichlet vs neumann_toposlope:  max|Δwtd| = {diff:.3e} m  (must exceed {dmin})")
print(f"  newton vs anderson (dirichlet):  max|Δwtd| = {newton:.3e} m  (tol 1e-6)")
ok = match <= mtol and diff >= dmin and newton <= 1e-6
if ok:
    print("PASS: land-edge ghost Dirichlet == old sea-level padding, distinct from Neumann, and Newton agrees")
    sys.exit(0)
if match > mtol:  print(f"FAIL: dirichlet vs padding {match:.3e} m > tol {mtol} m (the two should be the same BC)")
if diff < dmin:   print(f"FAIL: dirichlet vs neumann {diff:.3e} m < {dmin} m (selector had no effect?)")
if newton > 1e-6: print(f"FAIL: newton vs anderson {newton:.3e} m > 1e-6 m (Jacobian off-map Dirichlet tangent inconsistent?)")
sys.exit(1)
PY
