#!/usr/bin/env bash
# runoff_collector selector: the input-file choice of how the wtd<=0 seepage face is enforced. One boundary
# condition, three enforcements (see benchmark/SURFACE_WATER_ROUTING.md):
#   implicit : in-residual seepage (direct_to_runoff) -- pins wtd=0, dt-independent, exact (Anderson today).
#   explicit : post-solve clamp -- robust on every solver, a dt-lagged form of the same face.
#   off      : no collection -- above-surface water piles up (NONPHYSICAL; warns).
# On a partial-seepage fixture (interior driven to the surface) these are distinguishable by the peak water
# table. Asserts, on the matrix-free Anderson path unless noted:
#   IMPLICIT : table pinned at the surface (0 <= max wtd < 0.5 m: a seepage face, not a pile) with seeping cells.
#   EXPLICIT : table clamped to exactly the surface (|max wtd| < 1e-4 m) with seeping cells.
#   OFF      : water piles far above the surface (max wtd > 5 m) AND the nonphysical warning is printed.
#   UNSET    : the legacy band sink holds the table just BELOW the surface (-1 < max wtd < 0), managed (no pile).
#   AGREE    : implicit and explicit land within a few cm (same face, converging as dt->0).
#   SOLVER   : explicit also converges on the default Picard path (it needs no tangent).
# This test asserts the modes via the config KEY, so it also fences off the -wtm_<flag> 0 CLI mis-parse hazard.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/rcoll_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/rc_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem  collector-line
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
$2
cells_per_degree 120
southern_edge 0
deltat 2419200
total_cycles 120
cycles_to_save 120
maxiter 50
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region rcoll
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
run() { # stem  collector-line  extra-flags
  emit "$1" "$2"
  "$WTM" "$WORK/$1.cfg" -wtm_anderson $3 -wtm_eq_tol 0 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
run implicit "runoff_collector implicit" ""
run explicit "runoff_collector explicit" ""
run off      "runoff_collector off"      ""
run unset    ""                          ""
# explicit on the DEFAULT Picard path (no -wtm_anderson): must converge (no tangent needed)
emit picard "runoff_collector explicit"
"$WTM" "$WORK/picard.cfg" -wtm_eq_tol 0 > "$WORK/picard.log" 2>&1 \
  || { echo "RUN FAILED: explicit on Picard"; tail -3 "$WORK/picard.log"; exit 2; }
OFFWARN=$(grep -c "WARNING \[runoff_collector=off\]" "$WORK/off.log" || true)

IM=$(ls "$WORK"/implicit_*.tif | tail -1); EX=$(ls "$WORK"/explicit_*.tif | tail -1)
OF=$(ls "$WORK"/off_*.tif | tail -1);      UN=$(ls "$WORK"/unset_*.tif | tail -1)
OFFWARN="$OFFWARN" "$PY" - "$IM" "$EX" "$OF" "$UN" <<'PY'
import sys, os, numpy as np, rasterio
im, ex, of, un = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
def interior(a): return a[1:-1, 1:-1]
im_mx, ex_mx, of_mx, un_mx = (float(interior(a).max()) for a in (im, ex, of, un))
im_seep = int((interior(im) > -1e-3).sum()); ex_seep = int((interior(ex) > -1e-3).sum())
agree = float(np.max(np.abs(im - ex)))
offwarn = int(os.environ["OFFWARN"]) > 0
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}")
    ok = ok and cond
check("IMPLICIT (seepage face, not piled)", (0.0 - 1e-3 <= im_mx < 0.5) and im_seep > 0,
      f"max wtd = {im_mx:.4f} m, seeping cells = {im_seep}")
check("EXPLICIT (clamped to surface)",      abs(ex_mx) < 1e-4 and ex_seep > 0,
      f"max wtd = {ex_mx:.4e} m, seeping cells = {ex_seep}")
check("OFF (piles + warns)",                of_mx > 5.0 and offwarn,
      f"max wtd = {of_mx:.2f} m, warning printed = {offwarn}")
check("UNSET (legacy band sink, sub-surface)", -1.0 < un_mx < 0.0,
      f"max wtd = {un_mx:.4f} m (held below the surface)")
check("AGREE implicit vs explicit",         agree < 0.1,
      f"max|implicit - explicit| = {agree:.3e} m")
print("PASS: runoff_collector modes behave as specified" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
echo "  SOLVER: explicit converged on the default Picard path (no tangent needed)"
