#!/usr/bin/env bash
# Mask-aware ghost-boundary test (now the default; task #96). On a domain with REAL LAND at the N/S/E
# edges and ocean at the west (see make_inputs.py), verify that the off-map land-slope Neumann ghost is
# implemented consistently across every solver assembly site:
#
#   1. MPI determinism   -- cc (Anderson) with the ghost boundary is bit-for-bit identical on 1 vs N ranks
#                           (exercises the off-map reads under domain decomposition).
#   2. Cross-scheme      -- cc / TR-BDF2 / BDF2-on-V / Newton integrate the SAME steady residual, so under
#                           the ghost boundary they must converge to the SAME water table.
#   3. Newton Jacobian   -- ||J - Jfd||_F/||J||_F stays small (~1e-4) with the ghost boundary ON, i.e. the
#                           off-map land-slope tangent matches finite differences of the residual.
#
# (Picard is intentionally not asserted here: its cold-start non-convergence and its ~0.3 m fixed-point
#  offset from Anderson are PRE-EXISTING and reproduce flag-off on an ocean-ringed domain -- see
#  benchmark/BOUNDARY_CONDITIONS.md. This test isolates the BOUNDARY, not Picard's solver robustness.)
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
NPROCS="${2:-4}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/ghostbc_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/ghostbc_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-3}"        # metres; cross-scheme + MPI agreement under the ghost boundary
JTOL="${JTOL:-1e-2}"      # Newton ||J-Jfd||/||J|| ceiling (smooth-T tangent; piecewise kink keeps it >1e-8)
PY="${PY:-python3}"
MPIRUN="${MPIRUN:-mpirun}"
export OMP_NUM_THREADS=1

emit() { # stem cycles
  cat > "$WORK/$1.cfg" <<EOF
run_type transient
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1
southern_edge 0
deltat 2419200
total_cycles ${2}
cycles_to_save ${2}
maxiter 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region ghostbc
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

GB=""  # mask-aware ghost boundary is now the default (no flag needed)
BASE="-snes_stol 1e-8"
fail=0

# ---- 1. MPI determinism (cc, ghost boundary): 1 rank vs N ranks -------------------------------------
emit cc_n1 120
emit cc_nN 120
"$WTM" "$WORK/cc_n1.cfg" -wtm_anderson $GB $BASE > "$WORK/cc_n1.log" 2>&1 \
  || { echo "RUN FAILED: cc n=1"; tail -3 "$WORK/cc_n1.log"; exit 2; }
"$MPIRUN" -n "$NPROCS" "$WTM" "$WORK/cc_nN.cfg" -wtm_anderson $GB $BASE > "$WORK/cc_nN.log" 2>&1 \
  || { echo "RUN FAILED: cc n=$NPROCS"; tail -3 "$WORK/cc_nN.log"; exit 2; }

# ---- 2. Cross-scheme agreement (all serial, ghost boundary) -----------------------------------------
declare -A FLAG=( [cc]="-wtm_anderson" [tr]="-wtm_anderson -wtm_tr_bdf2" [bdf2v]="-wtm_anderson -wtm_bdf2_on_V" [newton]="-wtm_newton" )
for s in tr bdf2v newton; do
  emit "$s" 120
  "$WTM" "$WORK/$s.cfg" ${FLAG[$s]} $GB $BASE > "$WORK/$s.log" 2>&1 \
    || { echo "RUN FAILED: $s"; tail -3 "$WORK/$s.log"; exit 2; }
done

CC1=$(ls "$WORK"/cc_n1_*.tif | tail -1)
CCN=$(ls "$WORK"/cc_nN_*.tif | tail -1)
TRF=$(ls "$WORK"/tr_*.tif | tail -1)
BVF=$(ls "$WORK"/bdf2v_*.tif | tail -1)
NWF=$(ls "$WORK"/newton_*.tif | tail -1)

TOL="$TOL" NPROCS="$NPROCS" "$PY" - "$CC1" "$CCN" "$TRF" "$BVF" "$NWF" <<'PY'
import sys, os, numpy as np, rasterio
cc1, ccn, tr, bv, nw = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:6]]
m = np.ones_like(cc1, bool); m[:, 0] = False   # interior + land edges; exclude the ocean column
def mx(a, b): return float(np.max(np.abs((a - b)[m])))
tol = float(os.environ["TOL"]); n = os.environ["NPROCS"]
d_mpi = mx(cc1, ccn); d_tr = mx(cc1, tr); d_bv = mx(cc1, bv); d_nw = mx(cc1, nw)
print(f"  cc steady wtd: min {cc1[m].min():.3f} max {cc1[m].max():.3f} m (land, incl. edges)")
print(f"  1. MPI determinism  cc n=1 vs n={n}: max|d| = {d_mpi:.2e} m")
print(f"  2. cross-scheme vs cc:  tr={d_tr:.2e}  bdf2v={d_bv:.2e}  newton={d_nw:.2e} m")
ok = (d_mpi <= 1e-9) and max(d_tr, d_bv, d_nw) <= tol
print("PASS" if ok else "FAIL", "(cross-scheme / MPI agreement under the ghost boundary)")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && fail=1

# ---- 3. Newton Jacobian FD check (ghost boundary ON, smooth T so the tangent is exact) ---------------
emit jac 1
JR=$("$WTM" "$WORK/jac.cfg" -wtm_newton $GB \
        -wtm_ksat_surface_smoothing_width 0.5 -wtm_ksat_soilbottom_smoothing_width 0.5 \
        -snes_test_jacobian -snes_max_it 1 2>&1 \
     | grep -oE '\|\|J - Jfd\|\|_F/\|\|J\|\|_F = [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' | sort -g | tail -1)
if [ -z "$JR" ]; then
  echo "  3. Newton Jacobian FD: FAIL (no ratio produced)"; fail=1
else
  echo "  3. Newton Jacobian FD (ghost ON): max ||J-Jfd||/||J|| = $JR  (ceiling $JTOL)"
  awk -v r="$JR" -v t="$JTOL" 'BEGIN{exit !(r+0 <= t+0)}' \
    && echo "  PASS (off-map land-slope tangent matches finite differences)" \
    || { echo "  FAIL (Jacobian off-map tangent inconsistent)"; fail=1; }
fi

echo
[ $fail -eq 0 ] && echo "GHOST-BOUNDARY CHECKS PASSED" || echo "GHOST-BOUNDARY CHECKS FAILED" >&2
exit $fail
