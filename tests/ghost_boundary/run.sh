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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
NPROCS="${2:-4}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/ghostbc_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work ghostbc
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head: the model conserves water
# and judges every stopping criterion in water volume (#61/#65). Uniform phi = 0.25 on this fixture, so this
# is the old 1e-3 head bound x0.25 exactly -- the same strictness, correctly labelled.
TOL="${TOL:-2.5e-4}"      # cross-scheme + MPI agreement under the ghost boundary
JTOL="${JTOL:-1e-2}"      # Newton ||J-Jfd||/||J|| ceiling (smooth-T tangent; piecewise kink keeps it >1e-8)
PY="${PY:-python3}"
MPIRUN="${MPIRUN:-mpirun}"
export OMP_NUM_THREADS=1

emit() { # stem cycles   [env: METHOD=, DTC=]
  ../emit_config.sh > "$WORK/$1.yaml" <<EOF
${METHOD:+solver_method $METHOD}
${INTEG:+time_integration $INTEG}
${DTC:+dt_continuation $DTC}
run_type transient
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
deltat 2419200
total_time $(( ${2} * 50 * 2419200 ))s
save_nreport_interval ${2}
report_interval 50
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
METHOD=anderson emit cc_n1 120
METHOD=anderson emit cc_nN 120
"$WTM" "$WORK/cc_n1.yaml" $GB $BASE > "$WORK/cc_n1.log" 2>&1 \
  || { echo "RUN FAILED: cc n=1"; tail -3 "$WORK/cc_n1.log"; exit 2; }
"$MPIRUN" -n "$NPROCS" "$WTM" "$WORK/cc_nN.yaml" $GB $BASE > "$WORK/cc_nN.log" 2>&1 \
  || { echo "RUN FAILED: cc n=$NPROCS"; tail -3 "$WORK/cc_nN.log"; exit 2; }

# ---- 2. Cross-scheme agreement (all serial, ghost boundary) -----------------------------------------
declare -A FLAG=( [cc]="" [tr]="" [bdf2v]="" [newton]="" )
# newton is config-owned; it was a BARE flag here, i.e. PLAIN Newton, so continuation is declined
declare -A CFG=(  [cc]="anderson" [tr]="anderson" [bdf2v]="anderson" [newton]="newton" )
declare -A INTEG=([cc]="" [tr]="tr-bdf2" [bdf2v]="bdf2" [newton]="")
for s in tr bdf2v newton; do
  METHOD="${CFG[$s]}" INTEG="${INTEG[$s]}" DTC=$([ "${CFG[$s]}" = newton ] && echo false) emit "$s" 120
  "$WTM" "$WORK/$s.yaml" ${FLAG[$s]} $GB $BASE > "$WORK/$s.log" 2>&1 \
    || { echo "RUN FAILED: $s"; tail -3 "$WORK/$s.log"; exit 2; }
done

CC1=$(ls "$WORK"/cc_n1_*.tif | tail -1)
CCN=$(ls "$WORK"/cc_nN_*.tif | tail -1)
TRF=$(ls "$WORK"/tr_*.tif | tail -1)
BVF=$(ls "$WORK"/bdf2v_*.tif | tail -1)
NWF=$(ls "$WORK"/newton_*.tif | tail -1)

TOL="$TOL" NPROCS="$NPROCS" PHI="$INP/ghostbc_porosity.tif" TESTS="$(readlink -f ..)" \
  "$PY" - "$CC1" "$CCN" "$TRF" "$BVF" "$NWF" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

cc1, ccn, tr, bv, nw = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:6]]
m = np.ones_like(cc1, bool); m[:, 0] = False   # interior + land edges; exclude the ocean column
phi = VOL.read_band(os.environ["PHI"])
def mx(a, b): return float(VOL.volume_diff(a, b, phi)[m].max())   # WATER VOLUME, not head
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
METHOD=newton DTC=false emit jac 1   # PLAIN Newton: the FD check wants the raw Jacobian
JR=$("$WTM" "$WORK/jac.yaml" $GB \
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
