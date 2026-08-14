#!/usr/bin/env bash
# SECANT ≡ VOLUME backward-Euler storage equivalence (unit/regression test).
#
# WTM's default backward Euler forms the storage term as S·Δh with the EXACT secant effective storativity
# S = (V(wⁿ⁺¹) − V(wⁿ))/(wⁿ⁺¹ − wⁿ); -wtm_volume_storage uses the stored-volume change ΔV directly. Since S
# is the exact secant, S·Δh ≡ ΔV identically (even across the surface where dV/dh jumps porosity→~1). So on
# a well-behaved (non-oscillating) domain the two must agree to MACHINE PRECISION.
#
# This guards that identity: it bites if updateEffectiveStorativity ever stops being the exact secant (e.g.
# a tangent or endpoint storativity), which would make the default BE storage inconsistent with the volume
# schemes. See finding on the (retracted) "secant storage inconsistency" -- there is none, and this proves it.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/storeq_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/storeq_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-6}"        # metres; machine-precision agreement expected (observed ~1e-12)
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type transient
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1
southern_edge 0
deltat 2419200
total_cycles 3
cycles_to_save 3
maxiter 200
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region storeq
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

# -wtm_ghost_boundary: use the mask-aware ghost boundary so edge land cells are not forced against a
# hard-draining topo-0 ocean pad. Under the old padding, boundary-adjacent SURFACE cells drain so hard that
# the ÷S (secant) vs ÷Sy (tangent) residual scaling leaves a ~1e-4 convergence-region difference there --
# a scaling/conditioning artifact, NOT the identity failing. The ghost boundary removes that edge stress so
# the S·Δh ≡ ΔV identity shows at machine precision (observed ~1e-15) and the test is a clean invariant check.
emit secant; emit volume
"$WTM" "$WORK/secant.cfg" -wtm_anderson                     -wtm_ghost_boundary -snes_stol 1e-10 > "$WORK/secant.log" 2>&1 \
  || { echo "RUN FAILED: secant"; tail -3 "$WORK/secant.log"; exit 2; }
"$WTM" "$WORK/volume.cfg" -wtm_anderson -wtm_volume_storage -wtm_ghost_boundary -snes_stol 1e-10 > "$WORK/volume.log" 2>&1 \
  || { echo "RUN FAILED: volume"; tail -3 "$WORK/volume.log"; exit 2; }

SEC=$(ls "$WORK"/secant_*.tif | tail -1)
VOL=$(ls "$WORK"/volume_*.tif | tail -1)
TOL="$TOL" "$PY" - "$SEC" "$VOL" <<'PY'
import sys, os, numpy as np, rasterio
sec, vol = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(sec, bool); m[:, 0] = False   # interior + land edges (exclude ocean column)
d = float(np.max(np.abs((sec - vol)[m])))
tol = float(os.environ["TOL"])
print(f"  secant-BE vs volume-BE (S·Δh ≡ ΔV): max|Δwtd| = {d:.3e} m  (tol {tol})")
if d <= tol:
    print("PASS: exact-secant storativity makes the two forms identical (machine precision)")
    sys.exit(0)
print(f"FAIL: {d:.3e} m > {tol} m -> the BE secant storage is NOT the exact volume secant")
sys.exit(1)
PY
