#!/usr/bin/env bash
# Mask-aware ghost-boundary test (now the default; task #96). On a domain with REAL LAND at the N/S/E
# edges and ocean at the west (see make_inputs.py), verify that the off-map land-slope Neumann ghost is
# implemented consistently across every solver assembly site:
#
#   1. MPI determinism   -- cc (Anderson) with the ghost boundary agrees on 1 vs N ranks to the solver's
#                           own water tolerance (exercises the off-map reads under domain decomposition).
#                           NOT bit-for-bit: that was claimed here, and was never true -- see the measured
#                           tolerance-scaling note beside the assertion.
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
TOL="${TOL:-2.5e-4}"      # STEADY-STATE cross-scheme agreement under the ghost boundary.
                          # ITS OLD JUSTIFICATION WAS THE BUG (#34). It read: "NOT a cross-scheme bound:
                          # at the fixed point the schemes agree to 0.00e+00 by construction, so this
                          # tolerance is never the binding constraint". The 0.00e+00 was not "by
                          # construction" -- every compared field was identically zero, because the wedge
                          # had saturated and active_set pinned it. The agreement was read as a strong
                          # result and written down as a rationale, which is how it survived a review.
                          # Live values on the repaired fixture, in water: tr 1.34e-08, bdf2v ~1.1e-08,
                          # newton 2.80e-07 -- so TOL now sits ~900x above the largest real signal.
JTOL="${JTOL:-1e-2}"      # Newton ||J-Jfd||/||J|| ceiling (smooth-T tangent; piecewise kink keeps it >1e-8)
PY="${PY:-python3}"
MPIRUN="${MPIRUN:-mpirun}"
export OMP_NUM_THREADS=1

# THE CONFIGS ARE FILES NOW (#83): config.yaml (mode: adaptive, the anderson arms) and
# config_fixed.yaml (mode: fixed, the newton arms). TWO files because the two modes resolve different
# key sets -- adaptive carries grow/shrink/norm, fixed carries no controller dials at all -- and a
# config must state what its run resolves to rather than a superset.
emit_adaptive() { # $1 stem  $2 time_integration (REQUIRED: it is what distinguishes these arms)
    local stem="${1:?emit_adaptive needs a stem}"
    local integ="${2:?emit_adaptive needs a time_integration: it is the only thing that differs between
                      these arms, so name it rather than inherit it}"
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" -e "s|@INTEG@|$integ|g" \
        config.yaml > "$WORK/$stem.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$stem.yaml" && { echo "ERROR: unfilled slot in $stem.yaml"; exit 1; }
    return 0
}

emit_fixed() { # $1 stem  $2 cycles  $3 max_iterations  $4 ksat smoothing width
    local stem="${1:?}" cyc="${2:?}" maxit="${3:?}" ksm="${4:?emit_fixed needs a smoothing width}"
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" \
        -e "s|@TOTAL@|$(( cyc * 50 * 2419200 ))s|g" -e "s|@SAVE@|$cyc|g" \
        -e "s|@MAXIT@|$maxit|g" -e "s|@KSMOOTH@|$ksm|g" \
        config_fixed.yaml > "$WORK/$stem.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$stem.yaml" && { echo "ERROR: unfilled slot in $stem.yaml"; exit 1; }
    return 0
}

GB=""  # mask-aware ghost boundary is now the default (no flag needed)
BASE=""  # solver.tolerance is now a CONFIG key (snes_stol in the shim), not a CLI flag
fail=0

# ---- 1. MPI determinism (cc, ghost boundary): 1 rank vs N ranks -------------------------------------
# cc's integrator is backward-euler (see the INTEG note in section 2, which is where it is decided).
# Named literally here because INTEG is declared below; the two must not drift apart.
emit_adaptive cc_n1 backward-euler
emit_adaptive cc_nN backward-euler
"$WTM" "$WORK/cc_n1.yaml" $GB $BASE > "$WORK/cc_n1.log" 2>&1 \
  || { echo "RUN FAILED: cc n=1"; tail -3 "$WORK/cc_n1.log"; exit 2; }
"$MPIRUN" -n "$NPROCS" "$WTM" "$WORK/cc_nN.yaml" $GB $BASE > "$WORK/cc_nN.log" 2>&1 \
  || { echo "RUN FAILED: cc n=$NPROCS"; tail -3 "$WORK/cc_nN.log"; exit 2; }

# ---- 2. Cross-scheme agreement (all serial, ghost boundary) -----------------------------------------
declare -A FLAG=( [cc]="" [tr]="" [bdf2v]="" [newton]="" )
# newton is config-owned; it was a BARE flag here, i.e. PLAIN Newton, so continuation is declined.
# THE INTEGRATOR IS NAMED, NOT LEFT ABSENT -- and `cc` names tr-bdf2, which is WHAT IT ACTUALLY RUNS.
# It used to leave the key unset and take whatever `anderson` resolved to; writing that value down
# changes nothing about the run, which is the point of materialising a config.
#
# cc IS backward-euler NOW, AND THAT IS A DELIBERATE CHANGE OF WHAT THIS SUITE MEASURES (#34/#96).
# It was tr-bdf2, which made it a BYTE-IDENTICAL COPY of the tr arm -- so `tr=0.00e+00` in check 2 was
# a field compared against itself, and one of four arms measured nothing.
#
# The previous note here declined to change it, on the grounds that the arm's history was `-wtm_anderson`
# with no integrator flag ("whatever the default is", and tr-bdf2 IS the default), so pinning
# backward-euler would ASSERT a new intent rather than restore an old one. That reasoning was right to
# refuse a QUIET change, and this is not one. The decisive argument is the header three lines up: check 2
# is stated as "cc / TR-BDF2 / BDF2-on-V / Newton ... must converge to the SAME water table" -- FOUR
# schemes. With cc = tr-bdf2 there were only three, so the duplicate contradicted the suite's own claim.
# Naming backward-euler makes the suite do what it says.
#
# Verified it earns its place rather than just being different: backward-euler converges on this fixture
# to the same steady wtd range (-1.9059 .. -0.7482) and sits 1.335e-08 m of water from tr-bdf2 -- a real,
# nonzero measurement, four orders inside TOL.
declare -A INTEG=([cc]="backward-euler" [tr]="tr-bdf2" [bdf2v]="bdf2")
for s in tr bdf2v newton; do
  if [ "$s" = newton ]; then emit_fixed "$s" 120 10000 0; else emit_adaptive "$s" "${INTEG[$s]}"; fi
  "$WTM" "$WORK/$s.yaml" ${FLAG[$s]} $GB $BASE > "$WORK/$s.log" 2>&1 \
    || { echo "RUN FAILED: $s"; tail -3 "$WORK/$s.log"; exit 2; }
done

CC1=$(ls "$WORK"/cc_n1_*.tif | tail -1)
CCN=$(ls "$WORK"/cc_nN_*.tif | tail -1)
TRF=$(ls "$WORK"/tr_*.tif | tail -1)
BVF=$(ls "$WORK"/bdf2v_*.tif | tail -1)
NWF=$(ls "$WORK"/newton_*.tif | tail -1)

TOL="$TOL" NPROCS="$NPROCS" PHI="$INP/ghostbc_porosity.tif" TESTS="$(readlink -f ..)" \
  WATER_TOL="$(awk -F: '/water_volume_tol:/{v=$2; sub(/^[ \t]+/,"",v); sub(/[ \t].*$/,"",v); print v; exit}' config.yaml)" \
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
print(f"  2. steady-state agreement vs cc:  tr={d_tr:.2e}  bdf2v={d_bv:.2e}  newton={d_nw:.2e} m")
# THE MPI BOUND IS THE SOLVER TOLERANCE, NOT 1e-9, AND IT IS MEASURED (#34, #84's class).
# It was `d_mpi <= 1e-9` with a header claiming "bit-for-bit identical on 1 vs N ranks". That claim was
# never tested: while this suite's field was identically zero, d_mpi was 0.00e+00 for free. On a live
# field it is not bit-identical, and it should not be expected to be -- a different domain decomposition
# sums the Anderson reductions in a different floating-point order.
#
# What settles it as NOISE rather than a ghost-cell error under decomposition: d_mpi tracks the solver
# tolerance instead of plateauing. Measured, same fixture, sweeping solver.tolerance and water_volume_tol
# together:
#     tr-bdf2         tol 1e-08 -> 7.341e-09 (0.73x)   1e-10 -> 6.631e-11 (0.66x)   1e-12 -> 9.131e-13 (0.91x)
#     backward-euler  tol 1e-08 -> 2.083e-08 (2.08x)   1e-10 -> 8.796e-11 (0.88x)   1e-12 -> 1.321e-12 (1.32x)
# Four orders of magnitude, two schemes, ratio always O(1) and never plateauing. A real decomposition
# error would stop shrinking as the tolerance tightened. This does.
#
# THE FACTOR OF 5 IS A CHOICE, and it is mine rather than something the model dictates: the six
# measurements above span 0.66x to 2.08x, so 5x clears the worst by ~2.4x. It is still 5e-08 on the
# shipped tolerance -- four orders tighter than TOL, and far tighter than any real off-map ghost error
# could hide under. Raise it only with a measurement, never to make a red test green.
MPI_TOL_FACTOR = 5.0
mpi_tol = MPI_TOL_FACTOR * float(os.environ["WATER_TOL"])   # config's solver.convergence.water_volume_tol
print(f"     (MPI bound = {MPI_TOL_FACTOR:g}x the run's own solver water tolerance = {mpi_tol:g} m; see run.sh)")
ok = (d_mpi <= mpi_tol) and max(d_tr, d_bv, d_nw) <= tol
print("PASS" if ok else "FAIL", "(steady-state / MPI agreement under the ghost boundary)")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && fail=1

# ---- 3. Newton Jacobian FD check (ghost boundary ON, smooth T so the tangent is exact) ---------------
# PLAIN Newton: the FD check wants the raw Jacobian. Its three WTM settings are stated in the CONFIG --
# one iteration because this is a derivative check and not a solve, and the two ksat smoothing widths
# because a smooth T is what makes the analytic tangent exact. They used to be CLI flags, so this arm's
# config claimed the defaults (max_iterations 10000, smoothing 0) while the run used 1 and 0.5 -- a
# config that stated three values its own run did not use. -snes_test_jacobian stays on the command
# line: it is a PETSc diagnostic, not a WTM setting, and PETSc's flags keep their CLI surface.
emit_fixed jac 1 1 0.5
JR=$("$WTM" "$WORK/jac.yaml" $GB -snes_test_jacobian 2>&1 \
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
