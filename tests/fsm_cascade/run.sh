#!/usr/bin/env bash
# CASCADE fixture: surface water flows depression -> depression -> off map, with the lake-aware skim.
#
# pit A (floor 94, outlet sill 97) spills DOWN into basin B (floor 88, outlet sill 95), which spills OFF-MAP
# to the ocean. At steady state both fill to their sills, so this checks that the skim + FSM route a spill
# CHAIN correctly (A->B->ocean) and reach the known sill elevations, MPI-consistently and conserving mass.
# (The separate "one full, one not" heterogeneous state uses its own fixture; see #123.)
#
# Asserts, Anderson path + FSM every step + active-set skim:
#   CHAIN LEVELS : pit A fills to 97.0 m (its A->B sill) and basin B fills to 95.0 m (its off-map sill).
#   CONSERVATION : per-cycle water balance closes (|Δbudget_residual|/Δrecharge < 1e-4).
#   MPI CONSISTENT: n=1 == n=4 (byte-identical wtd).
#
# Usage:  tests/fsm_cascade/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/fsm_cascade_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work casc
PY="${PY:-python3}"; export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/fsm_cascade/config.yaml. Every setting the run resolves to is
# stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
# surface_water.routing: continuous is stated as the PRECONDITION it is -- with no routing there are
# no lakes, nothing spills, and every assertion here is about where spilled water ends up.
emit() { # $1 stem -- ONE arm, run at several rank counts; the rank count is not a config setting
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" config.yaml > "$WORK/$1.yaml"
  apply_test_iterations "$WORK/$1.yaml"
}
emit skim
"$WTM" "$WORK/skim.yaml" > "$WORK/skim.err" 2>&1 \
  || { echo "RUN FAILED: skim"; tail -3 "$WORK/skim.err"; exit 2; }
emit skim4
mpirun -n 4 "$WTM" "$WORK/skim4.yaml" \
    -da_processors_x 2 -da_processors_y 2 > "$WORK/skim4.err" 2>&1 \
  || { echo "RUN FAILED: skim4"; tail -3 "$WORK/skim4.err"; exit 2; }

SK=$(ls "$WORK"/skim_*.tif | tail -1); SK4=$(ls "$WORK"/skim4_*.tif | tail -1)
# PROMOTED FROM LITERALS (#121): reachable from outside so assertion_probe can tighten each
# and confirm the assertion still fails when it should.
# THE SILL ELEVATIONS (97 m, 95 m) STAY LITERAL -- they are the fixture's geometry, not tuning
# knobs. What is promoted is the TOLERANCE on the distance from them.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, ONE-SIDED with a GEOMETRIC scale: measured max|stage - sill| = 0.000 m
#   exactly -- both lakes land on their sills to the printed precision (pit A 97.000 vs 97, basin B
#   95.000 vs 95). The bound is sized by the fixture, not by the measurement: the two sills in this
#   chain are 2 m apart, so 0.2 m is 10x below the smallest distance that could let a lake settle
#   on the WRONG sill and still pass.
SILL_TOL="${SILL_TOL:-0.2}"   # |stage - sill| for each lake in the chain
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED and the BLUNTEST bound in the tree: measured max
#   |dbudget_residual|/drecharge = 3.047e-13 against a bound of 1e-4, i.e. 3.3e8 of headroom. It
#   would not notice per-cycle closure degrading by EIGHT orders. Recorded, not retightened --
#   changing it changes what the suite accepts, which is a decision and not a documentation fix.
CONS_TOL="${CONS_TOL:-1e-4}"   # per-cycle budget closure, relative
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED: measured max|dwtd| between n=1 and n=4 = 0.000e+00 m -- the
#   decomposition is bit-identical, as a correct halo exchange requires. Stated as a tolerance
#   rather than `== 0` so the assertion does not rest on float equality; at the O(100 m) scale of
#   this field, 1e-9 m is ~1e-11 relative, close to what double precision can even represent.
MPI_TOL="${MPI_TOL:-1e-9}"   # n=1 vs n=4 water table
SILL_TOL="$SILL_TOL" CONS_TOL="$CONS_TOL" MPI_TOL="$MPI_TOL" "$PY" - "$INP/fsm_cascade_t0_topography.tif" "$SK" "$SK4" "$WORK/skim.txt" <<'PY'
import sys, os, numpy as np, rasterio
sill_tol = float(os.environ["SILL_TOL"]); cons_tol = float(os.environ["CONS_TOL"])
mpi_tol = float(os.environ["MPI_TOL"])
topo, wk, wk4 = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
txt = sys.argv[4]
def surf(region):
    w = region_of(wk, region); p = w > 1e-6
    return float((region_of(topo, region) + w)[p].max()) if p.any() else -999.0
def region_of(a, r):
    return a[r]
A = (slice(4, 12), slice(11, 20))    # pit A
B = (slice(15, 27), slice(4, 26))    # basin B
sA, sB = surf(A), surf(B)
mpi = float(np.abs(wk - wk4).max())
rows = [l.split() for l in open(txt) if l and l[0].isdigit()]
R = np.array([float(r[8]) for r in rows]); res = np.array([float(r[15]) for r in rows])
rel = (np.abs(np.diff(res))[-4:] / np.where(np.abs(np.diff(R))[-4:] > 0, np.abs(np.diff(R))[-4:], 1)).max()
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
# UNITS: NOT CONVERTED TO WATER VOLUME, AND MUST NOT BE (#65). `surf()` returns (topo + w).max() --
# a LAKE SURFACE ELEVATION ABOVE DATUM -- and the assertion compares it against the fixture's SILL
# ELEVATIONS, 97 m and 95 m. Those are topographic heights, not water depths. "The lake filled to its
# sill" cannot be expressed as a volume without the basin hypsometry, and a volume bound here would be
# measuring a different claim. The other two checks need no conversion either: `rel` is already
# dimensionless (a budget ratio), and the MPI check is an IDENTITY (n=1 == n=4), which is unit-agnostic.
check("CHAIN LEVELS (A->97 sill, B->95 sill)", abs(sA - 97.0) < sill_tol and abs(sB - 95.0) < sill_tol,
      f"max|stage - sill| = {max(abs(sA - 97.0), abs(sB - 95.0)):.3f} m (tol SILL_TOL={sill_tol})"
      f" -- pit A {sA:.3f} m vs sill 97, basin B {sB:.3f} m vs sill 95")
check("CONSERVATION (per-cycle balance closes)", rel < cons_tol,
      f"max |Δbudget_residual|/Δrecharge = {rel:.3e} (tol CONS_TOL={cons_tol})")
check("MPI CONSISTENT (n=1 == n=4)", mpi < mpi_tol, f"max|Δwtd| = {mpi:.3e} m (tol MPI_TOL={mpi_tol})")
print("PASS: cascade A->B->ocean routes to the correct sills, conserving and MPI-consistent"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
