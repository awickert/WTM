#!/usr/bin/env bash
# NESTED-depression fixture: exercises (a) the FSM fullness walk on a real metadepression HIERARCHY, and
# (b) the #119 lake-aware-skim EQUILIBRIUM-ACCURACY check against a KNOWN spill elevation.
#
# Topography (make_inputs.py): two 90 m leaf pits inside a 95 m basin (-> a metadepression when they merge),
# ringed by a 100 m plateau, with a 97 m outlet notch to the ocean. Under net-positive forcing the basin
# fills and spills at its 97 m sill. Asserts, on the Anderson path with FSM every step:
#   HIERARCHY   : the fullness walk reports 3 depressions (2 leaves + 1 metadepression) -- a real hierarchy,
#                 not a single leaf.
#   SPILL LEVEL : with the lake-aware skim the basin fills to EXACTLY its 97 m outlet sill (|surf-97| < 0.2 m).
#                 A one-step-lag under-fill would land the stage below the sill -- this is the equilibrium-
#                 accuracy check for #119. (Pre-delivery-fix the skim drained the basin to 0 ponded: bites.)
#   SKIM == PLAIN : the skim and the plain collector reach the same spill stage (the skim is not draining or
#                 over-filling the lake).
#
# Usage:  tests/fsm_fullness/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/fsm_fullness_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work ff
PY="${PY:-python3}"; export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/fsm_fullness/config.yaml. Every setting the run resolves to is
# stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
# surface_water.routing: continuous is stated as the PRECONDITION it is -- with no routing there are
# no lakes, nothing spills, and every assertion here is about where spilled water ends up.
emit() { # $1 stem -- ONE arm, run at several rank counts; the rank count is not a config setting
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" config.yaml > "$WORK/$1.yaml"
}
run() { emit "$1"; "$WTM" "$WORK/$1.yaml" $2 > "$WORK/$1.err" 2>&1 \
        || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.err"; exit 2; }; }
run plain ""
run skim  ""
# MPI consistency of the skim: n=4 must match n=1 byte-for-byte. The skim reads per-cell starting_wtd and
# hands its captured water to a rank-0 FillSpillMerge, so this exercises the gather/scatter round-trip.
emit skim4
mpirun -n 4 "$WTM" "$WORK/skim4.yaml" \
    -da_processors_x 2 -da_processors_y 2 > "$WORK/skim4.err" 2>&1 \
    || { echo "RUN FAILED: skim4"; tail -3 "$WORK/skim4.err"; exit 2; }

NDEP=$(grep "FSM fullness" "$WORK/skim.err" | tail -1 | sed 's#.*/ \([0-9]*\) depressions.*#\1#')
SP=$(ls "$WORK"/plain_*.tif | tail -1); SK=$(ls "$WORK"/skim_*.tif | tail -1); SK4=$(ls "$WORK"/skim4_*.tif | tail -1)
# PROMOTED FROM LITERALS (#121): reachable from outside so assertion_probe can tighten each
# and confirm the assertion still fails when it should.
# THE SILL ELEVATIONS (97 m, 95 m) STAY LITERAL -- they are the fixture's geometry, not tuning
# knobs. What is promoted is the TOLERANCE on the distance from them.
SILL_TOL="${SILL_TOL:-0.2}"   # |stage - 97 m sill|, and |skim - plain|
MPI_TOL="${MPI_TOL:-1e-9}"   # n=1 vs n=4 water table
NDEP="$NDEP" SILL_TOL="$SILL_TOL" MPI_TOL="$MPI_TOL" "$PY" - "$INP/fsm_fullness_t0_topography.tif" "$SP" "$SK" "$SK4" <<'PY'
import sys, os, numpy as np, rasterio
sill_tol = float(os.environ["SILL_TOL"]); mpi_tol = float(os.environ["MPI_TOL"])
topo, wp, wk, wk4 = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
def surfmax(w):
    p = w > 1e-6
    return float((topo + w)[p].max()) if p.any() else -999.0
ndep = int(os.environ["NDEP"]); sk = surfmax(wk); pl = surfmax(wp)
ok = True
def check(name, cond, detail):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok = ok and cond
check("HIERARCHY (metadepression, not a lone leaf)", ndep >= 3,
      f"fullness walk reports {ndep} depressions (2 leaf pits + metadepression)")
# UNITS: NOT CONVERTED TO WATER VOLUME, AND MUST NOT BE (#65). Same reason as fsm_cascade: these are
# LAKE SURFACE ELEVATIONS against a 97 m topographic sill, and SKIM == PLAIN compares two such
# elevations to each other. HIERARCHY is a COUNT of depressions. The MPI check is an IDENTITY.
# Nothing here is a water-depth measurement, so there is nothing to convert.
check("SPILL LEVEL (skim fills to the 97 m sill)", abs(sk - 97.0) < sill_tol,
      f"skim lake surface = {sk:.3f} m (known outlet sill = 97.0)")
check("SKIM == PLAIN (skim neither drains nor over-fills)", abs(sk - pl) < sill_tol,
      f"skim {sk:.3f} vs plain {pl:.3f} m")
mpi = float(np.abs(wk - wk4).max())
check("SKIM MPI-CONSISTENT (n=1 == n=4)", mpi < mpi_tol,
      f"max|Δwtd| n1 vs n4 = {mpi:.3e} m")
print("PASS: nested hierarchy walked; lake-aware skim reaches the correct spill equilibrium"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
