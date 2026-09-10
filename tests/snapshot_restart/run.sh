#!/usr/bin/env bash
# Snapshot-filename + restart-from-snapshot regression.
#   FILENAME: output rasters are {prefix}{cycle:09}_{year}yr.tif. With deltat = 1 yr and report_interval = 1, the
#             simulated year MUST equal the cycle, so we assert e.g. cycle 5 -> ..._000000005_5yr.tif exists.
#   RESTART:  a warm restart from a mid-run snapshot (supplied_wt 1, starting_wt = that snapshot) must reach
#             the SAME equilibrium as a cold run AND get there in FEWER cycles -- which only happens if
#             supplied_wt actually loaded the snapshot (a broken load would silently restart cold and take
#             the same number of cycles).
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/snaptest_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work snap
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head: the model conserves water
# and judges every stopping criterion in water volume (#61/#65). Uniform phi = 0.25 on this fixture, so this
# is the old 0.05 head bound x0.25 exactly -- the same strictness, correctly labelled.
TOL="${TOL:-0.0125}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/snapshot_restart/config.yaml. Every setting the run resolves
# to is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# solver.time_step.mode: fixed is PINNED in that file, and it is load-bearing: this test asserts the
# output FILENAME encodes year == cycle, which holds only while dt is a fixed 1 yr with
# report_interval 1. An adaptive controller breaks that identity outright.
emit() { # $1 stem, $2 input dir, $3 initial_water_table, $4 equilibrium_stop.tol   (ALL REQUIRED)
  local dir="${2:?emit needs an input dir}"
  local iwt="${3:?emit needs an initial_water_table: saturated|supplied -- naming it IS the restart arm}"
  local et="${4:?emit needs an equilibrium_stop.tol}"
  sed -e "s|@INPUTS@|$dir|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  initial_water_table: saturated|  initial_water_table: $iwt|" \
      -e "s|^    tol: 0.001|    tol: $et|" config.yaml > "$WORK/$1.yaml"
}
stop_cycle() { grep -oE "stopping at cycle [0-9]+" "$1" | grep -oE "[0-9]+$"; }
BB=""

# --- cold full run (saves every cycle) ---
emit cold "$INP" saturated 0.001
"$WTM" "$WORK/cold.yaml" $BB > "$WORK/cold.log" 2>&1 || { echo "RUN FAILED: cold"; tail -3 "$WORK/cold.log"; exit 2; }
C_COLD=$(stop_cycle "$WORK/cold.log")

# --- (1) FILENAME format: year == cycle (deltat 1 yr, report_interval 1) ---
# Its own SHORT run with the equilibrium auto-stop DISABLED, so the cycle numbers exist regardless of
# how fast the model converges. Previously this checked cycles {1,3,5} of the cold equilibrium run,
# which broke when the default collector became active_set: that converges ~3x faster (2 cycles here
# vs 7 under implicit), so cycles 3 and 5 no longer existed. A filename-format assertion should not
# depend on convergence speed.
emit fname "$INP" saturated 0
sed -i "s#^  total:.*#  total: '6yr'#" "$WORK/fname.yaml"
"$WTM" "$WORK/fname.yaml" > "$WORK/fname.log" 2>&1 \
  || { echo "RUN FAILED: fname"; tail -3 "$WORK/fname.log"; exit 2; }
for k in 1 3 5; do
  f=$(printf "%s/fname_%09d_%dyr.tif" "$WORK" "$k" "$k")
  [[ -f "$f" ]] || { echo "FAIL: expected snapshot $(basename "$f") not found (filename year != cycle?)"; ls "$WORK"/fname_*.tif | sed 's#.*/##' | head; exit 1; }
done
for k in ; do
  f=$(printf "%s/cold_%09d_%dyr.tif" "$WORK" "$k" "$k")
  [[ -f "$f" ]] || { echo "FAIL: expected snapshot $(basename "$f") not found (filename year != cycle?)"; ls "$WORK"/cold_*.tif | sed 's#.*/##' | head; exit 1; }
done
echo "  filename format OK: {prefix}{cycle:09}_{year}yr.tif, year==cycle"

# --- (2) RESTART from a mid snapshot (clearly pre-equilibrium) ---
# Derived from the cold run's own length, not hardcoded. A fixed MID=4 broke when the default
# collector became active_set and the cold run began converging in 2 cycles instead of 7 -- cycle 4
# no longer existed. Half-way (min 1) is pre-equilibrium by construction at any convergence speed.
MID=$(( C_COLD / 2 )); [[ "$MID" -ge 1 ]] || MID=1
SNAP=$(printf "%s/cold_%09d_%dyr.tif" "$WORK" "$MID" "$MID")
[[ -f "$SNAP" ]] || { echo "FAIL: mid snapshot $(basename "$SNAP") missing"; exit 1; }
mkdir -p "$WORK/rinp"; cp "$INP"/*.tif "$WORK/rinp/"; cp "$SNAP" "$WORK/rinp/snaptest_ta_starting_wt.tif"
emit restart "$WORK/rinp" supplied 0.001
"$WTM" "$WORK/restart.yaml" $BB > "$WORK/restart.log" 2>&1 || { echo "RUN FAILED: restart"; tail -3 "$WORK/restart.log"; exit 2; }
C_RST=$(stop_cycle "$WORK/restart.log")

echo "  cold equilibrium: $C_COLD cycles;  warm restart from cycle $MID: $C_RST cycles"
[[ "$C_RST" -lt "$C_COLD" ]] || { echo "FAIL: restart took $C_RST >= cold $C_COLD cycles -> supplied_wt did not warm-start from the snapshot"; exit 1; }

COLD_TIF=$(ls "$WORK"/cold_*.tif | tail -1); RST_TIF=$(ls "$WORK"/restart_*.tif | tail -1)
TOL="$TOL" PHI="$INP/snaptest_porosity.tif" TESTS="$(readlink -f ..)" "$PY" - "$COLD_TIF" "$RST_TIF" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh

cold, rst = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(cold, bool); m[:, 0] = False   # exclude ocean column
phi = VOL.read_band(os.environ["PHI"])
d = float(VOL.volume_diff(rst, cold, phi)[m].max()); tol = float(os.environ["TOL"])
print(f"  restart vs cold equilibrium: max|ΔV| = {d:.4f} m water volume  (tol {tol})")
if d <= tol:
    print("PASS: snapshot filenames carry the simulated year, and restart-from-snapshot warm-starts to the same equilibrium")
    sys.exit(0)
print(f"FAIL: restart differs from cold equilibrium by {d:.4f} m > tol {tol}")
sys.exit(1)
PY
