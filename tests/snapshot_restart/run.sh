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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/snaptest_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/snap_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-0.05}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { # stem surfdir supplied_wt
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1
southern_edge 0
deltat 31536000
total_time 100yr
save_nreport_interval 1
report_interval 1
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $2
region snaptest
supplied_wt $3
textfilename $WORK/$1.txt
outfile_prefix $WORK/$1_
EOF
}
stop_cycle() { grep -oE "stopping at cycle [0-9]+" "$1" | grep -oE "[0-9]+$"; }
BB="-wtm_anderson -wtm_eq_tol 0.001 -wtm_eq_metric rms"

# --- cold full run (saves every cycle) ---
emit cold "$INP" 0
"$WTM" "$WORK/cold.cfg" $BB > "$WORK/cold.log" 2>&1 || { echo "RUN FAILED: cold"; tail -3 "$WORK/cold.log"; exit 2; }
C_COLD=$(stop_cycle "$WORK/cold.log")

# --- (1) FILENAME format: year == cycle (deltat 1 yr, report_interval 1) ---
for k in 1 3 5; do
  f=$(printf "%s/cold_%09d_%dyr.tif" "$WORK" "$k" "$k")
  [[ -f "$f" ]] || { echo "FAIL: expected snapshot $(basename "$f") not found (filename year != cycle?)"; ls "$WORK"/cold_*.tif | sed 's#.*/##' | head; exit 1; }
done
echo "  filename format OK: {prefix}{cycle:09}_{year}yr.tif, year==cycle"

# --- (2) RESTART from a mid snapshot (cycle 4, clearly pre-equilibrium) ---
MID=4
SNAP=$(printf "%s/cold_%09d_%dyr.tif" "$WORK" "$MID" "$MID")
[[ -f "$SNAP" ]] || { echo "FAIL: mid snapshot $(basename "$SNAP") missing"; exit 1; }
mkdir -p "$WORK/rinp"; cp "$INP"/*.tif "$WORK/rinp/"; cp "$SNAP" "$WORK/rinp/snaptest_ta_starting_wt.tif"
emit restart "$WORK/rinp" 1
"$WTM" "$WORK/restart.cfg" $BB > "$WORK/restart.log" 2>&1 || { echo "RUN FAILED: restart"; tail -3 "$WORK/restart.log"; exit 2; }
C_RST=$(stop_cycle "$WORK/restart.log")

echo "  cold equilibrium: $C_COLD cycles;  warm restart from cycle $MID: $C_RST cycles"
[[ "$C_RST" -lt "$C_COLD" ]] || { echo "FAIL: restart took $C_RST >= cold $C_COLD cycles -> supplied_wt did not warm-start from the snapshot"; exit 1; }

COLD_TIF=$(ls "$WORK"/cold_*.tif | tail -1); RST_TIF=$(ls "$WORK"/restart_*.tif | tail -1)
TOL="$TOL" "$PY" - "$COLD_TIF" "$RST_TIF" <<'PY'
import sys, os, numpy as np, rasterio
cold, rst = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(cold, bool); m[:, 0] = False   # exclude ocean column
d = float(np.max(np.abs((rst - cold)[m]))); tol = float(os.environ["TOL"])
print(f"  restart vs cold equilibrium: max|Δwtd| = {d:.4f} m  (tol {tol})")
if d <= tol:
    print("PASS: snapshot filenames carry the simulated year, and restart-from-snapshot warm-starts to the same equilibrium")
    sys.exit(0)
print(f"FAIL: restart differs from cold equilibrium by {d:.4f} m > tol {tol}")
sys.exit(1)
PY
