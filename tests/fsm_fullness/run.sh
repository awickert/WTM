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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/fsm_fullness_t0_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/ff_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"; export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.yaml" <<EOF
run:
  type: equilibrium
  total_time: 100yr
  supplied_wt: true
time:
  deltat: 31536000
  report_interval: 5
  save_nreport_interval: 9999
grid:
  cells_per_degree: 10
  southern_edge: -45
physics:
  fdepth:
    a: 200
    b: 150
    fmin: 2
  infiltration: false
  evaporation:
    mode: lakes
surface_water:
  fsm: true
  runoff_ratio: false
  runoff_collector: implicit
io:
  surfdatadir: $INP
  region: fsm_fullness
  time_start: t0
  time_end: t0
  textfilename: $WORK/$1.txt
  outfile_prefix: $WORK/${1}_
EOF
}
run() { emit "$1"; "$WTM" "$WORK/$1.yaml" -wtm_anderson $2 -wtm_eq_tol 0 > "$WORK/$1.err" 2>&1 \
        || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.err"; exit 2; }; }
run plain ""
run skim  "-wtm_dev_active_set"

NDEP=$(grep "FSM fullness" "$WORK/skim.err" | tail -1 | sed 's#.*/ \([0-9]*\) depressions.*#\1#')
SP=$(ls "$WORK"/plain_*.tif | tail -1); SK=$(ls "$WORK"/skim_*.tif | tail -1)
NDEP="$NDEP" "$PY" - "$INP/fsm_fullness_t0_topography.tif" "$SP" "$SK" <<'PY'
import sys, os, numpy as np, rasterio
topo, wp, wk = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
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
check("SPILL LEVEL (skim fills to the 97 m sill)", abs(sk - 97.0) < 0.2,
      f"skim lake surface = {sk:.3f} m (known outlet sill = 97.0)")
check("SKIM == PLAIN (skim neither drains nor over-fills)", abs(sk - pl) < 0.2,
      f"skim {sk:.3f} vs plain {pl:.3f} m")
print("PASS: nested hierarchy walked; lake-aware skim reaches the correct spill equilibrium"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
