#!/usr/bin/env bash
# Adaptive-dt + pure-water-depth-metric regression. On a small coastal wedge, a cold-start equilibrium must
# reach the SAME steady water table regardless of time-integration scheme or equilibrium-stop metric. Guards:
#   (1) -wtm_tr_bdf2 -wtm_dt_adaptive converges AND matches backward-Euler (cc) -> adaptive reaches the
#       correct equilibrium (not just "runs").
#   (2) -wtm_eq_metric water-rms stops on the pure-water-depth metric (|S*Δwtd|) AND reaches the same table.
# Bites if the adaptive controller or the water-depth metric ever produces a wrong field or fails to stop.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/adwater_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/adw_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-0.05}"       # metres; cross-scheme steady-state agreement
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 1
southern_edge 0
deltat 2419200
total_time 24192000000s
save_nreport_interval 200
report_interval 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region adwater
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

BB="-wtm_anderson"
emit cc; emit adapt; emit water
"$WTM" "$WORK/cc.cfg"    $BB -wtm_eq_metric rms       -wtm_eq_tol 0.001  > "$WORK/cc.log"    2>&1 \
  || { echo "RUN FAILED: cc";    tail -3 "$WORK/cc.log";    exit 2; }
"$WTM" "$WORK/adapt.cfg" $BB -wtm_tr_bdf2 -wtm_dt_adaptive -wtm_eq_metric rms -wtm_eq_tol 0.001 > "$WORK/adapt.log" 2>&1 \
  || { echo "RUN FAILED: adapt"; tail -3 "$WORK/adapt.log"; exit 2; }
"$WTM" "$WORK/water.cfg" $BB -wtm_eq_metric water-rms -wtm_eq_tol 0.0005 > "$WORK/water.log" 2>&1 \
  || { echo "RUN FAILED: water"; tail -3 "$WORK/water.log"; exit 2; }

# (1) adaptive must have actually reached equilibrium (not hit the total_time cap)
grep -q "equilibrium reached" "$WORK/adapt.log" || { echo "FAIL: adaptive did not reach equilibrium"; exit 1; }
# (2) the water arm must have stopped on the WATER-depth metric specifically
grep -q "equilibrium reached (water-rms metric)" "$WORK/water.log" \
  || { echo "FAIL: -wtm_eq_metric water-rms did not drive the stop"; grep -i "equilibrium reached" "$WORK/water.log"; exit 1; }

CC=$(ls "$WORK"/cc_*.tif | tail -1); AD=$(ls "$WORK"/adapt_*.tif | tail -1); WA=$(ls "$WORK"/water_*.tif | tail -1)
TOL="$TOL" "$PY" - "$CC" "$AD" "$WA" <<'PY'
import sys, os, numpy as np, rasterio
cc, ad, wa = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
m = np.ones_like(cc, bool); m[:, 0] = False   # exclude the ocean column
d_ad = float(np.max(np.abs((ad - cc)[m]))); d_wa = float(np.max(np.abs((wa - cc)[m])))
tol = float(os.environ["TOL"])
print(f"  adaptive (tr-bdf2+dt_adaptive) vs cc: max|Δwtd| = {d_ad:.4f} m")
print(f"  water-depth metric vs cc:            max|Δwtd| = {d_wa:.4f} m  (tol {tol})")
if d_ad <= tol and d_wa <= tol:
    print("PASS: adaptive dt and the pure-water-depth stop metric both reach cc's equilibrium")
    sys.exit(0)
print(f"FAIL: adaptive={d_ad:.4f} m, water={d_wa:.4f} m exceed tol {tol} m")
sys.exit(1)
PY
