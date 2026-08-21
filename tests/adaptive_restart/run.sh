#!/usr/bin/env bash
# Adaptive-restart robustness regression. The ρ-triggered proactive Anderson restart controller
# (-wtm_anderson -wtm_adaptive_restart) must run an equilibrium spin-up to completion and settle to the
# SAME water table as a plain Anderson solve.
#
# Bug this guards (robust-finish fix): near equilibrium the Anderson step floors just ABOVE the relative
# step tolerance, so the controller never formally declares true convergence; it then exhausts its restart
# budget and USED TO throw "The SNES solver has not converged" (aborting the run) instead of returning the
# tracked best iterate. A cold start on this gentle subsurface fixture reaches that near-equilibrium regime
# within ~13 cycles, so a bare `-wtm_adaptive_restart` run aborts without the fix -- this test bites.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/arestart_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/arst_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-0.001}"       # metres; adaptive-restart vs plain-Anderson steady-state agreement
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 100
southern_edge 0
deltat 2419200
total_time 60480000000s
save_nreport_interval 500
report_interval 50
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region arestart
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

BB="-wtm_eq_metric rms -wtm_eq_tol 0.001 -wtm_anderson"
emit ar; emit base
# (1) adaptive-restart must run to equilibrium WITHOUT aborting (the robustness claim)
"$WTM" "$WORK/ar.cfg" $BB -wtm_adaptive_restart > "$WORK/ar.log" 2>&1 \
  || { echo "FAIL: -wtm_adaptive_restart aborted (robust-finish regression):"; tail -4 "$WORK/ar.log"; exit 1; }
grep -q "equilibrium reached" "$WORK/ar.log" \
  || { echo "FAIL: -wtm_adaptive_restart ran but never reached equilibrium"; exit 1; }
# (2) and it must reach the SAME water table as a plain Anderson solve
"$WTM" "$WORK/base.cfg" $BB > "$WORK/base.log" 2>&1 \
  || { echo "FAIL: plain Anderson reference run failed"; tail -4 "$WORK/base.log"; exit 2; }

AR=$(ls "$WORK"/ar_*.tif | tail -1); BASE=$(ls "$WORK"/base_*.tif | tail -1)
TOL="$TOL" "$PY" - "$AR" "$BASE" <<'PY'
import sys, os, numpy as np, rasterio
ar, base = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
m = np.ones_like(ar, bool); m[:, 0] = False   # exclude the ocean column
d = float(np.max(np.abs((ar - base)[m]))); tol = float(os.environ["TOL"])
print(f"  adaptive-restart vs plain Anderson: max|Δwtd| = {d:.3e} m  (tol {tol})")
if d <= tol:
    print("PASS: -wtm_adaptive_restart runs to equilibrium and matches plain Anderson"); sys.exit(0)
print(f"FAIL: adaptive-restart differs from plain Anderson by {d:.3e} m > tol {tol} m"); sys.exit(1)
PY
