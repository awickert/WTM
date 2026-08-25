#!/usr/bin/env bash
# DIRECT-TO-RUNOFF gathering -- the routing-success view of the free-surface flicker (FSM off).
# See benchmark/FREE_SURFACE_FLICKER.md. The flicker is not a numerical problem when above-surface water has
# somewhere to GO: with -wtm_direct_to_runoff the in-residual exfiltration constraint routes the excess (max(0,wtd)/dt)
# into the runoff array, holding the table AT the surface (wtd = 0) instead of letting it pile up and slosh.
# POSITIVE test of that success, with a bite proving the routing is load-bearing. Asserts:
#   SETTLING     : with the routing on, the run reaches equilibrium (per-cycle |Δwtd| decays; no limit cycle).
#   GATHERING    : the table is held at the surface -- max wtd = 0 and wtd <= 0 everywhere (the exfiltration
#                  complementarity: wherever water is gathered to runoff the table is pinned at 0, none piled).
#   MASS BALANCE : at steady state the per-cycle recharge input equals what leaves via the runoff array + ocean
#                  outflow:  Δrecharge = Δtotal_surface_removed + Δtotal_ocean_outflow  (evap_mode 0, no evap).
#   BITE         : the SAME fixture with NO gathering (-wtm_dev_allow_aboveground_water_columns) piles the water
#                  far above the surface (max wtd >> 0) -- the failure the routing prevents. Proves the test
#                  fails without the fix.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/runoffgather_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/dtr_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-4}"          # metres; settled if final per-cycle |Δwtd| below this
SURF_TOL="${SURF_TOL:-0.5}" # metres; implicit pins the table at the surface to the SNES tolerance (a small
                            # cm-dm overshoot, no clamp backstop) -- a exfiltration constraint, not a pile
PILE_MIN="${PILE_MIN:-1.0}" # metres; without gathering the table piles far above this
MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 120
southern_edge 0
deltat 2419200
total_time 9676800000s
save_nreport_interval 80
report_interval 50
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region runoffgather
supplied_wt 1
runoff_collector $2
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
# eq_tol 0: run the full fixed cycle count so the per-cycle change is observed, not auto-stopped.
# gathered = implicit (in-residual exfiltration; pins wtd~0 to the SNES tolerance); piled = off (no collection).
emit gathered implicit
"$WTM" "$WORK/gathered.yaml" -wtm_anderson -wtm_eq_tol 0 > "$WORK/gathered.log" 2>&1 \
  || { echo "RUN FAILED: gathered"; tail -3 "$WORK/gathered.log"; exit 2; }
emit piled off
"$WTM" "$WORK/piled.yaml" -wtm_anderson -wtm_eq_tol 0 > "$WORK/piled.log" 2>&1 \
  || { echo "RUN FAILED: piled"; tail -3 "$WORK/piled.log"; exit 2; }

# SETTLING (gathered): final per-cycle |Δwtd| (col 5) must be small (data rows only; skip the trailing "p" line).
gsettle=$(grep -E '^[0-9]' "$WORK/gathered.txt" | tail -1 | awk '{print $5}')
awk -v v="$gsettle" -v q="$TOL" 'BEGIN{exit !(v+0 <= q+0)}' \
  || { echo "FAIL: gathered did not settle -- final per-cycle |Δwtd|=$gsettle > $TOL"; exit 1; }

GAT=$(ls "$WORK"/gathered_*.tif | tail -1); PIL=$(ls "$WORK"/piled_*.tif | tail -1)
# MASS BALANCE from the runoff array: per-cycle deltas of cols 9 (recharge), 12 (surface_removed), 13 (ocean)
read -r dR dS dO < <(grep -E '^[0-9]' "$WORK/gathered.txt" | tail -2 | awk 'NR==1{r=$9;s=$12;o=$13} NR==2{print ($9-r), ($12-s), ($13-o)}')
TOL="$TOL" SURF_TOL="$SURF_TOL" MB_TOL="$MB_TOL" PILE_MIN="$PILE_MIN" GSETTLE="$gsettle" \
  "$PY" - "$GAT" "$PIL" "$dR" "$dS" "$dO" <<'PY'
import sys, os, numpy as np, rasterio
gat, pil = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
dR, dS, dO = map(float, sys.argv[3:6])
tol = float(os.environ["TOL"]); mbtol = float(os.environ["MB_TOL"]); pile_min = float(os.environ["PILE_MIN"])
surf_tol = float(os.environ["SURF_TOL"])
above = float(gat.max()); below_ok = bool((gat <= surf_tol).all())
at_surface = bool(abs(above) <= surf_tol)       # gathered: table pinned at the surface (exfiltration constraint, SNES-tol overshoot)
mb = abs(dR - dS - dO); rel = mb / max(abs(dR), 1e-30)
pile = float(pil.max())                          # without gathering: piles far above the surface
print(f"  SETTLING       : gathered final per-cycle |Δwtd| = {os.environ['GSETTLE']} m (<= {tol})")
print(f"  GATHERING      : gathered max wtd = {above:.3e} m (<= {surf_tol}, at surface), all wtd<={surf_tol}: {below_ok}")
print(f"  MASS BALANCE   : dRech={dR:.4e} dSurf_removed={dS:.4e} dOcean={dO:.4e} residual={mb:.3e} (rel {rel:.2e})")
print(f"  BITE           : no-gathering (ponding) max wtd = {pile:.3e} m (piles above surface; routing prevents this)")
ok = below_ok and at_surface and rel < mbtol and pile >= pile_min
print("PASS: direct-to-runoff gathers the excess and holds wtd=0; budget closes; without it the water piles"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
