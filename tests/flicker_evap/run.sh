#!/usr/bin/env bash
# FLICKER mechanism 2 -- the EVAPORATION DISCONTINUITY (below vs above ground) at the free surface (wtd=0).
# See benchmark/FREE_SURFACE_FLICKER.md. POSITIVE test: under the default smooth evaporation taper (taper 2)
# a fixture whose water table is driven across the surface -- and whose legacy hard evap switch limit-cycles
# -- SETTLES to a physically consistent state, with the water budget (now including evaporation) closing.
#
# Fixture (make_inputs.py): a low ocean-ringed plateau with ET < P < owe, so below the surface the cell fills
# toward wtd=0 while above it open-water evaporation drains it back -- opposite pushes across the surface. To
# let the above-surface (owe) branch fire with FSM off, above-surface water is permitted to persist via
# -wtm_dev_allow_aboveground_water_columns (surface clamp off), so the evaporation taper is the ONLY manager
# of the surface crossing. Asserts:
#   SETTLING     : with the smooth taper the run reaches equilibrium (per-cycle |Δwtd| decays; no limit cycle).
#   NO PONDING   : despite ponding being ALLOWED, the taper drives the table back to/below the surface
#                  (wtd <= 0 everywhere) -- the discontinuity is removed, not merely tolerated.
#   MASS BALANCE : at steady state the per-cycle recharge input equals what leaves via evaporation + the
#                  runoff array + ocean outflow:  Δrecharge = Δevap + Δsurface_removed + Δocean_outflow.
#   BITE         : the SAME fixture with the taper OFF (legacy hard evap_mode-1 switch) does NOT settle -- the
#                  per-cycle change stays large (period-2 limit cycle). Proves the fixture genuinely flickers
#                  and that taper 2 is the fix (a regression that fails without it).
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/flickevap_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/fe_XXXX); trap 'rm -rf "$WORK"' EXIT
QUIET="${QUIET:-1e-3}"      # metres; settled if final per-cycle |Δwtd| below this (a limit cycle stays large)
BITE_MIN="${BITE_MIN:-1.0}" # metres; the hard-switch limit cycle keeps per-cycle |Δwtd| far above QUIET
MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 1
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 120
southern_edge 0
deltat 2419200
total_time 14515200000s
save_nreport_interval 120
report_interval 50
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region flickevap
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
POND="-wtm_dev_allow_aboveground_water_columns"   # surface clamp off: let the owe branch fire (FSM off)
# eq_tol 0: run the full fixed cycle count so the per-cycle change is observed, not auto-stopped.
emit managed
"$WTM" "$WORK/managed.cfg" -wtm_anderson $POND -wtm_eq_tol 0 > "$WORK/managed.log" 2>&1 \
  || { echo "RUN FAILED: managed"; tail -3 "$WORK/managed.log"; exit 2; }
emit bare
"$WTM" "$WORK/bare.cfg" -wtm_anderson $POND -wtm_evap_taper 0 -wtm_extinction 0 -wtm_eq_tol 0 > "$WORK/bare.log" 2>&1 \
  || { echo "RUN FAILED: bare"; tail -3 "$WORK/bare.log"; exit 2; }

# SETTLING (managed): the largest per-cycle |Δwtd| (col 5) over the last few cycles must be small.
msettle=$(grep -E '^[0-9]' "$WORK/managed.txt" | tail -4 | awk 'BEGIN{m=0}{v=$5+0; if(v>m)m=v}END{print m}')
awk -v v="$msettle" -v q="$QUIET" 'BEGIN{exit !(v+0 <= q+0)}' \
  || { echo "FAIL: managed did not settle -- max recent per-cycle |Δwtd|=$msettle > $QUIET (taper not damping?)"; exit 1; }
# BITE (bare): the hard-switch run must NOT settle (limit cycle keeps the per-cycle change large).
bsettle=$(grep -E '^[0-9]' "$WORK/bare.txt" | tail -1 | awk '{print $5}')
awk -v v="$bsettle" -v b="$BITE_MIN" 'BEGIN{exit !(v+0 >= b+0)}' \
  || { echo "FAIL: bare (taper off) SETTLED (final |Δwtd|=$bsettle < $BITE_MIN) -- fixture no longer flickers; test does not bite"; exit 1; }

MAN=$(ls "$WORK"/managed_*.tif | tail -1)
# MASS BALANCE from the last two cycles: cols 9 (recharge), 18 (evap), 12 (surface_removed), 13 (ocean_outflow)
read -r dR dE dS dO < <(grep -E '^[0-9]' "$WORK/managed.txt" | tail -2 \
  | awk 'NR==1{r=$9;e=$18;s=$12;o=$13} NR==2{print ($9-r), ($18-e), ($12-s), ($13-o)}')
QUIET="$QUIET" MB_TOL="$MB_TOL" BSETTLE="$bsettle" MSETTLE="$msettle" \
  "$PY" - "$MAN" "$dR" "$dE" "$dS" "$dO" <<'PY'
import sys, os, numpy as np, rasterio
man = rasterio.open(sys.argv[1]).read(1).astype(float)
dR, dE, dS, dO = map(float, sys.argv[2:6])
q = float(os.environ["QUIET"]); mbtol = float(os.environ["MB_TOL"])
above = float(man.max()); below_ok = bool((man <= q).all())
mb = abs(dR - dE - dS - dO); rel = mb / max(abs(dR), 1e-30)
print(f"  SETTLING       : managed max recent per-cycle |Δwtd| = {os.environ['MSETTLE']} m (<= {q}); "
      f"bare (taper off) = {os.environ['BSETTLE']} m (limit cycle)")
print(f"  NO PONDING     : max wtd = {above:.3e} m, all wtd<=0: {below_ok} (ponding allowed, taper drove it back)")
print(f"  MASS BALANCE   : dRech={dR:.4e} dEvap={dE:.4e} dSurf={dS:.4e} dOcean={dO:.4e} residual={mb:.3e} (rel {rel:.2e})")
ok = below_ok and rel < mbtol
print("PASS: smooth taper settles the surface-crossing flicker; no ponding remains; budget closes with evaporation"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
