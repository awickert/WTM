#!/usr/bin/env bash
# FLICKER mechanism 2 -- the EVAPORATION DISCONTINUITY (below vs above ground) at the free surface (wtd=0).
# See benchmark/FREE_SURFACE_FLICKER.md. POSITIVE test: under the default smooth evaporation taper (taper 2)
# a fixture whose water table is driven across the surface -- and whose legacy hard evap switch limit-cycles
# -- SETTLES to a physically consistent state, with the water budget (now including evaporation) closing.
#
# Fixture (make_inputs.py): a low ocean-ringed plateau with ET < P < owe, so below the surface the cell fills
# toward wtd=0 while above it open-water evaporation drains it back -- opposite pushes across the surface. To
# let the above-surface (owe) branch fire with FSM off, above-surface water is permitted to persist via
# surface_water.collection.method: off -- no enforcement at all -- so the evaporation taper is the ONLY
# manager of the surface crossing. Asserts:
#   SETTLING     : with the smooth taper the run reaches equilibrium (per-cycle |Δwtd| decays; no limit cycle).
#   NO PONDING   : despite ponding being ALLOWED, the taper drives the table back to/below the surface
#                  (wtd <= 0 everywhere) -- the discontinuity is removed, not merely tolerated.
#   MASS BALANCE : at steady state the per-cycle recharge input equals what leaves via evaporation + the
#                  runoff array + ocean outflow:  Δrecharge = Δevap + Δsurface_removed + Δocean_outflow.
#   BITE         : the SAME fixture with the taper OFF (the legacy hard wtd=0 switch) does NOT settle -- the
#                  per-cycle change stays large (period-2 limit cycle). Proves the fixture genuinely flickers
#                  and that taper 2 is the fix (a regression that fails without it).
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # wtm_col: run-log columns BY NAME, not by field number
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/flickevap_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work fe
# metres OF WATER VOLUME (|S*Δwtd|, run-log column abs_change_volume_max), not head (#61/#65).
# THE SCALE FACTOR HERE IS 0.0158, NOT 0.25, and the reason is the whole argument for the metric.
# Column 5 is the max over cells of |Δwtd|; the volume column is the max over cells of |S*Δwtd| -- and
# those maxima fall on DIFFERENT CELLS. Measured on the bare (taper-off) arm: head 1.79714, volume
# 0.0283872. The largest head swing sits in a low-storativity cell that moves almost no water, so the
# "flicker" this fixture detects is far smaller in water than it looks in head.
QUIET="${QUIET:-2.5e-4}"    # settled if the final per-cycle |S*Δwtd| is below this (managed reads exactly 0)
BITE_MIN="${BITE_MIN:-0.015}" # metres OF WATER VOLUME; the hard-switch limit cycle stays far above QUIET.
                            # Set to PRESERVE THE ORIGINAL MARGIN rather than by scaling the old number:
                            # 1.0 against an achieved 1.79714 head was 1.80x, and 0.015 against an achieved
                            # 0.0283872 volume is 1.89x. Scaling 1.0 by 0.25 would have demanded 0.25 from a
                            # control that only reaches 0.0284, failing a fixture that has not changed.
MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { ../emit_config.sh > "$WORK/$1.yaml" <<EOF
taper_surface_transition ${TAPERS:-true}
taper_depth_extinction ${TAPERS:-true}
solver_method anderson
run_type equilibrium
fsm_on 0
# Pinned to the FORMER default collector on purpose. This test's subject is the EVAPORATION
# discontinuity taper: it needs a fixture that visibly flickers with the taper OFF, so the taper can be
# shown to remove it. Under the current default, active_set, the bare (taper-off) arm SETTLES -- the
# semismooth pin removes that flicker by itself -- so the negative control stops discriminating and the
# test reports "does not bite". Hold the collector fixed so the taper remains testable.
# (That active_set alone also kills the evaporation-discontinuity flicker is a real finding, recorded
# in benchmark/scheme_bench/README.md; it is not something this test can demonstrate.)
# collection.method: off -- above-surface water is genuinely UNMANAGED, so the evaporation taper is the
# ONLY thing that can bring the table back down. That is what this suite's header has always claimed,
# and what NO PONDING needs in order to mean anything.
#
# IT WAS `implicit` UNTIL 2026-09-10 (#87). The in-residual siphon removes above-surface water by
# itself, so `wtd <= 0` held whichever mechanism did the work and the assertion could not tell the
# taper from the collector. (The arm ALSO set dev.allow_aboveground_water_columns: true, which read as
# "clamp off" but was overwritten by the collector selector and did nothing at all -- #35.)
runoff_collector off
infiltration_on 0
runoff_ratio_on 0
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
eq_tol 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
# eq_tol 0: run the full fixed cycle count so the per-cycle change is observed, not auto-stopped.
emit managed
"$WTM" "$WORK/managed.yaml" > "$WORK/managed.log" 2>&1 \
  || { echo "RUN FAILED: managed"; tail -3 "$WORK/managed.log"; exit 2; }
TAPERS=false emit bare   # the BARE arm is the one with the tapers off -- that is its subject
"$WTM" "$WORK/bare.yaml" > "$WORK/bare.log" 2>&1 \
  || { echo "RUN FAILED: bare"; tail -3 "$WORK/bare.log"; exit 2; }

# SETTLING (managed): the largest per-cycle |Δwtd| (col 5) over the last few cycles must be small.
VC=$(wtm_col "$WORK/managed.txt" abs_change_volume_max) || exit 1
msettle=$(grep -E '^[0-9]' "$WORK/managed.txt" | tail -4 | awk -v c="$VC" 'BEGIN{m=0}{v=$c+0; if(v>m)m=v}END{print m}')
awk -v v="$msettle" -v q="$QUIET" 'BEGIN{exit !(v+0 <= q+0)}' \
  || { echo "FAIL: managed did not settle -- max recent per-cycle |S*Δwtd|=$msettle > $QUIET (taper not damping?)"; exit 1; }
# BITE (bare): the hard-switch run must NOT settle (limit cycle keeps the per-cycle change large).
BC=$(wtm_col "$WORK/bare.txt" abs_change_volume_max) || exit 1
bsettle=$(grep -E '^[0-9]' "$WORK/bare.txt" | tail -1 | awk -v c="$BC" '{print $c}')
awk -v v="$bsettle" -v b="$BITE_MIN" 'BEGIN{exit !(v+0 >= b+0)}' \
  || { echo "FAIL: bare (taper off) SETTLED (final |S*Δwtd|=$bsettle < $BITE_MIN) -- fixture no longer flickers; test does not bite"; exit 1; }

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
# THE DISCRIMINATOR, and the reason NO PONDING is worth asserting at all. surface_removed must be
# EXACTLY zero: no collector took any water away, so the only thing that could have driven the table
# back to wtd<=0 is the evaporation taper. Without this, NO PONDING passes whenever ANY enforcement is
# active -- which is how it passed for its whole life while a collector, not the taper, did the work
# (#87). If a collector is ever re-enabled here, this fails and says so instead of quietly agreeing.
taper_alone = (dS == 0.0)
if not taper_alone:
    print(f"  FAIL  TAPER-ALONE: surface_removed = {dS:.4e}, not 0. A collector removed water, so "
          f"NO PONDING above does not show the taper did it -- the assertion is vacuous as written.")
else:
    print(f"  TAPER ALONE    : surface_removed = {dS:.4e} -- no collector took any water, so the taper "
          f"is the only thing that brought the table back")

ok = below_ok and rel < mbtol and taper_alone
print("PASS: smooth taper settles the surface-crossing flicker; no ponding remains; budget closes with evaporation"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
