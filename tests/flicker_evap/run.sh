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
# SPREAD: 0   measured 2026-09-22 by repeat run; see this file's first bound.
# DERIVED 2026-09-22, SEPARATING: the managed (taper on) arm measures 1.8053e-11 m of per-cycle
#   |S*dwtd| -- effectively settled -- while the bare (taper off) arm sustains 2.838e-02 m. The bound
#   sits inside that nine-order gap. RENAMED from QUIET 2026-09-22: assertion_health.py only
#   recognises a bound whose NAME contains TOL/MIN/MAX/FLOOR/BAR, so this one was invisible to the
#   framework and its two assertions read as a coverage gap that did not exist.
QUIET_TOL="${QUIET_TOL:-2.5e-4}"    # settled if the final per-cycle |S*Δwtd| is below this (managed reads exactly 0)
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, SEPARATING, and the gap is NINE ORDERS: the managed (taper on) arm settles to
#   1.8053e-11 m of per-cycle motion while the bare (taper off) arm sustains 2.838e-02 m -- a limit
#   cycle that does not decay. The floor at 0.015 sits 1.9x below the flickering value. Sharp, and
#   deliberately so: if the fixture ever stops flickering this test proves nothing, and the guard
#   should fail loudly rather than pass quietly.
BITE_MIN="${BITE_MIN:-0.015}" # metres OF WATER VOLUME; the hard-switch limit cycle stays far above QUIET_TOL.
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
                            # Set to PRESERVE THE ORIGINAL MARGIN rather than by scaling the old number:
                            # 1.0 against an achieved 1.79714 head was 1.80x, and 0.015 against an achieved
                            # 0.0283872 volume is 1.89x. Scaling 1.0 by 0.25 would have demanded 0.25 from a
                            # control that only reaches 0.0284, failing a fixture that has not changed.
# DERIVED 2026-09-22, ONE-SIDED and BLUNT: measured |residual|/recharge = 2.125e-07, so the bound
#   is 4700x the measurement and blind to closure degrading by two orders. Same shape and same
#   value as direct_to_runoff's MB_TOL, and blunt for the same reason. Recorded, not retightened.
MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/flicker_evap/config.yaml. Every setting the run resolves to
# is stated there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# BOTH TAPERS ARE THE SUBJECT and both are supplied per arm. Since #88 removed evap_mode, the taper is
# the ONLY thing that removes surface water here, so the taper-off arm is a real control rather than
# a cosmetic change.
emit() { # $1 stem, $2 both evaporation.tapers (REQUIRED: true|false)
  local tp="${2:?emit needs the taper setting: it IS the subject, never inherit it}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|@TAPER@|$tp|g" config.yaml > "$WORK/$1.yaml"
}
# eq_tol 0: run the full fixed cycle count so the per-cycle change is observed, not auto-stopped.
emit managed true
"$WTM" "$WORK/managed.yaml" > "$WORK/managed.log" 2>&1 \
  || { echo "RUN FAILED: managed"; tail -3 "$WORK/managed.log"; exit 2; }
emit bare false   # the BARE arm is the one with the tapers off -- that is its subject
"$WTM" "$WORK/bare.yaml" > "$WORK/bare.log" 2>&1 \
  || { echo "RUN FAILED: bare"; tail -3 "$WORK/bare.log"; exit 2; }

# SETTLING (managed): the largest per-cycle |Δwtd| (col 5) over the last few cycles must be small.
VC=$(wtm_col "$WORK/managed.txt" abs_change_volume_max) || exit 1
msettle=$(grep -E '^[0-9]' "$WORK/managed.txt" | tail -4 | awk -v c="$VC" 'BEGIN{m=0}{v=$c+0; if(v>m)m=v}END{print m}')
awk -v v="$msettle" -v q="$QUIET_TOL" 'BEGIN{exit !(v+0 <= q+0)}' \
  || { echo "FAIL: managed did not settle -- max recent per-cycle |S*Δwtd|=$msettle > $QUIET_TOL (taper not damping?)"; exit 1; }
# BITE (bare): the hard-switch run must NOT settle (limit cycle keeps the per-cycle change large).
BC=$(wtm_col "$WORK/bare.txt" abs_change_volume_max) || exit 1
bsettle=$(grep -E '^[0-9]' "$WORK/bare.txt" | tail -1 | awk -v c="$BC" '{print $c}')
# PRINTS ON PASS TOO. Written as a bare `awk || { ... }` this guard was silent whenever it held,
# so the one number proving the fixture still flickers never reached the log and the bound could
# not be read by tests/assertion_health.py at all.
if awk -v v="$bsettle" -v b="$BITE_MIN" 'BEGIN{exit !(v+0 >= b+0)}'; then
  echo "  PASS  BITE           : bare (taper off) still flickers, final |S*Δwtd| = $bsettle m (min BITE_MIN=$BITE_MIN)"
else
  echo "FAIL: bare (taper off) SETTLED (final |S*Δwtd|=$bsettle < $BITE_MIN) -- fixture no longer flickers; test does not bite"; exit 1
fi

MAN=$(ls "$WORK"/managed_*.tif | tail -1)
# MASS BALANCE from the last two cycles: cols 9 (recharge), 18 (evap), 12 (surface_removed), 13 (ocean_outflow)
read -r dR dE dS dO < <(grep -E '^[0-9]' "$WORK/managed.txt" | tail -2 \
  | awk 'NR==1{r=$9;e=$18;s=$12;o=$13} NR==2{print ($9-r), ($18-e), ($12-s), ($13-o)}')
QUIET_TOL="$QUIET_TOL" MB_TOL="$MB_TOL" BSETTLE="$bsettle" MSETTLE="$msettle" \
  "$PY" - "$MAN" "$dR" "$dE" "$dS" "$dO" <<'PY'
import sys, os, numpy as np, rasterio
man = rasterio.open(sys.argv[1]).read(1).astype(float)
dR, dE, dS, dO = map(float, sys.argv[2:6])
q = float(os.environ["QUIET_TOL"]); mbtol = float(os.environ["MB_TOL"])
above = float(man.max()); below_ok = bool((man <= q).all())
mb = abs(dR - dE - dS - dO); rel = mb / max(abs(dR), 1e-30)
print(f"  SETTLING       : managed max recent per-cycle |Δwtd| = {os.environ['MSETTLE']} m (tol QUIET_TOL={q}); "
      f"bare (taper off) = {os.environ['BSETTLE']} m (limit cycle)")
print(f"  NO PONDING     : max wtd = {above:.3e} m, ponding allowed and the taper drove it back (tol QUIET_TOL={q})")
print(f"  MASS BALANCE   : dRech={dR:.4e} dEvap={dE:.4e} dSurf={dS:.4e} dOcean={dO:.4e} residual={mb:.3e}")
print(f"  MASS BALANCE   : |residual|/recharge = {rel:.3e} (tol MB_TOL={mbtol})")
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
if ok:
    print("PASS: smooth taper settles the surface-crossing flicker; no ponding remains; budget closes with evaporation")
else:
    # NAME WHAT FAILED (#119). A bare "FAIL" left a reader to compare the printed numbers
    # against their bounds by hand, and left the bite harness unable to tell a failed
    # assertion from a crash -- so every bound here reported INCONCLUSIVE.
    if not below_ok:     print(f"  FAIL  NO PONDING: water stands above the surface, max wtd = {above:.3e} m (tol QUIET_TOL={q})")
    if rel >= mbtol:     print(f"  FAIL  MASS BALANCE: |residual|/recharge = {rel:.3e} (tol MB_TOL={mbtol})")
    if not taper_alone:  print(f"  FAIL  TAPER ALONE: a collector removed {dS:.4e} m of water, so the taper is not what settled it")
    print("FAIL")
sys.exit(0 if ok else 1)
PY
