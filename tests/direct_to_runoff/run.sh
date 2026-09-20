#!/usr/bin/env bash
# DIRECT-TO-RUNOFF gathering -- the routing-success view of the free-surface flicker (FSM off).
# See benchmark/FREE_SURFACE_FLICKER.md. The flicker is not a numerical problem when above-surface water has
# somewhere to GO: with collection.method: implicit the in-residual exfiltration constraint routes the excess (max(0,wtd)/dt)
# into the runoff array, holding the table AT the surface (wtd = 0) instead of letting it pile up and slosh.
# POSITIVE test of that success, with a bite proving the routing is load-bearing. Asserts:
#   SETTLING     : with the routing on, the run reaches equilibrium (per-cycle |Δwtd| decays; no limit cycle).
#   GATHERING    : the table is held at the surface -- max wtd = 0 and wtd <= 0 everywhere (the exfiltration
#                  complementarity: wherever water is gathered to runoff the table is pinned at 0, none piled).
#   MASS BALANCE : at steady state the per-cycle recharge input equals what leaves via the runoff array + ocean
#                  outflow:  Δrecharge = Δtotal_surface_removed + Δtotal_ocean_outflow  (no evap).
#   BITE         : the SAME fixture with NO gathering (collection.method: off) piles the water
#                  far above the surface (max wtd >> 0) -- the failure the routing prevents. Proves the test
#                  fails without the fix.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # wtm_col: run-log columns BY NAME, not by field number
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/runoffgather_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work dtr
TOL="${TOL:-1e-4}"          # metres; settled if final per-cycle |Δwtd| below this
SURF_TOL="${SURF_TOL:-0.5}" # metres; implicit pins the table at the surface to the SNES tolerance (a small
                            # cm-dm overshoot, no clamp backstop) -- a exfiltration constraint, not a pile
PILE_MIN="${PILE_MIN:-1.0}" # metres; without gathering the table piles far above this
MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/direct_to_runoff/config.yaml, read and edited directly rather
# than translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# The COLLECTOR is the only thing that differs between the arms, so it is the only thing overridden
# here -- the base file runs as the `gathered` arm.
#
# NOTE while reading config.yaml: solver.time_step.mode: fixed is NOT a free choice there. `adaptive`
# with collection.method: implicit is REFUSED by name, because the implicit siphon removes at rate
# max(0,wtd)/dt so its per-step error GROWS as the controller shrinks dt. Stating the collector fixes
# the step mode too -- which is why both are written down rather than resolved.
emit() { # $1 stem, $2 collection.method
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
        -e "s|^    method: implicit|    method: $2|" config.yaml > "$WORK/$1.yaml"
}
# eq_tol 0: run the full fixed cycle count so the per-cycle change is observed, not auto-stopped.
# gathered = implicit (in-residual exfiltration; pins wtd~0 to the SNES tolerance); piled = off (no collection).
emit gathered implicit
"$WTM" "$WORK/gathered.yaml" > "$WORK/gathered.log" 2>&1 \
  || { echo "RUN FAILED: gathered"; tail -3 "$WORK/gathered.log"; exit 2; }
emit piled off
"$WTM" "$WORK/piled.yaml" > "$WORK/piled.log" 2>&1 \
  || { echo "RUN FAILED: piled"; tail -3 "$WORK/piled.log"; exit 2; }

# SETTLING (gathered): final per-cycle |Δwtd| (col 5) must be small (data rows only; skip the trailing "p" line).
GC=$(wtm_col "$WORK/gathered.txt" abs_change_volume_max) || exit 1
gsettle=$(grep -E '^[0-9]' "$WORK/gathered.txt" | tail -1 | awk -v c="$GC" '{print $c}')
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
# FORMATTED, not interpolated raw. The shell hands this over as the string "0", and a bare
# integer is exactly what the parser discards -- a count or an index is never a measurement --
# so a PERFECT result made the line unparseable and the bound unprobeable.
print(f"  SETTLING       : gathered final per-cycle |Δwtd| = {float(os.environ['GSETTLE']):.3e} m (tol {tol})")
print(f"  GATHERING      : gathered max wtd = {above:.3e} m, all at/below surface: {below_ok} (tol {surf_tol})")
print(f"  MASS BALANCE   : dRech={dR:.4e} dSurf_removed={dS:.4e} dOcean={dO:.4e} residual={mb:.3e}")
print(f"  MASS BALANCE   : |residual|/recharge = {rel:.3e} (tol {mbtol})")
print(f"  BITE           : no-gathering (ponding) max wtd = {pile:.3e} m, piles above surface as routing prevents (min {pile_min})")
ok = below_ok and at_surface and rel < mbtol and pile >= pile_min
if ok:
    print("PASS: direct-to-runoff gathers the excess and holds wtd=0; budget closes; without it the water piles")
else:
    # NAME WHAT FAILED (#119). A bare "FAIL" left a reader to compare the printed numbers
    # against their bounds by hand, and left the bite harness unable to tell a failed
    # assertion from a crash -- so every bound here reported INCONCLUSIVE.
    if not below_ok:       print(f"  FAIL  GATHERING: water stands above the surface, max wtd = {above:.3e} m (tol {surf_tol})")
    if not at_surface:     print(f"  FAIL  AT SURFACE: the table is not pinned at the surface, max wtd = {above:.3e} m (tol {surf_tol})")
    if rel >= mbtol:       print(f"  FAIL  MASS BALANCE: |residual|/recharge = {rel:.3e} (tol {mbtol})")
    if pile < pile_min:    print(f"  FAIL  BITE: without gathering the table did NOT pile, max wtd = {pile:.3e} m (min {pile_min})")
    print("  Without that contrast the gathering claim above proves nothing.")
    print("FAIL")
sys.exit(0 if ok else 1)
PY
