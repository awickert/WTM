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
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py -- bit-identical across repeat runs.
#             Headroom here therefore measures SENSITIVITY, never flake risk; see
#             tests/ASSERTION_HEALTH.md sec 3 for why that inverts how a low headroom reads.
# DERIVED 2026-09-22, ONE-SIDED: measured gathered final per-cycle |dwtd| = 0.000e+00 m -- the run
#   reaches a genuine fixed point, not a small residual motion. There is no broken arm HERE to
#   separate against, so the bound is a convention. For scale, and marked as a CROSS-SUITE
#   reference rather than a measurement on this fixture: a lakeshore limit cycle on a
#   surface-crossing domain shows per-cycle motion of order 1e-2 m (flicker_evap, active_set), two
#   orders above this bound.
TOL="${TOL:-1e-4}"          # metres; settled if final per-cycle |Δwtd| below this
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, SEPARATING: measured gathered max wtd = 2.324e-01 m against 9.207e+02 m for
#   the same fixture with gathering off (the BITE arm below). The bound sits 2.15x above the
#   gathered value and 3 orders below the piled one. It is NOT the SNES tolerance: the solve is
#   converged to water_volume_tol 1e-08, and the residual 0.23 m of standing water is physical --
#   water the routing has not yet moved -- so the bound is sized to the fixture, not the solver.
SURF_TOL="${SURF_TOL:-0.5}" # metres; implicit pins the table at the surface to the SNES tolerance (a small
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
                            # cm-dm overshoot, no clamp backstop) -- a exfiltration constraint, not a pile
# DERIVED 2026-09-22, SEPARATING: measured no-gathering max wtd = 9.207e+02 m against 2.324e-01 m
#   with gathering on. The floor sits 4.3x above the gathered value and 920x below the piled one,
#   inside a gap whose both edges this suite measures.
PILE_MIN="${PILE_MIN:-1.0}" # metres; without gathering the table piles far above this
# SPREAD: 0   measured 2026-09-22 by assertion_probe.py; see the note at this file's first bound.
# DERIVED 2026-09-22, ONE-SIDED, and BLUNT -- say so rather than leave it in a passing suite:
#   measured |residual|/recharge = 2.520e-07, so the bound is 4000x the measurement. It would not
#   notice closure degrading by two orders. The multiplier is a CONVENTION; the measurement is not.
#   The floor under the residual is the per-solve water tolerance (water_volume_tol 1e-08 in this
#   suite's config.yaml) accumulated over the run, which is why 2.5e-07 and not zero.
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
print(f"  SETTLING       : gathered final per-cycle |Δwtd| = {float(os.environ['GSETTLE']):.3e} m (tol TOL={tol})")
print(f"  GATHERING      : gathered max wtd = {above:.3e} m, all at/below surface: {below_ok} (tol SURF_TOL={surf_tol})")
print(f"  MASS BALANCE   : dRech={dR:.4e} dSurf_removed={dS:.4e} dOcean={dO:.4e} residual={mb:.3e}")
print(f"  MASS BALANCE   : |residual|/recharge = {rel:.3e} (tol MB_TOL={mbtol})")
print(f"  BITE           : no-gathering (ponding) max wtd = {pile:.3e} m, piles above surface as routing prevents (min PILE_MIN={pile_min})")
ok = below_ok and at_surface and rel < mbtol and pile >= pile_min
if ok:
    print("PASS: direct-to-runoff gathers the excess and holds wtd=0; budget closes; without it the water piles")
else:
    # NAME WHAT FAILED (#119). A bare "FAIL" left a reader to compare the printed numbers
    # against their bounds by hand, and left the bite harness unable to tell a failed
    # assertion from a crash -- so every bound here reported INCONCLUSIVE.
    if not below_ok:       print(f"  FAIL  GATHERING: water stands above the surface, max wtd = {above:.3e} m (tol SURF_TOL={surf_tol})")
    if not at_surface:     print(f"  FAIL  AT SURFACE: the table is not pinned at the surface, max wtd = {above:.3e} m (tol SURF_TOL={surf_tol})")
    if rel >= mbtol:       print(f"  FAIL  MASS BALANCE: |residual|/recharge = {rel:.3e} (tol MB_TOL={mbtol})")
    if pile < pile_min:    print(f"  FAIL  BITE: without gathering the table did NOT pile, max wtd = {pile:.3e} m (min PILE_MIN={pile_min})")
    print("  Without that contrast the gathering claim above proves nothing.")
    print("FAIL")
sys.exit(0 if ok else 1)
PY
