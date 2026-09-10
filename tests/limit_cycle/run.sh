#!/usr/bin/env bash
# LIMIT CYCLE -- mechanism 1: storativity-jump / exfiltration overshoot at the free surface (wtd=0).
# See benchmark/FREE_SURFACE_FLICKER.md. This is now a POSITIVE test: under the default exfiltration clamp
# the flicker-prone plateau SETTLES to a physically consistent free boundary, the exfiltrated water is fully
# accounted by the runoff array (mass balance), and two time-integration schemes agree. (The old negative
# "assert the bare flicker exists" version + the nonphysical -wtm_dev_allow_aboveground_water_columns switch are retired;
# the bare/unmanaged contrast is documented in FREE_SURFACE_FLICKER.md, not asserted here.)
#
# Fixture: a high plateau ringed by ocean with strong recharge, so the interior mound rises to the surface
# and exfiltrates -- exactly the regime where backward-Euler + Anderson would overshoot the storativity jump.
# Asserts, with the clamp on by default:
#   SETTLING        : the run reaches equilibrium (per-cycle change decays; it does NOT limit-cycle).
#   COMPLEMENTARITY : wtd <= 0 everywhere AND max wtd = 0 -- cells pinned exactly at the surface are the
#                     exfiltration constraint (the free-boundary complementarity condition).
#   MASS BALANCE    : at steady state (storage constant) the per-cycle recharge input equals what leaves via
#                     the runoff array + ocean outflow:  Δrecharge = Δtotal_surface_removed + Δtotal_ocean_outflow.
#                     runoff_ratio_on 0 so total_surface_removed is purely the exfiltration (exfiltration) runoff.
#   AGREEMENT       : backward-Euler (cc) and BDF2-on-V settle to the same water table.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # wtm_col: run-log columns BY NAME, not by field number
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/limitcyc_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work lc
# TOL is metres OF WATER VOLUME (#61/#65), the old 1e-4 head bound x0.25 on this uniform phi=0.25 fixture.
TOL="${TOL:-2.5e-5}"; MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/limit_cycle/config.yaml. Every setting the run resolves to is
# stated there, and tests/config_identity.py enforces it (this suite is on WTM_DECLARED_SUITES).
#
# equilibrium_stop.tol is 0 in that file, deliberately: this suite watches the PER-CYCLE change for a
# limit cycle, so it must run the clock out rather than stop as soon as the change looks small.
emit() { # $1 stem, $2 time_integration, $3 dev.under_relaxation   (BOTH REQUIRED)
  local ti="${2:?emit needs a time_integration: name the value for this arm, do not inherit it}"
  local rx="${3:?emit needs an under_relaxation: naming it is what keeps the damped arm distinct}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  time_integration: tr-bdf2|  time_integration: $ti|" \
      -e "s|^  under_relaxation: 1|  under_relaxation: $rx|" config.yaml > "$WORK/$1.yaml"
}
BB=""
QUIET="${QUIET:-1e-4}"   # metres; final per-cycle |Δwtd| below this = settled (a limit cycle would stay large)
emit cc tr-bdf2 1; emit bd bdf2 1
"$WTM" "$WORK/cc.yaml" $BB                > "$WORK/cc.log" 2>&1 || { echo "RUN FAILED: cc"; tail -3 "$WORK/cc.log"; exit 2; }
"$WTM" "$WORK/bd.yaml" $BB > "$WORK/bd.log" 2>&1 || { echo "RUN FAILED: bd"; tail -3 "$WORK/bd.log"; exit 2; }

# dev.under_relaxation, which lives here because flicker is what it was built to damp. It blends the
# COMMITTED step, w <- a*w_solve + (1-a)*w_prev, over the whole grid. Two arms, and the second is what
# keeps the first honest:
#   a = 1.0  must be BYTE-IDENTICAL to not setting it. The code has always claimed this ("a=1 -> byte-
#            identical") and nothing ever checked it. An off switch that is not exactly off is worse than
#            no off switch, because every result taken with it is quietly a different model.
#   a = 0.5  must DIFFER, or the key is inert and the check above proves nothing.
emit rx1 tr-bdf2 1.0; emit rx05 tr-bdf2 0.5
"$WTM" "$WORK/rx1.yaml"  $BB > "$WORK/rx1.log"  2>&1 || { echo "RUN FAILED: rx1";  tail -3 "$WORK/rx1.log";  exit 2; }
"$WTM" "$WORK/rx05.yaml" $BB > "$WORK/rx05.log" 2>&1 || { echo "RUN FAILED: rx05"; tail -3 "$WORK/rx05.log"; exit 2; }

# SETTLING: the final per-cycle |wtd change| (col 5) must be small -- a limit cycle would keep it large.
for a in cc bd; do
  LC=$(wtm_col "$WORK/$a.txt" abs_change_volume_max) || exit 1
  last=$(tail -1 "$WORK/$a.txt" | awk -v c="$LC" '{print $c}')
  awk -v v="$last" -v q="$QUIET" 'BEGIN{exit !(v+0 <= q+0)}' \
    || { echo "FAIL: $a did not settle -- final per-cycle |Δwtd|=$last > $QUIET (limit cycle?)"; exit 1; }
done

CC=$(ls "$WORK"/cc_*.tif | tail -1); BD=$(ls "$WORK"/bd_*.tif | tail -1)
RX1=$(ls "$WORK"/rx1_*.tif | tail -1); RX05=$(ls "$WORK"/rx05_*.tif | tail -1)
# MASS BALANCE from the runoff array: per-cycle deltas of cols 9 (recharge), 12 (surface_removed), 13 (ocean_outflow)
read -r dR dS dO < <(grep -E '^[0-9]' "$WORK/cc.txt" | tail -2 | awk 'NR==1{r=$9;s=$12;o=$13} NR==2{print ($9-r), ($12-s), ($13-o)}')
TOL="$TOL" MB_TOL="$MB_TOL" TESTS="$(readlink -f ..)" PHI="$INP/limitcyc_porosity.tif" "$PY" - "$CC" "$BD" "$dR" "$dS" "$dO" "$RX1" "$RX05" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL              # ONE verified V(wtd); see tests/verify_wtm_volume.sh
cc, bd = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
dR, dS, dO = map(float, sys.argv[3:6])
rx1, rx05 = [rasterio.open(p).read(1).astype(float) for p in sys.argv[6:8]]
tol = float(os.environ["TOL"]); mbtol = float(os.environ["MB_TOL"])
above = float(cc.max()); below_ok = bool((cc <= tol).all())
exfiltration = bool(abs(above) < tol)                 # some cells pinned exactly at the surface = the exfiltration constraint
mb = abs(dR - dS - dO)                            # steady-state runoff mass-balance residual
rel = mb / max(abs(dR), 1e-30)
phi = VOL.read_band(os.environ["PHI"])
agree = float(VOL.volume_diff(cc, bd, phi).max())
print(f"  COMPLEMENTARITY: max wtd = {above:.3e} (=0 exfiltration constraint), all wtd<=0: {below_ok}")
print(f"  MASS BALANCE (runoff): dRech={dR:.4e} dSurf_removed={dS:.4e} dOcean={dO:.4e} residual={mb:.3e} (rel {rel:.2e})")
print(f"  AGREEMENT cc vs bdf2v: max|ΔV| = {agree:.3e} m water volume")
# dev.under_relaxation. Asserted at EXACTLY zero: "off" that is only nearly off is worse than no off
# switch, because every result taken with it is quietly a different model.
d_rx1  = float(np.max(np.abs(rx1 - cc)))
d_rx05 = float(np.max(np.abs(rx05 - cc)))
print(f"  UNDER-RELAXATION a=1.0 is OFF: max|Δwtd| vs baseline = {d_rx1:.3e} m (must be exactly 0)")
print(f"  UNDER-RELAXATION a=0.5 DIFFERS: max|Δwtd| vs baseline = {d_rx05:.3e} m (> 0, else a=1 proves nothing)")
relax_ok = (d_rx1 == 0.0) and (d_rx05 > 0.0)

ok = below_ok and exfiltration and rel < mbtol and agree < tol and relax_ok
print("PASS: settles; wtd<=0 with a pinned exfiltration constraint; runoff+ocean close the budget; schemes agree" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
