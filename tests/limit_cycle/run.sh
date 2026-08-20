#!/usr/bin/env bash
# LIMIT CYCLE -- mechanism 1: storativity-jump / seepage overshoot at the free surface (wtd=0).
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
#                     seepage face (the free-boundary complementarity condition).
#   MASS BALANCE    : at steady state (storage constant) the per-cycle recharge input equals what leaves via
#                     the runoff array + ocean outflow:  Δrecharge = Δtotal_surface_removed + Δtotal_ocean_outflow.
#                     runoff_ratio_on 0 so total_surface_removed is purely the exfiltration (seepage) runoff.
#   AGREEMENT       : backward-Euler (cc) and BDF2-on-V settle to the same water table.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/limitcyc_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/lc_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-4}"; MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { cat > "$WORK/$1.cfg" <<EOF
run_type transient
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 120
southern_edge 0
deltat 2419200
total_cycles 60
cycles_to_save 60
maxiter 200
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region limitcyc
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
BB="-wtm_anderson"
QUIET="${QUIET:-1e-4}"   # metres; final per-cycle |Δwtd| below this = settled (a limit cycle would stay large)
emit cc; emit bd
"$WTM" "$WORK/cc.cfg" $BB                > "$WORK/cc.log" 2>&1 || { echo "RUN FAILED: cc"; tail -3 "$WORK/cc.log"; exit 2; }
"$WTM" "$WORK/bd.cfg" $BB -wtm_bdf2_on_V > "$WORK/bd.log" 2>&1 || { echo "RUN FAILED: bd"; tail -3 "$WORK/bd.log"; exit 2; }

# SETTLING: the final per-cycle |wtd change| (col 5) must be small -- a limit cycle would keep it large.
for a in cc bd; do
  last=$(tail -1 "$WORK/$a.txt" | awk '{print $5}')
  awk -v v="$last" -v q="$QUIET" 'BEGIN{exit !(v+0 <= q+0)}' \
    || { echo "FAIL: $a did not settle -- final per-cycle |Δwtd|=$last > $QUIET (limit cycle?)"; exit 1; }
done

CC=$(ls "$WORK"/cc_*.tif | tail -1); BD=$(ls "$WORK"/bd_*.tif | tail -1)
# MASS BALANCE from the runoff array: per-cycle deltas of cols 9 (recharge), 12 (surface_removed), 13 (ocean_outflow)
read -r dR dS dO < <(grep -E '^[0-9]' "$WORK/cc.txt" | tail -2 | awk 'NR==1{r=$9;s=$12;o=$13} NR==2{print ($9-r), ($12-s), ($13-o)}')
TOL="$TOL" MB_TOL="$MB_TOL" "$PY" - "$CC" "$BD" "$dR" "$dS" "$dO" <<'PY'
import sys, os, numpy as np, rasterio
cc, bd = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
dR, dS, dO = map(float, sys.argv[3:6])
tol = float(os.environ["TOL"]); mbtol = float(os.environ["MB_TOL"])
above = float(cc.max()); below_ok = bool((cc <= tol).all())
seepage = bool(abs(above) < tol)                 # some cells pinned exactly at the surface = the seepage face
mb = abs(dR - dS - dO)                            # steady-state runoff mass-balance residual
rel = mb / max(abs(dR), 1e-30)
agree = float(np.max(np.abs(cc - bd)))
print(f"  COMPLEMENTARITY: max wtd = {above:.3e} (=0 seepage face), all wtd<=0: {below_ok}")
print(f"  MASS BALANCE (runoff): dRech={dR:.4e} dSurf_removed={dS:.4e} dOcean={dO:.4e} residual={mb:.3e} (rel {rel:.2e})")
print(f"  AGREEMENT cc vs bdf2v: max|Δwtd| = {agree:.3e}")
ok = below_ok and seepage and rel < mbtol and agree < tol
print("PASS: settles; wtd<=0 with a pinned seepage face; runoff+ocean close the budget; schemes agree" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
