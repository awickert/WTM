#!/usr/bin/env bash
# dt-SENSITIVITY: the equilibrium water table must NOT depend on the time step. The active-set semismooth
# seepage face (-wtm_dev_active_set) pins wtd=0 INSIDE the solve, so the free-surface equilibrium is
# dt-INDEPENDENT to machine precision. The legacy taper-1 band sink is NOT (its band width scales as
# 2*qmax*dt, so a table sitting in the band equilibrates at a dt-dependent depth). (The default `implicit`
# in-residual siphon is also dt-DEPENDENT at the face -- its finite 1/dt conductance leaves a dt*excess head
# above the surface -- which is exactly why the active-set face exists.) This test runs one equilibrium
# problem at two time steps (4x apart), holding maxiter fixed so only dt changes, and asserts:
#   DT-INDEPENDENT : under active-set, max|Δwtd| between the two dt is below DT_TOL (measured ~1e-14).
#   BITES          : under runoff_collector=legacy (the band sink), the SAME comparison is dt-DEPENDENT
#                    (max|Δwtd| above BITE_MIN) -- proving the fixture exercises the effect and the active-set
#                    face is what removes it (a regression test that fails without it).
# Total simulated time is matched across the two dt (total_cycles scales inversely), so both reach the same
# equilibrium; maxiter is fixed so the FSM/coupling frequency is not a variable here (that is a separate axis).
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/dtsens_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/dts_XXXX); trap 'rm -rf "$WORK"' EXIT
DT_TOL="${DT_TOL:-1e-3}"     # metres; the active-set equilibrium must match across the 4x dt change (it is ~1e-14)
BITE_MIN="${BITE_MIN:-0.3}"  # metres; the legacy band sink's dt-dependence (band ~2*qmax*dt) far exceeds this
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# The band sink's dt-dependence scales with ABSOLUTE dt (band = 2*qmax*dt), so use YEAR-scale steps to make it
# sharp: coarse = 1 yr x 100 cycles; fine = 0.25 yr x 400 cycles (same total simulated time, maxiter fixed).
emit() { # stem  deltat  total_cycles  collector
  cat > "$WORK/$1.cfg" <<EOF
run_type equilibrium
fsm_on 0
evap_mode 0
infiltration_on 0
runoff_ratio_on 0
runoff_collector $4
cells_per_degree 120
southern_edge 0
deltat $2
total_cycles $3
cycles_to_save $3
maxiter 20
fdepth_a 100
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end tb
surfdatadir $INP
region dtsens
supplied_wt 1
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}
run() { # stem deltat cycles collector extra_flags
  emit "$1" "$2" "$3" "$4"
  "$WTM" "$WORK/$1.cfg" -wtm_anderson $5 -wtm_eq_tol 0 > "$WORK/$1.log" 2>&1 \
    || { echo "RUN FAILED: $1"; tail -3 "$WORK/$1.log"; exit 2; }
}
COARSE=31536000; FINE=7884000   # 1 yr, 0.25 yr
run as_c  $COARSE 100 off    "-wtm_dev_active_set -snes_stol 1e-10"
run as_f  $FINE   400 off    "-wtm_dev_active_set -snes_stol 1e-10"
run leg_c $COARSE 100 legacy ""
run leg_f $FINE   400 legacy ""

AC=$(ls "$WORK"/as_c_*.tif|tail -1); AF=$(ls "$WORK"/as_f_*.tif|tail -1)
LC=$(ls "$WORK"/leg_c_*.tif|tail -1); LF=$(ls "$WORK"/leg_f_*.tif|tail -1)
DT_TOL="$DT_TOL" BITE_MIN="$BITE_MIN" "$PY" - "$AC" "$AF" "$LC" "$LF" <<'PY'
import sys, os, numpy as np, rasterio
ac, af, lc, lf = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:5]]
dt_tol = float(os.environ["DT_TOL"]); bite = float(os.environ["BITE_MIN"])
act = float(np.max(np.abs(ac - af)))   # active-set: dt sensitivity (should be ~0)
leg = float(np.max(np.abs(lc - lf)))   # legacy band sink: dt sensitivity (should be large)
print(f"  DT-INDEPENDENT : active-set        max|wtd(1yr) - wtd(0.25yr)| = {act:.3e} m  (<= {dt_tol})")
print(f"  BITES          : legacy band sink  max|wtd(1yr) - wtd(0.25yr)| = {leg:.3e} m  (>= {bite})")
ok = act <= dt_tol and leg >= bite
print("PASS: the active-set seepage face gives a dt-independent equilibrium; the legacy band sink does not (test bites)"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
