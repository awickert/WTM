#!/usr/bin/env bash
# Solver-consistency differential oracle. The production solver is matrix-free Anderson, which has no
# independent Jacobian to validate it. This suite runs the two matrix-based solvers as independent oracles
# on a gentle, purely-subsurface equilibrium (make_inputs.py) and asserts all three agree:
#   anderson (-wtm_anderson)                    -- the matrix-free production path
#   picard   (-wtm_picard)                       -- frozen-coefficient backward-Euler operator
#   newton   (-wtm_newton -wtm_dt_continuation)  -- analytic-Jacobian Newton, driven from cold in its
#                                                   designed dt-continuation mode (robust; plain cold Newton
#                                                   sits on a knife-edge in this regime)
# All three must (a) actually REACH equilibrium (not hit the cycle cap or diverge) and (b) land on the SAME
# water table. Bites if the Anderson residual, the Picard operator, or the analytic Jacobian ever drifts
# apart -- an independent cross-check no single-solver test can give.
#
# NOTE the regime restriction: Picard/Newton DIVERGE at a pinned free surface (issue #97), so this fixture
# is deliberately gentle/subsurface. Do NOT crank the recharge -- that would break the oracle by design.
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/sconsist_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
WORK=$(mktemp -d /tmp/scons_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-0.001}"       # metres; cross-solver steady-state agreement (1 mm on a ~6 m mound)
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

emit() { ../emit_config.sh > "$WORK/$1.yaml" <<EOF
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
region sconsist
supplied_wt 0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

# Stop 10x TIGHTER than the agreement bound (TOL): each solver only settles to ~eq_tol of the true steady
# state, and eq_tol is a WATER depth (|S*Δwtd|) while the comparison is in wtd -- so a solver stopped at
# eq_tol water sits ~eq_tol/S in wtd from truth. Newton's dt-continuation path lands it on the far side of
# that ball from Anderson (Picard, on a near-identical path, agrees to ~1e-7). eq_tol=1e-4 water puts all
# three within ~1e-4 wtd, well inside the 1e-3 agreement tol. (Converge tighter than you compare.)
BB="-wtm_eq_metric rms -wtm_eq_tol 0.0001"
emit anderson; emit picard; emit newton
run() { # arm  extra-flags...
  local arm="$1"; shift
  "$WTM" "$WORK/$arm.yaml" $BB "$@" > "$WORK/$arm.log" 2>&1 \
    || { echo "FAIL: $arm did not run cleanly (diverged?):"; grep -oE "DIVERGED[A-Z_]*" "$WORK/$arm.log" | tail -1; tail -3 "$WORK/$arm.log"; exit 1; }
  grep -q "equilibrium reached" "$WORK/$arm.log" \
    || { echo "FAIL: $arm ran but never reached equilibrium (hit the cycle cap)"; exit 1; }
}
run anderson -wtm_anderson
run picard   -wtm_picard
run newton   -wtm_newton -wtm_dt_continuation

AN=$(ls "$WORK"/anderson_*.tif | tail -1); PI=$(ls "$WORK"/picard_*.tif | tail -1); NE=$(ls "$WORK"/newton_*.tif | tail -1)
TOL="$TOL" "$PY" - "$AN" "$PI" "$NE" <<'PY'
import sys, os, numpy as np, rasterio
an, pi, ne = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
m = np.ones_like(an, bool); m[:, 0] = False   # exclude the ocean column
d_pi = float(np.max(np.abs((pi - an)[m]))); d_ne = float(np.max(np.abs((ne - an)[m])))
tol = float(os.environ["TOL"])
interior = an[m]
print(f"  equilibrium mound elevation: {100 + interior.min():.2f} .. {100 + interior.max():.2f} m "
      f"(all subsurface: {bool((interior < 0).all())})")
print(f"  picard vs anderson: max|Δwtd| = {d_pi:.3e} m")
print(f"  newton vs anderson: max|Δwtd| = {d_ne:.3e} m   (tol {tol})")
if not (interior < 0).all():
    print("FAIL: equilibrium is not purely subsurface -> the fixture drifted into the pinned-surface regime "
          "where Picard/Newton are invalid; regenerate inputs / lower the recharge"); sys.exit(1)
if d_pi <= tol and d_ne <= tol:
    print("PASS: Anderson, Picard, and Newton converge to the same interior water table"); sys.exit(0)
print(f"FAIL: picard={d_pi:.3e} m, newton={d_ne:.3e} m exceed tol {tol} m"); sys.exit(1)
PY
