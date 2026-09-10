#!/usr/bin/env bash
# Adaptive-restart robustness regression. The ρ-triggered proactive Anderson restart controller
# (-wtm_adaptive_restart) must run an equilibrium spin-up to completion and settle to the
# SAME water table as a plain Anderson solve.
#
# Bug this guards (robust-finish fix): near equilibrium the Anderson step floors just ABOVE the relative
# step tolerance, so the controller never formally declares true convergence; it then exhausts its restart
# budget and USED TO throw "The SNES solver has not converged" (aborting the run) instead of returning the
# tracked best iterate. A cold start on this gentle subsurface fixture reaches that near-equilibrium regime
# within ~13 cycles, so a bare `-wtm_adaptive_restart` run aborts without the fix -- this test bites.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/arestart_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work arst
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not metres of head: the model
# conserves water and judges every stopping criterion in water volume (#61), so an agreement bound belongs
# in the same units. Uniform phi = 0.25 here, so this is the old 1e-3 m head bound x0.25 exactly.
TOL="${TOL:-0.00025}"     # 0.25 mm of water; adaptive-restart vs plain-Anderson steady-state agreement
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/adaptive_restart/config.yaml, read and edited directly rather
# than translated from legacy key/value lines. Every setting the run resolves to is stated there, and
# tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# The restart CONSTANTS (rho, patience, max_it, max_restarts) are in that file at their shipped values,
# and they are NOT decoration: they set how fast the restart budget is exhausted, which IS the
# near-equilibrium regime the guarded bug lived in. Change one and this test is measuring something
# else -- which is the argument for having them written down rather than inherited.
# AR_ON IS REQUIRED, WITH NO DEFAULT, and that is deliberate. A default here makes an unset value
# silently pick a side: with `:-true` both arms run the restart controller, the test compares a config
# against ITSELF, reports max|dV| = 0.000e+00 and PASSES. That is what a vacuous arm looks like (#24),
# and it is the same way this suite's sibling storage_equivalence went vacuous when a default moved.
emit() { # $1 stem, $2 restart.enabled (REQUIRED: true|false)
  local on="${2:?emit needs restart.enabled: name the value for this arm, do not inherit it}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^      enabled: true|      enabled: $on|" config.yaml > "$WORK/$1.yaml"
}

BB=""
# The restart loop is a CONFIG key now (solver.anderson.restart.enabled); the -wtm_adaptive_restart
# options-database entry is gone, and a -wtm_ nothing reads aborts. Only the `ar` arm enables it --
# `base` is the plain-Anderson control it must match.
emit ar true; emit base false
# (1) adaptive-restart must run to equilibrium WITHOUT aborting (the robustness claim)
"$WTM" "$WORK/ar.yaml" $BB > "$WORK/ar.log" 2>&1 \
  || { echo "FAIL: solver.anderson.restart.enabled aborted (robust-finish regression):"; tail -4 "$WORK/ar.log"; exit 1; }
grep -q "equilibrium reached" "$WORK/ar.log" \
  || { echo "FAIL: solver.anderson.restart.enabled ran but never reached equilibrium"; exit 1; }
# (2) and it must reach the SAME water table as a plain Anderson solve
"$WTM" "$WORK/base.yaml" $BB > "$WORK/base.log" 2>&1 \
  || { echo "FAIL: plain Anderson reference run failed"; tail -4 "$WORK/base.log"; exit 2; }

AR=$(ls "$WORK"/ar_*.tif | tail -1); BASE=$(ls "$WORK"/base_*.tif | tail -1)
TOL="$TOL" PHI="$(readlink -f inputs/arestart_porosity.tif)" TESTS="$(readlink -f ..)" \
  "$PY" - "$AR" "$BASE" <<'PY'
import sys, os, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh
ar, base = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:3]]
phi = VOL.read_band(os.environ["PHI"])
m = np.ones_like(ar, bool); m[:, 0] = False   # exclude the ocean column
d = float(VOL.volume_diff(ar, base, phi)[m].max()); tol = float(os.environ["TOL"])
print(f"  adaptive-restart vs plain Anderson: max|ΔV| = {d:.3e} m water volume  (tol {tol})")
if d <= tol:
    print("PASS: solver.anderson.restart.enabled runs to equilibrium and matches plain Anderson"); sys.exit(0)
print(f"FAIL: adaptive-restart differs from plain Anderson by {d:.3e} > tol {tol} m water"); sys.exit(1)
PY
