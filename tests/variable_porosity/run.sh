#!/usr/bin/env bash
# The suite's only SPATIALLY VARYING POROSITY fixture, and the only place the head-vs-volume
# distinction can actually be seen.
#
# Every other fixture is uniform phi = 0.25 and purely subsurface, where V(wtd) has slope exactly phi
# everywhere -- so converting a comparison from head to water volume is an exact x0.25 rescale, the
# cell ranking never changes, and nothing that passes can start failing. The distinction the whole
# #61/#65 arc turns on is invisible there by construction. Production data has variable porosity.
#
# Three things are asserted, and the third is the one that keeps this suite honest:
#   AGREEMENT  two integrators reach the same equilibrium, judged in WATER VOLUME
#   BUDGET     the exact budget closes on a variable-phi domain
#   DISCRIMINATES  the head norm and the volume norm pick DIFFERENT cells. Without this the fixture
#              could drift into uniform-equivalent behaviour and the other two arms would still pass
#              while proving nothing about porosity -- the vacuous-arm failure, one level up.
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
[[ -f inputs/varphi_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
PY="${PY:-python3}"
# Metres OF WATER VOLUME; cross-integrator agreement (cc = backward-euler vs tr-bdf2).
#
# DERIVED 2026-09-19 (#84) FROM A MEASURED SPREAD, because the obvious anchor is WRONG. The natural
# hypothesis -- that two integrators agree to within the adaptive step target, so the bound should be
# k x solver.time_step.error_tol -- was tested and REFUTED. Sweeping error_tol over 8x, everything
# else held, both arms rebuilt from config.yaml:
#
#     error_tol   max|dV| cc-vs-tr   ratio to prev
#       1.0         1.1588e-03            --
#       0.5         1.9838e-03        0.584x     <- the value this suite asserts against
#       0.25        1.1213e-03        1.769x
#       0.125       2.1561e-03        0.520x
#
# NOT MONOTONE, and tightening the target made agreement WORSE twice. The disagreement PLATEAUS at
# 1.1e-03 .. 2.2e-03 rather than tracking the controller, which config.yaml already explains at
# error_tol: it is a PER-STEP LOCAL target and "does not bound the error of the answer". Two schemes
# of different order accumulate differently over a fixed span no matter how the steps are chosen.
#
# So the bound is 3x the WORST case of that spread: 3 x 2.1561e-03 = 6.47e-03, set to 0.0065 (3.01x).
# The previous 0.0125 was a label with no derivation behind it, at 5.8x the same worst case.
#
# WHY 3x AND NOT TIGHTER: run-to-run variation here is ZERO -- this suite is serial, and 1.9838e-03
# reproduces bit-identically across runs and across the dt_min work. So margin buys nothing against
# flakiness; it buys robustness against legitimate future drift, and the sweep is the only empirical
# handle on how much this quantity moves when something legitimate changes (1.9x, under an 8x change
# in the controller target). 3x covers that with room. Tighter would be defensible and was considered
# (2x = 4.3e-03); 3x is the chosen margin, not a measured one.
#
# WHAT IT IS NOT INSURED AGAINST, stated so nobody assumes otherwise: the RUN SPAN. Disagreement
# between a 1st- and a 2nd-order scheme accumulates, so this bound is derived AT time.total
# 604800000s and is invalid if that changes. Re-derive with the sweep above, do not scale it.
# SPREAD: 0   verified 2026-09-20 by assertion_probe.py -- bit-identical across repeat runs, so
#             headroom here measures SENSITIVITY only and can never flake. BITES at 1.786e-03
#             (assertion_probe tightened the bound below the measured value and the suite failed).
TOL="${TOL:-0.0065}"
# |exact_budget_residual| / recharge. MEASURED on this fixture at a fixed span: cc 1.062e-06,
# tr 1.513e-06, so 1e-5 leaves 9.4x and 6.6x margin. Both arms cover identical simulated time.
# RE-MEASURED 2026-09-19 (#84) and the tr figure had drifted: recorded as 1.57e-06, actually
# 1.513e-06. cc was exact. Both reproduce BIT-IDENTICALLY across runs, so this bound carries no
# flake risk -- it ranks "thin" only because it was DERIVED tightly, which is what derived means.
# SPREAD: 0   verified 2026-09-20 by assertion_probe.py -- bit-identical across repeat runs, so
#             headroom here measures SENSITIVITY only and can never flake. BITES at 1.362e-06
#             (assertion_probe tightened the bound below the measured value and the suite failed).
BUDGET_TOL="${BUDGET_TOL:-1e-5}"
make_work varphi
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/variable_porosity/config.yaml, read and edited directly
# rather than translated from legacy key/value lines. Every setting the run resolves to is stated
# there, and tests/config_identity.py enforces it for every suite (#79 Phase 5).
#
# io.region in that file is the LOAD-BEARING line: this is the suite's only spatially varying porosity
# fixture, and the only place the head-vs-volume distinction is visible at all. The integrator is the
# only thing that differs between the arms, so it is the only thing overridden here.
emit() { # $1 stem, $2 solver.time_integration (REQUIRED -- naming it is what keeps the arms distinct)
  local ti="${2:?emit needs a time_integration: name the value for this arm, do not inherit it}"
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  time_integration: backward-euler|  time_integration: $ti|" config.yaml > "$WORK/$1.yaml"
}
run() { emit "$1" "$2"
  WTM_COVERAGE_TAG="variable_porosity/$1" "$WTM" "$WORK/$1.yaml" > "$WORK/$1.log" 2>&1 \
    || { echo "FAIL: $1 did not run cleanly"; tail -4 "$WORK/$1.log"; exit 2; }
}
run cc backward-euler
run tr tr-bdf2

TOL="$TOL" BUDGET_TOL="$BUDGET_TOL" PHI="$INP/varphi_porosity.tif" TESTS="$(readlink -f ..)" \
  "$PY" - "$WORK" <<'PYEOF'
import os, sys
import numpy as np
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL
import wtm_log as LOG
W = sys.argv[1]
phi = VOL.read_band(os.environ["PHI"])
tol, btol = float(os.environ["TOL"]), float(os.environ["BUDGET_TOL"])
cc = VOL.read_band(VOL.latest_output(f"{W}/cc_"))
tr = VOL.read_band(VOL.latest_output(f"{W}/tr_"))
m = np.ones_like(cc, bool); m[:, 0] = False          # exclude the ocean column

fail = 0
def check(name, ok, detail):
    global fail
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<14} {detail}")
    if not ok: fail = 1

print(f"  porosity across the domain: {phi[m].min():.3f} .. {phi[m].max():.3f}")

# --- AGREEMENT, in water volume -----------------------------------------------------------------
dv = VOL.volume_diff(cc, tr, phi)
check("AGREEMENT", float(dv[m].max()) <= tol,
      f"cc vs tr-bdf2: max|dV| = {dv[m].max():.3e} m water volume (tol {tol})")

# --- BUDGET ---------------------------------------------------------------------------------------
for stem in ("cc", "tr"):
    log = LOG.read_log(f"{W}/{stem}.txt")
    rech = abs(log.last("total_recharge_added")) or 1.0
    rel  = abs(log.last("exact_budget_residual")) / rech
    check(f"BUDGET {stem}", rel <= btol, f"|exact residual|/recharge = {rel:.3e} (tol {btol})")

# --- DISCRIMINATES: head and volume must disagree about which cell is worst -----------------------
# This is what makes the fixture worth having. On every uniform-phi fixture these two argmaxes are
# the SAME CELL by construction, so a head norm and a volume norm rank identically and the units
# question cannot be posed. Here they must differ.
dh = np.abs(cc - tr)
jh, ih = np.unravel_index(np.argmax(np.where(m, dh, -np.inf)), dh.shape)
jv, iv = np.unravel_index(np.argmax(np.where(m, dv, -np.inf)), dv.shape)
check("DISCRIMINATES", (jh, ih) != (jv, iv),
      f"head worst at ({jh},{ih}) phi={phi[jh,ih]:.3f}; volume worst at ({jv},{iv}) phi={phi[jv,iv]:.3f}"
      + ("" if (jh, ih) != (jv, iv) else "  -- SAME cell: the fixture no longer exercises porosity"))
print(f"       head max {dh[m].max():.4e} m   volume max {dv[m].max():.4e} m water volume")
sys.exit(fail)
PYEOF
rc=$?
[ "$rc" -eq 0 ] && echo "VARIABLE POROSITY: ALL PASSED" || echo "VARIABLE POROSITY: FAILED"
exit $rc
