#!/usr/bin/env bash
# Solver-consistency differential oracle. The production solver is matrix-free Anderson, which has no
# independent Jacobian to validate it. This suite runs the two matrix-based solvers as independent oracles
# on a gentle, purely-subsurface equilibrium (make_inputs.py) and asserts all three agree:
#   anderson (-wtm_anderson)                    -- the matrix-free production path
#   picard   (-wtm_picard)                       -- frozen-coefficient backward-Euler operator
#   newton   (method: newton, time_step.mode: ramp)  -- analytic-Jacobian Newton, driven from cold in its
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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
# .tif inputs are gitignored -> generate them if absent (needs rasterio, like the other suites)
[[ -f inputs/sconsist_ta_topography.tif ]] || python3 make_inputs.py >/dev/null
INP=$(readlink -f inputs)
make_work scons
# metres OF WATER VOLUME (|V(wtd_a) - V(wtd_b)|, tests/wtm_volume.py), not metres of head. The model
# conserves water and every stopping criterion is judged in water since #61, so an agreement bound
# belongs in the same units. This fixture is uniform phi = 0.25 and purely subsurface, so the
# conversion from the old 1e-3 m head bound is exactly x0.25 and nothing about what passes changes
# today -- it starts to matter the moment porosity varies (it does in production) or the table
# reaches the surface, where dV/dwtd runs from phi up to 1.
TOL="${TOL:-0.00025}"    # 0.25 mm of water on a ~6 m mound (was 1e-3 m of head)
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/solver_consistency/config.yaml. Every setting the run
# resolves to is stated there, and tests/config_identity.py enforces it (this suite is on
# unconditional since #79 Phase 5).
#
# NAMING THE METHOD FORCES NAMING TWO NEIGHBOURS, and both are arguments here rather than defaults:
#   time_integration  anderson resolves to tr-bdf2, picard and newton to backward-euler
#   time_step.mode    newton needs `ramp` -- it does not converge from a cold start without
#                     dt-continuation -- while the others take adaptive
#   collection.method picard needs `explicit`: active_set is REFUSED on Picard, because the pin is
#                     absent from the Picard operator and RHS, so the constraint would silently fall
#                     through (measured: water piles over 1440 of ~1444 land cells with FSM off)
# That is the point of the suite: the oracle only works if the three solvers are ACTUALLY different,
# and the model downgrades some combinations. An inherited value is how a downgrade goes unnoticed.
emit() { # $1 stem, $2 solver.method, $3 time_integration, $4 time_step.mode, $5 convergence.metric,
         # $6 output.trace, $7 collection.method   (ALL REQUIRED)
  local m="${2:?emit needs a solver.method -- the whole claim is that the arms differ}"
  local ti="${3:?emit needs a time_integration: it follows the method, so name it}"
  local sm="${4:?emit needs a time_step.mode: newton requires ramp}"
  local cm="${5:?emit needs a convergence.metric}"
  local tr="${6:?emit needs an output.trace value, e.g. [] or [water_step]}"
  local col="${7:?emit needs a collection.method: active_set is refused on Picard}"
  # THE RAMP ARM DIFFERS STRUCTURALLY, not just in values: under time_step.mode: ramp the model
  # records solver.newton.dt0 and solver.time_step.dt_max, records NO step norm, and resolves a
  # different error_tol. So that arm adds two keys and drops one rather than merely setting values.
  local ramp=(-e "/^#@RAMP@/d" -e "/^#@RAMPNEWTON@/d")
  if [ "$sm" = ramp ]; then
      ramp=(-e "/^    norm: rms/,+1d"
            -e 's|^#@RAMP@|    dt_max: "2419200000s"   # ramp only: the ceiling the continuation climbs to|'
            -e "s|^    error_tol: 0.0001.*|    error_tol: 0.1   # the value resolved under mode: ramp|"
            -e "s|^#@RAMPNEWTON@|  newton:\n    dt0: \"12096s\"   # ramp only: the continuation start step (deltat/200). The `s` is REQUIRED:\n           #          a bare number is parsed as YEARS|")
  fi
  sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
      -e "s|^  method: anderson|  method: $m|" \
      -e "s|^  time_integration: tr-bdf2|  time_integration: $ti|" \
      -e "s|^    mode: adaptive|    mode: $sm|" \
      -e "s|^    metric: volume|    metric: $cm|" \
      -e "s|^  trace: \[\]|  trace: $tr|" \
      -e "s|^    method: active_set|    method: $col|" \
      "${ramp[@]}" config.yaml > "$WORK/$1.yaml"

}

# Stop 10x TIGHTER than the agreement bound (TOL): each solver only settles to ~eq_tol of the true steady
# state, and eq_tol is a WATER depth (|S*Δwtd|) while the comparison is in wtd -- so a solver stopped at
# eq_tol water sits ~eq_tol/S in wtd from truth. Newton's dt-continuation path lands it on the far side of
# that ball from Anderson (Picard, on a near-identical path, agrees to ~1e-7). eq_tol=1e-4 water puts all
# three within ~1e-4 wtd, well inside the 1e-3 agreement tol. (Converge tighter than you compare.)
# eq_metric/eq_tol now travel in the CONFIG (run.equilibrium_stop.*), so BB is empty.
BB=""
emit anderson anderson tr-bdf2        adaptive volume "[]" active_set
emit picard   picard   backward-euler adaptive volume "[]" explicit
emit newton   newton   backward-euler ramp     volume "[]" active_set
# THE ORACLE IS ONLY AN ORACLE IF THE THREE SOLVERS ARE ACTUALLY DIFFERENT. This suite's whole claim
# is that two matrix-based solvers independently corroborate the matrix-free one -- so if `picard`
# silently downgraded to anderson (which the model DOES do in some combinations, and announces with a
# note), the test would be comparing anderson with itself and would pass while proving nothing. That
# is the vacuous-arm failure in its most damaging form: not a missing check, a fake corroboration.
# expect_resolved reads the fingerprint the MODEL writes after every override and downgrade.
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"   # defer to the suite's log if set
run() { # arm  [expected solver]  [extra flags...]
  local arm="$1" want="${2:-}"; shift; shift || true
  WTM_COVERAGE_TAG="solver_consistency/$arm" "$WTM" "$WORK/$arm.yaml" $BB "$@" > "$WORK/$arm.log" 2>&1 \
    || { echo "FAIL: $arm did not run cleanly (diverged?):"; grep -oE "DIVERGED[A-Z_]*" "$WORK/$arm.log" | tail -1; tail -3 "$WORK/$arm.log"; exit 1; }
  grep -q "equilibrium reached" "$WORK/$arm.log" \
    || { echo "FAIL: $arm ran but never reached equilibrium (hit the cycle cap)"; exit 1; }
  [ -n "$want" ] && { expect_resolved "$WTM_COVERAGE_LOG" "solver=$want" || exit 3; }
  return 0
}
run anderson anderson
run picard picard
run newton newton

# FOURTH ARM: the same Anderson solve with the volume-step DIAGNOSTIC registered
# (-wtm_snes_volume_conv, not _govern). Three things are asserted below, and the fixture is the reason
# they can be: it is gentle and purely SUBSURFACE, so every cell sits on the porosity branch of V(wtd).
emit volconv anderson tr-bdf2 adaptive volume "[water_step]" active_set
run volconv anderson

# FIFTH ARM: the same solve judged in HEAD (solver.convergence.metric: head) instead of water. Water is
# the DEFAULT since #61, so this arm is the deviation and the other four are the control -- it was the
# other way round until the default flipped, at which point asking for `water` here would have made this
# arm a second copy of `anderson` and the assertion below vacuous.
# A convergence criterion decides WHEN a solve stops, never WHERE it converges, so both metrics must land
# on the same equilibrium. That is the whole claim, and it is what makes the default safe to change.
emit volgov  anderson tr-bdf2 adaptive head   "[water_step]" active_set
run volgov anderson

AN=$(ls "$WORK"/anderson_*.tif | tail -1); PI=$(ls "$WORK"/picard_*.tif | tail -1); NE=$(ls "$WORK"/newton_*.tif | tail -1)
VC=$(ls "$WORK"/volconv_*.tif | tail -1); VG=$(ls "$WORK"/volgov_*.tif | tail -1)
TOL="$TOL" PHI="$(readlink -f inputs/sconsist_porosity.tif)" TESTS="$(readlink -f ..)" \
  "$PY" - "$AN" "$PI" "$NE" "$VC" "$WORK/volconv.log" "$VG" <<'PY'
import sys, os, re, numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                      # ONE verified V(wtd); see tests/verify_wtm_volume.sh
an, pi, ne = [rasterio.open(p).read(1).astype(float) for p in sys.argv[1:4]]
vc_tif, vc_log, vg_tif = sys.argv[4], sys.argv[5], sys.argv[6]
phi = VOL.read_band(os.environ["PHI"])
m = np.ones_like(an, bool); m[:, 0] = False   # exclude the ocean column
# Compare in WATER VOLUME. Subtracting the rasters directly would be a head norm with no label on it.
d_pi = float(VOL.volume_diff(pi, an, phi)[m].max()); d_ne = float(VOL.volume_diff(ne, an, phi)[m].max())
tol = float(os.environ["TOL"])
interior = an[m]
print(f"  equilibrium mound elevation: {100 + interior.min():.2f} .. {100 + interior.max():.2f} m "
      f"(all subsurface: {bool((interior < 0).all())})")
print(f"  picard vs anderson: max|ΔV| = {d_pi:.3e} m water")
print(f"  newton vs anderson: max|ΔV| = {d_ne:.3e} m water volume   (tol {tol} m water)")
if not (interior < 0).all():
    print("FAIL: equilibrium is not purely subsurface -> the fixture drifted into the pinned-surface regime "
          "where Picard/Newton are invalid; regenerate inputs / lower the recharge"); sys.exit(1)
# ---- the volume-step diagnostic ----------------------------------------------------------------
# WHAT snorm IS. PETSc hands a convergence test ||dx||, the 2-norm of the step just taken, and
# -snes_stol converges when snorm < stol*xnorm. On the MATRIX-FREE ANDERSON path that step cannot be
# read back: SNESGetSolutionUpdate returns Anderson's raw PRE-MIXING update, measured ~10x the accepted
# step. So VolumeStepConverged keeps its own previous accepted iterate and differences against it.
# recon == snorm is the proof that the reconstruction measures the same step PETSc does -- and nothing
# else in the suite looks at it, so a change to the Anderson update path would silently invalidate the
# water-step machinery that eq_tol and dt_tol are built on.
rows = [tuple(map(float, m.groups())) for m in
        re.finditer(r"\[vol-conv diag\] it=(\d+)\s+head snorm=(\S+) \(recon (\S+)\).*?water max=\S+ L2=(\S+)",
                    open(vc_log).read())]
ok_vc = True
def vcheck(name, cond, detail):
    global ok_vc
    print(f"  {'OK  ' if cond else 'FAIL'} {name}: {detail}"); ok_vc = ok_vc and cond

vcheck("DIAGNOSTIC EMITS", len(rows) > 10, f"{len(rows)} per-iteration lines parsed")

# Printed at %.3e, so equality is asserted at the printed precision, not to machine zero.
worst = max((abs(r - s) / s) for _, s, r, _ in rows if s > 0)
vcheck("RECON == snorm (the reconstruction tracks PETSc's step)", worst < 1e-3,
       f"max |recon - snorm| / snorm = {worst:.3e} over {len(rows)} iterations (< 1e-3)")

# On a purely subsurface fixture V(wtd) = porosity*wtd, so the water step is exactly phi times the head
# step. phi = 0.25 here (make_inputs.py). This is what makes the water metric MEAN something: it is not a
# rescaling of head, it is head weighted by what each cell can actually store.
#
# Divided by PETSc's snorm, NOT by our reconstruction, deliberately: that keeps this check independent of
# the one above, so a broken reconstruction fails exactly one of them rather than both or neither.
ratios = [w / h for _, h, _, w in rows if h > 0]
r_med = float(np.median(ratios))
vcheck("WATER/snorm RATIO == porosity (subsurface fixture)", abs(r_med - 0.25) < 1e-3,
       f"median water_L2 / snorm = {r_med:.6f} (phi = 0.25)")

# ANSWER-NEUTRALITY, which is what makes the diagnostic safe to leave on. Without _govern it must only
# print; if it ever perturbs the solve, this is the arm that says so.
d_vc = float(VOL.volume_diff(rasterio.open(vc_tif).read(1).astype(float), an, phi)[m].max())
vcheck("DIAGNOSTIC IS ANSWER-NEUTRAL", d_vc == 0.0,
       f"max|ΔV(diagnostic) - ΔV(plain anderson)| = {d_vc:.3e} m water (must be exactly 0)")

# GOVERNING. A convergence test decides when to STOP, not where to converge, so swapping the water
# step (the default since #61) for the head step must not move the equilibrium -- and must not be a
# no-op either, or the switch would be untestable by construction.
d_vg = float(VOL.volume_diff(rasterio.open(vg_tif).read(1).astype(float), an, phi)[m].max())
vcheck("GOVERNING lands on the same equilibrium", d_vg <= tol,
       f"max|ΔV(head-governed) - ΔV(water-governed)| = {d_vg:.3e} m water (tol {tol})")
vcheck("GOVERNING is not a no-op", d_vg > 0.0,
       f"the same figure is nonzero, so the criterion really did change the stopping")

if d_pi <= tol and d_ne <= tol and ok_vc:
    print("PASS: Anderson, Picard, and Newton converge to the same interior water table"); sys.exit(0)
if d_pi > tol or d_ne > tol:
    print(f"FAIL: picard={d_pi:.3e}, newton={d_ne:.3e} m water volume exceed tol {tol} m water")
else:
    print("FAIL: the solvers agree, but a volume-step diagnostic assertion above failed")
sys.exit(1)
PY
