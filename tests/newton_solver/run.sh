#!/usr/bin/env bash
# NEWTON solver coverage: the analytic Jacobian, and the contract for using it.
#
# WHY THIS EXISTS. Newton is a shipped solver that had almost no coverage. Its only budget arm was
# added late, and its Jacobian was FD-checked in exactly one configuration (tests/ghost_boundary, on a
# fixture whose steady water table is flat at 0 m) -- so the ACTIVE-SET tangent, an identity row on the
# pinned cells, had never been verified on a fixture where any cell is actually pinned. A tangent that
# is wrong only where the constraint is active would pass everything we had.
#
# WHAT IT ASSERTS
#   1. PRECONDITION  the active set is genuinely non-empty on this fixture, so check 2 is not vacuous.
#   2. JACOBIAN      ||J - Jfd||_F/||J||_F stays under the ceiling for each collector whose tangent is
#                    claimed to be wired. Measured: active_set 0.00415, explicit 0.000993.
#                    `implicit` is a KNOWN HOLE at 0.845 -- its kink tangent is wired into the Anderson
#                    residual and the Picard operator but NOT the Newton Jacobian, and the code already
#                    warns so. Held as a guarded expected-inconsistency: the WARNING must be emitted,
#                    and the ratio must stay large. If it ever goes small, the tangent has been wired
#                    and this arm must be promoted to a real check.
#   3. SAME ROOT     Newton and Anderson differentiate/iterate the SAME residual, so at equilibrium
#                    they must find the same water table. Measured 2.518e-02 m (rms 9.410e-03).
# THE TEST CARRIES ITS OWN POSITIVE CONTROL. `implicit` is a collector whose tangent is known NOT to
# be in the Jacobian, and it reads 0.845 while the two wired collectors read 0.00415 and 0.000993 --
# so the FD check is demonstrably able to tell a missing tangent from a present one, on this fixture,
# at this ceiling. That is stronger than asserting a small number and hoping it means something.
#
#   4. CONTRACT      plain Newton AT FIXED dt does not converge on these fixtures. It needs EITHER
#                    solver.newton.dt_continuation OR adaptive stepping -- both arms are run, so the
#                    rule is demonstrated rather than asserted. Pinned so the requirement is recorded
#                    rather than folklore, and so the day Newton needs neither, this test says so.
#
#                    The "OR adaptive" half is new (2026-09-03). It was found when adaptive_dt: auto
#                    began resolving TRUE for this arm -- dt_continuation: false, collector active_set
#                    -- and plain Newton CONVERGED, failing a contract written when fixed dt was the
#                    only option. The controller's reject-and-shrink does the globalising the
#                    continuation ramp was doing. Checked on two other fixtures before rewording:
#                    boundary_consistency (fixed: not converged; adaptive: equilibrium reached) and
#                    solver_consistency (fixed: not converged; adaptive: ran the full span without
#                    diverging, though without settling). So adaptive PREVENTS THE DIVERGENCE; it does
#                    not promise equilibrium in a given budget, and it does NOT replace continuation --
#                    adaptive is bound to the report span while the ramp is bound only by dt_max, so
#                    the ramp still reaches steady state in far larger steps.
#
# A NOTE ON WHAT NOT TO COMPARE. Newton is normally run with -wtm_dt_continuation, whose loop runs
# report_steps STEPS at a dt it may grow -- so a continuation cycle does NOT cover one report span, and
# after N cycles Newton and Anderson have simulated DIFFERENT amounts of time. Comparing them at a
# fixed cycle count therefore compares two different instants: it showed a 35 m discrepancy that was
# entirely an artefact of one run having reached 3.2 years and the other 20. Compare at EQUILIBRIUM,
# where elapsed time no longer matters, and check the elapsed_time column when in doubt.
#
# Usage:  tests/newton_solver/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work newton
JTOL="${JTOL:-1e-2}"      # ||J-Jfd||/||J|| ceiling; the piecewise kink keeps it well above 1e-8
# metres OF WATER VOLUME (|V(wtd_a)-V(wtd_b)|, tests/wtm_volume.py), not head (#61/#65).
# THE VALUE DOES NOT SCALE BY phi HERE, and the reason is the point of the whole conversion: on
# this fixture the move to water CHANGES WHICH CELL GOVERNS. Measured, Anderson vs Newton:
#     HEAD  max 4.7388e-02 at (12,13), wtd = -30.16  -- a DEEP cell, where that head difference
#                                                       is only ~1.2e-02 m of actual water
#     WATER max 4.2714e-02 at (9,9),   wtd = +0.175  -- a cell AT THE SURFACE, dV/dh = 0.999
# So the old head norm was reporting a disagreement that barely moved any water, while the real
# largest disagreement in water sat somewhere else entirely. A blind x0.25 would have set the
# bound to 0.0125 and failed a test that had not regressed.
# 0.05 m OF WATER keeps the original numeric bound where it governs (the surface, dV/dh ~ 1) and
# is 4x STRICTER below ground, so this is not a loosening in any regime.
# NOTE the margin is thin either way: 0.05 against an achieved 4.27e-02 is 1.17x (the old head
# pairing was tighter still, 1.055x). This test runs close to its limit by nature.
# DERIVED, NOT CHOSEN (#84). 0.05 was a round number with 1.17x of headroom -- the thinnest margin in
# the suite, and the first thing tol_margin.py flagged when it was finally wired in.
#
# THE HYPOTHESIS FOR WHERE IT SHOULD COME FROM WAS WRONG, and the refutation is the useful part.
# tol_margin.py's own docstring named this assertion as an example: "each solver converged to the run's
# own water tolerance, so THAT is what bounds their disagreement -- and it moves automatically if
# someone tightens the solver." MEASURED by sweeping the equilibrium tolerance both arms share, over
# three orders of magnitude:
#       eq_tol 1e-3   max|dV| = 4.307e-02
#       eq_tol 1e-4   max|dV| = 4.271e-02      <- what this suite runs
#       eq_tol 1e-5   max|dV| = 4.268e-02
#       eq_tol 1e-6   max|dV| = 4.267e-02
# A 1000x tightening moves it 1%. Noise scales; this PLATEAUS. So the disagreement is not solver slop
# bounded by the stopping criterion -- it is a REAL, CONVERGED difference between the two solvers'
# fixed points, stable to 1%. Tightening the solver would not move this bound, and deriving it from
# solver tolerance would have been deriving it from something that does not govern it.
#
# SO THE BOUND IS DERIVED FROM THE MEASUREMENT: 2x the largest value over that sweep,
# 2 x 4.307e-02 = 8.6e-02. The factor is stated rather than implied -- it fails when the two solvers
# disagree TWICE as much as they measurably do, which is a change worth reporting, while 1% run-to-run
# variation cannot flip it. Re-derive by re-running the sweep above if the fixture or either solver
# changes.
#
# STILL OPEN, and deliberately not closed here: WHY they differ by 4.3e-02 at all. Both are pinned to
# fixed dt so the stepping is held equal, but they stop at different cycles (466 vs 463), and with FSM
# on a different step count can flip a discrete fill/spill decision. That is a question about the
# model, not about this tolerance, and it should not be settled by widening a bound.
AGREE_TOL="${AGREE_TOL:-0.086}"   # metres OF WATER VOLUME; see the derivation above
export OMP_NUM_THREADS=1

# THE CONFIGS ARE FILES NOW (#83): config.yaml (mode: fixed), config_ramp.yaml, config_adaptive.yaml.
# THREE files because the three step modes resolve DIFFERENT KEY SETS -- ramp alone carries dt_max and
# solver.newton.dt0, adaptive alone carries norm, fixed carries no controller dials at all -- and a
# config must state what its run resolves to, not a superset.
#
# mkcfg RENDERS one; every per-arm value is REQUIRED, with no default. That is deliberate: these arms
# differ from each other in exactly these keys, and a default is how an arm silently becomes a copy of
# another one (#24). ROUTING is among them because an ABSENT routing key resolves PER COLLECTOR --
# measured: active_set -> continuous, explicit -> impulse, implicit -> continuous -- so the three
# Jacobian arms were each getting a different coupling with nothing saying so.
mkcfg() { # $1 stem  $2 collector  $3 total_time  $4 routing  $5 method  $6 ksmooth  $7 eq_tol  $8 maxit
    local stem="${1:?mkcfg needs a stem}"        coll="${2:?mkcfg needs a collector}"
    local total="${3:?mkcfg needs total_time}"   routing="${4:?mkcfg needs a routing: it resolves PER COLLECTOR, so name it}"
    local method="${5:?mkcfg needs a solver method}" ksm="${6:?mkcfg needs a ksat smoothing width}"
    local eqt="${7:?mkcfg needs an eq_tol}"   maxit="${8:?mkcfg needs a max_iterations}"
    # time_integration FOLLOWS FROM the solver -- it is not an independent choice, so it is
    # derived here rather than passed in, and the config states the value the run resolves to.
    local integ=backward-euler; [ "$method" = anderson ] && integ=tr-bdf2
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" \
        -e "s|@COLLECTOR@|$coll|g" -e "s|@TOTAL@|$total|g" -e "s|@ROUTING@|$routing|g" \
        -e "s|@METHOD@|$method|g" -e "s|@KSMOOTH@|$ksm|g" -e "s|@EQ_TOL@|$eqt|g" \
        -e "s|@INTEG@|$integ|g" -e "s|@MAXIT@|$maxit|g" \
        config.yaml > "$WORK/$stem.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$stem.yaml" && { echo "ERROR: unfilled slot in $stem.yaml"; exit 1; }
    return 0
}

mkcfg_mode() { # $1 stem  $2 ramp|adaptive -- the single-arm files; no per-arm slots but the paths
    local stem="${1:?}" mode="${2:?}"
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" \
        "config_${mode}.yaml" > "$WORK/$stem.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$stem.yaml" && { echo "ERROR: unfilled slot in $stem.yaml"; exit 1; }
    return 0
}

# PETSc prints the ratio per Jacobian evaluation; take the worst.
# The two ksat smoothing widths come from the CONFIG (the mkcfg ksmooth argument), not from -wtm_ flags:
# those options-database entries are gone, and a -wtm_ nothing reads is an abort.
# -snes_test_jacobian stays on the command line because it is PETSc's own diagnostic with no config
# key. -snes_max_it does NOT: solver.max_iterations sets it (WTM.cpp), and because that uses
# set_opt_if_unset the CLI value SILENTLY WON -- the config said 10000 while the run used 1. The
# Jacobian arms now declare max_iterations: 1 and nothing overrides it (#79).
fd_ratio() { # $1 = stem
    "$WTM" "$WORK/$1.yaml" \
        -snes_test_jacobian 2>&1 | tee "$WORK/$1.fd.log" \
      | grep -oE '\|\|J - Jfd\|\|_F/\|\|J\|\|_F = [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' | sort -g | tail -1
}

echo "=== Newton solver: analytic Jacobian and its contract ==="
echo "WTM binary: $WTM"
echo
fail=0

# ---- 1. PRECONDITION: the pin actually fires on this fixture -------------------------------------
mkcfg_mode pre ramp
"$WTM" "$WORK/pre.yaml" \
    > "$WORK/pre.log" 2>&1
REM=$(awk '$1 ~ /^[0-9]+$/ && NF>=23 {s=$12} END{print s+0}' "$WORK/pre.txt" 2>/dev/null || echo 0)
if awk -v r="$REM" 'BEGIN{exit !(r > 0)}'; then
    echo "  PASS  PRECONDITION  the active set is non-empty (surface_removed = $REM > 0)"
else
    echo "  FAIL  PRECONDITION  no water pinned -- the Jacobian check below would not touch the"
    echo "        active-set tangent at all (surface_removed = $REM)"
    fail=1
fi

# ---- 2. Jacobian vs finite differences, per collector --------------------------------------------
# MODE=fixed on the arms below is load-bearing. Under the old two-boolean scheme they set
# dt_continuation: false, adaptive_dt then resolved TRUE, and the FD comparison was made at a state the
# controller chose rather than at the fixed-dt state these ratios were characterised on: active_set read 0.0736 and explicit 1.0785
# against a 1e-2 ceiling. Nothing was wrong with the Jacobian -- pinned back to fixed dt they return to
# 0.00415 and 7.36e-08. (explicit is four orders BETTER than its recorded 0.000993: the volume storage
# default, 879a188, makes the analytic Jacobian match its own residual exactly.)
for coll in active_set explicit; do
    # an absent routing key resolves per collector (measured); state the one this arm gets
    routing=continuous; [ "$coll" = explicit ] && routing=impulse
    mkcfg "j_$coll" "$coll" "2yr" "$routing" newton 0.5 0 1   # raw Jacobian: PLAIN Newton, as the bare flag gave
    R=$(fd_ratio "j_$coll")
    if [ -z "$R" ]; then
        echo "  FAIL  JACOBIAN   $coll -- no ratio produced"; fail=1
    elif awk -v r="$R" -v t="$JTOL" 'BEGIN{exit !(r+0 <= t+0)}'; then
        echo "  PASS  JACOBIAN   $coll: ||J-Jfd||/||J|| = $R  (ceiling $JTOL)"
    else
        echo "  FAIL  JACOBIAN   $coll: ||J-Jfd||/||J|| = $R  exceeds $JTOL"; fail=1
    fi
done

mkcfg j_implicit implicit "2yr" continuous newton 0.5 0 1
R=$(fd_ratio j_implicit)
WARNED=$(grep -c "NOT the Newton Jacobian" "$WORK/j_implicit.fd.log" || true)
if awk -v r="${R:-0}" 'BEGIN{exit !(r+0 > 0.1)}' && [ "$WARNED" -gt 0 ]; then
    echo "  xfail   JACOBIAN   implicit: ||J-Jfd||/||J|| = $R, and the code warns -- KNOWN HOLE"
else
    echo "  FAIL  JACOBIAN   implicit: ratio=$R warned=$WARNED"
    echo "        -> either the kink tangent is now WIRED (promote this to a real check) or the"
    echo "           warning has been dropped while the hole remains."
    fail=1
fi

# ---- 3. SAME ROOT: Newton and Anderson share the residual -----------------------------------------
# BOTH arms are pinned to FIXED dt, and that is the whole point: this arm compares SOLVERS, so anything
# else that could move the answer has to be held equal. With FSM ON, a different step sequence flips
# FillSpillMerge's discrete fill/spill decisions -- measured on this fixture, fixed vs adaptive Anderson
# at equilibrium:
#     FSM off   max|dwtd| = 9.84e-06 m      FSM on    max|dwtd| = 8.25e-02 m
# so at equilibrium the stepping is answer-neutral for the groundwater solve ALONE but not once lake
# routing is in the loop.
#
# eq_newt used to need the continuation ramp, which made this comparison confound the SOLVER with the
# STEPPING: the ramp never reached equilibrium at all on this fixture (2000 cycles, per-cycle max|dw|
# median decaying 0.494 -> 0.039 but a max stuck near 5.0 m in every window -- recurring excursions, not
# slow convergence), so SAME ROOT was measuring an equilibrium against a run that had not converged. It
# read 2.537e-01 m that way. Plain Newton no longer needs the ramp (see the CONTRACT section below),
# which is what makes matching the stepping possible:
#     Newton + continuation ramp   max|dwtd| 2.537e-01 m   (never reached equilibrium)
#     Newton plain, fixed dt       max|dwtd| 4.739e-02 m   equilibrium at cycle 466, vs eq_and's 463
mkcfg eq_and  active_set "2000yr" continuous anderson 0 1e-4 10000
mkcfg eq_newt active_set "2000yr" continuous newton   0 1e-4 10000
"$WTM" "$WORK/eq_and.yaml"                  > "$WORK/eq_and.log"  2>&1
"$WTM" "$WORK/eq_newt.yaml" > "$WORK/eq_newt.log" 2>&1
WORK="$WORK" AGREE_TOL="$AGREE_TOL" PHI="$INP/fsm_test_porosity.tif" TESTS="$(readlink -f ..)" \
  python3 - <<'PY' || fail=1
import glob, os, sys
import numpy as np, rasterio
sys.path.insert(0, os.environ["TESTS"])
import wtm_volume as VOL                  # ONE verified V(wtd); see tests/verify_wtm_volume.sh
# NOT aliased to W: this block already binds W to the work directory, and the collision made the
# helper vanish behind a str at runtime.

W, tol = os.environ["WORK"], float(os.environ["AGREE_TOL"])
def last(stem):
    try:
        f = VOL.latest_output(f"{W}/{stem}_")   # guards against a stem that is a prefix of another
    except FileNotFoundError:
        return None
    return rasterio.open(f).read(1).astype(float)
a, n = last("eq_and"), last("eq_newt")
if a is None or n is None:
    print("  FAIL  SAME ROOT  missing output"); sys.exit(1)
phi = VOL.read_band(os.environ["PHI"])
d = VOL.volume_diff(a, n, phi)   # WATER VOLUME, not head
ok = d.max() < tol
# rms AFTER the bound. It is context, not the asserted quantity -- the verdict is on max|dV| -- and
# printed before the bound it sits NEAREST it, so a reader's eye lands on the wrong number while the
# parser takes the largest. The two rules disagreeing is exactly what #118's lint flags.
print(f"  {'PASS' if ok else 'FAIL'}  SAME ROOT  Anderson vs Newton at equilibrium: "
      f"max|dV| = {d.max():.3e} m water volume (tol AGREE_TOL={tol}); rms = {np.sqrt((d**2).mean()):.3e}")
sys.exit(0 if ok else 1)
PY

# ---- 4. CONTRACT: plain Newton at FIXED dt now CONVERGES ------------------------------------------
# THIS ARM WAS INVERTED on 2026-09-04. It used to assert that plain Newton at fixed dt FAILS -- that was
# the documented contract, and the reason solver.newton.dt_continuation existed as a requirement rather
# than an option. It no longer fails.
# WHEN it stopped failing is NOT established. It was already converging at the FIRST measurement taken
# in the 2026-09-04 session, before the active-set obstacle fix (19ee097) and before the FSM delta
# stopped being scaled (69a0d0c) -- so neither of those is the cause, and an earlier commit message
# wrongly credited them. No pre-flip Newton run exists to compare against, so the change may predate
# this session entirely. Recorded as unknown rather than guessed; see task #50.
# Assert the NEW behaviour rather than delete the arm, so a regression back to needing the ramp is still
# caught. ADAPT=false remains load-bearing: without it adaptive_dt: auto resolves TRUE here and the arm
# would not be testing fixed dt at all.
mkcfg contract active_set "2yr" continuous newton 0 0 10000
# Run through an inner shell so that IT owns the child: this arm is EXPECTED to abort, and the
# reporting shell's "Aborted (core dumped)" notice then goes to the inner shell's stderr -- which is
# redirected into the log -- instead of surfacing in the suite output looking like a real crash.
if sh -c '"$0" "$1"' \
        "$WTM" "$WORK/contract.yaml" > "$WORK/contract.log" 2>&1; then
    echo "  PASS  CONTRACT/a plain Newton at FIXED dt CONVERGES (it no longer needs the ramp)"
elif grep -q "The SNES solver has not converged" "$WORK/contract.log"; then
    echo "  FAIL  CONTRACT/a plain Newton at FIXED dt no longer converges -- this REGRESSED to needing"
    echo "        solver.newton.dt_continuation. It converged unaided as of 2026-09-04."
    fail=1
else
    echo "  FAIL  CONTRACT/a plain Newton at fixed dt failed for an UNEXPECTED reason:"
    grep -m1 -E "^ERROR: |what\(\):" "$WORK/contract.log" | sed 's/^/        /'
    fail=1
fi

# 4b. the SAME configuration with adaptive stepping must SUCCEED. This is the half that makes 4a a
# statement about fixed dt rather than about Newton, and it is the positive control for 4a: if 4b also
# failed, 4a would be proving only that the fixture is hard.
mkcfg_mode contract_adapt adaptive
if sh -c '"$0" "$1"' \
        "$WTM" "$WORK/contract_adapt.yaml" > "$WORK/contract_adapt.log" 2>&1; then
    echo "  PASS  CONTRACT/b the same run with adaptive stepping CONVERGES -- the ramp is not the only"
    echo "                   way to globalise plain Newton"
else
    echo "  FAIL  CONTRACT/b plain Newton + adaptive did NOT converge, so CONTRACT/a shows only that"
    echo "                   this fixture is hard, not that fixed dt is what defeats plain Newton:"
    grep -m1 -E "^ERROR: |what\(\):" "$WORK/contract_adapt.log" | sed 's/^/        /'
    fail=1
fi

echo
if [[ $fail -eq 0 ]]; then echo "NEWTON SOLVER: ALL PASSED"; else echo "NEWTON SOLVER: FAILED" >&2; fi
exit $fail
