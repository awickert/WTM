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
#   4. CONTRACT      plain -wtm_newton does NOT converge on these fixtures -- it needs
#                    -wtm_dt_continuation. Pinned so the requirement is recorded rather than folklore,
#                    and so the day Newton becomes robust enough to drop it, this test says so.
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
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/newton_XXXX); trap 'rm -rf "$WORK"' EXIT
JTOL="${JTOL:-1e-2}"      # ||J-Jfd||/||J|| ceiling; the piecewise kink keeps it well above 1e-8
AGREE_TOL="${AGREE_TOL:-0.05}"   # metres; same band tests/recharge_consistency uses cross-scheme
export OMP_NUM_THREADS=1

mkcfg() { # $1 = stem, $2 = collector, $3 = total_time
    { cat <<EOF
run_type equilibrium
total_time $3
supplied_wt 1
deltat 31536000
report_interval 1
save_nreport_interval 1
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_ratio 0
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
eq_tol ${EQ_TOL:-0}
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
      [ -n "$2" ] && echo "runoff_collector $2"; } | ../emit_config.sh > "$WORK/$1.yaml"
}

# PETSc prints the ratio per Jacobian evaluation; take the worst.
fd_ratio() { # $1 = stem
    "$WTM" "$WORK/$1.yaml" -wtm_newton \
        -wtm_ksat_surface_smoothing_width 0.5 -wtm_ksat_soilbottom_smoothing_width 0.5 \
        -snes_test_jacobian -snes_max_it 1 2>&1 | tee "$WORK/$1.fd.log" \
      | grep -oE '\|\|J - Jfd\|\|_F/\|\|J\|\|_F = [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' | sort -g | tail -1
}

echo "=== Newton solver: analytic Jacobian and its contract ==="
echo "WTM binary: $WTM"
echo
fail=0

# ---- 1. PRECONDITION: the pin actually fires on this fixture -------------------------------------
mkcfg pre active_set "2yr"
"$WTM" "$WORK/pre.yaml" -wtm_newton -wtm_dt_continuation -snes_stol 1e-10 \
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
for coll in active_set explicit; do
    mkcfg "j_$coll" "$coll" "2yr"
    R=$(fd_ratio "j_$coll")
    if [ -z "$R" ]; then
        echo "  FAIL  JACOBIAN   $coll -- no ratio produced"; fail=1
    elif awk -v r="$R" -v t="$JTOL" 'BEGIN{exit !(r+0 <= t+0)}'; then
        echo "  PASS  JACOBIAN   $coll: ||J-Jfd||/||J|| = $R  (ceiling $JTOL)"
    else
        echo "  FAIL  JACOBIAN   $coll: ||J-Jfd||/||J|| = $R  exceeds $JTOL"; fail=1
    fi
done

mkcfg j_implicit implicit "2yr"
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
EQ_TOL=1e-4 mkcfg eq_and  active_set "2000yr"
EQ_TOL=1e-4 mkcfg eq_newt active_set "2000yr"
"$WTM" "$WORK/eq_and.yaml"  -wtm_anderson                 -snes_stol 1e-10 > "$WORK/eq_and.log"  2>&1
"$WTM" "$WORK/eq_newt.yaml" -wtm_newton -wtm_dt_continuation -snes_stol 1e-10 > "$WORK/eq_newt.log" 2>&1
WORK="$WORK" AGREE_TOL="$AGREE_TOL" python3 - <<'PY' || fail=1
import glob, os, sys
import numpy as np, rasterio
W, tol = os.environ["WORK"], float(os.environ["AGREE_TOL"])
def last(stem):
    fs = sorted(glob.glob(f"{W}/{stem}_[0-9]" + "[0-9]"*8 + "_*yr.tif"))
    return rasterio.open(fs[-1]).read(1).astype(float) if fs else None
a, n = last("eq_and"), last("eq_newt")
if a is None or n is None:
    print("  FAIL  SAME ROOT  missing output"); sys.exit(1)
d = np.abs(a - n)
ok = d.max() < tol
print(f"  {'PASS' if ok else 'FAIL'}  SAME ROOT  Anderson vs Newton at equilibrium: "
      f"max|dwtd| = {d.max():.3e} m, rms = {np.sqrt((d**2).mean()):.3e} m  (tol {tol})")
sys.exit(0 if ok else 1)
PY

# ---- 4. CONTRACT: Newton needs dt-continuation ----------------------------------------------------
mkcfg contract active_set "2yr"
# Run through an inner shell so that IT owns the child: this arm is EXPECTED to abort, and the
# reporting shell's "Aborted (core dumped)" notice then goes to the inner shell's stderr -- which is
# redirected into the log -- instead of surfacing in the suite output looking like a real crash.
if sh -c '"$0" "$1" -wtm_newton -snes_stol 1e-10' \
        "$WTM" "$WORK/contract.yaml" > "$WORK/contract.log" 2>&1; then
    echo "  FAIL  CONTRACT   plain -wtm_newton CONVERGED -- it no longer needs -wtm_dt_continuation."
    echo "        That is good news; update this arm and the docs that say otherwise."
    fail=1
elif grep -q "The SNES solver has not converged" "$WORK/contract.log"; then
    echo "  PASS  CONTRACT   plain -wtm_newton fails as documented; -wtm_dt_continuation is required"
else
    echo "  FAIL  CONTRACT   plain -wtm_newton failed for an UNEXPECTED reason:"
    grep -m1 "what():" "$WORK/contract.log" | sed 's/^/        /'
    fail=1
fi

echo
if [[ $fail -eq 0 ]]; then echo "NEWTON SOLVER: ALL PASSED"; else echo "NEWTON SOLVER: FAILED" >&2; fi
exit $fail
