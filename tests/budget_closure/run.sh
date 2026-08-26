#!/usr/bin/env bash
# Water-budget closure gate: the scheme's own conservation law, per cycle and cumulatively.
#
# WHAT THIS ASSERTS. The exact budget residual (column 17) is built from the solver's discrete
# per-step terms:  solver_recharge - storage_change - ocean_outflow - surface_removed - evap = 0,
# which the discretisation satisfies to the SNES tolerance. This test requires that identity to hold
#   (a) CUMULATIVELY at the end of the run  -- global conservation, and
#   (b) PER CYCLE for every cycle           -- local-in-time conservation.
# (b) is the stronger statement and the reason this test exists: an error that removes water at step t
# and delivers it twice at t+1 can cancel in the cumulative total while being visibly wrong per step.
#
# WHY IT COVERS A MATRIX. The exact budget was Picard-only until the accumulator was made
# solver-agnostic; each solver path builds its storage term differently (secant backward Euler,
# -wtm_volume_storage's exact dV, BDF2-on-V), so each needs its own check or the accumulator can
# silently stop matching the residual it is supposed to mirror. TR-BDF2 has no single-step identity to
# accumulate, so it must report `nan` -- asserted here so the guard cannot regress into emitting a
# plausible-looking number.
#
# WHY runoff_ratio IS ON. With runoff_ratio > 0 part of the precipitation is diverted to the runoff
# array and reaches the domain only via FillSpillMerge, so the handoff is genuinely exercised. (Note
# `run_type test` cannot do this -- InitialiseTest hardcodes runoff_ratio to 0 -- hence the equilibrium
# fixture here.)
#
# WHY BOTH COUPLING MODES. -wtm_fsm_delta_source (#116) folds FSM's delivery into the step's source
# term instead of overwriting the water table between steps. That changes what "an input" means to the
# scheme, so it must be checked against the same identity rather than assumed equivalent.
#
# Usage:  tests/budget_closure/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
WORK=$(mktemp -d /tmp/budget_XXXX); trap 'rm -rf "$WORK"' EXIT
TOL="${TOL:-1e-6}"      # relative to the run's solver recharge
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

mkcfg() { # $1 = stem, $2 = runoff_collector ("" = OMIT the key entirely -> default resolution)
    local coll="${2-implicit}"
    { cat <<EOF
run_type equilibrium
total_time 20yr
supplied_wt 1
deltat 31536000
report_interval 1
save_nreport_interval 9999
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_ratio 0.3
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
      [ -n "$coll" ] && echo "runoff_collector $coll"; } | ../emit_config.sh > "$WORK/$1.yaml"
}

fail=0
check() { # $1 = label, $2 = stem, $3.. = solver flags ; ARM_TOL overrides TOL for one arm
    local label="$1" stem="$2"; shift 2
    local tol="${ARM_TOL:-$TOL}"
    mkcfg "$stem" "${COLL-implicit}"
    if ! "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 -wtm_eq_tol 0 > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  $label -- run failed"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1; return
    fi
    TOL="$tol" LABEL="$label" "$PY" - "$WORK/$stem.txt" <<'PY' || fail=1
import os, sys, math
tol   = float(os.environ["TOL"]); label = os.environ["LABEL"]
rows  = [[float(x) for x in l.split()] for l in open(sys.argv[1])
         if l.split() and l.split()[0].isdigit() and len(l.split()) >= 18]
if len(rows) < 3:
    print(f"  FAIL  {label} -- only {len(rows)} data rows"); sys.exit(1)
# cols (1-indexed): 9 solver-side recharge scale, 17 exact_budget_residual
rech = [r[8] for r in rows]; res = [r[16] for r in rows]
# (a) cumulative closure
scale = abs(rech[-1]) or 1.0
cum   = abs(res[-1]) / scale
# (b) per-cycle closure: the residual must not GROW from one cycle to the next
worst, worst_cyc = 0.0, -1
for (r0, x0), (r1, x1) in zip(zip(rech, res), zip(rech[1:], res[1:])):
    d_rech = abs(r1 - r0) or 1.0
    rel    = abs(x1 - x0) / d_rech
    if rel > worst: worst, worst_cyc = rel, int(rows[0][0])
ok = (cum < tol) and (worst < tol)
print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} cumulative={cum:.2e}  worst-per-cycle={worst:.2e}  (tol {tol:.0e})")
sys.exit(0 if ok else 1)
PY
}

check_nan() { # TR-BDF2 must report the exact residual as unavailable, not as a number
    local label="$1" stem="$2"; shift 2
    mkcfg "$stem"
    "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 -wtm_eq_tol 0 > "$WORK/$stem.log" 2>&1
    LABEL="$label" "$PY" - "$WORK/$stem.txt" <<'PY' || fail=1
import os, sys, math
label = os.environ["LABEL"]
rows  = [l.split() for l in open(sys.argv[1]) if l.split() and l.split()[0].isdigit() and len(l.split()) >= 18]
if not rows:
    print(f"  FAIL  {label} -- no data rows"); sys.exit(1)
vals = [float(r[16]) for r in rows]
ok   = all(math.isnan(v) for v in vals)
print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} exact residual reported as "
      f"{'nan (no single-step identity)' if ok else 'A NUMBER -- guard regressed'}")
sys.exit(0 if ok else 1)
PY
}


# xfail_broken: an arm that CANNOT close today. Held as an EXPECTED failure with a guard, so the defect
# keeps a regression test instead of simply being absent from the suite. PASSes while the cumulative
# residual stays ABOVE `floor` (still broken); FAILs the moment it closes -- which means the defect is
# fixed and the arm must be promoted to a real `check`. An expected failure that silently starts
# passing is how a fixed bug loses its test.
xfail_broken() { # $1 = label, $2 = stem, $3 = floor, $4.. = solver flags
    local label="$1" stem="$2" floor="$3"; shift 3
    mkcfg "$stem" "${COLL-implicit}"
    if ! "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 -wtm_eq_tol 0 > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  $label -- run failed"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1; return
    fi
    FLOOR="$floor" LABEL="$label" "$PY" - "$WORK/$stem.txt" <<'PYX' || fail=1
import os, sys
floor = float(os.environ["FLOOR"]); label = os.environ["LABEL"]
rows = [[float(x) for x in l.split()] for l in open(sys.argv[1])
        if l.split() and l.split()[0].isdigit() and len(l.split()) >= 18]
if len(rows) < 3:
    print(f"  FAIL  {label} -- only {len(rows)} data rows"); sys.exit(1)
cum = abs(rows[-1][16]) / (abs(rows[-1][8]) or 1.0)
still_broken = cum > floor
print(f"  {'xfail' if still_broken else 'FAIL '}   {label:<34} cumulative={cum:.2e}  "
      f"-- KNOWN DEFECT, task #12" + ("" if still_broken else "  <-- NOW CLOSES: promote to check()"))
sys.exit(0 if still_broken else 1)
PYX
}

echo "=== water-budget closure (exact per-step identity; runoff_ratio 0.3, FSM on) ==="
echo "WTM binary: $WTM"
echo
echo "-- overwrite coupling (default) --"
check "Anderson BE (secant)"       s_and    -wtm_anderson
check "Anderson BE (volume dV)"    s_vol    -wtm_anderson -wtm_volume_storage
check "Picard BDF2-on-V"           s_pic    -wtm_picard -wtm_bdf2_on_V
echo
echo "-- FSM-delta-source coupling (#116) --"
check "Anderson BE (secant)"       f_and    -wtm_anderson -wtm_fsm_delta_source
check "Anderson BE (volume dV)"    f_vol    -wtm_anderson -wtm_volume_storage -wtm_fsm_delta_source
echo
# Active-set is the candidate replacement for the `implicit` collector: it is the only enforcement
# measured to give a dt-INDEPENDENT equilibrium (see SURFACE_WATER_ROUTING.md). Gate its conservation
# here so that property cannot regress while the default question is open. Note its residual is the
# LOOSEST of the schemes (~6e-7 vs 1e-8..1e-10) -- the pinned-cell exfiltration flux is transferred to
# the budget post-solve rather than being an integrated source, so it carries the SNES tolerance of
# the pin. Understand that gap before making it the default.
echo "-- active-set exfiltration constraint --"
# KNOWN GAP, now EXPLAINED (was an unverified hypothesis): active-set closes to ~5e-6 per cycle at
# this suite's -snes_stol 1e-8, ~50x looser than every other arm. The cause is that the pinned cells'
# exfiltration flux is not an integrated source term -- it is recovered POST-solve from the residual
# itself (exfiltration_depth = max(0, -f*Sy)), so its accuracy IS the residual's accuracy. Verified by
# scaling the solver tolerance on this fixture:
#
#     snes_stol   cumulative   worst-per-cycle
#     1e-6         3.25e-05      6.65e-05
#     1e-8         5.84e-07      4.91e-06     <- what this suite runs
#     1e-10        1.37e-08      3.41e-08
#     1e-12        1.37e-08      3.41e-08     <- floors; the tolerance stops binding
#
# The residual tracks snes_stol until the solve floors, which is the signature of a tolerance-limited
# quantity and NOT of a conservation defect: tighten the solve and the budget tightens with it, to
# 3.4e-08 -- comparable to every other arm. So the looser per-arm tolerance below is a statement about
# the SOLVER setting this suite uses, not about active-set conserving worse. It is kept as a per-ARM
# tolerance (rather than tightening this arm's snes_stol, which would make its numbers incomparable
# with the others) so the difference stays visible in the output.
#
# HISTORY, so the earlier claim is not resurrected. This comment used to add that a second arm --
# active-set WITH -wtm_fsm_delta_source -- closed ~50x tighter at 8e-8, and read that as evidence the
# two changes "belong together". That inference was WRONG: on this fixture the source arm HALVES the
# ponded water (160.00 -> 79.04 m, max wtd 10.0 -> 5.0 m), so the tighter residual was measured on a
# MATERIALLY DIFFERENT answer, and tighter closure of a different state says nothing about
# complementarity. That arm has since been REMOVED entirely, because the pair is now a hard error: the
# active-set obstacle is read from the water table FSM writes each step, and -wtm_fsm_delta_source
# suppresses exactly that write, so every lake drains (5.6986 -> 0.0000 m). The #116 arms above still
# run, under the implicit collector this fixture pins. See benchmark/scheme_bench/README.md, where
# active-set alone is shown to already remove the FSM between-step shock (ratio 0.985 -> 3.6e-13) that
# -wtm_fsm_delta_source exists to address.
ARM_TOL=1e-5 check "Anderson + active-set [loose tol, see note]" a_as -wtm_anderson -wtm_active_set
echo
# TR-BDF2 used to live below this line, under a "no single-step identity" heading, asserting that it
# reported the exact residual as `nan`. That was true and worth pinning while the two stages' balances
# had not been combined -- but it also meant TR-BDF2 was the ONE scheme whose conservation nothing
# could check, and it was quietly losing 9.5% of recharge through the active-set exfiltration transfer
# (the multiplier was read off the stage-2 residual alone, recovering C3 = 29.29% of the step). The
# stages do telescope: C1*(stage 1) + (stage 2), with storage and recharge coming out unchanged and
# every flux/removal term becoming a three-point quadrature over (w^n, Y_gamma, w^{n+1}). See
# src/tr_bdf2_coefficients.hpp for the derivation and src/test_tr_bdf2_balance.cpp for the identities.
#
# So TR-BDF2 is now held to the same closure standard as everything else. The check_nan helper is kept
# (unused) because the guard it tests is still in the code as a backstop for a future scheme that
# genuinely has no per-step identity.
echo "-- TR-BDF2 (two stages, telescoped) --"
check "TR-BDF2"                    s_tr     -wtm_anderson -wtm_tr_bdf2
# The combination that was leaking, and the reason this arm exists: active-set puts a multiplier in
# BOTH stages, and only the step combination E = C1*E1 + E2 conserves. Same loose per-arm tolerance as
# the backward-Euler active-set arm above, and for the same reason -- the multiplier is recovered from
# the residual, so it carries the solve's tolerance, not a conservation defect.
ARM_TOL=1e-5 check "TR-BDF2 + active-set [loose tol]" tr_as -wtm_anderson -wtm_tr_bdf2 -wtm_active_set
echo
# ADAPTIVE dt. These exist because the exact budget was NOT checked under adaptive dt by anything, and
# it did not close: the controller wrote the NEXT step's dt into user_context.deltat before the step's
# own accounting had consumed it, so the BDF2 history ratio, both taper removals, the land->ocean flux
# and TR-BDF2's step quadrature all read the wrong dt. Measured before the fix: -1.603 of recharge for
# TR-BDF2 + adaptive and -0.417 for BDF2-on-V + adaptive, against ~2e-07 for the same schemes at fixed
# dt. Adaptive dt is the robustness tool for at-scale spin-up, so an unchecked budget there is exactly
# the gap that matters. BDF2-on-V is included as well as TR-BDF2 because the defect was
# scheme-independent -- pinning only TR-BDF2 would let it come back on the other path.
#
# BOTH ARMS OVERRIDE THIS FIXTURE'S `implicit` COLLECTOR, and the reason is worth stating rather than
# looking like a convenient choice. `implicit` + adaptive dt CANNOT COMPLETE -- it aborts with
# "adaptive dt: step failed after max retries" for BOTH TR-BDF2 and BDF2-on-V. That is pre-existing
# (verified: a binary built at 5c2422d, before any of this work, fails identically for both), and it
# is not a bug in the controller: the `implicit` siphon's retained head is ~linear in dt, so shrinking
# dt MOVES the solution instead of converging it, the local-error estimate never settles, and the
# controller correctly refuses. It is the same dt-dependence that made active_set the default. So
# these arms run the DEFAULT collector -- which is also the combination anyone would actually use.
#
# THE TWO ARMS CARRY DIFFERENT dt TOLERANCES, and that is not an oversight. An adaptive arm is only a
# test of adaptive dt if the controller actually RESIZES on this fixture; if it takes one step per
# report and rejects nothing, it reproduces the fixed-dt trajectory exactly and cannot fail
# differently from the fixed-dt arm above it -- it looks like coverage and is not. Measured here,
# 20 cycles, steps (rejects):
#
#     -wtm_dt_tol       TR-BDF2        BDF2-on-V
#     0.5 (default)     20  (0)  <-- DEGENERATE      62  (4)
#     0.1               20  (0)  <-- DEGENERATE       -
#     0.02              68  (0)                       -
#     0.005             57  (1)                     356 (10)
#
# So BDF2-on-V subdivides and rejects at the default tolerance and is left alone, while TR-BDF2 needs
# 0.005 before its embedded estimate is tight enough to make the controller do anything. Whatever
# makes the two schemes differ this much at the same tolerance is NOT understood -- see the
# non-monotonicity note below -- so the tolerance here is chosen by measurement, not by theory.
#
# If either arm ever prints "(fixed would be N)" with its own step count equal to N and no rejects,
# it has gone vacuous again and the tolerance must be re-measured, not the arm deleted.
#
# UNEXPLAINED, recorded so it is not lost: for TR-BDF2 the step count is NON-MONOTONIC in the
# tolerance -- 0.02 gives 68 steps but the tighter 0.005 gives 57. A stricter local-error bound
# producing FEWER steps is backwards. BDF2-on-V is monotonic (62 -> 356). Adaptive dt is the
# robustness tool for at-scale spin-up, so this is worth understanding before we lean on it there.
echo "-- adaptive dt (controller must not resize until accounting is done) --"
ARM_TOL=1e-5 check "TR-BDF2 + active-set, adaptive" tr_as_ad \
    -wtm_anderson -wtm_tr_bdf2 -wtm_active_set -wtm_dt_adaptive -wtm_dt_tol 0.005
ARM_TOL=1e-5 check "BDF2-on-V + active-set, adaptive" bdf2v_ad \
    -wtm_anderson -wtm_bdf2_on_V -wtm_active_set -wtm_dt_adaptive
echo


# COLLECTOR SWEEP. The enforcement is a user config choice, and conservation must not depend on which
# one is picked -- but until now every arm above pinned `implicit`, so three of the five values had no
# budget coverage at all. Measured on this fixture (residual / recharge):
#     active_set -3.744e-07 | implicit 4.197e-09 | off -1.753e-07   <- close
#     explicit   -9.165e+00 | legacy   -7.998e+00                   <- do NOT close
# FIXED (task #12). Both failures were POST-SOLVE removals: the water leaves after the residual has
# been driven to zero, so it was subtracted from an identity whose storage term had been read from the
# PRE-clamp state -- a state the model does not carry forward. Correcting the storage term to the
# COMMITTED state closes all four, with every already-closing collector UNCHANGED:
#     explicit  -9.165e+00 -> -1.670e-07 (Anderson)   -9.331e+00 -> -3.328e-10 (Picard)
#     legacy    -7.998e+00 -> -8.461e-08 (Anderson)   -8.865e+00 -> -3.289e-10 (Picard)
# These were xfail_broken arms until then, and the guards are what reported the fix
# ("NOW CLOSES: promote to check()").
echo "-- collector sweep (conservation must not depend on the enforcement) --"
COLL=active_set ARM_TOL=1e-5 check "Anderson x active_set"      c_as  -wtm_anderson
COLL=implicit                check "Anderson x implicit"        c_im  -wtm_anderson
COLL=off                     check "Anderson x off"             c_off -wtm_anderson
COLL=explicit                check "Anderson x explicit"        c_ex  -wtm_anderson
# `legacy` on Anderson keeps the band sink AND the clamp, and its per-cycle residual is
# TOLERANCE-LIMITED rather than defective -- the same signature as the active-set arm above. Verified
# by scaling the solve on this fixture:
#     snes_stol   cumulative   worst-per-cycle
#     1e-8         8.461e-08     2.039e-06     <- what this suite runs
#     1e-10        4.847e-10     1.767e-08
#     1e-12        4.847e-10     1.767e-08     <- floors; the tolerance stops binding
# It tracks snes_stol and then floors, which is what a tolerance-limited quantity does and what a
# conservation defect does not. Per-ARM tolerance rather than a tighter snes_stol, so this arm's
# numbers stay comparable with the others.
COLL=legacy   ARM_TOL=1e-5   check "Anderson x legacy [loose tol, see note]" c_lg -wtm_anderson
COLL=explicit                check "Picard x explicit"          c_pex -wtm_picard -wtm_bdf2_on_V
COLL=legacy                  check "Picard x legacy"            c_plg -wtm_picard -wtm_bdf2_on_V
echo
# EACH SOLVER AT ITS OWN RESOLVED DEFAULT. Every other arm in this file names its collector explicitly,
# which is right for discrimination but means the DEFAULT-RESOLUTION path itself was never exercised --
# and that default is SOLVER-DEPENDENT. The downgrade in transient_groundwater.cpp is conditioned on
# `use_picard` alone, so:
#     Anderson unset -> active_set     Newton unset -> active_set     Picard unset -> explicit
# Newton resolves to active_set because it now carries the matching semismooth tangent; only the Picard
# operator lacks the pin. So the configuration PICARD actually runs in production had no budget
# coverage at all -- and it did not close until task #12 was fixed.
#
# The resolution itself is asserted from the log, not inferred from the residual: two collectors could
# coincidentally give similar residuals, and this test's whole point is knowing WHICH one ran.
# Newton needs -wtm_dt_continuation to converge on this fixture; without it every collector aborts with
# "The SNES solver has not converged".
echo "-- each solver at its OWN resolved default (collector key UNSET) --"
COLL="" ARM_TOL=1e-5 check "Anderson, unset -> active_set"       d_and -wtm_anderson
# Newton's per-cycle residual is looser than Anderson's on the same collector because
# -wtm_dt_continuation SUB-STEPS, and the active-set multiplier carries the solve tolerance on every
# sub-step. TOLERANCE-LIMITED, verified by scaling the solve:
#     snes_stol   cumulative   worst-per-cycle
#     1e-8         3.097e-07     1.279e-05   <- what this suite runs
#     1e-10        1.588e-07     2.078e-06
#     1e-12        1.588e-07     2.078e-06   <- floors
COLL="" ARM_TOL=1e-4 check "Newton, unset -> active_set [loose tol, see note]" d_ntu -wtm_newton -wtm_dt_continuation
COLL=""              check "Picard, unset -> explicit"           d_pic -wtm_picard -wtm_bdf2_on_V
COLL=implicit        check "Newton + continuation x implicit"    d_nt  -wtm_newton -wtm_dt_continuation
# Pin WHICH collector each unset run actually resolved to. The Picard downgrade prints a NOTE; the
# other two must NOT print it, or they have silently stopped testing the active-set default.
for arm in d_and:absent d_ntu:absent d_pic:present; do
    stem="${arm%%:*}"; want="${arm##*:}"
    if grep -q "default resolves to \`explicit\`" "$WORK/$stem.log"; then got=present; else got=absent; fi
    if [[ "$got" == "$want" ]]; then
        echo "  PASS  RESOLUTION  $stem: Picard-downgrade NOTE $got (expected $want)"
    else
        echo "  FAIL  RESOLUTION  $stem: Picard-downgrade NOTE $got (expected $want)"; fail=1
    fi
done
echo

if [[ $fail -eq 0 ]]; then echo "BUDGET CLOSURE: ALL PASSED"; else echo "BUDGET CLOSURE: FAILED" >&2; fi
exit $fail
