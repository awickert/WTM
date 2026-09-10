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
# WHY BOTH COUPLING MODES. fsm_coupling: continuous (#116) folds FSM's delivery into the step's source
# term instead of overwriting the water table between steps. That changes what "an input" means to the
# scheme, so it must be checked against the same identity rather than assumed equivalent.
#
# Usage:  tests/budget_closure/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

FSMDIR=$(readlink -f ../fsm_consistency)
[[ -f "$FSMDIR/inputs/fsm_test_t0_topography.tif" ]] || ( cd "$FSMDIR" && python3 make_inputs.py >/dev/null )
INP="$FSMDIR/inputs"
make_work budget
# Defer to the suite's coverage log when run under run_all.sh, so the aggregated matrix sees these
# arms; fall back to a per-run file so the resolution assertions still work standalone.
export WTM_COVERAGE_LOG="${WTM_COVERAGE_LOG:-$WORK/coverage.txt}"
TOL="${TOL:-1e-6}"      # relative to the run's solver recharge
PY="${PY:-python3}"
export OMP_NUM_THREADS=1

# THE CONFIGS ARE FILES NOW (#83): config.yaml (adaptive), config_fixed.yaml, config_ramp.yaml.
# THREE files because the three step modes resolve DIFFERENT KEY SETS, and which mode an arm gets is
# NOT a free choice -- it follows from solver x integrator x collector. Each file's header states the
# rule and lists the arms that run from it.
#
# THE COLLECTOR IS A BLOCK, NOT A VALUE, because three arms (d_and, d_ntu, d_pic) exist precisely to
# assert what an UNDECLARED collector resolves to. For those, mkcfg substitutes an OPTIONAL marker
# instead of the key -- the config then states, in the author's voice, that the parameter is left to
# automatic resolution and what it is expected to resolve to (#92). Writing the key would delete the
# property under test: the arm would become a copy of the arm that sets it.
mkcfg() { # $1 stem  $2 collector ("" = DELIBERATELY ABSENT)  $3 routing  $4 mode  $5 method
          #   $6 integrator  $7 storage  $8 snes_stol  $9 dt error_tol
    local stem="${1:?mkcfg needs a stem}" coll="${2-}" routing="${3:?mkcfg needs a routing}"
    local mode="${4:?mkcfg needs a step mode: it RESOLVES from solver x integrator x collector, so name
                     the one this arm gets rather than letting the file choose}"
    local method="${5:?mkcfg needs a solver method}" integ="${6-}" storage="${7-}"
    local stol="${8:?mkcfg needs a snes_stol}" dttol="${9-}"
    # An EMPTY slot is not a default -- it renders `key:` with no value, which the model then reports
    # as `null` or a stod failure. Refuse it HERE, where the arm that caused it can still be named,
    # rather than letting a half-rendered config reach the model.
    local block
    if [ -n "$coll" ]; then
        block="  collection:\n    method: $coll"
    else
        # The marker config_identity.py reads. The PARAMETER is optional, not the value: an undeclared
        # key still resolves, and the marker commits to what it should resolve TO -- which the arm own
        # WANT_COLL assertion then checks independently.
        block="# OPTIONAL: surface_water.collection.method -- resolved automatically when not declared,\n"
        block="$block# and THAT RESOLUTION IS THIS ARM'S SUBJECT. Setting the key would delete the property under\n"
        block="$block# test, turning this arm into a copy of the arm that sets it.\n"
        block="$block# Expect: \${WANT_COLL:?an arm that leaves the collector undeclared must say what it expects}"
    fi
    local src=config.yaml
    [ "$mode" = fixed ] && src=config_fixed.yaml
    [ "$mode" = ramp  ] && src=config_ramp.yaml
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$stem|g" \
        -e "s|@ROUTING@|$routing|g" -e "s|@METHOD@|$method|g" -e "s|@STOL@|$stol|g" \
        -e "s|@INTEG@|$integ|g" -e "s|@STORAGE@|$storage|g" -e "s|@DT_TOL@|$dttol|g" \
        -e "s|@COLLECTION@|$block|" \
        "$src" > "$WORK/$stem.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$stem.yaml" && { echo "ERROR: $stem.yaml has an unfilled slot"; exit 1; }
    # A key with NO VALUE means a slot was filled with "". The model then reports it as `null` (an
    # enum) or dies in stod (a number), several layers away from the arm that caused it -- which is
    # exactly how twelve arms broke at once. Catch it here, where the arm can still be named.
    #
    # THE VALUE IS WHAT IS LEFT AFTER THE COMMENT IS STRIPPED. Every slot in these files carries a
    # trailing `# PER-ARM: ...` note, so an unfilled one renders as `key:    # PER-ARM: ...` and any
    # check anchored at end-of-line sees a comment and passes. That is the first version of this guard,
    # and it missed the very bug it was written for.
    #
    # A section header legitimately has no value; it is told apart by the next content line being MORE
    # indented than it is.
    local empty
    empty=$(sed 's/#.*//' "$WORK/$stem.yaml" | awk '
        /^[[:space:]]*$/ { next }
        { ind = match($0, /[^ ]/) - 1
          if (pn && ind <= pind) printf "line %d: %s\n", pn, ptxt
          if ($0 ~ /^[[:space:]]*[a-zA-Z_]+:[[:space:]]*$/) { pn = NR; pind = ind; ptxt = $0; sub(/[[:space:]]+$/, "", ptxt) }
          else pn = 0 }
        END { if (pn) printf "line %d: %s\n", pn, ptxt }')
    if [ -n "$empty" ]; then
        echo "ERROR: $stem.yaml has a key with an EMPTY value -- a slot was filled with nothing:"
        printf '%s\n' "$empty" | sed 's/^/        /'
        exit 1
    fi
    return 0
}

fail=0
check() { # $1 = label, $2 = stem, $3.. = solver flags ; ARM_TOL overrides TOL, ARM_STOL the solve
    local label="$1" stem="$2"; shift 2
    local tol="${ARM_TOL:-$TOL}"
    # A closure assertion is only meaningful if the SOLVE is resolved tighter than the closure it
    # asserts; otherwise the arm measures solver noise. Default 1e-8 suits every arm here except the
    # Newton sub-stepping one, which carries the solve tolerance on every sub-step (see its note).
    local stol="${ARM_STOL:-1e-8}"
    mkcfg "$stem" "${COLL-implicit}" "${ROUTING:?each arm must NAME the routing it resolves to}" \
          "${MODE:?each arm must NAME its step mode: it resolves from solver x integrator x collector,
                   and RE-DERIVING that rule here is how a test stops testing the model and starts
                   testing its own copy of the policy}" \
          "${METHOD:-anderson}" "${INTEG-}" "${STORAGE-}" "$stol" "${DT_TOL-}"
    if ! WTM_COVERAGE_TAG="budget_closure/$stem" "$WTM" "$WORK/$stem.yaml" "$@" \
            > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  $label -- run failed"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1; return
    fi
    # DID THIS ARM RUN WHAT IT ASKED FOR? (#24, #72) Only axes this arm SET are checked, against the
    # fingerprint the MODEL emits after every override. Deliberately NOT a re-derivation of the auto
    # policy: that would be this script asserting its own guess, and the guess was wrong -- the first
    # version of this check "found" three defects that are all documented auto-resolutions
    # (time_integration: auto -> tr-bdf2 for anderson; adaptive_dt: auto -> off under the implicit
    # collector; fsm_coupling: auto -> impulse under the explicit collector). What it DID find is below.
    local -a want=()
    if   [ -n "${WANT_COLL:-}" ];    then want+=("collector=$WANT_COLL")        # COLL="" arms: the resolution IS the claim
    elif [ -n "${COLL-implicit}" ];  then want+=("collector=${COLL-implicit}")
    fi
    case "${INTEG:-}" in
        tr-bdf2)        want+=("integrator=tr_bdf2") ;;
        bdf2)           want+=("integrator=bdf2") ;;
        backward-euler) want+=("integrator=$([ "${STORAGE:-volume}" = secant ] && echo be_secant || echo be_volume)") ;;
    esac
    [ -n "${METHOD:-}" ]  && want+=("solver=$METHOD")
    # ROUTING and MODE replaced COUPLING= and ADAPT= when the arms started NAMING what they resolve to
    # (#83). Both are REQUIRED on every arm, so unlike the old optional vars they cannot go empty and
    # silently drop out of the fingerprint -- which is how a check stops checking.
    want+=("coupling=${ROUTING:?the fingerprint needs the routing this arm resolves to}")
    [ "${MODE:?the fingerprint needs the step mode this arm resolves to}" = adaptive ] && want+=("dtctl=adaptive")
    expect_resolved "$WTM_COVERAGE_LOG" "${want[@]}" || fail=1
    TOL="$tol" LABEL="$label" TESTS="$(readlink -f ..)" "$PY" - "$WORK/$stem.txt" <<'PY' || fail=1
import os, sys, math
tol   = float(os.environ["TOL"]); label = os.environ["LABEL"]
sys.path.insert(0, os.environ["TESTS"])
import wtm_log as LOG               # columns BY NAME; see tests/log_schema
I     = LOG.index_map(sys.argv[1])
rows  = [[float(x) for x in l.split()] for l in open(sys.argv[1])
         if l.split() and l.split()[0].isdigit() and len(l.split()) >= 18]
if len(rows) < 3:
    print(f"  FAIL  {label} -- only {len(rows)} data rows"); sys.exit(1)
# cols (1-indexed): 9 solver-side recharge scale, 17 exact_budget_residual
rech = [r[I["total_recharge_added"]] for r in rows]; res = [r[I["exact_budget_residual"]] for r in rows]
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
    local stol="${ARM_STOL:-1e-8}"
    mkcfg "$stem" "${COLL-implicit}" "${ROUTING:?each arm must NAME the routing it resolves to}" \
          "${MODE:?each arm must NAME its step mode: it resolves from solver x integrator x collector,
                   and RE-DERIVING that rule here is how a test stops testing the model and starts
                   testing its own copy of the policy}" \
          "${METHOD:-anderson}" "${INTEG-}" "${STORAGE-}" "$stol" "${DT_TOL-}"
    WTM_COVERAGE_TAG="budget_closure/$stem" "$WTM" "$WORK/$stem.yaml" "$@" > "$WORK/$stem.log" 2>&1
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
xfail_broken() { # $1 = label, $2 = stem, $3 = floor, $4.. = solver flags ; XTASK names the defect
    local label="$1" stem="$2" floor="$3"; shift 3
    local stol="${ARM_STOL:-1e-8}"
    mkcfg "$stem" "${COLL-implicit}" "${ROUTING:?each arm must NAME the routing it resolves to}" \
          "${MODE:?each arm must NAME its step mode: it resolves from solver x integrator x collector,
                   and RE-DERIVING that rule here is how a test stops testing the model and starts
                   testing its own copy of the policy}" \
          "${METHOD:-anderson}" "${INTEG-}" "${STORAGE-}" "$stol" "${DT_TOL-}"
    if ! WTM_COVERAGE_TAG="budget_closure/$stem" "$WTM" "$WORK/$stem.yaml" "$@" \
            > "$WORK/$stem.log" 2>&1; then
        echo "  FAIL  $label -- run failed"; tail -3 "$WORK/$stem.log" | sed 's/^/        /'; fail=1; return
    fi
    FLOOR="$floor" LABEL="$label" XTASK="${XTASK:-#12}" "$PY" - "$WORK/$stem.txt" <<'PYX' || fail=1
import os, sys
floor = float(os.environ["FLOOR"]); label = os.environ["LABEL"]; xtask = os.environ["XTASK"]
rows = [[float(x) for x in l.split()] for l in open(sys.argv[1])
        if l.split() and l.split()[0].isdigit() and len(l.split()) >= 18]
if len(rows) < 3:
    print(f"  FAIL  {label} -- only {len(rows)} data rows"); sys.exit(1)
cum = abs(rows[-1][16]) / (abs(rows[-1][8]) or 1.0)
still_broken = cum > floor
print(f"  {'xfail' if still_broken else 'FAIL '}   {label:<34} cumulative={cum:.2e}  "
      f"-- KNOWN DEFECT, task {xtask}" + ("" if still_broken else "  <-- NOW CLOSES: promote to check()"))
sys.exit(0 if still_broken else 1)
PYX
}

echo "=== water-budget closure (exact per-step identity; runoff_ratio 0.3, FSM on) ==="
echo "WTM binary: $WTM"
echo
# PIN THE COUPLING ON BOTH ARMS. These two blocks exist to compare the two couplings, so neither may
# take it from the default: when the default flipped to `continuous` this first block -- which set no
# COUPLING -- silently BECAME the second, and `impulse` lost its budget-closure coverage entirely. The
# tell was in the output all along, the two blocks reporting bit-identical cumulative=5.67e-09 and
# worst-per-cycle=6.94e-07. An arm that names its configuration cannot be repurposed by a default.
echo "-- impulse coupling --"
ROUTING=impulse MODE=fixed INTEG=backward-euler STORAGE=secant check "Anderson BE (secant)"       s_and
ROUTING=impulse MODE=fixed INTEG=backward-euler STORAGE=volume check "Anderson BE (volume dV)" s_vol
STORAGE=volume ROUTING=impulse MODE=fixed METHOD=picard INTEG=bdf2 check "Picard BDF2-on-V" s_pic
echo
echo "-- continuous coupling (the default; #116) --"
ROUTING=continuous MODE=fixed INTEG=backward-euler STORAGE=secant check "Anderson BE (secant)"       f_and
ROUTING=continuous MODE=fixed INTEG=backward-euler STORAGE=volume check "Anderson BE (volume dV)" f_vol
echo
# Active-set is the candidate replacement for the `implicit` collector: it is the only enforcement
# measured to give a dt-INDEPENDENT equilibrium (see SURFACE_WATER_ROUTING.md). Gate its conservation
# here so that property cannot regress while the default question is open. Note its residual is the
# LOOSEST of the schemes (~6e-7 vs 1e-8..1e-10) -- the pinned-cell exfiltration flux is transferred to
# the budget post-solve rather than being an integrated source, so it carries the SNES tolerance of
# the pin. Understand that gap before making it the default.
echo "-- active-set exfiltration constraint --"
# KNOWN GAP, now EXPLAINED (was an unverified hypothesis): active-set closes to ~5e-6 per cycle at
# this suite's, ~50x looser than every other arm. The cause is that the pinned cells'
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
# active-set WITH fsm_coupling: continuous -- closed ~50x tighter at 8e-8, and read that as evidence the
# two changes "belong together". That inference was WRONG: on this fixture the source arm HALVES the
# ponded water (160.00 -> 79.04 m, max wtd 10.0 -> 5.0 m), so the tighter residual was measured on a
# MATERIALLY DIFFERENT answer, and tighter closure of a different state says nothing about
# complementarity. That arm has since been REMOVED entirely, because the pair is now a hard error: the
# active-set obstacle is read from the water table FSM writes each step, and fsm_coupling: continuous
# suppresses exactly that write, so every lake drains (5.6986 -> 0.0000 m). The #116 arms above still
# run, under the implicit collector this fixture pins. See benchmark/scheme_bench/README.md, where
# active-set alone is shown to already remove the FSM between-step shock (ratio 0.985 -> 3.6e-13) that
# fsm_coupling: continuous exists to address.
DT_TOL=0.5 ROUTING=continuous MODE=adaptive INTEG=tr-bdf2 COLL=active_set ARM_TOL=1e-5 check "Anderson + active-set [loose tol, see note]" a_as
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
STORAGE=volume ROUTING=continuous MODE=fixed INTEG=tr-bdf2 check "TR-BDF2" s_tr
# The combination that was leaking, and the reason this arm exists: active-set puts a multiplier in
# BOTH stages, and only the step combination E = C1*E1 + E2 conserves. Same loose per-arm tolerance as
# the backward-Euler active-set arm above, and for the same reason -- the multiplier is recovered from
# the residual, so it carries the solve's tolerance, not a conservation defect.
DT_TOL=0.5 ROUTING=continuous MODE=adaptive COLL=active_set INTEG=tr-bdf2 ARM_TOL=1e-5 check "TR-BDF2 + active-set [loose tol]" tr_as
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
# RE-MEASURED under the WATER convergence metric (#61), which moved every number here; the previous
# table was taken when 88% of solves were exiting early on the head step test. 20 cycles, steps (rejects):
#
#     -wtm_dt_tol       TR-BDF2                  BDF2-on-V
#     0.5 (default)     20  (0)  <-- DEGENERATE   83  (2)
#     0.1               20  (0)  <-- DEGENERATE  128 (11)
#     0.05              20  (0)  <-- DEGENERATE  109 (13)
#     0.02              20  (0)  <-- DEGENERATE  181 (23)
#     0.01              40  (1)                  210 (31)
#     0.005             ABORTS                   293 (44)
#
# So BDF2-on-V subdivides and rejects at the default tolerance and is left alone, while TR-BDF2 has a
# ONE-VALUE WINDOW: degenerate at every tolerance from 0.02 up, and at 0.005 the target is below what
# the solve can actually deliver, so the controller shrinks to dtc_max_retries and the run aborts.
# 0.01 is the only value that both completes and makes the controller do anything, so that is what the
# arm uses. THIS IS FRAGILE BY CONSTRUCTION -- a one-value window has no margin on either side, and any
# change to the estimator or the convergence metric can close it. Re-measure this table, do not nudge
# the tolerance until the run stops failing.
#
# WHY 0.005 STOPPED WORKING, since "the test needed a looser tolerance" is exactly what a papered-over
# regression looks like: it never worked, it only appeared to. Under the old head-judged step test the
# solves stopped early, which made the error estimate small, which let the controller believe it had
# met 0.005. Judged in water the estimate is honest, 0.005 is below the achievable floor for this
# fixture, and the controller correctly refuses. The arm is measuring the same thing as before against
# a target it can actually reach.
#
# If either arm ever prints "(fixed would be N)" with its own step count equal to N and no rejects,
# it has gone vacuous again and the tolerance must be re-measured, not the arm deleted.
#
# THE OLD NON-MONOTONICITY IS NOT REPRODUCED. It was recorded here as unexplained -- TR-BDF2 giving 68
# steps at 0.02 but 57 at the tighter 0.005 -- and under the water metric the sequence above is
# monotone non-decreasing instead. Those numbers were measured under head-judged convergence, so at
# least part of that backwards behaviour was an artifact of solves stopping at a tolerance-dependent
# point rather than at the solution. Not claimed as fully explained; recorded as no longer visible.
echo "-- adaptive dt (controller must not resize until accounting is done) --"
ROUTING=continuous MODE=adaptive COLL=active_set INTEG=tr-bdf2 DT_TOL=0.01 ARM_TOL=1e-5 check "TR-BDF2 + active-set, adaptive" tr_as_ad
DT_TOL=0.5 ROUTING=continuous MODE=adaptive COLL=active_set INTEG=bdf2 ARM_TOL=1e-5 check "BDF2-on-V + active-set, adaptive" bdf2v_ad
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
DT_TOL=0.5 ROUTING=continuous MODE=adaptive INTEG=tr-bdf2 COLL=active_set ARM_TOL=1e-5 check "Anderson x active_set"      c_as 
STORAGE=volume ROUTING=continuous MODE=fixed INTEG=tr-bdf2 COLL=implicit                check "Anderson x implicit"        c_im 
DT_TOL=0.5 ROUTING=continuous MODE=adaptive INTEG=tr-bdf2 COLL=off                     check "Anderson x off"             c_off
DT_TOL=0.5 ROUTING=impulse MODE=adaptive INTEG=tr-bdf2 COLL=explicit                check "Anderson x explicit"        c_ex 
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
DT_TOL=0.5 ROUTING=impulse MODE=adaptive COLL=explicit METHOD=picard INTEG=bdf2 check "Picard x explicit" c_pex
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
DT_TOL=0.5 ROUTING=continuous MODE=adaptive INTEG=tr-bdf2 COLL="" WANT_COLL=active_set ARM_TOL=1e-5 check "Anderson, unset -> active_set"       d_and
# Newton's per-cycle residual is looser than Anderson's on the same collector because
# -wtm_dt_continuation SUB-STEPS, and the active-set multiplier carries the solve tolerance on every
# sub-step. TOLERANCE-LIMITED, verified by scaling the solve. RE-MEASURED 2026-09-04, after the FSM
# delta stopped being scaled (69a0d0c) and the estimator stopped counting it as truncation error
# (e8d568b):
#     snes_stol   cumulative   worst-per-cycle        (was, before those two)
#     1e-8         5.733e-08     1.780e-04            3.097e-07   1.279e-05
#     1e-10        6.000e-09     1.277e-05            1.588e-07   2.078e-06
#     1e-12        6.000e-09     1.277e-05  <- floors 1.588e-07   2.078e-06
# The cumulative residual IMPROVED ~5x; the per-cycle FLOOR moved ~6x the other way (2.078e-06 ->
# 1.277e-05) and is a real change, not noise -- worth knowing if it drifts further. It is still 8x
# inside ARM_TOL. What broke this arm is that at snes_stol 1e-8 the solver noise (1.780e-04) now
# EXCEEDS the closure being asserted (1e-4), so the arm was measuring the solver, not the budget.
# Resolve the solve past the assertion instead of loosening the assertion.
ROUTING=continuous MODE=ramp INTEG=backward-euler COLL="" WANT_COLL=active_set METHOD=newton ARM_TOL=1e-4 ARM_STOL=1e-10 check "Newton, unset -> active_set [tight solve, see note]" d_ntu
DT_TOL=0.5 ROUTING=impulse MODE=adaptive COLL="" WANT_COLL=explicit METHOD=picard INTEG=bdf2 check "Picard, unset -> explicit" d_pic
# THE COUPLING IS WHAT BREAKS THIS ARM, and it is worth two arms rather than one. `implicit` closes
# perfectly well under impulse; under continuous it does not. Measured at snes_stol 1e-10 (past the
# solver floor, so this is the model and not the solve):
#     impulse    x implicit   cumulative 2.491e-07   worst-per-cycle 3.521e-07   <- closes
#     continuous x implicit   cumulative 1.222e-06   worst-per-cycle 2.244e-05   <- does not
# That completes a pattern the code already half-records: `continuous` composes with active_set (the
# pairing #40 built) and with nothing else. It is REFUSED with explicit (does not converge, #44) and
# with implicit under adaptive (the siphon's error grows as dt shrinks); this is the third face of the
# same incompatibility, and the only one that fails QUIETLY -- the budget simply stops closing.
# So: keep the closure assertion on the coupling that closes, and hold the broken pairing as an
# EXPECTED failure so it keeps a regression test instead of vanishing from the suite.
ROUTING=impulse MODE=ramp INTEG=backward-euler COLL=implicit METHOD=newton check "Newton + continuation x implicit (impulse)" d_nt
ROUTING=continuous MODE=ramp INTEG=backward-euler COLL=implicit METHOD=newton XTASK="#48 (continuous composes only with active_set)" \
    xfail_broken "Newton + continuation x implicit (continuous)" d_ntc 2e-6
# Pin WHICH collector each unset run actually resolved to. The Picard downgrade prints a NOTE; the
# other two must NOT print it, or they have silently stopped testing the active-set default.
#
# THE PATTERN WENT STALE ONCE ALREADY. It read "default resolves to \`explicit\`"; the model has said
# "surface_water.collection.method defaults to \`explicit\` on the Picard solver" since d437dee moved
# the message, so the grep matched NOTHING and reported the note ABSENT on the one arm that must
# print it. A grep against another program's prose is a fragile joint, and this one failed in the
# quiet direction -- "not found" and "not printed" look identical.
#
# So the pattern is now the STABLE part of the sentence (key + verb), and a stale pattern is made
# distinguishable from a real absence: the Picard arm MUST match, and if it does not, the check says
# the pattern is the suspect rather than blaming the model.
NOTE_RE="surface_water.collection.method defaults to"
if ! grep -q "$NOTE_RE" "$WORK/d_pic.log"; then
    echo "  FAIL  RESOLUTION  the NOTE pattern matched nothing even on the Picard arm, which prints it"
    echo "                    by construction -- so the PATTERN is stale, not the model. It looks for:"
    echo "                      $NOTE_RE"
    echo "                    and the model actually said:"
    grep -iE "^NOTE:" "$WORK/d_pic.log" | sed 's/^/                      /' | head -2
    fail=1
fi
for arm in d_and:absent d_ntu:absent d_pic:present; do
    stem="${arm%%:*}"; want="${arm##*:}"
    if grep -q "$NOTE_RE" "$WORK/$stem.log"; then got=present; else got=absent; fi
    if [[ "$got" == "$want" ]]; then
        echo "  PASS  RESOLUTION  $stem: Picard-downgrade NOTE $got (expected $want)"
    else
        echo "  FAIL  RESOLUTION  $stem: Picard-downgrade NOTE $got (expected $want)"; fail=1
    fi
done
echo

if [[ $fail -eq 0 ]]; then echo "BUDGET CLOSURE: ALL PASSED"; else echo "BUDGET CLOSURE: FAILED" >&2; fi
exit $fail
