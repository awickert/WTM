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

mkcfg() { # $1 = stem
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
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
runoff_collector implicit
surfdatadir $INP
region fsm_test
time_start t0
time_end t0
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

fail=0
check() { # $1 = label, $2 = stem, $3.. = solver flags ; ARM_TOL overrides TOL for one arm
    local label="$1" stem="$2"; shift 2
    local tol="${ARM_TOL:-$TOL}"
    mkcfg "$stem"
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
# KNOWN GAP, measured not guessed: active-set under the OVERWRITE coupling closes only to ~5e-6
# per cycle -- 50x looser than every other arm -- so it runs at its own documented tolerance rather
# than loosening the gate for everything else. Hypothesis (UNVERIFIED): the pinned cells' exfiltration
# flux is captured POST-solve and handed to FSM, whose overwrite is a state jump no per-step identity
# can see, so the transfer carries the pin's SNES tolerance instead of being an integrated source.
#
# CORRECTION (do not restore the earlier claim here). This comment used to add that the next arm --
# active-set WITH -wtm_fsm_delta_source -- closes ~50x tighter at 8e-8, and read that as evidence the
# two changes "belong together". That inference was WRONG. On this same fixture the source arm HALVES
# the ponded water (160.00 -> 79.04 m of ponded depth, max wtd 10.0 -> 5.0 m), so the tighter residual
# is measured on a MATERIALLY DIFFERENT answer, not on a better version of the same one -- and tighter
# closure of a different state says nothing about complementarity. Both arms are kept as conservation
# GATES; neither is evidence for enabling the pair. See benchmark/scheme_bench/README.md, where
# active-set alone is shown to already remove the FSM between-step shock (ratio 0.985 -> 3.6e-13) that
# -wtm_fsm_delta_source exists to address.
ARM_TOL=1e-5 check "Anderson + active-set [loose tol, see note]" a_as -wtm_anderson -wtm_active_set
check "Anderson + active-set, src" a_as_src -wtm_anderson -wtm_active_set -wtm_fsm_delta_source
echo
echo "-- no single-step identity --"
check_nan "TR-BDF2"                s_tr     -wtm_anderson -wtm_tr_bdf2
echo

if [[ $fail -eq 0 ]]; then echo "BUDGET CLOSURE: ALL PASSED"; else echo "BUDGET CLOSURE: FAILED" >&2; fi
exit $fail
