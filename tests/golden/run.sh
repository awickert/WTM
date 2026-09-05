#!/usr/bin/env bash
# Golden (expected-results) regression. For each case, runs the model and checks
# the final water table against a committed reference captured from a trusted
# n=1 run. Runs at n=1 AND n=4 so the references also serve as an absolute
# cross-rank check (subsuming a before/after binary comparison: the reference is
# the "before"). This catches regressions that perturb every rank count equally
# -- which the n=1-vs-n=N consistency tests cannot.
#
#   tests/golden/run.sh              # check against committed references
#   tests/golden/run.sh --generate   # (re)generate references; review the diff!
#
# Regenerate ONLY when a behavior change is intended and understood.
set -uo pipefail
cd "$(dirname "$0")"

GEN=0
[[ "${1:-}" == "--generate" ]] && { GEN=1; shift; }
WTM=$(readlink -f "${1:-../../build/wtm.x}")
shift || true
RANKS="${*:-1 2 4 6 8}"   # cross-rank check counts (run_all.sh passes the tier's set); default = full sweep
REFDIR="${GOLDEN_REFDIR:-reference}"
# Overridable so the references can be re-derived at a TIGHTER solve and diffed against the committed
# set -- the check that they are converged rather than merely different. Production value: 1e-10.
GOLDEN_STOL="${GOLDEN_STOL:-1e-10}"
mkdir -p "$REFDIR"

if [[ ! -x "$WTM" ]]; then echo "ERROR: WTM binary not found at $WTM" >&2; exit 1; fi

# Ensure fixtures exist.
[[ -f ../fsm_consistency/inputs/fsm_test_t0_topography.tif ]] || ( cd ../fsm_consistency && python3 make_inputs.py >/dev/null )
[[ -f ../ghost_cell/inputs/ghost_cell_test_t0_topography.tif ]] || ( cd ../ghost_cell && python3 make_inputs.py >/dev/null )
[[ -f inputs/transient_test_ta_topography.tif ]] || python3 make_transient_inputs.py >/dev/null
[[ -f inputs_runoff/runoff_test_t0_topography.tif ]] || python3 make_runoff_inputs.py >/dev/null

WORK=$(mktemp -d /tmp/golden_XXXX)
trap 'rm -rf "$WORK"' EXIT

# Each case: name | surfdatadir | region | extra config lines (key value; ...)
# The extra lines override the defaults in emit_cfg.
emit_cfg() { # sdir region extra... -> stdout config
    local sdir="$1" region="$2"; shift 2
    cat <<EOF
solver_method anderson
run_type           equilibrium
fsm_on             0
evap_mode          0
infiltration_on    0
runoff_ratio_on    0
cells_per_degree   10
southern_edge      -45
deltat             31536000
total_time       6yr
report_interval            2
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        $sdir
region             $region
supplied_wt        0
save_nreport_interval     9999
EOF
    for kv in "$@"; do echo "$kv"; done
}

# case name -> emits config body via the function above
case_cfg() {
    local GHOST FSM TRANS RUNOFF
    GHOST=$(readlink -f ../ghost_cell/inputs)
    FSM=$(readlink -f ../fsm_consistency/inputs)
    TRANS=$(readlink -f inputs)
    RUNOFF=$(readlink -f inputs_runoff)
    case "$1" in
      below_ground)  emit_cfg "$GHOST" ghost_cell_test ;;
      fsm_evap0)     emit_cfg "$FSM" fsm_test "fsm_on 1" "supplied_wt 1" "evap_mode 0" ;;
      fsm_evap1)     emit_cfg "$FSM" fsm_test "fsm_on 1" "supplied_wt 1" "evap_mode 1" ;;
      fsm_runoff)    emit_cfg "$RUNOFF" runoff_test    "fsm_on 1" "supplied_wt 1" "evap_mode 1" "runoff_ratio_on 1" ;;
      fsm_runoff_hi) emit_cfg "$RUNOFF" runoff_test_hi "fsm_on 1" "supplied_wt 1" "evap_mode 1" "runoff_ratio_on 1" ;;
      # adaptive_dt PINNED FALSE. A golden is an exact-reproduction assertion, so nothing in it may be
      # chosen by a controller whose input is a global reduction. With adaptive dt on, the embedded error
      # estimate came out DECOMPOSITION-DEPENDENT on this fixture -- at the same dt of 4.7304e+07 s the
      # estimate was 4.007673531e-02 at n=1, 3.113780608e-02 at n=2 and 3.987584816e-02 at n=6, a 22%
      # spread from states agreeing to 2e-11. That moved the growth factor (1.0374 / 1.1769 / 1.0400),
      # hence the SUB-STEP SIZES (step 7 ran at 3.4121e+07 / 3.5478e+07 / 3.4169e+07 s), hence the
      # trajectory. Different steps give different -- and equally valid -- answers, so no comparison
      # tolerance and no tie-break can reconcile them; the fix is to stop the test asking the question.
      # The estimate's sensitivity is itself a real defect (task #56): it is built by a reduction over a
      # cell set chosen by an exact `!= 0.0` float test (transient_groundwater.cpp:1883, :1922, which its
      # own note says drops 17-22% of land cells) on a `dev` that is a difference of two nearly-equal
      # stored volumes. Pinning the step here does NOT fix that -- it stops this test from depending on it.
      # Same reasoning, and same fix, as tests/coupling_convergence and tests/dt_invariance (2af7e67).
      transient)     emit_cfg "$TRANS" transient_test "run_type transient" "fsm_on 1" "time_start ta" "time_end tb" "total_time 8yr" "adaptive_dt false" ;;
      # fsm_impulse: the SAME case as fsm_evap1 under the non-default coupling. It exists because
      # surface_water.fsm_coupling now defaults to `continuous`, which would leave `impulse`
      # unexercised by every arm here -- and an alternative nobody runs is one that rots quietly.
      fsm_impulse)   emit_cfg "$FSM" fsm_test "fsm_on 1" "supplied_wt 1" "evap_mode 1" "fsm_coupling impulse" ;;
      *) echo "unknown case $1" >&2; return 1 ;;
    esac
}

# fsm_runoff exercises runoff_ratio_on with FSM on (a 2D-sinusoid fixture: two hills, two
# closed depressions, deep water table). The recharge is split by the runoff ratio, so the
# distributed recharge must compute rech and its runoff and gather the runoff to rank-0
# arp.runoff for FSM -- reproducing the serial rank-0 recharge bit-identically. The case
# is strongly sensitive to the runoff path (runoff_ratio on vs off shifts the water table
# ~35 m) and cross-rank stable (smooth gradient -> deterministic FSM routing). fsm_runoff_hi
# is the same setup on higher-overtone terrain (more, smaller depressions) -- exercising the
# runoff path over a richer routing pattern, still band-limited and cross-rank stable.
CASES=(below_ground fsm_evap0 fsm_evap1 fsm_runoff fsm_runoff_hi transient fsm_impulse)

# A golden is only as trustworthy as the run that produced it. A run that aborts partway leaves
# its EARLY cycles on disk, and golden.py's last_tif takes the newest file that exists -- so a
# truncated run looks exactly like a finished one. Under --generate that silently enshrines a
# half-finished run as the reference; under check, a truncated run then PASSES against it. This
# is not hypothetical: at 5feb2c3 the transient reference was overwritten with the cycle-0
# INITIAL CONDITION (bit-exactly the masked input water table), and the case passed vacuously
# until later fixes let the run complete. Hence two preconditions below, both fatal: the model
# must exit 0, and it must have written the output for the configured total_time.
run_case() { # name nranks -> sets $PREFIX; nonzero if the run did not finish
    local name="$1" n="$2"
    local cfg="$WORK/${name}_n${n}.yaml"
    local log="$WORK/${name}_n${n}.log"
    PREFIX="$WORK/${name}_n${n}_"
    { case_cfg "$name" | sed "s|__X__|x|"
      echo "eq_tol 0"
      echo "textfilename   $WORK/${name}_n${n}.txt"
      echo "outfile_prefix $PREFIX"
    } | ../emit_config.sh > "$cfg"
    # -wtm_eq_tol 0: run the full fixed total_time so the reference and the cross-rank checks compare at the
    # SAME cycle (the equilibrium auto-stop default could otherwise fire at MPI-decomposition-dependent cycles).
    # -snes_stol 1e-10, NOT 1e-8. snes_stol is a STEP tolerance: it stops when the iterate stops
    # moving, which in a near-null direction of this operator leaves real positional slack in the
    # answer. At 1e-8 the transient fixture's step-7 solve landed 1.3916e-01 m apart at n=1 vs n=4 --
    # from IDENTICAL inputs (lake stage agreeing to 5.8e-13) with ZERO difference in the active set --
    # and that fed a 2.126e-03 m difference in the final field, which read as MPI non-reproducibility.
    # Measured across the tolerance: 1e-8 -> 1.3916e-01 m, 1e-10 -> 2.9179e-10 m, 1e-12 -> 2.9179e-10,
    # 1e-14 -> 2.9179e-10. It collapses to round-off at 1e-10 and SATURATES there, so 1e-10 is
    # sufficient and nothing pathological sits underneath.
    # The principle is budget_closure's, applied here: an assertion is only meaningful if the SOLVE is
    # resolved tighter than the agreement it asserts, or the arm measures solver noise. These goldens
    # assert 1e-6..1e-5 m, so they must not be solved to a tolerance that permits 1e-1 m.
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "$cfg" -snes_stol "$GOLDEN_STOL" >"$log" 2>&1 )
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        printf "  %-14s n=%-2s : MODEL FAILED (exit %d) -- refusing to use its output\n" "$name" "$n" "$rc" >&2
        tail -n 15 "$log" | sed 's/^/      | /' >&2
        return "$rc"
    fi
    # The run exited 0; require the output for the CONFIGURED end time, so a short run cannot pass
    # itself off as a finished one. WTM names outputs <prefix><cycle>_<elapsed>.tif.
    local tt
    tt=$(case_cfg "$name" | awk '$1=="total_time"{v=$2} END{print v}')
    if ! compgen -G "${PREFIX}*_${tt}.tif" >/dev/null; then
        printf "  %-14s n=%-2s : INCOMPLETE -- no output at total_time=%s (have: %s)\n" \
               "$name" "$n" "$tt" "$(basename -a ${PREFIX}*.tif 2>/dev/null | tr '\n' ' ')" >&2
        return 1
    fi
    return 0
}

# Per-case cross-rank comparison tolerance (metres). All cases use the default (~1e-6, above FP-
# reduction noise and below any real change). Under the Picard default the groundwater solve is
# cross-rank consistent to ~1e-9 EVEN on the FSM-routing-threshold cases (measured: fsm_evap1 and
# fsm_runoff both ~1e-9 at n=2..8), so the discontinuous spill/merge routing stays deterministic and
# no per-case relaxation is needed. (This is a Picard win: under the older matrix-free Anderson
# default those two fixtures sat near a routing threshold where Anderson's larger cross-rank GW noise
# was amplified by the discontinuous routing into ~mm-cm differences and needed physical tolerances;
# if you run the tests under, expect that to return.)
# Per-case tolerance override (empty = golden.py's default 1e-6 m).
#
# BOTH PER-CASE RELAXATIONS ARE RETIRED (2026-09-05). Every case now passes at the 1e-6 default at
# n = 1, 2, 4, 6 and 8. The two diagnoses below were correct about the MECHANISM; what changed is that
# neither needs a relaxation any more, once the solve is resolved tighter than the assertion
# (snes_stol 1e-10) and the transient case no longer lets a controller choose its step. Kept as history.
#
# transient: WAS 1e-5 m. Under the default active_set enforcement this case reproduces across MPI rank
# counts only to ~2e-6 m (measured: n=1 and n=4 match the n=1 reference exactly; n=2/6/8 differ by
# 1.2e-6, 2.1e-6 and 1.2e-6 m). That is MICROMETRE-scale round-off on a field with 21 m features,
# sitting at the SNES tolerance: the semismooth pin's active set can differ in its last bits between
# domain decompositions near a cell that is marginally at the free surface. It is round-off, not a
# decomposition-dependent algorithm -- a real MPI inconsistency would not pass at n=1 and n=4 nor stay
# within 2 um. Under the former `implicit` default the same case reproduced below 1e-6, so this is a
# genuine (tiny) loss of cross-rank reproducibility that comes with the constraint being solved rather
# than approximated. Recorded rather than hidden; if this ever needs to be TIGHT again, the fix is an
# active-set tie-break that is decomposition-independent, not a looser number here.
# fsm_runoff: WAS 1e-5 m, and this one is a DIAGNOSED relaxation rather than a shrug. Under the default
# fsm_coupling: continuous, cross-rank drift COMPOUNDS instead of being reset: `impulse` overwrote every
# rank's starting_wtd from rank 0 every step (WTM.cpp, the impulse branch), which wiped accumulated drift;
# continuous never runs that line. Measured n=1 vs n=6 on this fixture: 8.2e-10 -> 1.3e-09 -> 7.1e-09 under
# continuous against a flat 9.0e-12 -> 1.6e-11 under impulse. This arm lands at 1.214e-06, just over the
# 1e-6 default.
#
# The relaxation was only defensible because the behaviour is measured DIRECTLY, by tests/xrank_growth,
# which asserts the REGIME (flat vs compounding) rather than a magnitude. That test remains where the #39
# finding lives and must not be deleted. What the retirement shows is that the 1.214e-06 which pushed this
# arm over 1e-6 was mostly SOLVER NOISE rather than coupling drift: at snes_stol 1e-10 the arm clears 1e-6
# outright. The compounding regime is real; its magnitude HERE was inflated by an under-resolved solve.
# See task #39.
case_tol() { case "$1" in *) echo "" ;; esac; }   # no per-case relaxations; all cases at the 1e-6 default

fail=0
for name in "${CASES[@]}"; do
    if [[ $GEN -eq 1 ]]; then
        if ! run_case "$name" 1; then
            echo "  REFUSING to regenerate $name from a run that did not finish" >&2; fail=1; continue
        fi
        python3 golden.py generate "$PREFIX" "$REFDIR/${name}.txt"
    else
        for n in $RANKS; do
            if ! run_case "$name" "$n"; then
                printf "  %-14s n=%-2s : FAIL (run did not finish)\n" "$name" "$n"; fail=1; continue
            fi
            if python3 golden.py check "$PREFIX" "$REFDIR/${name}.txt" $(case_tol "$name"); then
                printf "  %-14s n=%-2s : PASS\n" "$name" "$n"
            else
                printf "  %-14s n=%-2s : FAIL\n" "$name" "$n"; fail=1
            fi
        done
    fi
done

echo
if [[ $GEN -eq 1 ]]; then
    [[ $fail -eq 0 ]] && echo "GOLDEN REFERENCES REGENERATED -- review the diff before committing" \
                      || echo "GOLDEN REGENERATION INCOMPLETE (see refusals above)" >&2
else
    [[ $fail -eq 0 ]] && echo "GOLDEN CHECKS PASSED" || echo "GOLDEN CHECKS FAILED" >&2
fi
exit $fail
