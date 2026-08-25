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
REFDIR=reference
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
      transient)     emit_cfg "$TRANS" transient_test "run_type transient" "fsm_on 1" "time_start ta" "time_end tb" "total_time 8yr" ;;
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
CASES=(below_ground fsm_evap0 fsm_evap1 fsm_runoff fsm_runoff_hi transient)

run_case() { # name nranks -> sets $PREFIX
    local name="$1" n="$2"
    local cfg="$WORK/${name}_n${n}.yaml"
    PREFIX="$WORK/${name}_n${n}_"
    { case_cfg "$name" | sed "s|__X__|x|"
      echo "textfilename   $WORK/${name}_n${n}.txt"
      echo "outfile_prefix $PREFIX"
    } | ../emit_config.sh > "$cfg"
    # -wtm_eq_tol 0: run the full fixed total_time so the reference and the cross-rank checks compare at the
    # SAME cycle (the equilibrium auto-stop default could otherwise fire at MPI-decomposition-dependent cycles).
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "$cfg" -snes_stol 1e-8 -wtm_eq_tol 0 >"$WORK/${name}_n${n}.log" 2>&1 )
}

# Per-case cross-rank comparison tolerance (metres). All cases use the default (~1e-6, above FP-
# reduction noise and below any real change). Under the Picard default the groundwater solve is
# cross-rank consistent to ~1e-9 EVEN on the FSM-routing-threshold cases (measured: fsm_evap1 and
# fsm_runoff both ~1e-9 at n=2..8), so the discontinuous spill/merge routing stays deterministic and
# no per-case relaxation is needed. (This is a Picard win: under the older matrix-free Anderson
# default those two fixtures sat near a routing threshold where Anderson's larger cross-rank GW noise
# was amplified by the discontinuous routing into ~mm-cm differences and needed physical tolerances;
# if you run the tests under -wtm_anderson, expect that to return.)
# Per-case tolerance override (empty = golden.py's default 1e-6 m).
#
# transient: 1e-5 m. Under the default active_set enforcement this case reproduces across MPI rank
# counts only to ~2e-6 m (measured: n=1 and n=4 match the n=1 reference exactly; n=2/6/8 differ by
# 1.2e-6, 2.1e-6 and 1.2e-6 m). That is MICROMETRE-scale round-off on a field with 21 m features,
# sitting at the SNES tolerance: the semismooth pin's active set can differ in its last bits between
# domain decompositions near a cell that is marginally at the free surface. It is round-off, not a
# decomposition-dependent algorithm -- a real MPI inconsistency would not pass at n=1 and n=4 nor stay
# within 2 um. Under the former `implicit` default the same case reproduced below 1e-6, so this is a
# genuine (tiny) loss of cross-rank reproducibility that comes with the constraint being solved rather
# than approximated. Recorded rather than hidden; if this ever needs to be TIGHT again, the fix is an
# active-set tie-break that is decomposition-independent, not a looser number here.
case_tol() { case "$1" in transient) echo "1e-5" ;; *) echo "" ;; esac; }

fail=0
for name in "${CASES[@]}"; do
    if [[ $GEN -eq 1 ]]; then
        run_case "$name" 1
        python3 golden.py generate "$PREFIX" "$REFDIR/${name}.txt"
    else
        for n in $RANKS; do
            run_case "$name" "$n"
            if python3 golden.py check "$PREFIX" "$REFDIR/${name}.txt" $(case_tol "$name"); then
                printf "  %-14s n=%-2s : PASS\n" "$name" "$n"
            else
                printf "  %-14s n=%-2s : FAIL\n" "$name" "$n"; fail=1
            fi
        done
    fi
done

if [[ $GEN -eq 0 ]]; then
    echo
    [[ $fail -eq 0 ]] && echo "GOLDEN CHECKS PASSED" || echo "GOLDEN CHECKS FAILED" >&2
fi
exit $fail
