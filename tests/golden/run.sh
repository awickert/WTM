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
total_cycles       3
maxiter            2
fdepth_a           200
fdepth_b           150
fdepth_fmin        2
time_start         t0
time_end           t0
surfdatadir        $sdir
region             $region
supplied_wt        0
cycles_to_save     9999
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
      transient)     emit_cfg "$TRANS" transient_test "run_type transient" "fsm_on 1" "time_start ta" "time_end tb" "total_cycles 4" ;;
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
    local cfg="$WORK/${name}_n${n}.cfg"
    PREFIX="$WORK/${name}_n${n}_"
    case_cfg "$name" | sed "s|__X__|x|" > "$cfg"
    echo "textfilename       $WORK/${name}_n${n}.txt" >> "$cfg"
    echo "outfile_prefix     $PREFIX" >> "$cfg"
    ( cd "$WORK" && OMP_NUM_THREADS=1 mpirun -n "$n" "$WTM" "$cfg" -snes_stol 1e-8 >"$WORK/${name}_n${n}.log" 2>&1 )
}

# Per-case cross-rank comparison tolerance (metres). Default (golden.py) is ~1e-6 -- above FP-
# reduction noise, below any real change. fsm_evap1 needs a physical (cm) tolerance: with the
# corrected conservative flux (benchmark/GRID_CONVENTION.md) its equilibrium sits near an FSM
# spill/merge routing THRESHOLD, where the ~1e-13 machine-eps rank-dependence of the parallel
# Anderson solve (FP-reduction non-associativity) is amplified by the DISCONTINUOUS FSM routing into
# ~mm cross-rank differences. This is inherent (parallel reductions x discontinuous routing), NOT a
# flux/solver bug -- the groundwater solve alone is MPI-consistent to ~1e-13 (fsm_on 0). A breach of
# this cm tolerance would mean a routing FLIP (lake-scale) -- the separate FSM-determinism issue.
# fsm_runoff is the same class at a smaller scale: under the damped Anderson default (beta=0.5) its
# equilibrium sits near a runoff spill/merge threshold, so the (now ~1e-8, not 1e-13) cross-rank GW
# noise is amplified into ~3 um differences on some rank counts -- physically negligible and, again,
# discontinuous-routing amplification, not a solver bug. 1e-4 m sits well above that jitter and far
# below any real routing change (this fixture moves ~35 m when the runoff path actually flips).
case_tol() { case "$1" in fsm_evap1) echo 5e-2 ;; fsm_runoff) echo 1e-4 ;; *) echo "" ;; esac; }

fail=0
for name in "${CASES[@]}"; do
    if [[ $GEN -eq 1 ]]; then
        run_case "$name" 1
        python3 golden.py generate "$PREFIX" "$REFDIR/${name}.txt"
    else
        for n in 1 2 4 6 8; do
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
