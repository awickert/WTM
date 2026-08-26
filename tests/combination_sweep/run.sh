#!/usr/bin/env bash
# COMBINATION SWEEP: attempt every solver x integrator x collector, at BOTH run types, and record what
# each one actually does.
#
# WHY THIS REPLACES A CURATED LIST. tests/coverage_matrix.py used to carry a hand-written list of
# "combinations WTM refuses by design", so a blank cell in the matrix meant either "untested" or
# "unreachable for a reason someone typed in once" -- and the reader could not tell which. That list
# was an opinion. This sweep is evidence: it TRIES every combination. What runs is coverage; what
# refuses documents itself, with the model's own message; and the matrix learns both from observation
# instead of being told.
#
# THE ASSERTION IS NOT "everything works". It is:
#
#     every combination must either RUN, or REFUSE with a legible message.
#
# A combination that aborts with no explanation is the failure this gates. That is the difference
# between a documented incompatibility and a crash.
#
# TWO KINDS OF FAILURE, KEPT APART. They mean opposite things and must not be lumped together:
#   DESIGN    the model deliberately throws with an explanation ("... is not supported on the Picard
#             solver"). That is a real incompatibility and it documents itself.
#   HARD      the solver did not converge. That is NOT a statement about the combination being
#             forbidden -- it is a statement about this dt on this fixture. So the sweep TRIES AGAIN
#             at dt/8 before recording anything, because "I could not make it work at my first
#             guess" is not evidence and must not be presented as if it were.
#
# ONE FIXTURE FOR BOTH RUN TYPES. The rech_test fixture carries `ta` and `tb` slices, so equilibrium
# (time_start = time_end = ta) and transient (ta -> tb) differ ONLY in run_type. Using different
# fixtures per run type would confound "this combination fails in transient" with "this combination
# fails on that fixture".
#
# Usage:  tests/combination_sweep/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

RECH=$(readlink -f ../recharge_consistency)
[[ -f "$RECH/inputs/rech_test_ta_topography.tif" ]] || ( cd "$RECH" && python3 make_inputs.py >/dev/null )
INP="$RECH/inputs"
WORK=$(mktemp -d /tmp/combo_XXXX); trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=1

mkcfg() { # $1 stem, $2 run_type, $3 collector, $4 deltat
    local tend="ta"; [ "$2" = transient ] && tend="tb"
    ../emit_config.sh > "$WORK/$1.yaml" <<EOF
run_type $2
fsm_on 1
infiltration_on 0
runoff_ratio 0
cells_per_degree 1
southern_edge 0
deltat $4
total_time 4yr
report_interval 2
save_nreport_interval 9999
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start ta
time_end $tend
surfdatadir $INP
region rech_test
supplied_wt 1
runoff_collector $3
textfilename $WORK/$1.txt
outfile_prefix $WORK/${1}_
EOF
}

# Solver and integrator are given as FLAG SETS; which integrator each actually resolves to is recorded
# by the model's own coverage fingerprint, not assumed here (active_set auto-enables volume storage,
# for instance, so the requested and resolved integrator differ).
declare -A SOLVERS=( [anderson]="-wtm_anderson"
                     [picard]="-wtm_picard"
                     [newton]="-wtm_newton -wtm_dt_continuation" )
declare -A INTEGS=(  [be]=""
                     [volume]="-wtm_volume_storage"
                     [bdf2v]="-wtm_bdf2_on_V"
                     [trbdf2]="-wtm_tr_bdf2" )
COLLECTORS=(active_set explicit implicit legacy off)
RUNTYPES=(equilibrium transient)

echo "=== combination sweep: solver x integrator x collector x run_type ==="
echo "WTM binary: $WTM"
echo
printf "  %-11s %-8s %-11s %-12s %s\n" solver integ collector run_type outcome
fail=0; nrun=0; nretry=0; ndesign=0; nhard=0; nbad=0
REFUSALS="${WTM_COVERAGE_LOG:-$WORK/refusals.txt}"

# Run one attempt. Returns 0 if it completed; otherwise leaves the message in $MSG (empty if none).
# Wrapped in `sh -c` so an expected abort's job-control notice goes to the log, not the suite output.
attempt() { # $1 stem, $2 extra flags...
    local stem="$1"; shift
    MSG=""
    if sh -c '"$@"' _ "$WTM" "$WORK/$stem.yaml" "$@" -snes_stol 1e-8 -wtm_eq_tol 0 \
            > "$WORK/$stem.log" 2>&1; then
        return 0
    fi
    MSG=$(grep -m1 "what():" "$WORK/$stem.log" || true); MSG="${MSG#*what():  }"
    return 1
}

for rt in "${RUNTYPES[@]}"; do
  for sv in anderson picard newton; do
    for ig in be volume bdf2v trbdf2; do
      for cl in "${COLLECTORS[@]}"; do
        stem="${rt:0:2}_${sv}_${ig}_${cl}"
        mkcfg "$stem" "$rt" "$cl" 31536000
        if attempt "$stem" ${SOLVERS[$sv]} ${INTEGS[$ig]}; then
            OUT="runs"; nrun=$((nrun+1))
        elif [ -z "$MSG" ]; then
            OUT="ABORTED WITH NO MESSAGE"; nbad=$((nbad+1)); fail=1
        elif [[ "$MSG" == *"Could not open"* || "$MSG" == *"No such file"* ]]; then
            # A missing input is THIS HARNESS being broken, not the model refusing anything. It must
            # fail loudly: silently filing it under "refused by design" is how 60 perfectly reachable
            # combinations got recorded as forbidden on the first run of this sweep.
            OUT="SETUP ERROR (this test is broken): ${MSG:0:50}"; nbad=$((nbad+1)); fail=1
        elif [[ "$MSG" == *"not converged"* || "$MSG" == *"max retries"* ]]; then
            # HARD, not forbidden: try again at dt/8 before recording a verdict.
            mkcfg "${stem}_s" "$rt" "$cl" 3942000
            if attempt "${stem}_s" ${SOLVERS[$sv]} ${INTEGS[$ig]}; then
                OUT="runs (needed dt/8)"; nretry=$((nretry+1))
            else
                OUT="did not converge, even at dt/8"; nhard=$((nhard+1))
                echo "refusal kind=hard run_type=$rt solver=$sv integrator=$ig collector=$cl msg=${MSG:0:100}" >> "$REFUSALS"
            fi
        else
            OUT="refused by design: ${MSG:0:56}"; ndesign=$((ndesign+1))
            echo "refusal kind=design run_type=$rt solver=$sv integrator=$ig collector=$cl msg=${MSG:0:100}" >> "$REFUSALS"
        fi
        printf "  %-11s %-8s %-11s %-12s %s\n" "$sv" "$ig" "$cl" "$rt" "$OUT"
      done
    done
  done
done

echo
echo "  ran at the nominal dt        : $nrun"
echo "  ran only after dropping to dt/8: $nretry"
echo "  refused BY DESIGN (documented) : $ndesign"
echo "  did not converge even at dt/8  : $nhard"
echo "  harness broken / no message    : $nbad"
if [[ $nbad -gt 0 ]]; then
    echo
    echo "  A combination that fails must SAY WHY, and a missing input is THIS TEST's fault, not the"
    echo "  model's. Neither may be quietly filed under \"refused by design\"."
fi
echo
if [[ $fail -eq 0 ]]; then echo "COMBINATION SWEEP: ALL PASSED"; else echo "COMBINATION SWEEP: FAILED" >&2; fi
exit $fail
