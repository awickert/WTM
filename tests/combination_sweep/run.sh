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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

RECH=$(readlink -f ../recharge_consistency)
[[ -f "$RECH/inputs/rech_test_ta_topography.tif" ]] || ( cd "$RECH" && python3 make_inputs.py >/dev/null )
INP="$RECH/inputs"
make_work combo
export OMP_NUM_THREADS=1

# THE CONFIG IS A FILE NOW (#83): tests/combination_sweep/config.yaml, one file with six per-cell
# slots. EVERY SLOT SUBSTITUTES A VALUE; none removes a line. That is not stylistic -- the first
# attempt at this used a deletable placeholder for dev.storage_form, and deleting the line let the
# model default apply, silently turning all 24 `be` cells into `volume` cells. Every one then hit a
# refusal, and the classifier of the day filed all 96 under "refused by design" and printed ALL PASSED
# with `ran: 0` (#94, now fixed -- which is what makes this materialisation checkable at all).
mkcfg() { # $1 stem, $2 run_type, $3 collector, $4 deltat   [env: METHOD, INTEG, STORAGE per cell]
    local tend="ta"; [ "$2" = transient ] && tend="tb"
    sed -e "s|@INPUTS@|$INP|g" -e "s|@WORK@|$WORK|g" -e "s|@STEM@|$1|g" \
        -e "s|@RUNTYPE@|$2|g" -e "s|@TEND@|$tend|g" -e "s|@COLLECTOR@|$3|g" -e "s|@DT@|$4|g" \
        -e "s|@METHOD@|${METHOD:?mkcfg needs a solver method}|g" \
        -e "s|@INTEG@|${INTEG:?mkcfg needs a time_integration}|g" \
        -e "s|@STORAGE@|${STORAGE:?mkcfg needs a storage form: an EMPTY one renders a valueless key, and
                                  the model reports that as null several layers from the cause}|g" \
        config.yaml > "$WORK/$1.yaml"
    grep -q "@[A-Z_]*@" "$WORK/$1.yaml" && { echo "ERROR: $1.yaml has an unfilled slot"; exit 1; }
    return 0
}

# Solver and integrator are given as FLAG SETS; which integrator each actually resolves to is recorded
# by the model's own coverage fingerprint, not assumed here (active_set auto-enables volume storage,
# for instance, so the requested and resolved integrator differ).
declare -A SOLVERS=( [anderson]=""
                     [picard]=""   # solver.method: picard -- config, via METHOD=
                     [newton]="" )   # solver.method: newton (implies continuation) -- via METHOD=
declare -A INTEGS=(  [be]=""
                     [volume]=""   # dev.storage_form: volume -- set via STORAGE= on mkcfg, not a flag
                     [bdf2v]=""   # solver.time_integration: bdf2 -- via INTEG=
                     [trbdf2]="" )   # solver.time_integration: tr-bdf2 -- via INTEG=
COLLECTORS=(active_set explicit implicit off)
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
    if WTM_COVERAGE_TAG="combination_sweep/$stem" sh -c '"$@"' _ "$WTM" "$WORK/$stem.yaml" "$@" \
            > "$WORK/$stem.log" 2>&1; then
        return 0
    fi
    # The model PRINTS its refusal as `ERROR: <msg>` and exits 1 (#57). It used to escape as an
    # uncaught exception, and this line read the crash artifact `what():` -- which vanished the moment
    # refusals stopped crashing, turning every documented refusal into "ABORTED WITH NO MESSAGE".
    # Prefer the designed message; fall back to the crash text so a REAL crash is still captured.
    MSG=$(grep -m1 "^ERROR: " "$WORK/$stem.log" || true); MSG="${MSG#ERROR: }"
    [ -n "$MSG" ] || { MSG=$(grep -m1 "what():" "$WORK/$stem.log" || true); MSG="${MSG#*what():  }"; }
    return 1
}

# THE DOCUMENTED REFUSALS -- the ONLY messages that may be filed under "refused by design" (#94).
#
# WHY A LIST RATHER THAN A CATCH-ALL. This classifier used to end in an `else` that swallowed every
# unmatched message into `refused by design`. A design refusal and a MALFORMED CONFIG are then
# indistinguishable -- both are a non-zero exit with a message, and both even begin "config: ". So a
# sweep whose whole purpose is recording WHICH COMBINATIONS RUN could report total success while
# running NOTHING. It did: a one-line change to how a key was passed sent all 96 cells into this
# bucket, and the suite printed ALL PASSED with `ran at the nominal dt: 0`.
#
# Each entry is a distinguishing SUBSTRING of a refusal the model raises deliberately, paired with the
# number of cells expected to hit it. The counts are MEASURED from a green run, not chosen: 16 + 6 + 4
# = 26, which is the `refused BY DESIGN` total. They are asserted at the end, so a refusal that
# silently spreads to more cells -- or stops firing -- is a failure rather than a quiet re-shuffle.
declare -A DESIGN_REFUSALS=(
  ["solver.time_integration: tr-bdf2 runs only on the matrix-free Anderson path"]=16
  ["dev.storage_form: secant cannot be used with surface_water.collection.method: active_set"]=6
  ["surface_water.collection.method: active_set is not supported"]=4
)
declare -A DESIGN_SEEN=()
for k in "${!DESIGN_REFUSALS[@]}"; do DESIGN_SEEN["$k"]=0; done

# The ran-count FLOOR. 64 of 96 cells ran at the nominal dt on the reference run. A collapse means the
# harness broke, not that the model changed its mind about 60 combinations, and it must FAIL rather
# than pass quietly -- the same non-vacuity guard multilake and dt_sensitivity carry.
RAN_FLOOR="${RAN_FLOOR:-64}"

for rt in "${RUNTYPES[@]}"; do
  for sv in anderson picard newton; do
    for ig in be volume bdf2v trbdf2; do
      for cl in "${COLLECTORS[@]}"; do
        stem="${rt:0:2}_${sv}_${ig}_${cl}"
        # Settings that used to be FLAGS are now config values, so they reach the model only through
        # mkcfg -- and there are TWO mkcfg calls per combination (nominal dt, and the dt/8 retry below).
        # These are per-iteration VARIABLES, not command prefixes, so both calls see them. As prefixes
        # they applied to the first call only, and the retry silently ran the DEFAULT solver: that turned
        # "picard cannot converge" into a false "picard runs at dt/8" for 12 combinations.
        METHOD="$sv"
        # NAME BOTH SCHEME AXES EXPLICITLY. `be` and `volume` used to leave INTEG empty, and an absent
        # solver.time_integration is `auto` -- which resolves to tr-bdf2 on the Anderson path. STORAGE
        # was likewise empty for `be`, and the storage-form default is volume. So `be` and `volume` were
        # the same run as each other on every solver, and on anderson both were also the same run as
        # `trbdf2` (auto -> tr-bdf2 there). 32 of the 96 cells were duplicates of another cell in the
        # same table -- 16 on anderson (3 columns resolving to 1, over 4 collectors x 2 run types) and
        # 8 each on picard and newton (be == volume) -- while the table reported them as distinct
        # coverage (#24, #37).
        case "$ig" in
            be)     INTEG=backward-euler; STORAGE=secant ;;
            volume) INTEG=backward-euler; STORAGE=volume ;;
            # These two left STORAGE empty and the shim filled in the model default. The config
            # states the key, so the value is NAMED here -- an empty slot is not a default.
            bdf2v)  INTEG=bdf2;           STORAGE=volume ;;
            trbdf2) INTEG=tr-bdf2;        STORAGE=volume ;;
        esac
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
            mkcfg "${stem}_s" "$rt" "$cl" 3942000   # STORAGE/METHOD/DTC still in scope -- see above
            if attempt "${stem}_s" ${SOLVERS[$sv]} ${INTEGS[$ig]}; then
                OUT="runs (needed dt/8)"; nretry=$((nretry+1))
            else
                OUT="did not converge, even at dt/8"; nhard=$((nhard+1))
                echo "refusal kind=hard run_type=$rt solver=$sv integrator=$ig collector=$cl msg=${MSG:0:100}" >> "$REFUSALS"
            fi
        else
            # Match against the DOCUMENTED list. Anything else is this harness being broken -- a
            # retired key, a typo, a half-rendered config -- and must fail loudly rather than pass as
            # a refusal the model never made.
            matched=""
            for pat in "${!DESIGN_REFUSALS[@]}"; do
                if [[ "$MSG" == *"$pat"* ]]; then matched="$pat"; break; fi
            done
            if [ -n "$matched" ]; then
                OUT="refused by design: ${MSG:0:56}"; ndesign=$((ndesign+1))
                DESIGN_SEEN["$matched"]=$(( ${DESIGN_SEEN["$matched"]} + 1 ))
                echo "refusal kind=design run_type=$rt solver=$sv integrator=$ig collector=$cl msg=${MSG:0:100}" >> "$REFUSALS"
            else
                OUT="UNDOCUMENTED FAILURE (this test is broken): ${MSG:0:44}"; nbad=$((nbad+1)); fail=1
                echo "refusal kind=UNDOCUMENTED run_type=$rt solver=$sv integrator=$ig collector=$cl msg=${MSG:0:200}" >> "$REFUSALS"
            fi
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

# NON-VACUITY: the sweep must actually have run something. Without this, breaking every cell reads as
# 96 design refusals and the suite passes (#94).
if [[ $nrun -lt $RAN_FLOOR ]]; then
    echo
    echo "  FAIL  only $nrun of 96 cells ran at the nominal dt, against a floor of $RAN_FLOOR."
    echo "        This sweep RECORDS WHICH COMBINATIONS RUN, so a collapse in that count means the"
    echo "        harness broke -- not that the model changed its mind about 60 combinations."
    fail=1
fi

# Each documented refusal must fire on exactly the cells it fired on when the list was measured. A
# refusal that spreads, or stops firing, is a change in what the model forbids and deserves a look.
for pat in "${!DESIGN_REFUSALS[@]}"; do
    want=${DESIGN_REFUSALS["$pat"]}; got=${DESIGN_SEEN["$pat"]}
    if [[ "$got" -ne "$want" ]]; then
        echo "  FAIL  refusal count moved: $got cells (expected $want) for:"
        echo "          ${pat:0:88}"
        fail=1
    fi
done
echo
if [[ $fail -eq 0 ]]; then echo "COMBINATION SWEEP: ALL PASSED"; else echo "COMBINATION SWEEP: FAILED" >&2; fi
exit $fail
