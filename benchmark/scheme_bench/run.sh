#!/usr/bin/env bash
# PRECISION-MATCHED speed comparison across WTM's time-integration / solver schemes.
#
# THE RULE THIS ENFORCES. Never compare wall time or iteration counts across schemes at their own
# stopping criteria: a shared eq_tol fires at a DIFFERENT converged precision for each scheme, so a
# naive table compares a high-precision method against a low-precision one and calls the sloppy one
# fast. Every scheme is therefore run to a FIXED cycle budget with the auto-stop disabled, its whole
# per-cycle trajectory is recorded, and cost is read off at MATCHED precision.
#
# Output (see report.py) is three parts, in this order:
#   1. ISO-PRECISION cost   -- cycles / SNES iterations / wall to first reach each target water-depth
#                              rms; "never" when the scheme's floor is coarser than the target.
#   2. PRECISION FLOOR      -- the finest rms each scheme reaches at all, and where it plateaus.
#   3. NATIVE STOP          -- what each scheme's own auto-stop would have cost, WITH the rms it
#                              stopped at, explicitly labelled NOT precision-matched.
#
# The precision axis is the per-cycle water-depth rms in mm-water (`|S*dwtd| ... rms=... mm-water`),
# the volume-consistent metric, not a head difference.
#
# Wall time is reported alongside iterations everywhere. Total wall is measured; the iso-precision
# wall is apportioned by iteration share and marked `~` because per-cycle wall is not instrumented.
# All arms run at the same rank count on an otherwise idle machine; treat wall as noisy at the ~10%
# level and iterations as the clean algorithmic metric.
#
# COUPLING (env, default `between`) selects HOW FillSpillMerge's water reaches the groundwater solve:
#   between -- FSM runs between steps and OVERWRITES the water table (the original behaviour)
#   during -- FSM's per-cell volume change is folded into the NEXT step's source term
#              (-wtm_fsm_delta_source, #116), so the water arrives DURING the step
# Run both and compare with compare.py. NOTE the two converge to genuinely DIFFERENT equilibria
# (ponded cells infiltrate under `during` instead of being re-pinned full each step), so the
# per-cycle rms compared here is a SETTLING-RATE metric -- how fast each stops changing -- and says
# nothing about which answer is more nearly right. Do not read it as accuracy.
#
# Usage:  benchmark/scheme_bench/run.sh [path/to/wtm.x] [ranks] [cycles]
#         COUPLING=during benchmark/scheme_bench/run.sh ...
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
RANKS="${2:-4}"
CYCLES="${3:-120}"
COUPLING="${COUPLING:-between}"
case "$COUPLING" in
  between) COUPLING_FLAGS="" ;;
  during)  COUPLING_FLAGS="-wtm_fsm_delta_source" ;;
  *) echo "ERROR: COUPLING must be 'between' or 'during' (got '$COUPLING')"; exit 1 ;;
esac
# COLLECTOR (env, default `implicit`) selects how the wtd<=0 exfiltration constraint is ENFORCED:
#   implicit   -- in-residual siphon max(0,wtd)/dt. Measured dt-DEPENDENT: retained head ~ linear in dt,
#                 which FSM then turns into dt-dependent lake depth (SURFACE_WATER_ROUTING.md).
#   active_set -- semismooth pin at wtd=0 inside the solve. Measured dt-INDEPENDENT (5.6986 m at
#                 dt = 1, 1/3, 1/6 week). Supersedes the collector removals and auto-enables
#                 -wtm_volume_storage. Its pin is in the ANDERSON residual only -- the Picard operator
#                 and Newton Jacobian have no tangent for it, so those arms are expected to be
#                 inconsistent here, exactly as Newton is under `implicit`. Reported, not hidden.
COLLECTOR="${COLLECTOR:-implicit}"
case "$COLLECTOR" in
  implicit)   COLLECTOR_FLAGS="" ;;
  active_set) COLLECTOR_FLAGS="-wtm_active_set" ;;
  *) echo "ERROR: COLLECTOR must be 'implicit' or 'active_set' (got '$COLLECTOR')"; exit 1 ;;
esac
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }

DOM=$(readlink -f ../island/domain)
[[ -f "$DOM/Esquibel_010000_topography.tif" ]] || { echo "ERROR: island fixture missing at $DOM"; exit 1; }
OUT="${OUT:-results_${COLLECTOR}_${COUPLING}}"; mkdir -p "$OUT"
export OMP_NUM_THREADS=1     # pure MPI: OpenMP x MPI oversubscription hangs this fixture at n>=4

# Cold start from a saturated table (supplied_wt 0) -- the spin-up regime, where the schemes actually
# differ. deltat 1 week matches production. eq_tol 0 disables the auto-stop so every arm runs the
# same budget and we see each one's floor rather than where it chose to quit.
mkcfg() {  # $1 = stem, $2 = extra legacy config lines (may be empty)
    { cat <<EOF
run_type equilibrium
total_time $((604800 * CYCLES))s
deltat 604800
report_interval 1
save_nreport_interval 9999
supplied_wt 0
cells_per_degree 900
southern_edge 55.3839465761
fdepth_a 100
fdepth_b 150
fdepth_fmin 2.5
fsm_on 1
infiltration_on 0
runoff_collector implicit
surfdatadir $DOM
region Esquibel
time_start 010000
time_end 010000
eq_tol 0
textfilename $OUT/$1.txt
outfile_prefix $OUT/${1}_
EOF
      [ -n "${2:-}" ] && echo "$2"; } | ../../tests/emit_config.sh > "$OUT/$1.yaml"
}

# stem | human label | solver flags | config lines (settings that are config keys, not flags)
#
# The fourth field exists because t_bar and adaptive_dt are no longer -wtm_ flags: they are
# solver.t_bar and solver.adaptive_dt in the config, so a per-arm value has to reach mkcfg rather
# than the command line. Keeping them in this table means each arm still declares its own setup in
# one place, which is what makes the rows comparable.
SCHEMES=(
  "and_be|Anderson BE (secant)|-wtm_anderson|"
  "and_vol|Anderson BE (volume dV)|-wtm_anderson -wtm_volume_storage|"
  "picard|Picard BDF2-on-V (plain)|-wtm_picard -wtm_bdf2_on_V|"
  "picard_tbar|Picard BDF2-on-V + Tbar|-wtm_picard -wtm_bdf2_on_V|t_bar true"
  "tr_fixed|TR-BDF2 (fixed dt)|-wtm_anderson -wtm_tr_bdf2|"
  "tr_adapt|TR-BDF2 + adaptive dt|-wtm_anderson -wtm_tr_bdf2|adaptive_dt true"
  "newton|Newton (plain)|-wtm_newton|"
  "newton_cont|Newton + dt-continuation|-wtm_stiff|"
)
# NOTE on fairness: Picard and Newton are known to fail from a COLD start at production dt -- plain
# arms are kept so that is visible, but each also gets its documented working recipe (log-mean
# transmissivity for Picard; dt-continuation for Newton) so no scheme is judged on a setup it was
# never claimed to handle. The continuation arm varies dt, so its CYCLE count is not comparable with
# the fixed-dt arms; its iterations and wall still are, and precision is matched by rms either way.
#
# KNOWN HANDICAP, not yet corrected: every arm here runs runoff_collector=implicit so the physics is
# identical across schemes, but the implicit exfiltration kink is NOT in the Newton Jacobian (the run
# prints a WARNING to that effect and the documented remedy is runoff_collector=explicit). So the
# Newton rows are a lower bound on Newton, measured outside its supported configuration. Fixing this
# properly means a second matched set at runoff_collector=explicit for ALL schemes -- comparing one
# scheme under explicit against the rest under implicit would manufacture a difference that is really
# a configuration artifact.

echo "=== scheme benchmark: island (117x75 = 8775 cells), cold start, dt = 1 week ==="
echo "binary: $WTM   ranks: $RANKS   cycle budget: $CYCLES   auto-stop: DISABLED (eq_tol 0)"
echo "FSM coupling: $COUPLING${COUPLING_FLAGS:+  ($COUPLING_FLAGS)}   collector: $COLLECTOR${COLLECTOR_FLAGS:+  ($COLLECTOR_FLAGS)}"
echo
printf "%-28s %10s %12s %12s\n" "scheme" "rc" "wall_s" "SNES_iters"
: > "$OUT/summary.csv"
echo "stem,label,rc,wall_s,iters,cycles_run" >> "$OUT/summary.csv"
for entry in "${SCHEMES[@]}"; do
    IFS='|' read -r stem label flags cfgextra <<< "$entry"
    mkcfg "$stem" "$cfgextra"; rm -f "$OUT/$stem.txt"
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086
    mpirun -n "$RANKS" "$WTM" "$OUT/$stem.yaml" $flags $COUPLING_FLAGS $COLLECTOR_FLAGS -snes_stol 1e-8 \
        > "$OUT/$stem.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    its=$(grep -o 'Number of nonlinear iterations = [0-9]*' "$OUT/$stem.log" | awk '{s+=$6} END{print s+0}')
    cyc=$(grep -c 'per-cycle' "$OUT/$stem.log")
    printf "%-28s %10s %12s %12s\n" "$label" "$rc" "$wall" "$its"
    echo "$stem,$label,$rc,$wall,$its,$cyc" >> "$OUT/summary.csv"
done

echo
python3 report.py "$OUT"
