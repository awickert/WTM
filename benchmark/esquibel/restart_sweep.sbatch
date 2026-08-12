#!/bin/bash -l
#SBATCH --job-name=wtm-restartsweep
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=1:30:00
#SBATCH --output=restartsweep-%j.out
# Is periodic-restart robust across periods, or is 15 a lucky number? Can the ADAPTIVE (difference)
# restart be tuned to fire usefully? One cold sub-step at 139M, m=10 throughout (consistency-safe).
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"; python3 make_tiled.py 19 19 >/dev/null 2>&1
D="$ESQ/tiled/8569x16207"; CFG="$D/rs.cfg"
sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/;s/^fsm_on.*/fsm_on 0/' "$D/eq_tiled.cfg" > "$CFG"
BASE="-wtm_anderson -snes_anderson_m 10 -wtm_fringe_source ksat -snes_stol 1e-6 -snes_max_it 90 -snes_converged_reason"
CSV="$ESQ/restart_sweep_results.csv"; echo "label,reason,total_it,wall_s" > "$CSV"
run() { local label="$1"; shift; local LOG="$D/rs_$label.log"
  local t0=$(date +%s); mpiexec -ppn 32 -n 32 "$BIN" "$CFG" $BASE "$@" > "$LOG" 2>&1; local rc=$?; local w=$(($(date +%s)-t0))
  local line=$(grep -aoE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" "$LOG" | tail -1)
  local reason=$(echo "$line" | grep -oE "(CONVERGED|DIVERGED)_[A-Z_]+"); local it=$(echo "$line" | grep -oE "[0-9]+$")
  echo "  $label: ${reason:-?} @${it:-?} its, ${w}s (rc=$rc)"; echo "${label},${reason:-?},${it:-?},${w}" >> "$CSV"
}
for P in 10 15 20 25; do run periodic_$P -snes_anderson_restart_type periodic -snes_anderson_restart $P; done
for K in 1 2 3;    do run difference_it$K -snes_anderson_restart_type difference -snes_anderson_restart_it $K; done
echo DONE; column -t -s, "$CSV"
