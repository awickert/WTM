#!/bin/bash -l
#SBATCH --job-name=wtm-flagfix
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=1:00:00
#SBATCH --output=flagfix-%j.out
# Confirm the one-flag Anderson fix converges MULTIPLE consecutive cold sub-steps at 139M (not just
# the first). restart_periodic keeps m=10 (cross-rank-consistent default); m=20 as comparison.
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"; python3 make_tiled.py 19 19 >/dev/null 2>&1
D="$ESQ/tiled/8569x16207"; CFG="$D/flagfix.cfg"
sed 's/^total_cycles.*/total_cycles 2/;s/^cycles_to_save.*/cycles_to_save 2/;s/^maxiter.*/maxiter 5/;s/^fsm_on.*/fsm_on 0/' "$D/eq_tiled.cfg" > "$CFG"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -snes_converged_reason"
run() { echo "### $1 ###"; local LOG="$D/flagfix_$1.log"; shift
  t0=$(date +%s); mpiexec -ppn 32 -n 32 "$BIN" "$CFG" $BASE "$@" > "$LOG" 2>&1; echo "  rc=$? wall=$(($(date +%s)-t0))s"
  echo "  per-substep reason (want ALL CONVERGED):"; tr '\r' '\n' < "$LOG" | grep -aoE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" | sed 's/^/    /'
}
run restart_periodic -snes_anderson_restart_type periodic -snes_anderson_restart 15
run m20 -snes_anderson_m 20
echo DONE
