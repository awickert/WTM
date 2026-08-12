#!/bin/bash -l
#SBATCH --job-name=wtm-confirm139
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=0:50:00
#SBATCH --output=confirm139-%j.out
# Confirm BOTH periodic-default and rho-adaptive converge a cold 139M solve, on a healthy node,
# with -snes_monitor (see progress / catch a hang) and a bounded max_it. One cold sub-step.
set -uo pipefail
WR="$HOME/models/WTM"; source "$WR/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$WR/benchmark/esquibel"; python3 make_tiled.py 19 19 >/dev/null 2>&1
D="$WR/benchmark/esquibel/tiled/8569x16207"
sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/;s/^fsm_on.*/fsm_on 0/' "$D/eq_tiled.cfg" > "$D/c139.cfg"
run() { echo "############### $1 ###############"; local L="$D/c139_$1.log"; shift
  local t0=$(date +%s)
  mpiexec -ppn 32 -n 32 "$WR/build/wtm.x" "$D/c139.cfg" -wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -snes_max_it 120 -snes_monitor -snes_converged_reason "$@" > "$L" 2>&1
  echo "  rc=$? wall=$(($(date +%s)-t0))s"
  tr '\r' '\n' < "$L" | grep -aE "SNES Function norm" | grep -avE threads | awk 'NR==1||NR%10==0' | awk '{printf "    it %-4s %s\n",$1,$5}'
  tr '\r' '\n' < "$L" | grep -aoE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" | tail -1 | sed 's/^/    /'
  tr '\r' '\n' < "$L" | grep -aE "adaptive_restart: best residual" | tail -1 | sed 's/^/    /'
}
run periodic_default
run adaptive_restart -wtm_adaptive_restart
echo DONE
