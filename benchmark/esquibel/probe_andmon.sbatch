#!/bin/bash -l
#SBATCH --job-name=wtm-andmon
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=0:45:00
#SBATCH --output=andmon-%j.out
#
# Watch Anderson's INTERNAL mixing (-snes_anderson_monitor) alongside the residual at 139M, to
# (a) see the coefficient blow-up that IS the flail, and (b) get the rho = |F_k|/|F_{k-1}| turnover
# curve for setting the hand-off trigger empirically. Baseline vs restart-difference.
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"
python3 make_tiled.py 19 19 >/dev/null 2>&1
D="$ESQ/tiled/8569x16207"; CFG="$D/andmon.cfg"
sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/' "$D/eq_tiled.cfg" > "$CFG"
BASE="-wtm_fringe_source ksat -snes_stol 1e-6 -snes_max_it 80 -snes_monitor -snes_anderson_monitor -snes_converged_reason"
run() {
  local label="$1"; shift
  local LOG="$D/andmon_${label}.log"
  echo "############### $label ###############"
  local t0=$(date +%s)
  mpiexec -ppn 32 -n 32 "$BIN" "$CFG" -wtm_anderson $BASE "$@" > "$LOG" 2>&1; local rc=$?
  echo "  rc=$rc wall=$(( $(date +%s)-t0 ))s"
  echo "  --- residual + rho=|F_k|/|F_k-1| (every 5th iter) ---"
  tr '\r' '\n' < "$LOG" | LC_ALL=C grep -aE "SNES Function norm" | grep -avE "threads\)" \
    | awk '{it=$1;n=$5; if(NR>1)r=n/p; p=n; if(it%5==0||it<3)printf "    it %-3s  res=%-12s  rho=%.3f\n",it,n,r}'
  tr '\r' '\n' < "$LOG" | LC_ALL=C grep -aoE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" | tail -1 | sed 's/^/    /'
}
run baseline_m10           -snes_anderson_m 10
run restart_difference     -snes_anderson_m 10 -snes_anderson_restart_type difference
echo "=== DONE ==="
