#!/bin/bash -l
#SBATCH --job-name=wtm-probe
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=0:40:00
#SBATCH --output=probe-%j.out
#
# Diagnose the 139M cold-solve non-convergence (task: solver scaling). ONE cold sub-step
# (maxiter 1), run twice: default snes_max_it (50) to reproduce, then 200 to test whether it
# is just the iteration budget. -snes_monitor + -snes_converged_reason capture the residual
# trajectory + reason live (flushed per iter -> survives an abort, unlike WTM's buffered print).
#   CONVERGES at 200 but not 50  -> pure iteration-budget (trivial: raise max_it at scale)
#   DIVERGES (residual rises/stalls) even at 200 -> true large-N divergence (deep; #49/precond)
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"
python3 make_tiled.py 19 19 >/dev/null 2>&1    # idempotent (already staged)
D="$ESQ/tiled/8569x16207"; CFG="$D/probe.cfg"
sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/' "$D/eq_tiled.cfg" > "$CFG"
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -snes_monitor -snes_converged_reason"
echo "job $SLURM_JOB_ID on $(hostname); 139M one cold sub-step"
for MI in 50 200; do
  echo "########## snes_max_it = $MI ##########"
  t0=$(date +%s)
  mpiexec -ppn 32 -n 32 "$BIN" "$CFG" $OF -snes_max_it $MI > "$D/probe_mi${MI}.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "  rc=$rc wall=$((t1-t0))s"
  echo "  --- SNES residual trajectory (first + last few) ---"
  tr '\r' '\n' < "$D/probe_mi${MI}.log" | grep -aE "SNES Function norm|CONVERGED|DIVERGED|nonlinear iterations|not converged" | grep -avE "threads\)" | sed 's/^/    /' | (head -5; echo "    ..."; tail -6)
done
echo "=== DONE ==="
