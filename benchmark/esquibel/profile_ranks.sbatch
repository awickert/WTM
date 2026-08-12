#!/bin/bash -l
#SBATCH --job-name=wtm-prof-ranks
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=1:10:00
#SBATCH --output=profranks-%j.out
#
# Is the 139M cold-solve cliff a DOMAIN-DECOMPOSITION artifact (how the grid slices across ranks /
# where slices fall vs tile seams) or intrinsic to problem size? Run the SAME 139M cold sub-step at
# rank counts 1/8/16/32 on ONE node, capped at 40 Anderson iters, and compare convergence RATE.
#   n=1 converges (or much faster) but n=32 stalls -> DECOMPOSITION/rank artifact
#   all rank counts stall the same way          -> intrinsic to size (decomposition-independent)
# -snes_monitor = residual per iter; -dm_view = the DMDA rank layout; -log_view = where time goes.
# Uses the current MSI binary (snes_monitor is a runtime PETSc flag; no rebuild needed).
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"
python3 make_tiled.py 19 19 >/dev/null 2>&1
D="$ESQ/tiled/8569x16207"; CFG="$D/profr.cfg"
sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/' "$D/eq_tiled.cfg" > "$CFG"
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -snes_max_it 40 -snes_monitor -snes_converged_reason"
echo "job $SLURM_JOB_ID on $(hostname); 139M one cold sub-step; rank sweep 1/8/16/32"
for n in 1 8 16 32; do
  echo "############### ranks = $n ###############"
  extra="-log_view"; [ "$n" = 32 ] && extra="-log_view -dm_view"
  t0=$(date +%s)
  mpiexec -ppn "$n" -n "$n" "$BIN" "$CFG" $OF $extra > "$D/profr_n${n}.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "  rc=$rc wall=$((t1-t0))s"
  echo "  --- residual @ iters 0,5,10,20,30,40 + reason ---"
  tr '\r' '\n' < "$D/profr_n${n}.log" | LC_ALL=C grep -aE "SNES Function norm" | grep -avE "threads\)" | \
    awk '{it=$1; norm=$5} it==0||it==5||it==10||it==20||it==30||it==40{printf "    it %-3s  %s\n", it, norm}'
  tr '\r' '\n' < "$D/profr_n${n}.log" | LC_ALL=C grep -aoE "(CONVERGED|DIVERGED)_[A-Z_]+ Number of nonlinear iterations = [0-9]+" | tail -1 | sed 's/^/    /'
  echo "  --- SNESSolve time (log_view) ---"
  tr '\r' '\n' < "$D/profr_n${n}.log" | LC_ALL=C grep -aE "^SNESSolve" | awk '{print "    SNESSolve "$4" s, "$NF" Mflop/s"}' | head -1
done
echo "=== DMDA rank layout (n=32, from -dm_view) ==="
tr '\r' '\n' < "$D/profr_n32.log" | LC_ALL=C grep -aiE "Processor|sizes|DMDA|M .* N " | head -6
echo "=== DONE ==="
