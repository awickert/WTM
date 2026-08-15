#!/bin/bash -l
#SBATCH --job-name=kc-vs-adapt
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=kc_vs_adapt-%j.out
#
# Esquibel (384k) COLD -> equilibrium: KCallaghan-style fixed-1-week backward-Euler (cc, our
# corrected code) vs our adaptive TR-BDF2 -- the head-to-head the shelved PR would have argued.
# All arms: same domain/physics/BC (ghost boundary, surface-exfiltration clamp, ksat fringe),
# same inner tol (snes_stol 1e-6), same frac equilibrium stop. Only the integrator/adaptive
# flags differ. Cost metric = SNES iterations (rank-independent); wall reported too (agsmall
# shared slice -> first-look wall, co-tenant noise possible; NOT --exclusive per Andy: pristine
# timing waits for finalized code).
set -uo pipefail
N="${N:-16}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"
BIN="$WTM_ROOT/build/wtm.x"
ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1
export PROJ_DATA="${PROJ_DATA:-/usr/share/proj}" PROJ_LIB="${PROJ_LIB:-/usr/share/proj}"
cd "$ESQ"

# --- STALE-BINARY GUARD: the detached-controller print string must be present -----------------
# grep -a scans the binary directly (no dependency on `strings`, which is not reliably on PATH
# after msi_env.sh's module purge + conda activate -- that misfired the guard on the first run).
if ! grep -qa "on the %s integrator" "$BIN"; then
  echo "FATAL: build/wtm.x does not contain the detached-controller code (stale binary). Rebuild."; exit 1
fi

R=results/kc_vs_adapt; mkdir -p "$R"
CSV="$R/summary.csv"; echo "arm,rc,wall_s,snes_its,stop_cycle,final_tif" > "$CSV"
BASE="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6 -wtm_surface_exfiltration_to_runoff -wtm_eq_tol 0.01 -wtm_eq_metric frac"

run() { # label  extra_flags...
  local lbl="$1"; shift
  local stem="$R/$lbl"
  # cold (supplied_wt 0), fsm on (KCallaghan's equilibrium config), 1-week base dt, 1000-cycle
  # backstop so the frac eq-stop (not the cap) ends each run.
  sed "s#^deltat.*#deltat 604800#;s#^total_cycles.*#total_cycles 1000#;s#^cycles_to_save.*#cycles_to_save 1000#;s#^supplied_wt.*#supplied_wt 0#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "${stem}.cfg"
  rm -f "${stem}.txt" "${stem}"_*.tif
  echo "=== $lbl : $BASE $* ==="
  local t0=$(date +%s.%N)
  mpiexec -ppn "$N" -n "$N" "$BIN" "${stem}.cfg" $BASE "$@" > "${stem}.log" 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
  local its=$(grep -o "Number of nonlinear iterations = [0-9]*" "${stem}.log" | awk '{s+=$6} END{print s+0}')
  local stop=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1} END{print c}' "${stem}.txt")
  local tif=$(ls -1 "${stem}"_*.tif 2>/dev/null | tail -1)
  echo "$lbl,$rc,$wall,$its,$stop,$tif" | tee -a "$CSV"
}

echo "### Esquibel cold->eq | N=$N | $(hostname) | job ${SLURM_JOB_ID:-?} ###"
run kcallaghan_cc                                                     # cc fixed 1 week (KCallaghan style)
run fixed_tr      -wtm_tr_bdf2                                        # TR-BDF2 fixed 1 week (2nd order, no adapt)
run adaptive_tr_0p5 -wtm_tr_bdf2 -wtm_dt_adaptive -wtm_dt_tol 0.5     # adaptive TR-BDF2 (ours)
run adaptive_tr_2   -wtm_tr_bdf2 -wtm_dt_adaptive -wtm_dt_tol 2.0     # adaptive TR-BDF2, looser tol

echo "### DONE ###"; cat "$CSV"
