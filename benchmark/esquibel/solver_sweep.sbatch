#!/bin/bash -l
#SBATCH --job-name=wtm-solversweep
#SBATCH --partition=msilong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120gb
#SBATCH --time=2:30:00
#SBATCH --output=solversweep-%j.out
#
# CRITICAL (#87): can a cold 139M GW solve be made to converge? Anderson goes UNSTABLE near
# convergence (residual bottoms ~44 @ iter40 then oscillates 44-285; #85). Test every fix on the
# SAME 139M cold sub-step (maxiter 1, -snes_monitor), plus our Picard/Newton, then a WHY experiment.
set -uo pipefail
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
source "$WTM_ROOT/msi_env.sh" test
export OMP_NUM_THREADS=1 PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj
cd "$ESQ"
BASE="-wtm_fringe_source ksat -snes_stol 1e-6 -snes_max_it 80 -snes_monitor -snes_converged_reason"
CSV="$ESQ/solver_sweep_results.csv"; echo "grid,label,min_res,min_it,final_res,total_it,reason,wall_s" > "$CSV"

parse() {  # <log> -> "min_res min_it final_res total_it reason"
python3 - "$1" <<'PY'
import re, sys
txt=open(sys.argv[1],errors="ignore").read().replace("\r","\n")
its=[(int(m.group(1)),float(m.group(2))) for m in re.finditer(r"(\d+) SNES Function norm\s+([0-9.eE+-]+)",txt)]
r=re.search(r"(CONVERGED|DIVERGED)_[A-Z_]+",txt); reason=r.group(0) if r else "NONE"
if not its: print("nan nan nan nan "+reason); sys.exit()
mi=min(its,key=lambda x:x[1]); print(f"{mi[1]:.4g} {mi[0]} {its[-1][1]:.4g} {its[-1][0]} {reason}")
PY
}
runcfg() {  # <grid_dir> <cells> <label> <extra flags...>
  local D="$1" cells="$2" label="$3"; shift 3
  local CFG="$D/sweep.cfg"; sed 's/^total_cycles.*/total_cycles 1/;s/^cycles_to_save.*/cycles_to_save 1/;s/^maxiter.*/maxiter 1/' "$D/eq_tiled.cfg" > "$CFG"
  local LOG="$D/sweep_${label}.log"
  echo "############### $label ($cells cells) ###############"
  local t0=$(date +%s)
  timeout 800 mpiexec -ppn 32 -n 32 "$BIN" "$CFG" $BASE "$@" > "$LOG" 2>&1; local rc=$?
  local t1=$(date +%s); local wall=$((t1-t0))
  read mr mi fr ti reason < <(parse "$LOG")
  [ $rc -eq 124 ] && reason="TIMEOUT"
  echo "  rc=$rc wall=${wall}s  min_res=$mr @it$mi  final_res=$fr @it$ti  $reason"
  echo "${cells},${label},${mr},${mi},${fr},${ti},${reason},${wall}" >> "$CSV"
}

echo "===== SWEEP 1: stabilization options at 139M ====="
python3 make_tiled.py 19 19 >/dev/null 2>&1; D139="$ESQ/tiled/8569x16207"; C139=138877783
runcfg "$D139" "$C139" and_m10_baseline  -wtm_anderson -snes_anderson_m 10
runcfg "$D139" "$C139" and_m5            -wtm_anderson -snes_anderson_m 5
runcfg "$D139" "$C139" and_m3            -wtm_anderson -snes_anderson_m 3
runcfg "$D139" "$C139" and_m20           -wtm_anderson -snes_anderson_m 20
runcfg "$D139" "$C139" and_beta0.5       -wtm_anderson -snes_anderson_m 10 -snes_anderson_beta 0.5
runcfg "$D139" "$C139" and_beta0.3       -wtm_anderson -snes_anderson_m 10 -snes_anderson_beta 0.3
runcfg "$D139" "$C139" and_restart_periodic -wtm_anderson -snes_anderson_m 10 -snes_anderson_restart_type periodic -snes_anderson_restart 15
runcfg "$D139" "$C139" and_restart_diff  -wtm_anderson -snes_anderson_m 10 -snes_anderson_restart_type difference

echo "===== SWEEP 2: WHY -- equal cells (110M) + equal tiles (288), latitude span only ====="
python3 make_tiled.py 12 24 >/dev/null 2>&1; Dwide="$ESQ/tiled/5412x20472"   # wide, low lat 55-61
python3 make_tiled.py 24 12 >/dev/null 2>&1; Dtall="$ESQ/tiled/10824x10236"  # tall, high lat 55-67
runcfg "$Dwide" 110788 wide_lowlat_12x24  -wtm_anderson -snes_anderson_m 10
runcfg "$Dtall" 110788 tall_highlat_24x12 -wtm_anderson -snes_anderson_m 10

echo "===== DONE ====="; column -t -s, "$CSV"
