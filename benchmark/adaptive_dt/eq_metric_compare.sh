#!/usr/bin/env bash
# Equilibrium-stop metric comparison (see README): for -wtm_eq_metric max|rms|frac on the adaptive-dt path,
# measure COST (stop cycle, GW solves, wall) vs PRECISION (final water table vs a long no-stop reference).
# Warm Esquibel, -20% dry settle. Re-runnable after #108 (generalized adaptive) to re-check the trade.
#   N=1 (serial, laptop clean room) or N=16 (MPI/MSI);  EQTOL, CAP, REFCAP overridable via env.
set -uo pipefail
N="${N:-1}"; EQTOL="${EQTOL:-0.05}"; CAP="${CAP:-30}"; REFCAP="${REFCAP:-40}"; EQFRAC="${EQFRAC:-0.001}"
WTM_ROOT="$(cd "$HOME/models/WTM" && pwd)"; BIN="$WTM_ROOT/build/wtm.x"; ESQ="$WTM_ROOT/benchmark/esquibel"
OUT="$WTM_ROOT/benchmark/adaptive_dt/results/eqmetric"; mkdir -p "$OUT"
cd "$ESQ"
PERT="$OUT/pert_dry"; python3 perturb_pet.py domain "$PERT" results/algo/transient/w_eq_correct.tif 0.8 >/dev/null 2>&1
BASE="-wtm_tr_bdf2 -wtm_dt_adaptive -wtm_dt_tol 5 -wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff -snes_stol 1e-6"
CSV="$OUT/eq_metric_compare.csv"; echo "run,rc,stop_cycle,solves,wall_s" > "$CSV"

run() {  # label cap extra-flags
  local L="$1" C="$2" EX="$3" stem="$OUT/$L"
  cat > ${stem}.cfg <<CFG
run_type transient
fsm_on 0
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat 604800
total_cycles $C
save_nreport_interval 1
report_interval            8
fdepth_a           100
fdepth_b           150
fdepth_fmin        2.5
time_start         010000
time_end           010000
surfdatadir $PERT/
region             Esquibel
supplied_wt 0
textfilename ${stem}.txt
outfile_prefix ${stem}_
CFG
  rm -f ${stem}_*.tif ${stem}.txt
  local t0=$(date +%s.%N)
  mpiexec -n "$N" "$BIN" ${stem}.cfg $BASE $EX > ${stem}.log 2>&1
  local rc=$?; local t1=$(date +%s.%N)
  local wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  local its=$(grep -aoc "Number of nonlinear iterations" ${stem}.log)
  local sc=$(grep -ao "stopping at cycle [0-9]*" ${stem}.log | grep -o "[0-9]*" | head -1); [ -z "$sc" ] && sc="none(cap$C)"
  echo "$L,$rc,$sc,$its,$wall" | tee -a "$CSV"
}

run ref  "$REFCAP" "-wtm_eq_tol 0"
run max   "$CAP"   "-wtm_eq_metric max  -wtm_eq_tol $EQTOL"
run rms   "$CAP"   "-wtm_eq_metric rms  -wtm_eq_tol $EQTOL"
run frac  "$CAP"   "-wtm_eq_metric frac -wtm_eq_tol $EQTOL -wtm_eq_frac $EQFRAC"

echo "=== precision: each stop-state vs the no-stop reference (highest-cycle tif) ==="
python3 - "$OUT" <<'PY'
import sys, os, glob, numpy as np, rasterio
D=sys.argv[1]
last=lambda l: (sorted(glob.glob(os.path.join(D,"%s_*.tif"%l)) ) or [None])[-1]
ref=rasterio.open(last("ref")).read(1).astype(float); fin0=np.isfinite(ref)
print("%-6s %-26s %10s %10s"%("metric","stop_state","mean|Δref|","max|Δref|"))
for l in ("max","rms","frac"):
    fp=last(l)
    if not fp: print("%-6s (no tif)"%l); continue
    a=rasterio.open(fp).read(1).astype(float); fin=fin0&np.isfinite(a)
    print("%-6s %-26s %10.4f %10.3f"%(l,os.path.basename(fp),
          float(np.mean(np.abs((a-ref)[fin]))), float(np.max(np.abs((a-ref)[fin])))))
PY
echo "EQ_METRIC_COMPARE DONE"
