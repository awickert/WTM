#!/bin/bash
# ALGORITHMIC speedup over "corrected Callaghan" on Esquibel (384,703 cells) -- the real
# runtime number (island is shape-only; it is the wrong regime for Tbar). "Corrected
# Callaghan" = OUR binary running Callaghan's algorithm cleanly: matrix-free Anderson,
# 1st-order backward Euler, restart OFF, no Tbar, no TR-BDF2 -- but with our mass-
# conservation fix, so the comparison isolates ALGORITHM from formulation-accuracy.
# Each comparison config changes exactly ONE thing on top of that baseline.
#
# EQUILIBRIUM half (this script): cc  vs  cc+Tbar.  (TR-BDF2 rings at equilibrium -> it
# belongs in the TRANSIENT half, algo_transient.sh, not here.)
#
# Cores FIXED (config-to-config delta = pure algorithm; everything beyond it is
# parallelization). fsm OFF (isolate the GW solve). dt swept 1..32x the 1-week base.
#
# METRIC -- two independent signals, because neither alone is honest:
#   (1) cyc2eq : cycles until the model's per-cycle max|Δwtd| auto-stop fires (eq_tol
#       0.01 m, 2 consecutive). This is the SPEED signal: fewer cycles at a bigger stable
#       dt is the win. At true steady state wtd_new=wtd_old so this -> 0 for ANY dt.
#   (2) maxabs : max|w - w_ref| of the run's FINAL raster vs a trusted reference
#       equilibrium (cc at dt1, tight tol). This is the dt-INDEPENDENT CORRECTNESS gate:
#       at equilibrium every dt/order lands on the SAME steady state, so a small maxabs
#       confirms the run really equilibrated (not a premature stop or a limit cycle).
# A run only "wins" if it took fewer cycles AND its maxabs is small. Both are reported
# always, so a premature-looking stop is visible rather than hidden.
#
#   Usage: ./algo_speedup.sh [N_cores]      (default 4)   FORCE_REF=1 to rebuild the ref
set -uo pipefail
cd "$(dirname "$0")"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
BIN=../../build/wtm.x
N=${1:-4}
BASEDT=604800                 # 1 week, in seconds
CAP=200                       # total_cycles ceiling per run
RUN_TIMEOUT=700               # wall-clock guard per run (s)
TOL_ABS=0.1                   # [m] max|w-w_ref| below this => "reached equilibrium"
ROOT=results/algo
WORK="$ROOT/equilibrium"      # cc/tbar dt-sweep outputs land here
REFD="$ROOT/reference"        # trusted reference equilibrium lives here
mkdir -p "$WORK" "$REFD"

# Common flags held fixed across ALL configs (formulation constant); only the tail varies.
COMMON="-wtm_fringe_source ksat -snes_stol 1e-6"
declare -A FLAGS=(
  [cc]="-wtm_anderson -snes_anderson_restart_type none"
  [tbar]="-wtm_anderson -snes_anderson_restart_type none -wtm_Tbar" )

mkcfg() {  # mkcfg <stem> <dt> <eq_tol> <cap>
  local stem=$1 dt=$2 eqt=$3 cap=$4
  sed "s#^fsm_on.*#fsm_on 0#;s#^total_cycles.*#total_cycles $cap#;s#^save_nreport_interval.*#save_nreport_interval $cap#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#;s#^deltat.*#deltat $dt#" eq_awickert.cfg > "${stem}.cfg"
  echo "$eqt"
}
runit() {  # runit <stem> <flags...>  -> echoes "rc wall lastcycle finaltif"
  local stem=$1; shift
  rm -f "${stem}.txt" "${stem}"_*.tif
  local t0 t1 rc
  t0=$(date +%s.%N)
  timeout "$RUN_TIMEOUT" mpirun -n "$N" "$BIN" "${stem}.cfg" "$@" > "${stem}.log" 2>&1
  rc=$?; t1=$(date +%s.%N)
  local wall cyc ftif
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  cyc=$(awk 'NF>=5 && $1~/^[0-9]+$/{c=$1} END{print c+0}' "${stem}.txt" 2>/dev/null)
  ftif=$(ls -1 "${stem}"_*.tif 2>/dev/null | sort | tail -1)
  echo "$rc $wall ${cyc:-0} ${ftif:-none}"
}

# ---- (0) trusted reference equilibrium: corrected-Callaghan, dt1, tight tol -------------
REF="$REFD/ref"
if [ "${FORCE_REF:-0}" = 1 ] || [ ! -s "${REF}_eq.tif" ]; then
  echo "### building reference equilibrium (cc, dt=1wk, eq_tol 1e-3, cap 300)..."
  mkcfg "$REF" "$BASEDT" "0.001" 300 >/dev/null
  sed -i "s#^report_interval.*#report_interval 50#" "${REF}.cfg"
  read rc wall cyc ftif <<<"$(runit "$REF" ${FLAGS[cc]} $COMMON -wtm_eq_tol 0.001)"
  if [ "$rc" -ne 0 ] || [ "$ftif" = none ]; then echo "REF build failed (rc=$rc). abort."; exit 1; fi
  cp "$ftif" "${REF}_eq.tif"
  echo "### reference: cyc=$cyc wall=${wall}s -> ${REF}_eq.tif"
fi
REFTIF="${REF}_eq.tif"

# ---- (1) the sweep ----------------------------------------------------------------------
echo "cores=$N  domain=Esquibel(384703)  fsm=off  eq_tol=0.01m/2cyc  cap=$CAP  tol_abs=${TOL_ABS}m"
printf "%-6s %-6s %-9s %-8s %-8s %-11s %-11s\n" config dt_wk result cyc2eq wall_s maxabs_m rms_m
for cfg in cc tbar; do
  for mult in 1 2 4 8 16 32; do
    dt=$((BASEDT*mult)); stem="$WORK/${cfg}_${mult}"
    mkcfg "$stem" "$dt" "0.01" "$CAP" >/dev/null
    read rc wall cyc ftif <<<"$(runit "$stem" ${FLAGS[$cfg]} $COMMON -wtm_eq_tol 0.01)"
    maxabs=nan; rms=nan
    if [ "$ftif" != none ]; then
      read _ maxabs _ rms _ <<<"$(python3 wtd_diff.py "$ftif" "$REFTIF" 2>/dev/null | sed 's/[a-z]*=//g')"
      maxabs=${maxabs:-nan}; rms=${rms:-nan}
    fi
    # classify
    if   [ "$rc" -eq 124 ]; then res=TIMEOUT
    elif [ "$rc" -ne 0 ];   then res=DIVERGE
    elif awk "BEGIN{exit !($maxabs<$TOL_ABS)}" 2>/dev/null; then
         if [ "${cyc:-$CAP}" -lt "$CAP" ]; then res=EQ; else res=EQ_slow; fi
    else res=noeq
    fi
    printf "%-6s %-6s %-9s %-8s %-8s %-11s %-11s\n" "$cfg" "$mult" "$res" "${cyc:-?}" "$wall" "$maxabs" "$rms"
  done
done
echo "DONE. EQ=auto-stopped AND matches ref (<${TOL_ABS}m); EQ_slow=matches ref but ran to cap;"
echo "      noeq=ran to cap far from ref; DIVERGE/TIMEOUT=solver failed/too slow."
echo "The win: same maxabs (converged), FEWER cyc2eq and less wall at a bigger stable dt."