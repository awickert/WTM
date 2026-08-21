#!/bin/bash
# Find a stable big-dt config that reaches the TRUE equilibrium fast, to serve as the
# reference (cc-dt1 crawls: still 0.96 m/cycle, 53k cells drifting, at cycle 300). At a
# true steady state wtd_new=wtd_old, so the BE fixed point is identical at ANY dt -- a
# stable big-dt run reaches the SAME equilibrium, faster. Probe cc & tbar at dt 8/16/32 wk;
# report the max|Δwtd| decay so we can see which settles below 1e-2 / 1e-3 and when.
set -uo pipefail
cd "$(dirname "$0")"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
BIN=../../build/wtm.x; N=4; BASEDT=604800; WORK=results/algo/probe; mkdir -p "$WORK"
COMMON="-wtm_fringe_source ksat -snes_stol 1e-6"
declare -A FLAGS=( [cc]="-wtm_anderson -snes_anderson_restart_type none"
                   [tbar]="-wtm_anderson -snes_anderson_restart_type none -wtm_Tbar" )
for cfg in cc tbar; do
  for mult in 8 16 32; do
    dt=$((BASEDT*mult)); stem="$WORK/${cfg}_${mult}"
    sed "s#^fsm_on.*#fsm_on 0#;s#^total_cycles.*#total_cycles 150#;s#^save_nreport_interval.*#save_nreport_interval 150#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#;s#^deltat.*#deltat $dt#" eq_awickert.cfg > "${stem}.cfg"
    rm -f "${stem}.txt" "${stem}"_*.tif
    t0=$(date +%s.%N)
    timeout 500 mpirun -n "$N" "$BIN" "${stem}.cfg" ${FLAGS[$cfg]} $COMMON -wtm_eq_tol 0.001 > "${stem}.log" 2>&1
    rc=$?; t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.0f\", $t1-$t0}")
    cyc=$(awk 'NF>=5 && $1~/^[0-9]+$/{c=$1} END{print c+0}' "${stem}.txt")
    last=$(grep -E "per-cycle max" "${stem}.log" | tail -1 | grep -oE "= [0-9.e+-]+ m" | head -1)
    # first cycle to cross 1e-2 and 1e-3
    c2=$(grep -E "per-cycle max" "${stem}.log" | awk '{v=$5+0; if(v<1e-2){print NR; exit}}')
    c3=$(grep -E "per-cycle max" "${stem}.log" | awk '{v=$5+0; if(v<1e-3){print NR; exit}}')
    printf "%-5s dt=%-3swk rc=%-3s cyc=%-4s wall=%-4ss  last_max|Δ|%-14s  <1e-2@%-5s <1e-3@%-5s\n" \
           "$cfg" "$mult" "$rc" "$cyc" "$wall" "$last" "${c2:-never}" "${c3:-never}"
  done
done
echo "PROBE DONE"