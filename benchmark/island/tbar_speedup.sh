#!/bin/bash
# ALGORITHMIC speedup over baseline Anderson (Callaghan-equivalent: matrix-free Anderson, 1st-order BE)
# from -wtm_Tbar and -wtm_tr_bdf2. FIXED cores (n=N) so the config-to-config difference is PURELY
# algorithmic -- everything beyond it is parallelization. Island equilibrium spin-up, fsm OFF (isolate
# the GW solve). Sweep dt; each config's fewer-cycles-at-a-bigger-stable-dt is the win. Metric:
# cycles-to-equilibrium (eq_tol auto-stop) and wall-clock. baseline hits a stiffness dt-ceiling that
# Tbar/TR-BDF2 should lift.  Usage: ./tbar_speedup.sh [N]
set -uo pipefail
cd "$(dirname "$0")"; export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
BIN=../../build/wtm.x; N=${1:-4}; BASEDT=604800; WORK=results/tbar_cmp; mkdir -p "$WORK"
declare -A FLAGS=(
  [base]="-wtm_anderson"
  [tbar]="-wtm_anderson -wtm_Tbar"
  [trbdf2]="-wtm_anderson -wtm_tr_bdf2"
  [both]="-wtm_anderson -wtm_Tbar -wtm_tr_bdf2" )
echo "cores=$N  domain=island(8775)  fsm=off  eq_tol=default(0.01m)  maxiter=50"
printf "%-8s %-6s %-9s %-8s %-8s\n" config dt_wk result cyc2eq wall_s
for cfg in base tbar trbdf2 both; do
  for mult in 1 2 4 8 16 32; do
    dt=$((BASEDT*mult)); stem="$WORK/${cfg}_${mult}"
    sed "s#^deltat.*#deltat $dt#;s#^fsm_on.*#fsm_on 0#;s#^total_cycles.*#total_cycles 300#;s#^cycles_to_save.*#cycles_to_save 1#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#" eq_awickert.cfg > "$stem.cfg"
    t0=$(date +%s.%N)
    timeout 200 mpirun -n "$N" "$BIN" "$stem.cfg" ${FLAGS[$cfg]} -wtm_fringe_source ksat -snes_stol 1e-6 > "$stem.log" 2>&1
    rc=$?; t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
    cyc=$(awk 'NF>=5 && $1~/^[0-9]+$/{c=$1} END{print c+0}' "$stem.txt" 2>/dev/null)
    if [ "$rc" -eq 124 ]; then res=TIMEOUT; elif [ "$rc" -ne 0 ]; then res=DIVERGE; elif [ "${cyc:-300}" -ge 299 ]; then res=noeq; else res=EQ; fi
    printf "%-8s %-6s %-9s %-8s %-8s\n" "$cfg" "$mult" "$res" "${cyc:-?}" "$wall"
  done
done
echo "DONE. EQ=reached equilibrium (cyc2eq cycles); DIVERGE=solver failed at this dt; noeq=ran 300 cyc w/o settling."
