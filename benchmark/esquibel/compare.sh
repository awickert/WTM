#!/usr/bin/env bash
# Matched-tolerance (snes_stol 1e-6) full-Esquibel comparison, two wall metrics per config:
#   (1) wall to MATCH v2.0.1's accuracy  -- v2.0.1's own floor, derived from its run (NOT
#       hardcoded 0.59: the mass-leak floor differs on the full domain)
#   (2) wall to COMPLETION (full total_cycles)
# Mirrors benchmark/island/compare.sh. Stage the domain first: ./make_esquibel.py
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
OURS=${OURS:-/home/awickert/models/WTM/build/wtm.x}
KC=${KC:-/home/awickert/models/kcallaghan-wtm/build/wtm.x}
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6"
KF="-snes_mf -snes_type anderson -snes_stol 1e-6"
RANKS=${RANKS:-"1 2 4 8"}
TIMEOUT=${TIMEOUT:-7200}          # seconds per run; Esquibel is large -- adjust as needed
R=results/cmp; mkdir -p "$R"
[ -f domain/Esquibel_010000_topography.tif ] || { echo "domain not staged -- run ./make_esquibel.py first"; exit 1; }

crossover() { awk -v t=$1 'NF>=5&&$1~/^[0-9]+$/{if($5+0<=t){print $1; exit}}' "$2"; }
run() { # base bin ranks ncyc flags... -> "wall_ms finalcyc finalD iters"
  local base=$1 bin=$2 nr=$3 nc=$4; shift 4
  sed "s/^total_cycles.*/total_cycles $nc/;s/^cycles_to_save.*/cycles_to_save $nc/;s#results/eq_[a-z]*#$R/$base#g" \
      "$([ "$bin" = "$KC" ] && echo eq_kcallaghan.cfg || echo eq_awickert.cfg)" > "$R/$base.cfg"
  : > "$R/$base.txt"
  local t0=$(date +%s%N); timeout $TIMEOUT mpirun -n $nr "$bin" "$R/$base.cfg" $* > "$R/$base.log" 2>&1; local rc=$?; local t1=$(date +%s%N)
  local it=$(grep -a "nonlinear iterations" "$R/$base.log"|grep -oE "= [0-9]+"|grep -oE "[0-9]+"|awk '{s+=$1;n++}END{if(n)printf"%.1f",s/n;else print"-"}')
  echo "$(( (t1-t0)/1000000 )) $(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1;d=$5}END{print c" "d}' "$R/$base.txt") $it"
}
NC=$(awk -F' *' '/^total_cycles/{print $2}' eq_awickert.cfg)  # completion cycle count

echo "### v2.0.1 full run (defines the match target = its floor) ###"
read vw vc vd vi <<<"$(run v201_full $KC 1 $NC $KF)"
THR=$(awk "BEGIN{printf \"%.6g\", $vd*1.001}")
echo "  v2.0.1 n=1: completion ${vw}ms  floorΔ=$vd  iters=$vi   -> match target Δ<=$THR"
COV=$(crossover $THR "$R/v201_full.txt")
read vmw vmc vmd vmi <<<"$(run v201_match $KC 1 $((COV+1)) $KF)"
echo "  v2.0.1 n=1: match ${vmw}ms (cycle $vmc)"

echo "### ours: full (completion) + to-match, across ranks ###"
declare -A CW FD IT MW; CO=""
for n in $RANKS; do
  read w c d it <<<"$(run ours_full_n$n $OURS $n $NC $OF)"
  CW[$n]=$w; FD[$n]=$d; IT[$n]=$it
  echo "  ours n=$n: completion ${w}ms  finalΔ=$d  iters=$it"
  [ -z "$CO" ] && CO=$(crossover $THR "$R/ours_full_n$n.txt")
done
echo "  (ours crossover to Δ<=$THR: cycle $CO)"
for n in $RANKS; do
  read w c d it <<<"$(run ours_m_n$n $OURS $n $((CO+1)) $OF)"
  MW[$n]=$w; echo "  ours n=$n: match ${w}ms (cycle $c)"
done

echo
echo "########## MATCHED-TOLERANCE (snes_stol 1e-6) ESQUIBEL COMPARISON ##########"
printf "%-14s %14s %16s %12s %10s\n" "config" "match(ms)" "completion(ms)" "finalΔ" "iters/solve"
printf "%-14s %14s %16s %12s %10s\n" "v2.0.1 n=1" "$vmw" "$vw" "$vd" "$vi"
for n in $RANKS; do printf "%-14s %14s %16s %12s %10s\n" "ours n=$n" "${MW[$n]}" "${CW[$n]}" "${FD[$n]}" "${IT[$n]}"; done
echo "(match target Δ<=$THR = v2.0.1's floor +0.1%; v2.0.1 may SEGV at n>=4)"
