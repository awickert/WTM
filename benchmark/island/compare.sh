#!/usr/bin/env bash
# Matched-tolerance (snes_stol 1e-6) island comparison.
# For each config: wall to MATCH v2.0.1 accuracy (Δ<=0.5904) and wall to COMPLETION (100 cyc).
set -uo pipefail
cd /home/awickert/models/WTM/benchmark/island
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
OURS=/home/awickert/models/WTM/build/wtm.x
KC=/home/awickert/models/kcallaghan-wtm/build/wtm.x
OF="-wtm_anderson -wtm_fringe_source ksat -snes_stol 1e-6"
KF="-snes_mf -snes_type anderson -snes_stol 1e-6"
THR=0.5904
R=results/cmp; mkdir -p "$R"

# crossover cycle: first cycle with Δ(col5) <= THR, from a trajectory file
crossover() { awk -v t=$THR 'NF>=5&&$1~/^[0-9]+$/{if($5+0<=t){print $1; exit}}' "$1"; }
# timed run: base bin ranks ncyc flags...  -> prints "wall_ms finalcyc finalD iters"
run() {
  local base=$1 bin=$2 nr=$3 nc=$4; shift 4
  sed "s/^total_cycles.*/total_cycles $nc/;s/^cycles_to_save.*/cycles_to_save $nc/;s#results/eq_[a-z]*#$R/$base#g" \
      "$([ "$bin" = "$KC" ] && echo eq_kcallaghan.cfg || echo eq_awickert.cfg)" > "$R/$base.cfg"
  : > "$R/$base.txt"
  local t0=$(date +%s%N); timeout 300 mpirun -n $nr "$bin" "$R/$base.cfg" $* > "$R/$base.log" 2>&1; local rc=$?; local t1=$(date +%s%N)
  local w=$(( (t1-t0)/1000000 ))
  local fc=$(awk 'NF>=5&&$1~/^[0-9]+$/{c=$1;d=$5}END{print c" "d}' "$R/$base.txt")
  local it=$(grep -a "nonlinear iterations" "$R/$base.log"|grep -oE "= [0-9]+"|grep -oE "[0-9]+"|awk '{s+=$1;n++}END{if(n)printf"%.1f",s/n;else print"-"}')
  echo "$w $fc $it"
}

echo "### PHASE 1: full 100-cycle runs (completion) + trajectories ###"
declare -A COMPL FIN ITERS
# ours n=1,2,4,8
for n in 1 2 4 8; do
  read w fc fd it <<<"$(run ours_full_n$n $OURS $n 100 $OF)"
  COMPL[ours$n]=$w; FIN[ours$n]=$fd; ITERS[ours$n]=$it
  echo "  ours n=$n: completion ${w}ms  finalΔ=$fd  iters=$it"
done
read w fc fd it <<<"$(run v201_full $KC 1 100 $KF)"
COMPL[v201]=$w; FIN[v201]=$fd; ITERS[v201]=$it
echo "  v2.0.1 n=1: completion ${w}ms  finalΔ=$fd  iters=$it"

# crossover cycles (rank-independent; use n=1 trajectories)
CO=$(crossover "$R/ours_full_n1.txt"); COV=$(crossover "$R/v201_full.txt")
echo "### crossover to Δ<=$THR: ours cycle $CO, v2.0.1 cycle $COV ###"

echo "### PHASE 2: to-match runs (stop at crossover) ###"
declare -A MATCH
for n in 1 2 4 8; do
  read w fc fd it <<<"$(run ours_m_n$n $OURS $n $((CO+1)) $OF)"
  MATCH[ours$n]=$w; echo "  ours n=$n: match ${w}ms (cycle $fc, Δ=$fd)"
done
read w fc fd it <<<"$(run v201_m $KC 1 $((COV+1)) $KF)"
MATCH[v201]=$w; echo "  v2.0.1 n=1: match ${w}ms (cycle $fc, Δ=$fd)"

echo
echo "########## MATCHED-TOLERANCE (snes_stol 1e-6) ISLAND COMPARISON ##########"
printf "%-14s %14s %16s %12s %10s\n" "config" "match_0.59(ms)" "completion(ms)" "finalΔ" "iters/solve"
printf "%-14s %14s %16s %12s %10s\n" "v2.0.1 n=1" "${MATCH[v201]}" "${COMPL[v201]}" "${FIN[v201]}" "${ITERS[v201]}"
for n in 1 2 4 8; do
  printf "%-14s %14s %16s %12s %10s\n" "ours n=$n" "${MATCH[ours$n]}" "${COMPL[ours$n]}" "${FIN[ours$n]}" "${ITERS[ours$n]}"
done
echo "(v2.0.1 floors at ~0.59: match == completion; SEGV at n>=4)"
