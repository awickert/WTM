#!/bin/bash
# Build the TRUE-equilibrium reference by WARM-starting big-dt from cc_1's near-eq state.
# Cold big-dt aborts (rc=134: one huge step from wtd=0 = the stiffness wall). Warm-started
# from cc_1 cycle-300 (max move ~1 m), big-dt steps are small -> should be stable and finish
# the slow deep-mode tail fast. At a true steady state wtd_new=wtd_old, so whatever fixed
# point this reaches is the SAME one cc_1 crawls toward (cross-checked separately).
#
# Seeds a warm domain (symlinks to domain/ + the wtd seed as Esquibel_010000_wtd.tif),
# then runs cc & tbar at dt 8/16/32 wk, supplied_wt 1, eq_tol 1e-3. Reports decay.
set -uo pipefail
cd "$(dirname "$0")"
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj OMP_NUM_THREADS=1
BIN=../../build/wtm.x; N=4; BASEDT=604800
ROOT=results/algo; REFD="$ROOT/reference"; WARM="$REFD/warm"; WDOM="$REFD/warmdom"
SEED="$REFD/ref_000000300.tif"     # cc_1 near-equilibrium state (cycle 300)
mkdir -p "$WARM" "$WDOM"
[ -s "$SEED" ] || { echo "seed $SEED missing -- abort"; exit 1; }

# warm domain = symlinks to every real input + the wtd seed under the region/time name
for f in domain/Esquibel_*.tif; do ln -sf "$(readlink -f "$f")" "$WDOM/$(basename "$f")"; done
cp -f "$SEED" "$WDOM/Esquibel_010000_starting_wt.tif"   # WTM reads {region}_{time}_starting_wt.tif (irf.cpp:139)

COMMON="-wtm_fringe_source ksat -snes_stol 1e-6"
declare -A FLAGS=( [cc]="-wtm_anderson -snes_anderson_restart_type none"
                   [tbar]="-wtm_anderson -snes_anderson_restart_type none -wtm_Tbar" )
echo "warm-start reference build: seed=$(basename "$SEED"), supplied_wt 1"
printf "%-5s %-6s %-5s %-5s %-7s %-16s %-10s %-10s\n" config dt_wk rc cyc wall last_max|Δ| '<1e-2@' '<1e-3@'
for cfg in cc tbar; do
  for mult in 8 16 32; do
    dt=$((BASEDT*mult)); stem="$WARM/${cfg}_${mult}"
    sed "s#^fsm_on.*#fsm_on 0#;s#^total_cycles.*#total_cycles 120#;s#^cycles_to_save.*#cycles_to_save 120#;s#^supplied_wt.*#supplied_wt 1#;s#^surfdatadir.*#surfdatadir $WDOM/#;s#^textfilename.*#textfilename ${stem}.txt#;s#^outfile_prefix.*#outfile_prefix ${stem}_#;s#^deltat.*#deltat $dt#" eq_awickert.cfg > "${stem}.cfg"
    rm -f "${stem}.txt" "${stem}"_*.tif
    t0=$(date +%s.%N)
    timeout 500 mpirun -n "$N" "$BIN" "${stem}.cfg" ${FLAGS[$cfg]} $COMMON -wtm_eq_tol 0.001 > "${stem}.log" 2>&1
    rc=$?; t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.0f\", $t1-$t0}")
    cyc=$(awk 'NF>=5 && $1~/^[0-9]+$/{c=$1} END{print c+0}' "${stem}.txt")
    last=$(grep -E "per-cycle max" "${stem}.log" | tail -1 | grep -oE "= [0-9.e+-]+ m" | head -1)
    c2=$(grep -E "per-cycle max" "${stem}.log" | awk '{v=$5+0; if(v<1e-2){print NR; exit}}')
    c3=$(grep -E "per-cycle max" "${stem}.log" | awk '{v=$5+0; if(v<1e-3){print NR; exit}}')
    printf "%-5s %-6s %-5s %-5s %-7s %-16s %-10s %-10s\n" "$cfg" "$mult" "$rc" "$cyc" "${wall}s" "$last" "${c2:-never}" "${c3:-never}"
  done
done
echo "WARM REF DONE"