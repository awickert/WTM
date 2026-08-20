#!/usr/bin/env bash
# General clean-room serial runner (laptop, n=1, idle): any method, FSM on/off.
# Usage: local_serial_run.sh <label> "<method flags>" <dt_wk> <cycle_cap> <fsm 0|1>
#   cc:      ""                          tr:     "-wtm_tr_bdf2"
#   trtbar:  "-wtm_tr_bdf2 -wtm_Tbar"    bdf2v:  "-wtm_bdf2_on_V"     cctbar: "-wtm_Tbar"
# Mirrors the MSI bigdt config (dry -20%, clamp on, ghost boundary, stol 1e-6). fsm_on parameterized.
# CSV: label,dt_wk,cycles,fsm,rc,wall_s,snes_its,s_per_it   (wall = total incl. FSM; s/it = wall/GW-iters)
set -uo pipefail
cd /home/awickert/models/WTM/benchmark/esquibel
BIN=/home/awickert/models/WTM/build/wtm.x
PERT=results/algo/transient/local_pert_dry
OUT=results/algo/transient/local_serial; mkdir -p "$OUT"
CSV="$OUT/serial_wall_fsm.csv"
[ -f "$CSV" ] || echo "label,dt_wk,cycles,fsm,rc,wall_s,snes_its,s_per_it" > "$CSV"

M="$1"; FLAGS="$2"; DTWK="$3"; CAP="${4:-0}"; FSM="${5:-0}"
DT=$(awk "BEGIN{printf \"%d\", $DTWK*604800}")
CYC=$(awk "BEGIN{printf \"%d\", 32/$DTWK + 0.5}")
[ "$CAP" -gt 0 ] && [ "$CYC" -gt "$CAP" ] && CYC=$CAP
stem="$OUT/dry_${M}_fsm${FSM}_dt${DTWK}wk"
cat > ${stem}.cfg <<CFG
run_type transient
fsm_on $FSM
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      55.338391020555555
deltat $DT
total_cycles $CYC
cycles_to_save $CYC
maxiter            50
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
rm -f ${stem}.txt ${stem}_*.tif
t0=$(date +%s.%N)
mpiexec -n 1 "$BIN" ${stem}.cfg -wtm_anderson -snes_anderson_restart_type none $FLAGS \
  -wtm_fringe_source ksat -wtm_surface_exfiltration_to_runoff \
  -snes_stol 1e-6 > ${stem}.log 2>&1
rc=$?; t1=$(date +%s.%N)
wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
its=$(grep -o "Number of nonlinear iterations = [0-9]*" ${stem}.log | awk '{s+=$6} END{print s+0}')
spi=$(awk "BEGIN{ if($its>0) printf \"%.5f\", $wall/$its; else print \"NA\" }")
echo "$M,$DTWK,$CYC,$FSM,$rc,$wall,$its,$spi" | tee -a "$CSV"
