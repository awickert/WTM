#!/usr/bin/env bash
# Island speed test: v2.0.1 baseline vs the current branch, reproducibly.
#
# Two experiments on the same island domain, same 1-week dt:
#   (EQ) cold-start (wtd=0) to equilibrium
#   (TR) warm-start from equilibrium, transient with P-ET perturbed x1.1 (see perturb_pet.py)
#
# v2.0.1 has no built-in Anderson; its default SNES fails on this stiff cold-start, so it is run
# with PETSc's matrix-free Anderson on the command line (the only way it converges at 1-wk dt) --
# this is the baseline. The current branch is run with its own -wtm_anderson -wtm_Tbar.
#
# Usage:  run_island_speedtest.sh <island_domain_dir> [ncores]
#   <island_domain_dir> holds region=Esquibel time=010000 rasters + Esquibel_horizontal_ksat/porosity.
#   ncores: OpenMP threads for BOTH binaries (default 8). v2.0.1 is also timed at 1 core for reference.
#
# Binaries (edit if your paths differ):
OURS=${OURS:-/home/awickert/models/WTM/build/wtm.x}
MASTER=${MASTER:-/home/awickert/models/WTM-mastertest/build/wtm.x}       # v2.0.1 (tag v2.0.1)
V201_FLAGS="-snes_mf -snes_type anderson -snes_stol 1e-6"                 # how v2.0.1 converges here
OURS_FLAGS="-wtm_anderson -wtm_Tbar"                                      # current-branch workhorse
set -uo pipefail
export PROJ_DATA=/usr/share/proj PROJ_LIB=/usr/share/proj

DOM=${1:?usage: run_island_speedtest.sh <island_domain_dir> [ncores]}
NC=${2:-8}
OUT=$DOM/speedtest; mkdir -p "$OUT"
DT=604800            # 1 week
SOUTH=$(cat "$DOM/_se.txt" 2>/dev/null || echo 55.3839465761)
SETTLE=1.0           # |wtd change| (col 5) threshold for "reached equilibrium"

# Write a v2.0.1/branch-compatible cfg (shared keys; per-binary flags differ, not the cfg).
mkcfg() { # name run_type deltat total_cycles supplied_wt
  local name=$1 rt=$2 dt=$3 ncyc=$4 swt=$5
  cat > "$OUT/$name.cfg" <<EOF
run_type           $rt
fsm_on             1
evap_mode          1
infiltration_on    0
runoff_ratio_on    1
cells_per_degree   900
southern_edge      $SOUTH
deltat             $dt
total_cycles       $ncyc
save_nreport_interval     $ncyc
report_interval            50
fdepth_a           100
fdepth_b           150
fdepth_fmin        2.5
time_start         010000
time_end           010000
surfdatadir        $DOM/
region             Esquibel
supplied_wt        $swt
textfilename       $OUT/$name.txt
outfile_prefix     $OUT/${name}_
EOF
  : > "$OUT/$name.txt"
}

run_timed() { # label binary cfg threads extra_flags...
  local label=$1 bin=$2 cfg=$3 thr=$4; shift 4
  OMP_NUM_THREADS=$thr /usr/bin/env bash -c "t0=\$(date +%s%N); timeout 1200 '$bin' '$cfg' $* > '$OUT/$label.log' 2>&1; rc=\$?; t1=\$(date +%s%N); echo \"WALL_MS \$(( (t1-t0)/1000000 )) RC \$rc\"" \
    | tee "$OUT/$label.time"
  local txt=${cfg%.cfg}.txt
  awk -v s=$SETTLE '$1~/^[0-9]+$/{c=$1;d=$5} $1~/^[0-9]+$/ && $5+0<s && !f{f=$1} END{print "  last cycle="c" (Δ="d"); reached Δ<"s" at cycle "(f?f:"NONE")}' "$txt"
}

echo "############ ISLAND SPEED TEST  (domain=$DOM, ncores=$NC, dt=1wk) ############"
echo "### (EQ) cold-start -> equilibrium ###"
mkcfg eq_v201  equilibrium $DT 100 0
mkcfg eq_ours  equilibrium $DT 100 0
echo "-- v2.0.1 (1 core) --";  run_timed eq_v201_n1 "$MASTER" "$OUT/eq_v201.cfg" 1  $V201_FLAGS
echo "-- v2.0.1 ($NC cores) --"; run_timed eq_v201_nN "$MASTER" "$OUT/eq_v201.cfg" $NC $V201_FLAGS
echo "-- ours ($NC cores) --";   run_timed eq_ours_nN "$OURS"   "$OUT/eq_ours.cfg" $NC $OURS_FLAGS

echo "### (TR) warm-start + P-ET x1.1, transient ###"
# Perturbed forcing: precip' = 1.1P - 0.1E so that (P'-E) = 1.1(P-E); warm-start from the EQ output.
PDOM=$OUT/pert_domain
python3 "$(dirname "$0")/perturb_pet.py" "$DOM" "$PDOM" "$OUT/eq_ours_000000100.tif" 2>&1 | tail -3 || { echo "perturb setup failed -- see perturb_pet.py"; exit 0; }
mkcfg_tr() { local name=$1 sw=$2; cat > "$OUT/$name.cfg" <<EOF
run_type transient
fsm_on 1
evap_mode 1
infiltration_on 0
runoff_ratio_on 1
cells_per_degree 900
southern_edge $SOUTH
deltat $DT
total_cycles 20
save_nreport_interval 20
report_interval 50
fdepth_a 100
fdepth_b 150
fdepth_fmin 2.5
time_start 010000
time_end 010000
surfdatadir $PDOM/
region Esquibel
supplied_wt $sw
textfilename $OUT/$name.txt
outfile_prefix $OUT/${name}_
EOF
: > "$OUT/$name.txt"; }
mkcfg_tr tr_v201 1
mkcfg_tr tr_ours 1
echo "-- v2.0.1 ($NC cores) --"; run_timed tr_v201_nN "$MASTER" "$OUT/tr_v201.cfg" $NC $V201_FLAGS
echo "-- ours ($NC cores) --";    run_timed tr_ours_nN "$OURS"   "$OUT/tr_ours.cfg" $NC $OURS_FLAGS
echo "############ DONE -- results + logs in $OUT ############"
