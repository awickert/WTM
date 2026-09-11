#!/usr/bin/env bash
# #103: WHICH PARAMETERS PERMIT THE LIMIT CYCLE. One knob at a time from the shipped
# tests/storage_equivalence config. Every arm verifies its own substitution landed before running.
cd /home/awickert/models/WTM
SW=/tmp/claude-1000/lc103; SE=tests/storage_equivalence; WTM=$(readlink -f build/wtm.x)
export OMP_NUM_THREADS=1
printf '%-28s %-10s %-7s %-7s %-10s %-10s %-6s %s\n' "arm" "settle(m)" "n>1mm" "atsurf" "early" "late" "l/e" "verdict"
run_arm() {
  local lab=$1 cpd=$2 check=$3; shift 3
  local cfg=$SW/$lab.yaml
  sed -e "s|@INPUTS@|$SW/in$cpd|g" -e "s|@WORK@|$SW|g" -e "s|@STEM@|$lab|g" \
      -e "s|^  save_every_n_reports:.*|  save_every_n_reports: 1|" -e "s|^  total:.*|  total: \"14515200000s\"|" "$@" $SE/config.yaml > $cfg
  if [ -n "$check" ] && ! grep -q "$check" $cfg; then
    printf '%-28s SUBSTITUTION FAILED (%s not in config) -- arm NOT run\n' "$lab" "$check"; return
  fi
  timeout 900 $WTM $cfg > $SW/$lab.log 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then
    if grep -qi "ERROR: config:" $SW/$lab.log; then
      printf '%-28s %s\n' "$lab" "REFUSED BY MODEL: $(grep -m1 -oP '(?<=ERROR: config: ).{0,60}' $SW/$lab.log)"
    else
      printf '%-28s rc=%d %s\n' "$lab" "$rc" "$(grep -m1 -oE 'DIVERGED_[A-Z_]+' $SW/$lab.log)"
    fi; return
  fi
  read s n a t <<< "$(python3 $SW/metric.py $SW/$lab)"
  read nc early late ratio <<< "$(python3 $SW/decay.py $SW/$lab.txt)"
  local v="DECAYING"
  awk -v r="$ratio" 'BEGIN{exit !(r+0 > 0.30)}' && v="NOT DECAYING"
  awk -v x="$s" 'BEGIN{exit !(x+0 < 1e-6)}' && v="settled"
  printf '%-28s %-10s %-7s %-7s %-10s %-10s %-6s %s\n' "$lab" "$s" "$n/$t" "$a/$t" "$early" "$late" "$ratio" "$v"
}
echo "--- baseline (shipped config: explicit, secant, adaptive, BE, smoothing 0.01, taper on) ---"
run_arm base_cpd64 64 ""
echo "--- KNOB: collection.method ---"
run_arm coll_activeset 64 "method: active_set"  -e 's|^    method: explicit|    method: active_set|'
run_arm coll_as_volume 64 "storage_form: volume" -e 's|^    method: explicit|    method: active_set|' -e 's|^  storage_form: secant|  storage_form: volume|'
run_arm coll_off       64 "method: off"          -e 's|^    method: explicit|    method: off|'
echo "--- KNOB: cell size (moves the field relative to the surface) ---"
run_arm cell_cpd16  16 ""
run_arm cell_cpd256 256 ""
echo "--- KNOB: solver.smoothing.storativity_surface ---"
run_arm smooth_0    64 "storativity_surface: 0$"   -e 's|^    storativity_surface: 0.01.*|    storativity_surface: 0|'
run_arm smooth_0p1  64 "storativity_surface: 0.1"  -e 's|^    storativity_surface: 0.01.*|    storativity_surface: 0.1|'
echo "--- KNOB: evaporation.tapers.surface_transition ---"
run_arm taper_off 64 "surface_transition: false" -e 's|^    surface_transition: true.*|    surface_transition: false|'
echo "--- KNOB: dev.storage_form ---"
run_arm store_volume 64 "storage_form: volume" -e 's|^  storage_form: secant.*|  storage_form: volume|'
echo "--- KNOB: time stepping ---"
run_arm step_fixed  64 "mode: fixed" -e 's|^    mode: adaptive.*|    mode: fixed|'
run_arm step_dt4    64 "dt: 604800"  -e 's|^    dt: 2419200.*|    dt: 604800|'
echo "--- KNOB: integrator ---"
run_arm integ_trbdf2 64 "time_integration: tr-bdf2" -e 's|^  time_integration: backward-euler.*|  time_integration: tr-bdf2|'
