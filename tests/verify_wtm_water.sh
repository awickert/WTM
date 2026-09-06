#!/usr/bin/env bash
# Prove tests/wtm_water.py::stored_volume IS storedVolume() from the C++, not a lookalike.
#
# The Python helper exists so tests can compare water tables in WATER. That is only worth anything
# if it computes the SAME V(wtd) the model does -- a helper that is subtly different would put a
# confident, wrongly-scaled number in front of every assertion in the suite. So compile the real
# translation unit, sample V over the range the fixtures actually visit (deep, at the surface
# smoothing kink, and ponded above it), across the porosities they use, and require agreement to
# machine precision. It also covers both branches of the smoothing width and extended_soil.
set -uo pipefail
cd "$(dirname "$0")"
SRC=../src/update_effective_storativity.cpp
WORK=$(mktemp -d /tmp/vwater_XXXX); trap 'rm -rf "$WORK"' EXIT
PY="${PY:-python3}"

cat > "$WORK/probe.cpp" <<'CPP'
#include "../src/update_effective_storativity.hpp"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
  if (argc > 1) g_storativity_surface_smoothing_width = atof(argv[1]);
  if (argc > 2) g_extended_soil = (atoi(argv[2]) != 0);
  const double wtds[] = {-4000, -40, -1, -0.1, -0.01, -1e-6, 0, 1e-6, 0.01, 0.1, 1, 40, 4000};
  const double phis[] = {0.05, 0.25, 0.4, 1.0};
  for (double p : phis)
    for (double w : wtds) printf("%.17g %.17g %.17g\n", w, p, storedVolume(w, p));
  return 0;
}
CPP
g++ -O2 -o "$WORK/probe" "$WORK/probe.cpp" "$SRC" -I../src 2>"$WORK/cc.log" \
  || { echo "FAIL: could not build the C++ probe"; tail -5 "$WORK/cc.log"; exit 1; }

fail=0
for cfg in "0.01 0" "0.5 0" "0.01 1"; do
  set -- $cfg
  "$WORK/probe" "$1" "$2" > "$WORK/cpp.txt" || { echo "FAIL: probe crashed (smoothing=$1 extended=$2)"; exit 1; }
  SMOOTH="$1" EXT="$2" "$PY" - "$WORK/cpp.txt" <<'PYEOF' || fail=1
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if False else ".")
import wtm_water as W
smooth = float(os.environ["SMOOTH"]); ext = os.environ["EXT"] == "1"
worst, worst_row = 0.0, None
n = 0
for line in open(sys.argv[1]):
    w, p, v_cpp = (float(x) for x in line.split())
    v_py = float(W.stored_volume(w, p, smoothing=smooth, extended_soil=ext))
    scale = max(abs(v_cpp), 1.0)
    rel = abs(v_py - v_cpp) / scale
    n += 1
    if rel > worst: worst, worst_row = rel, (w, p, v_cpp, v_py)
tag = "smoothing=%g extended_soil=%s" % (smooth, ext)
if worst <= 1e-15:
    print("  OK   %-34s %d samples agree with the C++ to %.1e (<= 1e-15)" % (tag, n, worst))
else:
    print("  FAIL %-34s worst relative disagreement %.3e at wtd=%g phi=%g: C++ %.17g vs py %.17g"
          % (tag, worst, worst_row[0], worst_row[1], worst_row[2], worst_row[3]))
    sys.exit(1)
PYEOF
done

[ "$fail" -eq 0 ] && { echo "WTM_WATER: the Python helper matches the C++ storedVolume"; exit 0; }
echo "WTM_WATER: FAILED"; exit 1
