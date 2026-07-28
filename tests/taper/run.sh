#!/usr/bin/env bash
# Surface-transition taper tests: cross-rank determinism + smoothness of the smooth surface-water
# transition (-wtm_surface_sink + -wtm_evap_taper) on the Anderson default path. This is the
# SURFACE_SINK_DESIGN sec 14d experiment sequence, made into an assertion. See taper_test.py.
#
#   tests/taper/run.sh [wtm.x] [nrank ...]
set -uo pipefail
cd "$(dirname "$0")"
WTM=$(readlink -f "${1:-../../build/wtm.x}")
shift || true
python3 taper_test.py "$WTM" "$@"
