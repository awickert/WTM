#!/usr/bin/env bash
# Surface-transition taper tests: cross-rank determinism + smoothness of the smooth surface-water
# transition (-wtm_evap_taper, tapers 2+3) on the Anderson default path. This is the
# SURFACE_SINK_DESIGN sec 14d experiment sequence, made into an assertion. See taper_test.py.
#
#   tests/taper/run.sh [wtm.x] [nrank ...]
set -uo pipefail
cd "$(dirname "$0")"
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM=$(readlink -f "${1:-../../build/wtm.x}")
shift || true
# THE DECLARED-CONFIG CHECK NEEDS A REAL WORK DIR (#79 Phase 5). taper used to build everything in
# per-study TemporaryDirectory()s, which vanish before _wtm_declared_check can read them -- so the
# check silently never fired on this suite, and taper sat off the enforced list for that reason alone
# rather than any problem with its configs. The FIXTURES still get their own per-study directories,
# because studies A and B write DIFFERENT rasters under the same region name and would collide; only
# the configs, outputs and provenance records land in $WORK, where the comparator looks for them.
make_work taper
export WTM_TAPER_WORK="$WORK"
python3 taper_test.py "$WTM" "$@"
