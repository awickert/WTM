#!/usr/bin/env bash
# Draft a suite's config.yaml from the harvest, ready for its provenance header (#83).
#
#   tests/draft_config.sh <suite>
#
# Runs materialize.py (harvest -> base + per-arm overrides, tokenized, translated to the routing key),
# keeps the BASE, then own_defaults.py attaches a reason to each unstated value it can justify FROM
# THE CONFIG'S OWN FACTS. Whatever it cannot justify keeps its UNSTATED marker, which is the point:
# a leftover marker is a visible to-do, a fabricated reason is a lie that reads like a decision.
#
# Prints the per-arm overrides afterwards, because those go in run.sh with the reasoning, not here.
set -uo pipefail
cd "$(dirname "$0")"
suite="${1:?usage: draft_config.sh <suite>}"
python3 materialize.py "$suite" > "/tmp/claude-1000/${suite}.derived" || exit 1
sed -n '/^=== BASE/,/^=== PER-ARM/p' "/tmp/claude-1000/${suite}.derived" | sed '1d;$d' > "$suite/config.yaml"
python3 own_defaults.py "$suite/config.yaml"
echo "--- per-arm overrides (these belong in run.sh) ---"
sed -n '/^=== PER-ARM/,/^=== UNSTATED/p' "/tmp/claude-1000/${suite}.derived" | sed '1d;$d'
