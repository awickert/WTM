#!/usr/bin/env bash
# CONFIG SCHEMA: an unrecognised YAML key must ABORT the run, with a message that names it.
#
# WHY THIS EXISTS. yaml-cpp reads by lookup, so a key nobody looks up is simply never seen. Before the
# schema check, a typo, a key retired by a migration, and a setting a user believed was in force were
# all indistinguishable from not writing the key at all: the run proceeded and reported success.
#
# The cost is not a lost setting, it is a LOST NEGATIVE RESULT. A sweep over a key nothing reads returns
# "no effect" for a reason that has nothing to do with the model, and it reads exactly like a finding.
# Two of those happened in this repo: `total_cycles` (retired when the config went nested YAML) sat in
# ten benchmark scripts doing nothing for weeks, and a step-size-controller sweep reported byte-identical
# results at every setting because the flag it varied was parsed on a different code path.
#
# WHAT IT ASSERTS
#   REFERENCE  the repo's own config.yaml passes the schema -- the check must not reject valid configs.
#              This is the arm that fails if a key is added to a reader and not to the dictionary.
#   REJECT     an unknown key aborts, non-zero, and the message NAMES the offending key. Doubles as the
#              positive control: if validation were compiled out, everything else here would pass
#              vacuously, so this arm proves the check can actually fire.
#   SUGGEST    a near-miss typo gets a "did you mean" pointing at the intended key.
#   NESTED     the walk reaches nested sections, not just the top level (a top-level-only check would
#              pass REJECT while ignoring every section key).
#   MULTI      all offending keys are reported at once, not one abort per run.
#   SHIM       every legacy key tests/emit_config.sh can emit still validates -- the whole test suite
#              builds its configs through that shim, so a dictionary that disagrees with it would break
#              every other test in the suite rather than this one.
#
# Also REPORTED (not asserted): keys the dictionary accepts that config.yaml never documents. Some are
# deliberate -- `grid` is deprecated (#124), `dev` is developer-only, `sink` is legacy -- so this is a
# list to read, not a gate. Documenting a user-facing reference is an editorial decision.
#
# Usage:  tests/config_schema/run.sh [path/to/wtm.x]
set -uo pipefail
cd "$(dirname "$0")"
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
ROOT=$(readlink -f ../..)
REF="$ROOT/config.yaml"
[ -f "$REF" ] || { echo "ERROR: reference config not found at $REF"; exit 1; }
WORK=$(mktemp -d /tmp/cfgschema_XXXX); trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=1

echo "=== config schema: unknown keys must abort, informatively ==="
echo "WTM binary: $WTM"
echo
fail=0

# The model aborts on missing input data long AFTER config parsing, so every arm here greps the MESSAGE
# rather than trusting the exit status: a config-schema abort and a missing-raster abort both exit
# non-zero. Run through an inner shell so the expected abort's "Aborted (core dumped)" job-control notice
# goes to that shell's stderr instead of this suite's output.
msg() { sh -c '"$0" "$1" 2>&1' "$WTM" "$1" 2>/dev/null; }

# Inject a key by ROUND-TRIPPING the YAML, not by text substitution. config.yaml mixes block and inline
# flow style ("fdepth: { a: 200, ... }"), so splicing a line after a key can land inside a flow mapping
# and produce invalid YAML -- which the model then reports as a PARSE error, and the arm fails for a
# reason that has nothing to do with the schema. (It did exactly that on first run.)
inject() { # $1 out-file, $2.. dotted paths to add as bogus keys
    local out="$1"; shift
    python3 - "$REF" "$out" "$@" <<'PY'
import sys, yaml
ref, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = yaml.safe_load(open(ref))
for p in paths:
    node, *rest = p.split(".")
    d = cfg
    for k in [node] + rest[:-1]:
        d = d.setdefault(k, {})
    d[rest[-1] if rest else node] = 1
yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False)
PY
}

# ---- REFERENCE: a valid config must NOT be rejected ------------------------------------------------
if msg "$REF" | grep -q "unrecognised key"; then
    echo "  FAIL  REFERENCE  the repo's own config.yaml is REJECTED by the schema:"
    msg "$REF" | grep -A3 "unrecognised key" | sed 's/^/        /'
    echo "        A reader gained a key that the dictionary in src/parameters.cpp does not list."
    fail=1
else
    echo "  PASS  REFERENCE  config.yaml passes the schema (no valid config is rejected)"
fi

# ---- REJECT: an unknown key aborts and is NAMED. Also the positive control for this whole file. -----
inject "$WORK/bogus.yaml" time.nonsense_key
OUT=$(msg "$WORK/bogus.yaml")
if echo "$OUT" | grep -q "unrecognised key" && echo "$OUT" | grep -q "nonsense_key"; then
    echo "  PASS  REJECT     an unknown key aborts and the message names it ('time.nonsense_key')"
else
    echo "  FAIL  REJECT     an unknown key was NOT rejected -- validation is not running."
    echo "        Every other arm in this file would then pass vacuously."
    fail=1
fi

# ---- SUGGEST: a near-miss gets a did-you-mean ------------------------------------------------------
python3 -c "import sys,yaml; c=yaml.safe_load(open('$REF')); c['time']['detlat']=c['time'].pop('deltat'); yaml.safe_dump(c,open('$WORK/typo.yaml','w'),default_flow_style=False)"
OUT=$(msg "$WORK/typo.yaml")
if echo "$OUT" | grep -q "did you mean 'time.deltat'"; then
    echo "  PASS  SUGGEST    'time.detlat' suggests 'time.deltat'"
else
    echo "  FAIL  SUGGEST    no did-you-mean for a one-transposition typo:"
    echo "$OUT" | grep -A2 "unrecognised key" | sed 's/^/        /'
    fail=1
fi

# ---- NESTED: the walk descends into sections ------------------------------------------------------
inject "$WORK/nested.yaml" transmissivity.fdepth.bogus_subkey
OUT=$(msg "$WORK/nested.yaml")
if echo "$OUT" | grep -q "transmissivity.fdepth.bogus_subkey"; then
    echo "  PASS  NESTED     a bad key inside a nested section is caught, with its full dotted path"
else
    echo "  FAIL  NESTED     nested sections are not walked (a top-level-only check would still pass REJECT)"
    fail=1
fi

# ---- MULTI: report every offender in one run ------------------------------------------------------
inject "$WORK/multi.yaml" time.bad_one io.bad_two
OUT=$(msg "$WORK/multi.yaml")
if echo "$OUT" | grep -q "bad_one" && echo "$OUT" | grep -q "bad_two"; then
    echo "  PASS  MULTI      both offending keys reported in one abort (not one run per typo)"
else
    echo "  FAIL  MULTI      only the first offender was reported; fix the whole file in one pass"
    fail=1
fi

# ---- SHIM: the suite's own config emitter must agree with the dictionary ---------------------------
# Every legacy key tests/emit_config.sh maps, in one config. If the dictionary and the shim disagree,
# this catches it HERE instead of as a mass failure across every other test in the suite.
cat > "$WORK/shim_keys.txt" <<'EOF'
run_type equilibrium
supplied_wt 1
deltat 31536000
total_time 20yr
report_interval 1
save_nreport_interval 1
cells_per_degree 10
southern_edge -45
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
infiltration_on 0
fsm_on 1
runoff_ratio 0.3
runoff_collector active_set
evap_mode 0
surfdatadir /nonexistent
region none
time_start t0
time_end t0
textfilename /dev/null
outfile_prefix /tmp/none_
EOF
bash ../emit_config.sh < "$WORK/shim_keys.txt" > "$WORK/shim.yaml"
OUT=$(msg "$WORK/shim.yaml")
if echo "$OUT" | grep -q "unrecognised key"; then
    echo "  FAIL  SHIM       emit_config.sh emits a key the dictionary rejects:"
    echo "$OUT" | grep -A3 "unrecognised key" | sed 's/^/        /'
    fail=1
else
    echo "  PASS  SHIM       every key emit_config.sh emits validates ($(grep -c . "$WORK/shim_keys.txt") legacy keys)"
fi

# ---- REPORT (not a gate): accepted keys that config.yaml does not document -------------------------
echo
python3 - "$REF" "$ROOT/src/parameters.cpp" <<'PY'
import sys, re, collections, yaml
ref, src = open(sys.argv[1]).read(), open(sys.argv[2]).read()
def walk(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            out[prefix].add(k); walk(v, f"{prefix}.{k}" if prefix else k, out)
doc = collections.defaultdict(set); walk(yaml.safe_load(ref), "", doc)
block = src[src.index("static const std::map<std::string, std::set<std::string>> schema"):src.index("return schema;")]
schema = {m.group(1): set(re.findall(r'"([a-z_]+)"', m.group(2)))
          for m in re.finditer(r'\{"([a-z_.]*)",\s*\{([^}]*)\}\}', block, re.S)}
gaps = {p: sorted(k - doc.get(p, set())) for p, k in schema.items() if k - doc.get(p, set())}
if not gaps:
    print("  note  DOCS       every accepted key appears in config.yaml")
else:
    n = sum(len(v) for v in gaps.values())
    print(f"  note  DOCS       {n} accepted keys are NOT in config.yaml (not a failure; some are")
    print("                   deliberately unadvertised -- grid is deprecated, dev is developer-only,")
    print("                   collection.sink is legacy -- but ordinary user keys here want documenting):")
    for p in sorted(gaps):
        print(f"                     {p or '<top level>'}: {' '.join(gaps[p])}")
PY

echo
if [[ $fail -eq 0 ]]; then echo "CONFIG SCHEMA: ALL PASSED"; else echo "CONFIG SCHEMA: FAILED" >&2; fi
exit $fail
