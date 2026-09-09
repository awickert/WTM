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
. ../lib.sh                            # make_work: keeps the work dir when a test FAILS
WTM="${1:-$(readlink -f ../../build/wtm.x)}"
[ -x "$WTM" ] || { echo "ERROR: WTM binary not found at $WTM"; exit 1; }
ROOT=$(readlink -f ../..)
REF="$ROOT/config.yaml"
[ -f "$REF" ] || { echo "ERROR: reference config not found at $REF"; exit 1; }
make_work cfgschema
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


# Set a dotted path to a VALUE (same round-trip rationale as inject above).
setval() { # $1 out-file, $2 dotted path, $3 value
    python3 - "$REF" "$1" "$2" "$3" <<'PY2'
import sys, yaml
ref, out, path, val = sys.argv[1:5]
cfg = yaml.safe_load(open(ref))
d = cfg
ks = path.split(".")
for k in ks[:-1]:
    d = d.setdefault(k, {})
d[ks[-1]] = val
yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False)
PY2
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
# Was time.detlat -> time.deltat until the step moved to solver.time_step.dt. Retargeted at another
# one-transposition typo of a key that still exists, so the arm keeps testing the same thing.
python3 -c "import sys,yaml; c=yaml.safe_load(open('$REF')); c['time']['totla']=c['time'].pop('total'); yaml.safe_dump(c,open('$WORK/typo.yaml','w'),default_flow_style=False)"
OUT=$(msg "$WORK/typo.yaml")
if echo "$OUT" | grep -q "did you mean 'time.total'"; then
    echo "  PASS  SUGGEST    'time.totla' suggests 'time.total'"
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

# ---- ENUM: a key's VALUE must be checked, not just its name ----------------------------------------
# The schema above validates KEYS. That left the same defect one level down: `solver.method: pickard`
# fell through the bridge's if/else chain to the DEFAULT and the run reported success, so a sweep over a
# misspelled solver silently compared Anderson with Anderson -- the lost-negative-result cost again, on
# the setting most likely to be swept. Five keys behaved that way (solver.method, time_integration,
# storage, boundaries.land, run.equilibrium_stop.metric); five others already validated.
#
# BOTH HALVES MATTER. A validator that rejected everything would pass a reject-only test, so every legal
# value is exercised too -- including the three RETIRED eq_metric spellings, which must keep working
# (the consumer maps them with a NOTE) rather than becoming errors.
ENUM_BAD=(solver.method solver.time_integration dev.storage_form boundaries.land run.equilibrium_stop.metric)
for k in "${ENUM_BAD[@]}"; do
    setval "$WORK/enum.yaml" "$k" "definitely_not_a_value"
    # CAPTURE FIRST. `msg ... | grep -q` would take the MODEL's exit status under `set -o pipefail`
    # (134 from the very abort being tested), so the test would fail whenever it should pass -- and its
    # ENUM-OK counterpart would pass vacuously. Every other arm in this file captures for the same reason.
    OUT=$(msg "$WORK/enum.yaml")
    if echo "$OUT" | command grep -q "config: $k must be"; then
        echo "  PASS  ENUM-BAD   $k rejects an unknown value and lists the legal ones"
    else
        echo "  FAIL  ENUM-BAD   $k ACCEPTED 'definitely_not_a_value'. It falls through to the default and"
        echo "        the run reports success -- a swept parameter that silently does nothing."
        fail=1
    fi
done

ENUM_OK=("solver.method anderson" "solver.method picard" "solver.method newton"
         "solver.time_integration backward-euler" "solver.time_integration bdf2"
         "solver.time_integration tr-bdf2" "dev.storage_form volume" "dev.storage_form secant"
         "boundaries.land neumann_toposlope" "boundaries.land dirichlet_sea_level"
         "run.equilibrium_stop.metric max" "run.equilibrium_stop.metric rms"
         "run.equilibrium_stop.metric frac" "run.equilibrium_stop.metric water"
         "run.equilibrium_stop.metric water-max" "run.equilibrium_stop.metric water-rms")
bad_ok=0
for kv in "${ENUM_OK[@]}"; do
    set -- $kv
    setval "$WORK/enum.yaml" "$1" "$2"
    OUT=$(msg "$WORK/enum.yaml")
    if echo "$OUT" | command grep -q "config: $1 must be"; then
        echo "  FAIL  ENUM-OK    $1: $2 is a LEGAL value but was rejected"
        bad_ok=1; fail=1
    fi
done
[ $bad_ok -eq 0 ] && echo "  PASS  ENUM-OK    all ${#ENUM_OK[@]} legal values across the 5 enums are still accepted"

# ---- RETIRED: a key that was REMOVED must abort, not drift ------------------------------------------
# dev.active_set was a SECOND YAML route to the same enforcement as surface_water.collection.method,
# and it silently OVERRODE an explicit method: with `method: explicit` plus `dev: {active_set: true}`
# the run used active_set instead, differing on 54 of 256 cells (max 0.127 m) with NO log line. It was
# removed 2026-09-01. This arm pins the REMOVAL: a config carrying the old key must say so and stop.
# Distinct from REJECT above, which uses an invented key -- this one is a real spelling that used to
# work, which is exactly the case a user upgrading an old config will hit.
inject "$WORK/retired.yaml" dev.active_set
OUT=$(msg "$WORK/retired.yaml")
if echo "$OUT" | grep -q "unrecognised key" && echo "$OUT" | grep -q "dev.active_set"; then
    echo "  PASS  RETIRED    the removed key 'dev.active_set' aborts and is named"
else
    echo "  FAIL  RETIRED    'dev.active_set' was accepted. It is a SECOND route to the active-set"
    echo "        enforcement and silently overrides surface_water.collection.method -- if it is back in"
    echo "        the schema, the dual-route defect is back. See task #28."
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
# The list above covered 22 of the shim's 44 keys while the comment claimed "every legacy key". That
# overclaim is why the `trace` gating defect survived: trace was not in the list, so nothing noticed
# the shim silently dropping it. The rest of the vocabulary follows, so SHIM and SHIM/BACK now mean
# what they say. Values are legal ones -- the model's own validator runs over this config.
# EVERY VALUE HERE MUST BE NON-DEFAULT. SHIM/BACK is differential -- it emits the config with and
# without each key and requires the two to DIFFER -- and the shim now emits every setting, defaulted
# (fc18e95). So a key set to ITS OWN DEFAULT is indistinguishable from an absent one, and six of these
# were: dt_continuation false, storage volume, convergence_metric volume, eq_frac 0.001,
# extinction_depth 8, et_sigmoid_width 0.1. The arm failed, correctly -- it can no longer prove those
# keys flow through. Choosing non-default values restores the assertion AND strengthens it: it now
# proves the key's VALUE reaches the config, not merely that some line with that name appears.
solver_method anderson
time_integration tr-bdf2
adaptive_dt false
dt_tol 0.5
dt_max 31536000
dt_continuation true
under_relaxation 0.9
t_bar true
storage secant
convergence_metric head
convergence_water_volume_tol 1e-9
eq_tol 0.002
eq_metric rms
eq_frac 0.002
land_boundary dirichlet
fsm_coupling continuous
runoff_ratio_on 1
extinction_depth 6
et_sigmoid_width 0.2
et_sigmoid_wtd_center 0.0
trace dt
run_dir /tmp/shim_explicit_rundir   # DISTINCT from the derived <prefix>prov, or thedifference  vanishes
EOF
bash ../emit_config.sh < "$WORK/shim_keys.txt" > "$WORK/shim.yaml"
OUT=$(msg "$WORK/shim.yaml")
if echo "$OUT" | grep -q "unrecognised key"; then
    echo "  FAIL  SHIM       emit_config.sh emits a key the dictionary rejects:"
    echo "$OUT" | grep -A3 "unrecognised key" | sed 's/^/        /'
    fail=1
else
    echo "  PASS  SHIM       every key emit_config.sh emits validates ($(grep -cE "^[a-z]" "$WORK/shim_keys.txt") legacy keys)"
fi

# ---- SHIM/BACK: every key the shim is GIVEN must CHANGE what it emits ------------------------------
# The arm above checks one direction only -- that the emitted YAML validates. That is exactly how the
# `trace` defect lived: the emitted config was perfectly valid, it just silently LACKED the key it had
# been asked for, because `trace` was gated on an output PATH key also being present. Valid and
# complete are different properties, and only one of them was being tested.
#
# THE TEST IS DIFFERENTIAL, not a search for the value in the output, and that distinction was learned
# the hard way: the first version of this arm grepped the emitted YAML for each key's VALUE, and it
# did NOT catch the trace defect when it was deliberately reintroduced -- `trace dt` looks for "dt",
# which already appears in `dt: 31536000` and `adaptive_dt`. A substring match on short values is
# almost no assertion at all. So instead: emit the config WITH and WITHOUT each key and require the
# two to DIFFER. A key that changes nothing is a key being dropped, whatever the reason.
missing=""
while read -r k v; do
    [ -z "$k" ] && continue
    case "$k" in \#*) continue ;; esac
    # EXEMPTIONS, each for a reason the shim documents -- not a way to quieten an inconvenient result.
    #   evap_mode        accepted-but-inert; it announces itself on stderr and is SUPPOSED to emit nothing.
    #   runoff_ratio_on  legitimately SHADOWED when a numeric runoff_ratio is present ("a numeric value
    #                    takes precedence; else runoff_ratio_on 1 requires the raster"). Both keys stay in
    #                    the list so SHIM still validates them; only this differential check skips the
    #                    shadowed one. Flagged by the test on its first run, then verified against the
    #                    shim's own documented precedence before being exempted.
    case "$k" in evap_mode|runoff_ratio_on) continue ;; esac
    grep -vE "^$k " "$WORK/shim_keys.txt" > "$WORK/without.txt"
    bash ../emit_config.sh < "$WORK/without.txt" > "$WORK/without.yaml" 2>/dev/null
    cmp -s "$WORK/shim.yaml" "$WORK/without.yaml" && missing="$missing $k"
done < "$WORK/shim_keys.txt"
if [ -n "$missing" ]; then
    echo "  FAIL  SHIM/BACK  removing these keys changes NOTHING in the emitted config, so the shim is"
    echo "                   silently dropping them -- every arm that sets one is VACUOUS:"
    for k in $missing; do echo "                     $k"; done
    fail=1
else
    echo "  PASS  SHIM/BACK  every legacy key fed in demonstrably changes the emitted config"
fi

# ---- SHIM/ALONE: every key must still do something when it is the ONLY key set ---------------------
# REMOVAL FROM THE FULL SET IS NOT ENOUGH, and the trace defect is the proof. It only manifested when
# NEITHER textfilename NOR outfile_prefix was present -- and the full key list sets both, so dropping
# `trace` from it still left the output block emitted and the key with it. SHIM/BACK passed with the
# defect deliberately reintroduced. A key gated on ANOTHER key can only be caught in isolation.
alone=""
while read -r k v; do
    [ -z "$k" ] && continue
    case "$k" in \#*) continue ;; esac
    # run_type IS the baseline this compares against, so it can never differ from it. The other two
    # exemptions carry over from SHIM/BACK above, for the reasons documented there.
    case "$k" in evap_mode|runoff_ratio_on|run_type) continue ;; esac
    printf 'run_type equilibrium\n%s %s\n' "$k" "$v" > "$WORK/alone.txt"
    printf 'run_type equilibrium\n'                    > "$WORK/bare.txt"
    bash ../emit_config.sh < "$WORK/alone.txt" > "$WORK/alone.yaml" 2>/dev/null
    bash ../emit_config.sh < "$WORK/bare.txt"  > "$WORK/bare.yaml"  2>/dev/null
    cmp -s "$WORK/alone.yaml" "$WORK/bare.yaml" && alone="$alone $k"
done < "$WORK/shim_keys.txt"
if [ -n "$alone" ]; then
    echo "  FAIL  SHIM/ALONE these keys emit NOTHING when set on their own, so they are gated on some"
    echo "                   other key being present -- set one by itself and it silently does nothing:"
    for k in $alone; do echo "                     $k"; done
    fail=1
else
    echo "  PASS  SHIM/ALONE every legacy key does something even when it is the only key set"
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
    print("                   deliberately unadvertised -- grid is deprecated (#124) and dev is")
    print("                   developer-only -- but ordinary user keys here want documenting):")
    for p in sorted(gaps):
        print(f"                     {p or '<top level>'}: {' '.join(gaps[p])}")
PY

# RESOLVED: every schema key must be emitted into full_config.yaml. That file is the record of what a
# run did -- under a defaults-heavy schema the config a user WRITES no longer describes its own run --
# so a key the dump cannot see is a setting whose value is unrecoverable after the fact. This is the
# exact drift parameters.cpp:463 records for the older Parameters::print(): it "went dead once" and
# "drifted behind the config walk's new keys while runs quietly logged nothing; that gap turned a
# silently overridden setting into a wrong conclusion."
#
# STATIC by design: it reads the emitter's source rather than running the model, so it needs no input
# rasters and fires in any checkout. The limitation is real and worth stating -- it proves each key is
# WRITTEN, not that the value written is the one in force. Round-tripping catches that second class, and
# did: a run with storativity_surface 0.37 emitted 0.01, because the smoothing widths are parsed inside
# the solve and the dump ran before them.
python3 - "$ROOT/src/parameters.cpp" "$ROOT/src/WTM.cpp" <<'RESOLVEDPY' || fail=1
import sys, re
schema_src, emit_src = open(sys.argv[1]).read(), open(sys.argv[2]).read()
block = schema_src[schema_src.index("static const std::map<std::string, std::set<std::string>> schema")
                   :schema_src.index("return schema;")]
schema = {m.group(1): set(re.findall(r'"([a-z_]+)"', m.group(2)))
          for m in re.finditer(r'\{"([a-z_.]*)",\s*\{([^}]*)\}\}', block, re.S)}
fn = emit_src[emit_src.index("static void write_full_config("):]
fn = fn[:fn.index("\n}\n")]
emitted = set(re.findall(r'"(?:\\n)*\s*([a-z_]+):', fn))
allkeys = {k for keys in schema.values() for k in keys}
missing = sorted(allkeys - emitted)
if missing:
    print("  FAIL  RESOLVED   %d schema key(s) are never written to full_config.yaml:" % len(missing))
    for k in missing:
        print("                     %s" % k)
    print("                   A key the resolved dump cannot see is a setting no one can recover")
    print("                   from a finished run. Add it to write_full_config in src/WTM.cpp.")
    sys.exit(1)
print("  PASS  RESOLVED   all %d schema keys are written to full_config.yaml" % len(allkeys))
RESOLVEDPY

echo
if [[ $fail -eq 0 ]]; then echo "CONFIG SCHEMA: ALL PASSED"; else echo "CONFIG SCHEMA: FAILED" >&2; fi
exit $fail
