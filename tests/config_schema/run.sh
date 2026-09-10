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

# ---- NAMESPACE: the -wtm_ command-line namespace is RETIRED, in both directions ---------------------
# Every WTM setting is a config key. The options-database round-trip -- YAML -> string -> PETSc options
# DB -> re-parsed -- is gone (#86), and with it the second route into a setting: set_opt_if_unset took
# the FIRST setter, so a -wtm_ typed on the command line silently WON over the config key it duplicated
# and full_config.yaml recorded the winner, leaving the file the user was reading wrong.
#
# THE SOURCE LOCK lives here; the RUNTIME lock lives in tests/runoff_collector, which has a fixture (a
# run started from this suite's reference config aborts in GDAL long before it reaches the guard).
# BOTH are needed: the runtime guard fires only on an option NOTHING READ, so one
# PetscOptionsGetReal(..., "-wtm_x", ...) would quietly make -wtm_x acceptable again and the runtime arm
# would still pass. Only counting call sites catches that.
NSRC=$(command grep -rc 'PetscOptions[A-Za-z]*(.*"-wtm_' "$ROOT/src" 2>/dev/null | command grep -v ':0' | wc -l)
if [ "$NSRC" -eq 0 ]; then
    echo "  PASS  NAMESPACE  no source file reads a -wtm_ option (0 call sites)"
else
    echo "  FAIL  NAMESPACE  $NSRC source file(s) read a -wtm_ option. Every setting must arrive through"
    echo "        Parameters; a PetscOptions read of -wtm_ reopens the command-line route:"
    command grep -rn 'PetscOptions[A-Za-z]*(.*"-wtm_' "$ROOT/src" | sed 's/^/          /'
    fail=1
fi

# ---- RETIRED: a key that was REMOVED must abort, not drift ------------------------------------------
# dev.active_set was a SECOND YAML route to the same enforcement as surface_water.collection.method,
# and it silently OVERRODE an explicit method: with `method: explicit` plus `dev: {active_set: true}`
# the run used active_set instead, differing on 54 of 256 cells (max 0.127 m) with NO log line. It was
# removed 2026-09-01. This arm pins the REMOVAL: a config carrying the old key must say so and stop.
# Distinct from REJECT above, which uses an invented key -- this one is a real spelling that used to
# work, which is exactly the case a user upgrading an old config will hit.
for rk in dev.active_set dev.allow_aboveground_water_columns; do
inject "$WORK/retired.yaml" "$rk"
OUT=$(msg "$WORK/retired.yaml")
if echo "$OUT" | grep -q "unrecognised key" && echo "$OUT" | grep -q "$rk"; then
    echo "  PASS  RETIRED    the removed key '$rk' aborts and is named"
else
    echo "  FAIL  RETIRED    '$rk' was accepted. It is a SECOND route to the active-set"
    echo "        enforcement and silently overrides surface_water.collection.method -- if it is back in"
    echo "        the schema, the dual-route defect is back. See task #28 (dev.active_set) and #35"
    echo "        (dev.allow_aboveground_water_columns, read and then overwritten by every branch of"
    echo "        the collector selector, so it could not affect any run)."
    fail=1
fi
done

# ---- MIGRATED: two keys merged into one, so the refusal must TRANSLATE, not just reject ------------
# surface_water.mode and surface_water.fsm_coupling became the single key surface_water.routing
# (2026-09-10). Distinct from RETIRED above: those keys went away, and "unrecognised key ... did you
# mean" is a sufficient answer. Here the VALUES remap as well --
#     mode: routed + fsm_coupling: continuous  ->  routing: continuous
#     mode: ponded (or removed)                ->  routing: off
# -- so a bare "did you mean routing?" would send a user to a key whose obvious value produces a
# DIFFERENT RUN. The refusal therefore has to carry the table, and this arm pins that it does.
#
# WHY THE MERGE: the split made a contradiction writable. `mode: ponded` with `fsm_coupling:
# continuous` asks for a coupling that cannot happen, and full_config.yaml then recorded
# `fsm_coupling: impulse` for a run in which NO coupling ran -- so 72 FSM-off test configs would have
# had to declare a mechanism they never used (#89).
for mk in surface_water.mode surface_water.fsm_coupling; do
python3 - "$REF" "$WORK/migrated.yaml" "$mk" <<'PYEOF'
import sys, yaml
ref, out, path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(ref))
# A REAL old spelling, not a bogus value: this is what an upgrading user's file actually contains.
cfg["surface_water"].pop("routing", None)
cfg["surface_water"][path.split(".")[1]] = "routed" if path.endswith("mode") else "continuous"
yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False)
PYEOF
OUT=$(msg "$WORK/migrated.yaml")
if echo "$OUT" | grep -q "surface_water.routing" && echo "$OUT" | grep -q -- "-> *routing: off"; then
    echo "  PASS  MIGRATED   '$mk' aborts, names surface_water.routing, and gives the translation"
else
    echo "  FAIL  MIGRATED   '$mk' did not produce the migration message with its value table."
    echo "        Either the key is accepted again -- which would let 'mode: ponded' coexist with"
    echo "        'fsm_coupling: continuous', the contradiction the merge removed -- or the refusal"
    echo "        lost the routed/ponded -> continuous/off translation, leaving an upgrading user to"
    echo "        guess a value. See task #89."
    fail=1
fi
done

# ---- ARGV: an argument nothing reads is refused, an option and its value are not -----------------
# The model aborts on a -wtm_ FLAG nothing consumed; until 59b7006 it had no equivalent for a
# positional ARGUMENT and took any number of them silently -- `wtm.x a.yaml b.yaml` ran a.yaml and
# ignored b.yaml, exit 0. That is the same defect class as a swallowed config key: a thing the caller
# asked for that had no effect and no complaint.
#
# It is tested HERE, beside the schema refusals, because it is the same property one level out: what
# the model is GIVEN must either be used or refused. The negative arm matters as much as the positive
# -- a guard that also rejected `-snes_stol 1e-8` would make every arm in this suite unrunnable.
ARGOK=0
OUT=$("$WTM" "$REF" second.yaml 2>&1 || true)
if echo "$OUT" | grep -q "nothing reads" && echo "$OUT" | grep -q "second.yaml"; then
    echo "  PASS  ARGV-STRAY a second positional argument aborts and is NAMED"
else
    echo "  FAIL  ARGV-STRAY 'wtm.x cfg second.yaml' did not abort. An extra argument has no effect on"
    echo "        the run, so accepting it silently means a caller can pass a second config -- or a"
    echo "        quoting mistake -- and never learn it did nothing. See 59b7006."
    ARGOK=1; fail=1
fi
# POSITIVE assertion, not "no error appeared". $WORK/valid.yaml -- which this arm used at first -- was
# never created by this suite, so the model aborted on a missing file, printed no "nothing reads", and
# the arm PASSED having tested nothing. Requiring the run to REACH the config-reading stage cannot be
# satisfied that way: the guard runs before it, so the message only appears if the option and its value
# were let through.
OUT=$("$WTM" "$REF" -snes_stol 1e-8 2>&1 || true)
if echo "$OUT" | grep -q "nothing reads" || ! echo "$OUT" | grep -q "Reading configuration file"; then
    echo "  FAIL  ARGV-OPT   an option and its VALUE were mistaken for stray arguments. '-snes_stol 1e-8'"
    echo "        is two argv entries and must pass; a guard this blunt would break every suite."
    ARGOK=1; fail=1
elif [ "$ARGOK" -eq 0 ]; then
    echo "  PASS  ARGV-OPT   '-snes_stol 1e-8' is accepted (an option's value is not a stray)"
fi

# THE FOUR SHIM ARMS WERE DELETED WITH THE SHIM (#83, 2026-09-10). They asserted that every key
# tests/emit_config.sh could emit validated against this dictionary, that every key it was GIVEN
# changed what it emitted, that each did something alone, and that fsm_on/fsm_coupling mapped onto
# surface_water.routing. All four took the shim as their SUBJECT, so they had nothing left to test
# once every suite read a real config file. Their history is in git, not in a stub.

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
