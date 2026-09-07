#!/usr/bin/env python3
"""A test's config must ALREADY SAY everything the run resolved to.

WHY. Every silent-configuration defect found in this repo has the same shape: a key the test
did not write, filled in by the model, meaning something other than what the test's author
assumed. `solver.time_integration` absent is not backward-euler, it is `auto`; `adaptive_dt`
absent is not off; `dev.storage_form` absent is volume. Four suites were measuring something
other than what their arm names claimed, and every one of them passed.

Annotating WHERE each value came from would make that visible. Requiring the input config to
already CONTAIN every resolved key makes it impossible, which is stronger and needs no
per-key bookkeeping: if the model had to supply anything, the test was not explicit.

Compares the config handed to the model against the full_config.yaml that run wrote.
  MISSING  resolved but not declared  -- the test left it implicit. THE DEFECT THIS CATCHES.
  EXTRA    declared but not resolved  -- full_config does not round-trip the key; a gap in
                                         the record itself, so a run is not reproducible from it.
  DIFFER   declared and resolved, but not to the same value -- the model overrode the request,
                                         which is the dev.active_set class of defect (#28).

Values are compared SEMANTICALLY, not textually: "20yr" == "630720000s" == 630720000, and
"1e9" == "1000000000". A test may not be failed for writing a duration the way a human does.
"""
import sys, re, yaml

_UNITS = {"s": 1, "min": 60, "h": 3600, "d": 86400, "wk": 604800, "yr": 31536000}

def _dur(v):
    """Seconds if v parses as a duration ('20yr', '1.5d', '630720000s'), else None."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-z]*)\s*", str(v))
    if not m:
        return None
    n, u = float(m.group(1)), m.group(2)
    return n * _UNITS[u] if u in _UNITS else None

def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = node
    return out

def same(a, b):
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):          # bool/int: True == 1 must NOT slip through
        if isinstance(x, bool) != isinstance(y, bool):
            return False
    try:
        if float(a) == float(b):
            return True
    except (TypeError, ValueError):
        pass
    da, db = _dur(a), _dur(b)
    if da is not None and db is not None and da == db:
        return True
    return str(a).strip().strip("'\"") == str(b).strip().strip("'\"")

# Sections of full_config.yaml that are NOT settings. A test cannot be required to declare these:
# they are read from the input raster, not chosen by anyone. See the `derived` note in parameters.cpp.
DERIVED_SECTIONS = ("derived",)

def _settings_only(d):
    return {k: v for k, v in d.items() if not k.startswith(DERIVED_SECTIONS)}

def compare(input_path, resolved_path):
    """Returns (missing, extra, differ) as sorted lists of dotted keys."""
    with open(input_path) as f:
        declared = _settings_only(flatten(yaml.safe_load(f) or {}))
    with open(resolved_path) as f:
        resolved = _settings_only(flatten(yaml.safe_load(f) or {}))
    missing = sorted(k for k in resolved if k not in declared)
    extra   = sorted(k for k in declared if k not in resolved)
    differ  = sorted(k for k in resolved if k in declared and not same(declared[k], resolved[k]))
    return missing, extra, differ, declared, resolved

def scan(workdir):
    """Every <stem>.yaml in workdir that has a <stem>_prov/full_config.yaml beside it."""
    import glob, os
    out = []
    for prov in sorted(glob.glob(os.path.join(workdir, "*_prov", "full_config.yaml"))):
        stem = os.path.basename(os.path.dirname(prov))[:-5]
        inp = os.path.join(workdir, stem + ".yaml")
        if os.path.exists(inp):
            out.append((stem, inp, prov))
    return out

def main():
    if len(sys.argv) == 3 and sys.argv[1] in ("--summary", "--report"):
        runs = scan(sys.argv[2])
        if not runs:
            return 1                      # nothing to say; the caller stays quiet
        bad = []
        for stem, inp, prov in runs:
            try:
                m, e, d, dec, res = compare(inp, prov)
            except Exception:
                continue
            if m or e or d:
                bad.append((stem, m, e, d, dec, res))
        if sys.argv[1] == "--summary":
            if not bad:
                print(f"{len(runs)} runs, all explicit")
            else:
                worst = max(len(b[1]) + len(b[2]) + len(b[3]) for b in bad)
                print(f"{len(runs) - len(bad)}/{len(runs)} runs explicit "
                      f"({len(bad)} with implicit keys, worst {worst})")
            return 0
        for stem, m, e, d, dec, res in bad:
            print(f"    {stem}")
            for k in m: print(f"      MISSING  {k}: {res[k]!r}")
            for k in e: print(f"      EXTRA    {k}: {dec[k]!r}")
            for k in d: print(f"      DIFFER   {k}: asked {dec[k]!r}, ran {res[k]!r}")
        return 0
    if len(sys.argv) != 3:
        print("usage: config_identity.py <input.yaml> <full_config.yaml>\n"
              "       config_identity.py --summary|--report <workdir>", file=sys.stderr)
        return 2
    missing, extra, differ, dec, res = compare(sys.argv[1], sys.argv[2])
    if not (missing or extra or differ):
        print(f"OK  config is fully explicit ({len(res)} keys, all declared)")
        return 0
    print(f"FAIL  {sys.argv[1]} is not explicit "
          f"({len(missing)} missing, {len(extra)} extra, {len(differ)} differing)")
    for k in missing:
        print(f"    MISSING  {k}: {res[k]!r}   (resolved by the model; the config is silent)")
    for k in extra:
        print(f"    EXTRA    {k}: {dec[k]!r}   (declared, but full_config does not record it)")
    for k in differ:
        print(f"    DIFFER   {k}: asked {dec[k]!r}, ran {res[k]!r}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
