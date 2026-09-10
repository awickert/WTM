#!/usr/bin/env python3
"""Derive a suite's config.yaml + per-arm overrides from harvested shim output (#83).

Reads $WTM_HARVEST_DIR/<suite>/*.yaml -- the configs the shim actually emitted during a run,
one per arm, in emission order -- and reports:

  BASE      lines identical across every arm; these become tests/<suite>/config.yaml
  OVERRIDE  keys that differ between arms; these stay in run.sh, where the reasoning lives
  TOKENS    fixture/work/stem paths rewritten to @INPUTS@ / @WORK@ / @STEM@

This does NOT write anything. It is a derivation aid: it tells you what the suite RESOLVED to,
so materialising it is transcription rather than reconstruction. The provenance comments the
shim emits (# baseline / # UNSTATED) are carried through untouched -- an UNSTATED marker in the
output is a value nobody chose, and must be decided rather than pasted.

Usage: materialize.py <suite> [harvest_dir]
"""
import os, re, sys
from collections import OrderedDict

def leaf_path(lines):
    """Map each line index -> dotted key path, using indentation. Comments/blank -> None."""
    stack, out = [], []
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith('#'):
            out.append(None); continue
        indent = len(ln) - len(ln.lstrip())
        key = ln.strip().split(':', 1)[0]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = '.'.join([k for _, k in stack] + [key])
        stack.append((indent, key))
        out.append(path)
    return out

def value_of(ln):
    """The value text of a line, provenance comment stripped."""
    if ':' not in ln: return ln.strip()
    v = ln.split(':', 1)[1]
    return v.split('#', 1)[0].strip()

def main():
    suite = sys.argv[1]
    hd = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('WTM_HARVEST_DIR', '/tmp/claude-1000/harvest')
    d = os.path.join(hd, suite)
    if not os.path.isdir(d):
        sys.exit(f"no harvest for {suite} at {d}")
    files = sorted(f for f in os.listdir(d) if f.endswith('.yaml'))
    if not files:
        sys.exit(f"no configs harvested for {suite}")
    arms = [open(os.path.join(d, f)).read().splitlines() for f in files]

    # Tokenise per-arm before comparing, so path differences do not masquerade as real overrides.
    toks = []
    for a in arms:
        src = stem = work = None
        for ln in a:
            s = ln.strip()
            if s.startswith('source:'):          src = value_of(ln).strip("'\"")
            if s.startswith('outfile_prefix:'):
                p = value_of(ln).strip("'\"")
                work, base = os.path.split(p)
                stem = base[:-1] if base.endswith('_') else base
        t = []
        for ln in a:
            if src:  ln = ln.replace(src, '@INPUTS@')
            if work: ln = ln.replace(work, '@WORK@')
            if stem: ln = re.sub(r'(?<=[/\'"])' + re.escape(stem) + r'(?=[_\'".])', '@STEM@', ln)
            t.append(ln)
        toks.append(t)

    # All arms share the shim's deterministic line ORDER, so compare by key path.
    per_arm = []
    for t in toks:
        paths = leaf_path(t)
        per_arm.append(OrderedDict((p, ln) for p, ln in zip(paths, t) if p))

    all_keys = list(per_arm[0].keys())
    for m in per_arm[1:]:
        for k in m:
            if k not in all_keys: all_keys.append(k)

    base, over = [], OrderedDict()
    for k in all_keys:
        vals = [m.get(k) for m in per_arm]
        if all(v == vals[0] for v in vals) and vals[0] is not None:
            base.append((k, vals[0]))
        else:
            over[k] = vals

    print(f"# suite: {suite}   arms: {len(arms)} ({', '.join(files)})")
    print(f"# BASE: {len(base)} settings identical across every arm")
    print(f"# OVERRIDE: {len(over)} settings differ between arms\n")
    print("=== BASE (this becomes tests/%s/config.yaml) ===" % suite)
    for _, ln in base: print(ln)
    print("\n=== PER-ARM OVERRIDES (keep in run.sh, with the reasoning) ===")
    for k, vals in over.items():
        print(f"  {k}:")
        for f, v in zip(files, vals):
            print(f"      {f}: {value_of(v) if v else '(absent)'}")
    uns = [ln for _, ln in base if 'UNSTATED' in ln]
    print(f"\n=== UNSTATED in base: {len(uns)} (each needs a decision, not a paste) ===")
    for ln in uns: print(ln)

main()
