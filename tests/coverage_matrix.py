#!/usr/bin/env python3
"""Turn the coverage fingerprints emitted by a suite run into a readable matrix.

WHERE THE DATA COMES FROM. WTM appends one line per run to the file named by WTM_COVERAGE_LOG,
recording what that run actually RESOLVED to -- after CLI overrides, after the solver-dependent
collector downgrade, after auto-enables. Not what a config file appears to say. run_all.sh sets the
variable and tags each test, so the log accumulates across the whole suite.

WHY CROSSINGS AND NOT JUST AXES. Every defect found on 2026-08-26 lived in an untested PAIR:
TR-BDF2 x active_set, adaptive x budget, explicit x budget, runoff_ratio x sub-stepping. Each axis was
covered on its own; the combination was not. So the useful view is which crossings nothing exercises.

Usage:
  coverage_matrix.py <fingerprint-log> [-o tests/COVERAGE.md]
"""
import collections
import sys

# The axes worth crossing, in the order they appear in the report.
AXES = ["run_type", "solver", "integrator", "dtctl", "collector",
        "fsm", "runoff_ratio", "infiltration", "recharge_path", "coupling", "boundary", "ranks"]

# Pairs to CROSS in the report. Not every axis pair is informative; these are the ones where a gap has
# historically meant a bug.
CROSSINGS = [("solver", "collector"), ("integrator", "collector"), ("dtctl", "collector"),
             ("run_type", "collector"), ("solver", "integrator"), ("run_type", "solver"),
             ("fsm", "collector"), ("runoff_ratio", "dtctl")]

# Combinations WTM refuses by design. Listed so the gap report does not cry wolf about them -- an
# absent cell here is correct behaviour, not missing coverage.
BY_DESIGN = {
    ("solver=picard", "collector=active_set"):
        "hard error: the pin is absent from the Picard operator",
    ("coupling=fsm_delta_source", "collector=active_set"):
        "hard error: the pin reads the table fsm_delta_source suppresses",
    ("dtctl=continuation", "solver=anderson"): "continuation is the Newton path",
    ("dtctl=continuation", "solver=picard"):   "continuation is the Newton path",
    ("dtctl=continuation", "collector=implicit"): "implicit + sub-stepping cannot converge",
    ("dtctl=adaptive", "collector=implicit"):
        "implicit is dt-DEPENDENT: shrinking dt moves the solution, so the controller cannot settle",
    ("integrator=be_secant", "collector=active_set"):
        "active_set needs a b=0 residual path, so it auto-enables volume storage: the secant "
        "backward-Euler form is unreachable with it",
}


# The COMBINATION view: the (solver, collector, integrator) triple against run_type. This is the
# question the pairwise tables CANNOT answer -- two pairs can each be covered while their combination
# never runs. Asked directly ("does the table verify we use every combination ... for both equilibrium
# and transient?") the pairwise answer was no, and this is what answers it.
TRIPLE = ("solver", "collector", "integrator")


def combination_rows(runs):
    """[(triple, {run_type: count})], sorted, plus the run_types actually seen anywhere."""
    seen = collections.defaultdict(collections.Counter)
    for r in runs:
        if all(k in r for k in TRIPLE) and "run_type" in r:
            seen[tuple(r[k] for k in TRIPLE)][r["run_type"]] += 1
    rts = sorted({rt for c in seen.values() for rt in c})
    return sorted(seen.items()), rts


def combination_table(runs):
    rows, rts = combination_rows(runs)
    out = ["| " + " | ".join(TRIPLE) + " | " + " | ".join(f"`{t}`" for t in rts) + " |",
           "|" + "---|" * (len(TRIPLE) + len(rts))]
    for triple, counts in rows:
        cells = [f"`{v}`" for v in triple] + [str(counts[t]) if counts[t] else "**0**" for t in rts]
        out.append("| " + " | ".join(cells) + " |")
    return out, rows, rts

BEGIN = "<!-- COVERAGE:BEGIN -->"
END = "<!-- COVERAGE:END -->"

# The README carries a COMPACT view: the crossings a reader actually needs to judge "which
# configurations are tested?", not all eight tables. The full detail lives in tests/COVERAGE.md.
README_CROSSINGS = [("solver", "collector")]


def crossing_table(runs, a, b):
    """One markdown table. Cell = number of runs, `n/a` where WTM refuses the pair, **0** for a gap."""
    va = sorted({r[a] for r in runs if a in r})
    vb = sorted({r[b] for r in runs if b in r})
    seen = collections.Counter((r.get(a), r.get(b)) for r in runs)
    out = [f"| {a} \\ {b} | " + " | ".join(f"`{y}`" for y in vb) + " |",
           "|" + "---|" * (len(vb) + 1)]
    for x in va:
        cells = []
        for y in vb:
            n = seen.get((x, y), 0)
            if n:
                cells.append(str(n))
            elif BY_DESIGN.get((f"{a}={x}", f"{b}={y}")) or BY_DESIGN.get((f"{b}={y}", f"{a}={x}")):
                cells.append("n/a")
            else:
                cells.append("**0**")
        out.append(f"| **`{x}`** | " + " | ".join(cells) + " |")
    return out


def write_readme(path, runs, gaps):
    try:
        src = open(path).read()
    except FileNotFoundError:
        return 0
    if BEGIN not in src or END not in src:
        return 0
    ntests = len({r.get("test") for r in runs})
    B = [BEGIN, ""]
    B.append(f"*Regenerated by every `tests/run_all.sh`. Each cell counts the runs that exercised that "
             f"combination: **0** marks a combination nothing tests, `n/a` one the model refuses by "
             f"design.*")
    B.append("")
    ctab, crows, crts = combination_table(runs)
    both = eqonly = 0
    if "equilibrium" in crts and "transient" in crts:
        both = sum(1 for _, c in crows if c["equilibrium"] and c["transient"])
        eqonly = sum(1 for _, c in crows if c["equilibrium"] and not c["transient"])
    B.append(f"**{len(runs)} runs** across **{ntests} tests**. **{len(crows)}** distinct "
             f"(solver, collector, integrator) combinations are exercised; of those **{both}** run in "
             f"both equilibrium and transient and **{eqonly}** are equilibrium-only. Full detail, "
             f"every pairwise crossing and the gap list: "
             f"[`tests/COVERAGE.md`](tests/COVERAGE.md).")
    B.append("")
    B.append("**Which combinations are tested, and at which run type**")
    B.append("")
    B += ctab
    for a, b in README_CROSSINGS:
        B.append("")
        B.append(f"**Coarser view: {a} × {b}**")
        B.append("")
        B += crossing_table(runs, a, b)
    B += ["", END]
    head, rest = src.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    open(path, "w").write(head + "\n".join(B) + tail)
    return 1

def parse(path):
    runs = []
    with open(path) as f:
        for line in f:
            if not line.startswith("coverage "):
                continue
            d = {}
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d[k] = v
            runs.append(d)
    return runs


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    log = sys.argv[1]
    argv = sys.argv[2:]
    out = argv[argv.index("-o") + 1] if "-o" in argv else None
    readme = argv[argv.index("--readme") + 1] if "--readme" in argv else None
    try:
        runs = parse(log)
    except FileNotFoundError:
        sys.stderr.write(f"no fingerprint log at {log}; run the suite with WTM_COVERAGE_LOG set\n")
        return 2
    if not runs:
        sys.stderr.write(f"{log} contains no coverage lines\n")
        return 2

    L = []
    w = L.append
    w("# WTM test-coverage matrix\n")
    w("**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it "
      "reflects what each run RESOLVED to rather than what its config appears to say. Do not edit "
      "by hand; re-run the suite.\n")
    w(f"Runs recorded: **{len(runs)}** across **{len({r.get('test') for r in runs})}** tests.\n")

    # ---- 0. combination coverage ------------------------------------------------------------------
    w("\n## 1. Combination coverage: every (solver, collector, integrator) against run type\n")
    w("The pairwise tables further down CANNOT answer this: two pairs can each be covered while their "
      "combination never runs. **0** means that combination has never been run at that run type.\n")
    ctab, crows, crts = combination_table(runs)
    for line in ctab:
        w(line)
    if "equilibrium" in crts and "transient" in crts:
        both = [t for t, c in crows if c["equilibrium"] and c["transient"]]
        eqonly = [t for t, c in crows if c["equilibrium"] and not c["transient"]]
        tronly = [t for t, c in crows if c["transient"] and not c["equilibrium"]]
        w(f"\n**{len(crows)}** distinct combinations are exercised at all. Of those, **{len(both)}** run "
          f"in BOTH equilibrium and transient, **{len(eqonly)}** are equilibrium-only and "
          f"**{len(tronly)}** transient-only.\n")
        if eqonly:
            w("Equilibrium-only, i.e. never exercised on the transient path:\n")
            for t in eqonly:
                w("- `" + "` x `".join(t) + "`")
            w("")

    # ---- 1. what each test covers -----------------------------------------------------------------
    w("\n## 2. What each test covers\n")
    by_test = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in runs:
        for a in AXES:
            if a in r:
                by_test[r.get("test", "unknown")][a].add(r[a])
    show = ["run_type", "solver", "integrator", "dtctl", "collector", "fsm", "runoff_ratio", "ranks"]
    w("| test | " + " | ".join(show) + " |")
    w("|" + "---|" * (len(show) + 1))
    for t in sorted(by_test):
        cells = [",".join(sorted(by_test[t].get(a, {"-"}))) for a in show]
        w(f"| `{t}` | " + " | ".join(cells) + " |")

    # ---- 2. crossings -----------------------------------------------------------------------------
    w("\n## 3. Pairwise crossings\n")
    w("A blank cell is a combination **no run exercises**. `by design` marks the ones WTM refuses on "
      "purpose -- those blanks are correct, not gaps.\n")
    gaps = []
    for a, b in CROSSINGS:
        va = sorted({r[a] for r in runs if a in r})
        vb = sorted({r[b] for r in runs if b in r})
        if not va or not vb:
            continue
        seen = collections.Counter((r.get(a), r.get(b)) for r in runs)
        w(f"\n### {a} x {b}\n")
        w("| " + a + " \\ " + b + " | " + " | ".join(vb) + " |")
        w("|" + "---|" * (len(vb) + 1))
        for x in va:
            row = []
            for y in vb:
                n = seen.get((x, y), 0)
                if n:
                    row.append(str(n))
                else:
                    why = BY_DESIGN.get((f"{a}={x}", f"{b}={y}")) or BY_DESIGN.get((f"{b}={y}", f"{a}={x}"))
                    if why:
                        row.append("by design")
                    else:
                        row.append(" ")
                        gaps.append((f"{a}={x}", f"{b}={y}"))
            w(f"| **{x}** | " + " | ".join(row) + " |")

    # ---- 3. the gap list --------------------------------------------------------------------------
    w("\n## 4. Uncovered pairwise crossings\n")
    if not gaps:
        w("None: every crossing above is either exercised or refused by design.\n")
    else:
        w(f"**{len(gaps)}** combinations are reachable but exercised by nothing:\n")
        for x, y in gaps:
            w(f"- `{x}` x `{y}`")
        w("\nEach is a place a defect could live unseen. That is not a demand to cover all of them -- "
          "some are uninteresting -- but the list should be read, and anything load-bearing should "
          "get an arm.\n")

    text = "\n".join(L) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {out} ({len(runs)} runs, {len(gaps)} uncovered crossings)")
    else:
        print(text)

    if readme:
        n = write_readme(readme, runs, gaps)
        if n:
            print(f"wrote the coverage summary into {readme}")
        else:
            sys.stderr.write(f"{readme}: no {BEGIN} / {END} markers found; summary not written\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
