#!/usr/bin/env python3
"""Is each assertion's bound a number somebody chose, and does the bound mean what a reader thinks?

READ tests/ASSERTION_HEALTH.md FIRST. It defines the vocabulary this script prints, and it exists
because the word this script used to print -- "thin" -- carried two opposite meanings for a day.

WHAT THIS REPLACES, AND WHY THE RENAME. tol_margin.py computed `margin = tol/value` and printed the
low end of the list under the heading "N run within 10x of failing -- those are the ones that can go
red without the model changing." Two things were wrong with that framing, and neither was the
arithmetic:

  1. IT ASSUMED NON-DETERMINISM. "can go red without the model changing" is true of a cross-rank or
     cross-compiler comparison and FALSE of a serial, bit-reproducible one -- which most of this
     suite is. For a quantity whose spread is zero, no ratio above 1.0 can ever produce a spurious
     failure, so a low ratio is not a risk at all. It is a SHARP test, and sharp is what we want.
  2. SO THE SORT PROMOTED THE GOOD BOUNDS. Deriving a bound from a measurement puts it close to the
     measurement. Ranking by ratio therefore puts the carefully-derived bounds at the top and buries
     the guessed ones below them. Measured 2026-09-19: of the six lowest-ratio assertions in the
     tree, FIVE already carried derivations, several with swept tables; the one that did not was
     ranked third.

So the ratio is renamed `headroom` -- how many times the value may grow before the assertion fails --
and it is no longer the verdict. The verdict is the pair (headroom, was this bound DERIVED), because
`derived` is the signal that actually separates a considered bound from a typed one.

NO THRESHOLD, FOR REAL THIS TIME. tol_margin's docstring said it "does not carry a threshold of its
own -- that would just be another invented number, one level up", and then used 10.0 to decide what
to print. This prints EVERY assertion, sorted, and flags the ones with no derivation. Nothing here is
a cutoff.

HOW `derived` IS DETECTED, and its limits. A derivation lives in a comment block, and this repo
puts it in ONE OF TWO PLACES -- which the first version of this script got wrong, and its own output
caught: it looked only above the shell default (`NAME="${NAME:-VALUE}"`) and reported
budget_closure's most carefully derived arm as UNDERIVED, because that arm's swept table sits above
its INVOCATION, thirty lines from the variable. So both are read:

  the VARIABLE block   above `NAME="${NAME:-VALUE}"`   -- for a bound shared by every arm
  the ARM block        above `check "<label>" ...`     -- for a bound one arm overrides or explains

An output line is tied to an arm by its LABEL, which is quoted verbatim in run.sh and printed
verbatim in the output, so that link is exact rather than heuristic. Only the CONTENT test is
heuristic: a scientific-notation number, or one of a few words, taken as evidence somebody measured
something. That will be wrong both ways -- a derivation phrased without numbers reads as absent, an
unrelated nearby figure reads as present -- so it only ever REPORTS, never fails a suite.

Its deeper limit, stated so nobody mistakes a green column for a guarantee: a bound can be
documented, stable, and still WRONG, because the reasoning behind it was wrong. Nothing mechanical
catches that.
"""
import re, sys, os, glob

# THE BOUND MAY NAME ITSELF: "(tol TOL=0.0065)". The name makes the link to the shell default
# EXACT instead of inferred, which is what six separate linkage defects were all caused by --
# two bounds sharing a default value, several arms sharing one, a bound set as an inline env
# prefix, a python-side default invisible to a run.sh scan, a name not matching the expected
# pattern, and a bound after a semicolon. Each was patched by making the guess cleverer. The
# name ends the guessing, and is the same move the declared-config rule made for configs (#79):
# have the program state what it used rather than have a reader work it out.
# The bare form stays readable so a suite is never broken by not having been converted yet.
_TOL = re.compile(r"\(tol\s+(?:([A-Z_][A-Z_0-9]*)=)?([0-9.eE+-]+)\)")
# A FLOOR, NOT A CEILING. Some assertions are `value >= X`: "a lake must persist", "the two arms must
# really differ". They are the NON-VACUITY guards -- the checks that stop a suite passing while
# comparing nothing -- so leaving them unreadable would hide exactly the guards that matter most.
# Printing them as `(tol X)` would have been read as `value <= X` and reported as FAILING.
# headroom is defined symmetrically: value/X for a floor, X/value for a ceiling. In both, > 1 means
# the assertion passes with room, so one column stays comparable across both shapes.
_MIN = re.compile(r"\(min\s+(?:([A-Z_][A-Z_0-9]*)=)?([0-9.eE+-]+)\)")
_NUM = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*[eE][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+)")
# NAME="${NAME:-VALUE}" -- the convention every suite uses for an overridable bound.
# NOT ANCHORED AT LINE START: several suites put two bounds on one line --
#     TOL="${TOL:-2.5e-5}"; MB_TOL="${MB_TOL:-1e-3}"; PY="${PY:-python3}"
# and an anchored pattern found only the first, so the second was never probed and never
# reported. Scanned with finditer over the whole line instead.
# NAMES CONTAINING TOL *OR* MIN/MAX/FLOOR/BAR. Requiring TOL made every FLOOR invisible: a lower
# bound is naturally called BITE_MIN or DISTINCT_MIN, never *_TOL, so active_set's four
# non-vacuity guards were promoted to overridable defaults (#121) and the scanner still
# reported "no bounds found". The bite guards are exactly the ones this must not miss.
_DEF = re.compile(r'([A-Z_]*(?:TOL|MIN|MAX|FLOOR|BAR)[A-Z_]*)="\$\{\1:-([^}]*)\}"')
_SPREAD = re.compile(r"^\s*#\s*SPREAD:\s*(\S+)", re.M)
# Evidence that somebody measured something near the bound, rather than typing a round number.
_EVIDENCE = re.compile(r"[0-9]\.?[0-9]*[eE][+-]?[0-9]|MEASURED|DERIVED|swept|sweep|floor|noise", re.I)


def _strip_spread(blk):
    """Remove the `# SPREAD:` note, and only it, before looking for a DERIVATION.

    Q6 (does the value wobble?) and Q4 (where did the bound come from?) are SEPARATE
    questions, and the note that answers the first says "measured ... by
    assertion_probe.py" -- which _EVIDENCE matches on the word MEASURED. Left in, every
    bound carrying a spread note reads as derived, and #114's worklist went from 49 to
    ZERO the moment the notes were written. A tool that reports no work left because of
    its own annotation is worse than one that reports nothing.

    The note is the SPREAD line plus its hanging-indent continuations (`#` then several
    spaces), so a real derivation written in the same block still counts.
    """
    out, skipping = [], False
    for ln in blk.split("\n"):
        if re.match(r"^\s*#\s*SPREAD:", ln):
            skipping = True
            continue
        if skipping and re.match(r"^\s*#\s{4,}\S", ln):
            continue
        skipping = False
        out.append(ln)
    return "\n".join(out)


def _value_and_tol(line):
    """(worst-case value, tol) for a line stating both, else None.

    UNCHANGED FROM tol_margin.py -- this parsing was earned by being wrong twice and is left alone.
    Two rules settle which number on the line is the compared one:
      1. a compared quantity is a ratio or small difference, so candidates have magnitude <= 1. That
         drops "(row 23, col 23)" and "1.896865911e+10".
      2. where several remain, take the largest -- the smallest headroom. Over-reporting costs a
         glance; under-reporting is the failure this exists to catch.
    A BARE INTEGER IS NEVER THE VALUE: "1yr" once contributed a 1 that dominated every candidate and
    reported three passing conservation checks as failing.
    """
    t = _TOL.search(line)
    floor = False
    if not t:
        t = _MIN.search(line)
        floor = True
    if not t:
        return None
    try:
        tol = float(t.group(2))
    except ValueError:
        return None
    # A FLOOR TAKES THE NEAREST NUMBER, NOT THE LARGEST SMALL ONE. The `magnitude <= 1` rule below
    # exists to keep cell indices and absolute magnitudes out of a RATIO comparison, and a floor's
    # value is expected to be LARGE -- "max wtd = 9.9212 m (min 1.0)" has its only candidate above 1,
    # so that rule returned nothing at all. For a floor the number immediately before the marker is
    # the compared one; these lines are short and state one quantity.
    if floor:
        # A BARE INTEGER IS NEVER THE COMPARED VALUE -- the same rule the ceiling path applies, and
        # it was missing here purely because floors were added later. Without it "1yr" in a label
        # contributed a 1 that outranked the measurement in the lint, and could have been returned
        # as the value on a line whose real quantity came earlier.
        for n in reversed(_NUM.findall(line[: t.start()])):
            if "." not in n and "e" not in n and "E" not in n:
                continue
            try:
                return abs(float(n)), tol, True, t.group(1)
            except ValueError:
                continue
        return None
    cands = []
    for n in _NUM.findall(line[: t.start()]):
        if "." not in n and "e" not in n and "E" not in n:
            continue
        try:
            v = abs(float(n))
        except ValueError:
            continue
        if v <= 1.0:
            cands.append(v)
    if not cands:
        # NO CANDIDATE <= 1. The magnitude rule keeps absolute quantities out of a RATIO comparison,
        # but a ratio may legitimately exceed 1: xrank_growth asserts "last/first = 2747.06
        # (tol FLAT_MAX=10.0)" and that parsed to NOTHING, so the line vanished and its bound was
        # unprobeable. Fall back to the number NEAREST the marker, as a floor does.
        # CONSERVATIVE BY CONSTRUCTION: this path runs only where the previous code returned None,
        # so no line that parses today can change its answer.
        for n in reversed(_NUM.findall(line[: t.start()])):
            if "." in n or "e" in n or "E" in n:
                try:
                    return abs(float(n)), tol, False, t.group(1)
                except ValueError:
                    continue
        return None
    return (max(cands), tol, floor, t.group(1))


def claims_bound(line):
    """Does this line carry a bound marker at all? (#123)

    A line containing "(tol ...)" or "(min ...)" is CLAIMING to be an assertion. If the parser then
    extracts no value from it, that is a defect IN THE LINE -- and today it is indistinguishable
    from a line that was never an assertion, because both are simply skipped.

    THE CASE THAT PROMPTED THIS. direct_to_runoff printed

        SETTLING : gathered final per-cycle |Δwtd| = 0 m (tol 0.0001)

    A bare integer is discarded by design, so a line WITH a bound yielded no assertion. It was not
    misread; it was ABSENT, and the totals shrank without saying so. I found it only because the
    probe happened to report "governs no assertion that printed" and I chased that rather than
    accepting it. A suite with no probeable bound looks exactly the same.

    Other ways in, none of which had been checked: a value that happens to be whole and prints
    without a decimal; a nan or inf; a format the number pattern misses.
    """
    return bool(_TOL.search(line) or _MIN.search(line))


def ambiguous(line):
    """Does more than one number on this line plausibly answer "what was compared"? (#118)

    THE FAILURE IT CATCHES, six times in one day and twice from my own edits: the parser takes the
    largest candidate of magnitude <= 1 before the marker, so a small decimal anywhere in the PROSE
    outranks the measurement and the assertion is compared against its own label.

        "ΔV(0.25yr) = 1.2e-14 m (tol 2.5e-04)"        -> 0.25 wins, from the label
        "(topo=0.05/cell): residual = 3.5e-08 (tol 1e-06)" -> 0.05 wins, from the label
        "  1.0000  fixed  max|dV| = 3.1e-06 (tol 1e-04)"   -> the TIMESTEP column wins

    IT DOES NOT NEED TO KNOW WHICH NUMBER IS RIGHT -- only that there is more than one plausible
    answer, which is a fact about the line rather than a judgment about the suite. That is what makes
    it mechanical. It therefore over-reports: a line with two candidates whose largest happens to be
    the compared one is flagged and is fine. Over-reporting costs a glance; the failure it prevents
    is a silently wrong number in the one output whose job is ranking risk.

    FLOORS ARE EXEMPT: their rule takes the number NEAREST the marker, not the largest, so a second
    candidate earlier in the line cannot outrank it.
    """
    t = _TOL.search(line)
    if not t:
        # FLOORS ARE NOT EXEMPT. I had written "floors use the nearest-number rule; nothing to
        # confuse" and never tested it. Measured: "max wtd = 9.9212 m over 5 cells (min 1.0)" parses
        # to 5.0, and "spread = 18.0x ... 0.5 threshold (min 5.0)" parses to 0.5 -- reporting a
        # PASSING assertion at headroom 0.1. The nearest number is not always the compared one.
        # The SAME disagreement test applies: a reader's eye goes to the most prominent value, the
        # parser takes the nearest, and the line is ambiguous exactly when those differ.
        t = _MIN.search(line)
        if not t:
            return False
        ns = []
        for n in _NUM.findall(line[: t.start()]):
            if "." not in n and "e" not in n and "E" not in n:
                continue          # same rule as the value path: an integer is not a measurement
            try:
                ns.append(abs(float(n)))
            except ValueError:
                pass
        return len(ns) > 1 and max(ns) != ns[-1]
    small = []
    for n in _NUM.findall(line[: t.start()]):
        if "." not in n and "e" not in n and "E" not in n:
            continue
        try:
            v = abs(float(n))
        except ValueError:
            continue
        if v <= 1.0:
            small.append(v)
    if len(small) < 2:
        return False
    # THE TEST IS DISAGREEMENT BETWEEN THE TWO PLAUSIBLE RULES, not merely "more than one number".
    # A first version flagged 17 of 22 lines, most of them harmless: budget_closure prints
    # "cumulative=2.49e-07 worst-per-cycle=3.52e-07" and the largest IS the asserted one. Correct --
    # but correct by luck, and too noisy to act on.
    # A well-formed line puts the compared quantity immediately before the bound. So the line is
    # ambiguous exactly when LARGEST and NEAREST pick different numbers: that is the case where the
    # parser's rule and a reader's eye would disagree, and every historical misparse has this shape.
    if max(small) != small[-1]:
        return True
    # AND THE CASE THE DISAGREEMENT RULE MISSES: the BOUND repeated in the prose. direct_to_runoff
    # printed "gathered max wtd = 4.2e-03 m (<= 0.5, at surface) (tol 0.5)", where 0.5 is both the
    # nearest candidate AND the largest, so the two rules agree -- on the wrong number. A candidate
    # equal to the bound is the bound, not a measurement.
    try:
        tol = abs(float(t.group(2)))
    except ValueError:
        return False
    return any(v == tol for v in small)

def source_evidence(tests_dir):
    """(bounds, labels) per suite: where a derivation could be, read from each suite's run.sh.

    bounds[suite] = [(name, value, derived, spread)]   -- keyed by the default's VALUE
    labels[suite] = [(label, derived, spread)]         -- keyed by the arm's quoted LABEL

    TWO SOURCES BECAUSE THE REPO USES TWO. A bound shared by every arm is explained above the
    variable; a bound one arm overrides, or a per-arm measurement, is explained above that arm's
    invocation. Reading only the first reported budget_closure's best-derived arm as UNDERIVED.
    """
    bounds, labels = {}, {}
    for rs in sorted(glob.glob(os.path.join(tests_dir, "*", "run.sh"))):
        suite = os.path.basename(os.path.dirname(rs))
        try:
            lines = open(rs, errors="ignore").read().splitlines()
        except OSError:
            continue

        def block_above(i):
            out, j = [], i - 1
            while j >= 0 and lines[j].lstrip().startswith("#"):
                out.append(lines[j]); j -= 1
            return "\n".join(reversed(out))

        b, l = [], []
        for i, line in enumerate(lines):
            ms = list(_DEF.finditer(line))
            if ms:
                blk = block_above(i)
                sp = _SPREAD.search(blk)
                for m in ms:
                    try:
                        val = float(m.group(2))
                    except ValueError:
                        continue
                    b.append((m.group(1), val, bool(_EVIDENCE.search(_strip_spread(blk))), sp.group(1) if sp else None))
                continue
            # An arm invocation: a quoted label long enough to be distinctive. Short strings like
            # "off" or a stem would match half the output lines.
            # PAIR QUOTES PROPERLY, then filter. `[^"$]{18,}` looked right and was wrong: on a line
            # carrying an empty argument (COLL="") it pairs that string's CLOSING quote with the
            # label's OPENING one and captures the shell words between them, so the real label is
            # never seen and its derivation reads as absent. Caught by this script reporting
            # budget_closure's best-documented arm as unlinked.
            for lab in [q for q in re.findall(r'"([^"]*)"', line) if len(q) >= 18 and "$" not in q]:
                blk = block_above(i)
                sp = _SPREAD.search(blk)
                l.append((lab, bool(_EVIDENCE.search(_strip_spread(blk))), sp.group(1) if sp else None))
        if b:
            bounds[suite] = b
        if l:
            labels[suite] = l
    return bounds, labels


def classify(headroom, derived, spread, inverted=False, unverified=False):
    """The verdict. ASSERTION_HEALTH.md sec. 3: spread INVERTS how headroom reads.

    `spread?` is not an accusation. It means the headroom is low and nobody has recorded whether the
    quantity varies, so the number cannot be read either way yet -- declare `# SPREAD: 0` beside the
    bound once it has been checked, and it becomes `sharp`.
    """
    # AN XFAIL ASSERTS THE OPPOSITE: `value > tol` is the EXPECTED state, recording a known defect
    # so it cannot vanish unnoticed. Its headroom is below 1 BY DESIGN and means nothing on the same
    # scale as the others, so it is labelled and excluded rather than ranked. Before this, such a
    # line was simply unparseable and therefore invisible -- safe but accidental. The moment a suite
    # printed its bound in the readable form, the tool would have called a working xfail a FAILURE.
    if unverified:
        return "NOT ASKED"         # a coverage gap wearing a test's clothes -- see sec. 5b
    if inverted:
        return "pinned"            # a known defect, measured and held
    if derived is None:
        return "unlinked"          # printed tol tied to no arm and no default
    if not derived:
        return "UNDERIVED"         # the strongest signal, independent of headroom
    if spread == "0":
        return "sharp" if headroom < 10 else "derived"
    return "derived" if headroom >= 10 else "spread?"


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: assertion_health.py <suite-output-file-or-dir>...", file=sys.stderr)
        return 2
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(os.path.join(a, "*.out"))) if os.path.isdir(a) else [a])

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    bounds, labels = source_evidence(tests_dir)

    rows, unparsed = [], []
    for p in paths:
        try:
            text = open(p, errors="ignore").read()
        except OSError:
            continue
        suite = os.path.basename(p).rsplit(".", 1)[0]
        for line in text.splitlines():
            vt = _value_and_tol(line)
            if not vt:
                if claims_bound(line):
                    unparsed.append((suite, line.strip()))
                continue
            val, tol, floor, bname = vt
            if tol <= 0:
                continue
            headroom = (val / tol if tol else float("inf")) if floor else \
                       (float("inf") if val == 0 else tol / val)
            low = line.lower()
            # THREE DIFFERENT THINGS, and collapsing them loses the one that matters.
            #   pinned      a KNOWN DEFECT, measured and held so it cannot vanish unnoticed. The test
            #               asked its question and got a bad answer. That is a working test.
            #   unverified  the fixture CANNOT EXERCISE THE CLAIM. The test never asked. Not a
            #               failure -- an absence of evidence, and therefore a COVERAGE GAP that
            #               must not be counted as a test.
            # Andy, 2026-09-20: "It is not a failure. It is just that the test is not asking the
            # question. We should not use xfail in this case."
            unverified = "unverified" in low or "not asked" in low
            inverted = ("xfail" in low) or unverified
            amb = ambiguous(line)
            # ARM FIRST, then the shared default: an arm that explains its own bound is the
            # more specific statement, and the label match is exact rather than value-matched.
            derived = spread = None
            # NAME FIRST, and it is EXACT. Value-matching was the root of six linkage defects; when
            # the line names its bound there is nothing left to infer.
            if bname:
                named = [b for b in bounds.get(suite, []) if b[0] == bname]
                if len(named) == 1:
                    derived, spread = named[0][2], named[0][3]
            arm = [a for a in labels.get(suite, []) if a[0] in line]
            if arm:
                best = max(arm, key=lambda a: len(a[0]))
                derived, spread = best[1], best[2]
            if not derived:
                hits = [b for b in bounds.get(suite, []) if b[1] == tol]
                if len(hits) == 1:
                    derived = derived or hits[0][2]
                    spread = spread or hits[0][3]
                elif derived is None and not arm:
                    derived = spread = None
            rows.append((headroom, suite, line.strip(), derived, spread, inverted, unverified, amb))
    if not rows:
        return 0
    rows.sort(key=lambda r: r[0])

    # A LINE THAT SAYS PASS CANNOT BE REPORTED AS FAILING. Kept verbatim from tol_margin: when the
    # computed headroom is below 1 but the suite printed PASS/OK, the disagreement is THIS SCRIPT'S
    # parse. Seen the first time it ran -- an arm labelled "(runoff 0.3, FSM never runs)" reported
    # 0.00x because 0.3 in the LABEL beat the 6.73e-08 being compared. A false alarm in the one output
    # whose job is ranking real risk is worse than over-reporting.
    verdicted = lambda l: any(w in l for w in ("PASS", "OK  ", "ok  ", " OK "))
    misparsed = [r for r in rows if r[0] < 1.0 and verdicted(r[2]) and not r[5]]
    rows = [r for r in rows if r not in misparsed]

    under = [r for r in rows if r[3] is False]
    notasked = [r for r in rows if r[6]]
    unlinked = [r for r in rows if r[3] is None]
    nospread = [r for r in rows if r[3] and r[4] is None]

    print(f"  {len(rows)} assertions parsed, of {len(rows) + len(unparsed)} lines carrying a bound.")
    print("  EVERY ONE IS LISTED -- this")
    print("  carries no threshold. headroom = tol/value; see tests/ASSERTION_HEALTH.md for why a low")
    print("  headroom is a VIRTUE on a bit-reproducible quantity and a risk only on a varying one.")
    print(f"  {len(under)} carry no detectable derivation. {len(nospread)} have no declared SPREAD,")
    print("  so their headroom cannot yet be read either way.")
    ambs = [r for r in rows if r[7]]
    if ambs:
        print(f"  {len(ambs)} AMBIGUOUS -- more than one number before the bound could be the one")
        print("  compared, so the value reported for them may be from the LABEL (#118). Fix the LINE:")
        for r in ambs:
            print(f"      {r[1]:<22} {r[2][:74]}")
        print()
    if notasked:
        print(f"  {len(notasked)} are NOT ASKED -- the fixture cannot exercise the claim. These are")
        print("  COVERAGE GAPS, not tests, and must not be counted as either passing or failing.")
    print()
    print(f"      {'headroom':>10}  {'verdict':<10} {'suite':<22} assertion")
    for headroom, suite, line, derived, spread, inverted, unverified, amb in rows:
        # An inverted assertion's headroom is below 1 BY DESIGN, so printing "(FAIL)" beside it
        # would say the opposite of the truth. It gets no number: the ratio is not on the same
        # scale as the others and ranking it against them would be meaningless.
        if inverted:
            h = "       n/a"
        elif headroom < 1:
            h = "  0.00 (FAIL)"
        elif headroom == float("inf"):
            h = "       inf"
        else:
            h = f"{headroom:10.2f}"
        print(f"      {h}  {classify(headroom, derived, spread, inverted, unverified):<10} {suite:<22} {line[:78]}")

    if unparsed:
        print(f"  {len(unparsed)} line(s) STATE A BOUND BUT YIELD NO VALUE (#123). A line carrying")
        print("  (tol ...) or (min ...) is claiming to be an assertion; if no value can be read from")
        print("  it, the line is at fault -- and it would otherwise vanish from every count silently:")
        for suite, line in unparsed:
            print(f"      {suite:<22} {line[:74]}")
        print()
    if unlinked:
        print()
        print(f"  {len(unlinked)} could not be tied to a tolerance in any run.sh, so `derived` is unknown")
        print("  for them -- the printed tol matched no shell default, or matched more than one.")
    if misparsed:
        print()
        print(f"  {len(misparsed)} NOT RANKED: the suite printed a pass but this parse puts them below")
        print("  their tolerance, so the parse is wrong -- usually a bare number in the LABEL.")
        for _, suite, line, _, _, _, _, _ in misparsed:
            print(f"      {suite:<22} {line[:78]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
