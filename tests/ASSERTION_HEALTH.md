# What makes a test assertion trustworthy

This document exists because one word carried two meanings for a day and cost three exchanges to
untangle. `tol_margin.py` called an assertion **thin** when `tol/value < 10`, and its docstring meant
that as *fragile* -- close enough to its bound that a rank count or a compiler could flip it. In
`#84` the same word got used for *suspicious, probably too loose*. Those are **opposite ends of the
same axis**, and conflating them produced a confident, wrong reading of the tool's output: six
top-ranked assertions were reported as needing repair when five of them were already correct, and
were ranked high *because* they were correct.

So: definitions first, in one place, and a statement of what each instrument does and does not see.

---

## 1. Definitions

An assertion is `value <= tol`. Three quantities describe its health, and they are not
interchangeable.

| term | definition | measured by |
|---|---|---|
| **value** | what the run actually produced | the suite, printed on the assertion line |
| **tol** | the bound it was compared against | the suite's `${NAME_TOL:-...}` |
| **headroom** | `tol / value` -- how many times `value` may grow before the assertion fails | `assertion_health.py` |
| **spread** | how much `value` moves under changes that are **not** defects: a repeat run, a different rank count, a different compiler or BLAS | **nothing, currently** -- see §5 |
| **derivation** | where `tol`'s number came from | a human, in a comment beside it |

**`headroom` was previously called `margin`, and the thin/wide language is retired.** A ratio needs a
name that says which direction is good, and `margin` does not; worse, "thin margin" reads as a
warning in English regardless of whether it is one here.

## 2. The two failure modes, and why headroom alone cannot tell them apart

| mode | what it looks like | what it costs |
|---|---|---|
| **FRAGILE** | `tol` sits barely above `value + spread` | the test goes red without the model changing. "Noise wearing a defect's clothes" -- it burns the reader's time and teaches nothing |
| **BLUNT** | `tol` sits so far above `value` that no plausible regression reaches it | the test is green forever. It reports success while asserting nothing |

Both are real, both have bitten this repo, and **they sit at opposite ends of the headroom axis.**
Low headroom suggests FRAGILE; high headroom suggests BLUNT. That is why a single sorted list cannot
be read as a worklist: the top of it and the bottom of it are different defects, and the middle is
healthy.

## 3. THE INVERSION RULE -- read this before interpreting any headroom number

> **When `spread == 0`, low headroom is not a risk. It is a virtue.**

If a quantity is bit-reproducible, no headroom above 1.0 can ever cause a spurious failure -- there is
nothing to perturb it. For such an assertion, headroom measures **only sensitivity**: how large a
regression must be before the test notices. A headroom of 2 means a regression need only double the
error to be caught. A headroom of 10 000 means it must grow four orders.

So for a deterministic quantity the reading **flips**: small headroom is a sharp test, large headroom
is a blunt one. `tol_margin.py`'s docstring assumed non-determinism -- *"a different rank count,
compiler, or BLAS will eventually flip it"* -- which is true of a cross-rank comparison and false of
a serial one. Most of this suite is serial.

**Measured 2026-09-19:** every one of the six lowest-headroom assertions in the tree reproduced its
recorded figure to every printed digit, across runs and across an unrelated model change. Their
spread is zero. Their low headroom was never a flake risk and five of the six were already correct.

**Corollary, and the reason this document exists:** *a tolerance ranks low on headroom precisely when
it was derived tightly.* Deriving a bound from a measurement puts it close to the measurement. An
instrument that sorts by headroom therefore promotes the well-made bounds and buries the guessed
ones.

## 4. The six questions a renewed suite must answer

The chain from "the binary ran" to "this verdict means something" has six links. Each has its own
failure, and each has bitten this repo at least once.

| # | question | failure when unchecked | machinery | state |
|---|---|---|---|---|
| **Q1** | Did the arm run the configuration it claims? | an arm silently becomes a copy of another; an omitted key resolves to something else | `config_identity.py`, coverage fingerprint | **enforced** |
| **Q2** | Does the compared quantity have structure? | comparing two identically-constant fields; a run that produced no answer | `nonvacuous.py` | **enforced** |
| **Q3** | Does the reference mean anything? | a golden regenerated for a forgotten reason | `golden.py` provenance, `#85` | **enforced** |
| **Q4** | Where did `tol` come from? | a round number nobody derived | `assertion_health.py` (detection only) | **partial** |
| **Q5** | Can the assertion fail at all? | a bound no regression could reach | ad hoc -- see §5 | **MISSING** |
| **Q6** | Will it fail spuriously? | flake; or, worse, a real failure dismissed as flake | nothing measures `spread` | **MISSING** |

Q1-Q3 are done and were each built after the corresponding defect was found green. Q4 is what `#84`
is. **Q5 and Q6 are the open gap, and Q6 is what makes Q4's output readable** -- without `spread` you
cannot say whether a given headroom is comfortable or reckless, which is exactly the confusion at the
top of this file.

## 5. What is missing, concretely

**Q6 -- measure `spread`.** Cheap where it matters: run the same arm twice and diff; where a suite
already runs multiple rank counts, diff across them. Most WTM quantities will come back exactly zero,
and that is the useful answer -- it licenses reading headroom as pure sensitivity. A suite can then
declare it once, beside the tolerance, rather than re-measuring every time:

```sh
# SPREAD: 0   verified <date> -- bit-identical across repeat runs and n=1,2,4
TOL="${TOL:-0.0065}"
```

**Q5 -- prove the assertion bites.** Every tolerance in this suite is already env-overridable
(`${NAME:-default}`), so a generic driver can re-run a suite with `NAME` set just below the measured
value and require a non-zero exit. Done by hand on 2026-09-19 for `variable_porosity`:

```
TOL=1e-3 ./run.sh   ->   FAIL AGREEMENT ... (tol 0.001)   RC=1
```

That is the whole test. It is not automated because one re-run per tolerance is ~30 runs of the
suite; it belongs in an occasional audit, not in `run_all.sh`.

**What neither would catch:** a bound that is documented, stable, and bites -- and is still the wrong
bound, because the derivation behind it was wrong. Only someone who knows the physics catches that.
The machinery here exists to spend that attention where it is needed, not to replace it.

## 5b. Inverted assertions (xfail) -- a fourth case the axis does not cover

Some suites assert `value > tol`: a KNOWN defect, recorded so it cannot vanish unnoticed, where
exceeding the bound is the EXPECTED state and falling below it is the alarm. `tests/storage_equivalence`
is the clearest -- `d <= tol` prints "UNEXPECTED PASS" and exits 1.

**Headroom is meaningless for these.** It is below 1 by design, so ranking them against ordinary
assertions compares two different things, and printing "(FAIL)" beside one says the opposite of the
truth. They are labelled `xfail`, given no headroom number, and excluded from the ranking.

**Why this is written down rather than just fixed:** an xfail line currently carries no parenthesised
bound, so it parses as nothing and is invisible. That is safe BY ACCIDENT. The moment such a suite is
brought into the print convention -- which is exactly what #117 does -- the tool would begin reading a
correctly-working xfail as a failure. Five of the sixteen already-visible suites contain xfail arms,
and three of the twelve to be onboarded do.

## 6. How to read `assertion_health.py`'s output

It sorts by headroom and reports both ends, labelled by what they mean rather than by a single word:

- **`SHARP`** -- low headroom, `spread` known to be 0. Healthy: a tight bound on a stable quantity.
- **`FRAGILE`** -- low headroom, `spread` unknown or non-zero. Look: this can go red on its own.
- **`BLUNT?`** -- high headroom on a quantity that is an approximation error rather than a
  conservation residual. Ask what regression this bound would actually catch.
- **`UNDERIVED`** -- no derivation found near the tolerance. Orthogonal to headroom, and the strongest
  single signal, because it marks a number nobody chose on purpose.

**A high headroom is not automatically BLUNT.** For a conservation assertion -- an exact-budget
residual at 1e-14 against a bound of 1e-6 -- a huge headroom is correct: any real break jumps orders,
and tightening toward the arithmetic noise floor would only manufacture fragility. The tool cannot
tell a conservation residual from an approximation error, and it does not try. It flags; a person
decides.
