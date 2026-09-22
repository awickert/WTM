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

## 5a. Two shapes of bound: ceiling and floor

Not every bound is an upper limit. `fsm_conservation` asserts `lake > 1.0` ("a lake must persist")
and `state_gap > 1e-3` ("the two couplings really differ"). **These are the NON-VACUITY guards** --
the checks that stop a suite passing while comparing nothing -- so leaving them unreadable would hide
exactly the guards that matter most.

| printed | assertion | headroom |
|---|---|---|
| `(tol X)` | `value <= X` | `X / value` |
| `(min X)` | `value >= X` | `value / X` |

**Headroom is defined so that > 1 means "passes with room" in both shapes**, which is what keeps one
column comparable across the two. Printing a floor as `(tol X)` would have been read as a ceiling and
reported as FAILING a test that was working.

One implementation note, because it is a real difference and not an oversight: a floor takes the
number NEAREST the marker, while a ceiling takes the largest candidate of magnitude <= 1. That
`<= 1` rule exists to keep cell indices and absolute magnitudes out of a *ratio* comparison, and a
floor's value is expected to be large -- `max wtd = 9.9212 m (min 1.0)` has its only candidate above
1, so the ceiling rule returned nothing at all.

## 5c. Seven assertions the convention cannot express, and never will

A full scan of every assertion in the twelve onboarded suites (2026-09-20) found 22 numeric bounds
-- 18 ceilings, 4 floors -- and **7 gates with no numeric bound at all**:

```
direct_to_runoff:  below_ok, at_surface
flicker_evap:      below_ok, taper_alone
limit_cycle:       below_ok, exfiltration, relax_ok
```

These are predicates: *every cell is at or below the surface*; *under-relaxation at a = 1.0 changes
nothing*. There is no threshold to state, so three of the six questions simply do not apply:

- **Q4 (derive the bound)** -- there is no bound.
- **Q6 (measure spread)** -- there is no number to watch drift.
- **Q5 (can it fail?)** -- **the bite harness cannot reach them.** It works by tightening a bound from
  outside; a predicate has none. Proving one of these can fail means perturbing THE MODEL, not the test.

**This is a limit of the framework, not a to-do.** It is written down because a silent gap is how a
count becomes a lie: "22 assertions checked" is true and also leaves seven unexamined. Whether a
predicate is worth checking by model perturbation is a separate question and a more expensive one.

**A NOTE ON WRITING THE LINES THEMSELVES.** Three assertions were mis-read on their first attempt
because a bare decimal elsewhere on the line outranked the measured value -- `ΔV(0.25yr)` in a label,
a `1.0000` dt column, a ladder of p-values. This is the documented failure mode, and its documented
remedy is to FIX THE LINE, not to teach the parser. In practice:

- put the compared quantity immediately before the marker, and nothing else numeric after it;
- move context (ladders, breakdowns, per-scheme detail) to a separate line or after the marker;
- print the DEVIATION when the assertion bounds a deviation -- `|p - expected| = 0.0021 (tol 0.2)`
  rather than leaving a reader to subtract two printed numbers;
- never put the bound itself in the prose before the marker: it is a number <= 1 and the parser will
  happily compare the assertion against itself.

## 5b. Three things that are not a pass, and only one of them is a failure

Some suites assert `value > tol`. Collapsing them all under `xfail` loses the distinction that
matters most, so they are separated:

| label | what it means | is the test working? |
|---|---|---|
| **`pinned`** | a KNOWN DEFECT, measured and held so it cannot vanish unnoticed. `budget_closure` names the task in `XTASK`; `newton_solver` pins a Jacobian mismatch; `estimator_order` records that BDF2-on-V has no order with FSM on | **yes.** It asked its question and got a bad answer, and it will shout if the answer moves |
| **`NOT ASKED`** | the fixture CANNOT EXERCISE THE CLAIM. `storage_equivalence` cannot satisfy its own stated preconditions, so the identity is *neither confirmed nor denied* | **no.** This is a COVERAGE GAP wearing a test's clothes |

> Andy, 2026-09-20: *"It is not a failure. It is just that the test is not asking the question. We
> should not use xfail in this case."*

**Why the distinction is the whole point.** `pinned` and `NOT ASKED` produce the same tidy line in a
test log forever. But `pinned` is evidence and `NOT ASKED` is the absence of it, and absence of
evidence is exactly the defect class this framework exists to catch -- the same shape as `#34`, `#91`
and `#96`, where suites ran green while comparing identically-constant fields. A suite that never
asks its question will report a stable status until someone reads the prose underneath it.

**So `NOT ASKED` is counted against COVERAGE, not as a test.** `assertion_health.py` reports it in its
own line and excludes it from both the passing and failing counts. Before this, `storage_equivalence`
appeared as a working suite in every figure quoted from these tools.

**Headroom is meaningless for both**, and is printed as `n/a` rather than as a ratio below 1 -- which
would have read as "(FAIL)" beside a test behaving exactly as designed.

**A note on how this was nearly missed.** These lines carry no parenthesised bound today, so they
parse as nothing and never reach the classifier. That safety was ACCIDENTAL. `#117` exists to bring
unreadable suites INTO the print convention, which would have turned an accidental invisibility into
an active misreading -- on the first suite queued for editing. Five of the sixteen already-visible
suites contain such arms; three of the twelve queued do.

## 5d. Two ways a line can be wrong about itself

Both are reported, never fatal, and they are opposite failures of the same parse.

**AMBIGUOUS (#118)** -- more than one number on the line could be the compared one, and the
parser's rule disagrees with where a reader's eye lands. Six real cases, two of them introduced
while writing this framework. The test is *largest vs nearest disagree*, plus the special case of
the bound repeated in the prose.

**NO VALUE (#123)** -- the line carries `(tol ...)` or `(min ...)`, so it CLAIMS to be an
assertion, and the parser can extract nothing from it. Today that would make it vanish: skipped
exactly like a line that was never an assertion, with the totals quietly shrinking.

> `direct_to_runoff` printed `|Δwtd| = 0 m (tol 0.0001)`. A bare integer is discarded by design,
> so a **perfect result** made the line unparseable and its bound unreachable. It was found by
> accident -- the probe said "governs no assertion that printed" and that was chased rather than
> accepted.

The headline count is therefore stated as **"N assertions parsed, of M lines carrying a bound"**.
If those differ, something is missing and says so.

### The rules for writing an assertion line

Earned by getting each one wrong at least once:

1. put the compared quantity **immediately before** the marker, and nothing numeric after it;
2. move ladders, breakdowns and per-scheme detail to their own line, or after the marker;
3. print the **deviation** when the assertion bounds a deviation -- `|p - expected| = 0.0021
   (tol PTOL=0.2)` rather than leaving a reader to subtract two printed numbers;
4. never print the bound itself in the prose before the marker -- the parser will compare the
   assertion against itself;
5. name the bound: `(tol TOL=0.0065)`, not `(tol 0.0065)`. The name makes the link exact instead
   of inferred, and tells a reader which variable to override;
6. a value of exactly zero still needs a decimal -- `0.000e+00`, never `0`.

## 5d-bis. How to write a derivation (Q4)

A derivation says **where the number came from**. It is not a restatement of what the bound
measures -- every bound here already carries that on its definition line. Two shapes, and the first
is far stronger.

### Shape 1 -- a SEPARATING bound (use this whenever the suite provides it)

Many suites run both a healthy arm and a broken-by-construction one, so they measure **both edges of
the gap the bound sits in**. Then the derivation needs no convention at all:

```
# DERIVED 2026-09-22, SEPARATING: `explicit` clamps to the surface and measures 0.0000e+00 m
#   exactly, while `off` in this same suite piles water to 192.46 m. The bound sits between two
#   MEASURED values, 4 orders above the clamped one and 6 below the unclamped.
```

Nothing is assumed. A regression has to cross a gap whose far side was observed, not guessed.

### Shape 2 -- a ONE-SIDED bound (only the healthy value is measurable)

State the measurement, then say plainly that the multiplier is a **convention**:

```
# DERIVED 2026-09-22, ONE-SIDED: measured 2.520e-07 on this fixture. Bound 1e-3 = 4000x that,
#   a CONVENTION not a measurement -- no arm here produces the broken value to bound against.
```

The words "a convention not a measurement" are the point. A multiplier dressed up as though it were
derived is worse than an undocumented number, because it stops anyone looking again.

### Say so when a bound is BLUNT

If the headroom runs to thousands and the quantity is bit-reproducible, the bound cannot catch a
small regression. That is a finding about the test, and it belongs in the derivation where the next
reader will see it -- not quietly left in a passing suite.

### What does NOT count as a derivation

- Restating the units or the subject ("metres of water volume") -- that is the definition line.
- "Seems reasonable", "conservative", "should be plenty".
- A spread note. Q6 and Q4 are different questions; `_strip_spread` enforces the separation after
  the notes' own wording ("measured ... by assertion_probe.py") marked 49 bounds derived by
  accident.

## 5d-ter. A bound's NAME must contain TOL, MIN, MAX, FLOOR or BAR

`assertion_health.py` finds bounds with

```python
_DEF = re.compile(r'([A-Z_]*(?:TOL|MIN|MAX|FLOOR|BAR)[A-Z_]*)="\$\{\1:-([^}]*)\}"')
```

so a shell default whose name lacks one of those five words is **invisible to the framework**, no
matter how correctly it is written. The regex is deliberately narrow: a rule like "any all-caps
shell default" would sweep up `WTM`, `PY`, `RANKS`, `WORK` and every other variable in a suite.

The cost of leaving this unwritten was real. `flicker_evap`'s `QUIET` was a properly promoted,
overridable, well-commented bound, and its two assertions were counted as a COVERAGE GAP on
2026-09-22 — a gap that did not exist. It is now `QUIET_TOL`.

**So: when promoting a bound, put one of the five words in its name.** If a name genuinely reads
better without one (`QUIET` did), rename it anyway and say so in the derivation; the tool's blindness
is worse than the slightly clumsier name.

## 5c-bis. COMPUTED bounds — derived from the run's own config, and better for it

A third thing the framework cannot express, alongside the seven booleans and the two framework-exempt
suites. `ghost_boundary` computes both of its bounds at runtime:

```python
MPI_TOL_FACTOR = 5.0
mpi_tol = MPI_TOL_FACTOR * float(os.environ["WATER_TOL"])   # solver.convergence.water_volume_tol
```
```sh
BUDGET_TOL=$(awk ... config.yaml)      # the run's own water_volume_tol
BUDGET_BOUND=$(awk -v t="$BUDGET_TOL" 'BEGIN{printf "%g", 5*t}')
```

The tool links a bound through a `NAME="${NAME:-value}"` shell default, and these have no fixed
value to put there — which is the point. **The bound tracks the solver tolerance that sets its own
floor.** Change the config's `water_volume_tol` and the assertion follows; a frozen default would
not, and would silently become wrong.

**Do not "fix" these into shell defaults.** It would trade a bound that stays correct for one the
tool can see. What they owe instead is what `MPI_TOL_FACTOR` already provides: the FACTOR derived
and measured in place — *"the six measurements span 0.66x to 2.08x, so 5x clears the worst by
~2.4x… Raise it only with a measurement, never to make a red test green."*

They will keep appearing as `unlinked` rows. That is the correct reading, and the reason the
misconfiguration alarm fires on a MAJORITY rather than on any unlinked row at all.

## 5e. The sweep of 2026-09-21/22 — what was actually measured

Every suite carrying a bound was run twice unchanged (Q6) and once per bound with that bound
tightened (Q5). **32 suites.**

### Q6: spread is ZERO, everywhere

**Not one printed value moved between repeat runs — across every suite, roughly 60 assertions.**
`budget_closure`'s 15, `dt_invariance`'s 14, `serial_recharge`'s 11, and every smaller set.

That settles the inversion rule for this tree, as a measurement rather than an assumption: **headroom
here cannot express flake risk, because nothing flakes.** It measures sensitivity alone, so a low
headroom is a sharp test and a high one is a blunt instrument. Every reading of a headroom number in
this repo rests on that.

### Q5: what the bite check found

| outcome | meaning |
|---|---|
| **BITES** | the bound was tightened below its measured value and the named assertion failed. It is live. |
| **exactly 0** | the quantity is identically zero, so no tighter bound exists — `local_ledger`'s stored-volume drift, `fsm_conservation`'s external input, `dt_invariance`'s span, `serial_recharge`'s split. **Results, not gaps**, and each is backed by a guard that does bite. |
| **INVERTED** | an xfail or NOT ASKED line. Tightening confirms what it already says; its liveness rests on the ratchets beside it. |
| **CANNOT PROBE** | no assertion printed that bound. Every instance found was a missing marker, and every one was fixed. |

**All 14 bite guards are live.** Each was an unreachable literal at the start of this work.

### Two suites are outside the framework entirely, and correctly so

- `golden` — `GOLDEN_STOL` is a SOLVER tolerance injected into each generated config, not an
  assertion bound. It tightens the runs; it does not judge them. Verdicts come from stored
  references via `golden.py`.
- `combination_sweep` — `RAN_FLOOR` is a floor on a COUNT ("only N of 96 cells ran"). The parser
  discards bare integers as values by design, so a count cannot be expressed as an assertion here.

Neither is a gap. They are listed because a silent absence is how a count becomes a lie.

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
