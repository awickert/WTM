# WTM handoff readiness

**What this is.** A live checklist of what stands between WTM and being a model someone else can pick
up and use. Not a design note and not a changelog: it answers one question — *if we handed this over
today, where would a new user get stuck?* Each item is either done (with the evidence), in progress, or
open with its blast radius stated.

**Scope: the model code and the surfaces a user touches** — configuration, discoverability, trustworthy
numbers, reproducible documented results. At-scale validation runs are a separate activity and are
deliberately not on this list. They are not a prerequisite for finishing the code, and while they were
listed they kept dragging the plan toward themselves.

Kept in the repo rather than in a session log because the answer outlives any one working session.

---

## The goal, stated plainly

Completing the model and handing it off for use. A model is ready when someone who did not write it
can configure it, run it, trust its numbers, reproduce its documented results, and discover what it can
do without reading the source.

That framing decides priority. Work that removes a way for a new user to be silently wrong outranks
work that makes an existing result slightly better.

---

## 1. Configuration surface — the thing a user touches first

A user configures WTM through a nested-YAML file. Historically a second channel existed: ~65 `-wtm_*`
command-line flags, many duplicating config keys, with no record anywhere of which channel won.

**Done**

- **Unknown YAML keys abort** with the offending key named, its section's valid keys listed, and a
  did-you-mean suggestion. Previously a typo or a retired key was silently ignored and the run
  reported success. (`tests/config_schema`, 6 arms.)
- **Unconsumed `-wtm_` flags abort** and name themselves. A flag nothing read — misspelled, retired,
  or parsed on a code path this run did not take — had no effect and said nothing.
- **17 of 17 "1:1" flags retired.** Each setting is now a `Parameters` member, parsed from the config,
  schema-checked, and read directly by its consumer; flag and bridge entry deleted. See
  `benchmark/CONFIG_FLAG_COVERAGE.md` for the full classification of all 65 flags.
- **`solver.method: newton` works from YAML alone.** It had been a documented config value that
  crashed: Newton does not converge from a cold start without dt-continuation, which the config could
  not express, so `solver.method: newton` aborted with `DIVERGED_LINE_SEARCH`. The config value now
  means the working recipe and is byte-identical to `-wtm_newton -wtm_dt_continuation`.
- **All eight ABSTRACTED flags are verified equal to their config key**, byte-for-byte
  (`tests/route_equality`). That claim -- "the config expresses this, the flag is the primitive" -- had
  never been tested; the suite covered each mechanism but never the equivalence of the two routes to it.
- **The taper-1 band sink is retired, and with it `collection.method: legacy` and the three surface
  flags** (fork issue #7, now closed). Its band width was `2·qmax·dt`, so its equilibrium water table
  moved with the time step. The semismooth `active_set` pin the issue prescribed had already shipped and
  is the default, so the niche was gone. `legacy` collapsed exactly onto `explicit`/`implicit` first
  (max|Δ| = 0.000e+00), which is what made the removal safe.
- **The active-set dual route is closed.** `dev.active_set` is removed. It was a second YAML key for
  the same enforcement, and it silently *overrode* an explicit `collection.method` — measured at 54 of
  256 cells and 0.127 m max, with nothing in the log to say so. An old config carrying the key now
  aborts and names it (`tests/config_schema`, `RETIRED` arm, shown to fail with the key restored).
- **Every run states its surface-water enforcement, with the source named.** It was previously written
  only to the coverage file, so a run's own output could not say which boundary condition produced it.
- **`extended_soil` is a collection *mode*, not a rival switch.** It joined the
  `surface_water.collection.method` enumeration, so "extended soil AND a collector" — a contradiction
  that silently cost a day of debugging — is now unrepresentable rather than merely detected.

**Open**

- **Three gaps block documented workflows** (was four; the Newton one is fixed, see Done). The two
  evaporation taper toggles are configurable in their parameters but not their on/off switch, and
  `-wtm_stiff` is a preset, and presets belong in a config.
- **~24 advanced flags have no config expression at all** — the step-size controller (`dtc_*`,
  `dt_norm_*`) and the Anderson restart/handoff machinery (`ar_*`, `handoff*`, `aa_picard`). These want
  grouping under the method that owns them (`solver.adaptive_dt`, `solver.method: anderson`), not 24
  top-level keys.

## 2. Discoverability — can a user find what exists?

**Partly fixed; the remainder is still open.** The worst of it is closed: `config.yaml` had been
shipping `method: implicit` — the *former* default, and the one enforcement measured to carry a spurious
dt-dependence — while the real default `active_set` appeared neither as the value nor in the enumeration.
A reference config that is present and *wrong* is worse than a missing key, and it now ships `active_set`
with all six modes described.

**Still open:** `config.yaml` is the reference a new user reads, and **16 keys the model accepts do not
appear in it** (17 before `dev.active_set` was removed). Some absences are deliberate (`grid:` is
deprecated and `dev:` is developer-only), but `solver.dt_max`,
`solver.water_volume_timestep_error_tol` and `surface_water.runoff_ratio` are ordinary user settings
that are currently undiscoverable. `tests/config_schema` reports the list on every run.

## 3. Trustworthy numbers

**Done**

- **The water budget closes** across every solver and integrator, cumulatively *and per cycle*
  (`tests/budget_closure`). Per-cycle is the stronger claim: an error that removes water at step *t*
  and returns it at *t+1* cancels in the total while being visibly wrong per step.
- **The adaptive controller's error estimate has an order test** (`tests/estimator_order`). It asserts
  an *order*, not a value — "est is small" is worthless because a constant is small too.
- **Known holes are `xfail`s, not folklore.** The history-based estimator is invalid across the
  FillSpillMerge operator split (observed order 0.00); that is pinned as an expected failure which
  fails loudly if it changes in either direction.
- **Cross-rank consistency** at n = 1,2,4,6,8 on every golden fixture.

**Open**

- **`-wtm_extended_soil` restores GW-step order 2 but its production half was never implemented** — the
  above-surface mound must be truncated at the FSM handoff, and is not. It remains `[WIP]` and
  nonphysical: honour restored, utility not.
- **The default surface-water enforcement has only been validated at small scale.** `active_set` is the
  default `collection.method`, and every result supporting that choice comes from fixtures of **8775
  cells or fewer**. It is the only enforcement whose equilibrium carries no spurious dt-dependence,
  which is why it is the default — but nothing here demonstrates it at production grid sizes. A user
  running a large domain should know that, and should watch the per-cycle convergence metric rather
  than assume the small-domain behaviour carries over. (Establishing this at scale is a validation run,
  deliberately out of scope for this checklist; the caveat is in scope.)
- **With FSM on, first-order Lie splitting caps the whole scheme at order 1** regardless of integrator.
  Measured: TR-BDF2 drops from ~2.0 to 1.00 when FSM is switched on. This bounds every
  dt-refinement accuracy argument anyone will make, and is worth stating wherever accuracy is claimed.

## 4. Reproducible documented results

**Open, and larger than it looks.** ~25 benchmark scripts cannot run at all — they write a legacy flat
`.cfg`, which the model now rejects, and several also depend on a fixture whose geotransform predates
#124. Affected: all of `benchmark/picard/*`, most of `benchmark/esquibel/*.sbatch`, plus `island`,
`speedtest`, `tbar_suite`, `adaptive_dt`.

This matters more than "some scripts are stale": **design notes cite these scripts as the reproduction
path**. `BDF2_RECHARGE_ORDER.md` §15 ends "Reproduce the whole story with
`benchmark/picard/recharge_free_boundary.py`" — a sentence that was false for weeks, and whose failure
mode was three silent breakages deep. That one script is now repaired and reproduces §15 to within
`0.002031 mm` vs the recorded `0.0019 mm`. The rest are not.

A handed-off model whose benchmark suite does not execute is incomplete, and its claims decay into
folklore.

---

## Production configuration, as it stands

`solver.method: anderson` (default) · `time_integration: tr-bdf2` ·
`collection.method: active_set` (default) · `adaptive_dt: true`

This is the combination that is verified green: TR-BDF2 is the only integrator whose error estimator
measures a clean order 2 with FSM on, and active-set is the only surface enforcement whose equilibrium
does not carry a spurious dt-dependence.

---

## The open queue, in implementation order (2026-09-09)

One ordered list rather than a grouped one and an ordered one — two views of the same queue is how
the drift this file just spent a day correcting begins. The **group** column keeps the kind visible:
**C**onfig arc · **T**est integrity · **M**odel defect · **P**hysics/measurement · **D**epression
hierarchy.

**Three constraints fix the sequence.** Schema-movers must precede configs being written to files.
Answer-movers must precede any number being pinned. Units must precede tolerances. Everything else
is movable, and items marked *(parallel-safe)* have no dependants at all.

### First — the three decisions, because they block the largest item

| n | # | grp | item | why here |
|---|---|---|---|---|
| 1 | 38 | C | `solver.time_step.mode: fixed \| adaptive \| ramp` replaces `adaptive_dt` + `newton.dt_continuation` | **Last schema-mover.** 18 callers now; 18 callers *plus ~37 written configs* if it lands after #83 |
| 2 | 86 | C | Decide CLI-vs-config precedence: announce, refuse, or document | Touches the bridge #83 must describe; announcing is one edit |
| 3 | 36 | C | Step 10 only: reconcile the code defaults to what `config.yaml` ships — re-check first, it names a key that no longer exists | Changes what *omission* resolves to, so it must settle before resolutions are written down |

### Then — cheap cleanups that reduce noise before the big migration

| n | # | grp | item | why here |
|---|---|---|---|---|
| 4 | 57 | M | Catch `std::runtime_error` at top level: a refusal should print and exit, not core-dump | Self-contained, user-facing, no interactions. A handoff wart |
| 5 | 46 | M | Coupling vocabulary — `-wtm_fsm_continuous` (4 sites) and "overwrite" in prose | Includes the `WTM.cpp` comment still saying the default is `source` |
| 6 | 44 | M | Re-verify `fsm_coupling: continuous` × `collection.method: explicit` under the current name | May close for free — #40, #47 and #56 all moved underneath it |
| 7 | 35 | M | Reproduce: does the collector selector drop `dev.allow_aboveground_water_columns`? | Verified still present in code; one two-run diff settles it |

### Then — the config arc, which is the only live level of the stack

| n | # | grp | item | why here |
|---|---|---|---|---|
| 8 | 83 | C | Materialize the configs; the shim becomes a one-time generator, then is deleted | Needs the schema still, which items 1–3 deliver. **Subsumes #37** |
| 9 | 79 | C | Phase 5 — enforce the declared rule for every suite, delete `WTM_DECLARED_SUITES` | Enforcement freezes what the configs say, so it follows the authoring |
| 10 | 80 | C | Close the stack: L3 done means all four levels are done | Bookkeeping, but it is the thread's end |

### Then — answer-movers, before any number is blessed

| n | # | grp | item | why here |
|---|---|---|---|---|
| 11 | 60 | P | Order-aware retry — patch is written and unapplied; measure it | Changes step sizes, so it moves trajectories. Cheapest answer-mover |
| 12 | 64 | P | Sub-cycle the FSM coupling to bound the delta admitted per step | **Will move the goldens.** Biggest item on the list; a negative result is a valid outcome |
| 13 | 54 | T | Partition every budget assertion boundary-ring vs interior | The instrument for #52. Folds in **#34** (a vacuous boundary assertion) |
| 14 | 52 | M | Land→ocean outflow mis-booked at pinned boundary cells | Item 13 is what catches it; the water table is unaffected, so no goldens move |
| 15 | 48 | T | Fix the metric: normalise spreads by cumulative recharge, assert convergence under `continuous` | The defect is the test's metric, not the model |
| 16 | 42 | P | dt-scaling of the continuous coupling's budget lag | **Check #48 first** — it may already have answered this |
| 17 | 39 | T | Assert cross-rank difference does not *grow*, rather than pinning a threshold | The growth rate is the durable handle |

### Then — pin the numbers, in this order

| n | # | grp | item | why here |
|---|---|---|---|---|
| 18 | 65 | T | Convert the last 5 head-unit measurements to water volume | **Units before tolerances.** Reversing these two derives all 24 tolerances twice |
| 19 | 84 | T | Give all 24 assertion tolerances a measured basis | Only meaningful once units and answers have settled |
| 20 | 85 | T | Self-describing golden headers — commit, config, date, reason | Do it *with* any regold item 12 forces |
| 21 | 53 | T | Restrict state-vs-accumulator to two-level schemes, and say why | Small, and it prevents a future false alarm on `bdf2` |

### Then — measurement infrastructure and the remaining measurements

| n | # | grp | item | why here |
|---|---|---|---|---|
| 22 | 73 | T | Rewrite `analyze_adapt_bench.py`; add `compare_series` | The surviving children of the #66 audit |
| 23 | 66 | T | Promote the six themes and the "what already works" patterns into a durable doc, then close | ~9 of 13 ranked fixes are already done |
| 24 | 50 | P | Does Newton still need the ramp for cold starts from far? | Five documentation sites are waiting on the answer |
| 25 | 6 | P | Re-run `scheme_bench` now that Newton carries the active-set tangent | *(parallel-safe)* |
| 26 | 58, 59 | P | The blind-step reject trigger: measure whether it improves the N-sweep or grinds | Needs items 11–12 settled first |
| 27 | 77, 78 | P | Recorded measurements: smoothing inert under active_set; the unverified backward-euler residual | *(parallel-safe)* — #78 needs a purpose-built config, not a sed of another |

### Independent of everything above

| n | # | grp | item |
|---|---|---|---|
| 28 | 82 | D | Vendored DH: content-diff the 11 richdem commits; reframe the integration plan *(parallel-safe)* |

### Closed 2026-09-09

**#32** flag walk-through, 8 of 8 — kirchhoff retired, five groups already done.
**#30** flag retirement, 14 of 14 — the nine "abstracted" flags have zero source call sites.
**#81** DH cell-area bug — fixed and suite-confirmed; the diff to port lives in #82.

### What the sweep found, and the habit it argues for

Six task bodies described work that was already finished, and every `transient_groundwater.cpp` line
citation in the list was wrong by ~64 lines after the kirchhoff removal. In every case the
*analysis* held — the measurements, mechanisms and refutations were all still good — and only the
**status** and the **addresses** had decayed. So: verify status in the same turn you act on it, and
cite by symbol rather than line number.

### Standing

Nothing has left this machine: **249 commits unpushed** to the `awickert` fork. Push to `origin`
(KCallaghan) is disabled. Pushing, tagging and releasing each need their own explicit go-ahead.
