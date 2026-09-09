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

## The open queue, in full (updated 2026-09-09, after a staleness sweep)

Every open item with its task number, grouped by kind. Ordering *within* a group is not a ranking;
the ordering *between* the two constraints below is real.

**Two constraints decide sequence.** Work that changes the **schema** must precede work that pins
configs into files. Work that changes **answers** must precede work that pins numbers. Everything
else can move.

### A. The configuration arc — the only live level of the stack

| # | item |
|---|---|
| 38 | **Schema-mover, do first.** One key `solver.time_step.mode: fixed \| adaptive \| ramp`, replacing `solver.adaptive_dt` and `solver.newton.dt_continuation`. ~18 callers. Needs Andy: it changes a user-facing key |
| 83 | Materialize the configs: the shim becomes a one-time generator, then is deleted |
| 79 | Declared-config rule – authoring pass, then Phase 5 enforces for every suite |
| 86 | A `-wtm_` transport flag silently overrides its config key. Needs Andy: announce, refuse, or document |
| 36 | Sketch D – only step 10 remains, and it is part-stale (names a key that no longer exists) |
| 80 | The stack – L1, L2 and L4 are closed; L3 is what is left |

### B. Test integrity

| # | item |
|---|---|
| 84 | 24 assertion tolerances state what they *are*, not where they *came from* |
| 85 | Golden references record shape, not origin |
| 34 | `boundary_analytic`'s Neumann vertex assertion is defeated by `nan` |
| 65 | Convert head-unit measurements to water volume – 5 suites left of ~15 |
| 73 | `analyze_adapt_bench.py` and `compare_series` – the surviving children of the #66 audit |
| 37 | Integrator coverage is now whatever suites declare – 8 of 39 state it. Folds into #83 |
| 53 | State-vs-accumulator is invalid only for 3-level schemes |
| 66 | Keep the six themes and the "what already works" patterns; ~9 of 13 ranked fixes are done |

### C. Open model defects

| # | item |
|---|---|
| 52 | Land→ocean outflow at pinned boundary cells – 211 cells of the outer ring, corners 1.5× |
| 57 | A deliberate config refusal exits via uncaught exception and a core dump |
| 35 | The collector selector reassigns the same global the `dev` key sets – verified still present |
| 44 | Re-verify under the current name: `fsm_coupling: continuous` × `collection.method: explicit` |
| 48 | Not a blocker – `continuous` is already the default. The budget is first-order in dt, and the *test's* metric is the defect |
| 39 | Cross-rank drift: impulse resets it each step from rank 0, continuous accumulates it |
| 46 | Coupling vocabulary – `-wtm_fsm_continuous` (4 sites) and "overwrite" in prose |

### D. Measurement, physics, performance

| # | item |
|---|---|
| 64 | **Answer-mover.** Sub-cycle the FSM coupling to bound the delta admitted per step |
| 60 | **Answer-mover.** Order-aware retry – real speedup, exonerated; patch written and unapplied |
| 42 | dt-scaling of the continuous coupling's in-flight budget lag – check #48 first, it may already answer it |
| 50 | Whether Newton still needs the ramp for cold starts from far – five doc sites depend on the answer |
| 54 | Test and guard the budget at boundaries – the gap #52 fell into |
| 77 | Storativity smoothing is inert under active_set – a recorded measurement, not an action |
| 78 | Backward-euler residual at fine dt – unverified, needs a purpose-built reproduction |
| 6 | Re-run `scheme_bench` now that Newton carries the active-set tangent |
| 58, 59 | The estimator bump: solved as rate×exposure, plus the measured negative |

### E. Depression hierarchy

| # | item |
|---|---|
| 82 | Vendored DH: content-diff the 11 richdem commits; reframe the integration plan |

### Blocked on a decision rather than on work

**#38** (a user-facing key changes), **#86** (precedence: announce, refuse, or document), and **#36**
step 10 (three code defaults, one of which no longer exists).

### Closed 2026-09-09

**#32** flag walk-through, 8 of 8 – kirchhoff retired, five groups already done.
**#30** flag retirement, 14 of 14 – the nine "abstracted" flags have zero source call sites.
**#81** DH cell-area bug – fixed and suite-confirmed; the diff to port lives in #82.

### What the sweep found, and the habit it argues for

Six task bodies described work that was already finished, and every `transient_groundwater.cpp` line
citation in the list was wrong by ~64 lines after the kirchhoff removal. In every case the
*analysis* held – the measurements, mechanisms and refutations were all still good – and only the
**status** and the **addresses** had decayed. So: verify status in the same turn you act on it, and
cite by symbol rather than line number.

### Standing

Nothing has left this machine: **249 commits unpushed** to the `awickert` fork. Push to `origin`
(KCallaghan) is disabled. Pushing, tagging and releasing each need their own explicit go-ahead.
