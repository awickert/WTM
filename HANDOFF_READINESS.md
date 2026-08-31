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
- **`extended_soil` is a collection *mode*, not a rival switch.** It joined the
  `surface_water.collection.method` enumeration, so "extended soil AND a collector" — a contradiction
  that silently cost a day of debugging — is now unrepresentable rather than merely detected.

**Open**

- **`-wtm_active_set` is reachable from two YAML keys** (`dev.active_set` and
  `collection.method: active_set`). The same dual-channel hazard, entirely inside the config. Resolve
  to one.
- **Four gaps block documented workflows.** `-wtm_dt_continuation` has no config route, and Newton does
  not converge without it — so `solver.method: newton` is *unusable from YAML alone*. The two
  evaporation taper toggles are configurable in their parameters but not their on/off switch.
  `-wtm_stiff` is a preset, and presets belong in a config.
- **~24 advanced flags have no config expression at all** — the step-size controller (`dtc_*`,
  `dt_norm_*`) and the Anderson restart/handoff machinery (`ar_*`, `handoff*`, `aa_picard`). These want
  grouping under the method that owns them (`solver.adaptive_dt`, `solver.method: anderson`), not 24
  top-level keys.

## 2. Discoverability — can a user find what exists?

**Open, and this is a defect rather than a matter of taste.** `config.yaml` is the reference a new user
reads, and **17 keys the model accepts do not appear in it**. Some absences are deliberate (`grid:` is
deprecated, `dev:` is developer-only, `collection.sink` is legacy), but `solver.dt_max`,
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
