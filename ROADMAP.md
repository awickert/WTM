# WTM: the road to a finalized, usable model

The **running list**. This file is the single source of truth for what is open, in what order, and why
each item earns its place. `HANDOFF_READINESS.md` describes the *state* of the handoff; this file
describes the *work remaining*. Update it as items close — do not keep a second copy elsewhere.

**Last updated: 2026-09-10.**

## The goal

A model someone else can run, trust and reproduce without the person who built it. That splits into
three things, and most of the list below serves the third:

1. **It computes the right answer** — mass conserved, boundaries correct, defaults defensible.
2. **It can be driven** — one configuration surface, YAML, defaults resolved by the model and recorded.
3. **Its numbers can be re-derived** — every tolerance, golden and benchmark traceable to how it was
   measured, by someone who was not there.

## Closed: the configuration arc

The whole L1–L4 stack of `#80` is closed. Configuration is YAML, defaults resolve in one dedicated
module (`src/resolve_defaults.cpp`), every run records what it resolved (`full_config.yaml`), and every
one of the 39 test suites reads a real config file a human wrote.

| # | what closed | evidence |
|---|---|---|
| 32, 30, 86 | The `-wtm_` namespace: 23 round-trips to zero, locked at runtime and in source | `config_schema` NAMESPACE arm |
| 38, 89 | One key sizes the step (`solver.time_step.mode`); routing merged into one key | schema is still |
| 45181ae | Geometry has ONE source: the input geotransform. The `grid:` block is gone | L4 |
| 83 | **All 39 suites materialized; `tests/emit_config.sh` DELETED** | 2c5330b |
| 79 | **The declared-config rule is unconditional**; `WTM_DECLARED_SUITES` gone; 37 of 39 enforced | 45a2820 |
| 92 | A parameter may be marked `OPTIONAL` where automatic resolution IS the subject | bcdb1c7 |

## Andy's stated next priorities

**#42, then #65** (recorded 2026-09-11). Take these before resuming the numbered order below.

## The open queue, in implementation order

**Three constraints fix the sequence.** Answer-movers before any number is pinned. Units before
tolerances. Measurement instruments before the defects they catch.

Group: **M**odel defect · **T**est integrity · **P**hysics/measurement · **D**ocumentation

| n | # | grp | step | how it serves the goal |
|---|---|---|---|---|
| - | 42 | D | **NEXT (Andy)** Stale vocabulary: `-wtm_fsm_delta_source`, "overwrite" | **Legibility.** Docs naming keys that no longer exist send a new user down dead ends |
| - | 65 | T | **THEN (Andy)** Convert the remaining head-unit measurements to water volume | **One currency.** The model judges every criterion in water; a test measuring head can pass while water is wrong |
| 1 | 60 | P | Order-aware retry — real speedup, exonerated; the blocker was #61 | **Speed.** Same hardware, faster run. Gain already measured |
| 2 | 64 | M | Sub-cycle the FSM coupling to bound the delta admitted per step | **Correctness at production settings.** Bounds a per-step coupling error that has no limit today. **Moves goldens — needs explicit authorization** |
| 3 | 54 | T | Test and guard the BUDGET at boundaries. Folds in #34 | **The largest correctness gap.** Solution-correctness at boundaries is covered; MASS-correctness is not. A water model can lose water where nobody looks |
| 4 | 52 | M | Land→ocean outflow mis-booked at pinned cells (211 of the outer ring, corners 1.5x) | **The defect #54 fences.** Localised already. Water table unaffected, so no goldens move |
| 5 | 48 | T | `continuous` is first-order in dt — fix the TEST's metric | **Trust in the shipped default.** Its convergence test measures the wrong thing, so the default is unverified |
| 7 | 39 | T | impulse resets cross-rank drift; continuous compounds | **Parallel trustworthiness.** Instrument built and green (`xrank_growth`); the confirmatory experiment remains. EVERY cross-rank tolerance was calibrated under `impulse`, i.e. on a resynchronised system |
| 9 | 84 | T | 24 laundered assertion tolerances across 19 suites | **Provenance.** A tolerance that says what it IS but not where it CAME FROM cannot be re-derived by a new maintainer |
| 10 | 85 | T | Golden references carry no in-file provenance | **Reproducibility.** Do it *with* any regold #64 forces |
| 11 | 53 | T | State-vs-accumulator is valid only for 3-level schemes | **Test validity.** Prevents a future false alarm on `bdf2` |
| 12 | 73 | T | Fix the measurement scripts: `analyze_adapt_bench.py`, `compare_series()`, norm lint | **The tools that produce the numbers** |
| 13 | 66 | D | Stale count in the ranked-fixes doc (~9 of 13 done, not 2 of 15) | **Accurate handoff state.** A doc overstating open work misdirects whoever picks this up |
| 14 | 50 | P | Does Newton still need the ramp for cold starts from far? | **Solver guidance.** Five documentation sites wait on the answer |
| 15 | 6 | P | Re-run `scheme_bench` | **The performance table users read.** Unblocked 2026-09-10: the script runs again |
| 16 | 58, 59 | P | The non-monotone bump: rate-vs-exposure; FSM activity does not predict it | **Controller understanding.** #59 is a measured NEGATIVE, kept so it is not re-walked |
| 17 | 77, 78 | P | Storativity smoothing inert under active_set; BE residual growth at fine dt | Recorded observations; #78 needs a purpose-built config before it can be believed |
| 18 | 37 | T | Integrator coverage — which suites DECLARE which integrator | Folded into #83's authoring pass; re-check and close |

## Filed later, not part of the agreed order

| # | item | note |
|---|---|---|
| 96 | ghost_boundary compares schemes at a converged steady state, where the integrator cannot matter | The `cc` arm's identity is an open DECISION, deliberately not taken |
| 97 | Tests that grep the model's PROSE go stale silently | One instance fixed in budget_closure; the sweep remains |
| 90 | active_set's three arms differ in coupling as well as collector | |
| 91 | recharge_consistency compares integrators that produce bit-identical fields | Same shape as #96 |
| 34 | boundary_analytic's Neumann vertex assertion is defeated by nan | Folds into #54 |
| 98 | budget_closure's `a_as` and `c_as` are the same run at the same tolerance | Low stakes |

## Audit items — changes made to TESTS that alter what they run

Neither touches `src/` or the shipped `config.yaml`; the model and its defaults are unchanged. Both are
one-line reverts if unwanted.

| where | change | basis |
|---|---|---|
| `tests/taper` config | Study A `equilibrium_stop.tol` 0.001 → **0** | The arm's own comment documented `eq_tol 0`, lost when the flag was retired. Study A now runs 5 cycles, not 4 |
| `benchmark/picard/recharge_free_boundary.py` | collector now **pinned** to `active_set` in the signature | Same value the model resolves today, but it no longer TRACKS the default if that default moves |

## Standing

Nothing has left this machine. Push to `origin` (KCallaghan) is disabled; pushing, tagging and
releasing each need their own explicit, current-message go-ahead.
