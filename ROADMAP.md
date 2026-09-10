# WTM: the road to a finalized, usable model

The **running list** — the single source of truth for what is open, in what order, and why.
`HANDOFF_READINESS.md` describes the *state* of the handoff; this file describes the *work remaining*.

**Rewritten 2026-09-11 after a staleness sweep over all 25 open items**, four agents verifying each
claim against the tree rather than against task prose. **Twelve closed as already-done or obsolete.**
**Updated later the same day:** `#34` closed, taking `#91` and `#96` with it; `#102` opened by the guard
that closing it produced. Fourteen closed, one new. 13 open → 12.

## The goal

A model someone else can run, trust and reproduce without the person who built it.

## Why this list keeps going stale — read before adding to it

Andy, 2026-09-11: *"I open one, you create a huge task list to bring our code base in line with it…
and in the end, that request is because the code is now far past the numbered to-do item."*

The fixes here have been **mechanism-level**, and each dissolved a whole class of numbered items at
once. An item can be obsolete without its claim ever having been wrong: the mechanism it depended on
stopped existing, or a general guard made its class unreachable.

| general mechanism | class it dissolved |
|---|---|
| declared-config rule + `full_config.yaml` | every "this test does not state X" item |
| `tests/nonvacuous.py` (structure guard) | every "this suite compares identical/empty fields" item (#34, #91, #96) |
| `src/resolve_defaults.cpp` | every per-key "what does an absent key mean" question |
| `tests/emit_config.sh` **deleted** | the shim's entire failure surface (#99) |
| `tests/wtm_volume.py` + fatal `lint_norms.sh` | head-vs-volume items, **and prevents recurrence** |
| `tests/wtm_log.py` (columns by name) | hardcoded-column-index items (#53's class) |
| `make_work`, `expect_resolved` | lost-evidence and did-this-arm-run-what-it-configured items |
| `-wtm_` namespace closed to **0** call sites | every flag-named item |

**So: re-read this list after each general fix, not before each individual one.** Before working an
item, verify its claim against the tree first.

## Closed by the sweep (verified, not assumed)

`#37` 37 of 37 configs state `time_integration` (claimed 8 of 39) · `#48` all four metric items
shipped · `#53` both prescriptions implemented, 456 step-checks · `#97` sweep performed, negative ·
`#99` moot, shim deleted · `#66` the doc it targets **does not exist in the repo** · `#76` every
Phase-A item closed · `#80` all four levels closed · `#39` `xrank_growth` ships and asserts the growth
rate · plus `#42`, `#65`, `#73` closed earlier today · and `#34` / `#91` / `#96` closed by the work in
section 1 below.

## THE OPEN LIST

### 1 — Done since the rewrite

`#34` is **CLOSED** (2026-09-11, `bc67220..de70f4e`), and it took `#91` and `#96` with it.

All three filed suites were repaired and each printed number is now a live measurement. The general
deliverable is `tests/nonvacuous.py`, wired into `lib.sh` at exit 4: a suite fails if its last snapshot
per stem is identically constant. It was **shown to bite on a suite every other check calls healthy** --
restore `recharge_consistency`'s 400-week window and the suite still prints `PASS` and `DECLARED` still
says `OK`, and the guard fails the run anyway.

On its first sweep it found **two more**: `adaptive_water` (3 of 3 zero, fixed) and `storage_equivalence`
(2 of 2 zero, now a guarded xfail — see `#102`).

**Two corrections worth carrying.** The vacuity exemption list I seeded with three suites was wrong in
all three: `direct_to_runoff`, `flicker_evap` and `config_schema` all measured fine, because holding
`wtd` at 0 does not make the output raster constant. It is empty, and the rule is now "add a suite only
after seeing it fail". And two failures that surfaced in `ghost_boundary` looked like model defects and
were **not** — the Jacobian's 0.208 was `active_set`'s semismooth `max()` being verified by a *smooth*
finite-difference reference, and "bit-for-bit MPI" was reduction order tracking the solver tolerance.

**A general mechanism, found on the way:** the `#124` geotransform migration converted **10 of 15**
fixture generators. The rest keep `from_bounds(0, 0, NX, NY, NX, NY)` — the placeholder meaning **one
degree per cell, ~111 km** — which is why those domains saturate. Three of the four unconverted suites
were vacuous because of it; `snapshot_restart` was the only one that was fine. `recharge_consistency`
still carries the placeholder and now measures correctly anyway, so converting it is optional and would
change its physics.

### 2 — Correctness of the model
### 2 — Correctness of the model

| # | item | state |
|---|---|---|
| **102** | `S·Δh ≡ ΔV` is **UNVERIFIED** where `S ≠ Sy` — the suite that asserts it was comparing two zero fields, and the model's own `secant × active_set` refusal message *cites that suite* as its authority. Live: 2.386e-03 m water where cells sit at the surface, not scaling with solver tolerance, not the clamp. | Held as a guarded **xfail**, not a declared defect: the table is in a surface limit cycle there, which is the claim's one documented exception. The fixture cannot meet its own two preconditions under any collector available to `secant`. **Andy's call** between three options in the task. |
| **78** | Budget residual grows as dt shrinks. **CONFIRMED, and worse than filed:** it is *not* backward-euler-specific — at fixed dt tr-bdf2 is worse at every step (3.08e-01 / 4.87e-02 / 5.01e-02). Reaches ~1-30% of recharge. | **UNRESOLVED DISCREPANCY:** my `fsm_consistency` run shows 1.16e-2 → 2.55e-2 growth (reproduced at `runoff_ratio` 0 **and** 0.3); an independent purpose-written config showed 1e-8 with no growth. **The discriminator is unidentified.** Find it before acting. |
| **52** | Land→ocean outflow mis-booked at pinned cells | Reproduces unchanged at **1.802e-05**; encoded as an xfail. Two stale pointers: `CreateSNES.hpp:40` now says the opposite (#40 moved it to `WTM.cpp:715`), and its probe patch no longer applies. |
| **54** | Test and guard the budget at boundaries | Valid, but **item 1 is costed wrong**: `BUDGETTRACE` emits eight *domain scalars*, so this needs model-side machinery, not a mask split. |
| **64** | Sub-cycle the FSM coupling | Untouched; **moves goldens, needs explicit authorization**. Its sub-item — say in `tests/golden/run.sh` that references are regression *pins*, not accuracy statements — is undone. |

### 3 — Provenance: no general mechanism exists yet

| # | item | state |
|---|---|---|
| **84** | 24 tolerances across 19 suites | Count exact. **17 of 24 have no recorded origin; 8 have no comment at all.** |
| **85** | Goldens carry no in-file provenance | `golden.py:54` writes only shape. Loader skips `#` lines, so provenance is backward-compatible. |

### 4 — Documentation that misleads

| # | item |
|---|---|
| **90** | `active_set`'s header still claims collector-independence that was retracted, still lists a deleted assertion as asserted, and calls active_set "EXPERIMENTAL and OFF BY DEFAULT" — it is **the default**. |
| **50** | Newton cold-start claim: 4 sites, not 5. `parameters.cpp:365/372` no longer contain it; it moved to `CreateSNES.cpp:255` and became a *warning*. |
| **58 / 59** | Keep the findings; **strike the "not implemented" sections** — `78d7188` made the blind step visible (measured `est = 4.34e-01` where the tasks record `0.0`), and #63 deliberately forbids the reject-trigger form. |
| **6** | `scheme_bench` re-run: its "active_set × during is a hard error" premise is **false** (that is the default pair); new blocker — `and_be` (secant) now throws under active_set. |
| **98** | `budget_closure` `a_as` ≡ `c_as`, character-identical. Andy's (a)/(b)/(c) call. |
| **77** | Storativity smoothing inert under active_set — a record; no action was claimed. |

## Known repo-hygiene items found by the sweep

- `tests/lib.sh` **contradicts itself**: line 171 says three suites are exempt and names
  `runoff_collector`; line 184 lists two. The variable is right, the prose is stale.
- `tests/lib.sh:155-164` still carries the **pre-#79 ratchet paragraph** ("494 of 494 runs currently
  leave at least one key implicit") directly above the paragraph saying the rule is unconditional.
- `#52`, `#58`, `#59` cite `scratchpad/*.patch` reproduction routes. **That directory is not in the
  repo**, and 3 of 4 patches no longer apply — those routes are unusable by the next session.

## Standing

Nothing has left this machine. Push to `origin` (KCallaghan) is disabled; pushing, tagging and
releasing each need their own explicit, current-message go-ahead.
