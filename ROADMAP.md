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

Andy, 2026-09-11: *"go through your numbered steps. All those that have to do with issues around the
tests rather than the code itself: set aside. I am interested only in improving the code."* The list is
split on that line. Section A is work on the model; section B is real but is about the harness, and is
parked until the model work is done.

### A — THE CODE

| # | item | state |
|---|---|---|
| **104** | The per-solve water-step test declared convergence after 4-7 iterations on the shipped `active_set` path, committing a first step tens of metres from the answer | **FIXED** (`db54072`, `2cc272a`, `5101991`). The verdict is now judged against what the run has DEMONSTRATED it can reach, not a fixed reduction. 10 of 20 sweep arms disagreeing → 0. Three conditions, each proven load-bearing by ablation. Pinned by `tests/tolerance_independence`. |
| **103** | `explicit` sustains a **permanent** surface limit cycle (no decay over 460 yr, 56 of 88 cells) and the equilibrium stop declares convergence *inside* it — `stopping at cycle 4 of 30` with 0.0606 m of within-cycle motion | `active_set` cures it completely (4.6e-08 vs 0.0912 m). Three options; needs Andy. |
| **102** | `S·Δh ≡ ΔV` unverified where `S ≠ Sy`, and the model's own `secant × active_set` refusal cites the suite that never checked it | Blocked by #103 — `explicit` is the only surface-reaching collector `secant` may use, and it flickers. |
| **64** | Sub-cycle the FSM coupling | Untouched. **Moves goldens → needs authorization.** |
| **54** | Test and guard the budget at boundaries | Needs model-side machinery: `BUDGETTRACE` emits domain scalars, so a mask split is not enough. |
| **50** | Re-measure whether Newton still needs `dt_continuation` for cold starts | A measurement about the model; the claim survives at 4 sites, not 5. |
| **60** | Order-aware retry for the adaptive controller | Parked by Andy: needs a case that would otherwise abort. |
| **6** | Re-run `scheme_bench` | Blocked by a live model refusal (`and_be` secant throws under `active_set`), which is itself #102's territory. |

### B — THE HARNESS, set aside

Real, and none of it changes an answer. `#84` tolerance provenance (now partly mechanised by
`tests/tol_margin.py`) · `#85` goldens carry no in-file provenance · `#98` `budget_closure`'s `a_as` ≡
`c_as` · `#90` `active_set`'s header still calls it experimental and off-by-default · `#50`'s doc half ·
`#58 / 59` strike the "not implemented" sections · `#77` a record, no action claimed.

### Closed today beyond the sweep

`#34` (with `#91`, `#96`) · `#78` → root-caused into `#104` · **`#52`** — both encoded reproductions now
close, at 319x and ~5800x margin, and are promoted to plain checks.

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
