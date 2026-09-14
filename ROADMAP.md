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
| **60** | Order-aware retry for the adaptive controller | Parked by Andy: needs a case that would otherwise abort. The v2 rework is **in `git stash stash@{0}`**, not in the tree — grepping `src/` for it finds nothing and reads as lost work. |
| **6** | Re-run `scheme_bench` | Blocked by a live model refusal (`and_be` secant throws under `active_set`), which is itself #102's territory. |

### Measured and NOT shipped: option A, a continuous PI law on the coupling error

The adaptive controller reads `est_int` only. The FSM coupling error reaches it through ONE line —
`if (est_cpl > dt_tol) factor = min(factor, 1.0)` — a binary gate in a controller that is otherwise
fully continuous, and the conservative residue of `#63` removing `est_cpl` from the reject test.

That gate looked worth replacing, because on `fsm_runoff_hi` `est_int` sits at **1.3e-06 .. 9.7e-05**
against `dt_tol` 0.5 — four to five orders BELOW tolerance — so the PI law saturates `dtc_grow` every
step and the factor actually taken is only ever **1.0 or 1.5, nothing between**. Step size was being
decided entirely by the gate, while `est_cpl` (0 .. 2.82, up to 5.6× tolerance) carried all the dynamic
range and was allowed to say only "hold".

Replacing it works *mechanically*: factors become continuous (0.808, 0.536, 1.260, 1.002), `dt` responds
to the coupling error and recovers when it falls, and it never pins at the floor. **But it does not fix
the accuracy problem**, measured against `#64`'s own dt-refined ground truth (fixed `dt` = 1/1000 yr):

| fixture | arm | max \|dV\| (m) | rms (m) | cells>1cm |
|---|---|---|---|---|
| `transient` | before (binary cap) | 2.9632e+00 | 2.1729e-01 | 90 of 256 |
| `transient` | after (continuous PI) | 2.9630e+00 | **2.1740e-01** | 90 of 256 |
| `fsm_runoff_hi` | before | 4.0063e-01 | 3.5011e-02 | 4 of 256 |
| `fsm_runoff_hi` | after | 3.9361e-01 | 3.3938e-02 | 4 of 256 |

0.007% better in max and 0.05% WORSE in rms on the widespread case; 1.8% / 3.1% better on the localised
one. And it **breaks 10 golden checks**, so shipping it costs a full re-gold for no measured gain.

Consistent with `#64`'s own diagnosis, and that is the lesson: the splitting error is delivered O(1) per
coupling event and comes down only under **uniformly** smaller steps. A shrinks `dt` when `est_cpl` is
large *at that moment*; it never makes the whole ladder finer, so it cannot reach the regime that
converges.

NOT SHIPPED. The patch is in `git stash` — "#64 A: continuous PI on est_cpl". Do not re-derive it.

### B — THE HARNESS, set aside

Real, and none of it changes an answer. `#84` tolerance provenance (now partly mechanised by
`tests/tol_margin.py`) · `#85` goldens carry no in-file provenance · `#98` `budget_closure`'s `a_as` ≡
`c_as` · `#90` `active_set`'s header still calls it experimental and off-by-default · `#50`'s doc half ·
`#58 / 59` strike the "not implemented" sections · `#77` a record, no action claimed.

### Closed today beyond the sweep

`#34` (with `#91`, `#96`) · `#78` → root-caused into `#104` · **`#52`** — both encoded reproductions now
close, at 319x and ~5800x margin, and are promoted to plain checks. · **`#50`** — measured, and the
answer is that the ramp IS still needed: plain Newton fails at 5 of 6 `dt`, and the single success at
1.0 wk is the same single-`dt` trap. The claim in the docs stands, unchanged.

**`#105` — the default land boundary leaked mass.** Under `boundaries.land: neumann_toposlope` the
ghost head is `h_edge + (topo_edge − topo_inland)`, which is zero flux relative to the *land surface*,
not zero Darcy flux; wherever terrain rises away from the edge it drives water **in**. The solve used
that flux and the budget ignored it. Unaccounted inflow ran to **44.5× recharge**; the exact residual
went `4.4521e+01` → `8.3964e-10`, with `dirichlet_sea_level` and flat terrain both bit-unchanged. It
hid for the suite's whole life because **every other fixture is ocean-ringed**, so the term was a
structural zero and every budget check was true and empty. Now reported as run-log column 26,
`boundary_inflow_gw`, and pinned by `tests/ghost_boundary` — which also asserts the term is nonzero, so
the check cannot go quietly vacuous the way it did before.

**`#54` — CLOSED, 3 of 5 items done, 2 declined with reasons.** Done: the boundary *budget* is asserted,
not just the boundary solution (item 3); the collector × boundary matrix was swept and is clean for
everything that runs, with one standing arm kept for the constraint-on-a-boundary-face crossing that
both `#52` and `#105` needed (item 4); and the off-map flux is now recomputed by a second independent
route — the pre-fix budget *gap* and the term the model *books* agree to five significant figures,
`4.4521e+01` both ways (item 5).

Declined, with the reasoning recorded so it is not re-opened blind:

- *Item 1's remainder, a region-partitioned budget.* It would evaluate
  `storage_change = recharge + boundary_inflow − ocean_outflow − surface_removed − evap` over a subset
  of cells rather than the whole domain. **It does not diagnose boundary conditions.** The budget asks
  whether all the water was accounted for, not whether the right water was moved: on one fixture,
  `neumann_toposlope` and `dirichlet_sea_level` differ by **93.2 m** at 460 of 480 cells and *both*
  close, at 8.4e-10 and 5.6e-09. A wrong ghost formula keeps the solve and the books in agreement and
  the budget stays silent. BC correctness is guarded instead by the Jacobian-vs-finite-difference check
  on the off-map tangent, serial/MPI agreement, and the analytic expectation. For mass-accounting
  defects the coarse discrimination already exists in column 26 — a failure with a large
  `boundary_inflow_gw` points at the edge, one with `boundary_inflow_gw = 0` points at the interior.
  What a per-cell map adds is attribution of the book-vs-solve mismatch, which is a debugging
  convenience, not a guard, and the guard is already at 1e-10.

- *Item 2, a perimeter-scaling property test.* Needs a new fixture whose land-edge extent varies with
  local conditions held fixed. It would measure the missed-edge and double-counted-corner cases; both
  were read in source and look right, but reading is not measuring. Declined as speculative against its
  fixture cost.

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
