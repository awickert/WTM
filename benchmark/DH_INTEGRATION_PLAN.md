# Depression Hierarchy integration plan

Swap WTM's vendored **serial** depression-hierarchy builder for the local **parallel**
Depression Hierarchy (Barnes 2019), so the two forks converge on one source of truth and WTM can
scale to the 30″ global grid. This is the WTM-side companion to the DH repo's
`archive/WTM_API_REVIEW.md`, `archive/PARALLEL_DEPHIER_DESIGN.md`, and `DH_API_NAMING_REVIEW.md`.

**Status:** planning. No code changed yet. Prerequisite (below) must pass first.

## Scope

The swap is the **depression-hierarchy builder only** (`GetDepressionHierarchy`). WTM's
**FillSpillMerge stays in WTM** (`src/fill_spill_merge.hpp`, ~1773 lines) — the local DH repo does
not provide FSM. FSM runs rank-0-only and is *not* being parallelized (see
`benchmark/FSM_SERIAL_DESIGN.md`); this work does not change that.

## Current state — the API gap

WTM calls (vendored `src/dephier.hpp`, namespace `richdem::dephier`):
- `GetDepressionHierarchy<float, Topology::D8>(topo, cell_area, label, final_label, flowdirs)`
  → `DepressionHierarchy<float>` (= `std::vector<Depression<float>>`)   [WTM.cpp:249, 483]
- `FillSpillMerge(params, deps, arp)` → void, mutates `deps` + `arp.wtd` in place   [WTM.cpp:379]
- `deps` is rebuilt **every cycle** in transient (topography changes), **once** in equilibrium.

Local DH (`~/dataanalysis/Barnes2019-DepressionHierarchy`, `include/dephier/dephier.hpp`, same
`richdem::dephier` namespace) vs. what WTM/FSM need:

| # | Gap | Local DH today | WTM / FSM needs | Rec (WTM_API_REVIEW) |
|---|-----|----------------|-----------------|----------------------|
| 1 | Builder signature | `(dem, label, flowdirs)` | `(dem, cell_area, label, final_label, flowdirs)` | superset wrapper |
| 2 | Label type | `dh_label_t = uint32_t` (+ latent `-3` assign, ~line 946) | `int32_t` | revert to `int32_t` |
| 3 | Volumes | cell-weighted | area-weighted (WTM ENH-4) | land ENH-4 |
| 4 | `final_label` | not produced | produced (FSM reads it) | add to output |
| 5 | `Depression` fields | has `total_elevation` | needs `wtd_vol, wtd_only, my_cells, dep_area` | make `Depression` a superset |

The local DH already carries the naming cleanup (`DH_API_NAMING_REVIEW`: done) and the determinism
fixes (`RICHARD_REVIEW_NOTES`: tiled build bit-identical to serial on all 107 fixtures) — so the
serial entry point is deterministic and split-invariant. Only gaps 1–5 remain.

## Architecture decision

**Chosen: single source of truth.** Apply recs 1–5 in the DH repo so its serial `GetDepressionHierarchy`
is a superset drop-in; WTM then *deletes* `src/dephier.hpp` and includes the DH header directly.
Rejected alternative: a thin adapter kept inside WTM (leaves two diverging forks). Single-source is the
`WTM_API_REVIEW` recommendation and avoids perpetual re-sync. (Decision revisitable if the DH repo turns
out not to be the canonical copy to modify — confirm before Phase 1.)

## Plan

### Phase 0 — Prerequisite: current WTM runs standalone (baseline)
Confirm the current *self-contained* (vendored-dephier) WTM builds and runs a full simulation to
completion, and **capture a reference output** (the `island_equilibrium` Corsica run: real DEM,
exercises GW solve + dephier + FSM + GeoTIFF output). This is the before/after baseline for the swap.
Must pass before Phase 1.

### Phase 1 — Make the DH repo a superset drop-in (recs 1–5), tested in-repo
In `~/dataanalysis/Barnes2019-DepressionHierarchy` (on a branch):
- Rec 2: `dh_label_t` `uint32_t` → `int32_t`; fix the latent `-3` assignment.
- Rec 1/4: superset `GetDepressionHierarchy` signature — accept `cell_area`, emit `final_label`.
- Rec 3: port WTM's area-weighted marginal volumes (ENH-4) into `CalculateMarginalVolumes`.
- Rec 5: add the FSM fields (`wtd_vol, wtd_only, my_cells, dep_area`) to `Depression` (superset; FSM
  populates them, the builder just declares/zeroes them).
- Validate with the DH repo's own serial≡tiled oracle (the 107-fixture correctness suite).

### Phase 2 — Swap in WTM
- Wire WTM to the DH header (submodule or include path; both already share `richdem` at the awickert
  fork, so versions align). Delete `src/dephier.hpp`. Keep `src/fill_spill_merge.hpp`.
- Confirm FSM compiles against the superset `Depression`; build WTM.

### Phase 3 — Validate WTM
- Full suite (`tests/run_all.sh`) green — especially golden, mpi_consistency, FSM tests.
- Diff WTM output against the Phase-0 baseline. See risk below.
- `island_equilibrium` Corsica serial≡parallel example.

## Risks

- **Golden shift from determinism fixes (most likely).** The local DH's deterministic tie-breaks may
  build a *different but more-deterministic* hierarchy than the vendored one on ties → WTM output can
  move. Not a bug. Mitigation: diff Phase-0 baseline vs post-swap; trace every change to a documented
  determinism fix before regenerating goldens; regenerate only with sign-off.
- **`Depression` superset correctness.** The FSM fields must be added *and* zeroed/managed so FSM's
  existing reads/writes behave identically. Guard: the FSM tests (`test_fill_spill_merge`) plus the
  output diff.
- **Cross-repo build wiring.** WTM and the DH both pull `richdem` (awickert submodule); confirm one
  consistent richdem across both to avoid ODR/type mismatches.

## Validation summary
Phase 0 baseline (Corsica output) + DH-repo 107-fixture oracle + WTM full suite + before/after output
diff. Serial must stay bit-identical to the vendored path except where a determinism fix explains the
change.


## Which upstream owns this code today (settled 2026-09-10, #82b/c)

**WTM's `src/dephier.hpp` is a VENDORED COPY**, not the `common/richdem` submodule. Its ancestor is
richdem's `include/richdem/depressions/depression_hierarchy.hpp`, fork point `0ffc4c2` (2022-05-14),
established by normalised-diff bisect (see the provenance header in the file itself).

**A correction worth recording, because it is an easy wrong turn.** The file is ABSENT at the richdem
commit WTM pins (`4308687`) -- but it was never removed upstream. `4308687` is on awickert's
`submodule-gdal-pin` line, which is NOT a descendant of the fork point; `origin/master` still carries
the file. "Absent at the pinned HEAD" is not "deleted upstream", and reading it that way would have
retired a live upstream.

### The ten upstream commits since the fork point, checked BY CONTENT

Commit counts are an upper bound on divergence, not a defect list. Each commit's newly-introduced
identifiers were checked for presence in WTM's copy:

| commit | subject | new identifiers present in WTM |
|---|---|---|
| `91ded4a` | Fix logging | 100% of 24 |
| `5520625` | Add more fine-grained timing info to DH | 100% of 6 |
| `e821e7a` | Remove extraneous newlines | 100% of 6 |
| `67230b3` | Add some brackets | 100% of 3 |
| `12e6f1b` | Use pair as the basis for OutletLink | 88% of 9 |
| `c98ac96` | Rename outlet hashing class for clarity | 81% of 11 |
| `5ac329e` | BUGFIX: Stop MV search at ocean link | 72% of 29 -- **already present**, reached independently (the `ocean_parent -> clabel = OCEAN` early exit) |
| `2bfde4a` | Simplify `Array2D::iToxy()` | touches Array2D, not DH content |
| **`c5a183a`** | **Adds CachingOutletChecker** | **57% of 73 -- GENUINELY ABSENT** |
| **`4b491a1`** | **Start of serialization** | **66% of 33 -- GENUINELY ABSENT** |

Verified directly: `CachingOutletChecker`, `serialize` and `cereal` appear **zero** times in
`src/dephier.hpp`; `OutletLink` (8) and `ocean_parent` (6) do appear.

### So only two upstream changes are actually missing, and neither is a defect

**`CachingOutletChecker` (`c5a183a`) is a PERFORMANCE cache**, by its own comment: "Accelerates marginal
volume calculations by caching previous DH lookups" -- when consecutive cells belong to the same leaf
depression, the same outlet set applies, so it avoids revisiting the DH. It changes no answer.
NOTE THE COLLISION RISK BEFORE PORTING IT: `CalculateMarginalVolumes` is exactly where WTM's
area-weighted volumes live (its one correctness-bearing local change, DH_INTEGRATION gap 3 / ENH-4).
A cache over that function must be reconciled with the area weighting, not dropped on top of it.

**`4b491a1` "Start of serialization"** adds cereal-based serialization. WTM does not serialize the DH,
so this is unused surface.

CONCLUSION: the vendored copy is not behind on anything that changes a result. The divergence that
matters runs the OTHER way -- WTM's local additions (area-weighted volumes, the FSM fields, `is_full`),
which upstream does not have and which any replacement must preserve.
