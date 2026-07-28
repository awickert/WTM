# WTM — review notes for the work since v2.0.1

A companion to `CHANGELOG.md` for reviewing/merging this branch. It focuses on the things a
maintainer needs to weigh: the one results-affecting default change, the motivating finding, what is
production-ready vs. experimental, and the decisions that are yours to make.

## TL;DR

- **Most new capability is opt-in** — behind `-wtm_*` runtime flags — and the default solve is
  behavior-preserving *except* for the two items below. The regression suites (golden,
  MPI-consistency, mass-balance, ghost-cell) pass.
- **Two default-behavior changes:** (1) the default solver is **Anderson** (m = 10); (2) a
  **conservative finite-volume flux fix**. The flux fix is a *discretization-correctness* change whose
  numerical impact is **modest on realistic terrain** (quantified below) and which restores exact flux
  conservation.
- **Motivating finding (worth your attention):** FillSpillMerge lake formation could be
  **MPI-rank-count-dependent** near the water-table = surface evaporation threshold — the hard `wtd=0`
  switch amplifies unavoidable ~1e-13 parallel-reduction noise into lake-scale differences. The new
  (experimental) surface-transition tapers remove it.
- **Memory scaling:** the full grid is no longer replicated on every MPI rank (distributed data
  model), lifting the ceiling on single-node core counts.

## 1. The one results-affecting default change — the conservative flux fix

**What it corrects.** The groundwater flux discretization had the two horizontal directions' grid
spacings **swapped**: the east–west (longitude) flux was divided by the north–south spacing and vice
versa. Away from the equator this is wrong by a factor of ~cos²(latitude), and it was
non-conservative across north–south cell faces. The fix rewrites the flux in conservative
finite-volume form (shared-face conductances that cancel exactly) with the correct per-row geometry.
See `benchmark/GRID_CONVENTION.md`. The Picard operator uses the volume form (for exact symmetry); the
matrix-free Anderson residual uses the head form (same root, better conditioning).

**Impact — quantified (old flux vs. new flux, identical inputs).** From the regenerated golden
references (16×16 fixtures at `southern_edge = -45`, i.e. cos²(45) ≈ 0.5):

| case (fixture) | mean \|Δwtd\| | max \|Δwtd\| | Δ(Σ depths)/\|Σ\| |
|---|---|---|---|
| `fsm_runoff` (realistic 2-D terrain) | 0.003 m | 0.028 m | 2e-7 |
| `fsm_runoff_hi` (richer terrain) | 0.004 m | 0.028 m | 6e-6 |
| `transient` (2-D) | 0.012 m | 0.21 m | 3e-4 |
| `fsm_evap0/1` (plateau + supplied surface water) | ~1–1.6 m | 6–8 m | ~1e-4 |
| `below_ground` (degenerate 3×12 1-D transect) | 3.3 m | 15.5 m | 0.18 |

Read: on **realistic 2-D terrain the change is small** (mm–cm per cell, negligible in the summed
water table). The larger per-cell numbers appear only where the geometry amplifies the anisotropy
correction — the plateau fixtures with strong supplied surface water, and the degenerate 1-D
`below_ground` transect (a thin land strip with a steep sub-surface gradient, not representative). The
effect scales with cos²(latitude), so it is largest toward the poles. (These are the exact old-flux
vs. new-flux golden references on identical inputs — a clean A/B with no other confounds. A bespoke
higher-resolution / higher-latitude run can be produced on request.)

**Correctness gain.** Beyond the anisotropy, the new form makes the per-step water budget close to
**machine zero** (see `benchmark/WATER_BUDGET.md`); the old form did not conserve exactly across N–S
faces. Golden references were regenerated for the fix.

## 2. The FillSpillMerge determinism finding (motivates the tapers)

WTM's recharge branch is a hard switch at wtd = 0: sub-surface cells lose land-surface ET, surface
cells lose open-water evaporation. When an equilibrium sits on that threshold, a ~1e-13 rank-dependent
difference in the parallel groundwater solve (floating-point reductions are non-associative) can tip
cells across wtd = 0, flip the evaporation regime and the FSM fill/spill routing, and produce
**lake-scale, rank-count-dependent** differences. This is why the `fsm_evap1` golden case carries a
centimetre tolerance rather than the usual ~1e-6. The groundwater solve *alone* (FSM off) is
MPI-consistent to ~1e-13; it is the discontinuous routing that amplifies it.

## 3. New capabilities (opt-in, off by default)

- **Semi-implicit Picard solver** (`-wtm_picard`) and **2nd-order transient time integration**
  (`-wtm_bdf2`, `-wtm_bdf2_on_V`, adaptive `-wtm_dt_adaptive`). Validated; not the default.
- **Smooth surface-water transition (experimental).** Two implicit, order-preserving pieces that
  together remove the finding-2 non-determinism: a **sub-surface sink** (`-wtm_surface_sink`) that
  hands near-surface exfiltration to FSM smoothly, and a **demand-identity evaporation taper**
  (`-wtm_evap_taper`, requires `evap_mode 1`) that replaces the hard ET↔open-water switch with one
  smooth transition. See `benchmark/SURFACE_SINK_DESIGN.md`. New regression suite (`tests/taper/`)
  asserts cross-rank determinism through the threshold and a smooth pond/shoreline. One piece is
  intentionally deferred — an accessibility / extinction-depth clamp for arid `ET > precip` cells
  (fork issue #4).
- **Extended-soil** (`-wtm_extended_soil`) and **configurable T/S smoothing widths**.

## 4. Memory scaling — distributed data model

The water table, per-cycle recharge, and static solve fields are carried in distributed DMDA vectors
(FillSpillMerge and the depression hierarchy stay on rank 0), so the full grid is no longer replicated
on every rank. This is the change that lets single-node many-core runs go past the previous
replicated-memory ceiling. See `benchmark/DISTRIBUTED_ARP_DESIGN.md`.

## 5. Other changes worth flagging

- **Anderson (m = 10) is the default solver**, with the piecewise Fan transmissivity.
- **The Newton-Krylov (analytic-Jacobian) path is disabled** — it was unused and its Jacobian is
  unmaintained (missing the sink/evap-taper terms), so it is refused at setup with a message pointing
  to Anderson or `-wtm_picard`. The Jacobian code is retained in-source for a future rebuild. (This is
  a removed option — flagged for your awareness.)
- CMake defaults to a Release build.

## 6. Testing

New/expanded suites, all in `tests/` + `tests/run_all.sh`: **golden** (expected results),
**MPI-consistency** (n = 1 vs N), **mass-balance**, **ghost-cell**, **taper** (surface-transition
determinism + smoothness), and DMDA unit tests; plus synthetic terrain generators.

## 7. Production-ready vs. experimental

- **Ready to adopt:** the conservative flux fix, the Anderson default, the distributed data model, and
  the test suites.
- **Validated but opt-in:** Picard / BDF2 / adaptive time stepping.
- **Experimental (WIP):** the surface-transition tapers and extended-soil — off by default; the tapers
  need the deferred accessibility piece (#4) before they could be a sensible default.

## 8. Decisions that are yours

1. **Should the surface-transition tapers become the default?** They fix the finding-2
   non-determinism, but promoting them is a deliberate step: it would rebaseline golden and pulls in
   the deferred accessibility clamp (#4). Until then they stay opt-in.
2. **The Newton-Krylov removal** — confirm you are fine retiring that path (code retained).
3. Anything on the flux-fix numbers you would like re-run on a specific region/latitude.
