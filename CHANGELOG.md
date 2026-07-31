# Changelog

All notable changes to WTM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Work since v2.0.1. The **default surface-water / evaporation model is now the smooth
transition** — the surface-transition tapers (1–3, below) are on by default, replacing the
hard `wtd = 0` switch. Other new capabilities remain experimental and off by default (gated
behind `-wtm_*` runtime options). The full regression suite (`tests/run_all.sh`: DMDA unit
tests, ghost-cell, MPI-consistency, mass-balance, FillSpillMerge-consistency, golden, and
taper) passes.

### Added

#### Solvers and time integration
- **Semi-implicit Picard groundwater solver** (`-wtm_picard`): builds an SPD operator solved
  with CG + GAMG, as an alternative to the matrix-free default. See `benchmark/picard/PICARD_MATH.md`.
- **Second-order transient time integration.** Fixed-step BDF2 (`-wtm_bdf2`), a volume-form
  variant that is genuinely 2nd order under recharge (`-wtm_bdf2_on_V`), variable-step BDF2, and
  **adaptive time stepping** (`-wtm_dt_adaptive`). See `benchmark/picard/BDF2_ADAPTIVE_DESIGN.md`.
- **Extended-soil option** (`-wtm_extended_soil`, _experimental_): continues the aquifer above the
  land surface to remove the water-table-depth = 0 free boundary from the groundwater step.
- **Configurable transmissivity / storativity smoothing** (`-wtm_ksat_soilbottom_smoothing_width`,
  `-wtm_ksat_surface_smoothing_width`, `-wtm_storativity_surface_smoothing_width`): optional rounding
  of the piecewise T/S boundaries, applied consistently across all solver paths.

#### Smooth surface-water transition (now the default — three tapers, per-taper off-switches)
The hard `wtd = 0` switch is replaced by three smooth, implicit, order-preserving tapers, all **on by
default** and each individually disabled by its flag (e.g. `-wtm_evap_taper 0`):
- **Taper 1 — sub-surface sink** (`-wtm_surface_sink`): a near-surface removal that holds the water
  table at/below the surface and hands the exfiltrated water to FillSpillMerge (it stays in the
  domain) — the smooth replacement for the hard "surface water → runoff" handoff, preserving 2nd-order
  accuracy across `wtd = 0`. On both the Anderson default and Picard/BDF2-on-V paths. Its width scales
  with the timestep (`width = C·qmax·dt`, C = 2) for stability at every step (tight — mm–cm — at small
  transient dt, wider only at large equilibrium dt); `qmax` default 1 m/yr, `-wtm_surface_sink_width`
  overrides.
- **Taper 2 — demand-identity evaporation** (`-wtm_evap_taper`): a single smooth, implicit transition
  from land-surface evapotranspiration (deep) to open-water evaporation (at/above the surface),
  replacing the hard ET↔open-water switch. Makes FillSpillMerge lake formation cross-rank deterministic
  at the evaporation threshold. Works in both `evap_mode 1` and `evap_mode 0` (it supersedes evap_mode
  0's remove-all).
- **Taper 3 — accessibility / extinction-depth clamp** (`-wtm_extinction`, depth `-wtm_extinction_depth`,
  default 8 m): gates taper 2's sub-surface evaporative deficit by depth, so an arid table (`ET > precip`)
  draws down only within the extinction depth (phreatic ET) rather than without bound. Depth basis:
  rooting depths (Canadell et al. 1996) / groundwater-ET extinction depths (Shah et al. 2007), see
  `benchmark/SURFACE_SINK_DESIGN.md` §14f.

Any configuration other than all-three-on emits a warning (arid-unsafe, inert, or the legacy
hard-switch model). See `benchmark/SURFACE_SINK_DESIGN.md`.

#### Scaling and memory
- **Distributed data model** for single-node, many-core runs. The full grid is no longer replicated
  on every MPI rank: the water table, per-cycle recharge, and the static solve fields are carried in
  distributed DMDA vectors, while FillSpillMerge and the depression hierarchy run on rank 0. This
  lifts the replicated-memory ceiling that previously bounded core counts. See
  `benchmark/DISTRIBUTED_ARP_DESIGN.md`.

#### Water budget
- **Land → ocean outflow accounting** (Darcy interface flux) and an **exact per-step discrete water
  budget that closes to machine zero** on the Picard path, with separate loss channels for the
  sub-surface sink and the evaporation taper. See `benchmark/WATER_BUDGET.md`.

#### Testing and tooling
- Regression suites: **golden** (expected-results, transient + evap-mode coverage), **MPI-consistency**
  (n = 1 vs n = N), **mass-balance**, **ghost-cell**, **taper** (surface-transition cross-rank
  determinism + smoothness — the `SURFACE_SINK_DESIGN` §14d experiment sequence), and DMDA
  gather/scatter **unit tests**; synthetic terrain generators (spectral / Fourier-mode and fractal).
- **`BUILD_HPC.md`** cluster build guide (MSI worked example), single-node scaling/memory study drivers,
  a solve profiler, publication figure and dataset generators, and design notes.

### Changed
- **Default surface-water / evaporation model is now the smooth transition** (surface-transition
  tapers 1–3 on). This replaces the hard `wtd = 0` ET↔open-water switch — which made FillSpillMerge
  lake formation rank-dependent (non-deterministic across MPI rank counts) and applied no phreatic ET —
  with a cross-rank-deterministic, 2nd-order-preserving transition, and it lets arid tables draw down
  physically (phreatic ET to an extinction depth). Recharge becomes the precip source with evaporation
  carried by the smooth `E_eff`, and runoff becomes `runoff_ratio · precip` (a split of the source).
  Golden references were regenerated. Disable per taper (e.g. `-wtm_evap_taper 0`) for the legacy
  behavior; see the tapers under _Added_.
- **Solver defaults are chosen automatically; no PETSc solver flags are required on the command line.**
  Equilibrium runs use the matrix-free Anderson solver (acceleration window m = 10) with the piecewise
  Fan transmissivity and mild damping (`-snes_anderson_beta 0.5`) for robust convergence on steep,
  heterogeneous real terrain, at a step tolerance of `1e-8` (the damped solver converges linearly, so
  the previous `1e-6` stopped a hair loose and left µm-scale cross-rank rank-dependence). Everything
  remains overridable via the usual `-snes_*` options.
- **Transient runs default to the genuinely 2nd-order `-wtm_bdf2_on_V` (Picard) solver.** On a
  transient the time-discretization accuracy is the answer, and the matrix-free Anderson path silently
  under-converges (and diverges when pushed) on stiff transient drainage, so the semi-implicit path is
  both more accurate and more robust there. Force the matrix-free path on a transient with
  `-wtm_anderson`; explicit `-wtm_*` path flags also take precedence. Equilibrium runs are unchanged
  (Anderson — the steady state does not depend on the time scheme).
- **Conservative finite-volume flux discretization.** Corrects a longitude/latitude grid-spacing swap
  (the east–west and north–south fluxes had each been divided by the _other_ direction's spacing) and
  restores exact flux conservation across shared cell faces. This is a discretization-correctness fix;
  its numerical effect on results is quantified in `benchmark/REVIEW_NOTES_since_v2.0.1.md`.
  Golden references were regenerated. See `benchmark/GRID_CONVENTION.md`.
- **Recharge computation distributed across ranks** (previously serial on rank 0) where FillSpillMerge
  coupling permits; a warning is emitted for the configuration that must stay on the serial path
  (`infiltration_on` with FSM on).
- **CMake defaults to a Release build** when no build type is specified.

### Fixed
- **MPI ghost-cell bug** in the groundwater solve, with a dedicated ghost-cell validation test.
- **Integer division** that froze transient forcing at its start-time values.
- Water-budget diagnostics made **MPI-consistent** (owned-cells-only partials + scalar reduction).
- Picard post-equilibrium **false divergence** (a sensible default `-snes_atol`).
- **Anderson solver stall on steep real terrain** (`DIVERGED_MAX_IT` at the iteration cap; reported on a
  real DEM). The undamped default diverged on steep, heterogeneous topography; the default now damps
  (`-snes_anderson_beta 0.5`), verified to converge on the Corsica DEM at 1–4 MPI ranks. The damping is
  the fix — a wider acceleration window also converges but adds per-iteration parallel reductions that
  the discontinuous FillSpillMerge routing amplifies into ~mm cross-rank differences, so m stays at 10.

### Deprecated
- **Newton-Krylov solver path disabled.** The analytic-Jacobian path (`-snes_type newtonls` _without_
  `-wtm_picard`) is unused, and its Jacobian is unmaintained (it lacks the sink / evaporation-taper
  terms), so it would diverge. It is now refused at solver setup with a message pointing to the default
  Anderson solver or `-wtm_picard`. The Jacobian code is retained in-source so the path can be rebuilt.
  (The `-wtm_picard` path, which drives `newtonls` internally, is unaffected.)

### Removed
- The `-wtm_const_storativity` diagnostic path.

[Unreleased]: https://github.com/KCallaghan/WTM/compare/v2.0.1...HEAD
