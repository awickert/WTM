# Changelog

All notable changes to WTM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Work since v2.0.1. Several new capabilities are **experimental and off by default**
(gated behind `-wtm_*` runtime options); the default solve path is unchanged in behavior
except where noted under _Changed_, and the regression suites (golden, MPI-consistency,
mass-balance) pass.

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

#### Smooth surface-water transition (experimental, off by default)
- **Sub-surface sink** (`-wtm_surface_sink`): a smooth, order-preserving near-surface removal that
  holds the water table at/below the surface and hands the exfiltrated water to FillSpillMerge as a
  per-cell input — the smooth replacement for the hard "surface water → runoff" handoff. Available on
  both the Picard/BDF2-on-V and the Anderson default paths.
- **Demand-identity evaporation taper** (`-wtm_evap_taper`, requires `evap_mode 1`): replaces the
  hard switch at the surface between land-surface evapotranspiration and open-water evaporation with a
  single smooth, implicit transition. Restores cross-rank determinism of FillSpillMerge lake formation
  at the evaporation threshold. See `benchmark/SURFACE_SINK_DESIGN.md`.

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
- **Default solver is Anderson** (acceleration window m = 10) using the piecewise Fan transmissivity.
- **Conservative finite-volume flux discretization.** Corrects a longitude/latitude grid-spacing swap
  (the east–west and north–south fluxes had each been divided by the _other_ direction's spacing) and
  restores exact flux conservation across shared cell faces. This is a discretization-correctness fix;
  its effect on results is **relatively minor** — a small change in the summed water-table depths.
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

### Deprecated
- **Newton-Krylov solver path disabled.** The analytic-Jacobian path (`-snes_type newtonls` _without_
  `-wtm_picard`) is unused, and its Jacobian is unmaintained (it lacks the sink / evaporation-taper
  terms), so it would diverge. It is now refused at solver setup with a message pointing to the default
  Anderson solver or `-wtm_picard`. The Jacobian code is retained in-source so the path can be rebuilt.
  (The `-wtm_picard` path, which drives `newtonls` internally, is unaffected.)

### Removed
- The `-wtm_const_storativity` diagnostic path.

[Unreleased]: https://github.com/KCallaghan/WTM/compare/v2.0.1...HEAD
