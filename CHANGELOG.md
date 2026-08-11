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
- **Analytic-Jacobian Newton solver** (`-wtm_newton`): a true Newton–Krylov path (GMRES + GAMG) driven
  by an exact, finite-difference-verified Jacobian of the conservative-FV residual — including the
  surface-sink and evaporation / accessibility taper tangents. Paired with **dt-continuation**
  (`-wtm_dt_continuation`): a pseudo-transient ramp that starts `deltat` small, so a far / cold initial
  water table stays inside the Newton basin, and grows it after each converged step until the table
  settles. On this dt-continuation path the convergence-based early stop (`-wtm_eq_tol`) only engages once
  `deltat` has ramped up, so it cannot trip early. The convenience bundle **`-wtm_stiff`** turns on all three
  at once (equivalent to `-wtm_newton -wtm_dt_continuation -wtm_eq_tol 0.01`) for hard equilibrium cold-starts
  on stiff terrain. See `benchmark/EQUILIBRIUM_ROBUSTNESS.md`. (`-wtm_eq_tol` is now a general, all-paths
  equilibrium stop that is on by default — see "Automatic equilibrium stop" below.)
- **Second-order transient time integration.** Fixed-step BDF2 (`-wtm_bdf2`), a volume-form
  variant that is genuinely 2nd order under recharge (`-wtm_bdf2_on_V`), variable-step BDF2, and
  **adaptive time stepping** (`-wtm_dt_adaptive`). See `benchmark/picard/BDF2_ADAPTIVE_DESIGN.md`.
  Time-order is decoupled from the solver: **`-wtm_bdf2_on_V` composes with `-wtm_anderson`** to give
  the matrix-free Anderson solver a genuine 2nd-order-in-time residual (no operator/preconditioner),
  so a run can be both fast (Anderson's cheap matrix-free iterations) and 2nd-order in time. It shares
  the Picard BDF2-on-V fixed point exactly, and leaves Anderson's stable time step unchanged (measured).
- **TR-BDF2 for matrix-free Anderson** (`-wtm_tr_bdf2`): an L-stable, strongly (monotonically) damped
  2nd-order-in-time alternative to plain BDF2-on-V (whose stiff-mode damping is oscillatory). One step is
  two staged implicit solves (trapezoidal to `t + γΔt`, then BDF2 to `t + Δt`; γ = 2−√2, self-starting).
  Measured on a warm 2× perturbation it takes **2× the stable step (8 vs 4 weeks)** of BE / BDF2-on-V and
  needs ~6× fewer iterations near the ceiling (no ringing), for ~11 % more work per step at small `deltat`.
  See `benchmark/TBAR_TIME_AVERAGING.md`.
- **Time-averaged interblock transmissivity** (`-wtm_Tbar`, _experimental_): uses each cell's
  step-time-averaged transmissivity `T̄ = (Φ(wᵗ⁺¹) − Φ(wᵗ)) / (wᵗ⁺¹ − wᵗ)` (the Kirchhoff-potential
  difference — the log-mean of the exponential deep T, the arithmetic mean of the affine soil T, and the
  constant surface T, continuously) as the per-cell value feeding the unchanged harmonic interblock mean,
  instead of the instantaneous start-of-step T. This addresses the exponential T's frozen-coefficient lag
  that makes the outer iteration oscillate on stiff steps. Same physics and same equilibrium (`T̄ → T` at
  steady state); it composes with every solver (Anderson residual, Picard operator, exact Newton
  Jacobian). Requires the piecewise Fan T (refused with ksat smoothing, extended soil, or Kirchhoff).
  See `benchmark/TBAR_TIME_AVERAGING.md`.
- **Predictor-seeded initial guess** (`-wtm_predict_guess`, _experimental_): seeds the solve's initial
  guess (and hence the iteration-1 `T̄` coefficient) with a guarded 2nd-order history extrapolation
  `wᵗ⁺¹ ≈ wᵗ + ω(wᵗ − wᵗ⁻¹)` (forward-Euler `wᵗ + Δt·f(wᵗ)` on the first step, which has no history),
  instead of `wᵗ`. Without it the iteration-1 `T̄` collapses to the instantaneous `T(wᵗ)`, so `T̄`'s
  before-and-after advantage is unrealized on the first residual evaluation. Measured with `-wtm_Tbar` it
  cuts nonlinear iterations (~11 % on a 2-week warm transient; ~48 % at 8-week steps; ~34 % on the first
  step) — a **speed** win only: it does **not** change the equilibrium and does **not** raise the stable
  step ceiling (that is set by the operator, not the guess). Off by default.
- **Extended-soil option** (`-wtm_extended_soil`, _experimental_): continues the aquifer above the
  land surface to remove the water-table-depth = 0 free boundary from the groundwater step.
- **Configurable transmissivity / storativity smoothing** (`-wtm_ksat_soilbottom_smoothing_width`,
  `-wtm_ksat_surface_smoothing_width`, `-wtm_storativity_surface_smoothing_width`): optional rounding
  of the piecewise T/S boundaries, applied consistently across all solver paths.
- **Automatic equilibrium stop, on by default** (`-wtm_eq_tol`). The convergence-based early stop now works
  on **all** solver paths (it was previously silently ignored except on the Newton dt-continuation path) and
  gates on the **per-cycle** water-table change `max|wtdᴺ − wtdᴺ⁻¹|` over land — the honest steady-state
  signal, free of the cosmetic within-cycle free-boundary flicker that the per-sub-step change carries. It is
  **on by default for equilibrium runs** (0.01 m ≈ 1 cm per ~1-yr cycle; two consecutive cycles below the
  tolerance → stop) and **off for transient runs**, which must play out in full. `-wtm_eq_tol 0` disables it;
  any value overrides. Each cycle prints the per-cycle change alongside a within-cycle flicker diagnostic.
- **Sub-step under-relaxation** (`-wtm_relax a`, default 1 = off): blends each sub-step's solved water table
  with the previous one, `w ← a·w_solve + (1−a)·w_prev`, damping the period-2 flicker at pinned free
  boundaries (a FillSpillMerge lake surface, or a `-wtm_direct_to_runoff` seepage cell). Inert at steady
  state — the equilibrium is unchanged; only the transient march is damped.

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

- **Direct-to-runoff seepage face** (`-wtm_direct_to_runoff`, _opt-in alternative to taper 1_): removes the
  above-surface water-table excess to runoff each sub-step at rate `max(0,wtd)/dt`, pinning the table at the
  land surface with no rate cap (so no runaway pile) and no below-surface band (so no artificial depression).
  A simpler, tuning-free surface→runoff handoff; the removed water is routed to FillSpillMerge, so it stays
  in the domain. Also modestly **faster** (~24 % per cycle on Esquibel), because pinning the seepage cells
  removes the above-surface churn that otherwise keeps each cycle's solve stiff. Off by default.

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
- **The default solver is now the semi-implicit BDF2-on-V (Picard) path, for both run types; no PETSc
  solver flags are required on the command line.** Equilibrium reaches steady state in a handful of
  large, stable steps — Picard's Newton + GAMG solve has a nearly step-size-independent cost, so
  `deltat` can be raised by orders of magnitude (measured flat at ~28 SNES iterations from `deltat` = 1
  to 1000 yr on a real DEM). Transient gets genuine 2nd-order-in-time accuracy from the same solver.
  This **replaces the previous matrix-free Anderson default**, which — having no preconditioner — is
  stiffness-limited: it diverges once `deltat` is raised (so it cannot take the large equilibrium
  steps) and it under-converges on stiff transients. Anderson is retained as an opt-in, `-wtm_anderson`
  (faster per step at small `deltat`, bit-exact across ranks; for small-`deltat` / fast-science cases);
  explicit `-wtm_*` path flags and `-snes_*` options still take precedence. As an incidental win, the
  Picard default is cross-rank consistent to ~1e-9 even on the FSM-routing-threshold fixtures, so the
  golden tests no longer need the physical (mm–cm) tolerances the Anderson default required.
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
  real DEM). The undamped Anderson path diverged on steep, heterogeneous topography; the Anderson
  (now opt-in, `-wtm_anderson`) path damps (`-snes_anderson_beta 0.5`), verified to converge on the
  Corsica DEM at 1–4 MPI ranks. The damping is the fix — a wider acceleration window also converges but
  adds per-iteration parallel reductions that the discontinuous FillSpillMerge routing amplifies into
  ~mm cross-rank differences, so m stays at 10.

### Removed
- The `-wtm_const_storativity` diagnostic path.

[Unreleased]: https://github.com/KCallaghan/WTM/compare/v2.0.1...HEAD
