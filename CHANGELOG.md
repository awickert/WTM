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

#### Surface-water routing
- **`runoff_collector` config-file selector** (`implicit` | `explicit` | `off` | `legacy`) unifies how the
  `wtd = 0` seepage face — where above-surface water is routed to runoff / Fill-Spill-Merge — is enforced.
  `implicit` is the in-residual seepage: exact, dt-independent, pins `wtd = 0`, wired into the Anderson
  residual **and** the Picard operator (a frozen active-set diagonal); Newton still warns (its Jacobian needs a
  semismooth/active-set treatment of the discontinuous kink). `explicit` is the post-solve clamp: robust on
  every solver *and* under adaptive-dt, within ~1 cm of `implicit` and converging as `dt → 0`. `off` collects
  nothing (above-surface water piles up — nonphysical, warns; supersedes
  `-wtm_dev_allow_aboveground_water_columns`). `legacy` keeps the old `-wtm_surface_sink` band-sink defaults.
  The modes are mutually exclusive (no hidden clamp backstop under `implicit`, so its misbehaviour stays
  visible). **Default is AUTO** (key omitted): `implicit` normally, but `explicit` under `-wtm_dt_adaptive`
  (the implicit kink can't be adaptively step-sized). See `benchmark/SURFACE_WATER_ROUTING.md`.

#### Boundary conditions
- **Selectable land-edge boundary condition** (`-wtm_land_boundary neumann_toposlope|dirichlet`): ocean edges
  are always Dirichlet `h = 0`; land edges default to terrain-following no-flow (`neumann_toposlope`) but can be
  set to sea-level Dirichlet (`dirichlet`), where a land edge behaves exactly as an ocean neighbour (head 0 and
  surface transmissivity via ghost nodes). Wired into all solver paths — the matrix-free residual, the Newton
  analytic Jacobian (FD-verified), and the Picard operator+RHS — and the water budget. The `dirichlet` mode
  reproduces the legacy sea-level padding to machine precision (7e-12 m). Not compatible with `-wtm_kirchhoff`.
  Regression: `tests/boundary_consistency/`. See `benchmark/BOUNDARY_CONDITIONS.md`.

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
  **adaptive time stepping** (`-wtm_dt_adaptive`). See `benchmark/BDF2_ADAPTIVE_DESIGN.md`.
  The **adaptive controller is detached from the integrator**: `-wtm_dt_adaptive` composes with any of
  them — `-wtm_anderson` → 1st-order backward-Euler (ring-proof), `-wtm_tr_bdf2` → TR-BDF2,
  `-wtm_bdf2_on_V` → BDF2-on-V — the error *estimate* being the only method-specific piece (TR-BDF2's
  embedded two-stage estimate, else the generic linear-history predictor) feeding one shared
  grow/shrink/reject controller. Its error norm **includes the free surface** (where stability is set),
  which is what lets it settle a cold start rather than growing Δt into a surface limit cycle. A **PI
  step-size controller** damps the Δt "hunting" that otherwise locks into resonance dead-bands, so on an
  equilibrium/spin-up run the step tolerance is **derived from the convergence target** — `dt_tol =
  min(50·eq_tol, 0.5 m)` unless `-wtm_dt_tol` is set — making **`eq_tol` the single knob**, with no toxic
  `dt_tol`/`eq_tol` combination and no fixed dt that can ring on unfamiliar terrain.
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
- **Adaptive time stepping for TR-BDF2** (`-wtm_tr_bdf2 -wtm_dt_adaptive`, _experimental_): a self-tuning
  step that stays as large as accuracy and convergence allow — "long enough to be efficient but not so long
  it fails" on terrain / conditions whose stable-step ceiling varies. Two coupled mechanisms: a
  **reject/retry feasibility floor** — a non-converged stage or step shrinks `deltat` and retries from the
  uncommitted state (accumulators rolled back), so a step too large for the local conditions cannot crash
  the run — and an **embedded error estimator** from TR-BDF2's two stages, `h_pred = [Y_γ − (1−γ)hⁿ]/γ`
  (exact for linear-in-time, `O(Δt²)` for curvature; needs no history, valid on the first step): `est >
  -wtm_dt_tol` shrinks and retries, otherwise `deltat` grows toward the tolerance, capped by the step's
  convergence headroom (`-wtm_dtc_easy_iters`) and `-wtm_dtc_dt_max`. The error norm **excludes surface
  cells** (`wtd ≥ −band`) so the non-smooth free-surface clamp cannot spike the estimate and force `deltat`
  tiny (the failure that shelved the earlier history-extrapolation estimator). Measured on Esquibel
  (dry −20 %): `-wtm_dt_tol 1/5/20 m` → 33/12/8 steps (6/3/1 rejected), all converged — monotone in the
  tolerance. Off by default. See `benchmark/BDF2_ADAPTIVE_DESIGN.md`.
- **Robust equilibrium auto-stop** (`-wtm_eq_metric`, `-wtm_eq_frac`). The `-wtm_eq_tol` per-cycle early
  stop now applies on **every** spin-up pathway (fixed-dt, Newton-continuation, and the adaptive-dt
  controller — previously skipped). Its aggregation of the per-cycle water-table change is selectable:
  `frac` (**default**: converged when < `-wtm_eq_frac` = 0.1 % of land cells still exceed `eq_tol`), `max`
  (the old strict worst-cell criterion), or `rms`. The default changed from `max` to `frac` because `max`
  is worst-cell-hostage — one slow deep lowland cell filling to the surface can keep it from ever firing
  even though the bulk has converged (diagnosed as a metric artifact, not a physical oscillation). Measured
  trade at `eq_tol` 0.05 m: `max` never stops, `rms` stops early but loose (14.6 m worst-cell residual),
  `frac` stops with a 4.3 m worst-cell residual — the robust middle. See `benchmark/adaptive_dt/`.
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
- **The default solver is now matrix-free Anderson** (was semi-implicit BDF2-on-V/Picard). Anderson is the
  robust production worker, carries the exact in-residual seepage face, and is bit-exact across MPI ranks;
  it is 1st-order-in-time (backward-Euler cc, the right choice for equilibrium). Opt into BDF2-on-V/Picard
  (large stable steps, 2nd-order) with `-wtm_bdf2_on_V`. Picard is retained as the cross-rank-deterministic
  **grounding reference** the golden tests hold Anderson against. Combined with the `runoff_collector` AUTO
  default (implicit surface seepage; explicit under `-wtm_dt_adaptive`), the golden references were
  regenerated. The old default is exactly reproducible with `runoff_collector legacy -wtm_bdf2_on_V` (verified
  bit-for-bit against the pre-flip goldens), so every golden delta is the intended change, not a regression:
  below_ground 4.37 m (solver order over the deliberately short 6-step run), fsm_evap0/1 10.0 m (band sink →
  implicit seepage face), fsm_runoff/hi 36.5/31.6 m max but 0.28 m mean (a few FSM routing-threshold cells
  flip), transient unchanged.
- **The mask-aware ghost boundary is now the default** (no flag needed). It applies Dirichlet `h = 0`
  (constant head) at ocean edges and land-slope Neumann (constant flux) at land edges, computed at the true
  domain edge, and replaces the legacy edge-padding (`setEdges(0)`). Behaviour-neutral on ocean-ringed domains
  (where padding and the ghost BC coincide); it changes results only where real land meets a domain edge. The
  legacy sea-level-padding boundary is retained as a verification tool behind `-wtm_dev_padded_dirichlet`
  (which forces every edge to ocean `h = 0` and **fails loudly** unless the domain boundary is already all
  ocean, so it cannot silently discard edge land). See task #96 and `benchmark/BOUNDARY_CONDITIONS.md`.
- **The surface-water exfiltration clamp is now the default on every solver path**
  (`-wtm_surface_exfiltration_to_runoff`, on; disable with `-wtm_surface_exfiltration_to_runoff false`).
  It pins the water table at/below the surface and routes above-surface water to Fill-Spill-Merge (or to
  runoff when FSM is off), the physical Fan & Miguez-Macho behaviour, so physical runs never leave water
  ponded above ground as a raised water table and never flicker at the free surface. Previously default-on
  only for the matrix-free Anderson path; now also on for the default Picard/Newton paths (a post-solve
  clamp there, complementing the in-residual taper-1 sink; the operator-consistent in-residual seepage
  `-wtm_direct_to_runoff` is task #100). Golden references were regenerated: the subsurface case is
  unchanged, and cases that generated above-surface water shifted (fsm cases up to ~9.4 m, the 4-cycle
  cold-start transient up to ~24.9 m as routing early surface water changes the trajectory).
- **Renamed the nonphysical developer switch `-wtm_allow_surface_ponding` to
  `-wtm_dev_allow_aboveground_water_columns`** (the `-wtm_dev_` prefix marks it developer/nonphysical at the
  point of use, and the new name says what it actually permits — vertical water columns standing above the land
  surface, not lakes). It still disables both runoff clamps and prints a warning; it is a
  testing/diagnostics-only regime (used by `tests/boundary_analytic` to reach the constant-transmissivity
  ponded-parabola solution), never a valid model configuration.
- **Snapshot output filenames now carry the simulated year.** Water-table rasters are written as
  `{outfile_prefix}{cycle:09}_{years}yr.tif` (was `{outfile_prefix}{cycle:09}.tif`), so each periodic
  snapshot is self-describing by simulated time — essential for transient runs, informative for spin-up
  progress. `years = cycles_done · maxiter · deltat / seconds_in_a_year` (a cycle spans a fixed
  `maxiter·deltat` even under adaptive dt). The zero-padded cycle stays the leading field, so any
  `glob(prefix + "*.tif")` + sort still orders by cycle (the golden suite is unaffected). Downstream
  analysis scripts that constructed the exact old name now glob for the final output.
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
- **Anderson near-convergence instability at large scale** (`DIVERGED_MAX_IT`; a cold ~139-million-cell
  equilibrium solve). Near convergence the Anderson residual-difference vectors go nearly linearly
  dependent, the least-squares mixing coefficients blow up, and the residual *reverses and oscillates*
  instead of settling. The fix is a **periodic history restart, now on by default for the Anderson path**
  (`-snes_anderson_restart_type periodic -snes_anderson_restart 20`): purging the history before it
  degenerates lets the solve converge (~40 iterations at 139M, robustly across periods 10–25). It is a
  **safe conditional default** — small grids converge in fewer than 20 iterations so the restart never
  fires (the full regression suite is byte-identical with it on), and only large / stiff runs engage it.
  Chosen over widening `m` (which converges but doubles the per-iteration reductions and reopens the
  cross-rank-consistency issue) and over the adaptive *difference* restart (which triggers on a residual
  rise that only occurs *at* the flail — too late). The instability is driven by high-latitude `cos(lat)`
  cell anisotropy, so it is a real property of large real-world domains, not just a test artifact.
  Disable with `-snes_anderson_restart_type none`. See `benchmark/esquibel/` sweeps.
- **Optional ρ-adaptive Anderson restart** (`-wtm_adaptive_restart`): a *proactive* alternative to the
  fixed-period default that restarts Anderson's history when the convergence *rate* degrades
  (ρ = ‖F_k‖/‖F_{k-1}‖ → 1 — the flail precursor, which appears *before* the residual rises), restarting
  each phase from the best iterate. Because it triggers on the rate rather than a fixed count, it adapts
  to a flail arriving at an unknown iteration — robustness for scales beyond those tested (the road to
  global). Confirmed at 139M: converges, and slightly faster than the periodic default (~112 s vs ~140 s,
  by restarting only when needed). Off by default; tunable via `-wtm_ar_rho / _patience / _max_it /
  _max_restarts`. Robust finish: near equilibrium the Anderson step floors just above the relative step
  tolerance, so true convergence is never formally declared — the controller now returns the tracked best
  iterate (and a phase that diverges after a good iterate falls back to it, with a warning) instead of
  aborting once restarts are exhausted. Regression: `tests/adaptive_restart/`.

### Removed
- The `-wtm_const_storativity` diagnostic path.

[Unreleased]: https://github.com/KCallaghan/WTM/compare/v2.0.1...HEAD
