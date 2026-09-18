# Changelog

All notable changes to WTM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Work since v2.0.1. Two headline changes: **configuration is now nested YAML**, and the
**default surface-water / evaporation model is the smooth transition** (the surface-transition
tapers, 1–3 below, are on by default, replacing the hard `wtd = 0` switch). Other new
capabilities remain experimental and off by default.

**Configuration is now nested YAML** (`yaml-cpp`), replacing the legacy `key value` `.cfg`
format. Settings are grouped into sections – `run` / `time` / `io` / `output` / `boundaries`
/ `transmissivity` / `evaporation` / `surface_water` / `solver` / `parallel` / `dev` – see the
annotated `config.yaml`. Grid geometry is derived from the input's GDAL geotransform (the
`grid` block is deprecated). Solver and numerics choices are now config keys. Seventeen
settings that were previously reachable only through a `-wtm_*` flag are now config-owned and the
flags are **removed**; the flags that remain still override the config when passed. The
regression suite is fully migrated to the new format and is **green end-to-end** (34/34 suites, all
configs built through `tests/emit_config.sh`).

### Known limitations

Four limits measured during this work. All are properties of the model as it stands rather than
defects to be worked around, and the second and third can invalidate a naive dt-refinement argument.

- **Where lakes drain to the sea, the answer depends on an arbitrary choice when two outlets are level.**
  FillSpillMerge fills a depression and sends the surplus out through the depression's lowest outlet.
  When two or more outlets sit at the same elevation, one is picked arbitrarily. That is a deliberate
  choice in the depression hierarchy, and its own source says so: *"If a depression has more than one
  outlet at the same level one of them is arbitrarily chosen; hopefully this happens only rarely in
  natural environments."*

  On real terrain exact ties are rare, which is the assumption the method is built on. On FLAT terrain
  they are universal, and then the arbitrary choice decides the whole answer. Measured on a test island
  with a perfectly flat plateau and a pit in the exact centre, a case symmetric to machine precision in
  every input on both axes:

  | | left-right | up-down |
  |---|---|---|
  | groundwater only | **0.000 m** | **0.000 m** |
  | with lake routing on | 8.43 m | 9.04 m |

  The groundwater solve is exactly symmetric. The lake routing is not, because every cell around the
  plateau's rim is a tied outlet and one of them wins. Give the same terrain a slope of 1 cm per cell so
  that exactly one outlet is lowest, and the preference disappears completely: two mirror-image runs
  then agree to **1.8e-08 m**.

  **What this means in practice.** Two consequences, and neither is a reason to distrust a normal run:

  1. On terrain with real relief, outlets rarely tie and this does not arise.
  2. Where it does arise (a flat plateau, a plain, a synthetic test surface) the water still balances
     exactly and the total is right. What moves is WHICH way it leaves, and therefore which cells near
     the outlet end up wet. Refining the time step does not help, because the choice is not a
     time-stepping error: the finest run made it slightly worse, not better.

  If you need a deterministic outlet on flat ground, add a small gradient to the surface. `tests/fsm_exit_path`
  demonstrates the whole thing, including the tilted case that removes it.

- **With FillSpillMerge on, first-order operator splitting caps the whole scheme at order 1**,
  regardless of integrator. Measured by solution convergence at fixed model time: TR-BDF2 gives
  ~2.0 with FSM off and **1.04 / 1.00** with it on. Refining `deltat` therefore buys first-order
  accuracy in any production configuration, whatever the integrator claims.

- **The adaptive controller's error estimate is invalid on non-TR integrators when FSM is on.** A
  history-based estimate must difference two states, and in an operator-split step every such pair
  straddles the FSM handoff, so the jump – which does not shrink with `dt` – lands in the estimate.
  Measured order of the estimate itself, over a 64× refinement: **0.80, −0.37, 0.49** – no order at all,
  and one rung negative, so the estimate does not even move monotonically with `dt`. (This entry
  previously read "0.00, constant in `dt`". That number had never been taken: the test's probe read the
  first traced step, which for a three-level scheme carries no history, so it differenced 0/0 and
  asserted an empty result. Corrected 2026-09-06 along with the test.) TR-BDF2 is immune structurally,
  its estimate being embedded within a single step, and measures 2.00. Pinned as an expected failure in
  `tests/estimator_order`, now as a bound on |p| rather than a target, so it fails loudly the day the
  estimator acquires an order.
  **Practical consequence: prefer `solver.time_integration: tr-bdf2` when using `solver.adaptive_dt`.**

- **The default surface-water enforcement is validated only at small scale.** Every result supporting
  `collection.method: active_set` as the default comes from fixtures of 8775 cells or fewer.

### Changed

- **BREAKING – `solver.time_step.mode: adaptive` now REQUIRES `solver.time_step.dt_min`.** A config
  that states the adaptive mode and omits the floor is refused by name, with the reason and a
  suggested value computed from its own `dt`. The key previously carried a default of `1e-5 × dt`,
  which has been **removed**.

  The change is about who owns a decision, not about a number. Every code surveyed does one of two
  things and neither is to guess on the user's behalf: **SUNDIALS and PETSc** ship no floor at all,
  letting the controller shrink without bound; **MODFLOW 6 and ParFlow** require the user to state
  one as part of opting into adaptive stepping - MODFLOW's `DTMIN` lives inside the ATS package you
  add, ParFlow's bounds inside the `TimeStep.Type` you select. WTM now follows the second.

  What settled it is that the floor is **not the safety mechanism it resembles**. An unconditional
  roundoff guard - independent of this key and impossible to switch off - already aborts any run
  whose step has collapsed to where `t + dt == t`, with Hairer's `dop853`/`radau5` and ParFlow as
  precedent; `max_retries` bounds the reject loop besides. What the floor actually decides is how
  much **accuracy a run may lose in silence**, because a step clamped at the floor runs looser than
  the `error_tol` that was requested. That is a policy, and a guessed default made the user's policy
  choice for them and then hid it.

  `dt_min: "0s"` is a first-class choice rather than an opt-out: it disables the floor - the
  SUNDIALS/PETSc behaviour - so a collapsing step aborts legibly instead of being clamped and carried
  on. Prefer it if you would rather a run die than quietly return an answer at an accuracy you did
  not ask for. When a floor does bind, the run says so in ordinary output, naming the number of
  clamped steps and that they ran looser than `error_tol`.

  **The requirement attaches to the explicit key, not to the resolved mode**, and the distinction
  matters: `adaptive` is what an ordinary run resolves to when `mode` is omitted, so requiring it
  there would refuse the simplest possible config. A resolved adaptive run therefore gets the
  SUNDIALS/PETSc behaviour - no floor - while writing the mode down means owning its bounds.

  **To migrate:** add `dt_min: "<seconds>s"` beside any `mode: adaptive` you have written. `1e-5 × dt`
  reproduces the old behaviour exactly and is MODFLOW 6's own recommendation; `"0s"` disables the
  floor. The refusal message computes the first of these for your config and prints it.

  Everything in the tree that states the mode was migrated with it - 27 suite configs, the annotated
  `config.yaml`, `examples/island_equilibrium/demo.py` and `benchmark/scheme_bench` - all at
  `1e-5 × dt`, so **no result moved**: `tests/golden` passes all 35 runs against unchanged
  references. Two suites derive the floor per arm rather than fixing it, because a constant floor
  would bind at the fine end of a `dt` ladder and bend what is being measured - `tests/estimator_order`
  still observes order `p = 1.99 2.00 2.00`.

- **BREAKING – the per-solve convergence test is now judged in WATER VOLUME, not head.** New
  default `solver.convergence.metric: volume` (`head` is the off-switch). The step that ends a solve is
  measured as |S·Δwtd| -- water volume per unit area, so a depth in metres, not m³ -- rather than |Δh|, so all three "close enough" gates - this one,
  `run.equilibrium_stop`, and `solver.time_step.error_tol` - finally speak one language. It changes
  the answer, and it changes it toward the converged one.

  PETSc's `-snes_stol`, which WTM set itself at 1e-8, is a STEP-SIZE test: it stops when the iterate
  stops moving, not when the residual is small. It is a stagnation detector, and WTM was treating
  `CONVERGED_SNORM_RELATIVE` as success. Measured on `tests/budget_closure`, **259 of 294 solves -
  88 % - exited that way** at `error_tol` 0.005, and 38 of 40 - 95 % - at 0.05. The offenders are
  well-warm-started steps, where the update is already tiny at the first iteration and the step test
  trips before the residual is ever driven down. With everything else in this release in place, the
  step-test exits reduce the residual by a median 7.40e-08 and a worst 5.20e-07, against 9.46e-09 and
  9.49e-09 for the solves that exit on the residual test: looser by roughly one decade in the median
  and by 55x at worst.

  The units are the crux. A head step tolerance is a LENGTH and the water budget it has to agree with
  is a VOLUME; on this fixture `stol` 1e-8 ends the solve once the update falls below ~2.4e-06 m of
  head, and a few microns of head over cells of ~1e8 m² is a great deal of water. The exact budget is
  an algebraic consequence of the discrete equations being satisfied, so those steps left real
  unbalanced mass and the ledger reported it faithfully: the ledger was right, the solve was not.
  Measured head against water on the same fixture and binary, `error_tol` 0.05: the worst per-step
  budget residual improves from 2.88e-07 of recharge to 4.56e-08, a factor of 6.3, at **identical wall
  time** (1.11 s both ways, 20 solves both ways). It also stops a few very deep cells, where a metre of
  head is very little water, from speaking for the whole grid.

  Three golden references moved, but only ONE of them was actually stale. Measuring the previously
  committed references against a set re-derived three decades tighter: `fsm_runoff_hi` was off by
  **1.72e-01 m** and genuinely needed replacing - it is the case the controller defect below had been
  aborting - while `fsm_runoff` (1.40e-10 m) and `fsm_impulse` (5.90e-08 m) were already converged and
  their new values differ only in the last digits, well inside the 1e-6 check tolerance. The four
  unchanged references were already exact. The new default reproduces the tight reference on all seven
  cases, and so does the old head metric once the controller defect is fixed - a convergence criterion
  decides WHEN a solve stops, not WHERE it converges, which is what `tests/solver_consistency` asserts.

  One consequence worth knowing before tightening anything: `solver.time_step.error_tol: 0.005`, the
  tightest value the adaptive benchmarks used, is **not actually attainable** on the budget fixture.
  It only ever appeared to be, because the solves were stopping early and reporting a small error
  estimate. Judged honestly the controller correctly refuses and the run aborts on max retries.

- **BREAKING – FillSpillMerge's water now reaches the groundwater as a CONTINUOUS SOURCE, not an
  IMPULSE.** New key `surface_water.fsm_coupling: continuous | impulse`, defaulting to `continuous`.
  Under the old behaviour (`impulse`, still selectable) FSM's post-routing water table replaced the step
  baseline, so the solver's state jumped at every step boundary. Under `continuous` the table is left
  alone and FSM's per-cell volume change feeds the next step's recharge instead.

  The reason is physical rather than numerical. FSM is instantaneous by construction, so under `impulse`
  the state always carries its fully equilibrated lake - a depression is full from the instant there is
  water to fill it, and evaporates at the open-water rate for the whole step. Refining the step does not
  soften that; it re-equilibrates more often. Real water flows in over the interval and evaporates as it
  arrives. Measured on `tests/fsm_consistency` at a fixed 8 yr, cumulative evaporation converges to
  ~8.55e09 m³ under `impulse` against ~7.65e09 under `continuous`: the old default **over-exposed surface
  water to open-water evaporation by ~11 %**, and the gap GREW under refinement rather than vanishing.

  What did *not* decide it, since both were checked: `continuous` does **not** restore second order
  (1.16/1.25/1.60 against 1.13/1.23/1.59 - the first-order splitting cap survives both couplings), and its
  flicker benefit is already spent by `collection.method: active_set`, which is the default collector.
  Both couplings reach the same equilibrium (20.08 % apart at 8 yr, 0.30 % at 400 yr), so this matters for
  **transients** far more than for equilibrium runs.

  Two consequences to know about. `continuous` with `collection.method: explicit` is **refused** - that
  pair does not converge (observed order goes negative under refinement; the post-solve clamp and the
  source term fight). And cross-rank drift now **compounds** rather than being reset each step: the
  `impulse` path overwrote every rank's state from rank 0 every step, which silently wiped accumulated
  divergence. That means the old default's tight cross-rank agreement was partly an artefact of a
  broadcast rather than evidence the parallel solve agrees. `tests/xrank_growth` measures the regime
  directly, and `tests/golden`'s `fsm_runoff` arm carries a documented 1e-5 tolerance as a result.

  Every FSM-on golden reference moved. A `fsm_impulse` golden arm keeps the non-default coupling covered.


- **BREAKING – an unrecognised YAML key now ABORTS the run.** Previously a key nobody read was simply
  never seen: a typo, a key retired by the schema migration, and a setting a user believed was in force
  all behaved identically to not writing it, and the run reported success. The abort names every
  offending key with its full dotted path, lists the valid keys for that section, and suggests the
  nearest match:

      config file 'x.yaml' has 1 unrecognised key:
        unknown key 'time.detlat'  -- did you mean 'time.deltat'?
            known keys in 'time': deltat, report_interval, save_every_n_reports, total

  **If you have an existing config with a stale or misspelled key, it will now stop rather than quietly
  ignore it.** That is the point: the cost of the old behaviour was not a lost setting but a lost
  NEGATIVE RESULT – a parameter sweep over a key nothing reads returns "no effect" for a reason that has
  nothing to do with the model, and reads exactly like a finding.

- **BREAKING – a `-wtm_*` flag that nothing consumed now ABORTS the run.** PETSc's options database
  accepts any string, so a misspelled flag, a retired flag, or a flag whose parse site sits on a code
  path the run did not take had no effect and said nothing. Checked once, after the first cycle
  (`update()` re-reads options every cycle, so at init many legitimately-passed flags have not been
  queried yet). PETSc's own options (`-snes_`, `-ksp_`, `-pc_`, `-mat_`) are deliberately not policed –
  they are legitimately unused depending on the solver path.


- **DEFAULT surface-water enforcement is now `active_set`** (`surface_water.collection.method`),
  replacing `implicit`. The semismooth exfiltration constraint is solved *inside* the residual rather
  than approximated by an in-residual siphon. Why:
  - **`implicit` is dt-DEPENDENT.** Its removal rate is `max(0,wtd)/dt`, so the retained head is
    ~linear in `dt` – measured with FSM off, isolating the face: **1.97 / 0.68 / 0.34 m** at
    `dt` = 1, 1/3, 1/6 week. With FSM on, that dt-dependent excess is what FillSpillMerge routes, so
    **lake depth inherits it** (5.38 / 2.50 / 2.02 m), and on a multi-lake fixture the lake *count*
    itself moves with `dt` (6 → 5). Under `active_set` the same lakes hold **5.6986 m at every `dt`**.
  - **It eliminates the between-step FSM shock.** FSM was undoing essentially the whole groundwater
    step every cycle: shock ratio **0.985 → 3.6e-13**. The solve now arrives at a state FSM agrees with.
  - **It is cheaper**, 2–100× across solvers (island, cold start, matched precision): Anderson
    2869 → 1364 SNES iterations, TR-BDF2 1771 → 957, TR-BDF2+adaptive 3769 → 957 (adaptive stops
    subdividing and becomes identical to fixed-`dt`), Newton+continuation 261203/412.5 s → 2478/5.3 s.
  - **All eight schemes now agree exactly** on the resulting water table, where previously they did not.
- **The default is solver-dependent – and only Picard falls back.** The active-set pin lives in the
  matrix-free residual, which is the Anderson path *and* Newton's, since Newton differentiates that
  same function and now carries the matching semismooth tangent. Only the **Picard** operator and RHS
  are a separate formulation with no pin, and selecting active-set there also disables every collector
  removal, so the constraint would be silently unenforced. With the key unset the default therefore
  resolves to `active_set` on **Anderson and Newton**, and to `explicit` on **Picard**, with a NOTE.
  (An earlier draft of this entry said Picard *and Newton* downgrade; the downgrade is conditioned on
  `use_picard` alone. `tests/budget_closure` now asserts which collector each solver resolves to, from
  the log rather than by inference.) An explicit choice is always honoured, with a warning on the
  solver that cannot enforce it consistently.
- `runoff_collector` accepts a new `active_set` value, and it is the DEFAULT. Both flag spellings
  (`-wtm_active_set`, `-wtm_dev_active_set`) were retired before release and now abort by name.
- **Golden references regenerated** for the surface-water cases. `below_ground` (no surface water) is
  unchanged, as it must be. Changes are ≤1 m except the `transient` case, where 52 cells move >1 m
  (max 21.7 m) at cells previously held near the surface by the siphon and now free to drain – the
  known collector divergence at rim cells, now resolved to one enforcement-independent answer. The
  `transient` golden tolerance is loosened 1e-6 → 1e-5 m: under active-set that case reproduces across
  rank counts only to ~2e-6 m (micrometre round-off from the active set differing in its last bits
  between decompositions), where `implicit` reproduced below 1e-6.

### Fixed

- **The default land boundary leaked mass: its off-map flux was never booked.** Under
  `boundaries.land: neumann_toposlope` the ghost head is set to `h_ghost = h_edge + (topo_edge −
  topo_inland)`, which is zero flux relative to the *land surface*, not zero Darcy flux. Wherever the
  terrain rises away from the domain edge, that ghost sits above the edge head and drives water **in**.
  The solve used this flux, correctly, and the water budget then ignored it, so the arriving water
  appeared from nowhere. On `tests/ghost_boundary`'s coastal fixture the unaccounted inflow ran to
  **44.5 times the recharge**, and the exact budget residual fell from `4.4521e+01` to `8.3964e-10`
  once the term was accounted.

  The flux is now accumulated per step, with the TR-BDF2 stage weights, and enters the exact budget as
  a **source** rather than as part of the ocean outflow: water arriving from off-map upslope is not
  ocean outflow, and folding the two together would close the ledger while describing the wrong
  physics. Both controls are bit-unchanged - `dirichlet_sea_level` and flat terrain each book exactly
  zero - so no existing answer moves.

  This hid for the whole life of the test suite for one reason: **every other fixture is
  ocean-ringed**. With no land edge there is no off-map flux, the term is a structural zero, and every
  budget check was true and empty. `tests/edge_composition.py` now reports, for each fixture, which
  boundary condition its mask actually exercises, so a blind spot of this shape is visible rather than
  inferred.

- **New run-log column, `boundary_inflow_gw`** (column 26, appended), reporting that off-map ghost
  flux, signed `+` for inflow. It had existed only inside `exact_budget_residual`, a sum it shares
  with five other terms, where two errors can cancel and read as a closed budget - and where the
  term's null cases cannot be asserted at all, since a test cannot say "this must be exactly zero on
  flat ground" about a quantity the model never prints. Reporting it also recomputes the fix by a
  second, independent route: the budget *gap* before the fix and the term the model now *books* agree
  to five significant figures (`4.4521e+01` both ways).

- **The adaptive step controller could shrink `dt` without bound, and abort, chasing an error that
  `dt` cannot reduce.** The embedded estimate is split into an integrator part and an FSM-coupling
  part, and these were combined by taking the larger. Only one of them answers to `dt`: the
  integrator error is O(dt²), while the coupling deviation comes from FillSpillMerge's delta, which
  is delivered whole regardless of step size and is therefore O(1) in `dt`. Measured on
  `tests/golden` `fsm_runoff_hi`, the estimate sat at 0.6043593790 while `dt` was driven from
  4.4e+07 s down to 4.1e-02 s - **nine orders of magnitude** - moving only in the ninth significant
  figure. Handing that to the reject test asks the controller to fix by shrinking something shrinking
  cannot fix, so `dt` collapses to `max_retries` and the run aborts. Not a conservative choice, an
  unsatisfiable one.

  A controller may only steer on error it can control, so the accept/reject decision and the PI
  shrink now read the integrator part alone. The coupling part may **withhold growth** - all it can
  honestly say is "FillSpillMerge is moving a lot of water here, do not get greedy" - but it can no
  longer force a shrink. `output.trace: [dt]` now reports both parts (`eint=`, `ecpl=`) so the split
  is visible rather than inferred. This was aborting three suites outright, all now passing: the
  `fsm_runoff_hi` golden at every rank count, `budget_closure`'s TR-BDF2 adaptive arm, and
  `xrank_growth`'s continuous arm.

- **The adaptive-restart phase declared convergence on a head step.** `AdaptiveRestartTest` compared
  PETSc's head `snorm` against `ar_stol`, and that path was also the one place the volume-weighted
  test was deliberately not registered, so nothing about `-wtm_adaptive_restart` was judged in water.
  It could call a phase converged once the iterate stopped moving in head, which on a warm start
  happens well before the water it still owes has been driven out. The water-step computation is now
  shared between the two convergence tests. The tolerance is unchanged; only the metric moved.

- **The water budget's baseline was one cycle late, which on a cold start was most of the reported
  residual.** `stored_volume_initial` was captured on the first `PrintValues` call, and `PrintValues`
  runs at the *end* of a cycle, so the baseline was the state after cycle 0 had already stepped. Every
  flux accumulator - recharge, evaporation, Darcy ocean outflow, FSM spill - starts *at* cycle 0. The
  closure therefore differenced a storage change over cycles 1..N against fluxes over cycles 0..N, and
  the first cycle's storage change was absent from the books. Measured on `tests/fsm_consistency`,
  120 yr, `active_set` with FillSpillMerge: the whole-run gap was 34.84 % of recharge, and the same run
  stopped after one cycle - where `d_stored` is zero by construction - reproduced it almost exactly
  (−7.261e10 against −7.264e10). The supplied initial table drains and FSM spills 6.69e10 m³ to the
  ocean in year one; none of the storage drop feeding it was counted. The baseline is now taken at the
  end of `initialise()`, before any stepping, and both it and every later report call one
  `ComputeStoredVolume`, so the two cannot drift apart. `PrintValues` aborts if the baseline was never
  captured rather than falling back to zero, which would silently turn `d_stored` into the absolute
  volume. Result: **34.841 % → 0.040 %** for the overwrite coupling and **2.739 % → 0.014 %** for
  `fsm_delta_source`, the former being precisely the "≈0.04 % once spun up" `WATER_BUDGET.md` had
  always predicted. Water tables are untouched: `stored_volume` and the exact residual (column 17) are
  identical to every printed digit, so this moves the diagnostic and not the physics.

- **Columns 9 and 19 counted internal water as external input under `fsm_delta_source`.** `rech_dist`
  served two roles at once: the solve's source term, and the quantity booked as
  `total_recharge_direct`, which `WATER_BUDGET.md` defines as the water entering the domain. Folding
  FillSpillMerge's per-cell delta into it therefore booked redistribution as input. Cumulative column 9
  ran to −6.34e10 by cycle 1 - a negative cumulative external input - and `ocean_loss_closing` and
  column 16 are built on top of it. The delta now has its own carrier, so the solve reads
  `rech_dist + fsm_delta_dist` while the booking reads `rech_dist` alone. Conservation is untouched by
  construction: the exact budget reads `rech_vec`, the full source term, which is the definition the
  scheme's own conservation law requires once FillSpillMerge's water arrives *during* a step. Column 19
  now agrees between the two couplings to all twelve printed digits. This also supersedes the
  "≈18 % is a definitional mismatch" note in `WATER_BUDGET.md`: most of that was this defect, and what
  remains on a fixture that ships is 2.7e-02.

- **BREAKING – an invalid config ENUM VALUE now aborts.** The schema check validated config KEYS; it did
  not validate their VALUES, which left the same defect one level down. `solver.method: pickard` fell
  through the bridge's `if/else` chain to the DEFAULT and the run reported success – so a sweep over a
  misspelled solver silently compared Anderson with Anderson, and `solver.method: newtno` silently lost
  the dt-continuation the correct spelling now implies. Five keys behaved this way: `solver.method`,
  `solver.time_integration`, `solver.storage`, `boundaries.land` and `run.equilibrium_stop.metric`. Five
  others (`run.type`, `surface_water.mode`, `collection.method`, `output.verbosity`, `output.if_exists`)
  already validated, so the message form follows theirs:

      config: solver.method must be anderson | picard | newton, got 'pickard'

  `solver.storage` gains an explicit name for its second value, `secant` – the codebase's own term for
  the S·Δh form – which was previously reachable only by writing something the model did not recognise.
  The three retired `eq_metric` spellings (`water`, `water-max`, `water-rms`) remain accepted and keep
  self-announcing. Covered by ENUM-BAD and ENUM-OK arms in `tests/config_schema`; both were shown to
  fail, ENUM-BAD by removing a validator and ENUM-OK by dropping a legal value from one.

- **`solver.method: newton` was a documented config value that crashed.** Newton does not converge from
  a cold start without dt-continuation, and at the time `-wtm_dt_continuation` had no config expression – so
  selecting Newton from YAML alone aborted with `DIVERGED_LINE_SEARCH` after 4 iterations. The config
  value now means the *working recipe*: `solver.method: newton` implies
  `solver.dt_continuation: true` and is byte-identical to it (the flag has since been retired too).
  `solver.dt_continuation: false` opts
  out – legitimate for a warm finish, where continuation is wasted – and warns that a cold start will
  diverge. The bare `-wtm_newton` flag is **unchanged** and still means plain Newton: `tests/newton_solver`
  pins a contract that it does not converge, `benchmark/scheme_bench` measures it, and
  `EQUILIBRIUM_ROBUSTNESS.md` documents it as the thing that needs the recipe. The flags remain the
  primitive layer and the config key is the abstraction over them, as `collection.method: legacy`
  already is for the `-wtm_` surface flags.

- **A stray NUL byte made `src/CreateSNES.cpp` invisible to every text search.** A scripted edit had
  written a raw NUL where the source should read `'\0'`, so the file was `data` rather than text: `grep`,
  `file` and every code-search tool skipped it silently. It compiled and ran correctly, which is why it
  survived – but the file holds **16 of the model's 32 option-parse sites** (exactly half), including the whole adaptive
  step-size controller (`dtc_*`, `dt_norm_*`) and the Anderson restart/handoff machinery, so an audit of
  "which flags does the model read?" came back missing 26 of 48 flags with no indication anything had been
  skipped. One byte; the fix is `'\0'`.

- **BREAKING – `dev.active_set` is removed; it silently overrode an explicit
  `surface_water.collection.method`.** The active-set exfiltration enforcement was reachable from two
  YAML keys at once, and the developer key won without saying so: a config asking for
  `collection.method: explicit` while also setting `dev: {active_set: true}` ran **active_set**. Measured
  on `tests/fsm_consistency` (256 cells, 5 yr, Anderson + TR-BDF2), the two configs differ on **54 of 256
  cells, max 0.127 m, rms 0.0106 m** – with no NOTE, no WARNING, and nothing in the log to distinguish
  them. This is the same dual-channel hazard as the flag-versus-config ambiguity the nested-YAML
  migration set out to remove, but living entirely inside the config. One setting, one key: select the
  enforcement with `surface_water.collection.method: active_set`, which is the documented route and
  already the default. An existing config carrying the old key now aborts and names it, rather than
  drifting. Pinned by the `RETIRED` arm of `tests/config_schema`.

- **Every run now states which surface-water enforcement it used.** The resolved
  `collection.method` – the single most consequential surface-water choice, since it moves the
  equilibrium head and through FSM the lake count – was written only to the coverage file, so an
  ordinary run's output could not say which boundary condition produced it. It is now announced once per
  run, with its source named, because `active_set [default]` and `active_set [config]` are different runs
  to anyone auditing a result later:

      surface-water exfiltration enforcement: explicit  [surface_water.collection.method]

- **`config.yaml` shipped the wrong collector and an incomplete list.** The reference config a new user
  copies carried `method: implicit` – the *former* default, and the one enforcement measured to leave a
  spurious dt-dependence – while the actual default, `active_set`, appeared neither as the value nor in
  the commented enumeration (nor did `extended_soil`). Distinct from a merely undocumented key: the
  reference was present and steering users to the wrong value. Now ships `active_set` with all six modes
  described.

- **`-wtm_extended_soil` was defeated by a second mechanism wired to the same flag.** A post-solve
  surface-truncation experiment added later keyed off `g_extended_soil` and clamped the above-surface
  mound back to the surface every GW step – reinstating exactly the `wtd = 0` free boundary that
  extended soil exists to remove. The flag therefore printed its mode banner while doing the opposite of
  what it claimed. Measured on the design note's own harness (`benchmark/picard/recharge_free_boundary.py`,
  arm E), order in dt: **1.22 / 1.08 / 1.02** with the collision, **2.07 / 2.01 / 2.00** without, against
  the recorded 2.07/2.07/2.00. Extended soil is now a member of `surface_water.collection.method`, so
  "extended soil AND a collector" – the contradictory request underneath this – is unrepresentable
  rather than merely detected. It remains `[WIP]` and nonphysical: its production half (truncating the
  mound at the FSM handoff rather than per step) is still unimplemented.

- **The step-size controller's own knobs were unreachable on the adaptive path.** `-wtm_dtc_grow`,
  `-wtm_dtc_shrink`, `-wtm_dtc_dt_max` and `-wtm_dtc_easy_iters` were parsed only inside
  `if (use_newton_continuation)`, yet the adaptive controller reads all four on every step. On a plain
  `-wtm_dt_adaptive` run PETSc accepted them and nothing consumed them, so the controller silently ran
  on compiled-in defaults. Effect once live, same fixture: `-wtm_dtc_easy_iters` 0 / 4 / 8 / 100000
  gives 229506 (still running) / 2464 / 57 / 57 steps, where all four previously gave 57.

- **`benchmark/picard/recharge_free_boundary.py` had not run since two migrations.**
  `BDF2_RECHARGE_ORDER.md` §15 tells the reader to reproduce its result with that script; the script had
  been broken for weeks by the nested-YAML config change and by #124 making the GDAL geotransform
  authoritative (its fixture wrote a pixel-space transform, giving latitudes past 90° and negative cell
  areas). Neither failed loudly, because the harness swallowed the model's exit status. Repaired, and it
  now reproduces §15 to within **0.002031 mm** against the recorded **0.0019 mm**. Roughly 25 other
  scripts under `benchmark/` share this breakage and are **not** yet repaired.


- **The runoff-ratio share was routed by solve count, not by elapsed time.** It was handed to
  FillSpillMerge at full *nominal* step size on every accepted sub-step and never scaled, so under
  adaptive `dt` or Newton continuation the model routed the wrong amount of water – a **mass** error,
  not a reporting one. Measured at `runoff_ratio 0.3`: 1.36148e10 over 20 solves (fixed `dt`) against
  9.53038e09 over 14 (adaptive), a ratio of 0.700 against a solve-count ratio of 0.700, producing a
  **6.6 % divergence in stored volume** against 0.16 % with the channel off. Fixed by making the routed
  channel *lazy* like the direct one: the nominal depth is held, and both delivery and booking happen
  at the handoff scaled by the accepted step's `dt`. Column 20 is now exactly invariant to solve count
  (0.000e+00) and the stored-volume spread falls to 7.293e-04. Fixed-`dt` runs are unaffected (the
  scale is exactly 1; goldens unchanged).
- **Neither post-solve collector closed the exact budget.** `explicit` and `legacy` reported residuals
  of 8 to 9.3 *times* recharge, on every solver, because `accumulate_budget_terms` read the storage
  from the **pre-clamp** `w^{n+1}` while the commit loop stored the post-clamp value – so the budget
  described a state the model does not carry forward, and the residual came out at exactly
  `−total_surface_removed`. No water was ever lost; this was a defect in the conservation *check*. It
  mattered because `explicit` is what **Picard** resolves to when the collection method is unset.
  Correcting the storage term to the committed state closes all four:
  explicit −9.165e+00 → −1.670e-07 (Anderson), −9.331e+00 → −3.328e-10 (Picard); legacy −7.998e+00 →
  −8.461e-08 and −8.865e+00 → −3.289e-10. Every collector that already closed is unchanged.
- **The water budget's "water in" term counted only one of the two input channels.**
  Precipitation-less-evaporation is split by the runoff ratio the moment it is computed
  (`runoff = rratio*rech_dist; rech_dist -= runoff`), and `total_recharge_added` summed only what was
  left. The routed share was therefore never counted as an input, while the lakes FillSpillMerge
  builds from it *did* appear in `stored_volume`, so `ocean_loss_closing` and `budget_residual` could
  not close whenever `runoff_ratio > 0`. Measured: column 9 scaled as exactly `(1 − runoff_ratio)`.
  The same accumulator also booked the **unscaled** `rech_dist` while the solver integrates
  `rech_dist * rech_dt_scale`, so under sub-stepping it tracked the *solve count* rather than elapsed
  time (2.55× inflation under adaptive `dt`). Both are fixed by reporting the channels for what they
  are: new **column 19 `recharge_direct`** and **column 20 `runoff_to_surface`**, with column 9 now
  their sum – which is what its name always promised. Columns are **appended**, so every existing
  index is unchanged. Verified: column 9 is now identical at 20 solves (fixed `dt`) and 15 (adaptive),
  and at `runoff_ratio = 1.0` the direct channel is exactly zero.
- **Adaptive `dt` sized the next step before the current one had been accounted.** The controller
  wrote its newly-sized `dt` straight into `user_context.deltat`, but five consumers downstream are
  still accounting the step just *taken* and read it as that step's: the BDF2 history ratio
  `ω = Δt/Δt_{n-1}`, the taper-1 sink and taper-2/3 evaporation removal depths, the land→ocean flux
  accumulation, and TR-BDF2's step-flux quadrature. Exact residual as a fraction of recharge:
  **−1.603 → −2.100e-07** for TR-BDF2 + adaptive and **−0.417 → −8.139e-08** for BDF2-on-V +
  adaptive, the former now identical to its fixed-`dt` value to the last digit. Sizing is now the
  last thing `update()` does; the reject path is unchanged, since it returns before any accounting
  runs. This is not only a diagnostic fix – `bdf2_prev_dt` feeds the BDF2 time ratio, so
  adaptive + BDF2 *results* move. Nothing had ever checked the water budget under adaptive `dt`,
  which is how it survived; `tests/budget_closure` now carries an adaptive arm per integrator.
- **TR-BDF2 was losing water through the active-set exfiltration transfer.** The multiplier the
  constraint hands to FillSpillMerge was read off whichever residual evaluation ran last, which under
  a two-stage scheme is stage 2. Stage 2 carries only `C3 = 29.29%` of the step and knows nothing of
  what stage 1 shed, so the step's exfiltration was understated by `1/C3 = 3.4142` and that water was
  neither delivered nor accounted. Measured on `tests/multilake` (`active_set`, `dt = 0.25 yr`):
  **5.97e11 m³ delivered against backward Euler's 2.00e12 m³**, and a physical budget residual of
  **9.5% of recharge** where BDF2-on-V – also multi-level, also second order – closes at 0.2%. The
  step multiplier is `E = C1·E1 + E2`; both stage multipliers are now captured by an explicit
  post-solve residual evaluation at the *accepted* state, which for stage 1 is required rather than
  merely tidy.
- **TR-BDF2's flux and removal budget terms were accumulated with backward-Euler weighting.** The
  land→ocean Darcy flux, the taper-2/3 evaporation, and the taper-1 sink were each evaluated once at
  `w^{n+1}` over the full step. Under TR-BDF2 every such term is a three-point quadrature over
  `(w^n, Y_γ, w^{n+1})` with weights `(C1γ/2, C1γ/2, C3)`. Evaporation dominated in practice – it
  carries 91% of the water leaving the domain on `tests/multilake` – and the taper-1 sink drove the
  exact residual to **191% of recharge** under the `implicit` collector, against backward Euler's
  6.5e-09 on the same arm. The sink's quadrature depth is also what is now handed to FSM, so the
  aquifer and the surface agree on the amount transferred.

### Added

- **`tests/route_equality` – a config key and the flag it abstracts must produce the same run.** All
  eight `ABSTRACTED` flags are now asserted **byte-identical** between their two routes, plus a positive
  control that `solver.method: newton` runs from YAML alone. The claim that the config expresses what a
  flag does had never been tested: the suite covered every mechanism but never the *equivalence of the
  two routes to it*, which is precisely where this repo's config defects have lived (`dev.active_set`
  overriding an explicit method; `-wtm_extended_soil` and the post-solve truncation masking each other).
  Subsumes the redundant triplicated budget-closure check across three active-set arms, which
  re-established closure three times while never asserting the property actually at stake – that the
  three routes agree.


- **`-wtm_dt_trace`** – reports `(dt, est, tol, factor, iters, accepted)` for every adaptive step,
  accepted or rejected, in a machine-readable line. The local-error estimate steers the whole
  integration and was previously computed each step and reported nowhere, so nothing could see whether
  it responded to `dt` at all. Off by default.


- **TR-BDF2 now has an exact per-step water budget** (`exact_budget_residual`, column 17), where it
  previously reported `nan` on the grounds that two stages have no single-step identity. They do:
  `C1·(stage 1) + (stage 2)` telescopes, because `C1 − C2 = 1` and `C1γ + C3 = 1` exactly. Storage
  and recharge come out as the backward-Euler forms unchanged; only the flux, removal and
  exfiltration terms differ. Column 17 now closes at **2.2e-8 of recharge** under TR-BDF2 against
  backward Euler's 3.1e-8, tolerance-limited at `-snes_stol 1e-8`. This closes the gap where second
  order and verifiable conservation were mutually exclusive – TR-BDF2 is the integrator the adaptive
  controller drives, and it had been the one scheme whose conservation nothing could check.
  - `src/tr_bdf2_coefficients.hpp` – the derivation, with the stage coefficients and the step
    quadrature weights named separately (the duplication of bare locals is how a stage weight came to
    stand in for a step weight in the first place).
  - `src/test_tr_bdf2_balance.cpp` – 7 unit cases pinning every identity the derivation rests on,
    including the second-order condition `W_YGAMMA·γ + W_NEW = 1/2` and the order barrier above it.
  - `tests/budget_closure` – plain TR-BDF2 and TR-BDF2 + active-set arms, replacing the assertion
    that TR-BDF2 reports `nan`. Shown to bite three ways: against a pre-fix binary, and against two
    weight injections that reproduce the original defects numerically.

#### Configuration (nested YAML)
- **Nested-YAML config** (`yaml-cpp`), replacing `key value` `.cfg`. Key moves from the old flat keys:
  `total_time`→`time.total`, `supplied_wt`→`run.initial_water_table` (`saturated` | a starting-WT path),
  `save_nreport_interval`→`time.save_every_n_reports`, `physics.fdepth`→`transmissivity.fdepth`,
  `physics.infiltration`→`surface_water.infiltration_during_flow`, `surface_water.fsm`→`surface_water.mode`
  (`routed` | `ponded` | `removed`), `runoff_collector`→`surface_water.collection.method`, `surfdatadir`→
  `io.source`, `outfile_prefix`/`textfilename`→`output.*`. `physics.evaporation.mode` dropped (vestigial once
  the ET sigmoid – the default – is on). `grid` deprecated (geometry from the GDAL geotransform).
- **Solver / numerics as config keys**, bridged to the existing `-wtm_*`/`-snes_*` options (an explicit CLI
  flag still overrides): `solver.{method, tolerance, max_iterations, time_integration, adaptive_dt, dt_max,
  wtd_step_error_tol, t_bar, storage}`, `run.equilibrium_stop.{tol, metric, frac}`, `boundaries.land`,
  `transmissivity.additive_background_transmissivity`, `evaporation.{et_sigmoid, extinction_depth}`,
  `parallel.threads_per_rank`, `dev.*`. The resolved SNES tolerances are logged at start-up.
- **`surface_water.runoff_ratio`** accepts a uniform number in `[0, 1]`, the string `raster`, or omission (off).
- **`output` run directories** – each run writes to `output.directory/run<NNN>_<timestamp>/` (never
  clobbered; `output.if_exists`: `increment` | `overwrite` | `error`), with a `latest` symlink and an
  auto-written `provenance.yaml` (git commit/state, PETSc version, command line, host / time / MPI ranks).
- **`output.verbosity`** – `quiet` | `normal` | `verbose` (`verbose` adds the PETSc per-solve monitors).

#### Time stepping and surface–subsurface coupling
- **`report_interval` / `save_nreport_interval` config knobs.** `report_interval` sets the cadence of the
  equilibrium check + log line + raster output – as a step count (`report_interval 100`) or a simulated time
  (`report_interval 50yr` / `1000s`, resolved via `deltat`). `save_nreport_interval` saves a raster every K
  reports. Both **default with a loud warning** if omitted (100 steps / 1). They replace `maxiter` /
  `cycles_to_save` (deprecated), which are no longer coupling intervals now that FillSpillMerge runs every
  timestep (see *Changed*). Per-report `t GW time / FSM time` lines are summed from timers around each GW and
  each FSM step.
- **`surface_water.collection.method: active_set`** – semismooth active-set exfiltration constraint: enforces
  `wtd ≤ 0` as a min-NCP constraint *inside* the matrix-free Anderson solve (`f = max(w_c, f)`), pinning the
  free surface every iteration rather than via a post-solve clamp (`explicit`) or in-residual siphon
  (`implicit`). dt-independent to machine precision, mass-conserving (captured exfiltration → FillSpillMerge), and
  collector-independent with FSM on. Auto-enables `-wtm_volume_storage` (it needs a `b = 0` residual path) and
  supersedes the `runoff_collector` removals. Picard/Newton tangents and a full default decision are future
  work. See `benchmark/FSM_EVERY_STEP_DESIGN.md`.

#### Surface-water routing
- **`runoff_collector` config-file selector** (`implicit` | `explicit` | `off` | `legacy`) unifies how the
  `wtd = 0` exfiltration constraint – where above-surface water is routed to runoff / Fill-Spill-Merge – is enforced.
  `implicit` is the in-residual exfiltration: exact, dt-independent, pins `wtd = 0`, wired into the Anderson
  residual **and** the Picard operator (a frozen active-set diagonal); Newton still warns (its Jacobian needs a
  semismooth/active-set treatment of the discontinuous kink). `explicit` is the post-solve clamp: robust on
  every solver *and* under adaptive-dt, within ~1 cm of `implicit` and converging as `dt → 0`. `off` collects
  nothing (above-surface water piles up – nonphysical, warns; supersedes
  `-wtm_dev_allow_aboveground_water_columns`). `legacy` keeps the old `-wtm_surface_sink` band-sink defaults.
  The modes are mutually exclusive (no hidden clamp backstop under `implicit`, so its misbehaviour stays
  visible). **Default is `implicit`.** Adaptive-dt (`-wtm_dt_adaptive`) handles `implicit` by clamping the
  error-estimate predictor to the feasible set (`wtd ≤ 0`), so its discontinuous exfiltration kink no longer
  spikes the step-size controller (verified: implicit-adaptive == implicit-fixed-cc to ~1 cm). See
  `benchmark/SURFACE_WATER_ROUTING.md`.

#### Boundary conditions
- **Selectable land-edge boundary condition** (`boundaries.land: neumann_toposlope | dirichlet_sea_level`): ocean edges
  are always Dirichlet `h = 0`; land edges default to terrain-following no-flow (`neumann_toposlope`) but can be
  set to sea-level Dirichlet (`dirichlet`), where a land edge behaves exactly as an ocean neighbour (head 0 and
  surface transmissivity via ghost nodes). Wired into all solver paths – the matrix-free residual, the Newton
  analytic Jacobian (FD-verified), and the Picard operator+RHS – and the water budget. The `dirichlet` mode
  reproduces the legacy sea-level padding to machine precision (7e-12 m). Not compatible with `-wtm_kirchhoff`.
  Regression: `tests/boundary_consistency/`. See `benchmark/BOUNDARY_CONDITIONS.md`.

#### Solvers and time integration
- **Semi-implicit Picard groundwater solver** (`-wtm_picard`): builds an SPD operator solved
  with CG + GAMG, as an alternative to the matrix-free default. See `benchmark/picard/PICARD_MATH.md`.
- **Analytic-Jacobian Newton solver** (`-wtm_newton`): a true Newton–Krylov path (GMRES + GAMG) driven
  by an exact, finite-difference-verified Jacobian of the conservative-FV residual – including the
  surface-sink and evaporation / accessibility taper tangents. Paired with **dt-continuation**
  (`-wtm_dt_continuation`): a pseudo-transient ramp that starts `deltat` small, so a far / cold initial
  water table stays inside the Newton basin, and grows it after each converged step until the table
  settles. On this dt-continuation path the convergence-based early stop (`-wtm_eq_tol`) only engages once
  `deltat` has ramped up, so it cannot trip early. The convenience bundle **`-wtm_stiff`** turns on all three
  at once (equivalent to `-wtm_newton -wtm_dt_continuation -wtm_eq_tol 0.01`) for hard equilibrium cold-starts
  on stiff terrain. See `benchmark/EQUILIBRIUM_ROBUSTNESS.md`. (`-wtm_eq_tol` is now a general, all-paths
  equilibrium stop that is on by default – see "Automatic equilibrium stop" below.)
- **Second-order transient time integration.** Fixed-step BDF2 (`-wtm_bdf2`), a volume-form
  variant that is genuinely 2nd order under recharge (`-wtm_bdf2_on_V`), variable-step BDF2, and
  **adaptive time stepping** (`-wtm_dt_adaptive`). See `benchmark/BDF2_ADAPTIVE_DESIGN.md`.
  The **adaptive controller is detached from the integrator**: `-wtm_dt_adaptive` composes with any of
  them – `-wtm_anderson` → 1st-order backward-Euler (ring-proof), `-wtm_tr_bdf2` → TR-BDF2,
  `-wtm_bdf2_on_V` → BDF2-on-V – the error *estimate* being the only method-specific piece (TR-BDF2's
  embedded two-stage estimate, else the generic linear-history predictor) feeding one shared
  grow/shrink/reject controller. Its error norm **includes the free surface** (where stability is set),
  which is what lets it settle a cold start rather than growing Δt into a surface limit cycle. A **PI
  step-size controller** damps the Δt "hunting" that otherwise locks into resonance dead-bands, so on an
  equilibrium/spin-up run the step tolerance is **derived from the convergence target** – `dt_tol =
  min(50·eq_tol, 0.5 m)` unless `-wtm_dt_tol` is set – making **`eq_tol` the single knob**, with no toxic
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
  difference – the log-mean of the exponential deep T, the arithmetic mean of the affine soil T, and the
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
  step) – a **speed** win only: it does **not** change the equilibrium and does **not** raise the stable
  step ceiling (that is set by the operator, not the guess). Off by default.
- **Adaptive time stepping for TR-BDF2** (`-wtm_tr_bdf2 -wtm_dt_adaptive`, _experimental_): a self-tuning
  step that stays as large as accuracy and convergence allow – "long enough to be efficient but not so long
  it fails" on terrain / conditions whose stable-step ceiling varies. Two coupled mechanisms: a
  **reject/retry feasibility floor** – a non-converged stage or step shrinks `deltat` and retries from the
  uncommitted state (accumulators rolled back), so a step too large for the local conditions cannot crash
  the run – and an **embedded error estimator** from TR-BDF2's two stages, `h_pred = [Y_γ − (1−γ)hⁿ]/γ`
  (exact for linear-in-time, `O(Δt²)` for curvature; needs no history, valid on the first step): `est >
  -wtm_dt_tol` shrinks and retries, otherwise `deltat` grows toward the tolerance, capped by the step's
  convergence headroom (`-wtm_dtc_easy_iters`) and `-wtm_dtc_dt_max`. The error norm **excludes surface
  cells** (`wtd ≥ −band`) so the non-smooth free-surface clamp cannot spike the estimate and force `deltat`
  tiny (the failure that shelved the earlier history-extrapolation estimator). Measured on Esquibel
  (dry −20 %): `-wtm_dt_tol 1/5/20 m` → 33/12/8 steps (6/3/1 rejected), all converged – monotone in the
  tolerance. Off by default. See `benchmark/BDF2_ADAPTIVE_DESIGN.md`.
- **Robust equilibrium auto-stop** (`-wtm_eq_metric`, `-wtm_eq_frac`). The `-wtm_eq_tol` per-cycle early
  stop now applies on **every** spin-up pathway (fixed-dt, Newton-continuation, and the adaptive-dt
  controller – previously skipped). Its aggregation of the per-cycle water-table change is selectable:
  `frac` (**default**: converged when < `-wtm_eq_frac` = 0.1 % of land cells still exceed `eq_tol`), `max`
  (the old strict worst-cell criterion), or `rms`. The default changed from `max` to `frac` because `max`
  is worst-cell-hostage – one slow deep lowland cell filling to the surface can keep it from ever firing
  even though the bulk has converged (diagnosed as a metric artifact, not a physical oscillation). Measured
  trade at `eq_tol` 0.05 m: `max` never stops, `rms` stops early but loose (14.6 m worst-cell residual),
  `frac` stops with a 4.3 m worst-cell residual – the robust middle. See `benchmark/adaptive_dt/`.
- **Extended-soil option** (`collection.method: extended_soil`, _experimental_): continues the aquifer above the
  land surface to remove the water-table-depth = 0 free boundary from the groundwater step.
- **Configurable transmissivity / storativity smoothing** (`-wtm_ksat_soilbottom_smoothing_width`,
  `-wtm_ksat_surface_smoothing_width`, `-wtm_storativity_surface_smoothing_width`): optional rounding
  of the piecewise T/S boundaries, applied consistently across all solver paths.
- **Automatic equilibrium stop, on by default** (`-wtm_eq_tol`). The convergence-based early stop now works
  on **all** solver paths (it was previously silently ignored except on the Newton dt-continuation path) and
  gates on the **per-cycle** water-table change `max|wtdᴺ − wtdᴺ⁻¹|` over land – the honest steady-state
  signal, free of the cosmetic within-cycle free-boundary flicker that the per-sub-step change carries. It is
  **on by default for equilibrium runs** (0.01 m ≈ 1 cm per ~1-yr cycle; two consecutive cycles below the
  tolerance → stop) and **off for transient runs**, which must play out in full. `-wtm_eq_tol 0` disables it;
  any value overrides. Each cycle prints the per-cycle change alongside a within-cycle flicker diagnostic.
- **Sub-step under-relaxation** (`-wtm_relax a`, default 1 = off): blends each sub-step's solved water table
  with the previous one, `w ← a·w_solve + (1−a)·w_prev`, damping the period-2 flicker at pinned free
  boundaries (a FillSpillMerge lake surface, or a `-wtm_direct_to_runoff` exfiltration cell). Inert at steady
  state – the equilibrium is unchanged; only the transient march is damped.

#### Smooth surface-water transition (now the default – three tapers, per-taper off-switches)
The hard `wtd = 0` switch is replaced by three smooth, implicit, order-preserving tapers, all **on by
default** and each individually disabled by its flag (e.g. `-wtm_evap_taper 0`):
- **Taper 1 – sub-surface sink** (`-wtm_surface_sink`): a near-surface removal that holds the water
  table at/below the surface and hands the exfiltrated water to FillSpillMerge (it stays in the
  domain) – the smooth replacement for the hard "surface water → runoff" handoff, preserving 2nd-order
  accuracy across `wtd = 0`. On both the Anderson default and Picard/BDF2-on-V paths. Its width scales
  with the timestep (`width = C·qmax·dt`, C = 2) for stability at every step (tight – mm–cm – at small
  transient dt, wider only at large equilibrium dt); `qmax` default 1 m/yr, `-wtm_surface_sink_width`
  overrides.
- **Taper 2 – demand-identity evaporation** (`-wtm_evap_taper`): a single smooth, implicit transition
  from land-surface evapotranspiration (deep) to open-water evaporation (at/above the surface),
  replacing the hard ET↔open-water switch. Makes FillSpillMerge lake formation cross-rank deterministic
  at the evaporation threshold. Works in both `evap_mode 1` and `evap_mode 0` (it supersedes evap_mode
  0's remove-all).
- **Taper 3 – accessibility / extinction-depth clamp** (`-wtm_extinction`, depth `-wtm_extinction_depth`,
  default 8 m): gates taper 2's sub-surface evaporative deficit by depth, so an arid table (`ET > precip`)
  draws down only within the extinction depth (phreatic ET) rather than without bound. Depth basis:
  rooting depths (Canadell et al. 1996) / groundwater-ET extinction depths (Shah et al. 2007), see
  `benchmark/SURFACE_SINK_DESIGN.md` §14f.

Any configuration other than all-three-on emits a warning (arid-unsafe, inert, or the legacy
hard-switch model). See `benchmark/SURFACE_SINK_DESIGN.md`.

- **Direct-to-runoff exfiltration constraint** (`-wtm_direct_to_runoff`, _opt-in alternative to taper 1_): removes the
  above-surface water-table excess to runoff each sub-step at rate `max(0,wtd)/dt`, pinning the table at the
  land surface with no rate cap (so no runaway pile) and no below-surface band (so no artificial depression).
  A simpler, tuning-free surface→runoff handoff; the removed water is routed to FillSpillMerge, so it stays
  in the domain. Also modestly **faster** (~24 % per cycle on Esquibel), because pinning the exfiltration cells
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
  determinism + smoothness – the `SURFACE_SINK_DESIGN` §14d experiment sequence), and DMDA
  gather/scatter **unit tests**; synthetic terrain generators (spectral / Fourier-mode and fractal).
- **`BUILD_HPC.md`** cluster build guide (MSI worked example), single-node scaling/memory study drivers,
  a solve profiler, publication figure and dataset generators, and design notes.

### Changed
- **Grid geometry derived from the input's GDAL geotransform** (#124). The per-cell N–S / E–W degree
  spacing, southern edge, and cos-latitude cell-size scaling are read from the topography raster's
  geotransform – the authoritative georeference – rather than the `grid:` config block, which is deprecated
  to a fallback used only for un-georeferenced inputs (a stray override alongside a real geotransform is
  ignored with a warning). Test fixtures were migrated to carry correct geotransforms via the shared
  `tests/wtm_testgrid.py` writer, and the golden suite validates the identical-results swap. Projected
  (non-lat-lon / metre-CRS) grids are not yet handled – a follow-up on #124.
- **Adaptive time-step error measured in water (volume), not head.** The TR-BDF2 / history embedded local-error
  estimate is now `|storedVolume(wtd) − storedVolume(wtd_pred)|` (= `|S·Δwtd|`, water moved), the same units as
  the equilibrium stop, so the per-step accuracy tolerance and the equilibrium tolerance are directly
  comparable. On an equilibrium run the step tolerance
  (`solver.water_volume_timestep_error_tol`, → `-wtm_dt_tol`) defaults to track `run.equilibrium_stop.tol`
  (capped at the free-surface ring bound); a transient run keeps its own accuracy default.
- **Equilibrium auto-stop judged on water moved, not head.** The per-cycle convergence metric
  (`run.equilibrium_stop.metric`: `max` | `rms` | `frac`) now measures the pure-water depth `|S·Δwtd|` (m of
  water) rather than the head change `|Δwtd|`, so deep low-storativity cells – a metre of head over ~zero water
  – can no longer hold a run "unconverged" at steady state. `run.equilibrium_stop.tol` is therefore a water
  depth (default `0.001` = 1 mm of water). The separate `water-max` / `water-rms` metric names are retired (all
  three metrics are water-based now; the old names map with a deprecation note). Raw head is still printed each
  cycle as a diagnostic.
- **`arp.runoff` (the Fill-Spill-Merge input carrier) is assembled additively.** It is zeroed once per step,
  then every contributor – the runoff-ratio channel, the exfiltration sink, and FSM's own ponded water – adds into
  it before the single FSM, replacing a fragile overwrite-vs-add scheme with an order-independent lifecycle.
  Byte-identical on the FSM tests (conservation closes to 0.0, MPI bit-identical); restores the additive
  pattern of the upstream FillSpillMerge.
- **FillSpillMerge now runs EVERY timestep (tight surface–subsurface coupling).** Previously FSM ran once per
  `maxiter` groundwater sub-steps; it now runs after every accepted step. This removes the coupling-interval
  (`maxiter` / `niter`) dependence of the marginal-lake / lakeshore equilibria – the free-surface "flicker" and
  the collector×FSM equilibrium divergence were both artifacts of *batching* FSM. `fsm_off` runs are
  byte-identical; `fsm_on` equilibria are now report-interval-independent (verified to 0.0 m), and FSM MPI
  consistency still holds. Affordable at ~4% wall single-node (FSM compute is microseconds; the driver for
  parallel FSM is memory at global scale, not compute – see `benchmark/esquibel/FSM_COST.md`). The four `fsm_on`
  golden references were regenerated to the new tight-coupled equilibria (fsm_evap0/1 +3.5 m, fsm_runoff/hi
  ~+40 m; below_ground / transient unchanged) – the N-independent answers, not a regression. This **supersedes
  the golden deltas quoted in the solver entry below**, which were the earlier Anderson+implicit intermediate.
  See `benchmark/FSM_EVERY_STEP_DESIGN.md`. The FillSpillMerge progress bar is silenced in the `wtm` library
  (`RICHDEM_NO_PROGRESS`) since it would otherwise spam once per step; the standalone `dephier.x` keeps it.
- **The default solver is now matrix-free Anderson** (was semi-implicit BDF2-on-V/Picard). Anderson is the
  robust production worker, carries the exact in-residual exfiltration constraint, and is bit-exact across MPI ranks;
  it is 1st-order-in-time (backward-Euler cc, the right choice for equilibrium). Opt into BDF2-on-V/Picard
  (large stable steps, 2nd-order) with `-wtm_bdf2_on_V`. Picard is retained as the cross-rank-deterministic
  **grounding reference** the golden tests hold Anderson against. Combined with the `runoff_collector` AUTO
  default (implicit surface exfiltration; explicit under `-wtm_dt_adaptive`), the golden references were
  regenerated. The old default is exactly reproducible with `runoff_collector legacy -wtm_bdf2_on_V` (verified
  bit-for-bit against the pre-flip goldens), so every golden delta is the intended change, not a regression:
  below_ground 4.37 m (solver order over the deliberately short 6-step run), fsm_evap0/1 10.0 m (band sink →
  implicit exfiltration constraint), fsm_runoff/hi 36.5/31.6 m max but 0.28 m mean (a few FSM routing-threshold cells
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
  clamp there, complementing the in-residual taper-1 sink; the operator-consistent in-residual exfiltration
  `-wtm_direct_to_runoff` is task #100). Golden references were regenerated: the subsurface case is
  unchanged, and cases that generated above-surface water shifted (fsm cases up to ~9.4 m, the 4-cycle
  cold-start transient up to ~24.9 m as routing early surface water changes the trajectory).
- **Renamed the nonphysical developer switch `-wtm_allow_surface_ponding` to
  `-wtm_dev_allow_aboveground_water_columns`** (the `-wtm_dev_` prefix marks it developer/nonphysical at the
  point of use, and the new name says what it actually permits – vertical water columns standing above the land
  surface, not lakes). It still disables both runoff clamps and prints a warning; it is a
  testing/diagnostics-only regime (used by `tests/boundary_analytic` to reach the constant-transmissivity
  ponded-parabola solution), never a valid model configuration.
- **Snapshot output filenames now carry the simulated year.** Water-table rasters are written as
  `{outfile_prefix}{cycle:09}_{years}yr.tif` (was `{outfile_prefix}{cycle:09}.tif`), so each periodic
  snapshot is self-describing by simulated time – essential for transient runs, informative for spin-up
  progress. `years = cycles_done · maxiter · deltat / seconds_in_a_year` (a cycle spans a fixed
  `maxiter·deltat` even under adaptive dt). The zero-padded cycle stays the leading field, so any
  `glob(prefix + "*.tif")` + sort still orders by cycle (the golden suite is unaffected). Downstream
  analysis scripts that constructed the exact old name now glob for the final output.
- **Default surface-water / evaporation model is now the smooth transition** (surface-transition
  tapers 1–3 on). This replaces the hard `wtd = 0` ET↔open-water switch – which made FillSpillMerge
  lake formation rank-dependent (non-deterministic across MPI rank counts) and applied no phreatic ET –
  with a cross-rank-deterministic, 2nd-order-preserving transition, and it lets arid tables draw down
  physically (phreatic ET to an extinction depth). Recharge becomes the precip source with evaporation
  carried by the smooth `E_eff`, and runoff becomes `runoff_ratio · precip` (a split of the source).
  Golden references were regenerated. Disable per taper (e.g. `-wtm_evap_taper 0`) for the legacy
  behavior; see the tapers under _Added_.
- **The default solver is now the semi-implicit BDF2-on-V (Picard) path, for both run types; no PETSc
  solver flags are required on the command line.** Equilibrium reaches steady state in a handful of
  large, stable steps – Picard's Newton + GAMG solve has a nearly step-size-independent cost, so
  `deltat` can be raised by orders of magnitude (measured flat at ~28 SNES iterations from `deltat` = 1
  to 1000 yr on a real DEM). Transient gets genuine 2nd-order-in-time accuracy from the same solver.
  This **replaces the previous matrix-free Anderson default**, which – having no preconditioner – is
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

### Deprecated
- **`maxiter`** – no longer a coupling interval (FillSpillMerge runs every timestep). Superseded by
  `report_interval` (steps or a time). Still parsed and mapped to `report_interval` with a warning; removal is
  planned.
- **`cycles_to_save`** – superseded by `save_nreport_interval` (save a raster every K reports). Still parsed and
  mapped with a warning.

### Fixed
- **MPI ghost-cell bug** in the groundwater solve, with a dedicated ghost-cell validation test.
- **Integer division** that froze transient forcing at its start-time values.
- Water-budget diagnostics made **MPI-consistent** (owned-cells-only partials + scalar reduction).
- Picard post-equilibrium **false divergence** (a sensible default `-snes_atol`).
- **Anderson solver stall on steep real terrain** (`DIVERGED_MAX_IT` at the iteration cap; reported on a
  real DEM). The undamped Anderson path diverged on steep, heterogeneous topography; the Anderson
  (now opt-in, `-wtm_anderson`) path damps (`-snes_anderson_beta 0.5`), verified to converge on the
  Corsica DEM at 1–4 MPI ranks. The damping is the fix – a wider acceleration window also converges but
  adds per-iteration parallel reductions that the discontinuous FillSpillMerge routing amplifies into
  ~mm cross-rank differences, so m stays at 10.
- **Anderson near-convergence instability at large scale** (`DIVERGED_MAX_IT`; a cold ~139-million-cell
  equilibrium solve). Near convergence the Anderson residual-difference vectors go nearly linearly
  dependent, the least-squares mixing coefficients blow up, and the residual *reverses and oscillates*
  instead of settling. The fix is a **periodic history restart, now on by default for the Anderson path**
  (`-snes_anderson_restart_type periodic -snes_anderson_restart 20`): purging the history before it
  degenerates lets the solve converge (~40 iterations at 139M, robustly across periods 10–25). It is a
  **safe conditional default** – small grids converge in fewer than 20 iterations so the restart never
  fires (the full regression suite is byte-identical with it on), and only large / stiff runs engage it.
  Chosen over widening `m` (which converges but doubles the per-iteration reductions and reopens the
  cross-rank-consistency issue) and over the adaptive *difference* restart (which triggers on a residual
  rise that only occurs *at* the flail – too late). The instability is driven by high-latitude `cos(lat)`
  cell anisotropy, so it is a real property of large real-world domains, not just a test artifact.
  Disable with `-snes_anderson_restart_type none`. See `benchmark/esquibel/` sweeps.
- **Optional ρ-adaptive Anderson restart** (`-wtm_adaptive_restart`): a *proactive* alternative to the
  fixed-period default that restarts Anderson's history when the convergence *rate* degrades
  (ρ = ‖F_k‖/‖F_{k-1}‖ → 1 – the flail precursor, which appears *before* the residual rises), restarting
  each phase from the best iterate. Because it triggers on the rate rather than a fixed count, it adapts
  to a flail arriving at an unknown iteration – robustness for scales beyond those tested (the road to
  global). Confirmed at 139M: converges, and slightly faster than the periodic default (~112 s vs ~140 s,
  by restarting only when needed). Off by default; tunable via `-wtm_ar_rho / _patience / _max_it /
  _max_restarts`. Robust finish: near equilibrium the Anderson step floors just above the relative step
  tolerance, so true convergence is never formally declared – the controller now returns the tracked best
  iterate (and a phase that diverges after a good iterate falls back to it, with a warning) instead of
  aborting once restarts are exhausted. Regression: `tests/adaptive_restart/`.

### Removed

- **BREAKING – five more `-wtm_*` flags are retired in favour of their config keys**, continuing the move
  to a single configuration surface. **27 of the 65 flags are now gone.**

  | retired flag | config key |
  |---|---|
  | `-wtm_land_boundary` | `boundaries.land` (`dirichlet` → `dirichlet_sea_level`) |
  | `-wtm_volume_storage` | `solver.storage: volume` |
  | `-wtm_dt_continuation` | `solver.dt_continuation`, implied by `solver.method: newton` |
  | `-wtm_picard` | `solver.method: picard` |
  | `-wtm_active_set` | `surface_water.collection.method: active_set` (the default) |

  Each was proved equivalent **before** removal rather than after: the flag route and the config route
  were run and required byte-identical. Passing any of them now aborts by name.

  **`-wtm_active_set` was not a synonym for its mode**, and that matters to anyone porting a script. As a
  flag it was an *orthogonal switch* that superseded whatever collector was configured; as a member of the
  `collection.method` enumeration it is one mode among six, mutually exclusive with the rest. "Active set
  AND a collector" is now unrepresentable rather than resolved by precedence – the intended end state, but
  a retired behaviour rather than a renamed one. Two regression arms that tested the supersession were
  replaced rather than deleted: `tests/active_set` now asserts active_set *differs* from both plain
  collectors, and `tests/dt_sensitivity`'s positive control moved from the retired band sink to `implicit`.

- **BREAKING – the two alias flags are retired: `-wtm_extended_soil` and `-wtm_dev_active_set`.** Both
  were older spellings for something the config names directly, and both now abort by name rather than
  being honoured with a deprecation notice – a deprecation warning is only useful while something still
  reads it. Use `surface_water.collection.method: extended_soil` and `: active_set` (the latter is the
  default). `-wtm_extended_soil` did more than alias: it *selected* the mode when no method was
  configured, and a configured method superseded it with a warning; both halves go with the flag, so the
  mode is now set only where it is resolved. Two error messages that told you to "remove
  `-wtm_extended_soil`" – the `-wtm_Tbar` and `-wtm_kirchhoff` guards – now name the config key instead.
  With this, the ALIAS category is empty and **22 of the 65 originally classified flags are retired**.

- **BREAKING – the taper-1 sub-surface band sink is retired**, together with
  `surface_water.collection.method: legacy`, the flags `-wtm_surface_sink`, `-wtm_direct_to_runoff` and
  `-wtm_surface_exfiltration_to_runoff`, the `-wtm_fringe_*` capillary-fringe knobs, and the whole
  `surface_water.collection.sink` config section. Fork issue #7, now closed.

  The sink held the water table in a band *below* the land surface so that no cell ever crossed
  `wtd = 0`, buying 2nd-order time accuracy by **dodging** the free boundary rather than solving it. Its
  band width is `2·qmax·dt` and that dt-scaling is intrinsic – a fixed width overshoots for a rate-capped
  smooth sink – so its equilibrium water table moved with the time step: a plateau interior at **−1.56 m**
  at `dt` = 1 yr against **−0.79 m** at `dt` = 0.25 yr, while the pure groundwater solve is dt-independent
  to six decimals. That also disqualified it as an accuracy diagnostic, since second-order convergence
  toward a dt-dependent target measures nothing.

  Its niche – a differentiable tangent letting Picard and Newton cross the surface – is filled by
  `collection.method: active_set`, the primal-dual active-set / semismooth treatment of
  `wtd ≤ 0 ⊥ seepage ≥ 0` that issue #7 itself prescribed as the way to "let the taper go for good". That
  is the default, and the pin is in the analytic Newton Jacobian.

  **Two things went differently from the plan in that issue**, and both are recorded there. The
  replacement is `active_set`, **not** `implicit`: #7 called `implicit` the dt-independent exact face, but
  its retained head is itself ~linear in `dt` (**1.97 / 0.68 / 0.34 m** at `dt` = 1, 1/3, 1/6 week, FSM
  off), and with FSM on the *lake count* moves with `dt` – so that swap would have traded one
  dt-dependence for another. And it was **not** the golden regold the issue and the source comment both
  predicted: golden configs select no collector, so they already resolved to the default and the selector
  had been forcing the sink off for them.

  `legacy` was removed only after it was shown to have no content left: with the sink off it collapsed
  **exactly** onto the modes it wrapped (`legacy` + flag == `explicit` / `implicit`, `max|Δ| = 0.000e+00`,
  zero cells differing), and `budget_closure`'s two `legacy` arms had become bit-identical duplicates of
  their `explicit` counterparts, so they were deleted rather than kept as coverage.

  **Upgrading:** replace `collection.method: legacy` with `explicit` or `implicit` – or leave the key out
  and take the `active_set` default, which is the recommended choice. All the removed keys and flags now
  abort by name rather than being silently ignored.

  `tests/taper` studies A and B, written to validate the sink, were **re-pointed at `active_set`** rather
  than deleted: they assert properties rather than values – cross-rank determinism swept through
  `owe = precip`, monotonicity, and a pond forming with a cross-rank-identical shoreline – and those
  invariants matter more under an enforcement that sits *on* the crossing than under a taper that never
  engaged it. `benchmark/SURFACE_SINK_DESIGN.md` is kept as the record of the mechanism and its
  mathematics.


- **`evap_mode` is no longer a config key.** The member is frozen at 0 and is consulted only with the
  evaporation taper switched off, so the key could not express what it named. `tests/emit_config.sh`
  drops it. Note for anyone comparing against older runs: two benchmark arms that differed only by
  `evap_mode` were emitting byte-identical configs, so any "these agree" conclusion drawn from that pair
  was reading one measurement twice.


- **Seventeen `-wtm_*` flags, replaced by config keys.** Each setting is now parsed from the config,
  schema-checked, and read directly by its consumer. Passing a removed flag aborts by name (see the
  unconsumed-flag check above), so a stale script fails loudly rather than silently taking a default:

  | removed flag | config key |
  |---|---|
  | `-wtm_eq_tol` | `run.equilibrium_stop.tol` |
  | `-wtm_eq_metric` | `run.equilibrium_stop.metric` |
  | `-wtm_eq_frac` | `run.equilibrium_stop.frac` |
  | `-wtm_dt_adaptive` | `solver.adaptive_dt` |
  | `-wtm_dt_tol` | `solver.water_volume_timestep_error_tol` |
  | `-wtm_dtc_dt_max` | `solver.dt_max` |
  | `-wtm_Tbar` | `solver.t_bar` |
  | `-wtm_T_bedrock` | `transmissivity.additive_background_transmissivity` |
  | `-wtm_evap_taper_wtdc` | `evaporation.et_sigmoid.wtd_center` |
  | `-wtm_evap_taper_s` | `evaporation.et_sigmoid.logistic_width` |
  | `-wtm_extinction_depth` | `evaporation.extinction_depth` |
  | `-wtm_surface_sink_qmax` | `surface_water.collection.sink.qmax` |
  | `-wtm_surface_sink_width` | `surface_water.collection.sink.width` |
  | `-wtm_fringe_source` | `surface_water.collection.sink.fringe_source` |
  | `-wtm_fringe_cap` | `surface_water.collection.sink.fringe_cap` |
  | `-wtm_fringe_ksat_coef` | `surface_water.collection.sink.fringe_ksat_coef` |
  | `-wtm_fringe_length` | `surface_water.collection.sink.fringe_length` |

  Motivation beyond tidiness: while a setting lived only in PETSc's options database it was in no
  `Parameters` field and no resolved-configuration line, so a CLI flag that discarded a configured value
  left no trace anywhere. `benchmark/CONFIG_FLAG_COVERAGE.md` classifies all 65 `-wtm_*` flags and
  records which remain and why.

- The `-wtm_const_storativity` diagnostic path.

[Unreleased]: https://github.com/KCallaghan/WTM/compare/v2.0.1...HEAD
