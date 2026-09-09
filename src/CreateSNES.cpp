//#include "CreateSNES.hpp"
#include <cstring>  // std::strcmp for -wtm_eq_metric parsing

void InitialiseSNES(AppCtx& user_context, Parameters& params) {
  SNESCreate(PETSC_COMM_WORLD, &user_context.snes);

  user_context.cellsize_NS_squared = params.cellsize_n_s_metres * params.cellsize_n_s_metres;
  user_context.deltat              = params.deltat;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMDACreate2d(
      PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE,
      DM_BOUNDARY_NONE,
      DMDA_STENCIL_STAR,
      params.ncells_x,
      params.ncells_y,
      PETSC_DECIDE,
      PETSC_DECIDE,
      1,
      1,
      nullptr,
      nullptr,
      &user_context.da);
  DMSetFromOptions(user_context.da);
  DMSetUp(user_context.da);

  user_context.make_global_vectors();
  user_context.make_local_vectors();
  user_context.full_grid_gather = new DMDAFullGridGather(user_context.da);

  DMSetApplicationContext(user_context.da, &user_context);
  SNESSetDM(user_context.snes, user_context.da);

  // Default to Anderson mixing (matrix-free, robust for heterogeneous media).
  // Override at runtime: -snes_type newtonls -ksp_type gmres -pc_type gamg
  // to use Newton-Krylov with the analytic Jacobian and Picard preconditioner.
  SNESSetType(user_context.snes, SNESANDERSON);

  // Anderson defaults: narrow window m=10 PLUS mild damping beta=0.5. The undamped default (beta=1)
  // STALLS to DIVERGED_MAX_IT on steep, heterogeneous real DEMs (reproduced on the Corsica DEM; matches
  // Kerry's Esquibel hang); beta=0.5 converges there (Corsica: 10000-iter stall -> ~30-44 iters). It is
  // the DAMPING, not the window, that buys the robustness: m=30 (PETSc's default window) also converges
  // but its extra per-iteration vector reductions add cross-rank FP non-associativity that the
  // discontinuous FSM/runoff routing amplifies into ~2 mm rank-dependence (breaks fsm_runoff golden
  // consistency); m=10/beta=0.5 stays machine-consistent (~1e-8) AND keeps the narrow window's lower
  // per-iteration cost. Set only if the user did not override, so runtime -snes_anderson_m /
  // -snes_anderson_beta still win (raise beta toward 1 for speed on well-conditioned problems).
  PetscBool anderson_m_set = PETSC_FALSE, anderson_beta_set = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-snes_anderson_m", &anderson_m_set);
  PetscOptionsHasName(nullptr, nullptr, "-snes_anderson_beta", &anderson_beta_set);
  if (!anderson_m_set) PetscOptionsSetValue(nullptr, "-snes_anderson_m", "10");
  if (!anderson_beta_set) PetscOptionsSetValue(nullptr, "-snes_anderson_beta", "0.5");

  // Periodic restart (default period 20): purge the Anderson history before its near-convergence
  // least-squares degenerates. At large/stiff scale the residual-difference columns go linearly
  // dependent near convergence, the mixing coefficients blow up, and the residual REVERSES and
  // oscillates -- the "flail" that stalls a cold 139M solve to DIVERGED_MAX_IT. A proactive periodic
  // restart resets the history while still in the easy regime, so the degeneracy never accumulates:
  // 139M cold DIVERGES without it, CONVERGES in ~40 iters with it, robustly across periods 10-25.
  // SAFE as a default: small grids converge in FEWER than `period` iters so the restart NEVER fires
  // (byte-identical -- verified, full suite unchanged); only large/stiff runs hit it. Chosen over a
  // wider window (m=20 also converges but does 2x the per-iteration reductions AND may break cross-rank
  // consistency like m=30) and over the ADAPTIVE (difference) restart, which fires on a residual RISE
  // that only occurs AT the flail -- too late (confirmed: fails at restart_it 1/2/3). Overridable
  // (-snes_anderson_restart_type none to disable, -snes_anderson_restart N to retune). See #85/#87.
  PetscBool restart_type_set = PETSC_FALSE, restart_period_set = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-snes_anderson_restart_type", &restart_type_set);
  PetscOptionsHasName(nullptr, nullptr, "-snes_anderson_restart", &restart_period_set);
  // When -wtm_adaptive_restart drives restarts from the outer rho loop (update()), the internal
  // restart must be OFF (else the two mechanisms fight and the internal one muddies the rho signal).
  const bool adaptive_here = params.ar_enabled;
  if (!restart_type_set)
    PetscOptionsSetValue(nullptr, "-snes_anderson_restart_type", adaptive_here ? "none" : "periodic");
  if (!restart_period_set && !adaptive_here) PetscOptionsSetValue(nullptr, "-snes_anderson_restart", "20");

  // Step-tolerance default 1e-8. The damped default (beta=0.5) converges LINEARLY, so it stops right
  // AT the requested step tolerance rather than over-shooting it as the undamped solver does; at the
  // looser 1e-6 that left ~1e-6 (um-scale) rank-dependence in the water table (each rank converges to
  // its own 1e-6-accurate solution). 1e-8 is tight enough that the parallel solve is machine-consistent
  // (~1e-9 cross-rank) AND still reachable on steep terrain (Corsica converges in ~80 iters; a residual
  // -snes_atol criterion CANNOT be reached there -- the residual floors above 1e-8 -- so use a STEP
  // tolerance, which tracks solution change and is reachable). Set only if the user did not override.
  // NOTE (#61): this HEAD step tolerance is no longer the governing test on the default path -- the water
  // metric below is (solver.convergence.metric, default `water`). It still sets ar_stol and still governs
  // when the user asks for `metric: head`, so the default stays.
  PetscBool snes_stol_set = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-snes_stol", &snes_stol_set);
  if (!snes_stol_set) PetscOptionsSetValue(nullptr, "-snes_stol", "1e-8");

  // Volume-weighted per-solve convergence (#127): judge the SNES step in WATER (|S*Δwtd|) instead of head, so the
  // per-solve gate matches eq_tol / dt_tol. Opt-in; DIAGNOSTIC unless _govern. The test is registered in
  // transient_groundwater.cpp::update() (VolumeStepConverged); here we only read the flags into user_context.
  // WATER VOLUME IS THE DEFAULT (#61). `-wtm_snes_head_conv` (config `solver.convergence.metric: head`) is the
  // OFF-SWITCH, the same default-on/off-switch shape the evaporation tapers use. The old opt-in name is
  // kept as an explicit request for the default, so a config or command line that asks for volume still
  // reads correctly; when both arrive the off-switch wins, because only it can have been asked for
  // deliberately (volume needs no asking).
  user_context.snes_volume_conv_govern = !params.convergence_metric_head;
  user_context.vol_step_trace          = params.trace_water_step;  // INDEPENDENT of governing, see AppCtx
  user_context.snes_volume_conv_tol    = params.water_volume_tol;
  if (user_context.snes_volume_conv_govern)
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.convergence.metric: volume -- the per-solve step is judged as |S*Δwtd| (rel tol %g).\n",
                (double)user_context.snes_volume_conv_tol);
  else
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.convergence.metric: head -- the per-solve step is judged as |Δhead| (snes_stol). This is\n"
                "  NOT the default: a head step tolerance lets a few very deep cells speak for the whole grid, and\n"
                "  it does not match the water units of eq_tol/dt_tol or of the exact budget.\n");
  if (user_context.vol_step_trace)
    PetscPrintf(PETSC_COMM_WORLD, "output.trace: [water_step] -- per-iteration head-vs-water step lines.\n");

  // Semi-implicit Picard path (experimental; PICARD_MG_DESIGN.md / PICARD_MATH.md).
  // Gated behind solver.method: picard so the default Anderson path is untouched. When on,
  // allocate the SPD operator A(x) (also its own GAMG preconditioner) and a residual
  // work vector, and default the outer/inner solvers (below) unless the user overrode
  // them.
  // Time-integration flags nest: -wtm_dt_adaptive implies BDF2 implies the Picard path
  // (all live in the Picard operator/RHS). See BDF2_ADAPTIVE_DESIGN.md.
  PetscBool picard_flag = PETSC_FALSE, adaptive_flag = PETSC_FALSE;
  // config-owned (solver.method: picard); the -wtm_picard flag is retired. Kept as a PetscBool for the
  // same reason as adaptive_flag below -- the path resolution around it is written in PetscBool terms.
  picard_flag = (params.solver_method == "picard") ? PETSC_TRUE : PETSC_FALSE;
  // config-owned (solver.adaptive_dt); the -wtm_dt_adaptive flag is retired. Kept as a PetscBool
  // because the surrounding path-resolution logic below is written in PetscBool terms.
  adaptive_flag = params.adaptive_dt ? PETSC_TRUE : PETSC_FALSE;
  PetscBool bdf2v_flag = PETSC_FALSE;
  // config-owned (solver.time_integration: bdf2); the solver.time_integration: bdf2 flag is retired.
  bdf2v_flag = (params.time_integration == "bdf2") ? PETSC_TRUE : PETSC_FALSE;

  // The DEFAULT solver is the matrix-free Anderson path (selected ~60 lines below when no path flag is given):
  // the production worker -- robust across regimes, no preconditioner to tune, bit-exact across ranks, and it
  // carries the exact in-residual exfiltration constraint (runoff_collector=implicit). It is 1st-order-in-time
  // (backward-Euler cc), the right choice for equilibrium (a 2nd-order step oscillates at the free surface).
  // Opt into the semi-implicit volume-form BDF2-on-V/Picard path (large, ~dt-independent, 2nd-order steps) with
  // solver.time_integration: bdf2, matrix-free 2nd-order Anderson with solver.method: anderson solver.time_integration: bdf2, or Newton with
  // solver.method: newton. Any explicit path flag takes precedence.
  PetscBool force_anderson = PETSC_FALSE;
  // config-owned (solver.method: anderson); the solver.method: anderson flag is retired. It FORCES the matrix-free
  // path: an unset method defaults to Anderson too, but only an explicit choice keeps that path when a
  // time-integration key would otherwise select the Picard operator.
  force_anderson = (params.solver_method == "anderson") ? PETSC_TRUE : PETSC_FALSE;
  // solver.time_integration: tr-bdf2: L-stable strong-damping 2nd-order on the matrix-free Anderson path (two staged solves
  // per step). Implies the Anderson path (self-starting; no Picard operator, no BDF2 history vector).
  PetscBool tr_bdf2_flag = PETSC_FALSE;
  // config-owned (solver.time_integration: tr-bdf2); the solver.time_integration: tr-bdf2 flag is retired.
  if (!params.time_step_mode_set) {
    const char* why = params.time_step_mode == "ramp"  ? "solver.method: newton owns the step size on that path"
                      : params.time_step_mode == "fixed" ? "surface_water.collection.method: implicit cannot be "
                                                           "driven by an error controller -- its per-step error "
                                                           "grows as dt shrinks"
                                                         : "error-controlled stepping";
    PetscPrintf(PETSC_COMM_WORLD, "solver.time_step.mode: absent -> %s (%s).\n",
                params.time_step_mode.c_str(), why);
  }
  if (params.time_integration_auto)
    PetscPrintf(PETSC_COMM_WORLD, "solver.time_integration: auto -> %s (resolved from solver.method: %s).\n",
                params.time_integration.c_str(),
                params.solver_method.empty() ? "anderson" : params.solver_method.c_str());
  tr_bdf2_flag = (params.time_integration == "tr-bdf2") ? PETSC_TRUE : PETSC_FALSE;
  user_context.use_tr_bdf2 = (tr_bdf2_flag == PETSC_TRUE);
  // R2 (method uniqueness): tr-bdf2 no longer FORCES the Anderson path. It runs only there, so an
  // incompatible method is a contradiction and is refused by name. Previously the integrator silently
  // won: `solver.method: picard` + `solver.time_integration: tr-bdf2` ran ANDERSON without a word (#18).
  if (tr_bdf2_flag && !params.solver_method.empty() && params.solver_method != "anderson")
    throw std::runtime_error(
        "config: solver.time_integration: tr-bdf2 runs only on the matrix-free Anderson path, but "
        "solver.method: " + params.solver_method +
        " was requested. Set solver.method: anderson, or choose a different time_integration. (Before "
        "this check the integrator silently won and the run used Anderson.)");
  // solver.method: newton -- true Newton-Krylov path (analytic Jacobian). Like solver.method: anderson it selects a
  PetscBool newton_flag = PETSC_FALSE;
  // config-owned (solver.method: newton); the -wtm_newton flag is retired. NOTE the config value means
  // the WORKING RECIPE -- it implies solver.newton.dt_continuation -- whereas the bare flag meant PLAIN Newton.
  // A caller that wanted plain Newton must now say `dt_continuation: false` explicitly.
  newton_flag = (params.solver_method == "newton") ? PETSC_TRUE : PETSC_FALSE;

  const bool adaptive_restart_flag = params.ar_enabled;
  // A TUNING dial must not change the solver. solver.anderson.restart is Anderson's; asking for it with
  // another method is a contradiction, not a request to switch.
  if (adaptive_restart_flag && !params.solver_method.empty() && params.solver_method != "anderson")
    throw std::runtime_error(
        "config: solver.anderson.restart is a setting of the Anderson path, but solver.method: " +
        params.solver_method + " was requested. A tuning setting does not select the solver.");
  // R2, and Andy's call (2026-09-02) on the one case where uniqueness changes an ANSWER rather than
  // fixing a silent substitution. `solver.time_integration: bdf2` with no method used to select PICARD,
  // because use_bdf2 fed the use_picard expression. Under "the method is chosen only by solver.method"
  // it would become 2nd-order ANDERSON instead -- a silent change to every existing config that relies
  // on the old implication. Neither silence is acceptable, so it is refused and the user states the
  // method. This breaks such configs deliberately, and names the two ways to fix them.
  if (bdf2v_flag == PETSC_TRUE && params.solver_method.empty())
    throw std::runtime_error(
        "config: solver.time_integration: bdf2 used to IMPLY solver.method: picard, and the method is now "
        "chosen only by solver.method. State it explicitly:\n"
        "  solver.method: picard    -- the BDF2-on-V Picard operator (what this config did before)\n"
        "  solver.method: anderson  -- 2nd-order matrix-free Anderson (the time discretization is a "
        "property of the residual, not the solver)");
  const bool any_path_flag = (picard_flag || adaptive_flag || bdf2v_flag);
  if (!force_anderson && !newton_flag && !any_path_flag) {
    // Default solver: matrix-free Anderson -- the production worker. It is robust across regimes and
    // converges where the Picard/Newton free-boundary solve struggles, and it carries the exact
    // in-residual exfiltration constraint (runoff_collector=implicit). No flag is set here: Anderson is simply the
    // path taken when neither Picard nor Newton is selected. It is 1st-order-in-time (backward-Euler cc,
    // the right choice for equilibrium, where a 2nd-order step oscillates at the free surface). Opt into
    // the semi-implicit BDF2-on-V/Picard solver (large stable steps, 2nd-order) with solver.time_integration: bdf2,
    // matrix-free 2nd-order Anderson with solver.method: anderson solver.time_integration: bdf2, or solver.method: newton.
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Defaulting to the matrix-free Anderson solver (robust across regimes; the production worker;\n"
        "  1st-order-in-time). Opt into BDF2-on-V/Picard (2nd-order, large steps) with solver.time_integration: bdf2,\n"
        "  or Newton with solver.method: newton.\n");
  }

  // BDF2 is now singular: the head form (-wtm_bdf2) was retired 2026-09-04, so the surviving scheme is
  // the volume form and the `_on_V` qualifier named a distinction that no longer exists. The config
  // value is plain `bdf2`, and the field now matches it.
  user_context.use_bdf2 = (bdf2v_flag == PETSC_TRUE);
  user_context.use_dt_adaptive = (adaptive_flag == PETSC_TRUE);
  // The dt controller is DETACHED from the integrator: -wtm_dt_adaptive no longer forces the 2nd-order
  // BDF2 residual. The integrator (cc backward-Euler / TR-BDF2 / BDF2-on-V) is selected by its own flags,
  // and the controller sizes dt for whichever one is active (see the estimate/controller split in
  // transient_groundwater.cpp update()). So `solver.method: anderson -wtm_dt_adaptive` is 1st-order adaptive-cc,
  // `solver.time_integration: tr-bdf2 -wtm_dt_adaptive` is 2nd-order TR-BDF2, `solver.time_integration: bdf2 -wtm_dt_adaptive` is BDF2-on-V.
  // A forced Anderson path keeps the matrix-free residual even with a BDF2 time flag: solver.method: anderson
  // solver.time_integration: bdf2 gives 2nd-order-in-time Anderson (time discretization is a property of the residual,
  // not the solver). Only take the Picard operator path when Anderson is NOT forced.
  // R2: the integrator no longer feeds the method. use_bdf2 used to make this true, which is how
  // `time_integration: bdf2` selected Picard; that implication is refused above rather than resolved
  // silently, so the method comes from solver.method alone.
  user_context.use_picard      = (picard_flag == PETSC_TRUE);
  // Newton path is exclusive with Picard (a path flag wins if the user set both).
  user_context.use_newton      = (newton_flag == PETSC_TRUE) && !user_context.use_picard;
  user_context.use_adaptive_restart = (adaptive_restart_flag == PETSC_TRUE);
  user_context.ar_rho_threshold = params.ar_rho;
  user_context.ar_rho_patience  = params.ar_patience;
  user_context.ar_max_it        = params.ar_max_it;
  user_context.ar_max_restarts  = params.ar_max_restarts;

  // Newton dt-continuation (solver.newton.dt_continuation; needs solver.method: newton): equilibrium PTC that starts
  // deltat small (diagonally dominant -> non-singular Jacobian from a far guess) and grows it after
  // each converged step. Start dt defaults to params.deltat/200 (-wtm_dtc_dt0 overrides, seconds);
  // growth 1.5x/step (-wtm_dtc_grow); cap 1000*params.deltat (-wtm_dtc_dt_max). deltat persists across
  // cycles, so it ramps toward equilibrium. The WTM.cpp cycle loop drives the ramp. See
  // benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  // Config-owned (solver.newton.dt_continuation), resolved in Parameters -- solver.method: newton implies it.
  bool dtc_on = params.dt_continuation;
  user_context.use_newton_continuation = dtc_on && user_context.use_newton;
  // The opt-out warning lives here, not in the YAML bridge, because only this scope knows whether the
  // Newton path was actually selected.
  if (user_context.use_newton && params.time_step_mode_set && params.time_step_mode != "ramp")
    PetscPrintf(PETSC_COMM_WORLD,
                "WARNING [solver.method: newton + solver.time_step.mode: %s]: the continuation ramp is OFF, "
                "so this is PLAIN Newton. It converges from a WARM start but typically DIVERGES from a "
                "cold one (DIVERGED_LINE_SEARCH). Omit solver.time_step.mode to get the working recipe.\n",
                params.time_step_mode.c_str());
  // Convergence-based early stop (-wtm_eq_tol, a WATER depth in metres). Default ON for equilibrium runs
  // (0.001 m = 1 mm of water |S*Δwtd| per cycle), OFF for transient runs (a time-evolution run must play out
  // in full, so it is never auto-stopped). Pass -wtm_eq_tol 0 to disable on an equilibrium run, or any value
  // to override. Parsed for ALL solver paths (was previously only inside the Newton block below, so it was
  // silently ignored on the default Anderson/Picard path). run() checks the PER-CYCLE change against it.
  // config-owned (run.equilibrium_stop.tol); an absent key keeps the run-type default below
  if (params.eq_tol_set) user_context.eq_tol = params.eq_tol;
  if (!params.eq_tol_set)
    user_context.eq_tol = (params.run_type == "equilibrium") ? 0.001 : 0.0;
  // -wtm_eq_metric max|rms|frac: how the per-cycle change is aggregated for the equilibrium stop. ALL three
  // now judge the change in PURE-WATER DEPTH (|S*Δwtd|, m of water), NOT head -- deep low-storativity cells
  // (huge head swing, ~zero water moved) can no longer pin the stop, so it is FV-consistent and comparable
  // across cc and tr. eq_tol is therefore a WATER depth (default 0.001 = 1 mm). DEFAULT frac (converged when
  // < eq_frac of land cells exceed eq_tol) -- the measured best trade: max is worst-cell-hostage, rms is loose
  // (bulk only), frac both fires and stays precise. run.equilibrium_stop.frac sets the fraction (default 0.1%). Raw head is
  // still printed each cycle as a diagnostic. See benchmark/adaptive_dt.
  // config-owned (run.equilibrium_stop.metric)
  char eq_metric_str[16];
  std::strncpy(eq_metric_str, params.eq_metric.c_str(), sizeof(eq_metric_str) - 1);
  eq_metric_str[sizeof(eq_metric_str) - 1] = '\0';
  if (std::strcmp(eq_metric_str, "rms") == 0) user_context.eq_metric = 1;
  else if (std::strcmp(eq_metric_str, "max") == 0) user_context.eq_metric = 0;
  else if (std::strcmp(eq_metric_str, "water") == 0 || std::strcmp(eq_metric_str, "water-max") == 0 ||
           std::strcmp(eq_metric_str, "water-rms") == 0) {
    // Retired names: every metric now judges water, so water-max == max and water-rms == rms. Map them (rather
    // than silently falling through to frac) and note it, so old scripts keep their aggregation.
    user_context.eq_metric = (std::strcmp(eq_metric_str, "water-rms") == 0) ? 1 : 0;
    PetscPrintf(PETSC_COMM_WORLD, "NOTE: -wtm_eq_metric %s is retired (all metrics judge water now); using %s.\n",
                eq_metric_str, user_context.eq_metric == 1 ? "rms" : "max");
  }
  else user_context.eq_metric = 2;  // "frac" (default)
  user_context.eq_frac = params.eq_frac;  // config-owned; -wtm_eq_frac retired
  if (user_context.use_newton_continuation) {
    // dt0's default is DERIVED, not constant, so the _set flag is what distinguishes "the user chose
    // deltat/200" from "nobody asked".
    const double dt0 = params.dtc_dt0_set ? params.dtc_dt0 : params.deltat / 200.0;
    user_context.deltat = dt0;  // start small (overrides the params.deltat init above)
    user_context.dtc_dt0 = dt0;  // retained so full_config.yaml can report the resolved value
    user_context.dtc_grow   = params.dtc_grow;
    user_context.dtc_shrink = params.dtc_shrink;
    user_context.dtc_dt_max = 1000.0 * params.deltat;
    // config-owned (solver.dt_max); an unset key leaves THIS block's own default in place
    if (params.dtc_dt_max_set) user_context.dtc_dt_max = params.dtc_dt_max;
    user_context.dtc_easy_iters  = params.dtc_easy_iters;
    user_context.dtc_max_retries = params.dtc_max_retries;
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.newton.dt_continuation: Newton PTC, dt0=%g s, grow x%g if <=%d iters, shrink x%g on reject, "
                "dt_max=%g s.\n",
                dt0, user_context.dtc_grow, user_context.dtc_easy_iters, user_context.dtc_shrink,
                user_context.dtc_dt_max);
  }
  // output.trace: [budget] -- read UNCONDITIONALLY. It applies to every stepping path (fixed,
  // adaptive, continuation), so it must not be parsed inside the adaptive branch: on a fixed-dt run the
  // flag would then be set by the config bridge and read by nobody, and the unconsumed-flag guard would
  // abort the run. It did exactly that when this was first wired in.
  user_context.budget_trace = params.trace_budget;
  user_context.fsm_trace    = params.trace_fsm;
  if (user_context.use_dt_adaptive) {
    // config-owned (solver.water_volume_timestep_error_tol); unset keeps the eq_tol-tracking default
    const bool dt_tol_set = params.dt_tol_set;
    if (dt_tol_set) user_context.dt_tol = params.dt_tol;
    user_context.dt_trace = params.trace_dt;
    // The step-size controller's own knobs, parsed HERE as well as on the continuation path. They were
    // read ONLY inside `if (use_newton_continuation)`, yet the adaptive controller reads dtc_grow,
    // dtc_shrink, dtc_dt_max and dtc_easy_iters on EVERY adaptive step -- so on a plain
    // `-wtm_dt_adaptive` run these flags were accepted by PETSc, silently ignored, and the controller
    // ran on its compiled-in defaults.
    //
    // The cost is not only a lost knob: it makes a NEGATIVE experiment untrustworthy. Sweeping a flag
    // that is never parsed returns "no effect" for a reason that has nothing to do with the model, and
    // it reads exactly like a real result. It produced one: a growth-gate sweep at
    // -wtm_dtc_easy_iters 0 / 8 / 100000 returned byte-identical step counts, recorded as "the growth
    // gate is inert on this fixture" when the gate had never been varied at all. With the flag live,
    // the same sweep spans 57 steps (default 8) to 229506-and-still-running (0, growth forbidden).
    // Parsing these where they are used is also what makes the controller testable: tests/estimator_order
    // needs -wtm_dtc_grow 1 -wtm_dtc_shrink 1 to freeze dt and refine it from a fixed state.
    user_context.dtc_grow       = params.dtc_grow;
    user_context.dtc_shrink     = params.dtc_shrink;
    // config-owned (solver.dt_max); an unset key leaves THIS block's own default in place
    if (params.dtc_dt_max_set) user_context.dtc_dt_max = params.dtc_dt_max;
    user_context.dtc_easy_iters = params.dtc_easy_iters;
    // ...and max_retries with them. It was LEFT BEHIND when the other four were moved here: the adaptive
    // reject/retry loop reads dtc_max_retries on every rejected step (WTM.cpp:617), but the flag was
    // parsed only inside `if (use_newton_continuation)` above -- so on a plain adaptive run asking for it
    // did not tune the controller, it ABORTED the run ("the run was given 1 -wtm_ flag that nothing
    // read"). The adaptive loop was stuck on the compiled-in 15 with no way to reach it.
    user_context.dtc_max_retries = params.dtc_max_retries;
    // The adaptive step tolerance (dt_tol) is the per-step LOCAL ERROR in WATER (volume) units -- the SAME
    // units as the equilibrium-stop tolerance (eq_tol = |S·Δwtd|), because the embedded error estimate is now
    // volume-weighted (storedVolume difference; see transient_groundwater.cpp). They still measure different
    // things (dt_tol = how far one step strays from a linear prediction; eq_tol = when the run is steady), but
    // matched units make them COMPARABLE and coherent -- and coherence is required, since a time-marching scheme
    // cannot resolve a steady state finer than its own per-step error. SYMMETRY THROUGH CONVERGENCE: on an
    // equilibrium run the default step tol TRACKS eq_tol (integrate to the accuracy we detect), capped at the
    // free-surface ring bound so a LOOSE eq_tol still cannot let a big step ring the surface. This re-derivation
    // is now unit-correct -- the head/water unit-mismatch that forced the earlier decoupling is gone. A transient
    // run has no stop criterion -> the step tol is a pure accuracy knob (0.1 m water). -wtm_dt_tol overrides.
    const double dt_tol_ring_cap = 0.5;  // free-surface overshoot bound (m water); a bigger step rings
    if (!dt_tol_set) {
      if (params.run_type == "equilibrium" && user_context.eq_tol > 0.0)
        user_context.dt_tol = (user_context.eq_tol < dt_tol_ring_cap) ? user_context.eq_tol : dt_tol_ring_cap;
      else
        user_context.dt_tol = (params.run_type == "equilibrium") ? dt_tol_ring_cap : 0.1;  // never-stop eq | transient
    } else if (params.run_type == "equilibrium" && user_context.eq_tol > 0.0
               && user_context.dt_tol > user_context.eq_tol) {
      PetscPrintf(PETSC_COMM_WORLD,
                  "WARNING: -wtm_dt_tol %g m (water) is LOOSER than eq_tol %g m: a time-marching scheme cannot\n"
                  "  settle below its own per-step error, so this run will not reach equilibrium. Set the step\n"
                  "  tolerance <= eq_tol, or omit it to auto-track eq_tol.\n",
                  user_context.dt_tol, user_context.eq_tol);
    }
    // Adaptive error norm: RMS over land cells is the DEFAULT (robust on cold spin-up). MAX (worst-cell) is
    // opt-in via -wtm_dt_norm_max: under the water (volume) step-error a few surface-kink cells give a
    // dt-independent spike that the MAX norm is hostage to, stalling a cold start (GH #13). -wtm_dt_norm_rms is
    // still accepted (now the default); if BOTH are given, the explicit MAX wins.
    user_context.dt_norm_rms = params.dt_norm_rms;
    const char* integ = user_context.use_tr_bdf2      ? "TR-BDF2 (2nd-order)"
                        : user_context.use_bdf2  ? "BDF2-on-V (2nd-order)"
                        : user_context.use_picard     ? "backward-Euler Picard (1st-order)"
                                                      : "backward-Euler cc/Anderson (1st-order)";
    PetscPrintf(
        PETSC_COMM_WORLD,
        "-wtm_dt_adaptive set: adaptive dt (tol=%g m water, %s norm) on the %s integrator.\n",
        user_context.dt_tol,
        user_context.dt_norm_rms ? "RMS" : "MAX", integ);
  } else if (user_context.use_bdf2 && force_anderson == PETSC_TRUE) {
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.method: anderson + BDF2-on-V: 2nd-order-in-time matrix-free Anderson (BDF2-on-V residual, no\n"
                "  operator/preconditioner). Time-order decoupled from the solver.\n");
  } else if (user_context.use_bdf2 && user_context.use_newton) {
    // Was "-wtm_bdf2 set: enabling the Picard solver path (BDF2 requires it)", which stopped being true
    // when the method stopped being chosen by the integrator (R2). It was printing while the run took
    // the NEWTON path -- the announcement and the run disagreed.
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.method: newton + BDF2-on-V: 2nd-order-in-time Newton (analytic Jacobian on the\n"
                "  BDF2-on-V residual). Time-order decoupled from the solver.\n");
  }
  // BDF2 history carrier (w^{n-1}) is needed on ANY BDF2 path -- the Picard operator OR the matrix-free
  // Anderson residual (solver.method: anderson solver.time_integration: bdf2) -- by the predictor-seeded guess, AND by the detached
  // adaptive controller's generic linear-history error estimate (any non-TR integrator). Allocate it
  // whenever BDF2, the predictor, or adaptive dt is on, independent of use_picard.
  if (user_context.use_bdf2 || user_context.use_dt_adaptive) {
    VecDuplicate(user_context.x, &user_context.starting_wtd_prev);
    VecSet(user_context.starting_wtd_prev, 0.0);
    user_context.bdf2_prev_dt = user_context.deltat;  // ω=1 until Δt changes (adaptive)
  }
  // tr_expl: explicit old-state flux + removal at w^n, used by TR-BDF2's trapezoidal stage.
  if (user_context.use_tr_bdf2)
    VecDuplicate(user_context.x, &user_context.tr_expl);
  if (user_context.use_tr_bdf2) {
    VecDuplicate(user_context.x, &user_context.tr_ygamma);  // intermediate Y_gamma
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.time_integration: tr-bdf2: L-stable, strongly-damped 2nd-order matrix-free Anderson (TR-BDF2; two staged\n"
                "  solves/step, self-starting).\n");
  }
  if (user_context.use_picard) {
    DMCreateMatrix(user_context.da, &user_context.picard_A);
    VecDuplicate(user_context.x, &user_context.picard_r);

    // Defect-correction Picard is a modified-Newton iteration whose "Jacobian" is
    // the frozen operator A(x): each outer step solves A(x_k) dx = -(A x_k - b) via
    // the KSP, i.e. A(x_k) x_{k+1} = b(x_k). So the OUTER solver is a Newton type
    // (newtonls), NOT nrichardson (which would only do x <- x - lambda*F with no
    // linear solve). A basic (full-step) line search gives the plain Picard update.
    // The inner solve is CG+GAMG on the SPD A. (PETSc SNES ex15 fd/mf_picard.)
    PetscBool ksp_set = PETSC_FALSE, pc_set = PETSC_FALSE, snes_set = PETSC_FALSE, ls_set = PETSC_FALSE,
              atol_set = PETSC_FALSE, nsmooth_set = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, "-ksp_type", &ksp_set);
    PetscOptionsHasName(nullptr, nullptr, "-pc_type", &pc_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_type", &snes_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_linesearch_type", &ls_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_atol", &atol_set);
    PetscOptionsHasName(nullptr, nullptr, "-pc_gamg_agg_nsmooths", &nsmooth_set);
    if (!snes_set) PetscOptionsSetValue(nullptr, "-snes_type", "newtonls");            // modified Newton = Picard
    if (!ls_set)   PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");    // full-step (plain Picard)
    if (!ksp_set)  PetscOptionsSetValue(nullptr, "-ksp_type", "cg");                   // SPD inner solve
    if (!pc_set)   PetscOptionsSetValue(nullptr, "-pc_type", "gamg");                  // algebraic multigrid
    // Unsmoothed aggregation -> a reliably-SPD GAMG preconditioner. Smoothed aggregation
    // (the default) can produce a slightly INDEFINITE preconditioner as the operator turns
    // diffusion-dominated at large dt (BDF2 / adaptive), which makes CG bail with
    // DIVERGED_INDEFINITE_PC. Unsmoothed fixes that at no measured cost here (same ~2 inner
    // iterations on the elliptic operator). Overridable.
    if (!nsmooth_set) PetscOptionsSetValue(nullptr, "-pc_gamg_agg_nsmooths", "0");
    // Absolute residual tolerance so an already-converged (near-equilibrium) step stops
    // instead of chasing a RELATIVE reduction on a machine-zero residual -> SNES max-its
    // -> spurious "not converged" throw. The mid-transient residual norm (~S*h*sqrt(N),
    // 1e3 and up) is far above 1e-6, so this only fires at true equilibrium; it cannot
    // stop a real transient early. PETSc's default snes_atol (1e-50) effectively disables
    // this. Verified: default -> divergence after equilibrium; 1e-6 -> clean. Overridable.
    if (!atol_set) PetscOptionsSetValue(nullptr, "-snes_atol", "1e-6");
  } else if (user_context.use_newton) {
    // Newton-Krylov defaults. The analytic Jacobian (FormJacobianLocal, registered in update()) is
    // NON-symmetric (the dT/dw transmissivity-nonlinearity terms), so the inner solve is GMRES, not
    // CG. GAMG with unsmoothed aggregation preconditions the (near-elliptic) operator; a bt line
    // search globalizes from a far/cold start. snes_atol 1e-6 mirrors the Picard path (stop at a
    // machine-zero equilibrium residual instead of chasing a relative reduction). newtontr (trust
    // region) is the likely-more-robust alternative -- override with -snes_type newtontr. All set
    // only if the user did not, so runtime options win. Verify the Jacobian with -snes_test_jacobian
    // (needs -wtm_ksat_*_smoothing_width > 0 so the residual uses the smooth T that the tangent
    // differentiates); see FormJacobianLocal.
    PetscBool ksp_set = PETSC_FALSE, pc_set = PETSC_FALSE, snes_set = PETSC_FALSE, ls_set = PETSC_FALSE,
              atol_set = PETSC_FALSE, nsmooth_set = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, "-ksp_type", &ksp_set);
    PetscOptionsHasName(nullptr, nullptr, "-pc_type", &pc_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_type", &snes_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_linesearch_type", &ls_set);
    PetscOptionsHasName(nullptr, nullptr, "-snes_atol", &atol_set);
    PetscOptionsHasName(nullptr, nullptr, "-pc_gamg_agg_nsmooths", &nsmooth_set);
    if (!snes_set)    PetscOptionsSetValue(nullptr, "-snes_type", "newtonls");
    if (!ls_set)      PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    if (!ksp_set)     PetscOptionsSetValue(nullptr, "-ksp_type", "gmres");            // Jacobian is non-symmetric
    if (!pc_set)      PetscOptionsSetValue(nullptr, "-pc_type", "gamg");
    if (!nsmooth_set) PetscOptionsSetValue(nullptr, "-pc_gamg_agg_nsmooths", "0");
    if (!atol_set)    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-6");
  }

  // Anderson-accelerated GAMG-Picard: the OUTER SNES stays Anderson (default type; head-form residual is
  // registered in update()). Attach the GAMG-Picard solve as its NONLINEAR PRECONDITIONER -- allocate the
  // Picard operator and instantiate the NPC, defaulting it to a defect-correction Picard (newtonls + basic
  // line search, ONE sweep) with a CG+GAMG inner solve via -npc_-prefixed options that SNESSetFromOptions
  // applies below. The NPC's Picard callbacks are registered per solve in update().
  SNESSetFromOptions(user_context.snes);

  // Resolved-settings provenance: read the ACTUAL convergence tolerances/caps back from the SNES (after
  // SNESSetFromOptions) and log them, so the start-up record is authoritative rather than assuming PETSc's
  // per-type defaults (SNESANDERSON max_it defaults to 10000, NEWTONLS to 50). These map to the config's
  // solver.tolerance (snes_stol) and solver.max_iterations (snes_max_it).
  {
    SNESType  resolved_type;
    PetscReal r_atol, r_rtol, r_stol;
    PetscInt  r_maxit, r_maxf;
    SNESGetType(user_context.snes, &resolved_type);
    SNESGetTolerances(user_context.snes, &r_atol, &r_rtol, &r_stol, &r_maxit, &r_maxf);
    PetscPrintf(PETSC_COMM_WORLD,
                "solver: type=%s  tolerance(snes_stol)=%g  max_iterations(snes_max_it)=%d  (atol=%g rtol=%g)\n",
                resolved_type, (double)r_stol, (int)r_maxit, (double)r_atol, (double)r_rtol);
  }

  // -wtm_adaptive_restart: allocate the best-iterate carrier. The rho monitor + outer restart loop
  // live in update().
  if (user_context.use_adaptive_restart) {
    VecDuplicate(user_context.x, &user_context.ar_best_x);
    PetscOptionsGetReal(nullptr, nullptr, "-snes_stol", &user_context.ar_stol, nullptr);  // match the run's step tol
    PetscPrintf(PETSC_COMM_WORLD,
                "solver.anderson.restart: rho-triggered proactive Anderson restart (rho>%.2f for %d iters "
                "-> restart from best iterate; phase cap %d iters, <=%d restarts). Proactive vs the "
                "periodic default; generalizes to an unknown flail iteration (global scale). See #87.\n",
                (double)user_context.ar_rho_threshold, (int)user_context.ar_rho_patience,
                (int)user_context.ar_max_it, (int)user_context.ar_max_restarts);
  }
}
