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
  PetscBool ar_here = PETSC_FALSE, ar_growm_here = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_adaptive_restart", &ar_here);
  PetscOptionsHasName(nullptr, nullptr, "-wtm_adaptive_grow_m", &ar_growm_here);
  const bool adaptive_here = (ar_here == PETSC_TRUE || ar_growm_here == PETSC_TRUE);
  if (!restart_type_set)
    PetscOptionsSetValue(nullptr, "-snes_anderson_restart_type", adaptive_here ? "none" : "periodic");
  if (!restart_period_set && !adaptive_here) PetscOptionsSetValue(nullptr, "-snes_anderson_restart", "20");

  // Step-tolerance default 1e-8. The damped default (beta=0.5) converges LINEARLY, so it stops right
  // AT the requested step tolerance rather than over-shooting it as the undamped solver does; at the
  // looser 1e-6 that left ~1e-6 (um-scale) rank-dependence in the water table (each rank converges to
  // its own 1e-6-accurate solution). 1e-8 is tight enough that the parallel solve is machine-consistent
  // (~1e-9 cross-rank) AND still reachable on steep terrain (Corsica converges in ~80 iters; a residual
  // -snes_atol criterion CANNOT be reached there -- the residual floors above 1e-8 -- so use the step
  // tolerance, which tracks solution change and is reachable). Set only if the user did not override.
  PetscBool snes_stol_set = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-snes_stol", &snes_stol_set);
  if (!snes_stol_set) PetscOptionsSetValue(nullptr, "-snes_stol", "1e-8");

  // Semi-implicit Picard path (experimental; PICARD_MG_DESIGN.md / PICARD_MATH.md).
  // Gated behind -wtm_picard so the default Anderson path is untouched. When on,
  // allocate the SPD operator A(x) (also its own GAMG preconditioner) and a residual
  // work vector, and default the outer/inner solvers (below) unless the user overrode
  // them.
  // Time-integration flags nest: -wtm_dt_adaptive implies BDF2 implies the Picard path
  // (all live in the Picard operator/RHS). See BDF2_ADAPTIVE_DESIGN.md.
  PetscBool picard_flag = PETSC_FALSE, bdf2_flag = PETSC_FALSE, adaptive_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_picard", &picard_flag);
  PetscOptionsHasName(nullptr, nullptr, "-wtm_bdf2", &bdf2_flag);
  PetscOptionsHasName(nullptr, nullptr, "-wtm_dt_adaptive", &adaptive_flag);
  PetscBool bdf2v_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_bdf2_on_V", &bdf2v_flag);

  // The DEFAULT solver is the semi-implicit, volume-form BDF2-on-V (Picard) path -- for BOTH run types.
  // EQUILIBRIUM: Picard's Newton+GAMG takes large, stable time steps whose cost is nearly dt-independent
  // (Corsica: ~28 SNES iters flat from dt=1 to 1000 yr), so it reaches steady state in a handful of big
  // steps. The matrix-free Anderson default it replaces has NO preconditioner and is stiffness-limited:
  // it diverges once dt is raised (Corsica: fails at 10 yr) -- exactly the "10x dt did not converge"
  // that motivated this. TRANSIENT: BDF2-on-V is genuinely 2nd-order in time; the matrix-free path
  // silently under-converges (and diverges when pushed) on stiff transient drainage. Anderson is kept
  // only for the rare small-dt / fast-science case, opt-in via -wtm_anderson (it is also matrix-free ->
  // bit-exact across ranks). Any explicit path flag (-wtm_picard / -wtm_bdf2 / -wtm_bdf2_on_V /
  // -wtm_dt_adaptive) takes precedence over this default.
  PetscBool force_anderson = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_anderson", &force_anderson);
  // -wtm_tr_bdf2: L-stable strong-damping 2nd-order on the matrix-free Anderson path (two staged solves
  // per step). Implies the Anderson path (self-starting; no Picard operator, no BDF2 history vector).
  PetscBool tr_bdf2_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_tr_bdf2", &tr_bdf2_flag);
  user_context.use_tr_bdf2 = (tr_bdf2_flag == PETSC_TRUE);
  if (tr_bdf2_flag) force_anderson = PETSC_TRUE;  // take the matrix-free Anderson path
  // -wtm_aa_picard: Anderson-accelerated GAMG-Picard via nonlinear preconditioning. OUTER = Anderson on
  // the head-form residual; the GAMG-Picard solve is the outer's NONLINEAR PRECONDITIONER (wired below +
  // in update()). Implies the Anderson main path (matrix-free head-form residual); the Picard operator is
  // allocated for the NPC.
  PetscBool aa_picard_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_aa_picard", &aa_picard_flag);
  user_context.use_aa_picard = (aa_picard_flag == PETSC_TRUE);
  if (aa_picard_flag) force_anderson = PETSC_TRUE;  // outer solver = matrix-free Anderson (head form)
  // -wtm_predict_guess: seed the initial guess (and thus iteration-1 T̄) with the 2nd-order history
  // extrapolation instead of w^n. Needs the w^{n-1} history carrier (below).
  PetscBool predict_guess_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_predict_guess", &predict_guess_flag);
  user_context.use_predict_guess = (predict_guess_flag == PETSC_TRUE);
  // -wtm_newton: opt-in true Newton-Krylov path (analytic Jacobian). Like -wtm_anderson it selects a
  // matrix-free (non-Picard) residual path, so it also suppresses the Picard default below.
  PetscBool newton_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_newton", &newton_flag);
  // -wtm_handoff: Anderson globalizes, then hands off its best iterate to a Newton/Picard finisher
  // near convergence (see AppCtx). Selects the matrix-free Anderson MAIN path (phase 1); the finisher
  // SNES is built after SNESSetFromOptions below. -wtm_handoff_picard uses a Picard (CG+GAMG) finisher
  // instead of the default Newton (GMRES+GAMG). The main Picard/Newton defaults stay untouched.
  PetscBool handoff_flag = PETSC_FALSE, handoff_picard_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_handoff", &handoff_flag);
  PetscOptionsHasName(nullptr, nullptr, "-wtm_handoff_picard", &handoff_picard_flag);
  if (handoff_picard_flag) handoff_flag = PETSC_TRUE;
  if (handoff_flag) force_anderson = PETSC_TRUE;  // phase 1 = matrix-free Anderson (main path)
  // -wtm_adaptive_restart: rho-triggered proactive Anderson restart (see AppCtx). Selects the Anderson
  // main path; the outer restart loop lives in update(). -wtm_adaptive_grow_m widens m on each restart.
  PetscBool adaptive_restart_flag = PETSC_FALSE, adaptive_grow_m_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_adaptive_restart", &adaptive_restart_flag);
  PetscOptionsHasName(nullptr, nullptr, "-wtm_adaptive_grow_m", &adaptive_grow_m_flag);
  if (adaptive_grow_m_flag) adaptive_restart_flag = PETSC_TRUE;
  if (adaptive_restart_flag) force_anderson = PETSC_TRUE;  // rho-adaptive is an Anderson strategy
  // -wtm_stiff: convenience bundle for hard equilibrium cold-starts on stiff terrain. It is shorthand for
  // "-wtm_newton -wtm_dt_continuation -wtm_eq_tol 0.01": the analytic-Jacobian Newton path, dt-continuation
  // (ramp dt from small so a far/cold guess stays in-basin), and a default convergence early-stop so the
  // run terminates at equilibrium without hand-tuning total_cycles. Each piece stays individually
  // overridable; an explicit Picard/Anderson path flag still takes precedence (Newton is exclusive with
  // Picard -- a warning is printed below if that happens). See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  PetscBool stiff_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_stiff", &stiff_flag);
  if (stiff_flag) newton_flag = PETSC_TRUE;  // select the Newton path (a Picard/Anderson flag overrides below)
  const bool any_path_flag = (picard_flag || bdf2_flag || adaptive_flag || bdf2v_flag);
  bool default_picard = false;
  if (!force_anderson && !newton_flag && !any_path_flag) {
    bdf2v_flag     = PETSC_TRUE;
    default_picard = true;
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Defaulting to the semi-implicit BDF2-on-V (Picard) solver: large stable time steps (fast to\n"
        "  equilibrium) and 2nd-order-in-time accuracy. Override with -wtm_anderson for the matrix-free\n"
        "  Anderson solver (rare: small-dt / fast-science / bit-exact-across-ranks cases).\n");
  }

  user_context.use_bdf2_on_V   = (bdf2v_flag == PETSC_TRUE);
  user_context.use_dt_adaptive = (adaptive_flag == PETSC_TRUE);
  // -wtm_dt_adaptive normally implies the BDF2 history (needed by the non-TR forward error estimator).
  // With TR-BDF2 the embedded estimator uses the two stages (no history), so don't force the Picard-BDF2
  // residual onto the tr+adaptive combo -- it keeps the self-contained matrix-free TR-BDF2 residual.
  user_context.use_bdf2 = (bdf2_flag == PETSC_TRUE)
                          || (user_context.use_dt_adaptive && !user_context.use_tr_bdf2)
                          || user_context.use_bdf2_on_V;
  // A forced Anderson path keeps the matrix-free residual even with a BDF2 time flag: -wtm_anderson
  // -wtm_bdf2_on_V gives 2nd-order-in-time Anderson (time discretization is a property of the residual,
  // not the solver). Only take the Picard operator path when Anderson is NOT forced.
  user_context.use_picard      = (picard_flag == PETSC_TRUE || user_context.use_bdf2) && force_anderson != PETSC_TRUE;
  // Newton path is exclusive with Picard (a path flag wins if the user set both).
  user_context.use_newton      = (newton_flag == PETSC_TRUE) && !user_context.use_picard;
  user_context.use_handoff     = (handoff_flag == PETSC_TRUE);
  user_context.handoff_picard  = (handoff_picard_flag == PETSC_TRUE);
  PetscOptionsGetInt(nullptr, nullptr, "-wtm_handoff_patience", &user_context.handoff_patience, nullptr);
  PetscOptionsGetInt(nullptr, nullptr, "-wtm_handoff_max_it", &user_context.handoff_max_it, nullptr);
  user_context.use_adaptive_restart = (adaptive_restart_flag == PETSC_TRUE);
  user_context.adaptive_grow_m      = (adaptive_grow_m_flag == PETSC_TRUE);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_ar_rho", &user_context.ar_rho_threshold, nullptr);
  PetscOptionsGetInt(nullptr, nullptr, "-wtm_ar_patience", &user_context.ar_rho_patience, nullptr);
  PetscOptionsGetInt(nullptr, nullptr, "-wtm_ar_max_it", &user_context.ar_max_it, nullptr);
  PetscOptionsGetInt(nullptr, nullptr, "-wtm_ar_max_restarts", &user_context.ar_max_restarts, nullptr);
  if (stiff_flag && !user_context.use_newton) {
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_stiff has no effect: an explicit Picard/Anderson path flag takes precedence over the\n"
                "  Newton path it selects. Drop the Picard/Anderson flag to use the stiff cold-start recipe.\n");
  }

  // Newton dt-continuation (-wtm_dt_continuation; needs -wtm_newton): equilibrium PTC that starts
  // deltat small (diagonally dominant -> non-singular Jacobian from a far guess) and grows it after
  // each converged step. Start dt defaults to params.deltat/200 (-wtm_dtc_dt0 overrides, seconds);
  // growth 1.5x/step (-wtm_dtc_grow); cap 1000*params.deltat (-wtm_dtc_dt_max). deltat persists across
  // cycles, so it ramps toward equilibrium. The WTM.cpp cycle loop drives the ramp. See
  // benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  PetscBool dtc_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_dt_continuation", &dtc_flag);
  if (stiff_flag) dtc_flag = PETSC_TRUE;  // the bundle enables dt-continuation
  user_context.use_newton_continuation = (dtc_flag == PETSC_TRUE) && user_context.use_newton;
  // Convergence-based early stop (-wtm_eq_tol, metres). Default ON for equilibrium runs (0.01 m ~ 1 cm of
  // max water-table change per ~1-yr cycle), OFF for transient runs (a time-evolution run must play out in
  // full, so it is never auto-stopped). Pass -wtm_eq_tol 0 to disable on an equilibrium run, or any value to
  // override. Parsed for ALL solver paths (was previously only inside the Newton block below, so it was
  // silently ignored on the default Anderson/Picard path). run() checks the PER-CYCLE change against it.
  PetscBool eq_tol_set = PETSC_FALSE;
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_eq_tol", &user_context.eq_tol, &eq_tol_set);  // [m]; 0 = off
  if (!eq_tol_set)
    user_context.eq_tol = (params.run_type == "equilibrium") ? 0.01 : 0.0;
  // -wtm_eq_metric max|rms|frac: how the per-cycle change is aggregated for the equilibrium stop. DEFAULT
  // frac (converged when < eq_frac of land cells exceed eq_tol) -- the measured best trade: MAX is
  // worst-cell-hostage (one slow deep cell keeps it from ever firing), RMS is loose (bulk only); frac both
  // fires and stays precise. See the oscillation diagnosis in benchmark/adaptive_dt. -wtm_eq_frac sets the
  // fraction threshold (default 0.1%). Pass -wtm_eq_metric max for the old strict worst-cell criterion.
  char eq_metric_str[16] = "frac";
  PetscOptionsGetString(nullptr, nullptr, "-wtm_eq_metric", eq_metric_str, sizeof(eq_metric_str), nullptr);
  if (std::strcmp(eq_metric_str, "rms") == 0) user_context.eq_metric = 1;
  else if (std::strcmp(eq_metric_str, "max") == 0) user_context.eq_metric = 0;
  else user_context.eq_metric = 2;  // "frac" (default)
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_eq_frac", &user_context.eq_frac, nullptr);
  if (user_context.use_newton_continuation) {
    double dt0 = params.deltat / 200.0;
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dtc_dt0", &dt0, nullptr);
    user_context.deltat = dt0;  // start small (overrides the params.deltat init above)
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dtc_grow", &user_context.dtc_grow, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dtc_shrink", &user_context.dtc_shrink, nullptr);
    user_context.dtc_dt_max = 1000.0 * params.deltat;
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dtc_dt_max", &user_context.dtc_dt_max, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-wtm_dtc_easy_iters", &user_context.dtc_easy_iters, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-wtm_dtc_max_retries", &user_context.dtc_max_retries, nullptr);
    // The bundle defaults the early-stop to 1 cm/step (gated on dt in WTM.cpp so it cannot fire during the
    // ramp); a user -wtm_eq_tol below still wins.
    if (stiff_flag && user_context.eq_tol == 0.0) user_context.eq_tol = 0.01;
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_dt_continuation: Newton PTC, dt0=%g s, grow x%g if <=%d iters, shrink x%g on reject, "
                "dt_max=%g s.\n",
                dt0, user_context.dtc_grow, user_context.dtc_easy_iters, user_context.dtc_shrink,
                user_context.dtc_dt_max);
    if (stiff_flag)
      PetscPrintf(PETSC_COMM_WORLD,
                  "-wtm_stiff: hard cold-start recipe active (Newton + dt-continuation + eq_tol=%g m early stop).\n",
                  user_context.eq_tol);
  }
  if (user_context.use_dt_adaptive) {
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dt_tol", &user_context.dt_tol, nullptr);
    PetscBool norm_rms = PETSC_FALSE;
    PetscOptionsHasName(nullptr, nullptr, "-wtm_dt_norm_rms", &norm_rms);
    user_context.dt_norm_rms = (norm_rms == PETSC_TRUE);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "-wtm_dt_adaptive set: adaptive dt (tol=%g m, %s norm)%s.\n",
        user_context.dt_tol, user_context.dt_norm_rms ? "RMS" : "MAX",
        user_context.use_tr_bdf2 ? " on the TR-BDF2 path" : "; enabling the Picard solver path");
  } else if (user_context.use_bdf2 && force_anderson == PETSC_TRUE) {
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_anderson + BDF2-on-V: 2nd-order-in-time matrix-free Anderson (BDF2-on-V residual, no\n"
                "  operator/preconditioner). Time-order decoupled from the solver.\n");
  } else if (user_context.use_bdf2 && picard_flag != PETSC_TRUE && !default_picard) {
    PetscPrintf(PETSC_COMM_WORLD, "-wtm_bdf2 set: enabling the Picard solver path (BDF2 requires it).\n");
  }
  // BDF2 history carrier (w^{n-1}) is needed on ANY BDF2 path -- the Picard operator OR the matrix-free
  // Anderson residual (-wtm_anderson -wtm_bdf2_on_V) -- and also by the predictor-seeded guess. Allocate
  // it whenever BDF2 or the predictor is on, independent of use_picard.
  if (user_context.use_bdf2 || user_context.use_predict_guess) {
    VecDuplicate(user_context.x, &user_context.starting_wtd_prev);
    VecSet(user_context.starting_wtd_prev, 0.0);
    user_context.bdf2_prev_dt = user_context.deltat;  // ω=1 until Δt changes (adaptive)
  }
  if (user_context.use_predict_guess)
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_predict_guess: seeding the initial guess (and iteration-1 T̄) with the 2nd-order\n"
                "  history extrapolation (guarded).\n");
  // tr_expl (explicit old-state flux+removal at w^n) is used by TR-BDF2's trapezoidal stage AND by the
  // predictor's first-step forward-Euler bootstrap, so allocate it for either.
  if (user_context.use_tr_bdf2 || user_context.use_predict_guess)
    VecDuplicate(user_context.x, &user_context.tr_expl);
  if (user_context.use_tr_bdf2) {
    VecDuplicate(user_context.x, &user_context.tr_ygamma);  // intermediate Y_gamma
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_tr_bdf2: L-stable, strongly-damped 2nd-order matrix-free Anderson (TR-BDF2; two staged\n"
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
  if (user_context.use_aa_picard) {
    DMCreateMatrix(user_context.da, &user_context.picard_A);
    VecDuplicate(user_context.x, &user_context.picard_r);
    const auto setdef = [](const char* key, const char* val) {
      PetscBool set = PETSC_FALSE;
      PetscOptionsHasName(nullptr, nullptr, key, &set);
      if (!set) PetscOptionsSetValue(nullptr, key, val);
    };
    setdef("-npc_snes_type", "newtonls");          // NPC = modified Newton on A(x) = defect-correction Picard
    setdef("-npc_snes_linesearch_type", "basic");  // full-step (plain Picard update)
    setdef("-npc_snes_max_it", "1");               // one GAMG-Picard sweep per outer Anderson step
    // GMRES (not CG) for the NPC inner solve: the GAMG preconditioner comes out slightly INDEFINITE here
    // (CG bails DIVERGED_INDEFINITE_PC), and GMRES does not require an SPD preconditioner. GAMG unsmoothed.
    setdef("-npc_ksp_type", "gmres");
    setdef("-npc_pc_type", "gamg");
    setdef("-npc_pc_gamg_agg_nsmooths", "0");
    SNES npc;
    SNESGetNPC(user_context.snes, &npc);            // instantiate the NPC (inherits the outer DM)
    // LEFT nonlinear preconditioning: Anderson accelerates the residual of the NPC-preconditioned map.
    // Measured to converge cold where RIGHT side diverges (DIVERGED_INNER).
    SNESSetNPCSide(user_context.snes, PC_LEFT);
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_aa_picard: Anderson-accelerated GAMG-Picard (outer Anderson on the head-form residual;\n"
                "  GAMG-Picard nonlinear preconditioner, one sweep/step). Experimental.\n");
  }

  SNESSetFromOptions(user_context.snes);

  // -wtm_handoff: build the phase-2 finisher SNES. It SHARES the DM (so the DM-registered
  // FormFunctionLocal / FormJacobianLocal / Picard operator in update() are seen automatically)
  // but carries its own prefixed options so it is a Newton (GMRES+GAMG) or Picard (CG+GAMG) solve
  // independent of the main Anderson SNES. Anderson hands off its best iterate to this near
  // convergence, where the tame near-fixed T makes Newton/Picard fast and stable.
  if (user_context.use_handoff) {
    VecDuplicate(user_context.x, &user_context.handoff_best_x);
    SNESCreate(PETSC_COMM_WORLD, &user_context.snes_finish);
    SNESSetDM(user_context.snes_finish, user_context.da);
    SNESSetOptionsPrefix(user_context.snes_finish, "finish_");
    const auto setf = [](const char* key, const char* val) { PetscOptionsSetValue(nullptr, key, val); };
    if (user_context.handoff_picard) {
      // Picard finisher: SPD defect-correction (newtonls + basic line search + CG/GAMG on A(x)).
      // Its operator is registered per solve via SNESSetPicard in update(); allocate A + r here.
      if (!user_context.picard_A) DMCreateMatrix(user_context.da, &user_context.picard_A);
      if (!user_context.picard_r) VecDuplicate(user_context.x, &user_context.picard_r);
      setf("-finish_snes_type", "newtonls");
      setf("-finish_snes_linesearch_type", "basic");
      setf("-finish_ksp_type", "cg");
      setf("-finish_pc_type", "gamg");
      setf("-finish_pc_gamg_agg_nsmooths", "0");
    } else {
      // Newton finisher: analytic Jacobian (registered on the DM in update()); non-symmetric -> GMRES.
      setf("-finish_snes_type", "newtonls");
      setf("-finish_snes_linesearch_type", "bt");
      setf("-finish_ksp_type", "gmres");
      setf("-finish_pc_type", "gamg");
      setf("-finish_pc_gamg_agg_nsmooths", "0");
    }
    setf("-finish_snes_atol", "1e-6");  // stop at the (tame) equilibrium residual, like the Newton/Picard paths
    SNESSetFromOptions(user_context.snes_finish);
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_handoff: Anderson globalizes -> %s finisher near convergence (patience=%d stalled iters, "
                "phase-1 cap=%d). Nonlinear preconditioning; see #87.\n",
                user_context.handoff_picard ? "Picard(CG+GAMG)" : "Newton(GMRES+GAMG)",
                (int)user_context.handoff_patience, (int)user_context.handoff_max_it);
  }

  // -wtm_adaptive_restart: allocate the best-iterate carrier and seed the (possibly growing) window
  // from the Anderson m default. The rho monitor + outer restart loop live in update().
  if (user_context.use_adaptive_restart) {
    VecDuplicate(user_context.x, &user_context.ar_best_x);
    PetscInt m0 = 10;
    PetscOptionsGetInt(nullptr, nullptr, "-snes_anderson_m", &m0, nullptr);
    user_context.ar_current_m = m0;
    PetscOptionsGetReal(nullptr, nullptr, "-snes_stol", &user_context.ar_stol, nullptr);  // match the run's step tol
    PetscPrintf(PETSC_COMM_WORLD,
                "-wtm_adaptive_restart: rho-triggered proactive Anderson restart (rho>%.2f for %d iters "
                "-> restart from best iterate; phase cap %d iters, <=%d restarts). Proactive vs the "
                "periodic default; generalizes to an unknown flail iteration (global scale). See #87.\n",
                (double)user_context.ar_rho_threshold, (int)user_context.ar_rho_patience,
                (int)user_context.ar_max_it, (int)user_context.ar_max_restarts);
    if (user_context.adaptive_grow_m)
      PetscPrintf(PETSC_COMM_WORLD,
                  "-wtm_adaptive_grow_m: DEFERRED (mid-solve m-change via SNESReset is unreliable -- SEGV / "
                  "undamped blow-up); falling back to restart-only, which already converges 139M. See #88.\n");
  }
}
