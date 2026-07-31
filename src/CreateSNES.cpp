//#include "CreateSNES.hpp"

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

  // TRANSIENT runs default to the 2nd-order-in-time, volume-form BDF2-on-V (Picard) path. On a
  // transient the time-discretization accuracy IS the answer, and BDF2-on-V is far more accurate per
  // step than the matrix-free Anderson head-form (~100x on the benchmark/picard drainage fixture) and
  // lets the run take much larger steps toward the target state. EQUILIBRIUM runs keep the faster
  // matrix-free Anderson default -- the steady state is independent of the time scheme, so Anderson's
  // speed wins with no accuracy cost. Override on a transient with -wtm_anderson to force the
  // matrix-free path; any explicit path flag (-wtm_picard / -wtm_bdf2 / -wtm_bdf2_on_V /
  // -wtm_dt_adaptive) also takes precedence over this default.
  PetscBool force_anderson = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_anderson", &force_anderson);
  const bool any_path_flag = (picard_flag || bdf2_flag || adaptive_flag || bdf2v_flag);
  bool transient_default_picard = false;
  if (params.run_type == "transient" && !force_anderson && !any_path_flag) {
    bdf2v_flag                = PETSC_TRUE;
    transient_default_picard  = true;
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Transient run: defaulting to the 2nd-order BDF2-on-V (Picard) solver for time accuracy.\n"
        "  Override with -wtm_anderson to force the faster 1st-order matrix-free Anderson path.\n");
  }

  user_context.use_bdf2_on_V   = (bdf2v_flag == PETSC_TRUE);
  user_context.use_dt_adaptive = (adaptive_flag == PETSC_TRUE);
  user_context.use_bdf2        = (bdf2_flag == PETSC_TRUE) || user_context.use_dt_adaptive || user_context.use_bdf2_on_V;
  user_context.use_picard      = (picard_flag == PETSC_TRUE) || user_context.use_bdf2;
  if (user_context.use_dt_adaptive) {
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dt_tol", &user_context.dt_tol, nullptr);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "-wtm_dt_adaptive set: BDF2 + adaptive dt (tol=%g m); enabling the Picard solver path.\n",
        user_context.dt_tol);
  } else if (user_context.use_bdf2 && picard_flag != PETSC_TRUE && !transient_default_picard) {
    PetscPrintf(PETSC_COMM_WORLD, "-wtm_bdf2 set: enabling the Picard solver path (BDF2 requires it).\n");
  }
  if (user_context.use_picard) {
    DMCreateMatrix(user_context.da, &user_context.picard_A);
    VecDuplicate(user_context.x, &user_context.picard_r);
    if (user_context.use_bdf2) {
      VecDuplicate(user_context.x, &user_context.starting_wtd_prev);
      VecSet(user_context.starting_wtd_prev, 0.0);
      user_context.bdf2_prev_dt = user_context.deltat;  // ω=1 until Δt changes (adaptive)
    }

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
  }

  SNESSetFromOptions(user_context.snes);
}
