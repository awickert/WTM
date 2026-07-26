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

  // Default the Anderson window to 10 (PETSc default is 30). On this problem m=10
  // converges in the same iteration count as m=30 but with less vector work per
  // iteration (~10-15% faster); m=5 starts costing iterations at larger grids, so
  // 10 is the safe margin (benchmark/SOLVER_NOTES.md). Set only if the user did
  // not specify -snes_anderson_m, so a runtime override still wins.
  PetscBool anderson_m_set = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-snes_anderson_m", &anderson_m_set);
  if (!anderson_m_set) {
    PetscOptionsSetValue(nullptr, "-snes_anderson_m", "10");
  }

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
  user_context.use_bdf2_on_V   = (bdf2v_flag == PETSC_TRUE);
  user_context.use_dt_adaptive = (adaptive_flag == PETSC_TRUE);
  user_context.use_bdf2        = (bdf2_flag == PETSC_TRUE) || user_context.use_dt_adaptive || user_context.use_bdf2_on_V;
  user_context.use_picard      = (picard_flag == PETSC_TRUE) || user_context.use_bdf2;
  PetscBool smooth_T_flag = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_smooth_T", &smooth_T_flag);
  user_context.use_smooth_T = (smooth_T_flag == PETSC_TRUE);  // experiment: smooth T in the Picard operator
  if (user_context.use_dt_adaptive) {
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_dt_tol", &user_context.dt_tol, nullptr);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "-wtm_dt_adaptive set: BDF2 + adaptive dt (tol=%g m); enabling the Picard solver path.\n",
        user_context.dt_tol);
  } else if (user_context.use_bdf2 && picard_flag != PETSC_TRUE) {
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
