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

  SNESSetFromOptions(user_context.snes);
}
