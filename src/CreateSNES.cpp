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

  DMSetApplicationContext(user_context.da, &user_context);
  SNESSetDM(user_context.snes, user_context.da);

  // Default to Anderson mixing (matrix-free, robust for heterogeneous media).
  // Override at runtime: -snes_type newtonls -ksp_type gmres -pc_type gamg
  // to use Newton-Krylov with the analytic Jacobian and Picard preconditioner.
  SNESSetType(user_context.snes, SNESANDERSON);

  SNESSetFromOptions(user_context.snes);
}
