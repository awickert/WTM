#include "transient_groundwater.hpp"
#include "add_recharge.hpp"
#include "update_effective_storativity.hpp"

#include <omp.h>
#include <array>
#include <chrono>
#include <experimental/source_location>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscsnes.h>

///////////////////////
// PRIVATE FUNCTIONS //
///////////////////////

namespace FanDarcyGroundwater {

void PETSC_CHECK(
    const PetscErrorCode err,
    const std::experimental::source_location location = std::experimental::source_location::current()) {
  if (err) {
    throw std::runtime_error(
        "Petsc exception: " + std::to_string(err) + " at " + location.file_name() + ":" +
        std::to_string(location.line()));
  }
}

// get corners of arrays for individual processors
std::tuple<PetscInt, PetscInt, PetscInt, PetscInt> get_corners(const DM da) {
  PetscInt xs, ys, xm, ym;
  PETSC_CHECK(DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  return {xs, ys, xm, ym};
}

// declare functions
static PetscErrorCode FormRHS(AppCtx*, DM, Vec);
static PetscErrorCode FormInitialGuess(AppCtx*, DM, Vec);
static PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscScalar**, PetscScalar**, AppCtx*);
static PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscScalar**, Mat, Mat, AppCtx*);

// Semi-implicit Picard path (experimental; PICARD_MATH.md). Global SNES callbacks
// for SNESSetPicard: FormPicardRHS computes b(x), FormPicardOperator computes the
// SPD operator A(x). Gated behind -wtm_picard; default Anderson path unaffected.
static PetscErrorCode FormPicardRHS(SNES, Vec, Vec, void*);
static PetscErrorCode FormPicardOperator(SNES, Vec, Mat, Mat, void*);

//////////////////////
// PUBLIC FUNCTIONS //
//////////////////////

// Depth-integrated transmissivity. Two forms are kept side by side:
//
//   * depthIntegratedTransmissivity       -- the published Fan et al. (2013) S4/S6
//     PIECEWISE form; the PRODUCTION choice, used by the Anderson residual. Cheap:
//     an exp only for deep cells (wtd < -shallow); shallow and above-surface cells
//     are a single multiply. Profiling showed it is ~20% faster per core than the
//     smooth form below at identical iteration counts (~30% stacked with
//     -snes_anderson_m 5); see benchmark/SOLVER_NOTES.md.
//   * depthIntegratedTransmissivitySmooth -- a smooth (C-inf) blend of the same,
//     differentiable everywhere so it supports the analytic Jacobian
//     (dTransmissivityInverseDwtd) for a future Newton+multigrid path. Used by
//     FormJacobianLocal so residual/Jacobian stay consistent there; NOT used by
//     the Anderson production residual.
double depthIntegratedTransmissivity(const double wtd_T, const double fdepth, const double ksat) {
  constexpr double shallow = 1.5;
  // Global soil datasets include information for shallow soils.
  // if the water table is deeper than this, the permeability
  // of the soil sees an exponential decay with depth.
  if (fdepth <= 0) {
    // If the fdepth is zero, there is no water transmission below the surface
    // soil layer.
    // If it is less than zero, it is incorrect -- but no water transmission
    // also seems an okay thing to do in this case.
    return 0;
  } else if (wtd_T < -shallow) {  // Equation S6 from the Fan paper
    return std::max(0.0, fdepth * ksat * std::exp((wtd_T + shallow) / fdepth));
  } else if (wtd_T > 0) {
    // If wtd_T is greater than 0, max out rate of groundwater movement
    // as though wtd_T were 0. The surface water will get to move in
    // FillSpillMerge.
    return std::max(0.0, ksat * (0 + shallow + fdepth));
  } else {                                                    // Equation S4 from the Fan paper
    return std::max(0.0, ksat * (wtd_T + shallow + fdepth));  // max because you can't have a negative transmissivity.
  }
}

// Smooth (C-inf) depth-integrated transmissivity: a differentiable blend of the
// piecewise production form above. Kept for a future Newton path; its analytic
// derivative is dTransmissivityInverseDwtd, and FormJacobianLocal uses this
// version. NOT used by the Anderson production residual.
static double depthIntegratedTransmissivitySmooth(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0;
  constexpr double shallow = 1.5;
  constexpr double eps0    = 0.01;  // smooth clamping at WTD=0 boundary
  constexpr double eps1    = 0.01;  // smooth blend at WTD=-shallow boundary

  const double wtd_eff = (wtd_T - std::sqrt(wtd_T * wtd_T + eps0 * eps0)) * 0.5;
  const double u       = wtd_T + shallow;
  const double sigma_1 = 1.0 / (1.0 + std::exp(u / eps1));

  const double T_linear = ksat * (wtd_eff + shallow + fdepth);
  const double T_exp    = fdepth * ksat * std::exp(u / fdepth);

  return std::max(0.0, (1.0 - sigma_1) * T_linear + sigma_1 * T_exp);
}

// Analytic derivative of (1/T) with respect to wtd_T, matching the smooth T above.
static double dTransmissivityInverseDwtd(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  constexpr double eps0    = 0.01;
  constexpr double eps1    = 0.01;

  const double sq0      = std::sqrt(wtd_T * wtd_T + eps0 * eps0);
  const double wtd_eff  = (wtd_T - sq0) * 0.5;
  const double dwtd_eff = (1.0 - wtd_T / sq0) * 0.5;

  const double u       = wtd_T + shallow;
  const double sigma_1 = 1.0 / (1.0 + std::exp(u / eps1));
  const double dsigma1 = -sigma_1 * (1.0 - sigma_1) / eps1;

  const double T_linear = ksat * (wtd_eff + shallow + fdepth);
  const double T_exp    = fdepth * ksat * std::exp(u / fdepth);
  const double T        = std::max(0.0, (1.0 - sigma_1) * T_linear + sigma_1 * T_exp);
  if (T <= 0.0) return 0.0;

  const double dT = dsigma1 * (T_exp - T_linear)
                  + (1.0 - sigma_1) * ksat * dwtd_eff
                  + sigma_1 * ksat * std::exp(u / fdepth);
  return -dT / (T * T);
}

// Analytic derivative of S_eff with respect to my_new_wtd, matching the smooth S formula.
// Uses the same V(w) = [w(1+p) + sqrt(w²+eps²)(1-p)] / 2 construction.
static double dEffectiveStorativityDnew(
    const double my_original_wtd, const double my_new_wtd, const double my_porosity) {
  constexpr double eps = 0.01;
  const double dwtd    = my_new_wtd - my_original_wtd;

  const auto V = [&](double w) {
    return (w * (1.0 + my_porosity) + std::sqrt(w * w + eps * eps) * (1.0 - my_porosity)) * 0.5;
  };
  const auto Vprime = [&](double w) {
    return ((1.0 + my_porosity) + w * (1.0 - my_porosity) / std::sqrt(w * w + eps * eps)) * 0.5;
  };

  if (std::abs(dwtd) > 1e-10) {
    const double S = (V(my_new_wtd) - V(my_original_wtd)) / dwtd;
    return (Vprime(my_new_wtd) - S) / dwtd;
  }
  // Near convergence (new ≈ old): dS/d(new) → V''(old)/2
  const double w = my_original_wtd;
  return (1.0 - my_porosity) * eps * eps / (2.0 * std::pow(w * w + eps * eps, 1.5));
}

// The solve inputs are read from distributed DMDA arrays (indexed [y][x] over
// the owned range) rather than from full-grid arp arrays, so those arp arrays
// need not exist on non-root ranks: wtd from starting_wtd, recharge from
// rech_dist, land mask from mask, porosity from porosity_vec. cell_area is 1-D
// (Class-C) and stays replicated on all ranks. See DISTRIBUTED_ARP_DESIGN.md (2f-C).
void set_starting_values(
    ArrayPack& arp,
    PetscScalar** starting_wtd,
    PetscScalar** rech_dist,
    PetscScalar** mask,
    PetscScalar** porosity,
    PetscInt xs,
    PetscInt ys,
    PetscInt xm,
    PetscInt ym) {
  // no pragma because we're editing arp accumulators
  // Accumulate over this rank's OWNED cells only (DMDA owned range, which is
  // non-overlapping across ranks), so under MPI each ocean/recharge cell is
  // counted exactly once by its owner. total_loss_to_ocean_gw and
  // total_added_recharge are therefore per-rank partials; PrintValues reduces
  // them to global totals for reporting.
  // check to see if there is any non-zero water table in ocean
  // cells, and if so, record these values as changes to the ocean.
  for (int y = ys; y < ys + ym; y++) {
    for (int x = xs; x < xs + xm; x++) {
      if (mask[y][x] == 0) {
        if (starting_wtd[y][x] > 0)
          arp.total_loss_to_ocean_gw += starting_wtd[y][x] * arp.cell_area[y];
        else
          arp.total_loss_to_ocean_gw += starting_wtd[y][x] * arp.cell_area[y] * porosity[y][x];
        starting_wtd[y][x] = 0.;
      } else {
        double rech_count = rech_dist[y][x];
        if (starting_wtd[y][x] >= 0 && starting_wtd[y][x] + rech_dist[y][x] < 0)
          rech_count = -starting_wtd[y][x];

        arp.total_added_recharge += rech_count * arp.cell_area[y];
      }
    }
  }
}

int update(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  PetscInt its;                // iterations for convergence
  SNESConvergedReason reason;  // Check convergence

  // --- diagnostic: profile the non-PETSc O(N) overhead; appears in -log_view ---
  static PetscLogEvent EVENT_SETSTART = 0, EVENT_FULLREDUCE = 0;
  static bool events_registered = false;
  if (!events_registered) {
    PetscLogEventRegister("SetStartVals", 0, &EVENT_SETSTART);
    PetscLogEventRegister("FullGridReduce", 0, &EVENT_FULLREDUCE);
    events_registered = true;
  }

  // Get local array bounds
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);

  // compute any starting values needed for arrays (owned cells only).
  // wtd is carried in dmdapack.starting_wtd (populated once per cycle before the
  // maxiter loop, then maintained by the copy-back below), not in arp.wtd.
  PetscLogEventBegin(EVENT_SETSTART, 0, 0, 0, 0);
  set_starting_values(
      arp, dmdapack.starting_wtd, dmdapack.rech_dist, dmdapack.mask, dmdapack.porosity_vec, xs, ys, xm, ym);
  PetscLogEventEnd(EVENT_SETSTART, 0, 0, 0, 0);

//  values for storativity are reset each time; and recharge changes from one timestep to the next, so set these here
#pragma omp parallel for default(none) shared(arp, ys, ym, xs, xm, dmdapack, params) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      dmdapack.rech_vec[j][i] =
          add_recharge(dmdapack.rech_dist[j][i], dmdapack.starting_wtd[j][i], dmdapack.porosity_vec[j][i]);
    }
  }

  if (user_context.use_picard) {
    // Semi-implicit Picard path (PICARD_MATH.md).
    // PETSc solves A(x) x = b(x); FormPicardRHS supplies b(x) (so SNESSolve is
    // called with a NULL rhs), FormPicardOperator supplies the SPD A(x). A is its
    // own preconditioner (GAMG). Inner solve defaults to CG+GAMG (CreateSNES).
    SNESSetPicard(
        user_context.snes,
        user_context.picard_r,
        FormPicardRHS,
        user_context.picard_A,
        user_context.picard_A,
        FormPicardOperator,
        &user_context);

    FormInitialGuess(&user_context, user_context.da, user_context.x);
    SNESSolve(user_context.snes, nullptr, user_context.x);
  } else {
    // Set local function evaluation routine (always needed).
    DMDASNESSetFunctionLocal(
        user_context.da,
        INSERT_VALUES,
        (PetscErrorCode(*)(DMDALocalInfo*, void*, void*, void*))FormFunctionLocal,
        &user_context);

    // Register analytic Jacobian only for Newton-Krylov.
    // FormJacobianLocal accesses neighbor arrays from global (non-ghost) vectors,
    // which is safe only within a single MPI process partition boundary.
    // Registering it for Anderson causes PETSc to call it on divergence, triggering
    // a segfault under multi-process MPI. Skip it for Anderson (Jacobian unused).
    SNESType snes_type;
    SNESGetType(user_context.snes, &snes_type);
    if (std::string(snes_type) != std::string(SNESANDERSON)) {
      DMDASNESSetJacobianLocal(
          user_context.da,
          (PetscErrorCode(*)(DMDALocalInfo*, void*, Mat, Mat, void*))FormJacobianLocal,
          &user_context);
    }

    // Evaluate initial guess
    FormInitialGuess(&user_context, user_context.da, user_context.x);

    // set the RHS
    FormRHS(&user_context, user_context.da, user_context.b);
    // Solve nonlinear system
    SNESSolve(user_context.snes, user_context.b, user_context.x);
  }

  SNESGetIterationNumber(user_context.snes, &its);
  SNESGetConvergedReason(user_context.snes, &reason);

  PetscPrintf(
      PETSC_COMM_WORLD, "%s Number of nonlinear iterations = %" PetscInt_FMT "\n", SNESConvergedReasons[reason], its);

  if (reason != 2 && reason != 3 && reason != 4) {
    throw std::runtime_error("The SNES solver has not converged.");
  }

  // BDF2: before starting_wtd is overwritten with h^{n+1} below, save the current h^n
  // wtd as the next step's h^{n-1}. The first step captures h^0 and sets the history flag,
  // so BDF2 engages from the second step on (the first bootstraps with backward Euler).
  // (fsm_off / Phase A: history is continuous; Phase B will reset the flag after FSM.)
  if (user_context.use_bdf2) {
    PetscScalar** my_starting_wtd_prev;
    DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &my_starting_wtd_prev);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        my_starting_wtd_prev[j][i] = dmdapack.starting_wtd[j][i];
    DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &my_starting_wtd_prev);
    user_context.bdf2_have_history = true;
    user_context.bdf2_prev_dt      = user_context.deltat;  // this step's Δt becomes Δt_{n-1} next
  }

  // copy the result back into the distributed wtd carrier (starting_wtd), which
  // feeds the next solve in the maxiter loop and is assembled to arp.wtd once
  // per cycle by gather_wtd_to_all. Read topo/mask/porosity from DMDA arrays
  // (topo_vec is re-scattered each cycle in transient) so arp is not needed here.
  PetscScalar** my_topo;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      dmdapack.starting_wtd[j][i] = dmdapack.x[j][i] - my_topo[j][i];
      if (dmdapack.mask[j][i] == 0) {
        if (dmdapack.starting_wtd[j][i] > 0)
          arp.total_loss_to_ocean_gw += dmdapack.starting_wtd[j][i] * arp.cell_area[j];
        else
          arp.total_loss_to_ocean_gw += dmdapack.starting_wtd[j][i] * arp.cell_area[j] * dmdapack.porosity_vec[j][i];
        dmdapack.starting_wtd[j][i] = 0.;
      }
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);

  // The full wtd field is assembled once per cycle, after the maxiter loop, by
  // gather_wtd_to_all -- not here per solve (see benchmark/DISTRIBUTED_ARP_DESIGN.md).
  return 0;
}

// Assemble the full wtd field on every rank from each rank's owned cells of the
// distributed carrier (starting_wtd).
void gather_wtd_to_all(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.starting_wtd[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  // Gather to rank 0 only: wtd is consumed by the serial sections (FSM, recharge,
  // diagnostics, output), which all run on rank 0, and re-scattered to the solve
  // next cycle. Non-root ranks do not need the full field, so arp.wtd can be
  // rank-0-only.
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++)
        arp.wtd(i, j) = full[j * params.ncells_x + i];
}

// Gather the distributed per-cycle runoff (runoff_dist = runoff_ratio*rech) to rank-0
// arp.runoff, so the NEXT FillSpillMerge (rank 0) sees the recharge's runoff. Called only
// when runoff_ratio_on; otherwise the runoff is 0 and arp.runoff stays at FSM's own 0.
// Reuses the un-held wtd_global as the gather scratch (after gather_wtd_to_all has
// finished with it -- the two run sequentially). See DISTRIBUTED_ARP_DESIGN.md (2c).
void gather_runoff_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.runoff_dist[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++)
        arp.runoff(i, j) = full[j * params.ncells_x + i];
}

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
static PetscErrorCode FormInitialGuess(AppCtx* user_context, DM da, Vec X) {
  PetscScalar **x, **my_starting_wtd, **my_topo;

  DMDAVecGetArray(da, X, &x);
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));

  const auto [xs, ys, xm, ym] = get_corners(da);

#pragma omp parallel for default(none) shared(my_starting_wtd, my_topo, ys, ym, xs, xm, x) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      x[j][i] = my_starting_wtd[j][i] + my_topo[j][i];  // when land mask == 0, both topo and wtd have already been set
                                                        // to 0 elsewhere, so no need for another if statement here
    }
  }

  DMDAVecRestoreArray(da, X, &x);
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  return 0;
}

/*
   FormRHS - Forms constant RHS for the problem.

   Input Parameters:
   user - user-defined application context
   B - RHS vector

   Output Parameter:
   B - vector
 */
static PetscErrorCode FormRHS(AppCtx* user_context, DM da, Vec B) {
  PetscScalar **b, **my_starting_wtd, **my_topo;

  DMDAVecGetArray(da, B, &b);
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));

  const auto [xs, ys, xm, ym] = get_corners(da);

#pragma omp parallel for default(none) shared(ys, ym, xs, xm, b, my_starting_wtd, my_topo) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      b[j][i] = my_starting_wtd[j][i] + my_topo[j][i];  // when land mask == 0, both topo and wtd have already been set
                                                        // to 0 elsewhere, so no need for another if statement here
    }
  }
  DMDAVecRestoreArray(da, B, &b);
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));

  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo* info, PetscScalar** x, PetscScalar** f, AppCtx* user_context) {
  DM da = user_context->da;
  PetscScalar **cellsize_ew_sq, **my_mask, **my_fdepth, **my_ksat, **my_topo, **my_rech, **my_T, **my_starting_wtd,
      **my_porosity;

  /*
    Compute function over the locally owned part of the grid.
    topo/fdepth/ksat/T use local ghost vectors so neighbor accesses [j][i±1] are valid under MPI.
  */
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));

  // Compute 1/T over the full ghost range so neighbor lookups in the owned-range loop below are valid.
#pragma omp parallel for default(none) shared(info, my_T, x, my_topo, my_fdepth, my_ksat) collapse(2)
  for (auto j = info->gys; j < info->gys + info->gym; j++) {
    for (auto i = info->gxs; i < info->gxs + info->gxm; i++) {
      my_T[j][i] = 1. / depthIntegratedTransmissivity(x[j][i] - my_topo[j][i], my_fdepth[j][i], my_ksat[j][i]);
    }
  }

#pragma omp parallel for default(none)                                                                              \
    shared(info, cellsize_ew_sq, x, my_T, my_mask, my_rech, user_context, my_porosity, my_starting_wtd, my_topo, f) \
        collapse(2)
  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      if (my_mask[j][i] == 0) {
        // Dirichlet condition: x = 0 for ocean cells (topo = wtd = 0 there).
        // Writing f = x rather than f = 0 gives a unit Jacobian diagonal,
        // which is required for the Newton-Krylov linear solve to be
        // non-singular. Anderson is unaffected: x starts at 0 and stays at 0.
        f[j][i] = x[j][i];
      } else {
        double this_x          = x[j][i];
        double this_T          = my_T[j][i];
        const PetscScalar ux_E = (x[j][i + 1] - this_x);
        const PetscScalar ux_W = (this_x - x[j][i - 1]);
        const PetscScalar uy_N = (x[j + 1][i] - this_x);
        const PetscScalar uy_S = (this_x - x[j - 1][i]);
        const PetscScalar e_E  = 2. / (this_T + my_T[j][i + 1]);  //harmonic means
        const PetscScalar e_W  = 2. / (this_T + my_T[j][i - 1]);
        const PetscScalar e_N  = 2. / (this_T + my_T[j + 1][i]);
        const PetscScalar e_S  = 2. / (this_T + my_T[j - 1][i]);

        const PetscScalar uxx = (e_W * ux_W - e_E * ux_E) / user_context->cellsize_NS_squared;
        const PetscScalar uyy = (e_S * uy_S - e_N * uy_N) / cellsize_ew_sq[j][i];

        double my_storativity =
            updateEffectiveStorativity(my_starting_wtd[j][i], this_x - my_topo[j][i], my_porosity[j][i]);

        f[j][i] = (uxx + uyy) * user_context->deltat / my_storativity + this_x - my_rech[j][i];
        // my_rech is converted to appropriate recharge for this timestep and starting water
        // table outside of the solve.
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));

  PetscLogFlops(info->xm * info->ym * (72.0));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormJacobianLocal - Analytic 5-point Jacobian of FormFunctionLocal.

   For ocean cells (mask == 0): J = I (unit diagonal for Dirichlet).
   For land cells: differentiates
       f = (uxx + uyy) * dt/S + x - rech
   analytically through the smooth transmissivity T(x) and storativity S(x).
   All smoothing constants must match those in depthIntegratedTransmissivitySmooth
   and updateEffectiveStorativity.

   NOTE: neighbor arrays (fdepth, ksat, topo) are accessed via global DM
   vectors; for single-process runs this is always safe.  A future MPI
   extension should scatter those arrays to local vectors with ghosts first.
 */
static PetscErrorCode FormJacobianLocal(
    DMDALocalInfo* info, PetscScalar** x, Mat Jmat, Mat P, AppCtx* user_context) {
  DM           da = user_context->da;
  PetscScalar **cellsize_ew_sq, **my_mask, **my_fdepth, **my_ksat, **my_topo, **my_porosity, **my_starting_wtd;

  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_vec, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_vec, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));

  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      MatStencil row;
      row.j = j; row.i = i; row.c = 0;

      if (my_mask[j][i] == 0) {
        const PetscScalar one = 1.0;
        MatStencil col;
        col.j = j; col.i = i; col.c = 0;
        MatSetValuesStencil(Jmat, 1, &row, 1, &col, &one, INSERT_VALUES);
        MatSetValuesStencil(P,    1, &row, 1, &col, &one, INSERT_VALUES);
      } else {
        // WTD at center and 4 neighbours
        const double wtd_c = x[j][i]     - my_topo[j][i];
        const double wtd_E = x[j][i + 1] - my_topo[j][i + 1];
        const double wtd_W = x[j][i - 1] - my_topo[j][i - 1];
        const double wtd_N = x[j + 1][i] - my_topo[j + 1][i];
        const double wtd_S = x[j - 1][i] - my_topo[j - 1][i];

        // 1/T at centre and 4 neighbours; cap at 1e30 when T ≈ 0
        const auto T_inv = [](double T) { return T > 0.0 ? 1.0 / T : 1e30; };
        const double Tinv_c = T_inv(depthIntegratedTransmissivitySmooth(wtd_c, my_fdepth[j][i],     my_ksat[j][i]));
        const double Tinv_E = T_inv(depthIntegratedTransmissivitySmooth(wtd_E, my_fdepth[j][i + 1], my_ksat[j][i + 1]));
        const double Tinv_W = T_inv(depthIntegratedTransmissivitySmooth(wtd_W, my_fdepth[j][i - 1], my_ksat[j][i - 1]));
        const double Tinv_N = T_inv(depthIntegratedTransmissivitySmooth(wtd_N, my_fdepth[j + 1][i], my_ksat[j + 1][i]));
        const double Tinv_S = T_inv(depthIntegratedTransmissivitySmooth(wtd_S, my_fdepth[j - 1][i], my_ksat[j - 1][i]));

        // d(1/T)/dwtd at centre and 4 neighbours (needed for full Jacobian only)
        const double dTinv_c = dTransmissivityInverseDwtd(wtd_c, my_fdepth[j][i],     my_ksat[j][i]);
        const double dTinv_E = dTransmissivityInverseDwtd(wtd_E, my_fdepth[j][i + 1], my_ksat[j][i + 1]);
        const double dTinv_W = dTransmissivityInverseDwtd(wtd_W, my_fdepth[j][i - 1], my_ksat[j][i - 1]);
        const double dTinv_N = dTransmissivityInverseDwtd(wtd_N, my_fdepth[j + 1][i], my_ksat[j + 1][i]);
        const double dTinv_S = dTransmissivityInverseDwtd(wtd_S, my_fdepth[j - 1][i], my_ksat[j - 1][i]);

        // Harmonic-mean conductances and their sums
        const double sumE = Tinv_c + Tinv_E,  e_E = 2.0 / sumE;
        const double sumW = Tinv_c + Tinv_W,  e_W = 2.0 / sumW;
        const double sumN = Tinv_c + Tinv_N,  e_N = 2.0 / sumN;
        const double sumS = Tinv_c + Tinv_S,  e_S = 2.0 / sumS;

        // Head differences
        const double ux_E = x[j][i + 1] - x[j][i];
        const double ux_W = x[j][i]     - x[j][i - 1];
        const double uy_N = x[j + 1][i] - x[j][i];
        const double uy_S = x[j][i]     - x[j - 1][i];

        // Storativity at center and its derivative w.r.t. x[j,i]
        const double S         = updateEffectiveStorativity(my_starting_wtd[j][i], wtd_c, my_porosity[j][i]);
        const double dS_dnew   = dEffectiveStorativityDnew(my_starting_wtd[j][i], wtd_c, my_porosity[j][i]);
        const double dt_over_S = user_context->deltat / S;
        const double cns2      = user_context->cellsize_NS_squared;
        const double cew2      = cellsize_ew_sq[j][i];

        // uxx + uyy needed for the storativity part of the diagonal
        const double uxx = (e_W * ux_W - e_E * ux_E) / cns2;
        const double uyy = (e_S * uy_S - e_N * uy_N) / cew2;

        // ∂e_X/∂x[neighbour] = -2·dTinv_X / sumX²
        const double de_E_dxE = -2.0 * dTinv_E / (sumE * sumE);
        const double de_W_dxW = -2.0 * dTinv_W / (sumW * sumW);
        const double de_N_dxN = -2.0 * dTinv_N / (sumN * sumN);
        const double de_S_dxS = -2.0 * dTinv_S / (sumS * sumS);

        // ∂e_X/∂x[j,i] = -2·dTinv_c / sumX²  (centre changes all four conductances)
        const double de_E_dxc = -2.0 * dTinv_c / (sumE * sumE);
        const double de_W_dxc = -2.0 * dTinv_c / (sumW * sumW);
        const double de_N_dxc = -2.0 * dTinv_c / (sumN * sumN);
        const double de_S_dxc = -2.0 * dTinv_c / (sumS * sumS);

        // Full analytic Jacobian off-diagonal entries
        const double J_east  = -(de_E_dxE * ux_E + e_E) * dt_over_S / cns2;
        const double J_west  = (de_W_dxW * ux_W - e_W) * dt_over_S / cns2;
        const double J_north = -(de_N_dxN * uy_N + e_N) * dt_over_S / cew2;
        const double J_south = (de_S_dxS * uy_S - e_S) * dt_over_S / cew2;

        const double d_uxx_dc = (de_W_dxc * ux_W + e_W - de_E_dxc * ux_E + e_E) / cns2;
        const double d_uyy_dc = (de_S_dxc * uy_S + e_S - de_N_dxc * uy_N + e_N) / cew2;
        const double J_center = (d_uxx_dc + d_uyy_dc) * dt_over_S
                              - (uxx + uyy) * user_context->deltat * dS_dnew / (S * S)
                              + 1.0;

        // Symmetric Picard preconditioner: freeze T, average S between neighbors.
        // P[i,j][i+1,j] = P[i+1,j][i,j] by construction → symmetric → GAMG-compatible.
        const double S_E = updateEffectiveStorativity(my_starting_wtd[j][i+1], wtd_E, my_porosity[j][i+1]);
        const double S_W = updateEffectiveStorativity(my_starting_wtd[j][i-1], wtd_W, my_porosity[j][i-1]);
        const double S_N = updateEffectiveStorativity(my_starting_wtd[j+1][i], wtd_N, my_porosity[j+1][i]);
        const double S_S = updateEffectiveStorativity(my_starting_wtd[j-1][i], wtd_S, my_porosity[j-1][i]);

        const double P_east  = -e_E * user_context->deltat / (0.5 * (S + S_E) * cns2);
        const double P_west  = -e_W * user_context->deltat / (0.5 * (S + S_W) * cns2);
        const double P_north = -e_N * user_context->deltat / (0.5 * (S + S_N) * cew2);
        const double P_south = -e_S * user_context->deltat / (0.5 * (S + S_S) * cew2);
        // Diagonal = -(sum of off-diagonals) + 1; strictly diagonally dominant → SPD.
        const double P_center = -(P_east + P_west + P_north + P_south) + 1.0;

        MatStencil  cols[5];
        PetscScalar vals[5];
        cols[0].j = j;     cols[0].i = i + 1; cols[0].c = 0;
        cols[1].j = j;     cols[1].i = i - 1; cols[1].c = 0;
        cols[2].j = j + 1; cols[2].i = i;     cols[2].c = 0;
        cols[3].j = j - 1; cols[3].i = i;     cols[3].c = 0;
        cols[4].j = j;     cols[4].i = i;     cols[4].c = 0;

        vals[0] = J_east; vals[1] = J_west; vals[2] = J_north; vals[3] = J_south; vals[4] = J_center;
        MatSetValuesStencil(Jmat, 1, &row, 5, cols, vals, INSERT_VALUES);

        vals[0] = P_east; vals[1] = P_west; vals[2] = P_north; vals[3] = P_south; vals[4] = P_center;
        MatSetValuesStencil(P, 1, &row, 5, cols, vals, INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(Jmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Jmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_vec, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_vec, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormPicardRHS - right-hand side b(x) of the Picard system A(x) x = b(x)
   (PICARD_MATH.md sec 4). The production residual divides the whole flux
   divergence by the CENTRE storativity S_c; that makes the natural operator
   nonsymmetric, so each row is scaled by S_c to symmetrize it (this leaves the
   solution unchanged -- row scaling by a positive constant). The RHS carries the
   same S_c factor: on land, b = S_c * (starting_wtd + topo + rech); on ocean,
   b = 0 (Dirichlet h = 0). S_c depends on the current head, so b(x) genuinely
   depends on x here (S is frozen at the outer iterate, PETSc's SNESFunctionFn).
 */
static PetscErrorCode FormPicardRHS(SNES snes, Vec x, Vec b, void* ctx) {
  (void)snes;
  AppCtx* user_context = static_cast<AppCtx*>(ctx);
  DM      da           = user_context->da;
  PetscScalar **bb, **xx, **my_starting_wtd, **my_topo, **my_rech, **my_porosity, **my_mask;

  // Variable-step BDF2 (once an h^{n-1} exists): b = S_c*(b_c*h^n - c_c*h^{n-1} + rech) with
  // omega = dt_n/dt_{n-1}, b_c = 1+omega, c_c = omega^2/(1+omega) (b_c=2, c_c=1/2 when the
  // step is constant -> uniform BDF2). Backward Euler otherwise: b = S_c*(h^n + rech).
  // h^{n-1} = starting_wtd_prev + topo (centre only).
  const bool bdf2 = user_context->use_bdf2 && user_context->bdf2_have_history;
  double b_c = 0.0, c_c = 0.0;
  if (bdf2) {
    const double omega = user_context->deltat / user_context->bdf2_prev_dt;
    b_c                = 1.0 + omega;
    c_c                = omega * omega / (1.0 + omega);
  }
  PetscScalar** my_starting_wtd_prev = nullptr;

  PetscCall(DMDAVecGetArray(da, b, &bb));
  PetscCall(DMDAVecGetArray(da, x, &xx));  // owned range: S_c is a centre-cell quantity
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  if (bdf2) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));

  const auto [xs, ys, xm, ym] = get_corners(da);
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      if (my_mask[j][i] == 0) {
        bb[j][i] = 0.0;  // Dirichlet ocean cell: h = 0
      } else {
        const double S_c =
            updateEffectiveStorativity(my_starting_wtd[j][i], xx[j][i] - my_topo[j][i], my_porosity[j][i]);
        const double h_n = my_starting_wtd[j][i] + my_topo[j][i];
        if (bdf2) {
          const double h_nm1 = my_starting_wtd_prev[j][i] + my_topo[j][i];
          bb[j][i] = S_c * (b_c * h_n - c_c * h_nm1 + my_rech[j][i]);
        } else {
          bb[j][i] = S_c * (h_n + my_rech[j][i]);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, b, &bb));
  PetscCall(DMDAVecRestoreArray(da, x, &xx));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  if (bdf2) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormPicardOperator - the SPD operator A(x) of the Picard system A(x) x = b(x)
   (PICARD_MATH.md sec 4). It is the production backward-Euler operator

       (row c)   S_c*x_c + dt * sum_nbr e_{c,nbr} (x_c - x_nbr)

   i.e. the CENTRE-storativity discretization of the Anderson residual, ROW-SCALED
   by S_c so it is symmetric (the flux term dt*e is symmetric in the cell pair; the
   1/S_c that would otherwise multiply it -- and break symmetry -- is cleared by
   the scaling). The RHS carries the matching S_c factor (FormPicardRHS), so the
   scaling cancels and the fixed point is exactly the Anderson one. e is the
   harmonic mean of the PIECEWISE transmissivity. Diagonal = S_c + sum(dt*e) is
   strictly dominant -> SPD -> CG-compatible.

   Ocean (Dirichlet) rows/columns are eliminated symmetrically with
   MatZeroRowsColumnsStencil after assembly; h_ocean = 0, so no RHS correction is
   needed (x = b = NULL). This keeps each land cell's drain-to-ocean conductance in
   its diagonal while removing the asymmetric off-diagonal (PICARD_MATH.md 4.4).

   Only the harmonic-mean T needs neighbor values, so the iterate x is ghost-
   scattered here and read with topo/fdepth/ksat from their *_local ghost vectors;
   the centre-only S_c reads starting_wtd/porosity owned. A and P are the same
   matrix (A preconditions itself via GAMG).
 */
static PetscErrorCode FormPicardOperator(SNES snes, Vec x, Mat A, Mat P, void* ctx) {
  (void)snes;
  (void)P;  // A is its own preconditioner
  AppCtx* user_context = static_cast<AppCtx*>(ctx);
  DM      da           = user_context->da;

  // Ghost-scatter the current iterate so neighbor heads (for T) are valid under MPI.
  Vec xloc;
  PetscCall(DMGetLocalVector(da, &xloc));
  PetscCall(DMGlobalToLocalBegin(da, x, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(da, x, INSERT_VALUES, xloc));

  PetscScalar **xx, **my_topo, **my_fdepth, **my_ksat, **my_porosity, **my_starting_wtd, **my_mask, **cellsize_ew_sq,
      **my_T;
  PetscCall(DMDAVecGetArray(da, xloc, &xx));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));      // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));  // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(da, &info));

  // 1/T (piecewise production form) over the full ghost range so the neighbor
  // harmonic means on the owned range are valid. Mirrors FormFunctionLocal.
  for (auto j = info.gys; j < info.gys + info.gym; j++) {
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++) {
      my_T[j][i] = 1.0 / depthIntegratedTransmissivity(xx[j][i] - my_topo[j][i], my_fdepth[j][i], my_ksat[j][i]);
    }
  }

  const double dt   = user_context->deltat;
  const double cns2 = user_context->cellsize_NS_squared;

  // Variable-step BDF2 (once an h^{n-1} exists): a*h^{n+1} - b*h^n + c*h^{n-1} = dt*RHS,
  // with omega = dt_n/dt_{n-1}, a = (1+2w)/(1+w) [here], b,c on the RHS. The diffusion term
  // always carries the current dt; the storage diagonal carries a*S_c (a=3/2 when the step is
  // constant, i.e. w=1 -> uniform BDF2). Backward Euler is a=1. See BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2 = user_context->use_bdf2 && user_context->bdf2_have_history;
  double a_coeff  = 1.0;  // coefficient of h^{n+1} on the S_c diagonal (BE)
  if (bdf2) {
    const double omega = dt / user_context->bdf2_prev_dt;
    a_coeff            = (1.0 + 2.0 * omega) / (1.0 + omega);
  }

  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      const MatStencil row = {.k = 0, .j = j, .i = i, .c = 0};

      if (my_mask[j][i] == 0) {
        // Ocean: placeholder diagonal; MatZeroRowsColumnsStencil fixes it to identity.
        const PetscScalar one = 1.0;
        PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, &one, INSERT_VALUES));
      } else {
        const double cew2 = cellsize_ew_sq[j][i];

        // Centre storativity, frozen at the current x (matches FormFunctionLocal).
        const double S_c =
            updateEffectiveStorativity(my_starting_wtd[j][i], xx[j][i] - my_topo[j][i], my_porosity[j][i]);

        // Harmonic-mean interface conductances: 2 / (1/T_c + 1/T_nbr).
        const double e_E = 2.0 / (my_T[j][i] + my_T[j][i + 1]);
        const double e_W = 2.0 / (my_T[j][i] + my_T[j][i - 1]);
        const double e_N = 2.0 / (my_T[j][i] + my_T[j + 1][i]);
        const double e_S = 2.0 / (my_T[j][i] + my_T[j - 1][i]);

        // Row-scaled-by-S_c operator: off-diagonals are the (symmetric) flux terms
        // dt*e/h^2; diagonal is a_coeff*S_c + sum of the fluxes. RHS carries the matching
        // S_c. (BE: a_coeff=1; uniform BDF2: a_coeff=3/2.)
        const double A_east   = -e_E * dt / cns2;
        const double A_west   = -e_W * dt / cns2;
        const double A_north  = -e_N * dt / cew2;
        const double A_south  = -e_S * dt / cew2;
        const double A_center = a_coeff * S_c - (A_east + A_west + A_north + A_south);

        // 5-point stencil: east, west, north, south, centre.
        const MatStencil cols[5] = {
            {.k = 0, .j = j,     .i = i + 1, .c = 0},  // east
            {.k = 0, .j = j,     .i = i - 1, .c = 0},  // west
            {.k = 0, .j = j + 1, .i = i,     .c = 0},  // north
            {.k = 0, .j = j - 1, .i = i,     .c = 0},  // south
            {.k = 0, .j = j,     .i = i,     .c = 0},  // centre
        };
        const PetscScalar vals[5] = {
            A_east,
            A_west,
            A_north,
            A_south,
            A_center,
        };
        PetscCall(MatSetValuesStencil(A, 1, &row, 5, cols, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Symmetric Dirichlet elimination on ocean cells (see doc comment above).
  std::vector<MatStencil> ocean_rows;
  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      if (my_mask[j][i] == 0) {
        ocean_rows.push_back({.k = 0, .j = j, .i = i, .c = 0});
      }
    }
  }
  PetscCall(MatZeroRowsColumnsStencil(
      A, static_cast<PetscInt>(ocean_rows.size()), ocean_rows.data(), 1.0, nullptr, nullptr));

  PetscCall(DMDAVecRestoreArray(da, xloc, &xx));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  PetscCall(DMRestoreLocalVector(da, &xloc));
  return 0;
}

}  // namespace FanDarcyGroundwater
