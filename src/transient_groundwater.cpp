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
// Retained but no longer dispatched (the Newton-Krylov path is disabled in update()); kept so a
// Newton solver can be rebuilt. [[maybe_unused]] silences the now-uncalled-function warning.
[[maybe_unused]] static PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscScalar**, Mat, Mat, AppCtx*);

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
//     differentiable everywhere. Its analytic derivative (dTransmissivityInverseDwtd) is the
//     Newton-Krylov Jacobian's T term (FormJacobianLocal) -- a differentiable, INEXACT-Newton
//     approximation of the residual, since the Newton residual (FormFunctionLocal) itself uses
//     the PIECEWISE T. Also used by the Picard operator when a -wtm_ksat_*_smoothing_width is
//     set. NOT used by the Anderson production residual.
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
  } else if (wtd_T > 0 && !g_extended_soil) {
    // If wtd_T is greater than 0, max out rate of groundwater movement
    // as though wtd_T were 0. The surface water will get to move in
    // FillSpillMerge. (Extended-soil skips this clamp: the S4 form continues past the surface.)
    return std::max(0.0, ksat * (0 + shallow + fdepth));
  } else {                                                    // Equation S4 from the Fan paper (extended: also wtd>0)
    return std::max(0.0, ksat * (wtd_T + shallow + fdepth));  // max because you can't have a negative transmissivity.
  }
}

// Conductivity smoothing widths (metres) for the two kinks in the piecewise (C0) Fan
// transmissivity, each independent and each defaulting to 0 => sharp at that boundary:
//   * g_ksat_soilbottom_smoothing_width (-wtm_ksat_soilbottom_smoothing_width): the -1.5 m
//     soil-bottom transition, where conductivity switches from constant (shallow soil) to
//     exponential decay with depth.
//   * g_ksat_surface_smoothing_width (-wtm_ksat_surface_smoothing_width): the 0 m land-surface
//     clamp, where the water table reaches the surface and transmissivity is capped.
// The Picard operator uses the exact piecewise Fan T when both are 0 (production); if either is
// positive it uses the smooth (C-inf) form below with the respective bands (a positive width
// shifts the fixed point further from the piecewise Fan form).
static double g_ksat_soilbottom_smoothing_width = 0.0;  // eps1: -1.5 m conductivity transition
static double g_ksat_surface_smoothing_width    = 0.0;  // eps0: 0 m surface clamp

// --- Sub-surface surface-water sink (-wtm_surface_sink; WIP prototype) ---------------------------
// A smooth, compact-support removal in a band just BELOW the land surface that holds the water
// table strictly sub-surface (wtd < 0) while shunting the removed water on (to FSM, or discarded).
// Because no cell crosses wtd = 0, the model never engages the storativity jump / T-clamp free
// boundary, so BDF2-on-V stays 2nd order (the "no-crossing" regime) WITHOUT needing extended soil,
// and no above-surface water exists for open-water evaporation to act on. The ramp also crudely
// emulates near-surface evapotranspiration drawdown. Removal rate Q(wtd) = Qmax * g_w(wtd), with
// g_w a C2 quintic smoothstep rising 0 -> 1 across wtd in [-w, 0] (0 below the band, saturating at
// Qmax by the surface). Qmax must exceed the peak recharge rate to guarantee no breach.
// See benchmark/SURFACE_SINK_DESIGN.md sec 11.
static constexpr double SECONDS_IN_A_YEAR  = 31536000.0;
static bool             g_surface_sink       = false;
static double           g_surface_sink_qmax  = 0.0;  // Qmax: peak removal rate [m/s]
static double           g_surface_sink_width = 1.0;  // w: band width below the surface [m]

// Taper 2 -- demand-identity evaporation: the atmospheric loss transitions SMOOTHLY from the
// land-surface ET grid (deep) to the open-water rate owe (at/above the surface), as a logistic in
// wtd, treated IMPLICITLY in the solver (like the sink). Replaces the hard wtd>0 ? owe : ET recharge
// switch that sat on the wtd=0 knife-edge. E_eff(wtd) = ET + (owe-ET)*sigma((wtd - wtd_c)/s), a
// removal rate. Accessibility/extinction-depth (the max(0,P-ET) clamp) is deferred -> awickert/WTM#4.
// See SURFACE_SINK_DESIGN.md sec 14. ET/owe are per-cell (m/yr); the helpers return m/s.
static bool             g_evap_taper         = false;
static double           g_evap_taper_wtdc    = 0.05;  // wtd_c: half-rate depth [m] (small +, pond->exposed)
static double           g_evap_taper_s       = 0.1;   // s: transition width [m]

// Compact-support C2 quintic smoothstep ramp: 0 for wtd <= -w, smoothly rising to 1 at wtd = 0
// (p(u) = u^3(6u^2 - 15u + 10), p'(0)=p'(1)=0). Argument is wtd = h - topo (centre cell).
static double surfaceSinkRamp(const double wtd) {
  const double w = g_surface_sink_width;
  if (wtd <= -w) return 0.0;
  if (wtd >= 0.0) return 1.0;
  const double u = (wtd + w) / w;  // in (0,1)
  return u * u * u * (u * (6.0 * u - 15.0) + 10.0);
}
// d(ramp)/d(wtd) = p'(u)/w, with p'(u) = 30 u^2 (1-u)^2.
static double surfaceSinkRampTangent(const double wtd) {
  const double w = g_surface_sink_width;
  if (wtd <= -w || wtd >= 0.0) return 0.0;
  const double u = (wtd + w) / w;
  return 30.0 * u * u * (1.0 - u) * (1.0 - u) / w;
}
static double surfaceSink(const double wtd) { return g_surface_sink_qmax * surfaceSinkRamp(wtd); }
static double surfaceSinkTangent(const double wtd) {
  return g_surface_sink_qmax * surfaceSinkRampTangent(wtd);
}

// Taper 2 helpers. sigma is the logistic 1/(1+e^{-u}); u = (wtd - wtd_c)/s. E_eff(wtd) transitions
// ET -> owe as the table rises. Returns m/s (ET/owe supplied in m/yr). The tangent dE/dwtd is
// (owe-ET)*sigma*(1-sigma)/s; CLAMPED to >= 0 so it only ever strengthens the SPD storage diagonal
// (when ET > owe the raw tangent is negative). The clamp is applied identically in the operator and
// the RHS, so the Picard fixed point -- storage + dt*E_eff(w^{n+1}) = recharge -- is unchanged; only
// the linearization softens (a fixed-point step on that term). See SURFACE_SINK_DESIGN.md sec 14.
static double evapTaperSigma(const double wtd) {
  return 1.0 / (1.0 + std::exp(-(wtd - g_evap_taper_wtdc) / g_evap_taper_s));
}
static double evapTaper(const double wtd, const double et_yr, const double owe_yr) {
  return (et_yr + (owe_yr - et_yr) * evapTaperSigma(wtd)) / SECONDS_IN_A_YEAR;
}
static double evapTaperTangent(const double wtd, const double et_yr, const double owe_yr) {
  const double sig = evapTaperSigma(wtd);
  const double raw = (owe_yr - et_yr) * sig * (1.0 - sig) / g_evap_taper_s / SECONDS_IN_A_YEAR;
  return raw > 0.0 ? raw : 0.0;  // SPD-preserving clamp; see note above
}

// Smooth (C-inf) depth-integrated transmissivity: a differentiable blend of the
// piecewise production form above. Kept for a future Newton path; its analytic
// derivative is dTransmissivityInverseDwtd, and FormJacobianLocal uses this
// version. NOT used by the Anderson production residual.
static double depthIntegratedTransmissivitySmooth(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0;
  constexpr double shallow = 1.5;
  const double eps0        = g_ksat_surface_smoothing_width;     // smooth clamping at WTD=0 boundary
  const double eps1        = g_ksat_soilbottom_smoothing_width;  // smooth blend at WTD=-shallow boundary

  const double wtd_eff = (wtd_T - std::sqrt(wtd_T * wtd_T + eps0 * eps0)) * 0.5;
  const double u       = wtd_T + shallow;
  // eps1 == 0 => the sigmoid degrades to a step (sharp -1.5 m switch); eps0 == 0 is naturally sharp
  // (sqrt(wtd^2) = |wtd| in wtd_eff), so either boundary can be sharp independently.
  const double sigma_1 = (eps1 > 0.0) ? 1.0 / (1.0 + std::exp(u / eps1)) : (u < 0.0 ? 1.0 : 0.0);

  const double T_linear = ksat * (wtd_eff + shallow + fdepth);
  const double T_exp    = fdepth * ksat * std::exp(u / fdepth);

  return std::max(0.0, (1.0 - sigma_1) * T_linear + sigma_1 * T_exp);
}

// Analytic derivative of (1/T) with respect to wtd_T for the Newton-Krylov Jacobian
// (FormJacobianLocal): the derivative of the SMOOTH T. When a ksat smoothing width is set the
// residual (FormFunctionLocal) uses the smooth T with that width, so track it here to stay the
// exact derivative; when a width is 0 the residual uses the piecewise T, and we fall back to a
// fixed 0.01 m regularization so the Jacobian stays differentiable (a standard inexact-Newton
// approximation) and never divides by zero.
static double dTransmissivityInverseDwtd(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  const double eps0 = (g_ksat_surface_smoothing_width  > 0.0) ? g_ksat_surface_smoothing_width  : 0.01;
  const double eps1 = (g_ksat_soilbottom_smoothing_width > 0.0) ? g_ksat_soilbottom_smoothing_width : 0.01;

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

// Analytic derivative of S_eff with respect to my_new_wtd: the EXACT derivative of
// updateEffectiveStorativity, so it must use the same storativity smoothing width (not a
// hardcoded constant) to stay the true Jacobian for any -wtm_storativity_surface_smoothing_width.
// Uses the same V(w) = [w(1+p) + sqrt(w²+eps²)(1-p)] / 2 construction as storedVolume/specificYield.
static double dEffectiveStorativityDnew(
    const double my_original_wtd, const double my_new_wtd, const double my_porosity) {
  const double eps = g_storativity_surface_smoothing_width;
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

// Accumulate the water leaving through land->ocean faces this solve into arp.total_ocean_outflow_gw
// (a per-rank owned-cell partial; PrintValues reduces the partials to a global total). Ocean cells
// are Dirichlet h=0, so the crossing water is absorbed at the boundary and never appears as
// ocean-cell content -- the correct measure is the Darcy interface flux. It uses the SAME
// harmonic-mean conductance e = 2/(1/T_c + 1/T_nbr) the Picard operator assembles (mirroring its T
// construction), evaluated at the converged head, so the discrete budget closes exactly:
// recharge = d(storage) + ocean_outflow + surface_removed. Per land->ocean face the outflow volume
// is e * dt/(cell size)^2 * (h_land - 0) * cell_area, matching the operator's flux term (depth) times
// the cell area (volume). Needs ghost heads (x) and the ghost mask (mask_local); mirrors
// FormPicardOperator's ghost setup.
static void accumulate_ocean_outflow(AppCtx& user_context, ArrayPack& arp) {
  DM  da = user_context.da;
  Vec xloc;
  DMGetLocalVector(da, &xloc);
  DMGlobalToLocalBegin(da, user_context.x, INSERT_VALUES, xloc);
  DMGlobalToLocalEnd(da, user_context.x, INSERT_VALUES, xloc);

  PetscScalar **xx, **my_topo, **my_fdepth, **my_ksat, **my_mask, **my_T, **gew, **gn, **gs;
  DMDAVecGetArray(da, xloc, &xx);
  DMDAVecGetArray(da, user_context.topo_local, &my_topo);
  DMDAVecGetArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecGetArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecGetArray(da, user_context.mask_local, &my_mask);
  DMDAVecGetArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecGetArray(da, user_context.geom_n_vec, &gn);
  DMDAVecGetArray(da, user_context.geom_s_vec, &gs);
  DMDAVecGetArray(da, user_context.T_local, &my_T);

  DMDALocalInfo info;
  DMDAGetLocalInfo(da, &info);
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  for (auto j = info.gys; j < info.gys + info.gym; j++)
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++) {
      const double wtd_T = xx[j][i] - my_topo[j][i];
      my_T[j][i] = 1.0 / (smooth_T ? depthIntegratedTransmissivitySmooth(wtd_T, my_fdepth[j][i], my_ksat[j][i])
                                   : depthIntegratedTransmissivity(wtd_T, my_fdepth[j][i], my_ksat[j][i]));
    }

  const double dt = user_context.deltat;
  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      if (my_mask[j][i] == 0) continue;  // only LAND cells drain to ocean
      const double h_c = xx[j][i];
      // Volume flux through each land->ocean face = dt * G * h_c, with the SAME face conductance
      // G = e * (L_wall/d_centre) the operator assembles (E-W uses geom_ew, N/S the face geom_n/s).
      if (my_mask[j][i + 1] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j][i + 1]) * gew[j][i] * h_c;
      if (my_mask[j][i - 1] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j][i - 1]) * gew[j][i] * h_c;
      if (my_mask[j + 1][i] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j + 1][i]) * gn[j][i] * h_c;
      if (my_mask[j - 1][i] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j - 1][i]) * gs[j][i] * h_c;
    }
  }

  DMDAVecRestoreArray(da, xloc, &xx);
  DMDAVecRestoreArray(da, user_context.topo_local, &my_topo);
  DMDAVecRestoreArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecRestoreArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecRestoreArray(da, user_context.mask_local, &my_mask);
  DMDAVecRestoreArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecRestoreArray(da, user_context.geom_n_vec, &gn);
  DMDAVecRestoreArray(da, user_context.geom_s_vec, &gs);
  DMDAVecRestoreArray(da, user_context.T_local, &my_T);
  DMRestoreLocalVector(da, &xloc);
}

// Accumulate the solver's EXACT per-step discrete storage and specific-yield recharge over owned
// LAND cells, so the water budget closes to the SNES tolerance rather than the ~1% of the physical
// snapshot (whose gap is the BDF2-startup term, not a leak). The per-cell discrete balance the solve
// satisfies is  storage(w^{n+1},w^n,w^{n-1}) + dt*lateral_flux + dt*Q_sink = recharge_term, so summed
// over land cells (interior lateral fluxes cancel; the land->ocean flux is total_ocean_outflow):
//   total_storage_change = total_solver_recharge - total_ocean_outflow - total_surface_removed
// to SNES tolerance. The storage/recharge forms mirror FormPicardRHS exactly across its paths
// (BDF2-on-V uses V and specific yield; the secant paths use the effective storativity and heads).
// Called after the solve, BEFORE the BDF2 history overwrites w^{n-1}. Picard path only (the exact
// residual is not defined for the matrix-free Anderson default). See benchmark/WATER_BUDGET.md.
static void accumulate_budget_terms(AppCtx& user_context, ArrayPack& arp, DMDA_Array_Pack& dmdapack) {
  const bool bdf2      = user_context.use_bdf2 && user_context.bdf2_have_history;
  const bool bdf2_on_V = bdf2 && user_context.use_bdf2_on_V;
  double a_c = 1.0, b_c = 1.0, c_c = 0.0;  // backward-Euler weights (recharge form S_c*(h^{n+1}-h^n))
  if (bdf2) {
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    a_c                = (1.0 + 2.0 * omega) / (1.0 + omega);
    b_c                = 1.0 + omega;
    c_c                = omega * omega / (1.0 + omega);
  }

  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar **my_topo, **my_prev = nullptr;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  if (user_context.use_bdf2) DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &my_prev);
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      if (dmdapack.mask[j][i] == 0) continue;  // ocean: Dirichlet, no storage/recharge (flux counted separately)
      const double poro = dmdapack.porosity_vec[j][i];
      const double w1   = dmdapack.x[j][i] - my_topo[j][i];  // w^{n+1}
      const double w0   = dmdapack.starting_wtd[j][i];       // w^n (not yet overwritten by the copy-back)
      const double wm1  = my_prev ? my_prev[j][i] : 0.0;     // w^{n-1} (unused when c_c==0)
      const double rech = dmdapack.rech_vec[j][i];
      double storage, recharge;
      if (bdf2_on_V) {
        storage  = a_c * storedVolume(w1, poro) - b_c * storedVolume(w0, poro) + c_c * storedVolume(wm1, poro);
        recharge = specificYield(w1, poro) * rech;
      } else {
        const double S_c = updateEffectiveStorativity(w0, w1, poro);  // secant storativity (matches the RHS)
        const double h1 = dmdapack.x[j][i], h0 = w0 + my_topo[j][i], hm1 = wm1 + my_topo[j][i];
        storage  = S_c * (a_c * h1 - b_c * h0 + c_c * hm1);
        recharge = S_c * rech;
      }
      arp.total_storage_change  += storage * arp.cell_area[j];
      arp.total_solver_recharge += recharge * arp.cell_area[j];
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);
  if (my_prev) DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &my_prev);
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

  // Smoothing widths are physics modeling options and apply on ALL solver paths (Anderson,
  // Newton, Picard), so read them here -- before the solver branch -- rather than gating them
  // behind use_picard. Storativity land-surface transition (sub-grid roughness); default 0.01 m,
  // always on. The two ksat/transmissivity widths default to 0 (=> exact piecewise Fan T); any
  // positive width rounds that boundary in every path that evaluates T (residual and operator).
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_storativity_surface_smoothing_width", &g_storativity_surface_smoothing_width, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_ksat_soilbottom_smoothing_width", &g_ksat_soilbottom_smoothing_width, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_ksat_surface_smoothing_width", &g_ksat_surface_smoothing_width, nullptr);
  PetscBool extsoil = PETSC_FALSE;  // [WIP] -wtm_extended_soil: aquifer continues above surface (smooth GW step)
  PetscOptionsHasName(nullptr, nullptr, "-wtm_extended_soil", &extsoil);
  g_extended_soil = (extsoil == PETSC_TRUE);

  // Sub-surface sink [WIP]: Qmax supplied in m/yr (intuitive), stored as m/s. Requires
  // -wtm_bdf2_on_V (implemented only in the Picard RHS/operator). See SURFACE_SINK_DESIGN.md sec 11.
  PetscBool sink = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_surface_sink", &sink);
  g_surface_sink         = (sink == PETSC_TRUE);
  double sink_qmax_yr    = 0.0;
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_surface_sink_qmax", &sink_qmax_yr, nullptr);
  g_surface_sink_qmax = sink_qmax_yr / SECONDS_IN_A_YEAR;
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_surface_sink_width", &g_surface_sink_width, nullptr);

  // Taper 2 [WIP]: implicit demand-identity evaporation (ET -> owe). Read here AND early in
  // WTM.cpp::initialise() (before the initial recharge) via the same call, so the explicit-recharge
  // sites -- including irf.cpp's initial pass -- all see a consistent flag. See SURFACE_SINK_DESIGN.md 14.
  read_evap_taper_options(params);

  // Whether the sink was actually applied THIS solve (it lives only in the BDF2-on-V branch, which
  // needs an established history -- the BE bootstrap step has no sink). Captured before the solve,
  // since the copy-back below sets bdf2_have_history for the NEXT step. Used to account the removed
  // water in the same step it was removed.
  // Where the removals actually act, so we account exactly what the solve removed:
  //  * matrix-free path (Anderson/Newton, !use_picard): FormFunctionLocal applies them EVERY solve.
  //  * Picard path: only in the BDF2-on-V branch, once a history exists (the BE bootstrap has none).
  const bool matrix_free   = !user_context.use_picard;
  const bool picard_bdf2_V = user_context.use_bdf2 && user_context.bdf2_have_history && user_context.use_bdf2_on_V;
  const bool sink_active_this_step = g_surface_sink && (matrix_free || picard_bdf2_V);
  const bool evap_active_this_step = g_evap_taper && (matrix_free || picard_bdf2_V);

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

    // The Newton-Krylov path (analytic FormJacobianLocal) is DISABLED. It is unused -- the default
    // solver is Anderson, and the order-2 production path is -wtm_picard (semi-implicit) which drives
    // newtonls via SNESSetPicard and never reaches here. Its analytic Jacobian is also unmaintained:
    // it omits the sub-surface-sink and evaporation-taper tangents, so it would diverge. Refuse it
    // explicitly rather than run a stale, wrong-Jacobian solve. FormJacobianLocal is intentionally
    // RETAINED in-source below so a Newton solver can be rebuilt later (add the missing tangents,
    // then re-register here). Anderson (snes_type == SNESANDERSON) is matrix-free and skips this.
    SNESType snes_type;
    SNESGetType(user_context.snes, &snes_type);
    if (std::string(snes_type) != std::string(SNESANDERSON)) {
      throw std::runtime_error(
          std::string("The Newton-Krylov solver (-snes_type ") + snes_type +
          ") is disabled: its analytic Jacobian is unmaintained (missing the sink/evap-taper tangents) "
          "and would diverge. Use the default Anderson solver, or -wtm_picard for the semi-implicit "
          "(BDF2-on-V) path.");
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

  // Adaptive dt (forward, no-reject): estimate the local error as the max deviation of the
  // new head h^{n+1} from a LINEAR extrapolation of the history (h_pred = h^n + w(h^n -
  // h^{n-1}), w = dt_n/dt_{n-1}); this deviation ~ O(dt^2). Set the NEXT step to hold it near
  // dt_tol via dt_new = dt * clamp(safety*sqrt(tol/est), shrink, grow). Runs once history
  // exists, and BEFORE starting_wtd_prev is overwritten below. See BDF2_ADAPTIVE_DESIGN.md.
  if (user_context.use_dt_adaptive && user_context.bdf2_have_history) {
    PetscScalar **swp, **topo_e;
    DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecGetArray(user_context.da, user_context.topo_vec, &topo_e);
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    double local_max   = 0.0;
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (dmdapack.mask[j][i] != 0) {  // land only
          const double h_n    = dmdapack.starting_wtd[j][i] + topo_e[j][i];
          const double h_pred = h_n + omega * (dmdapack.starting_wtd[j][i] - swp[j][i]);
          const double dev    = std::abs(dmdapack.x[j][i] - h_pred);
          if (dev > local_max) local_max = dev;
        }
    DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &topo_e);

    double est = 0.0;
    MPI_Allreduce(&local_max, &est, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

    const double safety = 0.9, grow = 1.5, shrink = 0.5;
    double factor = (est > 0.0) ? safety * std::sqrt(user_context.dt_tol / est) : grow;  // est~O(dt^2)
    factor        = std::min(grow, std::max(shrink, factor));
    user_context.deltat *= factor;
  }

  // Exact budget-closing accounting (Picard path): the solver's discrete storage + recharge terms,
  // read while starting_wtd still holds w^n and starting_wtd_prev still holds w^{n-1} (both below
  // overwrite these). Together with total_ocean_outflow and total_surface_removed the budget then
  // closes to the SNES tolerance. See benchmark/WATER_BUDGET.md.
  if (user_context.use_picard) accumulate_budget_terms(user_context, arp, dmdapack);

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
  PetscScalar **my_evap = nullptr, **my_owe = nullptr;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  if (evap_active_this_step) {
    DMDAVecGetArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecGetArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
  }
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      dmdapack.starting_wtd[j][i] = dmdapack.x[j][i] - my_topo[j][i];
      if (dmdapack.mask[j][i] == 0) {
        if (dmdapack.starting_wtd[j][i] > 0)
          arp.total_loss_to_ocean_gw += dmdapack.starting_wtd[j][i] * arp.cell_area[j];
        else
          arp.total_loss_to_ocean_gw += dmdapack.starting_wtd[j][i] * arp.cell_area[j] * dmdapack.porosity_vec[j][i];
        dmdapack.starting_wtd[j][i] = 0.;
        continue;
      }
      // Land cells: the sink and the evaporation taper can both be active; account each in the same
      // sub-step it was removed, evaluated at the just-computed new head. Serial loop -> += race-free.
      if (sink_active_this_step) {
        // Sink removed dt*Q(w^{n+1}) (Q is m/s -> dt*Q is a depth). To FSM (stays in domain).
        const double removed_depth = user_context.deltat * surfaceSink(dmdapack.starting_wtd[j][i]);
        arp.total_surface_removed += removed_depth * arp.cell_area[j];  // budget-closing (WATER_BUDGET.md)
        dmdapack.sink_removed_dist[j][i] += removed_depth;              // per-cycle FSM input (taper 1)
      }
      if (evap_active_this_step) {
        // Taper 2 removed dt*E_eff(w^{n+1}) to the ATMOSPHERE (leaves the domain) -> its own budget
        // channel, kept separate from the sink's exfiltration-to-FSM (different destination).
        const double evap_depth =
            user_context.deltat * evapTaper(dmdapack.starting_wtd[j][i], my_evap[j][i], my_owe[j][i]);
        arp.total_evap_removed += evap_depth * arp.cell_area[j];
      }
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);
  if (evap_active_this_step) {
    DMDAVecRestoreArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
  }

  // Account the water that left through land->ocean faces this solve (Darcy interface flux at the
  // converged head), the term that closes the water budget against the Dirichlet ocean boundary.
  accumulate_ocean_outflow(user_context, arp);

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

// Whether the implicit sub-surface sink is configured this run (taper 1). Lets the cycle loop
// decide whether to gather the sink accumulator into arp.runoff for FSM without reaching into the
// file-static flag. Set in update() from -wtm_surface_sink, so valid by the post-solve gather.
bool surface_sink_on() { return g_surface_sink; }

// Whether the demand-identity evaporation taper is on (taper 2). Lets the explicit-recharge sites
// (irf.cpp, WTM.cpp) drop their hard ET<->owe switch and feed just precip, because the smooth
// implicit E_eff now carries that ET->open-water transition. Set by read_evap_taper_options().
bool evap_taper_on() { return g_evap_taper; }

// Read the taper-2 options (-wtm_evap_taper, wtd_c, s) into the file-static flags and enforce the
// evap_mode-1 requirement. Called BOTH early in WTM.cpp::initialise() (so irf.cpp's initial recharge
// sees the flag) AND in update() (so a standalone solve still parses it). Idempotent -- it just
// re-reads the same PETSc options -- so the double call is harmless.
void read_evap_taper_options(const Parameters& params) {
  PetscBool evap_taper = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_evap_taper", &evap_taper);
  g_evap_taper = (evap_taper == PETSC_TRUE);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_evap_taper_wtdc", &g_evap_taper_wtdc, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_evap_taper_s", &g_evap_taper_s, nullptr);
  // Taper 2 IS the ET->open-water evaporation transition; it only makes sense with the open-water
  // rate in play (evap_mode 1). evap_mode 0 (remove-all-surface-water) has no owe to transition to.
  if (g_evap_taper && !params.evap_mode)
    throw std::runtime_error("-wtm_evap_taper requires evap_mode 1 (it smooths the ET->open-water transition)");
}

// Gather this cycle's distributed sink removal (sink_removed_dist, depth m) to rank-0 arp.runoff,
// ADDING to it, so this cycle's FillSpillMerge routes the exfiltrated water the implicit sink pulled
// out of the solve (taper 1). Because the sink holds wtd<=0, FSM's own wtd>0->runoff handoff never
// fires -- this is its smooth, order-preserving replacement. It composes with the runoff_ratio
// channel (gather_runoff_to_zero OVERWRITES arp.runoff a cycle earlier, for the *next* FSM; this ADDS
// after that FSM has consumed the prior value). Reuses wtd_global as scratch, after gather_wtd_to_all
// has finished with it (the two run sequentially). See SURFACE_SINK_DESIGN.md sec 14 (taper 1).
void gather_sink_removed_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.sink_removed_dist[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++)
        arp.runoff(i, j) += full[j * params.ncells_x + i];
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
// GRID INDEX CONVENTION (see benchmark/GRID_CONVENTION.md) -- fixed once for the whole file:
//   i = column = EAST-WEST (longitude); spacing cellsize_e_w_metres[j], SHRINKS poleward.
//   j = row    = NORTH-SOUTH (latitude); spacing cellsize_n_s_metres, CONSTANT.
// Arrays are field[j][i]. NOTE the WTM paper (Callaghan et al. 2025, App. B) uses the OPPOSITE
// letters (paper x = S-N, y = W-E), so paper-Delta_x = cellsize_n_s and paper-Delta_y =
// cellsize_e_w. Divide the E-W (i +/- 1) flux by cellsize_e_w^2 and the N-S (j +/- 1) flux by
// cellsize_n_s^2 (with the face-centred E-W wall length) -- NOT the reverse.
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo* info, PetscScalar** x, PetscScalar** f, AppCtx* user_context) {
  DM da = user_context->da;
  PetscScalar **my_mask, **my_fdepth, **my_ksat, **my_topo, **my_rech, **my_T, **my_starting_wtd, **my_porosity, **gew,
      **gn, **gs;

  /*
    Compute function over the locally owned part of the grid.
    topo/fdepth/ksat/T use local ghost vectors so neighbor accesses [j][i±1] are valid under MPI.
  */
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // conservative-FV flux geometry
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscScalar **my_evap = nullptr, **my_owe = nullptr;  // taper 2: per-cell ET and open-water evap (m/yr)
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
  }

  // Use the smooth (C-inf) T when a ksat smoothing width is set (universal across solver paths);
  // otherwise the exact piecewise (C0) Fan form (production). Widths are read once in update().
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  // Compute 1/T over the full ghost range so neighbor lookups in the owned-range loop below are valid.
#pragma omp parallel for default(none) shared(info, my_T, x, my_topo, my_fdepth, my_ksat, smooth_T) collapse(2)
  for (auto j = info->gys; j < info->gys + info->gym; j++) {
    for (auto i = info->gxs; i < info->gxs + info->gxm; i++) {
      const double wtd_T = x[j][i] - my_topo[j][i];
      my_T[j][i] = 1. / (smooth_T ? depthIntegratedTransmissivitySmooth(wtd_T, my_fdepth[j][i], my_ksat[j][i])
                                  : depthIntegratedTransmissivity(wtd_T, my_fdepth[j][i], my_ksat[j][i]));
    }
  }

  const bool sink_on  = g_surface_sink;  // hoisted for the omp default(none) clause below
  const bool taper_on = g_evap_taper;
#pragma omp parallel for default(none)                                                                                \
    shared(info, gew, gn, gs, x, my_T, my_mask, my_rech, user_context, my_porosity, my_starting_wtd, my_topo, f,      \
           my_evap, my_owe, sink_on, taper_on) collapse(2)
  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      if (my_mask[j][i] == 0) {
        // Dirichlet condition: x = 0 for ocean cells (topo = wtd = 0 there).
        // Writing f = x rather than f = 0 gives a unit Jacobian diagonal,
        // which is required for the Newton-Krylov linear solve to be
        // non-singular. Anderson is unaffected: x starts at 0 and stays at 0.
        f[j][i] = x[j][i];
      } else {
        // Conservative finite-volume flux, HEAD form. The volume balance is
        //   A_j*S*(h - my_rech) + dt*(net outflow) = 0; we divide by A_j*S so the residual stays in
        // head units (O(metres)) -- the matrix-free Anderson solver diverges (DIVERGED_DTOL) on the
        // area-scaled volume-form residual, and, having no matrix, gains nothing from it: the root
        // (hence the solution and its conservation) is IDENTICAL. Face conductances G =
        // e*(L_wall/d_centre): E-W uses geom_ew, N/S the FACE-centred geom_n/geom_s, so shared-face
        // fluxes cancel (mass conserving) and the E-W/N-S cell sizes are no longer swapped. The
        // Picard OPERATOR keeps the volume form (it needs the exact symmetry). See GRID_CONVENTION.md.
        const double this_x = x[j][i];
        const double this_T = my_T[j][i];
        const double e_E    = 2. / (this_T + my_T[j][i + 1]);  // harmonic-mean interface transmissivities
        const double e_W    = 2. / (this_T + my_T[j][i - 1]);
        const double e_N    = 2. / (this_T + my_T[j + 1][i]);
        const double e_S    = 2. / (this_T + my_T[j - 1][i]);

        // Net outflow volume-rate = sum of face conductances * (h_c - h_nbr).
        const double net_outflow = e_E * gew[j][i] * (this_x - x[j][i + 1]) + e_W * gew[j][i] * (this_x - x[j][i - 1])
                                 + e_N * gn[j][i] * (this_x - x[j + 1][i]) + e_S * gs[j][i] * (this_x - x[j - 1][i]);

        const double A_j = user_context->cellsize_NS_squared / gew[j][i];  // cell area
        const double w_c = this_x - my_topo[j][i];                         // centre-cell wtd
        const double S   = updateEffectiveStorativity(my_starting_wtd[j][i], w_c, my_porosity[j][i]);

        // Sub-surface sink (taper 1) and demand-identity evaporation (taper 2), as head-form removals
        // dt*Q/S: evaluated at the current iterate, so implicit at the Anderson root. Matrix-free ->
        // no tangent needed (unlike the Picard operator). Off unless their flags are set.
        double removal = 0.0;  // m/s
        if (sink_on) removal += surfaceSink(w_c);
        if (taper_on) removal += evapTaper(w_c, my_evap[j][i], my_owe[j][i]);

        f[j][i] = (this_x - my_rech[j][i]) + user_context->deltat * net_outflow / (A_j * S)
                  + user_context->deltat * removal / S;
        // my_rech is converted to appropriate recharge for this timestep and starting water
        // table outside of the solve.
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
  }

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
  PetscScalar **bb, **xx, **my_starting_wtd, **my_topo, **my_rech, **my_porosity, **my_mask, **gew;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr;  // taper 2: per-cell ET and open-water evap (m/yr)
  const double  cns2 = user_context->cellsize_NS_squared;  // cell area A_j = cns2 / geom_ew (volume form)

  // Variable-step BDF2 (once an h^{n-1} exists): b = S_c*(b_c*h^n - c_c*h^{n-1} + rech) with
  // omega = dt_n/dt_{n-1}, b_c = 1+omega, c_c = omega^2/(1+omega) (b_c=2, c_c=1/2 when the
  // step is constant -> uniform BDF2). Backward Euler otherwise: b = S_c*(h^n + rech).
  // h^{n-1} = starting_wtd_prev + topo (centre only).
  const bool bdf2      = user_context->use_bdf2 && user_context->bdf2_have_history;
  const bool bdf2_on_V = bdf2 && user_context->use_bdf2_on_V;
  double a_c = 1.0, b_c = 0.0, c_c = 0.0;
  if (bdf2) {
    const double omega = user_context->deltat / user_context->bdf2_prev_dt;
    a_c                = (1.0 + 2.0 * omega) / (1.0 + omega);
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
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // for the cell area A_j
  if (bdf2) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
  }

  const auto [xs, ys, xm, ym] = get_corners(da);
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      if (my_mask[j][i] == 0) {
        bb[j][i] = 0.0;  // Dirichlet ocean cell: h = 0
      } else if (bdf2_on_V) {
        // BDF2-on-V: storage = a*V(w) - b*V(w^n) + c*V(w^{n-1}), Picard-linearized about x_k so
        // the diagonal a*Sy(w_k) (in the operator) cancels a*Sy(w_k)*x_k here, leaving a*V(w_k) at
        // the fixed point. Volume form: the whole storage+recharge+sink RHS scales by the cell area
        // A_j (matching the operator's a*Sy*A_j diagonal and dt*G face conductances).
        const double poro  = my_porosity[j][i];
        const double w_k   = xx[j][i] - my_topo[j][i];
        const double Sy    = specificYield(w_k, poro);
        const double A_j   = cns2 / gew[j][i];
        bb[j][i] = A_j * (a_c * Sy * xx[j][i] - a_c * storedVolume(w_k, poro)
                        + b_c * storedVolume(my_starting_wtd[j][i], poro)
                        - c_c * storedVolume(my_starting_wtd_prev[j][i], poro)
                        + Sy * my_rech[j][i]);
        if (g_surface_sink) {
          // Implicit sub-surface removal dt*Q(w^{n+1}), Picard-linearized about w_k like the storage
          // term; scaled by A_j to match the operator's dt*Q'(w_k)*A_j diagonal (volume form).
          const double dt = user_context->deltat;
          bb[j][i] += A_j * (dt * surfaceSinkTangent(w_k) * xx[j][i] - dt * surfaceSink(w_k));
        }
        if (g_evap_taper) {
          // Taper 2: implicit demand-identity evaporation dt*E_eff(w^{n+1}) (ET -> owe), Picard-
          // linearized about w_k with the SPD-clamped tangent (matches the operator's evap diagonal).
          const double dt = user_context->deltat;
          bb[j][i] += A_j * (dt * evapTaperTangent(w_k, my_evap[j][i], my_owe[j][i]) * xx[j][i]
                             - dt * evapTaper(w_k, my_evap[j][i], my_owe[j][i]));
        }
      } else {
        const double S_c =
            updateEffectiveStorativity(my_starting_wtd[j][i], xx[j][i] - my_topo[j][i], my_porosity[j][i]);
        const double h_n = my_starting_wtd[j][i] + my_topo[j][i];
        const double A_j = cns2 / gew[j][i];  // volume form: scale the storage/recharge by the cell area
        if (bdf2) {
          const double h_nm1 = my_starting_wtd_prev[j][i] + my_topo[j][i];
          bb[j][i] = A_j * S_c * (b_c * h_n - c_c * h_nm1 + my_rech[j][i]);
        } else {
          bb[j][i] = A_j * S_c * (h_n + my_rech[j][i]);
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
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
  }
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
      **my_T, **gew, **gn, **gs;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr;  // taper 2: per-cell ET and open-water evap (m/yr)
  PetscCall(DMDAVecGetArray(da, xloc, &xx));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));      // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));  // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // conservative-FV flux geometry
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
  }

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(da, &info));

  // 1/T over the full ghost range so the neighbor harmonic means on the owned range are valid
  // (mirrors FormFunctionLocal). Production uses the piecewise (C0) Fan form; a positive
  // -wtm_ksat_soilbottom_smoothing_width (-1.5 m) and/or -wtm_ksat_surface_smoothing_width (0 m)
  // swaps in the smooth (C-inf) form, rounding that boundary. Both 0 (default) => piecewise. The
  // widths are read once per cycle in update() (universal across solver paths).
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  for (auto j = info.gys; j < info.gys + info.gym; j++) {
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++) {
      const double wtd_T = xx[j][i] - my_topo[j][i];
      my_T[j][i] = 1.0 / (smooth_T ? depthIntegratedTransmissivitySmooth(wtd_T, my_fdepth[j][i], my_ksat[j][i])
                                   : depthIntegratedTransmissivity(wtd_T, my_fdepth[j][i], my_ksat[j][i]));
    }
  }

  const double dt   = user_context->deltat;
  const double cns2 = user_context->cellsize_NS_squared;

  // Variable-step BDF2 (once an h^{n-1} exists): a*h^{n+1} - b*h^n + c*h^{n-1} = dt*RHS,
  // with omega = dt_n/dt_{n-1}, a = (1+2w)/(1+w) [here], b,c on the RHS. The diffusion term
  // always carries the current dt; the storage diagonal carries a*S_c (a=3/2 when the step is
  // constant, i.e. w=1 -> uniform BDF2). Backward Euler is a=1. See BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2 = user_context->use_bdf2 && user_context->bdf2_have_history;
  double a_coeff  = 1.0;  // coefficient of h^{n+1} on the storativity diagonal (BE)
  if (bdf2) {
    const double omega = dt / user_context->bdf2_prev_dt;
    a_coeff            = (1.0 + 2.0 * omega) / (1.0 + omega);
  }
  // BDF2-on-V: use the TANGENT dV/dh on the diagonal (BDF2 applied to the volume), instead of the
  // backward-Euler secant storativity that caps the order at 1. Only once history exists (a BDF2
  // step); the BE bootstrap step keeps the secant. See BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2_on_V = bdf2 && user_context->use_bdf2_on_V;

  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      const MatStencil row = {.k = 0, .j = j, .i = i, .c = 0};

      if (my_mask[j][i] == 0) {
        // Ocean: placeholder diagonal; MatZeroRowsColumnsStencil fixes it to identity.
        const PetscScalar one = 1.0;
        PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, &one, INSERT_VALUES));
      } else {
        // Conservative FINITE-VOLUME (volume-form) assembly: each row is the cell's VOLUME balance,
        // so off-diagonals are the shared face conductances dt*G (G = e * L_wall/d_centre) -- exactly
        // symmetric across every face and mass-conservative -- and the storage/sink diagonal carries
        // the cell area A_j. See benchmark/GRID_CONVENTION.md. (Was head-form, which divided the E-W
        // flux by cellsize_n_s^2 and the N-S flux by cellsize_e_w^2 -- the two swapped, off by
        // cos^2(lat), and non-conservative across N-S faces.)
        const double A_j = cns2 / gew[j][i];  // cell area = cellsize_n_s^2 / (cellsize_n_s/cellsize_e_w)

        // Storativity diagonal coefficient, frozen at the current x. BDF2-on-V uses the tangent
        // dV/dh (specificYield); otherwise the backward-Euler secant (matches FormFunctionLocal).
        const double w_k = xx[j][i] - my_topo[j][i];
        const double S_c =
            bdf2_on_V ? specificYield(w_k, my_porosity[j][i])
                      : updateEffectiveStorativity(my_starting_wtd[j][i], w_k, my_porosity[j][i]);

        // Harmonic-mean interface transmissivities: 2 / (1/T_c + 1/T_nbr).
        const double e_E = 2.0 / (my_T[j][i] + my_T[j][i + 1]);
        const double e_W = 2.0 / (my_T[j][i] + my_T[j][i - 1]);
        const double e_N = 2.0 / (my_T[j][i] + my_T[j + 1][i]);
        const double e_S = 2.0 / (my_T[j][i] + my_T[j - 1][i]);

        // Face conductances G = e * (L_wall/d_centre): E-W uses geom_ew (per row); N/S use the
        // FACE-centred geom_n/geom_s, so G_N(j) = G_S(j+1) exactly (shared face) -> conservative.
        const double A_east   = -dt * e_E * gew[j][i];
        const double A_west   = -dt * e_W * gew[j][i];
        const double A_north  = -dt * e_N * gn[j][i];
        const double A_south  = -dt * e_S * gs[j][i];
        // Storage and sub-surface sink now scale with the cell area A_j (volume form). The sink
        // tangent dt*Q'(w_k)*A_j is >= 0, so the diagonal stays dominant -> SPD-preserving.
        const double sink_diag = (g_surface_sink && bdf2_on_V) ? dt * surfaceSinkTangent(w_k) * A_j : 0.0;
        // Taper 2 evaporation diagonal: dt*E_eff'(w_k)*A_j, SPD-clamped >= 0 (matches the RHS term).
        const double evap_diag =
            (g_evap_taper && bdf2_on_V) ? dt * evapTaperTangent(w_k, my_evap[j][i], my_owe[j][i]) * A_j : 0.0;
        const double A_center =
            a_coeff * S_c * A_j + sink_diag + evap_diag - (A_east + A_west + A_north + A_south);

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
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
  }
  PetscCall(DMRestoreLocalVector(da, &xloc));
  return 0;
}

}  // namespace FanDarcyGroundwater
