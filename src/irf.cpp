#include <limits>
#include "ArrayPack.hpp"
#include "fill_spill_merge.hpp"
#include "grid_geometry.hpp"
#include "parameters.hpp"
#include "update_effective_storativity.hpp"  // storedVolume, for the exact stored-water budget

#include <richdem/common/Array2D.hpp>

#define OMPI_SKIP_MPICXX 1  // we use the MPI C API only; skip the deprecated C++ bindings
#include <mpi.h>

namespace rd = richdem;
namespace dh = richdem::dephier;

// Boundary conditions use the mask-aware ghost-node scheme (task #96): Dirichlet h=0 at ocean edges,
// land-slope Neumann at land edges, computed at the true domain edge. The legacy alternative -- force
// every domain edge to sea-level ocean via setEdges(0) -- was retained behind -wtm_dev_padded_dirichlet
// until 2026-09-04 and is now retired; see benchmark/BOUNDARY_CONDITIONS.md for why, and
// tests/boundary_consistency for the standing check that the two agree on an ocean-ringed domain.


// Taper 2 accessor (defined in transient_groundwater.cpp): whether the smooth ET->open-water
// evaporation transition is on, so the initial recharge below feeds just precip. Forward-declared
// to avoid pulling the solver header (and its PETSc deps) into irf.cpp.
namespace FanDarcyGroundwater {
bool evap_taper_on();
}

constexpr double UNDEF             = -1.0e7;
constexpr double seconds_in_a_year = 31536000.;

/// We calculate the e-folding depth here, using temperature and slope.
double setup_fdepth(const Parameters& params, const double slope, const double temperature) {
  const auto fdepth = std::max(params.fdepth_a / (1 + params.fdepth_b * slope), params.fdepth_fmin);
  if (temperature > -5) {  // then fdepth = f from Ying's equation S7.
    return fdepth;
  } else if (temperature < -14) {  // then fdpth = f*fT, Ying's equations S7 and S8.
    return fdepth * std::max(0.05, 0.17 + 0.005 * temperature);
  } else {
    return fdepth * std::min(1.0, 1.5 + 0.1 * temperature);
  }
}

/// This function initialises those arrays that are needed only for transient
/// model runs. This includes both start and end states for slope, precipitation,
/// temperature, topography, ET, and relative humidity. We also have a land vs
/// ocean mask for the end time. It also includes the starting water table depth
/// array, a requirement for transient runs.
void InitialiseTransient(Parameters& params, ArrayPack& arp) {
  // width and height in number of cells in the array

  arp.topo_start = rd::Array2D<float>(params.get_path(params.time_start, "topography"));

  params.ncells_x = arp.topo_start.width();
  params.ncells_y = arp.topo_start.height();

  arp.slope_start           = rd::Array2D<float>(params.get_path(params.time_start, "slope"));
  arp.precip_start          = rd::Array2D<float>(params.get_path(params.time_start, "precipitation"));
  arp.evap_start            = rd::Array2D<float>(params.get_path(params.time_start, "evaporation"));
  arp.open_water_evap_start = rd::Array2D<float>(params.get_path(params.time_start, "open_water_evaporation"));
  arp.winter_temp_start     = rd::Array2D<float>(params.get_path(params.time_start, "winter_temperature"));
  arp.topo_end              = rd::Array2D<float>(params.get_path(params.time_end, "topography"));
  arp.slope_end             = rd::Array2D<float>(params.get_path(params.time_end, "slope"));
  arp.land_mask             = rd::Array2D<float>(params.get_path(params.time_end, "mask"));

  // land_mask: 1 where there is land, 0 in the ocean. Default: keep the real border (mask-aware ghost
  // boundary). Legacy verification path only: force the border to ocean (guarded to an all-ocean boundary).

  arp.precip_end          = rd::Array2D<float>(params.get_path(params.time_end, "precipitation"));
  arp.evap_end            = rd::Array2D<float>(params.get_path(params.time_end, "evaporation"));
  arp.open_water_evap_end = rd::Array2D<float>(params.get_path(params.time_end, "open_water_evaporation"));
  arp.winter_temp_end     = rd::Array2D<float>(params.get_path(params.time_end, "winter_temperature"));

  if (params.runoff_ratio_on && params.runoff_ratio_uniform < 0.0) {  // raster form
    arp.runoff_ratio_start = rd::Array2D<float>(params.get_path(params.time_start, "runoff_ratio"));
    arp.runoff_ratio_end   = rd::Array2D<float>(params.get_path(params.time_end, "runoff_ratio"));
  } else {  // uniform value (runoff_ratio_uniform >= 0) or off (0)
    const float rr = params.runoff_ratio_on ? static_cast<float>(params.runoff_ratio_uniform) : 0.0f;
    arp.runoff_ratio_start = rd::Array2D<float>(arp.topo_start, rr);
    arp.runoff_ratio_end   = rd::Array2D<float>(arp.topo_start, rr);
  }

  if (params.infiltration_on) {
    arp.vert_ksat = rd::Array2D<float>(params.get_path("vertical_ksat"));
  }

  // load in the wtd result from the previous time:
  arp.wtd = rd::Array2D<double>(params.get_path(params.time_start, "wtd"));

  // calculate the fdepth (e-folding depth, representing rate of decay of the
  // hydraulic conductivity with depth) arrays:
  arp.fdepth = rd::Array2D<double>(arp.topo_start, 0);

  for (size_t i = 0; i < arp.topo_start.size(); i++) {
    arp.fdepth(i) = setup_fdepth(params, arp.slope_start(i), arp.winter_temp_start(i));
  }

  // initialise the arrays to be as at the starting time:
  arp.topo            = arp.topo_start;
  arp.slope           = arp.slope_start;
  arp.precip          = arp.precip_start;
  arp.evap            = arp.evap_start;
  arp.open_water_evap = arp.open_water_evap_start;
  arp.winter_temp     = arp.winter_temp_start;
  if (params.runoff_ratio_on) {
    arp.runoff_ratio = arp.runoff_ratio_start;
  } else {
    arp.runoff_ratio = rd::Array2D<float>(arp.topo_start, 0.0);
  }
}

/// This function initialises those arrays that are needed only for equilibrium
/// model runs.
/// This includes a single array for each of slope, precipitation, temperature,
/// topography, ET, land vs ocean mask, and relative humidity.
/// It also includes setting the starting water table depth array to
/// zero everywhere.
void InitialiseEquilibrium(Parameters& params, ArrayPack& arp) {
  arp.topo = rd::Array2D<float>(params.get_path(params.time_start, "topography"));

  // width and height in number of cells in the array
  params.ncells_x = arp.topo.width();
  params.ncells_y = arp.topo.height();

  arp.slope     = rd::Array2D<float>(params.get_path(params.time_start, "slope"));
  arp.land_mask = rd::Array2D<float>(
      params.get_path(params.time_start, "mask"));  // A binary mask that is 1 where there is land and 0 in the ocean
  // Default: keep the real border (mask-aware ghost boundary). Legacy verification path only (guarded).

  arp.precip = rd::Array2D<float>(params.get_path(params.time_start, "precipitation"));  // Units: m/yr.
  arp.evap   = rd::Array2D<float>(params.get_path(params.time_start, "evaporation"));    // Units: m/yr.
  arp.open_water_evap =
      rd::Array2D<float>(params.get_path(params.time_start, "open_water_evaporation"));  // Units: m/yr.
  arp.winter_temp =
      rd::Array2D<float>(params.get_path(params.time_start, "winter_temperature"));  // Units: degrees Celsius

  if (params.runoff_ratio_on && params.runoff_ratio_uniform < 0.0) {  // raster form
    arp.runoff_ratio = rd::Array2D<float>(params.get_path(params.time_start, "runoff_ratio"));  // Units: m/yr.
  } else {  // uniform value (runoff_ratio_uniform >= 0) or off (0)
    const float rr = params.runoff_ratio_on ? static_cast<float>(params.runoff_ratio_uniform) : 0.0f;
    arp.runoff_ratio = rd::Array2D<float>(arp.topo, rr);  // Units: m/yr.
  }

  if (params.infiltration_on == true) {
    arp.vert_ksat = rd::Array2D<float>(params.get_path("vertical_ksat"));  // Units of ksat are m/s.
  }

  if (!params.initial_wt_path.empty()) {
    arp.wtd = rd::Array2D<double>(params.initial_wt_path);  // run.initial_water_table: <path>
  } else if (params.supplied_wt == true) {
    arp.wtd = rd::Array2D<double>(params.get_path(params.time_start, "starting_wt"));
  } else {
    arp.wtd = rd::Array2D<double>(arp.topo, 0.);
  }
  // we start with a water table at the surface for equilibrium runs.

  arp.fdepth = rd::Array2D<double>(arp.topo, 0);
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    arp.fdepth(i) = setup_fdepth(params, arp.slope(i), arp.winter_temp(i));
  }
}

void InitialiseTest(Parameters& params, ArrayPack& arp) {
  arp.topo  = rd::Array2D<float>(params.get_path("topography"));
  arp.slope = rd::Array2D<float>(params.get_path("slope"));  // Slope as a value from 0 to 1.

  // width and height in number of cells in the array
  params.ncells_x = arp.topo.width();
  params.ncells_y = arp.topo.height();

  if (params.infiltration_on) {
    arp.vert_ksat = rd::Array2D<float>(arp.topo, 0.00001f);  // Units of ksat are m/s.
  }

  // A binary mask that is 1 where there is land and 0 in the ocean
  arp.land_mask = rd::Array2D<float>(arp.topo, 1.f);

  arp.precip          = rd::Array2D<float>(arp.topo, 0.3);  // Units: m/yr.
  arp.runoff_ratio    = rd::Array2D<float>(arp.topo, 0.);   // Units: m/yr.
  arp.evap            = rd::Array2D<float>(arp.topo, 0.);   // Units: m/yr.
  arp.open_water_evap = rd::Array2D<float>(arp.topo, 0.4);  // Units: m/yr.

  arp.winter_temp = rd::Array2D<float>(arp.topo, 0);  // Units: deg C
  arp.wtd         = rd::Array2D<double>(arp.topo, 0.0);

  arp.fdepth = rd::Array2D<double>(arp.topo, 60);

  // border of 'ocean' with land everywhere else
  for (int y = 0; y < params.ncells_y; y++) {
    for (int x = 0; x < params.ncells_x; x++) {
      if (arp.land_mask.isEdgeCell(x, y)) {
        arp.land_mask(x, y) = 0.f;
      } else {
        arp.land_mask(x, y) = 1.f;
        if (std::isnan(arp.topo(x, y))) {
          arp.topo(x, y) = 0;
        }
      }
    }
  }
  // Default: keep the real border (mask-aware ghost boundary). Legacy verification path only (guarded).

  arp.ksat                  = rd::Array2D<float>(arp.topo, 0.0001f);  // Units of ksat are m/s.
  arp.porosity              = rd::Array2D<float>(arp.topo, 0.25);     // Units: unitless
  arp.effective_storativity = rd::Array2D<double>(arp.topo, 0.25);

  // Set arrays that start off with zero or other values,
  // that are not imported files. Just to initialise these -
  // we'll add the appropriate values later.

  // These two are just informational, to see how much change
  // happens in FSM vs in groundwater
  arp.wtd_old = arp.wtd;
  arp.wtd_mid = arp.wtd;

  arp.runoff = rd::Array2D<double>(arp.ksat, 0);
  arp.runoff_nominal = rd::Array2D<double>(arp.ksat, 0);

  // This is used to see how much change occurred in infiltration
  // portion of the code. Just informational.
  arp.infiltration_array = rd::Array2D<double>(arp.ksat, 0);

  arp.rech           = rd::Array2D<double>(arp.ksat, 0);
  arp.transmissivity = rd::Array2D<double>(arp.ksat, 0);

  // These are populated during the calculation of the depression hierarchy:
  // No cells are part of a depression
  arp.label = rd::Array2D<dh::dh_label_t>(params.ncells_x, params.ncells_y, dh::NO_DEP);
  // No cells are part of a depression
  arp.final_label = rd::Array2D<dh::dh_label_t>(params.ncells_x, params.ncells_y, dh::NO_DEP);
  // No cells flow anywhere
  arp.flowdirs = rd::Array2D<rd::flowdir_t>(params.ncells_x, params.ncells_y, rd::NO_FLOW);

  // Change undefined cells to 0
  for (size_t i = 0; i < arp.topo.size(); i++) {
    if (arp.topo(i) <= UNDEF) {
      arp.topo(i) = 0;
    }
  }

// get the starting runoff using precip and evap inputs:
#pragma omp parallel for default(none) shared(arp, params)
  for (size_t i = 0; i < arp.topo.size(); i++) {
    arp.rech(i) = (std::max(0., static_cast<double>(arp.precip(i)) - arp.evap(i))) / seconds_in_a_year * params.deltat;
    if (arp.porosity(i) <= 0) {
      arp.porosity(i) = 0.0000001f;  // not sure why it is sometimes processing cells with 0 porosity?
    }
  }

// Wtd is 0 in the ocean and under the ice:
#pragma omp parallel for default(none) shared(arp)
  for (size_t i = 0; i < arp.topo.size(); i++) {
    if (arp.land_mask(i) == 0) {  // || arp.ice_mask(i) == 1){
      arp.wtd(i)  = 0.;
      arp.topo(i) = 0.;
    }
  }

  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    arp.fdepth(i) = setup_fdepth(params, arp.slope(i), arp.winter_temp(i));
  }

// Label the ocean cells. This is a precondition for
// using `GetDepressionHierarchy()`.
#pragma omp parallel for default(none) shared(arp)
  for (unsigned int i = 0; i < arp.label.size(); i++) {
    if (arp.land_mask(i) == 0) {
      arp.label(i)       = dh::OCEAN;
      arp.final_label(i) = dh::OCEAN;
    }
  }
}


/// This function initialises those arrays that are used for both equilibrium
/// and transient model runs. This includes arrays that start off with zero
/// values, as well as the label, final_label, and flowdirs arrays.
void InitialiseBoth(const Parameters& params, ArrayPack& arp) {
  arp.ksat     = rd::Array2D<float>(params.get_path("horizontal_ksat"));
  arp.porosity = rd::Array2D<float>(params.get_path("porosity"));

  arp.effective_storativity = rd::Array2D<double>(arp.topo, 0.);
  // Set arrays that start off with zero or other values,
  // that are not imported files. Just to initialise these -
  // we'll add the appropriate values later.

  // These two are just informational, to see how much change
  // happens in FSM vs in groundwater
  arp.wtd_old = arp.wtd;
  arp.wtd_mid = arp.wtd;

  arp.runoff = rd::Array2D<double>(arp.ksat, 0);
  arp.runoff_nominal = rd::Array2D<double>(arp.ksat, 0);

  // These are used to see how much change occurred in infiltration
  // and updating lakes portions of the code. Just informational.
  arp.infiltration_array = rd::Array2D<double>(arp.ksat, 0);

  arp.rech           = rd::Array2D<double>(arp.ksat, 0);
  arp.transmissivity = rd::Array2D<double>(arp.ksat, 0);

  // These are populated during the calculation of the depression hierarchy:
  // No cells are part of a depression
  arp.label = rd::Array2D<dh::dh_label_t>(params.ncells_x, params.ncells_y, dh::NO_DEP);
  // No cells are part of a depression
  arp.final_label = rd::Array2D<dh::dh_label_t>(params.ncells_x, params.ncells_y, dh::NO_DEP);
  // No cells flow anywhere
  arp.flowdirs = rd::Array2D<rd::flowdir_t>(params.ncells_x, params.ncells_y, rd::NO_FLOW);

  // Change undefined cells to 0
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    if (arp.topo(i) <= UNDEF) {
      arp.topo(i) = 0;
    }
  }

#pragma omp parallel for default(none) shared(arp, params)
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    if (arp.porosity(i) <= 0) {
      arp.porosity(i) = 0.0000001f;  // not sure why it is sometimes processing cells with 0 porosity?
    }
  }

  // get the starting runoff using precip and evap inputs. Taper-first: the taper (2/3) governs
  // evaporation via the implicit E_eff, so feed just precip -- the smooth removal auto-zeroes standing
  // water, so no independent wtd=0 is needed under the taper. With the taper OFF, all surface water is
  // removed (wtd=0; GW-alone testing, Fan Reinfelder et al. 2013). Matches the per-cycle path.
  const bool evap_taper = FanDarcyGroundwater::evap_taper_on();
  std::cout << (evap_taper ? "p updating the recharge field (taper)" : "p removing all surface water")
            << std::endl;
#pragma omp parallel for default(none) shared(arp, params, evap_taper)
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    if (evap_taper) {
      arp.rech(i) = arp.precip(i) / seconds_in_a_year * params.deltat;
    } else if (arp.wtd(i) > 0) {  // surface water present
      arp.wtd(i) = 0;  // taper off: remove all surface water (GW-alone testing)
      arp.rech(i) = (arp.precip(i) - arp.open_water_evap(i)) / seconds_in_a_year * params.deltat;
    } else {  // water table below the surface; recharge is always positive
      arp.rech(i) =
          (std::max(0., static_cast<double>(arp.precip(i)) - arp.evap(i))) / seconds_in_a_year * params.deltat;
    }
    if (arp.rech(i) > 0) {
      // positive recharge may partly run off (runoff_ratio); subtract it from the recharge. Additive onto the
      // freshly-allocated (zeroed) carrier, matching the per-step arm in couple_surface_and_recharge (approach B).
      const double rr = arp.runoff_ratio(i) * arp.rech(i);
      arp.runoff(i) += rr;
      arp.rech(i) -= rr;
    }
  }

  // Wtd is 0 in the ocean and under the ice:
#pragma omp parallel for default(none) shared(arp)
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    if (arp.land_mask(i) == 0) {  //|| arp.ice_mask(i) ==1){
      arp.wtd(i)  = 0.;
      arp.topo(i) = 0.;
    }
  }

// Label the ocean cells. This is a precondition for
// using `GetDepressionHierarchy()`.
#pragma omp parallel for default(none) shared(arp)
  for (unsigned int i = 0; i < arp.label.size(); i++) {
    if (arp.land_mask(i) == 0) {
      arp.label(i)       = dh::OCEAN;
      arp.final_label(i) = dh::OCEAN;
    }
  }
}

/// In transient runs, we adjust the input arrays via a
// linear interpolation from the start state to the end state at each iteration.
/// We do so here, and also reset the label and flow direction arrays,
/// since the depression hierarchy needs to be
/// recalculated due to the changed topography.
void UpdateTransientArrays(const Parameters& params, ArrayPack& arp) {
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    const double f = static_cast<double>(params.cycles_done) / params.total_reports;

    arp.topo(i)            = (1 - f) * arp.topo_start(i) + f * arp.topo_end(i);
    arp.slope(i)           = (1 - f) * arp.slope_start(i) + f * arp.slope_end(i);
    arp.precip(i)          = (1 - f) * arp.precip_start(i) + f * arp.precip_end(i);
    arp.runoff_ratio(i)    = (1 - f) * arp.runoff_ratio_start(i) + f * arp.runoff_ratio_end(i);
    arp.evap(i)            = (1 - f) * arp.evap_start(i) + f * arp.evap_end(i);
    arp.open_water_evap(i) = (1 - f) * arp.open_water_evap_start(i) + f * arp.open_water_evap_end(i);
    arp.winter_temp(i)     = (1 - f) * arp.winter_temp_start(i) + f * arp.winter_temp_end(i);
    arp.fdepth(i)          = setup_fdepth(params, arp.slope(i), arp.winter_temp(i));

    arp.label(i)       = dh::NO_DEP;   // No cells are part of a depression
    arp.final_label(i) = dh::NO_DEP;   // No cells are part of a depression
    arp.flowdirs(i)    = rd::NO_FLOW;  // No cells flow anywhere
  }

#pragma omp parallel for default(none) shared(arp)
  for (unsigned int i = 0; i < arp.label.size(); i++) {
    if (arp.land_mask(i) == 0) {
      arp.label(i)       = dh::OCEAN;
      arp.final_label(i) = dh::OCEAN;
    }
  }
}

/// In this function, we use a few of the variables that were created for
/// informational purposes to help us understand how much the water table
/// is changing per iteration, and where in
/// the code that change is occurring. We print these values to a text file.
double ComputeStoredVolume(const Parameters& params, const ArrayPack& arp) {
  double v = 0.0;
  for (int y = 0; y < params.ncells_y; y++)
    for (int x = 0; x < params.ncells_x; x++)
      v += storedVolume(arp.wtd(x, y), arp.porosity(x, y)) * arp.cell_area[y];
  return v;
}

// THE BUDGET BASELINE, and the reason it is taken here rather than on the first report.
//
// PrintValues has ONE call site (WTM.cpp), at the END of a cycle. Capturing stored_volume_initial
// there meant the baseline was the state AFTER the first cycle had already run, while every flux
// accumulator -- recharge, evaporation, Darcy ocean outflow, FSM spill -- starts at cycle 0. The
// closure then differenced a storage change over cycles 1..N against fluxes over cycles 0..N, and
// the first cycle's storage change was silently absent from the books.
//
// On a cold start that term is not small; it is most of the residual. Measured on
// tests/fsm_consistency, 120 yr, active_set + FSM, overwrite coupling: the whole-run budget gap was
// -7.263575e+10 (34.84% of recharge), and the SAME run stopped after ONE cycle -- where d_stored is
// 0 by construction -- gave -7.261150e+10. The 120-year residual was the first cycle, essentially in
// its entirety: the supplied initial table drains, FSM spills 6.69e+10 m^3 to the ocean in year one,
// and none of the storage drop that fed it was in the budget.
//
// Rank 0 only, matching PrintValues, which returns early on every other rank and is the sole consumer.
void CaptureInitialStoredVolume(Parameters& params, const ArrayPack& arp) {
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank != 0) return;
  params.stored_volume_initial      = ComputeStoredVolume(params, arp);
  params.have_stored_volume_initial = true;
}

void PrintValues(Parameters& params, const ArrayPack& arp) {
  // total_added_recharge and total_loss_to_ocean_gw are per-rank owned-cell partials
  // (see set_starting_values), so reduce them to global totals. total_loss_to_ocean is
  // accumulated by FillSpillMerge on the full replicated grid on every rank, so it is
  // already global (rank 0's copy is correct) and must NOT be reduced. MPI_Allreduce is
  // collective -- every rank must reach these calls.
  double global_recharge_direct = 0.0;
  double global_runoff_to_surface = 0.0;
  double global_gw_loss_to_ocean = 0.0;
  double global_surface_removed = 0.0;
  double global_evap_removed = 0.0;
  double global_ocean_outflow = 0.0;
  double global_boundary_inflow = 0.0;   // #105: off-map ghost flux under neumann_toposlope; + is INFLOW
  double global_storage_change = 0.0;
  double global_solver_recharge = 0.0;
  MPI_Allreduce(&arp.total_recharge_direct, &global_recharge_direct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_runoff_to_surface, &global_runoff_to_surface, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // TOTAL external input = both channels. Column 9's name has always promised this; it used to deliver
  // only the direct share, so the runoff-ratio water was missing from the budget's "water in" while the
  // lakes it built were present in stored_volume. See benchmark/WATER_BUDGET.md.
  const double global_added_recharge = global_recharge_direct + global_runoff_to_surface;
  MPI_Allreduce(&arp.total_loss_to_ocean_gw, &global_gw_loss_to_ocean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_surface_removed, &global_surface_removed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_evap_removed, &global_evap_removed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_ocean_outflow_gw, &global_ocean_outflow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_boundary_inflow_gw, &global_boundary_inflow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_storage_change, &global_storage_change, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_solver_recharge, &global_solver_recharge, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank != 0) {
    return;  // only rank 0 writes the diagnostic text file
  }

  // Total ocean loss = groundwater part (owned-partial, reduced) + FSM part (already global).
  const double global_loss_to_ocean = global_gw_loss_to_ocean + arp.total_loss_to_ocean;

  std::ofstream textfile(params.textfilename, std::ios_base::app);
  // The run log is the water-BUDGET file: its purpose is verifying conservation to the SNES tolerance
  // (~1e-8 relative), and cross-checking columns against one another (col 9 == col 19 + col 20). The
  // stream default of 6 significant digits cannot support either -- a sum of two printed values misses
  // a printed total by ~5e-7, which is indistinguishable from a real accounting error. 12 digits is
  // still far short of double precision but comfortably below anything we assert on.
  textfile.precision(12);

  double abs_total_wtd_change = 0.0;
  double abs_wtd_mid_change   = 0.0;
  double abs_GW_wtd_change    = 0.0;
  double total_wtd_change     = 0.0;
  double wtd_mid_change       = 0.0;
  double GW_wtd_change        = 0.0;
  double wtd_sum              = 0.0;
  double stored_volume        = 0.0;  // exact Sum storedVolume(wtd)*cell_area -- the physical stored water

  for (int y = 0; y < params.ncells_y; y++) {
    for (int x = 0; x < params.ncells_x; x++) {
      abs_total_wtd_change += std::abs(arp.wtd(x, y) - arp.wtd_old(x, y));
      abs_wtd_mid_change += std::abs(arp.wtd(x, y) - arp.wtd_mid(x, y));
      abs_GW_wtd_change += std::abs(arp.wtd_mid(x, y) - arp.wtd_old(x, y));
      total_wtd_change += (arp.wtd(x, y) - arp.wtd_old(x, y));
      wtd_mid_change += (arp.wtd(x, y) - arp.wtd_mid(x, y));
      GW_wtd_change += (arp.wtd_mid(x, y) - arp.wtd_old(x, y));
      params.infiltration_change += arp.infiltration_array(x, y);
      if (arp.wtd(x, y) > 0) {
        wtd_sum += arp.wtd(x, y) * arp.cell_area[y];
      } else {
        wtd_sum += arp.wtd(x, y) * arp.porosity(x, y) * arp.cell_area[y];
      }
    }
  }

  stored_volume = ComputeStoredVolume(params, arp);

  // The baseline is captured at t=0 by CaptureInitialStoredVolume, NOT here. Taking it on the first
  // report would exclude the first cycle's storage change from d_stored while every flux accumulator
  // includes it -- which was most of the reported residual on a cold start. Assert rather than
  // silently fall back: a zero baseline would make d_stored the absolute volume and look plausible.
  if (!params.have_stored_volume_initial)
    throw std::runtime_error(
        "budget: the t=0 stored volume was never captured; CaptureInitialStoredVolume must run "
        "after the initial water table is loaded and before any stepping.");

  // Two ocean-loss measures, kept SEPARATE on purpose (see benchmark/WATER_BUDGET.md):
  //   * PHYSICAL   -- global_ocean_outflow: the direct Darcy flux across land->ocean faces.
  //   * BUDGET-CLOSING -- inferred by difference so the books balance exactly by construction:
  //       ocean_loss_closing = recharge_in - surface_removed - d(stored_volume).
  // Their difference is the conservation residual: ~0 confirms the physical flux is conservative;
  // a nonzero value is the discretisation-consistency gap (BDF2 startup term + the specific-yield
  // recharge definition), NOT a leak. See the math in WATER_BUDGET.md.
  const double d_stored            = stored_volume - params.stored_volume_initial;
  // Physical balance: recharge = d_stored + evap(->atmosphere) + ocean_outflow(Darcy) + loss_to_ocean(FSM).
  // surface_removed (water skimmed / exfiltrated to FSM) is an INTERNAL GW->FSM transfer, NOT a sink -- FSM
  // either keeps it in a lake (already counted in d_stored) or routes it to the ocean (loss_to_ocean).
  // Counting it here double-counts water FSM recycles into a persistent lake: a closed lake re-skims the same
  // recharge every step, inflating surface_removed to ~8% of recharge while the water sits in storage (found
  // via tests/fsm_fullness + fsm_conservation once the active-set skim delivers its captured water to FSM).
  // So the inferred TOTAL ocean loss is recharge - evap - d_stored, checked against BOTH ocean channels
  // (Darcy outflow + FSM spill). evap (taper 2) leaves to the atmosphere. See benchmark/WATER_BUDGET.md.
  const double ocean_loss_closing  = global_added_recharge - global_evap_removed - d_stored;
  const double budget_residual     = ocean_loss_closing - global_ocean_outflow - global_loss_to_ocean;

  // EXACT (machine-zero) budget residual from the solver's accumulated discrete terms (Picard path):
  // storage_change = solver_recharge - ocean_outflow - surface_removed holds to the SNES tolerance,
  // so this residual is ~0 (unlike the physical budget_residual, which carries the BDF2-startup gap).
  // Its departure from 0, once the numerics are exact, is a clean measure of any UNaccounted vertical
  // flux (e.g. evap_mode-0 surface discard / the water handed to FSM). See benchmark/WATER_BUDGET.md.
  // Meaningless if any step ran under a scheme the accumulator cannot express as one per-step identity.
  // Report NaN rather than a number, so a stale zero is never mistaken for a closed budget. Every
  // scheme currently in the code CAN be expressed, TR-BDF2 included (its two stages telescope; see
  // src/tr_bdf2_coefficients.hpp), so this guard is now a backstop for a future scheme rather than a
  // live case -- but it stays, because the failure it prevents is silent.
  // + global_boundary_inflow, and the SIGN IS THE POINT (#105): the off-map ghost flux under
  // boundaries.land: neumann_toposlope is a SOURCE, not a sink. Terrain rising away from the domain edge
  // puts h_ghost above h_edge and drives water IN. It is added alongside recharge rather than subtracted
  // alongside the outflows, and it is a separate term rather than part of global_ocean_outflow because
  // water arriving from off-map upslope is not ocean outflow -- folding them together would close the
  // ledger while describing the wrong physics. Zero by construction on a flat edge, and on every
  // ocean-ringed fixture (no land edge at all), which is why no budget suite ever saw it.
  const double exact_budget_residual = arp.exact_budget_valid
                                           ? global_solver_recharge + global_boundary_inflow
                                                 - global_storage_change - global_ocean_outflow
                                                 - global_surface_removed - global_evap_removed
                                           : std::numeric_limits<double>::quiet_NaN();

  textfile << params.cycles_done << " " << total_wtd_change << " " << GW_wtd_change << " " << wtd_mid_change << " "
           << abs_total_wtd_change << " " << abs_GW_wtd_change << " " << abs_wtd_mid_change << " "
           << params.infiltration_change << " " << global_added_recharge << " " << global_loss_to_ocean << " "
           << wtd_sum << " " << global_surface_removed << " " << global_ocean_outflow << " "
           << stored_volume << " " << ocean_loss_closing << " " << budget_residual << " "
           << exact_budget_residual << " " << global_evap_removed << " "
           // Columns 19-20: the two input channels whose sum is column 9. APPENDED so every existing
           // column index stays put -- three test scripts parse this file positionally.
           << global_recharge_direct << " " << global_runoff_to_surface << " "
           // Columns 21-23: the DENOMINATORS. Elapsed time is ACCUMULATED from the accepted steps, not
           // derived as cycles_done*report_seconds: that derivation assumes every cycle covers one
           // report span, which is true for the fixed-dt and adaptive loops but FALSE for
           // -wtm_dt_continuation, whose loop runs report_steps STEPS at a dt it may grow. The derived
           // form reported 20.000 yr for a continuation run that had simulated 5.77 yr.
           << params.elapsed_time_s << " "
           << params.solves_done << " " << params.rejects_done << " "
           // Columns 24-25: per-cycle change in WATER VOLUME (|S*Dwtd|), the units the model's own
           // equilibrium stop, adaptive step target and budget all use. Column 5 remains the HEAD change.
           << params.last_cycle_dw_volume << " " << params.last_cycle_rms_volume << " " << std::endl;

  textfile.close();
}
