#include "ArrayPack.hpp"
#include "fill_spill_merge.hpp"
#include "parameters.hpp"
#include "update_effective_storativity.hpp"  // storedVolume, for the exact stored-water budget

#include <richdem/common/Array2D.hpp>

#define OMPI_SKIP_MPICXX 1  // we use the MPI C API only; skip the deprecated C++ bindings
#include <mpi.h>

#include <petscsys.h>

namespace rd = richdem;
namespace dh = richdem::dephier;

// Boundary conditions default to the mask-aware ghost-node scheme (task #96): Dirichlet h=0 at ocean edges,
// land-slope Neumann at land edges, computed at the true domain edge. The legacy method -- force every domain
// edge to sea-level ocean via setEdges(0) ("1-cell sea-level padding, all-Dirichlet") -- is retained ONLY as
// a verification tool behind -wtm_dev_padded_dirichlet, to check that the ghost scheme reproduces it on an
// ocean-ringed domain (where the two coincide). Read the PETSc option here since irf runs before the solver
// parses its flags. Absent (default) => mask-aware ghost boundary; present => legacy padded all-Dirichlet.
static bool padded_dirichlet_requested() {
  PetscBool on = PETSC_FALSE;
  PetscOptionsGetBool(nullptr, nullptr, "-wtm_dev_padded_dirichlet", &on, nullptr);
  return on == PETSC_TRUE;
}

// The legacy padded-Dirichlet method (setEdges(0)) OVERWRITES the domain-edge ring to ocean. That silently
// discards a ring of real land data if the boundary is not already ocean, so guard it: require every
// domain-boundary cell to be ocean (sea level) and FAIL LOUDLY otherwise. Verification runs use an
// ocean-ringed (or zero-padded) domain, where setEdges is a no-op and this passes. Production land-edge runs
// must use the default mask-aware boundary (no-flow Neumann at land edges) instead.
static void require_ocean_boundary(const rd::Array2D<float>& mask) {
  const int H = mask.height(), W = mask.width();
  long land = 0;
  for (int x = 0; x < W; x++) { land += (mask(x, 0) != 0.f); land += (mask(x, H - 1) != 0.f); }
  for (int y = 0; y < H; y++) { land += (mask(0, y) != 0.f); land += (mask(W - 1, y) != 0.f); }
  if (land > 0)
    throw std::runtime_error(
        "-wtm_dev_padded_dirichlet requires an all-ocean domain boundary; found " + std::to_string(land) +
        " non-ocean boundary cell(s). Zero-pad the domain with an ocean ring, or drop the flag to use the "
        "default mask-aware boundary (no-flow at land edges).");
}

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
  if (padded_dirichlet_requested()) { require_ocean_boundary(arp.land_mask); arp.land_mask.setEdges(0); }

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
  if (padded_dirichlet_requested()) { require_ocean_boundary(arp.land_mask); arp.land_mask.setEdges(0); }

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
  if (padded_dirichlet_requested()) { require_ocean_boundary(arp.land_mask); arp.land_mask.setEdges(0); }

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

/// Here, we use the number of cells per degree (a user-defined value),
/// the southern-most latitude of the domain (also user-defined),
/// and the radius of the Earth to calculate the latitude of each row of cells,
/// the size of a cell in the N-S and E-W directions, and the area of each cell.
// Set the grid geometry (ns_deg_per_cell, ew_deg_per_cell, southern_edge) from the input topography's GDAL
// geotransform (#124) -- the authoritative source. The DEPRECATED grid: config block is used only as a
// fallback for inputs that carry no geotransform. Runs on rank 0, where arp.topo carries the geotransform;
// the three scalars are then broadcast (see WTM.cpp). The GDAL geotransform is [x0, dx, 0, y0, 0, dy] with
// (x0, y0) the top-left (north-west) corner and dy < 0 for a north-up raster. Pixels need not be square
// (dx != |dy|, e.g. a clipped tile).
void derive_grid_geometry(Parameters& params, const ArrayPack& arp) {
  const auto& gt = arp.topo.geotransform;  // [x0, dx, 0, y0, 0, dy]

  // RichDEM signals a missing geotransform with the sentinel {1000, 1, 0, 1000, 0, -1} (Array2D::loadGDAL).
  const bool no_georef =
      (gt.size() != 6) || (gt[0] == 1000. && gt[1] == 1. && gt[3] == 1000. && gt[5] == -1.);

  if (no_georef) {
    // Fallback: ungeoreferenced input -> use the deprecated grid: override, which must then be supplied.
    if (params.cells_per_degree <= 0 || std::isnan(params.southern_edge)) {
      throw std::runtime_error(
          "Input topography has no geotransform and no grid: override was given. Provide a georeferenced "
          "raster, or set the (deprecated) grid: block (cells_per_degree + southern_edge) in the config.");
    }
    std::cerr << "WARNING: the input topography carries no geotransform; falling back to the DEPRECATED "
                 "grid: block (cells_per_degree/southern_edge). Provide a georeferenced raster (#124)."
              << std::endl;
    params.ns_deg_per_cell = 1.0 / params.cells_per_degree;
    params.ew_deg_per_cell = 1.0 / params.cells_per_degree;  // fallback grid is square by construction
    return;                                                  // southern_edge already set from config
  }

  // A real geotransform is authoritative; a stray grid: override is ignored (warn once).
  if (params.cells_per_degree > 0) {
    std::cerr << "WARNING: the grid: block (cells_per_degree/southern_edge) is DEPRECATED and IGNORED because "
                 "the input topography carries a geotransform; geometry is read from it (#124). Remove grid:."
              << std::endl;
  }

  // Phase 1 supports geographic (lat-lon) grids only: a projected CRS puts dx/dy in metres and makes the
  // cos(lat) treatment wrong -> Phase 2 (#124, projected + ellipsoidal area).
  if (arp.topo.projection.find("PROJCRS") != std::string::npos ||
      arp.topo.projection.find("PROJCS") != std::string::npos) {
    throw std::runtime_error(
        "Projected CRS detected in the input topography; only geographic (lat-lon) grids are supported "
        "for now (#124 Phase 2).");
  }
  if (gt[5] >= 0.) {
    throw std::runtime_error("Expected a north-up input raster (geotransform dy < 0), but got dy >= 0.");
  }

  params.ew_deg_per_cell  = gt[1];                              // E-W degrees per cell (dx, longitude)
  params.ns_deg_per_cell  = -gt[5];                             // N-S degrees per cell (|dy|, latitude)
  params.southern_edge    = gt[3] + params.ncells_y * gt[5];    // north edge + H*dy(<0) = southern edge
  params.cells_per_degree = 1.0 / params.ns_deg_per_cell;       // nominal, for the run log/printout
}

void cell_size_area(Parameters& params, ArrayPack& arp) {
  // compute changing cell size and distances between
  // cells as these change with latitude:

  constexpr double earth_radius = 6371000.;    // metres
  constexpr double deg_to_rad   = M_PI / 180;  // convert degrees to radians

  // Radius * Pi = Distance from N to S pole
  // Distance / 180 = Meters / degree latitude
  const auto meters_per_degree = earth_radius * deg_to_rad;

  // N-S Meters per cell; distance between lines of latitude is a constant (ns_deg_per_cell from the
  // geotransform, #124). For a square override this equals meters_per_degree / cells_per_degree.
  params.cellsize_n_s_metres = meters_per_degree * params.ns_deg_per_cell;

  // initialise some arrays

  // size of a cell in the east-west direction at the centre of the cell (metres)
  arp.cellsize_e_w_metres.resize(params.ncells_y);
  // cell area (metres squared)
  arp.cell_area.resize(params.ncells_y);
  // Conservative finite-volume flux conductance geometry (see benchmark/GRID_CONVENTION.md).
  // For a face flux G = T * L_wall / d_centre, the per-row geometric factor L_wall/d_centre is:
  //   geom_ew = cellsize_n_s / cellsize_e_w[j]        (E-W face: N-S wall, E-W centre distance)
  //   geom_n  = cellsize_e_w[N face] / cellsize_n_s   (N face: E-W wall at the northern edge)
  //   geom_s  = cellsize_e_w[S face] / cellsize_n_s   (S face: E-W wall at the southern edge)
  // The N/S factors use the FACE (cell-edge) E-W length so a shared face gives equal-and-opposite
  // volume fluxes -> exact conservation. cell_area = cellsize_n_s^2 / geom_ew.
  arp.geom_ew.resize(params.ncells_y);
  arp.geom_n.resize(params.ncells_y);
  arp.geom_s.resize(params.ncells_y);

  // used to calculate cell latitude in radians.
  // southern edge of the domain in degrees, plus the number of cells up from this
  // location/the number of cells per degree, converted to radians.
  const auto cell_position_latitude = [&](const auto cell_idx) {
    return (params.southern_edge + cell_idx * params.ns_deg_per_cell) * deg_to_rad;
  };

  for (int j = 0; j < params.ncells_y; j++) {
    // latitude, in radians, at the southern edge of a cell:
    const double latitude_radians_S = cell_position_latitude(j);
    // latitude, in radians, at the northern edge of a cell (add a cell; equal to the southern edge of the next cell):
    const double latitude_radians_N = cell_position_latitude(j + 1);

    // distance between lines of longitude varies with latitude. The E-W base is the true longitude cell
    // width ew_deg_per_cell (from the geotransform, #124), which need not equal the N-S spacing; cos(lat)
    // does the metric conversion. For a square override this reduces to cellsize_n_s * cos(lat).

    // distance at the northern edge of the cell for the given latitude:
    const double cellsize_e_w_metres_N = meters_per_degree * params.ew_deg_per_cell * std::cos(latitude_radians_N);
    // distance at the southern edge of the cell for the given latitude:
    const double cellsize_e_w_metres_S = meters_per_degree * params.ew_deg_per_cell * std::cos(latitude_radians_S);

    arp.cellsize_e_w_metres[j] = (cellsize_e_w_metres_N + cellsize_e_w_metres_S) / 2.;

    // cell area computed as a trapezoid, using unchanging north-south distance,
    // and east-west average distance.
    arp.cell_area[j] = params.cellsize_n_s_metres * arp.cellsize_e_w_metres[j];

    if (arp.cell_area[j] < 0) {
      throw std::runtime_error("Cell with a negative area was found!");
    }

    // Conservative FV flux geometry (L_wall / d_centre per face; see GRID_CONVENTION.md). The N/S
    // factors use the cell-EDGE (face) E-W lengths so adjacent cells share the face value exactly.
    arp.geom_ew[j] = params.cellsize_n_s_metres / arp.cellsize_e_w_metres[j];
    arp.geom_n[j]  = cellsize_e_w_metres_N / params.cellsize_n_s_metres;
    arp.geom_s[j]  = cellsize_e_w_metres_S / params.cellsize_n_s_metres;
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

  // get the starting runoff using precip and evap inputs. Taper-first (mode-independent): the taper
  // (2/3) governs evaporation via the implicit E_eff, so feed just precip regardless of evap_mode --
  // the smooth removal auto-zeroes standing water, so no independent wtd=0 under the taper. Otherwise
  // mode 1 evaporates surface water at owe (persists); mode 0 removes all surface water (wtd=0;
  // GW-alone testing, Fan Reinfelder et al. 2013). Matches the per-cycle path.
  const bool evap_taper = FanDarcyGroundwater::evap_taper_on();
  std::cout << (evap_taper       ? "p updating the recharge field (taper)"
               : params.evap_mode ? "p updating the recharge field"
                                  : "p removing all surface water")
            << std::endl;
#pragma omp parallel for default(none) shared(arp, params, evap_taper)
  for (unsigned int i = 0; i < arp.topo.size(); i++) {
    if (evap_taper) {
      arp.rech(i) = arp.precip(i) / seconds_in_a_year * params.deltat;
    } else if (arp.wtd(i) > 0) {  // surface water present
      if (!params.evap_mode)
        arp.wtd(i) = 0;  // evap_mode 0: remove all surface water (GW-alone testing)
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
void PrintValues(Parameters& params, const ArrayPack& arp) {
  // total_added_recharge and total_loss_to_ocean_gw are per-rank owned-cell partials
  // (see set_starting_values), so reduce them to global totals. total_loss_to_ocean is
  // accumulated by FillSpillMerge on the full replicated grid on every rank, so it is
  // already global (rank 0's copy is correct) and must NOT be reduced. MPI_Allreduce is
  // collective -- every rank must reach these calls.
  double global_added_recharge = 0.0;
  double global_gw_loss_to_ocean = 0.0;
  double global_surface_removed = 0.0;
  double global_evap_removed = 0.0;
  double global_ocean_outflow = 0.0;
  double global_storage_change = 0.0;
  double global_solver_recharge = 0.0;
  MPI_Allreduce(&arp.total_added_recharge, &global_added_recharge, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_loss_to_ocean_gw, &global_gw_loss_to_ocean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_surface_removed, &global_surface_removed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_evap_removed, &global_evap_removed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&arp.total_ocean_outflow_gw, &global_ocean_outflow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
      stored_volume += storedVolume(arp.wtd(x, y), arp.porosity(x, y)) * arp.cell_area[y];
    }
  }

  // Capture the initial stored volume once, so d(stored_volume) can drive the budget-closing check.
  if (!params.have_stored_volume_initial) {
    params.stored_volume_initial      = stored_volume;
    params.have_stored_volume_initial = true;
  }

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
  const double exact_budget_residual = global_solver_recharge - global_storage_change - global_ocean_outflow
                                       - global_surface_removed - global_evap_removed;

  textfile << params.cycles_done << " " << total_wtd_change << " " << GW_wtd_change << " " << wtd_mid_change << " "
           << abs_total_wtd_change << " " << abs_GW_wtd_change << " " << abs_wtd_mid_change << " "
           << params.infiltration_change << " " << global_added_recharge << " " << global_loss_to_ocean << " "
           << wtd_sum << " " << global_surface_removed << " " << global_ocean_outflow << " "
           << stored_volume << " " << ocean_loss_closing << " " << budget_residual << " "
           << exact_budget_residual << " " << global_evap_removed << " " << std::endl;

  textfile.close();
}
