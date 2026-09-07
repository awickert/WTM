// Grid geometry shared by wtm.x and dephier.x -- see grid_geometry.hpp for why this is its own
// translation unit. Moved verbatim from irf.cpp; no logic changed.

#include "grid_geometry.hpp"

#include <richdem/common/Array2D.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace rd = richdem;

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
