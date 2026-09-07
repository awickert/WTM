#include "ArrayPack.hpp"
#include "grid_geometry.hpp"
#include "parameters.hpp"

#include <richdem/common/Array2D.hpp>
#include <richdem/common/ProgressBar.hpp>
#include <richdem/common/timer.hpp>

namespace rd = richdem;
namespace dh = richdem::dephier;

constexpr double deg_to_rad = M_PI / 180.0;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Syntax: " << argv[0] << " <Configuration File>" << std::endl;
    return -1;
  }

  ArrayPack arp;
  std::cerr << "Argv" << argv << std::endl;
  Parameters params(argv[1]);

  // load in the data files: topography and mask.
  arp.topo = rd::Array2D<float>(params.get_path(params.time_start, "topography"));

  arp.land_mask = rd::Array2D<float>(params.get_path(params.time_start, "mask"));

  // width and height in number of cells in the array
  params.ncells_x = arp.topo.width();
  params.ncells_y = arp.topo.height();

  // initialise the label and flow direction arrays:
  rd::Array2D<dh::dh_label_t> label(params.ncells_x, params.ncells_y, dh::NO_DEP);  // No cells are part of a depression

  rd::Array2D<dh::dh_label_t> final_label(
      params.ncells_x, params.ncells_y, dh::NO_DEP);  // No cells are part of a depression

  rd::Array2D<rd::flowdir_t> flowdirs(params.ncells_x, params.ncells_y, rd::NO_FLOW);  // No cells flow anywhere

  // GRID GEOMETRY -- from the shared translation unit, NOT computed here.
  //
  // This block used to be an inline copy of irf.cpp's cell_size_area, and it never used
  // ew_deg_per_cell: it assumed E-W spacing equals N-S spacing. On a non-square tile (dx != |dy|,
  // which is exactly what the GDAL geotransform work #124 added support for) that made every cell
  // area wrong by the aspect ratio -- measured 2.000x at ns_deg=0.1, ew_deg=0.2. cell_area is passed
  // straight into GetDepressionHierarchy below, so every depression VOLUME in the hierarchy this
  // tool built was wrong by that factor, silently, on any clipped tile.
  //
  // Two implementations of one piece of geometry is what caused it, so there is now exactly one.
  derive_grid_geometry(params, arp);   // ns/ew_deg_per_cell + southern_edge, from the geotransform
  cell_size_area(params, arp);         // per-row cell sizes, areas, and the FV face factors

  // Label the ocean cells. This is a precondition for using
  //`GetDepressionHierarchy()`.
#pragma omp parallel for default(none) shared(arp, final_label, label)
  for (unsigned int i = 0; i < label.size(); i++) {
    if (arp.land_mask(i) == 0) {
      label(i)       = dh::OCEAN;
      final_label(i) = dh::OCEAN;
    }
  }

  // Generate flow directions, label all the depressions, and get the hierarchy
  // connecting them
  std::cout << "going to do depression hierarchy" << std::endl;
  auto deps =
      dh::GetDepressionHierarchy<float, rd::Topology::D8>(arp.topo, arp.cell_area, label, final_label, flowdirs);

  // We are finished, save the result.
  std::cout << "done with processing" << std::endl;

  label.saveGDAL("label.tif");
  final_label.saveGDAL("final_label.tif");

  return 0;
}
