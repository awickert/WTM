#pragma once

// GRID GEOMETRY -- the single source of truth for cell size and area, shared by BOTH binaries.
//
// WHY ITS OWN TRANSLATION UNIT. wtm.x and dephier.x both need this, and dephier.x is deliberately
// PETSc-free, so it cannot link irf.cpp. Before this file existed, run_dephier.cpp carried its own
// inline copy -- which never used ew_deg_per_cell and so assumed square pixels. On a non-square tile
// (dx != |dy|, exactly what the GDAL geotransform work #124 added support for) its cell areas were
// wrong by the aspect ratio: measured 2.000x at ns_deg=0.1, ew_deg=0.2. Since cell_area is passed
// INTO GetDepressionHierarchy, every depression volume it built was wrong by that factor.
//
// Duplicated geometry is what caused that, so there must be exactly one copy. Anything that needs
// cell size or area includes this header; nothing recomputes it.
//
// Unit-tested against analytic spherical geometry in src/test_geometry.cpp, including the shared-face
// identity geom_n[j] == geom_s[j+1] that guarantees exact finite-volume conservation.

#include "ArrayPack.hpp"
#include "parameters.hpp"

/// Set ns_deg_per_cell / ew_deg_per_cell / southern_edge from the input topography's GDAL
/// geotransform (#124). See the definition for the deprecated grid: fallback.
void derive_grid_geometry(Parameters& params, const ArrayPack& arp);

/// Per-row cell sizes, areas and the conservative finite-volume face factors.
void cell_size_area(Parameters& params, ArrayPack& arp);
