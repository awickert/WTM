// Unit tests for cell_size_area (src/irf.cpp): the per-row cell geometry derived from the GDAL
// geotransform (#124). Spherical geometry -- N-S cell height is constant, E-W width shrinks with cos
// (latitude) -- plus the conservative finite-volume face factors (geom_ew / geom_n / geom_s, see
// benchmark/GRID_CONVENTION.md). The golden test proves this reproduces the old cells_per_degree path;
// these tests prove it independently matches ANALYTIC spherical geometry and, crucially, the shared-face
// identity geom_n[j] == geom_s[j+1] that guarantees exact FV conservation.
//
// Pure math over the Parameters/ArrayPack structs (no PETSc/MPI/IO); compiled into test_dmda.x and run by
// tests/run_unit_tests.sh. doctest's main lives in test_dmda_gather.cpp; this TU only registers TEST_CASEs.

#include "doctest.h"
#include "irf.hpp"

#include <cmath>

namespace {

constexpr double kEarthRadius = 6371000.0;      // must match cell_size_area
constexpr double kDegToRad    = M_PI / 180.0;
constexpr double kMPerDegree  = kEarthRadius * kDegToRad;

// Build a params + arp for a north-up domain and run the geometry computation.
void compute(Parameters& p, ArrayPack& a, double southern_edge, double ns_deg, double ew_deg, int ny) {
  p.southern_edge   = southern_edge;
  p.ns_deg_per_cell = ns_deg;
  p.ew_deg_per_cell = ew_deg;
  p.ncells_y        = ny;
  cell_size_area(p, a);
}

// Analytic E-W width at row j: trapezoidal average of cos(lat) at the cell's S and N edges.
double expect_ew(double southern_edge, double ns_deg, double ew_deg, int j) {
  const double latS = (southern_edge + j * ns_deg) * kDegToRad;
  const double latN = (southern_edge + (j + 1) * ns_deg) * kDegToRad;
  return kMPerDegree * ew_deg * 0.5 * (std::cos(latN) + std::cos(latS));
}

}  // namespace

TEST_CASE("cell_size_area: constant N-S height = R * dphi") {
  Parameters p; ArrayPack a;
  compute(p, a, 10.0, 0.5, 0.5, 8);
  CHECK(p.cellsize_n_s_metres == doctest::Approx(kMPerDegree * 0.5));
}

TEST_CASE("cell_size_area: E-W width follows cos(latitude), analytic + monotone") {
  Parameters p; ArrayPack a;
  const double south = 0.0, ns = 1.0, ew = 1.0;  const int ny = 60;  // lat 0..60N
  compute(p, a, south, ns, ew, ny);
  for (int j = 0; j < ny; j++)
    CHECK(a.cellsize_e_w_metres[j] == doctest::Approx(expect_ew(south, ns, ew, j)));
  // Strictly shrinking northward (cos decreases with |lat|); near the equator ~ the full R*dlon.
  for (int j = 1; j < ny; j++) CHECK(a.cellsize_e_w_metres[j] < a.cellsize_e_w_metres[j - 1]);
  CHECK(a.cellsize_e_w_metres[0] == doctest::Approx(kMPerDegree * ew).epsilon(2e-4));  // row spans 0..1 deg
}

TEST_CASE("cell_size_area: non-square spacing (ew != ns) scales the E-W base independently") {
  Parameters p; ArrayPack a;
  compute(p, a, 0.0, 1.0, 2.0, 5);  // E-W spacing twice N-S
  CHECK(p.cellsize_n_s_metres == doctest::Approx(kMPerDegree * 1.0));
  for (int j = 0; j < 5; j++)
    CHECK(a.cellsize_e_w_metres[j] == doctest::Approx(expect_ew(0.0, 1.0, 2.0, j)));
}

TEST_CASE("cell_size_area: area = height * width and strictly positive") {
  Parameters p; ArrayPack a;
  compute(p, a, -20.0, 0.5, 0.75, 40);  // straddles the equator (lat -20..0)
  for (int j = 0; j < 40; j++) {
    CHECK(a.cell_area[j] == doctest::Approx(p.cellsize_n_s_metres * a.cellsize_e_w_metres[j]));
    CHECK(a.cell_area[j] > 0.0);
  }
}

TEST_CASE("cell_size_area: FV shared-face identity geom_n[j] == geom_s[j+1] (exact conservation)") {
  Parameters p; ArrayPack a;
  compute(p, a, 12.34, 0.5, 0.9, 30);  // arbitrary, non-square, off-equator
  // The north face of cell j and the south face of cell j+1 are the SAME line of latitude, so their
  // geometric flux factors must be bit-for-bit equal -- this is what makes adjacent-cell fluxes
  // equal-and-opposite (exact mass conservation). An off-by-one in the edge latitudes would break it.
  for (int j = 0; j < 29; j++)
    CHECK(a.geom_n[j] == a.geom_s[j + 1]);  // exact bit equality: same latitude, same formula
  // geom_ew is the reciprocal-style E-W factor cellsize_n_s / cellsize_e_w[j].
  for (int j = 0; j < 30; j++)
    CHECK(a.geom_ew[j] == doctest::Approx(p.cellsize_n_s_metres / a.cellsize_e_w_metres[j]));
}

TEST_CASE("cell_size_area: hemisphere symmetry about the equator") {
  Parameters p; ArrayPack a;
  const double ns = 1.0, ew = 1.0;  const int ny = 20;  // lat -10..+10, symmetric about 0
  compute(p, a, -10.0, ns, ew, ny);
  // Row j and row (ny-1-j) sit at mirror-image latitude bands, so their E-W widths must match.
  for (int j = 0; j < ny / 2; j++)
    CHECK(a.cellsize_e_w_metres[j] == doctest::Approx(a.cellsize_e_w_metres[ny - 1 - j]));
}
