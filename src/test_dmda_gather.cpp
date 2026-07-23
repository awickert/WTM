// Unit tests for DMDAFullGridGather (src/dmda_gather.hpp) -- the gather/scatter
// primitives that the ArrayPack distribution (benchmark/DISTRIBUTED_ARP_DESIGN.md)
// is built on.
//
// These are MPI tests: build test_dmda.x and run under mpirun at several rank
// counts, e.g.  mpirun -n 1 ./test_dmda.x ; mpirun -n 2 ./test_dmda.x ; ...
// The row-major layout (index = j*Mx + i, x fastest) must hold independent of
// how PETSc decomposes the grid, so the value of these tests is running them at
// n > 1. tests/run_unit_tests.sh runs the full rank sweep.
//
// doctest provides the assertions; PetscInitialize/Finalize wrap main. Every
// rank runs every assertion; a failure on any rank makes that process exit
// non-zero, which mpirun propagates.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "dmda_gather.hpp"

#include <petscdm.h>
#include <petscdmda.h>

#include <vector>

namespace {

// Unique, exactly-representable value per (i,j) so equality checks are strict.
double encode(int i, int j) {
  return 1000000.0 * j + i + 0.25;
}

// Build a DMDA of the requested global size with the default (PETSC_DECIDE)
// decomposition over the run's communicator.
DM make_da(PetscInt Mx, PetscInt My) {
  DM da;
  DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, Mx, My, PETSC_DECIDE,
               PETSC_DECIDE, 1, 1, nullptr, nullptr, &da);
  DMSetFromOptions(da);
  DMSetUp(da);
  return da;
}

// Fill a global vector's owned cells with encode(i,j).
void fill_owned(DM da, Vec g) {
  PetscScalar** a;
  DMDAVecGetArray(da, g, &a);
  PetscInt xs, ys, xm, ym;
  DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr);
  for (PetscInt j = ys; j < ys + ym; j++)
    for (PetscInt i = xs; i < xs + xm; i++) a[j][i] = encode(i, j);
  DMDAVecRestoreArray(da, g, &a);
}

PetscMPIInt my_rank() {
  PetscMPIInt r;
  MPI_Comm_rank(PETSC_COMM_WORLD, &r);
  return r;
}

// A spread of grid shapes: square, wide, tall, sizes not divisible by common
// rank counts, and degenerate 1-wide / 1-tall strips.
const std::vector<std::pair<PetscInt, PetscInt>> kShapes = {
    {7, 5}, {5, 7}, {16, 16}, {13, 4}, {4, 13}, {1, 11}, {11, 1}, {100, 3}, {3, 100}};

}  // namespace

TEST_CASE("gatherToAll: every rank receives the exact row-major full field") {
  for (auto [Mx, My] : kShapes) {
    DM da = make_da(Mx, My);
    Vec g;
    DMCreateGlobalVector(da, &g);
    fill_owned(da, g);

    DMDAFullGridGather gs(da);
    CHECK(gs.width() == Mx);
    CHECK(gs.height() == My);

    std::vector<double> full;
    gs.gatherToAll(g, full);

    REQUIRE(full.size() == static_cast<size_t>(Mx * My));
    int bad = 0;
    for (PetscInt j = 0; j < My; j++)
      for (PetscInt i = 0; i < Mx; i++)
        if (full[j * Mx + i] != encode(i, j)) bad++;
    CHECK(bad == 0);  // holds on ALL ranks (that is the point of gatherToAll)

    VecDestroy(&g);
    DMDestroy(&da);
  }
}

TEST_CASE("gatherToZero: rank 0 gets the field, other ranks get empty") {
  for (auto [Mx, My] : kShapes) {
    DM da = make_da(Mx, My);
    Vec g;
    DMCreateGlobalVector(da, &g);
    fill_owned(da, g);

    DMDAFullGridGather gs(da);
    std::vector<double> full;
    gs.gatherToZero(g, full);

    if (my_rank() == 0) {
      REQUIRE(full.size() == static_cast<size_t>(Mx * My));
      int bad = 0;
      for (PetscInt j = 0; j < My; j++)
        for (PetscInt i = 0; i < Mx; i++)
          if (full[j * Mx + i] != encode(i, j)) bad++;
      CHECK(bad == 0);
    } else {
      CHECK(full.empty());
    }
    VecDestroy(&g);
    DMDestroy(&da);
  }
}

TEST_CASE("scatterFromZero then gatherToZero round-trips identity") {
  for (auto [Mx, My] : kShapes) {
    DM da = make_da(Mx, My);
    Vec g;
    DMCreateGlobalVector(da, &g);
    VecSet(g, -999.0);  // poison, so a missed cell shows up

    DMDAFullGridGather gs(da);

    std::vector<double> full;
    if (my_rank() == 0) {
      full.assign(Mx * My, 0.0);
      for (PetscInt j = 0; j < My; j++)
        for (PetscInt i = 0; i < Mx; i++) full[j * Mx + i] = encode(i, j);
    }

    gs.scatterFromZero(full, g);  // rank0 full -> distributed

    std::vector<double> back;
    gs.gatherToZero(g, back);  // distributed -> rank0 full

    if (my_rank() == 0) {
      int bad = 0;
      for (PetscInt j = 0; j < My; j++)
        for (PetscInt i = 0; i < Mx; i++)
          if (back[j * Mx + i] != encode(i, j)) bad++;
      CHECK(bad == 0);
    }
    VecDestroy(&g);
    DMDestroy(&da);
  }
}

TEST_CASE("scatterFromZero delivers each rank its correct owned cells") {
  for (auto [Mx, My] : kShapes) {
    DM da = make_da(Mx, My);
    Vec g;
    DMCreateGlobalVector(da, &g);
    VecSet(g, -999.0);

    DMDAFullGridGather gs(da);
    std::vector<double> full;
    if (my_rank() == 0) {
      full.assign(Mx * My, 0.0);
      for (PetscInt j = 0; j < My; j++)
        for (PetscInt i = 0; i < Mx; i++) full[j * Mx + i] = encode(i, j);
    }
    gs.scatterFromZero(full, g);

    // Each rank checks its OWNED cells directly against encode(i,j).
    PetscScalar** a;
    DMDAVecGetArray(da, g, &a);
    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr);
    int bad = 0;
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++)
        if (a[j][i] != encode(i, j)) bad++;
    DMDAVecRestoreArray(da, g, &a);
    CHECK(bad == 0);

    VecDestroy(&g);
    DMDestroy(&da);
  }
}

TEST_CASE("gatherToAll then scatterFromZero round-trips a distributed field") {
  for (auto [Mx, My] : kShapes) {
    DM da = make_da(Mx, My);
    Vec g;
    DMCreateGlobalVector(da, &g);
    fill_owned(da, g);

    DMDAFullGridGather gs(da);
    std::vector<double> full;
    gs.gatherToAll(g, full);  // distributed -> full on all ranks

    Vec g2;
    DMCreateGlobalVector(da, &g2);
    VecSet(g2, -999.0);
    gs.scatterFromZero(full, g2);  // full -> distributed (uses rank 0's copy)

    // g and g2 must now be identical.
    PetscReal nrm;
    VecAXPY(g2, -1.0, g);  // g2 -= g
    VecNorm(g2, NORM_INFINITY, &nrm);
    CHECK(nrm == 0.0);

    VecDestroy(&g);
    VecDestroy(&g2);
    DMDestroy(&da);
  }
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  const int res = ctx.run();
  PetscFinalize();
  return res;
}
