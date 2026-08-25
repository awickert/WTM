// Unit tests for the FillSpillMerge WATER LEDGER -- what FSM does to arp.wtd, arp.runoff and
// arp.total_loss_to_ocean, and the conservation identity that relates them.
//
// WHY THIS EXISTS. The GW<->FSM handoff has repeatedly been mis-read (see GH awickert/WTM#116, the
// FSM-delta-source work): the recurring error is to expect FSM's per-cell volume change to sum to
// ZERO over the domain, and then to read the (correct, non-zero) sum as evidence of a double-count.
// It is not zero. FSM is handed water in arp.runoff that was already debited from aquifer storage
// earlier in the step (the exfiltration constraint), and it RETURNS that water to the
// domain -- so the sum of its volume changes equals the runoff it consumed minus what it spilled to
// the ocean. These tests pin that identity so the contract is checkable instead of remembered.
//
// The three facts pinned here:
//   1. TRANSFER, NOT DUPLICATION. For a cell holding above-surface water, FSM does
//      `arp.runoff += arp.wtd; arp.wtd = 0` (fill_spill_merge.hpp:341-343). The cell's standing water
//      and its pre-existing runoff are SUMMED and routed once. Neither is counted twice, neither is
//      dropped.
//   2. THE LEDGER IDENTITY (the one that gets mis-remembered):
//         sum_cells [ v(wtd_post) - v(wtd_pre) ] * area
//              == sum_cells [ runoff_pre - runoff_post ] * area  -  delta(total_loss_to_ocean)
//      i.e. FSM's net volume change is (water handed in) - (water spilled to sea). NOT zero.
//   3. OCEAN cells absorb runoff into total_loss_to_ocean and are left holding nothing
//      (fill_spill_merge.hpp:348-351).
//
// Pure serial rank-0 logic (FillSpillMerge is serial by construction); compiled into test_dmda.x
// alongside the other unit TUs and run by tests/run_unit_tests.sh. doctest's main + PetscInitialize
// live in test_dmda_gather.cpp; this TU only registers TEST_CASEs.

#include "doctest.h"

#include "ArrayPack.hpp"
#include "fill_spill_merge.hpp"
#include "parameters.hpp"

#include <vector>

namespace rd = richdem;
namespace dh = richdem::dephier;

namespace {

constexpr int    W    = 9;      // grid width (x = 0 is ocean)
constexpr int    H    = 5;      // grid height
constexpr double AREA = 1.0e6;  // m^2 per cell; uniform, so every row has the same cell_area
constexpr double PORO = 0.25;   // uniform porosity

// FSM's stored-volume convention, per unit area: above the land surface water STANDS (slope 1);
// below it fills pore space (slope porosity) -- see the wtd_vol accumulation at
// fill_spill_merge.hpp:581. NOTE this is the UNSMOOTHED corner. storedVolume() in
// update_effective_storativity.hpp smooths it over g_storativity_surface_smoothing_width; FSM does
// not, so the ledger must be evaluated with this exact form, not with storedVolume().
double vol_per_area(double wtd) {
  return wtd > 0.0 ? wtd : PORO * wtd;
}

// East-west topography. Topography depends on x only, so the depression is a one-cell-wide trench
// spanning all H rows, which makes every volume below hand-checkable.
//
//   x      0      1     2     3     4     5     6     7     8
//  topo  -10.0   1.0   5.0   2.0   6.0   7.0   8.0   9.0  10.0
//        OCEAN         rim   PIT   <----- ground rising to the east ----->
//
// The trench at x=3 (floor 2.0) is closed by the rim at x=2 (5.0), so it holds PIT_DEPTH = 3.0 m of
// standing water over H cells before it overflows west, over the rim, down x=1 and into the ocean.
const double TOPO_X[W]   = {-10.0, 1.0, 5.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0};
constexpr int    PIT_X     = 3;
constexpr double PIT_DEPTH = 3.0;

struct Fixture {
  Parameters                     params;
  ArrayPack                      arp;
  dh::DepressionHierarchy<float> deps;

  Fixture() {
    params.infiltration_on     = false;  // the branch WTM production runs use (infiltration_during_flow)
    params.cellsize_n_s_metres = 1000.0;

    arp.topo               = rd::Array2D<float>(W, H, 0.0f);
    arp.porosity           = rd::Array2D<float>(W, H, static_cast<float>(PORO));
    arp.wtd                = rd::Array2D<double>(W, H, 0.0);
    arp.runoff             = rd::Array2D<double>(W, H, 0.0);
    arp.infiltration_array = rd::Array2D<double>(W, H, 0.0);
    arp.label              = rd::Array2D<dh::dh_label_t>(W, H, dh::NO_DEP);
    arp.final_label        = rd::Array2D<dh::dh_label_t>(W, H, dh::NO_DEP);
    arp.flowdirs           = rd::Array2D<rd::flowdir_t>(W, H, rd::NO_FLOW);

    arp.cell_area           = std::vector<double>(H, AREA);
    arp.cellsize_e_w_metres = std::vector<double>(H, 1000.0);

    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        arp.topo(x, y) = static_cast<float>(TOPO_X[x]);
        if (x == 0)
          arp.label(x, y) = dh::OCEAN;
      }
  }

  // The depression hierarchy must be rebuilt whenever topo changes; topo is fixed here, so once.
  void build() {
    deps = dh::GetDepressionHierarchy<float, rd::Topology::D8>(
        arp.topo, arp.cell_area, arp.label, arp.final_label, arp.flowdirs);
  }

  double sum_volume() const {
    double v = 0.0;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) v += vol_per_area(arp.wtd(x, y)) * arp.cell_area[y];
    return v;
  }

  double sum_runoff() const {
    double r = 0.0;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) r += arp.runoff(x, y) * arp.cell_area[y];
    return r;
  }
};

}  // namespace

TEST_CASE("FSM ledger: net volume change == runoff consumed - ocean spill (NOT zero)") {
  Fixture f;
  f.build();

  // Hand FSM water two ways at once, exactly as a production step does: standing water left in the
  // table by the solve (wtd > 0) AND water already collected into the runoff array by the exfiltration
  // collector.
  f.arp.wtd(7, 2)    = 2.0;  // standing water high on the eastern slope
  f.arp.runoff(6, 1) = 3.0;  // already debited from storage, handed to FSM to route
  f.arp.runoff(5, 3) = 1.5;

  const double vol_pre    = f.sum_volume();
  const double runoff_pre = f.sum_runoff();
  const double ocean_pre  = f.arp.total_loss_to_ocean;

  dh::FillSpillMerge(f.params, f.deps, f.arp);

  const double vol_post    = f.sum_volume();
  const double runoff_post = f.sum_runoff();
  const double spill       = f.arp.total_loss_to_ocean - ocean_pre;

  // THE IDENTITY. Everything FSM did to the water table is accounted for by the runoff it consumed
  // and the water it sent to sea.
  CHECK((vol_post - vol_pre) == doctest::Approx(runoff_pre - runoff_post - spill).epsilon(1e-9));

  // With infiltration off, FSM consumes the runoff array completely (it zeroes each cell as it goes,
  // fill_spill_merge.hpp:351/358), so the identity reduces to (volume change) = (runoff in) - (spill).
  CHECK(runoff_post == doctest::Approx(0.0));

  // ...and the net change is NOT zero: FSM returned 4.5 m x AREA of collected runoff to the domain.
  // This is the line that keeps getting mis-read as a double-count. It is conservation, not a leak.
  CHECK(spill == doctest::Approx(0.0));  // far too little water to overflow the 3 m trench
  CHECK((vol_post - vol_pre) == doctest::Approx(4.5 * AREA).epsilon(1e-9));
  CHECK((vol_post - vol_pre) > 0.0);
}

TEST_CASE("FSM ledger: standing water and runoff in the SAME cell are summed, not doubled") {
  Fixture f;
  f.build();

  // One cell carries both: 2 m standing in the table and 3 m already in the runoff array. The
  // transfer at fill_spill_merge.hpp:341-343 is `runoff += wtd; wtd = 0`, so exactly 5 m x AREA must
  // reach the trench -- not 4 (wtd counted twice), not 6 (runoff counted twice), not 3 (wtd dropped).
  f.arp.wtd(7, 2)    = 2.0;
  f.arp.runoff(7, 2) = 3.0;

  dh::FillSpillMerge(f.params, f.deps, f.arp);

  // The source cell gave up all of its standing water.
  CHECK(f.arp.wtd(7, 2) <= 0.0);
  CHECK(f.arp.runoff(7, 2) == doctest::Approx(0.0));

  // The trench (H cells of area AREA, floor at 2.0, rim at 5.0) received 5 m x AREA and stands at a
  // uniform 5/H = 1.0 m -- comfortably below the 3 m rim, so nothing overflows.
  const double expect_depth = 5.0 / H;
  for (int y = 0; y < H; y++)
    CHECK(f.arp.wtd(PIT_X, y) == doctest::Approx(expect_depth).epsilon(1e-6));
  CHECK(expect_depth < PIT_DEPTH);
  CHECK(f.arp.total_loss_to_ocean == doctest::Approx(0.0));
}

TEST_CASE("FSM ledger: below-surface space is filled at porosity, cell by cell (not as a level)") {
  Fixture f;
  // Dry out the trench so the arriving water must first fill pore space. 4 m of unsaturated column
  // at porosity PORO holds 4 * PORO = 1.0 m x AREA of water per cell, so the H-cell trench can take
  // 5.0 m x AREA before any standing water appears.
  const double DRY = -4.0;
  for (int y = 0; y < H; y++) f.arp.wtd(PIT_X, y) = DRY;
  f.build();

  const double supplied = 2.5;  // half the trench's 5.0 m x AREA of pore space
  f.arp.runoff(7, 2)    = supplied;

  const double vol_pre = f.sum_volume();
  dh::FillSpillMerge(f.params, f.deps, f.arp);

  // (1) CONSERVATION -- the derived contract. Volume is measured at porosity below the surface, and
  // the total rise across the trench accounts for exactly the water supplied.
  CHECK((f.sum_volume() - vol_pre) == doctest::Approx(supplied * AREA).epsilon(1e-9));

  // (2) No standing water was created: less water arrived than the available pore space.
  double rise_volume = 0.0;
  int    n_saturated = 0, n_untouched = 0;
  for (int y = 0; y < H; y++) {
    const double w = f.arp.wtd(PIT_X, y);
    CHECK(w <= 0.0);
    rise_volume += (w - DRY) * PORO;
    if (w == doctest::Approx(0.0)) n_saturated++;
    if (w == doctest::Approx(DRY)) n_untouched++;
  }
  CHECK(rise_volume == doctest::Approx(supplied).epsilon(1e-9));

  // (3) BEHAVIOUR, recorded rather than endorsed: FSM does NOT raise a uniform water level across
  // the depression's pore space. It saturates cells one at a time in the hierarchy's fill order, so
  // with half the pore space worth of water some trench cells come all the way up to the land
  // surface while others are left completely dry. All H cells here sit at the same elevation (2.0),
  // so which ones fill is set by traversal order, not by topography. If this ever becomes a
  // level-raise, this assertion is the tripwire -- and see the note in the header comment.
  CHECK(n_saturated > 0);
  CHECK(n_untouched > 0);
}

TEST_CASE("FSM ledger: runoff on an ocean cell leaves the domain and the cell holds nothing") {
  Fixture f;
  f.build();

  f.arp.runoff(0, 2) = 7.0;  // water routed onto an OCEAN cell
  const double ocean_pre = f.arp.total_loss_to_ocean;

  dh::FillSpillMerge(f.params, f.deps, f.arp);

  // fill_spill_merge.hpp:348-351: the volume is booked to total_loss_to_ocean and the cell is zeroed.
  CHECK((f.arp.total_loss_to_ocean - ocean_pre) == doctest::Approx(7.0 * AREA).epsilon(1e-9));
  CHECK(f.arp.runoff(0, 2) == doctest::Approx(0.0));
}

TEST_CASE("FSM ledger: overflow to the ocean shows up as spill, and the identity still closes") {
  Fixture f;
  f.build();

  // Far more than the trench can hold (H cells x 3 m depth = 15 m x AREA), so the excess must
  // overflow the rim and reach the ocean.
  const double supplied = 40.0;
  f.arp.runoff(7, 2)    = supplied;

  const double vol_pre   = f.sum_volume();
  const double ocean_pre = f.arp.total_loss_to_ocean;

  dh::FillSpillMerge(f.params, f.deps, f.arp);

  const double spill = f.arp.total_loss_to_ocean - ocean_pre;
  CHECK(spill > 0.0);  // it really did overflow
  // The identity holds with a non-zero spill term: volume change = runoff in - spill.
  CHECK((f.sum_volume() - vol_pre) == doctest::Approx(supplied * AREA - spill).epsilon(1e-9));
  CHECK(f.sum_runoff() == doctest::Approx(0.0));
}
