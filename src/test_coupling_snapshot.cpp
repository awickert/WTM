// Unit tests for the #112 coupling-iteration rollback (src/coupling_snapshot.hpp).
//
// WHAT THIS DEFENDS. Iterating the FSM<->recharge coupling means re-solving a step, which means
// undoing it. benchmark/FSM_COUPLING_ITERATION.md names the risk in one line -- "a missed item is a
// SILENT MASS ERROR" -- and a post-commit rollback is exactly where defects #13, #14, #15, #17 and
// #41 lived. A rollback that restores eight of nine accumulators loses water without saying so.
//
// WHY THE TEST IS NOT VACUOUS. The field list below is written from src/ArrayPack.hpp, NOT from
// CouplingSnapshot. If a field is absent from capture()/restore(), the round-trip leaves the SECOND
// perturbation in place and the assertion fires. A test that enumerated the snapshot's own members
// would pass with a field missing from both -- which is the trap this repo has hit six ways
// (tests/ASSERTION_HEALTH.md).
//
// THE TENTH ACCUMULATOR is the failure this cannot catch on its own: a field added to ArrayPack and
// to neither the snapshot nor this list. tests/lint_norms.sh pins that separately by comparing the
// `double total_*` sets in the two headers.
//
// Pure scalar math (no MPI); compiled into test_dmda.x. doctest's main lives in test_dmda_gather.cpp.
#include "doctest.h"
#include "coupling_snapshot.hpp"

TEST_CASE("coupling snapshot: every scalar round-trips EXACTLY") {
  Parameters params;
  ArrayPack  arp;
  AppCtx     uc;

  // FIRST perturbation -- the state we must get back. Distinct values so a cross-wired field
  // (restoring evap into surface_removed, say) is caught as well as a missing one.
  arp.total_recharge_direct    = 1.5;
  arp.total_runoff_to_surface  = 2.5;
  arp.total_loss_to_ocean      = 3.5;
  arp.total_loss_to_ocean_gw   = 4.5;
  arp.total_ocean_outflow_gw   = 5.5;
  arp.total_boundary_inflow_gw = 6.5;
  arp.total_surface_removed    = 7.5;
  arp.total_evap_removed       = 8.5;
  arp.total_solver_recharge    = 9.5;
  arp.total_storage_change     = 9.75;
  params.elapsed_time_s        = 10.5;
  uc.step.dt                   = 11.5;
  uc.step.elapsed_from         = 12.5;
  uc.step.elapsed_to           = 13.5;
  uc.deltat                    = 14.5;

  const auto snap = wtm::CouplingSnapshot::capture(params, arp, uc);

  // SECOND perturbation -- stands in for "the step ran". Every field moves, so a field the
  // snapshot does not carry is left holding this value instead of the first one.
  arp.total_recharge_direct    = -1.0;
  arp.total_runoff_to_surface  = -2.0;
  arp.total_loss_to_ocean      = -3.0;
  arp.total_loss_to_ocean_gw   = -4.0;
  arp.total_ocean_outflow_gw   = -5.0;
  arp.total_boundary_inflow_gw = -6.0;
  arp.total_surface_removed    = -7.0;
  arp.total_evap_removed       = -8.0;
  arp.total_solver_recharge    = -9.0;
  arp.total_storage_change     = -9.25;
  params.elapsed_time_s        = -10.0;
  uc.step.dt                   = -11.0;
  uc.step.elapsed_from         = -12.0;
  uc.step.elapsed_to           = -13.0;
  uc.deltat                    = -14.0;

  snap.restore(params, arp, uc);

  // EXACT equality, not doctest::Approx. These are copies of doubles; a tolerance here would hide
  // precisely the kind of partial restore this test exists to catch.
  CHECK(arp.total_recharge_direct    == 1.5);
  CHECK(arp.total_runoff_to_surface  == 2.5);
  CHECK(arp.total_loss_to_ocean      == 3.5);
  CHECK(arp.total_loss_to_ocean_gw   == 4.5);
  CHECK(arp.total_ocean_outflow_gw   == 5.5);
  CHECK(arp.total_boundary_inflow_gw == 6.5);
  CHECK(arp.total_surface_removed    == 7.5);
  CHECK(arp.total_evap_removed       == 8.5);
  CHECK(arp.total_solver_recharge    == 9.5);
  CHECK(arp.total_storage_change     == 9.75);
  CHECK(params.elapsed_time_s        == 10.5);
  CHECK(uc.step.dt                   == 11.5);
  CHECK(uc.step.elapsed_from         == 12.5);
  CHECK(uc.step.elapsed_to           == 13.5);
  CHECK(uc.deltat                    == 14.5);
}

TEST_CASE("coupling snapshot: capture does not mutate the model") {
  Parameters params;
  ArrayPack  arp;
  AppCtx     uc;
  arp.total_solver_recharge = 42.0;
  params.elapsed_time_s     = 7.0;

  const auto snap = wtm::CouplingSnapshot::capture(params, arp, uc);
  (void)snap;

  CHECK(arp.total_solver_recharge == 42.0);
  CHECK(params.elapsed_time_s     == 7.0);
}

TEST_CASE("coupling snapshot: the accumulator count is pinned at ten") {
  // Not a tautology: tests/lint_norms.sh checks this constant against the number of `double total_*`
  // declarations in ArrayPack.hpp, so raising one without the other fails there. The constant is
  // here so the two places that must agree are both in the build.
  CHECK(wtm::CouplingSnapshot::kAccumulators == 10);
}
