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
  params.runoff_booked_upto_s  = 10.25;   // the ELEVENTH scalar: the runoff-handoff watermark
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
  params.runoff_booked_upto_s  = -10.25;
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
  CHECK(params.runoff_booked_upto_s  == 10.25);
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

TEST_CASE("coupling snapshot: the Parameters-field count is pinned at two") {
  // The step body mutates exactly two params fields -- elapsed_time_s in the solve and
  // runoff_booked_upto_s in the coupling. tests/lint_norms.sh derives that set from the source and
  // compares it against this header, so a third appearing without a snapshot entry fails there.
  CHECK(wtm::CouplingSnapshot::kParamsFields == 2);
}

TEST_CASE("coupling snapshot: the accumulator count is pinned at ten") {
  // Not a tautology: tests/lint_norms.sh checks this constant against the number of `double total_*`
  // declarations in ArrayPack.hpp, so raising one without the other fails there. The constant is
  // here so the two places that must agree are both in the build.
  CHECK(wtm::CouplingSnapshot::kAccumulators == 10);
}

// ---------------------------------------------------------------------------------------------
// THE PROBE'S OTHER FAILURE MODE, and it took down a real run before it was understood: A STEP
// CREATES VECS. tr_head_old, vol_prev_x, tr_exfil_stage1 and tr_fwork are allocated lazily on
// first use (transient_groundwater.cpp:732, :1004, :1876, :1877), so a capture taken before the
// solve holds FEWER entries than are live afterwards -- measured on fsm_cascade as
// "captured=33 live=37". The original changed() walked saved[k] by POSITION and read off the end:
// SEGV, which the suite reported only as "RUN FAILED", exit 2. It was mislabelled for a day as the
// probe perturbing the run; it was never a perturbation, it was a crash.
//
// These three cases pin the name-matched behaviour. WHAT EACH ONE IS WORTH, checked by reverting
// changed() to position-matching and running them (not assumed -- the first guess was wrong):
//   CREATED   -- SIGSEGV. The crash case, and the one that took down fsm_cascade.
//   DESTROYED -- does NOT crash: it reports "x" instead of "tr_fwork [DESTROYED]", comparing x
//                against tr_fwork's saved copy. A SILENT WRONG ANSWER, which is the worse failure:
//                it would have named an innocent Vec for the rollback and omitted a guilty one.
//   VALUE     -- PASSES under the old implementation too. It is a CHARACTERISATION test, not a
//                regression test, and is kept as the ordinary case the other two must not swamp.
//                Saying so here rather than letting it pose as a third guard.
TEST_CASE("coupling vec probe: a Vec CREATED during the step is named, not walked past") {
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.x);
  VecSet(uc.x, 1.0);

  wtm::CouplingVecProbe probe;
  probe.capture(uc);   // one entry: x

  // "the step ran", and it created a history vector the capture never saw.
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.tr_head_old);
  VecSet(uc.tr_head_old, 7.0);

  const auto ch = probe.changed(uc);
  REQUIRE(ch.size() == 1);
  CHECK(ch[0] == "tr_head_old [CREATED]");

  VecDestroy(&uc.x);
  VecDestroy(&uc.tr_head_old);
}

TEST_CASE("coupling vec probe: a Vec DESTROYED during the step is named separately") {
  // The mirror case. It matters because the rollback's repair differs: a created Vec is undone by
  // DESTROYING it, a destroyed one by recreating it, and neither is a VecCopy.
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.x);
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.tr_fwork);
  VecSet(uc.x, 1.0);
  VecSet(uc.tr_fwork, 2.0);

  wtm::CouplingVecProbe probe;
  probe.capture(uc);   // two entries
  VecDestroy(&uc.tr_fwork);
  uc.tr_fwork = nullptr;

  const auto ch = probe.changed(uc);
  REQUIRE(ch.size() == 1);
  CHECK(ch[0] == "tr_fwork [DESTROYED]");

  VecDestroy(&uc.x);
}

TEST_CASE("coupling vec probe: a changed VALUE is named bare, and an untouched one is silent") {
  // The ordinary case, and the one that must not be swamped by the lifetime cases. Also pins the
  // exactness the rollback depends on: a Vec written and restored bit-for-bit reads as untouched.
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.x);
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.lake_stage);
  VecSet(uc.x, 1.0);
  VecSet(uc.lake_stage, 5.0);

  wtm::CouplingVecProbe probe;
  probe.capture(uc);

  VecSet(uc.lake_stage, 6.0);          // moved
  VecSet(uc.x, 2.0); VecSet(uc.x, 1.0);  // written, then put back exactly

  const auto ch = probe.changed(uc);
  REQUIRE(ch.size() == 1);
  CHECK(ch[0] == "lake_stage");

  VecDestroy(&uc.x);
  VecDestroy(&uc.lake_stage);
}

// ---------------------------------------------------------------------------------------------
// THE VEC ROLLBACK (CouplingVecSnapshot): the measured set is COPIED, everything else is
// FINGERPRINTED and checked. These pin both halves, and the second one pins the thing that makes
// the whole shape worth having -- an unmeasured Vec moving is REPORTED rather than lost.
TEST_CASE("coupling vec snapshot: a Vec in the measured set round-trips EXACTLY") {
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.lake_stage);
  VecSetValue(uc.lake_stage, 0, 3.25, INSERT_VALUES);   // distinct entries, so a zeroing "restore"
  VecSetValue(uc.lake_stage, 1, -7.5, INSERT_VALUES);   // cannot pass by accident
  VecAssemblyBegin(uc.lake_stage); VecAssemblyEnd(uc.lake_stage);

  wtm::CouplingVecSnapshot snap;
  snap.capture(uc);
  VecSet(uc.lake_stage, 99.0);            // "the step ran"
  snap.restore(uc);

  const PetscScalar* a;
  VecGetArrayRead(uc.lake_stage, &a);
  CHECK(a[0] == 3.25);
  CHECK(a[1] == -7.5);
  VecRestoreArrayRead(uc.lake_stage, &a);
  VecDestroy(&uc.lake_stage);
}

TEST_CASE("coupling vec snapshot: a Vec OUTSIDE the measured set that moves is REPORTED") {
  // The reason this shape was chosen over capturing the measured set alone. topo_vec changed on no
  // arm of the AMENDMENT 5 sweep, so it is fingerprinted, not copied -- and if some configuration
  // does move it, the run must SAY SO instead of silently failing to roll it back.
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.lake_stage);   // in the set
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.topo_vec);     // outside it
  VecSet(uc.lake_stage, 1.0);
  VecSet(uc.topo_vec, 2.0);

  wtm::CouplingVecSnapshot snap;
  snap.capture(uc);
  REQUIRE(snap.outside.size() == 1);       // NOT VACUOUS: there is something to check
  CHECK(snap.unrestorable(uc).empty());    // nothing moved yet

  VecSet(uc.topo_vec, 2.5);                // a Vec nobody measured moves
  const auto bad = snap.unrestorable(uc);
  REQUIRE(bad.size() == 1);
  CHECK(bad[0] == "topo_vec moved but is OUTSIDE the rollback set");

  VecDestroy(&uc.lake_stage);
  VecDestroy(&uc.topo_vec);
}

TEST_CASE("coupling vec snapshot: a set member created DURING the step is reported, not skipped") {
  // All four Vecs a step creates today are scratch (transient_groundwater.cpp:732, :1004, :1876,
  // :1877), so this is a notice rather than a defect. It is reported because a FUTURE lazily
  // created Vec might be history, and then restoring nothing into it is the silent case.
  AppCtx uc;
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.lake_stage);
  VecSet(uc.lake_stage, 1.0);

  wtm::CouplingVecSnapshot snap;
  snap.capture(uc);
  VecCreateSeq(PETSC_COMM_SELF, 4, &uc.tr_head_old);   // lazily allocated by the solve
  VecSet(uc.tr_head_old, 4.0);

  const auto bad = snap.unrestorable(uc);
  REQUIRE(bad.size() == 1);
  CHECK(bad[0] == "tr_head_old was CREATED during the step; nothing to restore");

  VecDestroy(&uc.lake_stage);
  VecDestroy(&uc.tr_head_old);
}

TEST_CASE("coupling vec snapshot: the rollback list is a strict subset of the AppCtx Vec list") {
  // Guards drift this file cannot otherwise see. A rollback name that left AppCtx would not
  // compile; a rollback list that quietly stopped matching the MEASURED 18 would. Both counts are
  // asserted, so neither can move alone, and the complement is pinned so the check cannot become
  // vacuous by the set swallowing everything.
  std::size_t in_appctx = 0, in_rollback = 0, members_found = 0;
#define X(name) ++in_appctx; if (wtm::CouplingVecSnapshot::in_rollback_set(#name)) ++members_found;
  WTM_APPCTX_VEC_LIST(X)
#undef X
#define X(name) ++in_rollback;
  WTM_COUPLING_ROLLBACK_LIST(X)
#undef X
  CHECK(in_appctx == 39);
  CHECK(in_rollback == wtm::CouplingVecSnapshot::kRollbackVecs);
  CHECK(in_rollback == 18);
  CHECK(members_found == in_rollback);   // every rollback name IS an AppCtx Vec
  CHECK(in_appctx - in_rollback == 21);  // and 21 are left for the fingerprint check
}
