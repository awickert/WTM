#pragma once
// #112 -- SNAPSHOT AND RESTORE FOR THE FSM<->RECHARGE COUPLING ITERATION.
//
// Iterating the coupling within a step means re-solving that step, which means UNDOING it first.
// This is structure B of benchmark/FSM_COUPLING_ITERATION.md: keep FillSpillMerge where it is and
// snapshot/restore at the WTM.cpp level, rather than pulling a rank-0 serial operation inside the
// distributed solver.
//
// WHY THIS FILE EXISTS SEPARATELY FROM WTM.cpp: so a unit test can reach it. The design names the
// risk exactly -- "the rollback surface is ours to get right, and a missed item is a SILENT MASS
// ERROR" -- and a post-commit rollback is precisely where defects #13, #14, #15, #17 and #41 lived.
//
// SCOPE OF THIS HEADER, stated so it is not mistaken for the whole rollback: the SCALARS only --
// the TEN arp.total_* accumulators, params.elapsed_time_s, the accepted-step record and deltat.
// The Vec state (starting_wtd, lake_stage, rech_vec) and the BDF2/TR-BDF2 history are a separate
// commit; they need PETSc objects and duplicate-and-copy rather than assignment.
#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ArrayPack.hpp"
#include "CreateSNES.hpp"
#include "parameters.hpp"

namespace wtm {

// THE TEN ACCUMULATORS, enumerated from src/ArrayPack.hpp rather than assumed -- the design said
// NINE and was wrong; see total_storage_change below. Adding a tenth
// without adding it here is the silent-mass-error case; tests/../test_coupling_snapshot.cpp pins
// the count so that addition fails loudly instead.
struct CouplingSnapshot {
  static constexpr std::size_t kAccumulators = 10;

  double total_recharge_direct    = 0;
  double total_runoff_to_surface  = 0;
  double total_loss_to_ocean      = 0;
  double total_loss_to_ocean_gw   = 0;
  double total_ocean_outflow_gw   = 0;
  double total_boundary_inflow_gw = 0;
  double total_surface_removed    = 0;
  double total_evap_removed       = 0;
  double total_solver_recharge    = 0;
  // THE TENTH, which the design's "enumerated from source, not assumed" list missed. It reads
  // like a derived quantity -- ArrayPack documents the identity it should satisfy -- but it is
  // ACCUMULATED independently (transient_groundwater.cpp:883 `+=`, :2442 `-=`), so restoring the
  // other nine does NOT restore it. Found by tests/lint_norms.sh on its first run.
  double total_storage_change     = 0;

  // THE PARAMETERS FIELDS THE STEP BODY MUTATES. There are exactly two, and that is enumerated
  // rather than believed: `params.<field> =/+=/++` appears in the solve (transient_groundwater.cpp)
  // and in couple_surface_and_recharge at precisely these two sites. Every other params mutation in
  // the tree is per-CYCLE (cycles_done, last_cycle_*), per-REPORT (infiltration_change, inside
  // PrintValues), initialisation, or one of the per-step COUNTERS that Amendment 6 decided are to
  // describe the accepted pass only. tests/lint_norms.sh pins this set.
  static constexpr std::size_t kParamsFields = 2;
  double elapsed_time_s = 0;   // params; advanced by transient_groundwater (:2475)
  // THE ELEVENTH SCALAR, and it was missed the same way the tenth was -- by not being an
  // `arp.total_*`, so the lint that caught the tenth could not see it. The coupling sets it to
  // elapsed_time_s (WTM.cpp:383) as the watermark for "runoff booked up to here", and the NEXT
  // coupling scales the runoff handoff by (elapsed_time_s - runoff_booked_upto_s) / params.deltat.
  // Leave it unrestored and a second pass computes interval = 0, falls through the `interval > 0`
  // guard to a scale of exactly 1.0, and hands FillSpillMerge a silently different amount of water.
  // Found by reading the coupling for a different reason, which is the argument for the new lint.
  double runoff_booked_upto_s = 0;
  double step_dt        = 0;   // user_context.step.*, the accepted-step record
  double step_from      = 0;
  double step_to        = 0;
  double deltat         = 0;   // user_context.deltat, the NEXT proposed step size

  // Capture the pre-solve state. Const in, value out: a snapshot that could mutate the model
  // would defeat its own purpose.
  static CouplingSnapshot capture(const Parameters& params, const ArrayPack& arp, const AppCtx& uc) {
    CouplingSnapshot s;
    s.total_recharge_direct    = arp.total_recharge_direct;
    s.total_runoff_to_surface  = arp.total_runoff_to_surface;
    s.total_loss_to_ocean      = arp.total_loss_to_ocean;
    s.total_loss_to_ocean_gw   = arp.total_loss_to_ocean_gw;
    s.total_ocean_outflow_gw   = arp.total_ocean_outflow_gw;
    s.total_boundary_inflow_gw = arp.total_boundary_inflow_gw;
    s.total_surface_removed    = arp.total_surface_removed;
    s.total_evap_removed       = arp.total_evap_removed;
    s.total_solver_recharge    = arp.total_solver_recharge;
    s.total_storage_change     = arp.total_storage_change;
    s.elapsed_time_s           = params.elapsed_time_s;
    s.runoff_booked_upto_s     = params.runoff_booked_upto_s;
    s.step_dt                  = uc.step.dt;
    s.step_from                = uc.step.elapsed_from;
    s.step_to                  = uc.step.elapsed_to;
    s.deltat                   = uc.deltat;
    return s;
  }

  // Put it all back, EXACTLY. These are plain doubles, so restoration is bit-for-bit by
  // construction -- no tolerance is involved and none should be introduced.
  void restore(Parameters& params, ArrayPack& arp, AppCtx& uc) const {
    arp.total_recharge_direct    = total_recharge_direct;
    arp.total_runoff_to_surface  = total_runoff_to_surface;
    arp.total_loss_to_ocean      = total_loss_to_ocean;
    arp.total_loss_to_ocean_gw   = total_loss_to_ocean_gw;
    arp.total_ocean_outflow_gw   = total_ocean_outflow_gw;
    arp.total_boundary_inflow_gw = total_boundary_inflow_gw;
    arp.total_surface_removed    = total_surface_removed;
    arp.total_evap_removed       = total_evap_removed;
    arp.total_solver_recharge    = total_solver_recharge;
    arp.total_storage_change     = total_storage_change;
    params.elapsed_time_s        = elapsed_time_s;
    params.runoff_booked_upto_s  = runoff_booked_upto_s;
    uc.step.dt                   = step_dt;
    uc.step.elapsed_from         = step_from;
    uc.step.elapsed_to           = step_to;
    uc.deltat                    = deltat;
  }
};

}  // namespace wtm

// ---------------------------------------------------------------------------------------------
// THE VEC HALF OF THE ROLLBACK -- MEASUREMENT FIRST, not a guess.
//
// The scalars above could be enumerated MECHANICALLY: `double total_*` is a syntactic signature, so
// tests/lint_norms.sh pins the set and caught a tenth accumulator the design had missed. THE VECS
// HAVE NO SUCH SIGNATURE. They are written through dmdapack's array views
// (`dmdapack.rech_vec[j][i] = ...`, transient_groundwater.cpp:1291), not through PETSc calls, so no
// grep can list which ones a step mutates.
//
// The design says "3 arrays + BDF2 history". AppCtx holds 39. Since the design's scalar list was
// already wrong by one, its Vec list is not something to build a silent-mass-error surface on.
//
// So: capture ALL of them, run a step, and ask which actually CHANGED. The answer is a measured
// list, prunable with evidence -- the same discipline as sweeping a tolerance instead of assuming
// it. Over-capturing is safe here (restoring an unchanged Vec is a no-op); under-capturing loses
// water silently, which is the asymmetry that decides the default.
#define WTM_APPCTX_VEC_LIST(X) \
  X(ar_best_x) \
  X(b) \
  X(cellsize_EW_squared) \
  X(evap_vec) \
  X(exfiltration_vec) \
  X(fdepth_local) \
  X(fdepth_vec) \
  X(fsm_delta_vec) \
  X(geom_ew_vec) \
  X(geom_n_vec) \
  X(geom_s_vec) \
  X(ksat_local) \
  X(ksat_vec) \
  X(lake_stage) \
  X(mask) \
  X(mask_local) \
  X(open_water_evap_vec) \
  X(picard_r) \
  X(porosity_vec) \
  X(precip_vec) \
  X(rech_source) \
  X(rech_vec) \
  X(runoff_dist_vec) \
  X(runoff_ratio_vec) \
  X(sink_removed_dist_vec) \
  X(starting_wtd) \
  X(starting_wtd_local) \
  X(starting_wtd_prev) \
  X(T_local) \
  X(topo_local) \
  X(topo_vec) \
  X(tr_exfil_stage1) \
  X(tr_expl) \
  X(tr_fwork) \
  X(tr_head_old) \
  X(tr_ygamma) \
  X(vol_prev_x) \
  X(wtd_global) \
  X(x)

namespace wtm {

// Capture-and-compare only. NO restore: this exists to MEASURE which Vecs a step touches, so the
// rollback can carry the measured set rather than a judged one. Restore lands once the list is known.
struct CouplingVecProbe {
  std::vector<std::pair<const char*, Vec>> saved;

  void capture(const AppCtx& uc) {
    destroy();
#define X(name) if (uc.name) { Vec d; VecDuplicate(uc.name, &d); VecCopy(uc.name, d); saved.emplace_back(#name, d); }
    WTM_APPCTX_VEC_LIST(X)
#undef X
  }

  // Names of the Vecs that differ from the capture, MATCHED BY NAME -- never by position.
  //
  // POSITION-MATCHING IS A SEGV, and it is what the first version of this did. A STEP CREATES VECS:
  // tr_head_old, vol_prev_x, tr_exfil_stage1 and tr_fwork are allocated lazily on first use
  // (transient_groundwater.cpp:732, :1004, :1876, :1877), so a capture taken before the solve held
  // 33 entries while 37 were live at compare time, and saved[k] walked off the end. Measured on
  // fsm_cascade: "captured=33 live=37 appeared: tr_exfil_stage1 tr_fwork tr_head_old vol_prev_x".
  //
  // The lifetime cases are reported SEPARATELY from the value case because the rollback must do
  // something different with each. A Vec that did not exist before the step has no pre-image to
  // copy back; undoing its creation means DESTROYING it. Folding that into "changed" would name
  // the right Vec and imply the wrong repair.
  //
  // Exact: a step that touches a Vec and puts it back bit-for-bit is correctly reported as
  // untouched, which is what the rollback cares about.
  std::vector<std::string> changed(const AppCtx& uc) const {
    std::vector<std::string> out;
    auto captured_as = [&](const char* nm) -> Vec {
      for (const auto& p : saved)
        if (std::string(p.first) == nm) return p.second;
      return nullptr;
    };
#define X(name) { Vec pre = captured_as(#name); \
      if (uc.name && !pre) out.emplace_back(std::string(#name) + " [CREATED]"); \
      else if (!uc.name && pre) out.emplace_back(std::string(#name) + " [DESTROYED]"); \
      else if (uc.name && pre) { PetscBool eq = PETSC_FALSE; VecEqual(uc.name, pre, &eq); \
                                 if (!eq) out.emplace_back(#name); } }
    WTM_APPCTX_VEC_LIST(X)
#undef X
    return out;
  }

  void destroy() {
    for (auto& p : saved) VecDestroy(&p.second);
    saved.clear();
  }
  ~CouplingVecProbe() { destroy(); }
};

}  // namespace wtm


// ---------------------------------------------------------------------------------------------
// THE ROLLBACK ITSELF: the MEASURED SET, copied -- plus a RUNTIME CHECK on everything else.
//
// Andy chose this shape (2026-09-23) over "capture the measured set" and "capture all 39". The
// measured set alone is smaller but breaks SILENTLY when a new integrator appears, and unlike the
// scalars there is no syntactic signature a lint could pin. Capturing all 39 cannot go stale but
// costs ~120 MB at Esquibel's 384,703 cells, linear in the grid. This carries the measured set and
// lets the RUN verify the enumeration, so a stale list fails loudly instead of losing water.
//
// THE CHECK USES FINGERPRINTS, NOT COPIES -- my choice, and it is the only way the shape makes
// sense: holding copies of the other 21 to compare against would cost exactly the memory the choice
// was made to avoid. Three norms per Vec (1, 2, infinity) is 3 doubles instead of a grid. WHAT THAT
// GIVES UP, stated rather than glossed: a mutation that preserves all three norms is invisible to
// it -- a permutation of entries between ranks, or a sign flip on a symmetric field. It is a
// tripwire for "this Vec is not inert after all", not a proof of equality.
//
// THE SET IS THE 18 MEASURED IN AMENDMENT 5, over seven arms on one fixture. It is a LOWER BOUND
// (see `ar_best_x` there: restart was enabled and still never fired), which is exactly why the
// check exists.
//
// WHAT THIS DOES NOT COVER, so the check is not read as broader than it is: rank-0 ArrayPack
// ARRAYS. The coupling writes five -- arp.wtd, arp.runoff, arp.rech, arp.wtd_mid and
// arp.runoff_nominal -- and none is copied or fingerprinted here. Each is believed safe because it
// is REDERIVED from state that is restored (wtd re-gathered from starting_wtd, runoff zeroed and
// re-armed every coupling, rech recomputed from precip/evap), but that is an argument rather than a
// measurement, and the evidence is only indirect: iterations 1/2/3 agree bit-for-bit and the exact
// budget holds at ~1e-11 relative. OPEN GAP, recorded in AMENDMENT 7. The fix is this same
// fingerprint shape applied to those five, and it is cheap because they are serial rank-0 arrays.
//
// FOUR OF THE 18 ARE PROVABLY SCRATCH and are captured anyway. tr_head_old is refilled from
// starting_wtd + topo over the whole owned range on every call (transient_groundwater.cpp:732-740);
// vol_prev_x is reset at it == 0 of each solve (:1004-1005); tr_fwork and tr_exfil_stage1 are
// written by SNESComputeFunction and VecCopy before they are read (:1876-1879). Their pre-step
// contents cannot matter. Capturing them is 4 grid vectors of 18 -- and the asymmetry that decides
// every question in this file says a wrong prune loses water silently while a wasted copy costs
// memory and says so. Prune them only with a measurement, never with this paragraph.
#define WTM_COUPLING_ROLLBACK_LIST(X) \
  X(exfiltration_vec) \
  X(fsm_delta_vec) \
  X(lake_stage) \
  X(picard_r) \
  X(rech_vec) \
  X(sink_removed_dist_vec) \
  X(starting_wtd) \
  X(starting_wtd_local) \
  X(starting_wtd_prev) \
  X(T_local) \
  X(tr_exfil_stage1) \
  X(tr_expl) \
  X(tr_fwork) \
  X(tr_head_old) \
  X(tr_ygamma) \
  X(vol_prev_x) \
  X(wtd_global) \
  X(x)

namespace wtm {

struct CouplingVecSnapshot {
  static constexpr std::size_t kRollbackVecs = 18;

  std::vector<std::pair<const char*, Vec>> saved;   // full copies: the measured set
  // name -> {1-norm, 2-norm, inf-norm} for every Vec OUTSIDE the measured set.
  std::vector<std::pair<const char*, std::array<double, 3>>> outside;

  static std::array<double, 3> fingerprint(Vec v) {
    std::array<double, 3> f{};
    VecNorm(v, NORM_1, &f[0]);
    VecNorm(v, NORM_2, &f[1]);
    VecNorm(v, NORM_INFINITY, &f[2]);
    return f;
  }

  void capture(const AppCtx& uc) {
    destroy();
#define X(name) if (uc.name) { Vec d; VecDuplicate(uc.name, &d); VecCopy(uc.name, d); saved.emplace_back(#name, d); }
    WTM_COUPLING_ROLLBACK_LIST(X)
#undef X
#define X(name) if (uc.name && !in_rollback_set(#name)) outside.emplace_back(#name, fingerprint(uc.name));
    WTM_APPCTX_VEC_LIST(X)
#undef X
  }

  // Put the measured set back, bit-for-bit. Matched by NAME, for the reason changed() is: a step
  // CREATES Vecs, so positions do not line up across a step.
  void restore(AppCtx& uc) const { restore_except(uc, nullptr); }

  // RESTORE EVERYTHING EXCEPT THE ITERATION VARIABLE -- what a coupling pass actually needs.
  //
  // A rollback that put fsm_delta_vec back would restore the very quantity the iteration is solving
  // for, and the loop could never move: pass k+1 would re-solve against pass k's INPUT rather than
  // its OUTPUT. So the one carrier FillSpillMerge writes for the next step is kept, and everything
  // else is undone. That is the fixed-point iteration written out:
  //
  //     Phi(w) = G(w_n, F(w))   -- restore the state, keep F(w).
  //
  // rech_vec is deliberately NOT kept, and the opposite is tempting. Its post-step value is the
  // recharge for step n+1, computed from the POST-step water table; a re-solve of step n must use
  // step n's own recharge, which the previous step's coupling set. The final accepted pass
  // recomputes the next step's value regardless, so restoring costs nothing -- while keeping it
  // would advance the forcing by one step inside the iteration. See AMENDMENT 6.
  void restore_keeping_fsm_delta(AppCtx& uc) const { restore_except(uc, "fsm_delta_vec"); }

  void restore_except(AppCtx& uc, const char* keep) const {
#define X(name) if (uc.name && !(keep && std::string(keep) == #name)) { \
      for (const auto& p : saved) if (std::string(p.first) == #name) { VecCopy(p.second, uc.name); break; } }
    WTM_COUPLING_ROLLBACK_LIST(X)
#undef X
  }

  // THE RUNTIME CHECK. Empty means the enumeration still holds for this run. Anything here is a
  // rollback that would have been incomplete -- report it loudly; do not filter it.
  std::vector<std::string> unrestorable(const AppCtx& uc) const {
    std::vector<std::string> out;
    // (a) a Vec outside the measured set whose fingerprint moved: the set is too small for this run.
#define X(name) if (uc.name && !in_rollback_set(#name)) { \
      for (const auto& p : outside) if (std::string(p.first) == #name) { \
        const auto now = fingerprint(uc.name); \
        if (now != p.second) out.emplace_back(std::string(#name) + " moved but is OUTSIDE the rollback set"); \
        break; } }
    WTM_APPCTX_VEC_LIST(X)
#undef X
    // (b) a Vec in the set that did not exist at capture. Today all four such Vecs are scratch
    // (see the header note), so this is a NOTICE rather than a defect -- but a future one might be
    // history, and then restoring nothing into it would be the silent case. So it is reported.
#define X(name) if (uc.name) { bool had = false; \
      for (const auto& p : saved) if (std::string(p.first) == #name) { had = true; break; } \
      if (!had) out.emplace_back(std::string(#name) + " was CREATED during the step; nothing to restore"); }
    WTM_COUPLING_ROLLBACK_LIST(X)
#undef X
    return out;
  }

  static bool in_rollback_set(const char* nm) {
#define X(name) if (std::string(nm) == #name) return true;
    WTM_COUPLING_ROLLBACK_LIST(X)
#undef X
    return false;
  }

  void destroy() {
    for (auto& p : saved) VecDestroy(&p.second);
    saved.clear();
    outside.clear();
  }
  ~CouplingVecSnapshot() { destroy(); }
};

}  // namespace wtm
