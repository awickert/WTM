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
// the nine arp.total_* accumulators, params.elapsed_time_s, the accepted-step record and deltat.
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

  double elapsed_time_s = 0;   // params; advanced by transient_groundwater
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

  // Names of the Vecs that differ from the capture. Exact: a step that touches a Vec and puts it
  // back bit-for-bit is correctly reported as untouched, which is what the rollback cares about.
  std::vector<std::string> changed(const AppCtx& uc) const {
    std::vector<std::string> out;
    std::size_t k = 0;
#define X(name) if (uc.name) { PetscBool eq = PETSC_FALSE; VecEqual(uc.name, saved[k].second, &eq); \
                              if (!eq) out.emplace_back(#name); ++k; }
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

