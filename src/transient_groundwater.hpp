#pragma once

#include "ArrayPack.hpp"
#include "CreateSNES.hpp"
#include "DMDA_array_pack.hpp"

#include <richdem/common/Array2D.hpp>

namespace FanDarcyGroundwater {

int update(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Assemble the full wtd field on every rank from each rank's owned cells.
// Called once per cycle after the maxiter solve loop -- the intermediate solves
// only need each rank's own owned wtd, so the full-grid assembly is not needed
// per solve. See benchmark/DISTRIBUTED_ARP_DESIGN.md (Phase 2f, lever #2).
void gather_wtd_to_all(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Gather the distributed per-cycle runoff to rank-0 arp.runoff for the next FSM,
// when runoff_ratio_on. See benchmark/DISTRIBUTED_ARP_DESIGN.md (2c).
void gather_runoff_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Whether the implicit sub-surface sink is configured this run (taper 1).
bool surface_sink_on();

// Whether the demand-identity evaporation taper is on (taper 2). The explicit-recharge sites feed
// just precip when this is set -- the smooth implicit E_eff carries the ET->open-water transition.
bool evap_taper_on();

// Read the -wtm_evap_taper options (+ wtd_c, s) and enforce evap_mode 1. Call early (before the
// initial recharge) so every explicit-recharge site sees a consistent flag. Idempotent.
void read_evap_taper_options(const Parameters& params);

// Add this cycle's implicit-sink removal (sink_removed_dist) into rank-0 arp.runoff so FSM routes
// it -- the order-preserving replacement for FSM's hard wtd>0->runoff handoff. See SURFACE_SINK_DESIGN.md.
void gather_sink_removed_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

}
