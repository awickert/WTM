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

}
