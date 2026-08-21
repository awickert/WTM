#pragma once

#include "ArrayPack.hpp"
#include "CreateSNES.hpp"
#include "DMDA_array_pack.hpp"

#include <richdem/common/Array2D.hpp>

namespace FanDarcyGroundwater {

int update(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Assemble the full wtd field on every rank from each rank's owned cells.
// Called once per cycle after the per-report step loop -- the intermediate solves
// only need each rank's own owned wtd, so the full-grid assembly is not needed
// per solve. See benchmark/DISTRIBUTED_ARP_DESIGN.md (Phase 2f, lever #2).
void gather_wtd_to_all(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Gather the distributed per-cycle runoff to rank-0 arp.runoff for the next FSM,
// when runoff_ratio_on. See benchmark/DISTRIBUTED_ARP_DESIGN.md (2c).
void gather_runoff_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

// Whether the implicit sub-surface sink is configured this run (taper 1).
bool surface_sink_on();
// Whether the direct-to-runoff (seepage-face) removal is configured this run (-wtm_direct_to_runoff).
bool direct_to_runoff_on();
// Whether the lake-aware active-set skim is on (so the post-solve gather hands its captured seepage to FSM).
bool active_set_on();

// Whether extended-soil surface truncation is on (-wtm_extended_soil): routes above-surface water to
// FSM via the sink accumulator, so the cycle loop must gather it just as for the sink.
bool extended_soil_on();

// Whether post-solve surface exfiltration-to-runoff collection is on (-wtm_surface_exfiltration_to_runoff): clamps wtd->0 and routes the exact
// above-surface excess to FSM via the sink accumulator, so the cycle loop must gather it as for the sink.
bool surface_exfiltration_to_runoff_on();

// Whether the demand-identity evaporation taper is on (taper 2). The explicit-recharge sites feed
// just precip when this is set -- the smooth implicit E_eff carries the ET->open-water transition.
bool evap_taper_on();

// Whether the accessibility / extinction-depth clamp is on (taper 3, awickert/WTM#4). Gates taper 2's
// sub-surface deficit so an arid table draws down only within the extinction depth; inert on its own.
bool extinction_on();

// Read the -wtm_evap_taper options (+ wtd_c, s) and enforce evap_mode 1. Call early (before the
// initial recharge) so every explicit-recharge site sees a consistent flag. Idempotent.
void read_evap_taper_options(const Parameters& params);

// Emit stderr warnings for surface-water evaporation configurations other than the blessed smooth
// transition (taper 2 + taper 3 both on). Call once on rank 0, after read_evap_taper_options.
void warn_taper_configuration(const Parameters& params);

// Add this cycle's implicit-sink removal (sink_removed_dist) into rank-0 arp.runoff so FSM routes
// it -- the order-preserving replacement for FSM's hard wtd>0->runoff handoff. See SURFACE_SINK_DESIGN.md.
void gather_sink_removed_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack);

}
