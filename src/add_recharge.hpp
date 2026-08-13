#pragma once

#include <algorithm>

#include "ArrayPack.hpp"

extern bool g_extended_soil;  // [WIP] aquifer continues above surface; see update_effective_storativity.hpp

// VOLUME-BASED recharge. `my_rech` is the recharge water DEPTH for this step (rate*dt), signed:
// >0 recharge, <0 net evaporation. Returns the effective DEPTH (volume per unit area) to add to the
// stored-water balance V(wtd); the residual scales it by 1/storativity for its head-form units, so the
// realized volume is exactly this depth regardless of scheme. The pore-space-vs-surface-water partition
// is NOT baked here -- it is resolved dynamically by storedVolume() in the residual, so a cell that
// crosses the land surface within a step is handled CONSISTENTLY across every time-integration scheme.
//
// This replaces the old `/porosity` HEAD conversion, which -- computed from the STARTING water table and
// then re-scaled by each scheme's storativity (secant for backward-Euler, tangent for TR-BDF2 /
// BDF2-on-V) -- over-scaled surface-crossing cells by S/porosity and made the schemes disagree in the
// dt->0 limit. That defect is inherited from v2.0.1 (the old body was byte-identical upstream); it is
// dormant at equilibrium and bites transient surface-crossing cells. See
// benchmark/TRANSIENT_RECHARGE_INCONSISTENCY.md.
//
// Evaporation (my_rech<0) removes only surface water (wtd>0), capped at what is present; it never draws
// down groundwater.
inline double add_recharge(const double my_rech, const double my_wtd, const double my_porosity) {
  (void)my_porosity;  // partition is now dynamic (storedVolume in the residual); no /porosity here.
  // Extended-soil: aquifer everywhere; the residual scales this depth by porosity. No evaporation branch.
  if (g_extended_soil) return (my_rech > 0.0) ? my_rech : 0.0;

  if (my_wtd >= 0.0) {  // surface water present (or exactly at the surface)
    // Positive recharge adds its full depth; evaporation removes surface water down to the surface at most.
    return (my_rech >= 0.0) ? my_rech : std::max(my_rech, -my_wtd);
  }
  // Below the surface: positive recharge adds its full depth (partition resolved dynamically in the
  // solve); negative forcing finds no surface water to evaporate.
  return (my_rech > 0.0) ? my_rech : 0.0;
}
