#pragma once

#include "ArrayPack.hpp"
#include "grid_geometry.hpp"  // derive_grid_geometry / cell_size_area -- shared with dephier.x
#include "parameters.hpp"

void InitialiseTransient(Parameters& params, ArrayPack& arp);

void InitialiseEquilibrium(Parameters& params, ArrayPack& arp);

void InitialiseTest(Parameters& params, ArrayPack& arp);

void InitialiseBoth(const Parameters& params, ArrayPack& arp);

void UpdateTransientArrays(const Parameters& params, ArrayPack& arp);

// The EXACT stored water, Sum storedVolume(w)*A over the full replicated grid (rank 0 only).
// Factored out so the budget's INITIAL volume is produced by the SAME code as every later one:
// a second hand-written copy of this sum is precisely how a baseline drifts from the series it
// is differenced against. See benchmark/WATER_BUDGET.md.
double ComputeStoredVolume(const Parameters& params, const ArrayPack& arp);

// Capture the t=0 stored volume as the budget baseline. MUST be called after the initial water
// table is loaded and BEFORE any stepping. See CaptureInitialStoredVolume's definition for why.
void CaptureInitialStoredVolume(Parameters& params, const ArrayPack& arp);

void PrintValues(Parameters& params, const ArrayPack& arp);
