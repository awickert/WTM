#pragma once

#include "ArrayPack.hpp"
#include "parameters.hpp"

void InitialiseTransient(Parameters& params, ArrayPack& arp);

void InitialiseEquilibrium(Parameters& params, ArrayPack& arp);

void InitialiseTest(Parameters& params, ArrayPack& arp);

void derive_grid_geometry(Parameters& params, const ArrayPack& arp);

void cell_size_area(Parameters& params, ArrayPack& arp);

void InitialiseBoth(const Parameters& params, ArrayPack& arp);

void UpdateTransientArrays(const Parameters& params, ArrayPack& arp);

void PrintValues(Parameters& params, const ArrayPack& arp);
