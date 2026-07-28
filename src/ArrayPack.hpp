#pragma once

#include <richdem/common/Array2D.hpp>
#include "dephier.hpp"

typedef richdem::Array2D<float> f2d;
typedef richdem::Array2D<double> d2d;

typedef std::vector<double> dvec;

struct ArrayPack {
  // input data files - transient:

  f2d evap_start;
  f2d evap_end;
  f2d open_water_evap_start;
  f2d open_water_evap_end;
  f2d precip_start;
  f2d precip_end;
  f2d runoff_ratio_start;
  f2d runoff_ratio_end;
  f2d slope_start;
  f2d slope_end;
  f2d topo_start;
  f2d topo_end;
  f2d winter_temp_start;
  f2d winter_temp_end;

  // input data files - equilibrium:

  f2d evap;
  f2d open_water_evap;
  f2d precip;
  f2d runoff_ratio;
  f2d slope;
  f2d topo;
  f2d winter_temp;

  // input data files - both

  f2d ksat;
  f2d land_mask;
  f2d porosity;
  f2d vert_ksat;

  // other data storage arrays:

  d2d effective_storativity;
  d2d fdepth;
  d2d infiltration_array;
  d2d rech;
  d2d runoff;
  d2d transmissivity;

  // arrays recording various states of water table depth:

  d2d wtd;
  d2d wtd_mid;
  d2d wtd_old;

  // arrays used for calculations in code:

  dvec cell_area;
  dvec cellsize_e_w_metres;
  // Per-row conservative-FV flux geometry factors (L_wall/d_centre); see GRID_CONVENTION.md.
  dvec geom_ew;  // E-W face: cellsize_n_s / cellsize_e_w[j]
  dvec geom_n;   // N face:   cellsize_e_w[N edge] / cellsize_n_s
  dvec geom_s;   // S face:   cellsize_e_w[S edge] / cellsize_n_s

  // labels and flow directions:

  richdem::Array2D<richdem::dephier::dh_label_t> label;        // No cells are part of a depression
  richdem::Array2D<richdem::dephier::dh_label_t> final_label;  // No cells are part of a depression
  richdem::Array2D<richdem::flowdir_t> flowdirs;               // No cells flow anywhere

  // Cumulative state variables
  double total_added_recharge = 0;
  double total_loss_to_ocean  = 0;
  // Groundwater-only ocean loss (set_starting_values + solve copy-back). Under MPI this is a
  // per-rank OWNED-cell partial, reduced to a global total in PrintValues. Kept separate from
  // total_loss_to_ocean, which FillSpillMerge accumulates on the full replicated grid on every
  // rank (already global, so it must NOT be reduced).
  double total_loss_to_ocean_gw = 0;
  // Water leaving through land->ocean faces (the Darcy interface flux), summed over owned cells and
  // substeps as a per-rank partial (reduced to a global total in PrintValues). Ocean cells are
  // Dirichlet h=0, so the crossing flux is absorbed at the boundary and does NOT show up as
  // ocean-cell content -- total_loss_to_ocean_gw (which counts that content) therefore misses it.
  // This term is what makes the budget close: recharge = d(storage) + ocean_outflow + surface_removed.
  double total_ocean_outflow_gw = 0;
  // Water removed by the sub-surface surface-water sink (-wtm_surface_sink), summed over owned
  // cells and substeps as a per-rank partial (reduced to a global total in PrintValues, like
  // total_loss_to_ocean_gw). For the no-FSM case the removed water is discarded, so this scalar is
  // what closes the water budget: d(wtd_sum) = added_recharge - loss_to_ocean - surface_removed.
  double total_surface_removed = 0;
  // Water evaporated by the implicit demand-identity taper (-wtm_evap_taper, taper 2), summed over
  // owned cells and substeps (per-rank partial, reduced in PrintValues). This water leaves to the
  // ATMOSPHERE (unlike the sink's exfiltration to FSM), so it is a separate budget-loss channel:
  // total_storage_change = total_solver_recharge - total_ocean_outflow - total_surface_removed
  // - total_evap_removed. See benchmark/SURFACE_SINK_DESIGN.md sec 14.
  double total_evap_removed = 0;
  // EXACT budget-closing accumulators (Picard/BDF2 path): the solver's per-step discrete storage
  // change and specific-yield recharge, summed over owned land cells (per-rank partials). The
  // discrete balance guarantees total_storage_change = total_solver_recharge - total_ocean_outflow
  // - total_surface_removed to the SNES tolerance, so the budget closes to ~machine zero (vs the
  // ~1% of the physical snapshot). See benchmark/WATER_BUDGET.md.
  double total_storage_change  = 0;
  double total_solver_recharge = 0;

  void check() const;
};
