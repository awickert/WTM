#include "fill_spill_merge.hpp"
#include "irf.hpp"
#include "transient_groundwater.hpp"

#include "CreateSNES.cpp"
#include "DMDA_array_pack.cpp"

#include <fmt/core.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscsnes.h>
#include <richdem/common/timer.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

namespace dh = richdem::dephier;
namespace rd = richdem;

constexpr double seconds_in_a_year = 31536000.;

std::string get_current_time_and_date_as_str() {
  const auto now       = std::chrono::system_clock::now();
  const auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

static constexpr char help[] = "trying petsc method to solve the problem using Newton";

void initialise(Parameters& params, ArrayPack& arp, AppCtx& user_context) {
  std::ofstream textfile(params.textfilename, std::ios_base::app);
  // Text file to save outputs of how much is changing and
  // min and max wtd at various times

  // Load the full-grid ArrayPack on rank 0 only, so its arrays are never
  // allocated on non-root ranks (the memory win: 2f-C). ncells_x/y are set from
  // the loaded topography on rank 0 and broadcast to all ranks (the DMDA needs
  // them everywhere). cell_size_area computes only 1-D Class-C arrays and runs
  // on all ranks. InitialiseBoth (labels/runoff/etc. for FSM) and arp.check
  // (full-grid dimension checks) are rank-0 only. See DISTRIBUTED_ARP_DESIGN.md.
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (params.run_type == "transient") {
    textfile << "Initialise transient" << std::endl;
    if (rank == 0)
      InitialiseTransient(params, arp);
    MPI_Bcast(&params.ncells_x, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ncells_y, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    cell_size_area(params, arp);
    textfile << "computed distances, areas, and latitudes" << std::endl;
    if (rank == 0)
      InitialiseBoth(params, arp);
  } else if (params.run_type == "equilibrium") {
    textfile << "Initialise equilibrium" << std::endl;
    if (rank == 0)
      InitialiseEquilibrium(params, arp);
    MPI_Bcast(&params.ncells_x, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ncells_y, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    cell_size_area(params, arp);
    textfile << "computed distances, areas, and latitudes" << std::endl;
    if (rank == 0)
      InitialiseBoth(params, arp);
  } else if (params.run_type == "test") {
    textfile << "Initialise test" << std::endl;
    if (rank == 0)
      InitialiseTest(params, arp);
    MPI_Bcast(&params.ncells_x, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ncells_y, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    cell_size_area(params, arp);
    textfile << "computed distances, areas, and latitudes" << std::endl;
  } else {
    throw std::runtime_error("That was not a recognised run type! Please choose transient or equilibrium.");
  }

  if (rank == 0)
    arp.check();

  InitialiseSNES(user_context, params);

  // Print column headings to textfile to match data that will be printed after each time step.
  textfile << "Cycles_done Total_wtd_change Change_in_GW_only Change_in_SW_only absolute_value_total_wtd_change "
              "abs_change_in_GW abs_change_in_SW change_in_infiltration total_recharge_added total_loss_to_ocean "
              "sum_of_water_tables "
           << std::endl;
  textfile.close();
}

// Distributed recharge for fsm-off runs. Computes the per-cell recharge over each
// rank's OWNED cells, reading forcing from the DMDA-distributed vecs and the water
// table from the distributed carrier (starting_wtd), writing rech_dist and (evap
// mode 0) zeroing surface water in starting_wtd. This is the same computation the
// serial rank-0 loop in update() does, but with no full-grid work and no arp -- so
// the O(N) serial recharge is removed at scale. Only valid with FSM OFF: with FSM on,
// arp.runoff (set by the serial loop) feeds the next cycle's FillSpillMerge, which is
// why the fsm-on path keeps the serial loop (see benchmark/DISTRIBUTED_ARP_DESIGN.md).
//
// Forcing is read into float locals so the arithmetic is bit-identical to the arp
// (float) loop: the surface-water branch subtracts precip-open_water_evap in float;
// the below-surface branch subtracts in double via the explicit cast -- exactly as
// there. runoff is computed and subtracted (a no-op when runoff_ratio_on is 0) but
// not persisted: with FSM off nothing downstream consumes it.
static void distributed_recharge(Parameters& params, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar **precip, **evap, **open_water_evap, **runoff_ratio;
  DMDAVecGetArray(user_context.da, user_context.precip_vec, &precip);
  DMDAVecGetArray(user_context.da, user_context.evap_vec, &evap);
  DMDAVecGetArray(user_context.da, user_context.open_water_evap_vec, &open_water_evap);
  DMDAVecGetArray(user_context.da, user_context.runoff_ratio_vec, &runoff_ratio);

#pragma omp parallel for default(none) \
    shared(params, dmdapack, precip, evap, open_water_evap, runoff_ratio, xs, ys, xm, ym) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      // The DMDA vecs hold double(float) values scattered from the (float) arp
      // arrays, so narrowing back to float is lossless and recovers the exact arp
      // operands -- required to reproduce the arp loop's float arithmetic bit-for-bit.
      const float precip_f = static_cast<float>(precip[j][i]);
      const float evap_f   = static_cast<float>(evap[j][i]);
      const float owe_f    = static_cast<float>(open_water_evap[j][i]);
      const float rratio_f = static_cast<float>(runoff_ratio[j][i]);

      if (params.evap_mode) {
        // Evap mode 1: use the computed open-water evaporation rate.
        if (dmdapack.starting_wtd[j][i] > 0) {  // surface water present
          dmdapack.rech_dist[j][i] = (precip_f - owe_f) / seconds_in_a_year * params.deltat;
        } else {  // water table below the surface; recharge is always positive
          dmdapack.rech_dist[j][i] =
              (std::max(0., static_cast<double>(precip_f) - evap_f)) / seconds_in_a_year * params.deltat;
        }
      } else {
        // Evap mode 0: remove all surface water (like Fan Reinfelder et al., 2013).
        if (dmdapack.starting_wtd[j][i] > 0) {  // surface water present
          dmdapack.starting_wtd[j][i] = 0;
          dmdapack.rech_dist[j][i]    = (precip_f - owe_f) / seconds_in_a_year * params.deltat;
        } else {
          dmdapack.rech_dist[j][i] =
              (std::max(0., static_cast<double>(precip_f) - evap_f)) / seconds_in_a_year * params.deltat;
        }
      }

      if (dmdapack.rech_dist[j][i] > 0) {
        // If there is positive recharge, some of it may run off; subtract that amount.
        const double runoff = rratio_f * dmdapack.rech_dist[j][i];
        dmdapack.rech_dist[j][i] -= runoff;
      }
    }
  }

  DMDAVecRestoreArray(user_context.da, user_context.precip_vec, &precip);
  DMDAVecRestoreArray(user_context.da, user_context.evap_vec, &evap);
  DMDAVecRestoreArray(user_context.da, user_context.open_water_evap_vec, &open_water_evap);
  DMDAVecRestoreArray(user_context.da, user_context.runoff_ratio_vec, &runoff_ratio);
}

// Scatter a rank-0 full-grid array (row-major) into a DMDA_Array_Pack-held owned
// array, through the un-held wtd_global scratch. The pack HOLDS its arrays across
// cycles (persistent DMDAVecGetArray), so we cannot scatter into them directly --
// we scatter into wtd_global, then copy its owned cells into the held destination.
// Templated on the source type so it serves double (wtd/rech) sources; scatterFromZero
// converts to PetscScalar.
template <typename T>
static void scatter_into_owned(AppCtx& user_context, const T* full_r0, PetscScalar** dest) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** scratch;
  user_context.full_grid_gather->scatterFromZero(full_r0, user_context.wtd_global);
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &scratch);
  for (auto j = ys; j < ys + ym; j++)
    for (auto i = xs; i < xs + xm; i++) dest[j][i] = scratch[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &scratch);
}

template <class elev_t>
void update(
    Parameters& params,
    ArrayPack& arp,
    AppCtx& user_context,
    DMDA_Array_Pack& dmdapack,
    richdem::dephier::DepressionHierarchy<elev_t>& deps) {
  richdem::Timer timer_overall;
  timer_overall.start();

  // wtd_old and wtd_mid are diagnostic snapshots read only by rank-0 PrintValues,
  // so maintain them on rank 0 only (they need not exist on non-root ranks).
  PetscMPIInt mpi_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  if (params.run_type == "transient") {
    // UpdateTransientArrays (linear interpolation of the forcing fields from
    // start to end, plus fdepth and the depression hierarchy rebuild) is serial
    // full-grid work; run it on rank 0 (2d). deps and the label/flowdir arrays
    // feed only FillSpillMerge (rank 0). See benchmark/DISTRIBUTED_ARP_DESIGN.md.
    PetscMPIInt trans_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &trans_rank);
    if (trans_rank == 0) {
      UpdateTransientArrays(params, arp);
      // with transient runs, we have to redo the depression hierarchy every time,
      // since the topography is changing.
      deps = dh::GetDepressionHierarchy<float, rd::Topology::D8>(
          arp.topo, arp.cell_area, arp.label, arp.final_label, arp.flowdirs);
    }
    // Re-scatter topo/fdepth (and ksat, unchanged) from rank-0 arp to the solve's
    // DMDA vectors so the groundwater solve uses the CURRENT topography this cycle.
    // Those vectors are otherwise scattered only once at init, so without this the
    // solve would ignore the transient topography change entirely. scatter_static_fields
    // now sources from rank 0, so no broadcast of arp.topo/fdepth is needed. See
    // benchmark/DISTRIBUTED_ARP_DESIGN.md (Phase 2e/2f).
    scatter_static_fields(user_context, arp);

    // fsm-off runs distribute the recharge, which reads the forcing (precip, evap,
    // open_water_evap, runoff_ratio) from the DMDA vecs. UpdateTransientArrays just
    // re-interpolated those on rank 0, so re-scatter them each cycle. (fsm-on keeps the
    // serial rank-0 recharge, which reads arp directly, so needs no scatter.)
    if (!params.fsm_on)
      scatter_forcing_fields(user_context, arp);
  }

  // TODO: How should equilibrium know when to exit?
  if ((params.cycles_done % params.cycles_to_save) == 0) {
    // Save the output every "cycles_to_save" iterations, under a new filename
    // so we can compare how the water table has changed through time.
    // wtd is fully assembled on all ranks by FanDarcyGroundwater::update; rank 0 writes.
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      arp.wtd.setNoData(-9999);
      arp.wtd.saveGDAL(fmt::format("{}{:09}.tif", params.outfile_prefix, params.cycles_done));
    }
  }

  if (mpi_rank == 0) {
    arp.wtd_old = arp.wtd;  // These are used to see how much change occurs
    arp.wtd_mid = arp.wtd;  // in FSM vs in the groundwater portion.
  }

  //////////////////////
  // Move groundwater //
  //////////////////////

  std::cerr << "Before GW time: " << get_current_time_and_date_as_str() << std::endl;

  richdem::Timer time_groundwater;
  time_groundwater.start();

  // These iterations refer to how many times to repeat the time step within the groundwater
  // portion of code before running FSM. For example, 1 year GW then FSM could also be run as
  // 2x 6 months GW then FSM.
  // Scatter the per-cycle solve inputs from rank-0 arp into the distributed
  // carriers: the wtd carrier (starting_wtd, advanced in place by the maxiter
  // solves) and the recharge source (rech_dist). Scatter through the un-held
  // wtd_global scratch, then copy its owned cells into the dmdapack-held arrays.
  // Sourcing from rank 0 lets arp.wtd/rech be dropped on non-root ranks. 2f-B / 2f-C.
  //
  // With FSM ON, the serial rank-0 recharge writes arp.rech and FSM writes arp.wtd
  // each cycle, so re-scatter every cycle. With FSM OFF the recharge is distributed
  // (writes rech_dist and starting_wtd in place) and nothing on rank 0 mutates
  // arp.wtd/rech between cycles, so the carriers persist -- scatter only at cycle 0
  // (the initial state). This removes both per-cycle scatters from the fsm-off path.
  if (params.fsm_on || params.cycles_done == 0) {
    scatter_into_owned(user_context, arp.wtd.data(), dmdapack.starting_wtd);
    scatter_into_owned(user_context, arp.rech.data(), dmdapack.rech_dist);
  }

  int iter_count = 0;
  while (iter_count++ < params.maxiter) {
    FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
  }
  // Assemble the full wtd field once, now that the solve loop is done (the
  // intermediate solves only need each rank's owned cells).
  FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);

  std::cerr << "t GW time = " << time_groundwater.lap() << std::endl;
  std::cerr << "t After GW time: " << get_current_time_and_date_as_str() << std::endl;

  if (mpi_rank == 0) {
    arp.wtd_mid = arp.wtd;
  }

  ////////////////////////
  // Move surface water //
  ////////////////////////

  if (params.fsm_on) {
    richdem::Timer fsm_timer;
    fsm_timer.start();

    // FillSpillMerge is a global serial algorithm; run it on rank 0, which holds
    // the full arp. Its wtd output stays on rank 0 (the following serial sections
    // and the next-cycle scatter all read rank-0 arp.wtd); no broadcast is needed.
    // See benchmark/DISTRIBUTED_ARP_DESIGN.md.
    if (mpi_rank == 0) {
      dh::FillSpillMerge(params, deps, arp);
    }

    std::cerr << "t FSM time = " << fsm_timer.lap() << std::endl;
    std::cerr << "t After FSM time: " << get_current_time_and_date_as_str() << std::endl;
  }

  /////////////////////////
  // Set recharge values //
  /////////////////////////

  // Check to see where there is surface water, and adjust how evaporation works
  // at these locations.
  richdem::Timer recharge_timer;
  recharge_timer.start();

  // With FSM ON, the recharge is serial full-grid work on rank 0: it writes
  // arp.rech (read by the next cycle's solve), arp.wtd (evap_mode 0's surface-water
  // removal), and arp.runoff, which the NEXT cycle's FillSpillMerge consumes -- so
  // this must stay on rank 0 alongside FSM. With FSM off, this block is skipped and
  // the recharge is distributed instead (below). See DISTRIBUTED_ARP_DESIGN.md.
  PetscMPIInt rech_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rech_rank);
  if (params.fsm_on && rech_rank == 0) {
    // Evap mode 1: Use the computed open-water evaporation rate
    if (params.evap_mode) {
      std::cout << "p updating the recharge field" << std::endl;
#pragma omp parallel for default(none) shared(arp, params)
      for (unsigned int i = 0; i < arp.topo.size(); i++) {
        if (arp.wtd(i) > 0) {  // if there is surface water present
          arp.rech(i) = (arp.precip(i) - arp.open_water_evap(i)) / seconds_in_a_year * params.deltat;
        } else {  // water table is below the surface
          // Recharge is always positive.
          arp.rech(i) =
              (std::max(0., static_cast<double>(arp.precip(i)) - arp.evap(i))) / seconds_in_a_year * params.deltat;
        }

        if (arp.rech(i) > 0) {
          // if there is positive recharge, some of it may run off.
          // set the amount of runoff based on runoff_ratio, and subtract this amount from the recharge.
          arp.runoff(i) = arp.runoff_ratio(i) * arp.rech(i);
          arp.rech(i) -= arp.runoff(i);
        }
      }
    }

    // Evap mode 0: remove all surface water (like Fan Reinfelder et al., 2013)
    else {
      std::cout << "p removing all surface water" << std::endl;
#pragma omp parallel for default(none) shared(arp, params)
      for (unsigned int i = 0; i < arp.topo.size(); i++) {
        if (arp.wtd(i) > 0) {  // if there is surface water present
          arp.wtd(i) = 0;      // use this option when testing GW component alone
          // still set recharge because it could be positive in this cell, and some may run off or move to neighbouring
          // cells
          arp.rech(i) = (arp.precip(i) - arp.open_water_evap(i)) / seconds_in_a_year * params.deltat;
        } else {  // water table is below the surface
          arp.rech(i) =
              (std::max(0., static_cast<double>(arp.precip(i)) - arp.evap(i))) / seconds_in_a_year * params.deltat;
        }
        if (arp.rech(i) > 0) {
          // if there is positive recharge, some of it may run off.
          // set the amount of runoff based on runoff_ratio, and subtract this amount from the recharge.
          arp.runoff(i) = arp.runoff_ratio(i) * arp.rech(i);
          arp.rech(i) -= arp.runoff(i);
        }
      }
    }
  }
  // (fsm-on) rech and wtd stay on rank 0 -- the next cycle scatters them from
  // rank-0 arp into the distributed solve carriers, so no broadcast is needed.

  // FSM off: compute the recharge distributed over each rank's owned cells
  // (writing rech_dist and, in evap_mode 0, zeroing surface water in starting_wtd),
  // then assemble the post-recharge wtd on rank 0 for PrintValues and the next
  // output. starting_wtd and rech_dist persist to the next cycle (the start-of-cycle
  // scatter is gated to cycle 0), so no per-cycle round-trip through arp.
  if (!params.fsm_on) {
    distributed_recharge(params, user_context, dmdapack);
    FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);
  }

  std::cerr << "t Set recharge time = " << recharge_timer.lap() << std::endl;
  std::cerr << "After setting recharge values: " << get_current_time_and_date_as_str() << std::endl;

  // Print values about the change in water table depth to the text file.
  PrintValues(params, arp);

  if (mpi_rank == 0) {
    arp.wtd_old = arp.wtd;
  }
  params.cycles_done += 1;
  std::cerr << "t Done time = " << get_current_time_and_date_as_str() << std::endl;
  std::cerr << "t WTM update time = " << timer_overall.lap() << std::endl;
}

void run(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  // Set the initial depression hierarchy.
  // For equilibrium runs, this is the only time this needs to be done.
  // deps feeds only FillSpillMerge (rank 0 only), so build it on rank 0 only;
  // on other ranks it stays default-constructed and unused.
  richdem::dephier::DepressionHierarchy<float> deps;
  PetscMPIInt deps_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &deps_rank);
  if (deps_rank == 0) {
    deps = dh::GetDepressionHierarchy<float, rd::Topology::D8>(
        arp.topo, arp.cell_area, arp.label, arp.final_label, arp.flowdirs);
  }

  while (params.cycles_done < params.total_cycles) {
    update(params, arp, user_context, dmdapack, deps);
  }
}

void finalise(Parameters& params, ArrayPack& arp, AppCtx& user_context) {
  std::ofstream textfile(params.textfilename, std::ios_base::app);

  textfile << "p done with processing" << std::endl;
  // Save the final answer. wtd is assembled on all ranks; only rank 0 writes to avoid conflicts.
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    arp.wtd.setNoData(-9999);
    arp.wtd.saveGDAL(fmt::format("{}{:09}.tif", params.outfile_prefix, params.cycles_done));
  }

  textfile.close();

  delete user_context.full_grid_gather;  // destroys PETSc scatter/vecs; must precede PetscFinalize
  user_context.full_grid_gather = nullptr;
  VecDestroy(&user_context.wtd_global);
  VecDestroy(&user_context.rech_source);

  SNESDestroy(&user_context.snes);
  DMDestroy(&user_context.da);
  VecDestroy(&user_context.x);
  VecDestroy(&user_context.b);
  VecDestroy(&user_context.cellsize_EW_squared);
  VecDestroy(&user_context.fdepth_vec);
  VecDestroy(&user_context.ksat_vec);
  VecDestroy(&user_context.mask);
  VecDestroy(&user_context.topo_vec);
  VecDestroy(&user_context.rech_vec);
  VecDestroy(&user_context.porosity_vec);
  VecDestroy(&user_context.starting_wtd);
  VecDestroy(&user_context.precip_vec);
  VecDestroy(&user_context.evap_vec);
  VecDestroy(&user_context.open_water_evap_vec);
  VecDestroy(&user_context.runoff_ratio_vec);
  VecDestroy(&user_context.topo_local);
  VecDestroy(&user_context.fdepth_local);
  VecDestroy(&user_context.ksat_local);
  VecDestroy(&user_context.T_local);
}

int main(int argc, char** argv) {
  // if (argc != 2) {
  //   // Make sure that the user is running the code with a configuration file.
  //   std::cerr << "Syntax: " << argv[0] << " <Configuration File>" << std::endl;
  //   return -1;
  // }

  std::cerr << "Reading configuration file '" << argv << "'..." << std::endl;
  Parameters params(argv[1]);

  ArrayPack arp;

  AppCtx user_context;

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));

  initialise(params, arp, user_context);

  // Structural acceptance check (2f-C, the memory win): the full-grid ArrayPack
  // is allocated only on rank 0; non-root ranks hold it empty. Assert it so a
  // regression that reintroduces full-grid allocation on non-root is caught.
  {
    PetscMPIInt r;
    MPI_Comm_rank(PETSC_COMM_WORLD, &r);
    const size_t expected = (r == 0) ? static_cast<size_t>(params.ncells_x) * params.ncells_y : 0;
    if (arp.topo.size() != expected) {
      throw std::runtime_error(
          "2f-C acceptance check failed: ArrayPack is not rank-0-only (topo size " + std::to_string(arp.topo.size()) +
          " on rank " + std::to_string(r) + ", expected " + std::to_string(expected) + ")");
    }
  }

  // Populate the static global vecs (mask/porosity from rank-0, cellsize from the 1-D array)
  // BEFORE DMDA_Array_Pack holds them, then scatter topo/fdepth/ksat to their local ghost vectors.
  populate_DMDA_array_pack(user_context, arp);
  scatter_static_fields(user_context, arp);

  DMDA_Array_Pack dmdapack(user_context);  // holds the now-populated vecs; must come after the above

  run(params, arp, user_context, dmdapack);

  dmdapack.release();

  finalise(params, arp, user_context);

  PetscCall(PetscFinalize());

  return 0;
}
