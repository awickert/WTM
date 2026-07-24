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
  // carriers once per cycle: the wtd carrier (starting_wtd, advanced in place by
  // the maxiter solves) and the recharge source (rech_dist, constant across the
  // loop). Scatter through the un-held wtd_global scratch, then copy its owned
  // cells into the dmdapack-held arrays. Sourcing from rank 0 lets arp.wtd/rech
  // be dropped on non-root ranks. 2f-B / 2f-C.
  {
    const auto [xs, ys, xm, ym] = get_corners(user_context.da);
    PetscScalar** scratch;

    user_context.full_grid_gather->scatterFromZero(arp.wtd.data(), user_context.wtd_global);
    DMDAVecGetArray(user_context.da, user_context.wtd_global, &scratch);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++) dmdapack.starting_wtd[j][i] = scratch[j][i];
    DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &scratch);

    user_context.full_grid_gather->scatterFromZero(arp.rech.data(), user_context.wtd_global);
    DMDAVecGetArray(user_context.da, user_context.wtd_global, &scratch);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++) dmdapack.rech_dist[j][i] = scratch[j][i];
    DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &scratch);
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

  // The recharge computation is serial full-grid work. Run it on rank 0 only
  // (arrays are still replicated, so rank 0 sees identical input), then
  // broadcast the outputs the rest of the model consumes: rech (read by the
  // next cycle's solve) and wtd (modified by evap_mode 0's surface-water
  // removal). runoff is overwritten by this same loop before it is read again,
  // so it needs no broadcast. See benchmark/DISTRIBUTED_ARP_DESIGN.md (Phase 2c).
  PetscMPIInt rech_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rech_rank);
  if (rech_rank == 0) {
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
  // rech and wtd stay on rank 0 -- the next cycle scatters them from rank-0 arp
  // into the distributed solve carriers, so no broadcast is needed.

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
