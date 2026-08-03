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

  // Taper 2/3: read -wtm_evap_taper / -wtm_extinction now, BEFORE the initial recharge (InitialiseBoth
  // below), so its explicit recharge sees the same flags as the per-cycle path. update() re-reads them
  // (idempotent). Warn once (rank 0) about configurations other than the blessed smooth transition.
  FanDarcyGroundwater::read_evap_taper_options(params);
  if (rank == 0)
    FanDarcyGroundwater::warn_taper_configuration(params);

  // The per-cycle recharge is distributed across ranks except when FSM is on AND
  // infiltration_on is set (then it stays on the serial rank-0 path, because the
  // cell-crossing runoff it produces feeds FSM; see benchmark/DISTRIBUTED_ARP_DESIGN.md).
  // Warn once so the user is not surprised that this configuration is slower than an
  // infiltration_on 0 run at high rank counts. The groundwater solve itself still runs
  // in parallel; only the recharge step is serial here.
  if (rank == 0 && params.fsm_on && params.infiltration_on) {
    std::cerr << "WARNING: infiltration_on is set with FSM on -- the per-cycle recharge runs on "
                 "the serial (rank-0) path and is NOT parallel-accelerated. Expect slower per-cycle "
                 "times at high MPI-rank counts than an equivalent infiltration_on 0 run. The "
                 "groundwater solve is still parallel." << std::endl;
  }

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
              "sum_of_water_tables total_surface_removed total_ocean_outflow "
              "stored_volume ocean_loss_closing budget_residual exact_budget_residual total_evap_removed "
           << std::endl;
  textfile.close();
}

// Distributed recharge for fsm-off runs. Computes the per-cell recharge over each
// rank's OWNED cells, reading forcing from the DMDA-distributed vecs and the water
// table from the distributed carrier (starting_wtd), writing rech_dist and (evap
// mode 0) zeroing surface water in starting_wtd. This is the same computation the
// serial rank-0 loop in update() does, but with no full-grid work and no arp -- so
// the O(N) serial recharge is removed at scale. It also writes the runoff
// (runoff_ratio*rech) into the distributed runoff_dist carrier; the caller gathers that
// to rank-0 arp.runoff for the next FSM when runoff_ratio_on (else it stays 0 and FSM's
// own cleanup keeps arp.runoff at 0). The caller (update()) gates this via
// distribute_recharge and, when fsm is on, resyncs starting_wtd from post-FSM arp.wtd
// first. Used for fsm-off and for fsm-on with infiltration off. See DISTRIBUTED_ARP_DESIGN.md.
//
// Forcing is read into float locals so the arithmetic is bit-identical to the arp
// (float) loop: the surface-water branch subtracts precip-open_water_evap in float;
// the below-surface branch subtracts in double via the explicit cast -- exactly as there.
static void distributed_recharge(Parameters& params, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar **precip, **evap, **open_water_evap, **runoff_ratio;
  DMDAVecGetArray(user_context.da, user_context.precip_vec, &precip);
  DMDAVecGetArray(user_context.da, user_context.evap_vec, &evap);
  DMDAVecGetArray(user_context.da, user_context.open_water_evap_vec, &open_water_evap);
  DMDAVecGetArray(user_context.da, user_context.runoff_ratio_vec, &runoff_ratio);

  // Taper 2: when the smooth ET->open-water transition is on, evaporation is the implicit E_eff in
  // the solve, so the explicit recharge here is just the precip source (runoff still partitions it).
  const bool evap_taper = FanDarcyGroundwater::evap_taper_on();
#pragma omp parallel for default(none) \
    shared(params, dmdapack, precip, evap, open_water_evap, runoff_ratio, xs, ys, xm, ym, evap_taper) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      // The DMDA vecs hold double(float) values scattered from the (float) arp
      // arrays, so narrowing back to float is lossless and recovers the exact arp
      // operands -- required to reproduce the arp loop's float arithmetic bit-for-bit.
      const float precip_f = static_cast<float>(precip[j][i]);
      const float evap_f   = static_cast<float>(evap[j][i]);
      const float owe_f    = static_cast<float>(open_water_evap[j][i]);
      const float rratio_f = static_cast<float>(runoff_ratio[j][i]);

      if (evap_taper) {
        // Evaporation is the implicit ET->owe taper; feed just the precip source (requires evap_mode 1).
        dmdapack.rech_dist[j][i] = precip_f / seconds_in_a_year * params.deltat;
      } else if (params.evap_mode) {
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

      dmdapack.runoff_dist[j][i] = 0.0;
      if (dmdapack.rech_dist[j][i] > 0) {
        // If there is positive recharge, some of it may run off; store the runoff (so
        // the caller can gather it to rank-0 arp.runoff for the next FSM when
        // runoff_ratio_on) and subtract it from the recharge. Matches the serial loop,
        // which writes arp.runoff only where rech > 0 and leaves it at FSM's 0 elsewhere.
        const double runoff        = rratio_f * dmdapack.rech_dist[j][i];
        dmdapack.runoff_dist[j][i] = runoff;
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
// Used for the cycle-0 solve-input load and the post-FSM wtd resync. Templated on
// the source type so it serves both double (wtd/rech) sources; scatterFromZero
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

  // Distribute the recharge (over each rank's owned cells) instead of the serial
  // rank-0 loop whenever it is safe. The one cross-boundary output the serial recharge
  // produces for FillSpillMerge is arp.runoff = runoff_ratio * rech; when runoff_ratio_on
  // the distributed recharge computes that too and gathers it to rank-0 arp.runoff before
  // the next FSM (below). "Cell-crossing runoff" (infiltration_on) is entirely FSM-internal
  // (rank 0) and does not affect recharge correctness, but no fixture exercises
  // infiltration_on, so we keep that case on the serial path until it has a test. fsm-off
  // always distributes (no FSM consumer at all). See benchmark/DISTRIBUTED_ARP_DESIGN.md.
  const bool distribute_recharge = !params.fsm_on || !params.infiltration_on;

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

    // Runs that distribute the recharge read the forcing (precip, evap,
    // open_water_evap, runoff_ratio) from the DMDA vecs. UpdateTransientArrays just
    // re-interpolated those on rank 0, so re-scatter them each cycle. (The serial
    // rank-0 recharge reads arp directly, so needs no scatter.)
    if (distribute_recharge)
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
  // Load the per-cycle solve inputs from rank-0 arp into the distributed carriers:
  // the wtd carrier (starting_wtd, advanced in place by the maxiter solves) and the
  // recharge source (rech_dist). Sourcing from rank 0 lets arp.wtd/rech be dropped
  // on non-root ranks. 2f-B / 2f-C.
  //
  // When the recharge is NOT distributed (serial rank-0 recharge writes arp.rech,
  // FSM writes arp.wtd, both each cycle), re-load every cycle. When it IS distributed
  // the carriers persist: rech_dist is written in place by distributed_recharge and
  // starting_wtd is resynced from arp.wtd only where FSM changed it (post-FSM, below),
  // so we load only at cycle 0 (the initial state). This removes the per-cycle wtd/rech
  // scatters from the distributed path.
  if (!distribute_recharge || params.cycles_done == 0) {
    scatter_into_owned(user_context, arp.wtd.data(), dmdapack.starting_wtd);
    scatter_into_owned(user_context, arp.rech.data(), dmdapack.rech_dist);
  }

  // Reset the per-cycle sink-removal accumulator (taper 1): it sums the implicit sink's removed depth
  // over this cycle's sub-steps, then is gathered into arp.runoff for FSM below. Zeroing here (owned
  // cells) makes each cycle start fresh; a harmless no-op when the sink is off (it stays 0).
  {
    const auto [xs, ys, xm, ym] = get_corners(user_context.da);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        dmdapack.sink_removed_dist[j][i] = 0.0;
  }

  if (user_context.use_dt_adaptive) {
    // Adaptive stepping covers the SAME cycle duration as the fixed loop would
    // (maxiter * base deltat), but with variable, error-controlled sub-steps chosen by
    // the controller in FanDarcyGroundwater::update (which mutates user_context.deltat to
    // the next proposed size). Clamp each step to the time remaining in the cycle so we
    // land exactly on the target. See benchmark/BDF2_ADAPTIVE_DESIGN.md.
    const double cycle_duration = params.maxiter * params.deltat;
    double       t              = 0.0;
    int          nsteps         = 0;
    while (t < cycle_duration * (1.0 - 1e-9) && nsteps < 1000000) {
      const double remaining = cycle_duration - t;
      if (user_context.deltat > remaining) user_context.deltat = remaining;
      const double dt_taken = user_context.deltat;
      FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
      t += dt_taken;
      nsteps++;
    }
    PetscPrintf(PETSC_COMM_WORLD, "adaptive dt: %d steps to cover %g s (fixed would be %d)\n",
                nsteps, cycle_duration, params.maxiter);
  } else if (user_context.use_newton_continuation) {
    // Newton pseudo-transient continuation (equilibrium): march maxiter ACCEPTED steps with a
    // Newton-iteration-controlled dt. Start deltat small so the storage term S/deltat keeps the
    // Jacobian diagonally dominant (a large step from a far guess overshoots into a SINGULAR Jacobian,
    // which fails even a direct solve); GROW dt after an easy step (converged in <= dtc_easy_iters),
    // HOLD it when a step was hard (near the safe ceiling), and REJECT+shrink+retry when a step does
    // not converge (update() returns -1 without committing, so the state is preserved). Reject/retry is
    // what lets it survive the dt overshoot AND the per-cycle FSM perturbation that defeats the Picard
    // adaptive path. deltat persists across cycles, ramping toward a near-steady large dt as the state
    // warms; the recharge is rescaled to rate*deltat in update() so the steady state is correct at any
    // dt. A rejected step's pre-solve budget accumulators (set_starting_values) are rolled back so a
    // retry does not double-count. See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
    int accepted = 0, retries = 0;
    while (accepted < params.maxiter) {
      const double rech_snap  = arp.total_added_recharge;   // roll back on a rejected step
      const double ocean_snap = arp.total_loss_to_ocean_gw;
      const double dt_try     = user_context.deltat;
      const int    its        = FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
      if (its < 0) {  // rejected (non-converged): restore accumulators, shrink dt, retry same step
        arp.total_added_recharge  = rech_snap;
        arp.total_loss_to_ocean_gw = ocean_snap;
        user_context.deltat        = dt_try * user_context.dtc_shrink;
        if (++retries > user_context.dtc_max_retries)
          throw std::runtime_error("dt-continuation: step failed to converge after max retries; deltat too "
                                   "small or the guess is too far (lower -wtm_dtc_grow / raise -wtm_dtc_dt0).");
        continue;  // do NOT advance `accepted`
      }
      accepted++;
      retries = 0;
      if (its <= user_context.dtc_easy_iters) user_context.deltat = dt_try * user_context.dtc_grow;  // else HOLD
      if (user_context.deltat > user_context.dtc_dt_max) user_context.deltat = user_context.dtc_dt_max;
    }
    PetscPrintf(PETSC_COMM_WORLD, "dt-continuation: deltat now %g s after this cycle.\n", user_context.deltat);
  } else {
    int iter_count = 0;
    while (iter_count++ < params.maxiter) {
      FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
    }
  }
  // Assemble the full wtd field once, now that the solve loop is done (the
  // intermediate solves only need each rank's owned cells).
  FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);

  // Taper 1: hand this cycle's implicit-sink removal to FSM. The sink held wtd<=0 during the solve,
  // so FSM's own wtd>0->runoff handoff won't fire; gather the accumulated exfiltration depth into
  // rank-0 arp.runoff (adding) so this cycle's FillSpillMerge routes it. No-op when the sink is off.
  if (params.fsm_on && FanDarcyGroundwater::surface_sink_on())
    FanDarcyGroundwater::gather_sink_removed_to_zero(params, arp, user_context, dmdapack);

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

    // FSM changed arp.wtd on rank 0. When the recharge is distributed, it reads the
    // water table from the distributed carrier (starting_wtd), so resync starting_wtd
    // from the post-FSM arp.wtd. (When the recharge is serial it reads arp.wtd on
    // rank 0 directly, so no resync is needed.) This is the fsm-on leg of the round
    // trip: gather wtd -> FSM (rank 0) -> scatter wtd back. See DISTRIBUTED_ARP_DESIGN.md.
    if (distribute_recharge)
      scatter_into_owned(user_context, arp.wtd.data(), dmdapack.starting_wtd);
  }

  /////////////////////////
  // Set recharge values //
  /////////////////////////

  // Check to see where there is surface water, and adjust how evaporation works
  // at these locations.
  richdem::Timer recharge_timer;
  recharge_timer.start();

  // The serial rank-0 recharge is kept only when the recharge is NOT distributed
  // (fsm_on && (runoff_ratio_on || infiltration_on)): it writes arp.rech (read by the
  // next cycle's solve), arp.wtd (evap_mode 0's surface-water removal), and arp.runoff,
  // which the NEXT cycle's FillSpillMerge consumes -- so it must stay on rank 0
  // alongside FSM. Otherwise this block is skipped and the recharge is distributed
  // (below). See DISTRIBUTED_ARP_DESIGN.md.
  PetscMPIInt rech_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rech_rank);
  if (!distribute_recharge && rech_rank == 0) {
    // The taper (taper 2/3) governs evaporation via the implicit E_eff, so feed just the precip source
    // regardless of evap_mode -- the smooth removal auto-zeroes standing water, so there is NO
    // independent wtd=0 under the taper. Otherwise the hard evap_mode split: mode 1 evaporates surface
    // water at owe (it persists); mode 0 removes all surface water (wtd=0; GW-alone testing, Fan
    // Reinfelder et al. 2013). Taper-first (not nested in evap_mode 1) so it also works in evap_mode 0.
    const bool evap_taper = FanDarcyGroundwater::evap_taper_on();
    std::cout << (evap_taper       ? "p updating the recharge field (taper)"
                 : params.evap_mode ? "p updating the recharge field"
                                    : "p removing all surface water")
              << std::endl;
#pragma omp parallel for default(none) shared(arp, params, evap_taper)
    for (unsigned int i = 0; i < arp.topo.size(); i++) {
      if (evap_taper) {
        arp.rech(i) = arp.precip(i) / seconds_in_a_year * params.deltat;
      } else if (arp.wtd(i) > 0) {  // surface water present
        if (!params.evap_mode)
          arp.wtd(i) = 0;  // evap_mode 0: remove all surface water (GW-alone testing)
        arp.rech(i) = (arp.precip(i) - arp.open_water_evap(i)) / seconds_in_a_year * params.deltat;
      } else {  // water table below the surface; recharge is always positive
        arp.rech(i) =
            (std::max(0., static_cast<double>(arp.precip(i)) - arp.evap(i))) / seconds_in_a_year * params.deltat;
      }
      if (arp.rech(i) > 0) {
        // positive recharge may partly run off (runoff_ratio); subtract it from the recharge.
        arp.runoff(i) = arp.runoff_ratio(i) * arp.rech(i);
        arp.rech(i) -= arp.runoff(i);
      }
    }
  }
  // (serial path) rech and wtd stay on rank 0 -- the next cycle re-loads them from
  // rank-0 arp into the distributed solve carriers, so no broadcast is needed.

  // Distributed recharge: compute over each rank's owned cells (writing rech_dist
  // and, in evap_mode 0, zeroing surface water in starting_wtd), then assemble the
  // post-recharge wtd on rank 0 for PrintValues and the next output. starting_wtd and
  // rech_dist persist to the next cycle (cycle-0-gated load above; fsm-on resyncs
  // starting_wtd post-FSM), so no per-cycle round-trip through arp for the recharge.
  if (distribute_recharge) {
    distributed_recharge(params, user_context, dmdapack);
    FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);
    // When runoff_ratio_on and FSM is on, the recharge's runoff (runoff_ratio*rech)
    // feeds the next FSM, which runs on rank 0 -- so gather the distributed runoff to
    // rank-0 arp.runoff. Otherwise the runoff is 0 and arp.runoff stays at FSM's own 0,
    // so no gather is needed (fsm-off has no FSM consumer at all).
    if (params.fsm_on && params.runoff_ratio_on)
      FanDarcyGroundwater::gather_runoff_to_zero(params, arp, user_context, dmdapack);
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
  VecDestroy(&user_context.runoff_dist_vec);
  VecDestroy(&user_context.sink_removed_dist_vec);

  SNESDestroy(&user_context.snes);
  DMDestroy(&user_context.da);
  VecDestroy(&user_context.x);
  VecDestroy(&user_context.b);
  VecDestroy(&user_context.cellsize_EW_squared);
  VecDestroy(&user_context.geom_ew_vec);
  VecDestroy(&user_context.geom_n_vec);
  VecDestroy(&user_context.geom_s_vec);
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
  VecDestroy(&user_context.mask_local);

  // Picard path (nullptr / no-op when -wtm_picard was not set).
  MatDestroy(&user_context.picard_A);
  VecDestroy(&user_context.picard_r);
  VecDestroy(&user_context.starting_wtd_prev);  // BDF2 history (nullptr / no-op otherwise)
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

  // Ghost-scatter the (static) mask once, so ocean-outflow accounting can identify land->ocean faces
  // at rank boundaries. Done before the pack holds the mask global. Assumes the mask is static within
  // a run (true for equilibrium; a transient coastline change would need a re-scatter).
  DMGlobalToLocalBegin(user_context.da, user_context.mask, INSERT_VALUES, user_context.mask_local);
  DMGlobalToLocalEnd(user_context.da, user_context.mask, INSERT_VALUES, user_context.mask_local);

  DMDA_Array_Pack dmdapack(user_context);  // holds the now-populated vecs; must come after the above

  run(params, arp, user_context, dmdapack);

  dmdapack.release();

  finalise(params, arp, user_context);

  PetscCall(PetscFinalize());

  return 0;
}
