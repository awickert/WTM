#include "fill_spill_merge.hpp"
#include "git_version.hpp"  // baked-in git commit + clean/dirty state (provenance)
#include "irf.hpp"
#include "transient_groundwater.hpp"
#include "update_effective_storativity.hpp"  // per-cycle pure-water-depth metric (S*Δwtd)

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

#include <yaml-cpp/yaml.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <unistd.h>  // gethostname (provenance)

namespace dh = richdem::dephier;
namespace rd = richdem;

constexpr double seconds_in_a_year = 31536000.;

// Snapshot output filename: report number _ elapsed simulated years, underscore-separated (e.g.
// "<prefix>000000015_1yr.tif"). Each report spans report_seconds of simulated time (report_steps*deltat, or
// the user's report_interval time) -- true even under adaptive dt (the controller varies the sub-step, not the
// report duration) -- so elapsed years is well-defined. The report index keeps the files uniquely ordered; the
// year is the physically meaningful label (essential for transient runs, informative for spin-up progress).
static std::string snapshot_filename(const Parameters& params) {
  const double years = params.cycles_done * params.report_seconds / seconds_in_a_year;
  return fmt::format("{}{:09}_{:.0f}yr.tif", params.outfile_prefix, params.cycles_done, years);
}

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

  // Provenance: record which code produced this run -- the git commit + clean/dirty state baked into the
  // binary at build time. Printed on rank 0; a dirty tree is flagged loudly (results are not reproducible
  // from the hash alone). Referencing these accessors also keeps git_version in the linked binary.
  if (rank == 0) {
    std::cout << "c WTM git commit         = " << wtm_git_commit() << " (" << wtm_git_state() << ")"
              << std::endl;
    if (wtm_git_dirty())
      std::cerr << "WARNING: this WTM binary was built from a DIRTY git tree; its results are not "
                   "reproducible from the commit hash alone."
                << std::endl;
  }

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
    // Grid geometry from the input geotransform (#124), derived on rank 0 (which holds arp.topo's
    // geotransform), then broadcast like ncells so all ranks build the same 1-D Class-C geometry.
    if (rank == 0)
      derive_grid_geometry(params, arp);
    MPI_Bcast(&params.ns_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ew_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.southern_edge, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
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
    // Grid geometry from the input geotransform (#124), derived on rank 0 (which holds arp.topo's
    // geotransform), then broadcast like ncells so all ranks build the same 1-D Class-C geometry.
    if (rank == 0)
      derive_grid_geometry(params, arp);
    MPI_Bcast(&params.ns_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ew_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.southern_edge, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
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
    // Grid geometry from the input geotransform (#124), derived on rank 0 (which holds arp.topo's
    // geotransform), then broadcast like ncells so all ranks build the same 1-D Class-C geometry.
    if (rank == 0)
      derive_grid_geometry(params, arp);
    MPI_Bcast(&params.ns_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.ew_deg_per_cell, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&params.southern_edge, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
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
              "recharge_direct runoff_to_surface elapsed_time_s solves_done rejects_done "
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
static void distributed_recharge(Parameters& params, ArrayPack& arp, AppCtx& user_context,
                                 DMDA_Array_Pack& dmdapack) {
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
    shared(arp, params, dmdapack, precip, evap, open_water_evap, runoff_ratio, xs, ys, xm, ym, evap_taper) \
    collapse(2)
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
        // NOT booked here. runoff_dist holds the NOMINAL depth; both the DELIVERY to FillSpillMerge and
        // the col-20 booking happen together at the handoff (gather_runoff_to_zero), scaled by the dt
        // the step actually took. Booking at preparation and delivering at the handoff would let the
        // diagnostic and the physics disagree.
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

// As scatter_into_owned, but ADDS the scattered rank-0 field onto the held owned array (+=). Used to fold
// the FSM per-step delta into rech_dist as a source (-wtm_fsm_delta_source, the FSM-delta-source work) via the wtd_global scratch.
template <typename T>
static void accumulate_into_owned(AppCtx& user_context, const T* full_r0, PetscScalar** dest) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** scratch;
  user_context.full_grid_gather->scatterFromZero(full_r0, user_context.wtd_global);
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &scratch);
  for (auto j = ys; j < ys + ym; j++)
    for (auto i = xs; i < xs + xm; i++) dest[j][i] += scratch[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &scratch);
}

// Per-timestep surface-water coupling (tight coupling: FillSpillMerge runs EVERY step -- see
// benchmark/FSM_EVERY_STEP_DESIGN.md). Assemble the water table on rank 0, hand this step's above-surface
// removal to FSM, run FillSpillMerge (timed into fsm_seconds), scatter the post-FSM table back, then set the
// recharge for the NEXT step -- which also re-arms arp.runoff (= runoff_ratio*rech) that the next step's FSM
// consumes (so recharge must run per step, not just per report). With fsm_on this is called after every
// ACCEPTED groundwater step; with fsm_off it is called once per report and only assembles the table + sets
// recharge (the FSM/scatter parts are skipped).
template <class elev_t>
static void couple_surface_and_recharge(Parameters& params, ArrayPack& arp, AppCtx& user_context,
                                        DMDA_Array_Pack& dmdapack,
                                        richdem::dephier::DepressionHierarchy<elev_t>& deps, int mpi_rank,
                                        bool distribute_recharge, double& fsm_seconds) {
  // Assemble the full wtd on rank 0 (the intermediate solves only touch each rank's owned cells).
  FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);

  // Hand the runoff-ratio share of (P-ET) to FillSpillMerge for the step that just ran, SCALED by the
  // dt that step actually took. It used to be handed over at the END of the previous call, at nominal
  // step size and never scaled -- so under sub-stepping the model routed an amount proportional to the
  // SOLVE COUNT rather than to elapsed time. Measured on tests/fsm_consistency at runoff_ratio 0.3:
  // 1.36148e10 over 20 solves (fixed dt) against 9.53038e09 over 14 (adaptive), a ratio of 0.700
  // against a solve-count ratio of 0.700. That was a MASS error, not a reporting one -- adaptive and
  // fixed diverged 6.6% in stored volume with the routed channel on, against 0.16% with it off.
  //
  // Scaling here rather than at preparation is what makes it exact: at preparation the next step's dt
  // is not final (the loop can still clamp it to the cycle remainder, and a rejected step re-runs
  // smaller), whereas dt_committed is the dt of the step that has been ACCEPTED. Placed BEFORE the
  // exfiltration gather so the accumulation order into arp.runoff is unchanged, keeping fixed-dt runs
  // bit-identical (dt_committed == params.deltat there, so the scale is exactly 1).
  // Note the guard is runoff_ratio_on ALONE, not `fsm_on &&`. With FSM off the runoff still has to be
  // accounted -- it leaves the domain rather than being routed -- and gating the booking on fsm_on is
  // what made it vanish silently.
  if (params.runoff_ratio_on) {
    // Scale by the interval since the LAST handoff, not by one step. With FSM on that interval is one
    // accepted step (so this is identical to the previous behaviour); with FSM off the coupling runs
    // once per REPORT, and scaling by a single step under-counted by the steps-per-report factor --
    // measured: col 19 fell by 5.21e9 while col 20 recorded only 6.81e8.
    const double interval = params.elapsed_time_s - params.runoff_booked_upto_s;
    params.runoff_booked_upto_s = params.elapsed_time_s;
    const double dt_scale = (params.deltat > 0.0 && interval > 0.0) ? interval / params.deltat : 1.0;
    if (distribute_recharge) {
      FanDarcyGroundwater::gather_runoff_to_zero(params, arp, user_context, dmdapack, dt_scale,
                                                 params.fsm_on != 0);
    } else if (mpi_rank == 0) {
      // Serial rank-0 recharge path: same handoff, same scale, from arp.runoff_nominal instead of the
      // distributed carrier. Rank 0 holds the whole grid and every other rank contributes 0, so the
      // Allreduce in PrintValues still yields the correct global total for col 20.
      for (int j = 0; j < params.ncells_y; j++)
        for (int i = 0; i < params.ncells_x; i++) {
          const double routed = arp.runoff_nominal(i, j) * dt_scale;
          if (params.fsm_on) {
            arp.runoff(i, j) += routed;
          } else {
            arp.total_loss_to_ocean += routed * arp.cell_area[j];  // it leaves; see the gather
          }
          arp.total_runoff_to_surface += routed * arp.cell_area[j];
        }
    }
  }

  // -wtm_fsm_delta_source (FSM-delta-source): rank-0 buffer for FSM's per-cell volume change V(post-FSM)-V(pre-FSM),
  // row-major (matching arp.wtd.data()). Populated below in the FSM block; injected into rech_dist further down.
  const bool fsm_delta_source = FanDarcyGroundwater::fsm_delta_source_on() && params.fsm_on && distribute_recharge;
  std::vector<double> fsm_delta_r0;

  // Hand this step's above-surface removal (sink / extended-soil / exfiltration / direct-to-runoff) into
  // rank-0 arp.runoff so FillSpillMerge routes it. No-op when all are off (stays 0).
  if (params.fsm_on && (FanDarcyGroundwater::surface_sink_on() || FanDarcyGroundwater::extended_soil_on()
                        || FanDarcyGroundwater::surface_exfiltration_to_runoff_on()
                        || FanDarcyGroundwater::direct_to_runoff_on()
                        || FanDarcyGroundwater::active_set_on()))
    FanDarcyGroundwater::gather_sink_removed_to_zero(params, arp, user_context, dmdapack);

  if (mpi_rank == 0) arp.wtd_mid = arp.wtd;  // table after GW, before FSM (GW-vs-FSM change diagnostic)

  if (params.fsm_on) {
    richdem::Timer fsm_timer;
    fsm_timer.start();
    // FillSpillMerge is a global serial algorithm; run it on rank 0, which holds the full arp.
    if (mpi_rank == 0) dh::FillSpillMerge(params, deps, arp);
    fsm_seconds += fsm_timer.lap();
    if (fsm_delta_source) {
      // -wtm_fsm_delta_source (FSM-delta-source): instead of overwriting the carrier with the post-FSM table (an IC jump
      // that breaks 2nd-order accuracy on TR-BDF2/adaptive), KEEP the smooth pre-FSM GW result as starting_wtd
      // and record FSM's per-cell volume change V(post)-V(pre) to fold into the next step's recharge below.
      if (mpi_rank == 0) {
        const size_t n = arp.wtd.size();
        fsm_delta_r0.resize(n);
        const auto* w   = arp.wtd.data();       // post-FSM
        const auto* wm  = arp.wtd_mid.data();   // pre-FSM (== starting_wtd this step)
        const auto* por = arp.porosity.data();
        for (size_t k = 0; k < n; k++)
          fsm_delta_r0[k] = storedVolume(w[k], por[k]) - storedVolume(wm[k], por[k]);
      }
    } else if (distribute_recharge) {
      // FSM changed rank-0 arp.wtd; resync the distributed carrier so the next solve's recharge reads it.
      scatter_into_owned(user_context, arp.wtd.data(), dmdapack.starting_wtd);
    }
  }

  // Carrier reset (approach B). FillSpillMerge has now CONSUMED this step's runoff (routed it to lakes/ocean),
  // so zero the rank-0 arp.runoff carrier explicitly before the next step's contributors accumulate into it.
  // Every writer of arp.runoff is ADDITIVE -- the runoff-ratio arm below, the exfiltration-sink gather
  // (gather_sink_removed_to_zero), and FSM's own ponded `+= wtd`. So the carrier's lifecycle is one clean loop:
  // zero here -> += all contributors -> one FSM consumes -> zero. This replaces the prior scheme (the arm
  // OVERWRITING, relying on FSM to leave exactly 0), which left stale runoff on the runoff_ratio-off / rech<=0
  // paths where the overwrite was skipped.
  if (mpi_rank == 0) arp.runoff.setAll(0);

  // Set recharge for the NEXT step. Adjust evaporation where there is surface water, and ADD the runoff-ratio
  // runoff (runoff_ratio*rech) into the freshly-zeroed carrier for the next step's FSM. Serial rank-0 form when
  // the recharge is not distributed; otherwise distributed over owned cells.
  PetscMPIInt rech_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rech_rank);
  if (!distribute_recharge && rech_rank == 0) {
    const bool evap_taper = FanDarcyGroundwater::evap_taper_on();
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
        const double rr = arp.runoff_ratio(i) * arp.rech(i);
        arp.rech(i) -= rr;
        // HELD, not delivered. Same reasoning as the distributed path: this is a NOMINAL step's depth,
        // and the dt of the step it will be routed over is not final yet. Handed to FillSpillMerge --
        // and booked into col 20 -- at the top of the next coupling call, scaled by dt_committed.
        arp.runoff_nominal(i) = rr;
      }
    }
  }
  if (distribute_recharge) {
    distributed_recharge(params, arp, user_context, dmdapack);
    // -wtm_fsm_delta_source (FSM-delta-source): fold FSM's per-cell volume change onto the recharge source for the next
    // step. Added AFTER distributed_recharge so the runoff ratio does not take a second cut of water FSM has
    // already routed -- that ordering is load-bearing.
    //
    // KNOWN DEFECT (why this is still opt-in). SumDV is NOT zero: it is rr + s - spill, where rr is the
    // runoff-ratio share (genuinely new water, correctly booked here), s the exfiltrated excess (already
    // booked once as recharge, and again in total_surface_removed) and spill FSM's discharge to sea (already
    // booked in total_loss_to_ocean). Because total_added_recharge counts ALL of rech_dist
    // (transient_groundwater.cpp, set_starting_values), s is double-counted as an input and the spill is
    // double-subtracted, so the water budget stops closing: measured on the dome fixture at 18.5% of recharge
    // vs 1.09% for the overwrite path, with storage/evap/ocean fluxes identical to 5-6 figures. The physics
    // may be unaffected -- but the diagnostic that would prove it is what this breaks, so do not default this
    // on until rech_dist's solve role and total_added_recharge's budget role are separated.
    //
    // Also unverified: keeping the pre-FSM table as the step baseline leaves above-surface water in the
    // column for one more step, where the exfiltration rule sees it while this term also removes it.
    if (fsm_delta_source)
      accumulate_into_owned(user_context, fsm_delta_r0.data(), dmdapack.rech_dist);
    FanDarcyGroundwater::gather_wtd_to_all(params, arp, user_context, dmdapack);
    // The runoff-ratio share is NO LONGER handed over here. runoff_dist is left holding the NOMINAL
    // depth and is gathered at the TOP of the next call, scaled by the dt that step actually took --
    // see the note there.
  }
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
      // since the topography is changing. Only when FillSpillMerge will actually use it -- see the
      // equilibrium site for why the guard matters beyond saving the work.
      if (params.fsm_on)
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
  if ((params.cycles_done % params.save_nreport_interval) == 0) {
    // Save the output every "save_nreport_interval" reports, under a new filename
    // so we can compare how the water table has changed through time.
    // wtd is fully assembled on all ranks by FanDarcyGroundwater::update; rank 0 writes.
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      arp.wtd.setNoData(-9999);
      arp.wtd.saveGDAL(snapshot_filename(params));
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


  // These iterations refer to how many times to repeat the time step within the groundwater
  // portion of code before running FSM. For example, 1 year GW then FSM could also be run as
  // 2x 6 months GW then FSM.
  // Load the per-cycle solve inputs from rank-0 arp into the distributed carriers:
  // the wtd carrier (starting_wtd, advanced in place by the report's steps) and the
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

  // FIRST CYCLE ONLY: hand the INITIAL runoff split to the carrier the handoff reads.
  //
  // irf.cpp's initialisation already performs the split (`rr = runoff_ratio*rech; arp.runoff += rr;
  // arp.rech -= rr`), but it deposits the share in arp.runoff -- the rank-0 FillSpillMerge carrier --
  // which couple_surface_and_recharge ZEROES before the handoff ever reads it. So the first interval's
  // runoff was subtracted from recharge and then wiped: delivered to nobody with FSM off, and
  // delivered but never BOOKED with FSM on. Measured on a strictly-positive-recharge fixture, col 20
  // came out at exactly (N-1)/N of its true value -- 3/4, 1/2, 0/1 at 4, 2 and 1 cycles.
  //
  // Moving it into runoff_dist / runoff_nominal and clearing arp.runoff makes the handoff the SINGLE
  // path that both delivers and books the routed share, in every configuration. Splitting the
  // difference between two paths is what produced the off-by-one.
  if (params.cycles_done == 0 && params.runoff_ratio_on) {
    scatter_into_owned(user_context, arp.runoff.data(), dmdapack.runoff_dist);
    if (mpi_rank == 0) {
      arp.runoff_nominal = arp.runoff;  // the serial (rank-0) recharge path reads this one
      arp.runoff.setAll(0);             // consumed above; must not also reach FSM directly
    }
  }

  // Reset the per-STEP sink-removal accumulator (taper 1 / exfiltration): update() sums the removed depth for one
  // step, then couple_surface_and_recharge hands it to FSM. Zeroed before EACH step (FSM now runs every step),
  // so every step starts fresh; a harmless no-op when no sink/exfiltration is active (stays 0).
  const auto zero_sink = [&]() {
    const auto [xs, ys, xm, ym] = get_corners(user_context.da);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        dmdapack.sink_removed_dist[j][i] = 0.0;
  };
  // Per-report wall-time accumulators, summed from the timers around each GW step and each FSM step below.
  double gw_seconds = 0.0, fsm_seconds = 0.0;

  if (user_context.use_dt_adaptive) {
    // Adaptive stepping covers the SAME cycle duration as the fixed loop would
    // (report_seconds), but with variable, error-controlled sub-steps chosen by
    // the controller in FanDarcyGroundwater::update (which mutates user_context.deltat to
    // the next proposed size). Clamp each step to the time remaining in the cycle so we
    // land exactly on the target. See benchmark/BDF2_ADAPTIVE_DESIGN.md.
    const double cycle_duration = params.report_seconds;
    double       t              = 0.0;
    int          nsteps         = 0;
    int          rejects        = 0;
    int          retries        = 0;
    while (t < cycle_duration * (1.0 - 1e-9) && nsteps < 1000000) {
      const double remaining = cycle_duration - t;
      if (user_context.deltat > remaining) user_context.deltat = remaining;
      const double dt_taken   = user_context.deltat;
      const double rech_snap  = arp.total_recharge_direct;    // roll back on a rejected step (non-converged
      const double ocean_snap = arp.total_loss_to_ocean_gw;  // OR too-inaccurate), as the continuation loop does
      zero_sink();
      richdem::Timer tgw_a;
      tgw_a.start();
      const int    its        = FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
      gw_seconds += tgw_a.lap();
      if (its < 0) {  // REJECT: update() shrank deltat and did NOT commit; retry the same step
        arp.total_recharge_direct  = rech_snap;
        arp.total_loss_to_ocean_gw = ocean_snap;
        rejects++;
        params.rejects_done++;
        if (++retries > user_context.dtc_max_retries)
          throw std::runtime_error("adaptive dt: step failed after max retries; -wtm_dt_tol too tight "
                                   "or the local stability ceiling is below the smallest tried dt.");
        continue;  // do NOT advance t
      }
      retries = 0;
      t += dt_taken;
      nsteps++;
      params.solves_done++;
      if (params.fsm_on)
        couple_surface_and_recharge(params, arp, user_context, dmdapack, deps, mpi_rank, distribute_recharge,
                                    fsm_seconds);
    }
    PetscPrintf(PETSC_COMM_WORLD, "adaptive dt: %d steps (%d rejected) to cover %g s (fixed would be %d)\n",
                nsteps, rejects, cycle_duration, params.report_steps);
  } else if (user_context.use_newton_continuation) {
    // Newton pseudo-transient continuation (equilibrium): march report_steps ACCEPTED steps with a
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
    while (accepted < params.report_steps) {
      const double rech_snap  = arp.total_recharge_direct;   // roll back on a rejected step
      const double ocean_snap = arp.total_loss_to_ocean_gw;
      const double dt_try     = user_context.deltat;
      zero_sink();
      richdem::Timer tgw_n;
      tgw_n.start();
      const int    its        = FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
      gw_seconds += tgw_n.lap();
      if (its < 0) {  // rejected (non-converged): restore accumulators, shrink dt, retry same step
        arp.total_recharge_direct = rech_snap;
        arp.total_loss_to_ocean_gw = ocean_snap;
        user_context.deltat        = dt_try * user_context.dtc_shrink;
        params.rejects_done++;
        if (++retries > user_context.dtc_max_retries)
          throw std::runtime_error("dt-continuation: step failed to converge after max retries; deltat too "
                                   "small or the guess is too far (lower -wtm_dtc_grow / raise -wtm_dtc_dt0).");
        continue;  // do NOT advance `accepted`
      }
      accepted++;
      params.solves_done++;
      retries = 0;
      // Grow Δt after an EASY step (converged in <= dtc_easy_iters), HOLD when hard (near the free-
      // boundary ceiling). NOTE: a residual/state-change SER controller (grow ∝ Δw_prev/Δw) was tried
      // and is WORSE here -- during the long drainage transient Δw is large and only slowly shrinking,
      // so SER holds Δt small and never advances; growing on solve-EASE advances far better. last_dh_max
      // is still tracked (below) as an equilibrium detector, not a step controller.
      if (its <= user_context.dtc_easy_iters) user_context.deltat *= user_context.dtc_grow;  // else HOLD
      if (user_context.deltat > user_context.dtc_dt_max) user_context.deltat = user_context.dtc_dt_max;
      if (params.fsm_on)
        couple_surface_and_recharge(params, arp, user_context, dmdapack, deps, mpi_rank, distribute_recharge,
                                    fsm_seconds);
    }
    PetscPrintf(PETSC_COMM_WORLD,
                "dt-continuation: deltat now %g s after this cycle; last max|Δw| = %g m (-> 0 at equilibrium).\n",
                user_context.deltat, user_context.last_dh_max);
  } else {
    int iter_count = 0;
    while (iter_count++ < params.report_steps) {
      zero_sink();
      richdem::Timer tgw;
      tgw.start();
      FanDarcyGroundwater::update(params, arp, user_context, dmdapack);
      gw_seconds += tgw.lap();
      params.solves_done++;
      if (params.fsm_on)
        couple_surface_and_recharge(params, arp, user_context, dmdapack, deps, mpi_rank, distribute_recharge,
                                    fsm_seconds);
    }
  }
  // fsm_off: assemble the water table + set recharge ONCE per report (FillSpillMerge is skipped inside the
  // helper). fsm_on already coupled (gather -> hand-off -> FSM -> scatter -> recharge) after every step above.
  if (!params.fsm_on)
    couple_surface_and_recharge(params, arp, user_context, dmdapack, deps, mpi_rank, distribute_recharge,
                                fsm_seconds);
  std::cerr << "t GW time (report) = " << gw_seconds << " s;  FSM time (report) = " << fsm_seconds << " s"
            << std::endl;
  if (params.fsm_on && mpi_rank == 0)
    std::cerr << "t FSM fullness (last step) = " << arp.fsm_n_full_depressions << " / "
              << arp.fsm_n_depressions << " depressions full" << std::endl;

  // Per-CYCLE convergence metric: max change in the (post-FSM) water table since the previous cycle. This
  // is the HONEST steady-state signal -- unlike the per-sub-step max|Δw|, it excludes the cosmetic within-
  // cycle oscillation at lake/shore free boundaries (which returns to the same value each cycle and so
  // over-reports non-convergence). Wired to -wtm_eq_tol in run(). (Distributed-recharge path: starting_wtd
  // is post-FSM here; the serial path measures pre-FSM, still a valid cycle-to-cycle change.)
  {
    const auto [pxs, pys, pxm, pym] = get_corners(user_context.da);
    PetscScalar **prevw;
    DMDAVecGetArray(user_context.da, user_context.prev_cycle_wtd, &prevw);
    double dw_local = 0.0, sq_local = 0.0;
    double dv_local = 0.0, sqv_local = 0.0;   // pure-water-depth (|S*Δwtd|) analogues, in m of water
    long   n_local = 0, above_local = 0;
    for (int j = pys; j < pys + pym; j++)
      for (int i = pxs; i < pxs + pxm; i++) {
        if (dmdapack.mask[j][i] != 0) {
          const double d = std::abs(dmdapack.starting_wtd[j][i] - static_cast<double>(prevw[j][i]));
          dw_local = std::max(dw_local, d);
          sq_local += d * d;
          // Pure-water depth = |ΔV|/area = S*|Δh| with the SECANT effective storativity (S*Δh ≡ water moved).
          // Deep low-S cells contribute ~0 even when their head swings metres -- the FV-consistent measure.
          const double S  = updateEffectiveStorativity(static_cast<double>(prevw[j][i]),
                                                        dmdapack.starting_wtd[j][i],
                                                        dmdapack.porosity_vec[j][i]);
          const double dv = d * S;
          dv_local = std::max(dv_local, dv);
          sqv_local += dv * dv;
          n_local++;
          // frac metric counts cells by PURE-WATER change (dv = |S*Δwtd|), not head, so eq_tol is a water depth
          if (user_context.eq_tol > 0.0 && dv > user_context.eq_tol) above_local++;
        }
        prevw[j][i] = dmdapack.starting_wtd[j][i];
      }
    // Aggregate the per-cycle change three ways so the equilibrium stop (-wtm_eq_metric) can pick: MAX
    // (worst cell), RMS (bulk), and the fraction of cells still exceeding eq_tol. All are cheap Allreduces.
    double gmax = 0.0, gsq = 0.0, gvmax = 0.0, gvsq = 0.0;
    long   gn = 0, gabove = 0;
    MPI_Allreduce(&dw_local, &gmax, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    MPI_Allreduce(&sq_local, &gsq, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&dv_local, &gvmax, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    MPI_Allreduce(&sqv_local, &gvsq, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&n_local, &gn, 1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&above_local, &gabove, 1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);
    user_context.last_cycle_dw        = gmax;
    user_context.last_cycle_rms       = (gn > 0) ? std::sqrt(gsq / (double)gn) : 0.0;
    user_context.last_cycle_fracabove = (gn > 0) ? (double)gabove / (double)gn : 0.0;
    user_context.last_cycle_dw_water  = gvmax;
    user_context.last_cycle_rms_water = (gn > 0) ? std::sqrt(gvsq / (double)gn) : 0.0;
    DMDAVecRestoreArray(user_context.da, user_context.prev_cycle_wtd, &prevw);
  }

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
  // ONLY when FillSpillMerge will use it. deps feeds nothing else -- label/final_label/flowdirs are
  // read by FSM alone -- so building it with surface water off was wasted work on every cycle of a
  // transient run, and worse: GetDepressionHierarchy THROWS "No OCEAN cells found" on an all-land
  // domain, so WTM could not run a closed basin at all even with the surface-water model switched off.
  // That is a legitimate configuration (and the only way to test the lateral flux operator's
  // conservation in isolation -- see tests/local_ledger arm B).
  if (deps_rank == 0 && params.fsm_on) {
    deps = dh::GetDepressionHierarchy<float, rd::Topology::D8>(
        arp.topo, arp.cell_area, arp.label, arp.final_label, arp.flowdirs);
  }

  while (params.cycles_done < params.total_reports) {
    update(params, arp, user_context, dmdapack, deps);
    // output.verbosity: quiet suppresses the per-cycle progress line (the equilibrium-reached line and
    // warnings/errors still print); normal/verbose show it.
    if (params.verbosity != "quiet")
      PetscPrintf(PETSC_COMM_WORLD,
                  "cycle %d: per-cycle |Δwtd| max=%g rms=%g frac>tol=%.4f m  |S·Δwtd| max=%.4g rms=%.4g mm-water  "
                  "[within-cycle max|Δw| = %g m, %d cells>1mm]\n",
                  params.cycles_done, user_context.last_cycle_dw, user_context.last_cycle_rms,
                  user_context.last_cycle_fracabove,
                  1000.0 * user_context.last_cycle_dw_water, 1000.0 * user_context.last_cycle_rms_water,
                  user_context.last_dh_max, user_context.last_dh_nflicker);
    // Convergence-based early stop (opt-in via -wtm_eq_tol): stop once the PER-CYCLE water-table change
    // stays below eq_tol for two consecutive cycles -- the equilibrium auto-stop, on EVERY spin-up pathway.
    // Uses the per-cycle metric (not the per-sub-step max|Δw|), so the cosmetic within-cycle lake/shore
    // flicker cannot hold the run hostage. Each cycle of the fixed-dt march AND the adaptive-dt controller
    // spans a FIXED physical time (report_seconds), so a small per-report change genuinely means "steady",
    // not "tiny step" -- both are trustworthy directly. The Newton dt-CONTINUATION path is the sole
    // exception: one "cycle" there is a single variable-dt step, so a small change is only meaningful once
    // dt has ramped near its ceiling.
    const bool settle_trustworthy =
        user_context.use_newton_continuation
            ? (user_context.deltat >= 0.5 * user_context.dtc_dt_max)
            : true;
    if (user_context.eq_tol > 0.0 && settle_trustworthy) {
      // -wtm_eq_metric selects how the per-cycle change is judged against eq_tol: max (worst cell, strict),
      // rms (bulk), or frac (converged when < eq_frac of cells still exceed eq_tol -- robust to a slow
      // handful of deep cells; see the oscillation diagnosis in benchmark/adaptive_dt).
      bool        converged;
      const char* mname;
      // All metrics judge the per-cycle change in PURE-WATER DEPTH (|S*Δwtd|, m of water): deep low-storativity
      // cells (huge head swing, ~zero water moved) cannot pin the stop, so it is FV-consistent across cc and tr.
      // eq_tol is a WATER depth (default 0.001 = 1 mm). Raw head (last_cycle_dw/_rms) is still printed below.
      if (user_context.eq_metric == 1) {
        converged = user_context.last_cycle_rms_water < user_context.eq_tol;   mname = "rms";
      } else if (user_context.eq_metric == 2) {
        converged = user_context.last_cycle_fracabove < user_context.eq_frac;  mname = "frac";
      } else {
        converged = user_context.last_cycle_dw_water < user_context.eq_tol;    mname = "max";
      }
      if (converged) {
        if (++user_context.settled_count >= 2) {
          PetscPrintf(PETSC_COMM_WORLD,
                      "equilibrium reached (%s metric): max=%g rms=%g frac>tol=%.4f water(max/rms)=%.4g/%.4g mm "
                      "(eq_tol=%g, eq_frac=%g) for 2 cycles; stopping at cycle %d of %d.\n",
                      mname, user_context.last_cycle_dw, user_context.last_cycle_rms, user_context.last_cycle_fracabove,
                      1000.0 * user_context.last_cycle_dw_water, 1000.0 * user_context.last_cycle_rms_water,
                      user_context.eq_tol, user_context.eq_frac, params.cycles_done, params.total_reports);
          break;
        }
      } else {
        user_context.settled_count = 0;
      }
    }
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
    arp.wtd.saveGDAL(snapshot_filename(params));
  }

  textfile.close();

  delete user_context.full_grid_gather;  // destroys PETSc scatter/vecs; must precede PetscFinalize
  user_context.full_grid_gather = nullptr;
  VecDestroy(&user_context.wtd_global);
  VecDestroy(&user_context.rech_source);
  VecDestroy(&user_context.runoff_dist_vec);
  VecDestroy(&user_context.sink_removed_dist_vec);
  VecDestroy(&user_context.exfiltration_vec);

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
  VecDestroy(&user_context.fringe_width_vec);
  VecDestroy(&user_context.prev_cycle_wtd);
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
  VecDestroy(&user_context.starting_wtd_local);
  VecDestroy(&user_context.tr_ygamma);  // TR-BDF2 (no-op if unallocated)
  VecDestroy(&user_context.tr_expl);
  VecDestroy(&user_context.tr_exfil_stage1);  // TR-BDF2 x active-set (lazily allocated)
  VecDestroy(&user_context.tr_fwork);
  VecDestroy(&user_context.tr_head_old);  // TR-BDF2 step-flux quadrature (lazily allocated)

  // Picard path (nullptr / no-op when -wtm_picard was not set).
  MatDestroy(&user_context.picard_A);
  VecDestroy(&user_context.picard_r);
  VecDestroy(&user_context.starting_wtd_prev);  // BDF2 history (nullptr / no-op otherwise)
}

// Feed a YAML value to a PETSc option ONLY if the user did not already pass that flag on the CLI, so an
// explicit -flag always wins over the config file (precedence: CLI > YAML config > default). PETSc copies the
// value into its options DB immediately, so passing a temporary's c_str() is safe.
static void set_opt_if_unset(const char* flag, const char* val) {
  PetscBool has = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, flag, &has);
  if (!has) PetscOptionsSetValue(nullptr, flag, val);
}

// Phase 2b bridge: translate the CLI-flag-backed config sections (run.equilibrium_stop, boundaries,
// transmissivity background, solver, dev) into PETSc options, and parallel.threads_per_rank into a direct
// omp_set_num_threads call -- so the existing -wtm_*/-snes_* parsing (CreateSNES / transient_groundwater)
// picks them up unchanged. MUST run AFTER PetscInitialize (the options DB must exist) and before the SNES is
// built. Lives here (not parameters.cpp) so the PETSc-free dephier.x stays PETSc-free. TODO (Phase 2b cont.):
// evaporation et_sigmoid/extinction_depth, surface_water.collection.sink; output.verbosity/if_exists are NEW
// behavior (no existing flag).
void apply_config_petsc_options(const std::string& config_file) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_file);
  } catch (...) {
    return;
  }
  if (!root.IsMap()) return;

  // run.equilibrium_stop -> -wtm_eq_tol / -wtm_eq_metric / -wtm_eq_frac
  if (auto n = root["run"]["equilibrium_stop"]["tol"])    set_opt_if_unset("-wtm_eq_tol", n.as<std::string>().c_str());
  if (auto n = root["run"]["equilibrium_stop"]["metric"]) set_opt_if_unset("-wtm_eq_metric", n.as<std::string>().c_str());
  if (auto n = root["run"]["equilibrium_stop"]["frac"])   set_opt_if_unset("-wtm_eq_frac", n.as<std::string>().c_str());

  // boundaries.land -> -wtm_land_boundary (translate the prototype value)
  if (auto n = root["boundaries"]["land"]) {
    const std::string b = n.as<std::string>();
    set_opt_if_unset("-wtm_land_boundary", (b == "dirichlet_sea_level") ? "dirichlet" : "neumann_toposlope");
  }

  // transmissivity.additive_background_transmissivity -> -wtm_T_bedrock
  if (auto n = root["transmissivity"]["additive_background_transmissivity"])
    set_opt_if_unset("-wtm_T_bedrock", n.as<std::string>().c_str());

  // evaporation.et_sigmoid (the always-on soil<->open-water ET transition) + extinction_depth. The taper
  // on/off toggles (-wtm_evap_taper / -wtm_extinction) stay default-on; only the parameters are exposed.
  if (auto n = root["evaporation"]["et_sigmoid"]["wtd_center"])
    set_opt_if_unset("-wtm_evap_taper_wtdc", n.as<std::string>().c_str());
  if (auto n = root["evaporation"]["et_sigmoid"]["logistic_width"])
    set_opt_if_unset("-wtm_evap_taper_s", n.as<std::string>().c_str());
  if (auto n = root["evaporation"]["extinction_depth"])
    set_opt_if_unset("-wtm_extinction_depth", n.as<std::string>().c_str());

  // solver
  if (auto n = root["solver"]["method"]) {
    const std::string m = n.as<std::string>();
    if (m == "picard")      set_opt_if_unset("-wtm_picard", "true");
    else if (m == "newton") set_opt_if_unset("-wtm_newton", "true");
    // "anderson" = default (no flag)
  }
  if (auto n = root["solver"]["tolerance"]) set_opt_if_unset("-snes_stol", n.as<std::string>().c_str());
  if (auto n = root["solver"]["max_iterations"]) {
    const std::string v = n.as<std::string>();
    if (v != "auto") set_opt_if_unset("-snes_max_it", v.c_str());
  }
  if (auto n = root["solver"]["time_integration"]) {
    const std::string t = n.as<std::string>();
    if (t == "tr-bdf2")   set_opt_if_unset("-wtm_tr_bdf2", "true");
    else if (t == "bdf2") set_opt_if_unset("-wtm_bdf2_on_V", "true");
    // "backward-euler" = default (no flag)
  }
  if (auto n = root["solver"]["adaptive_dt"]) { if (n.as<bool>()) set_opt_if_unset("-wtm_dt_adaptive", "true"); }
  if (auto n = root["solver"]["dt_max"]) {
    const std::string v = n.as<std::string>();
    if (v != "auto")
      set_opt_if_unset("-wtm_dtc_dt_max", std::to_string(parse_time_seconds(v, "solver.dt_max")).c_str());
  }
  if (auto n = root["solver"]["water_volume_timestep_error_tol"]) {
    const std::string v = n.as<std::string>();  // adaptive per-step WATER (volume) local-error tol; auto = track eq_tol
    if (v != "auto") set_opt_if_unset("-wtm_dt_tol", v.c_str());
  }
  if (auto n = root["solver"]["t_bar"])   { if (n.as<bool>()) set_opt_if_unset("-wtm_Tbar", "true"); }
  if (auto n = root["solver"]["storage"]) { if (n.as<std::string>() == "volume") set_opt_if_unset("-wtm_volume_storage", "true"); }

  // dev
  if (auto n = root["dev"]["active_set"])                      { if (n.as<bool>()) set_opt_if_unset("-wtm_active_set", "true"); }
  if (auto n = root["dev"]["allow_aboveground_water_columns"]) { if (n.as<bool>()) set_opt_if_unset("-wtm_dev_allow_aboveground_water_columns", "true"); }
  if (auto n = root["dev"]["padded_dirichlet"])               { if (n.as<bool>()) set_opt_if_unset("-wtm_dev_padded_dirichlet", "true"); }

  // surface_water.collection.sink (legacy band-sink parameters; effective only with collection.method: legacy)
  if (auto s = root["surface_water"]["collection"]["sink"]) {
    if (auto m = s["qmax"])             set_opt_if_unset("-wtm_surface_sink_qmax", m.as<std::string>().c_str());
    if (auto m = s["width"])            set_opt_if_unset("-wtm_surface_sink_width", m.as<std::string>().c_str());
    if (auto m = s["fringe_source"])    set_opt_if_unset("-wtm_fringe_source", m.as<std::string>().c_str());
    if (auto m = s["fringe_cap"])       { if (!m.IsNull()) set_opt_if_unset("-wtm_fringe_cap", m.as<std::string>().c_str()); }
    if (auto m = s["fringe_ksat_coef"]) { if (!m.IsNull()) set_opt_if_unset("-wtm_fringe_ksat_coef", m.as<std::string>().c_str()); }
    if (auto m = s["fringe_length"])    { if (!m.IsNull()) set_opt_if_unset("-wtm_fringe_length", m.as<std::string>().c_str()); }
  }

  // output.verbosity: verbose -> per-solve PETSc monitors. (quiet-level suppression of WTM's own per-cycle
  // lines is TODO -- it needs gating in run(), not a PETSc flag; normal = the current default.)
  if (auto n = root["output"]["verbosity"]) {
    if (n.as<std::string>() == "verbose") {
      set_opt_if_unset("-snes_monitor", "");
      set_opt_if_unset("-snes_converged_reason", "");
    }
  }

  // parallel.threads_per_rank -> omp_set_num_threads (item 6; a direct call, not a PETSc option)
#ifdef _OPENMP
  if (auto n = root["parallel"]["threads_per_rank"]) {
    const int t = n.as<int>();
    if (t > 0) omp_set_num_threads(t);
  }
#endif
}

// output.directory management: give each run its own subdirectory so outputs never clobber. When
// output.directory is empty the LEGACY behavior is kept (outfile_prefix / run_log are literal paths).
// Otherwise resolve a run directory per output.if_exists and rewrite params.outfile_prefix / textfilename to
// live inside it. Rank-0 only (output is gathered to and written by rank 0); must run before any output.
static std::string resolve_output_directory(Parameters& params) {
  namespace fs = std::filesystem;
  if (params.output_directory.empty()) return "";  // legacy: use outfile_prefix / run_log as-is

  fs::path parent = params.output_directory;
  fs::path run_dir;
  if (params.if_exists == "overwrite") {
    run_dir = parent;
    fs::create_directories(run_dir);
  } else if (params.if_exists == "error") {
    if (fs::exists(parent) && !fs::is_empty(parent))
      throw std::runtime_error("output.directory '" + parent.string() + "' exists and is not empty (if_exists: error)");
    run_dir = parent;
    fs::create_directories(run_dir);
  } else {  // "increment" (default): parent/run<NNN>_<timestamp>/
    fs::create_directories(parent);
    int next = 0;
    for (const auto& e : fs::directory_iterator(parent)) {
      const std::string name = e.path().filename().string();
      if (name.rfind("run", 0) == 0 && name.size() > 3 && std::isdigit(static_cast<unsigned char>(name[3]))) {
        const int n = std::atoi(name.c_str() + 3);
        if (n >= next) next = n + 1;
      }
    }
    std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H%M%S", std::localtime(&t));
    char nnn[8];
    std::snprintf(nnn, sizeof(nnn), "%03d", next);
    run_dir = parent / ("run" + std::string(nnn) + "_" + ts);
    fs::create_directories(run_dir);
    // 'latest' symlink -> the newest run dir (relative target, best-effort)
    std::error_code ec;
    fs::remove(parent / "latest", ec);
    fs::create_directory_symlink(run_dir.filename(), parent / "latest", ec);
  }

  params.outfile_prefix = (run_dir / params.outfile_prefix).string();
  params.textfilename   = (run_dir / params.textfilename).string();
  PetscPrintf(PETSC_COMM_WORLD, "output: run directory = %s\n", run_dir.string().c_str());
  return run_dir.string();
}

// Write provenance.yaml into the run directory (rank 0): when + where it ran, the build's git commit/state,
// PETSc version, and the exact command line. Best-effort; a failure to write is not fatal.
static void write_provenance(const std::string& run_dir, int argc, char** argv) {
  std::ofstream p((std::filesystem::path(run_dir) / "provenance.yaml").string());
  if (!p) return;
  std::time_t t = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  PetscMPIInt nranks = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &nranks);
  std::string cmd;
  for (int i = 0; i < argc; i++) {
    cmd += argv[i];
    if (i + 1 < argc) cmd += ' ';
  }
  p << "run:\n";
  p << "  started: " << ts << "\n";
  p << "  hostname: " << host << "\n";
  p << "  mpi_ranks: " << nranks << "\n";
  p << "build:\n";
  p << "  git_commit: " << wtm_git_commit() << "\n";
  p << "  git_state: " << wtm_git_state() << "\n";
  p << "  petsc_version: " << PETSC_VERSION_MAJOR << "." << PETSC_VERSION_MINOR << "." << PETSC_VERSION_SUBMINOR
    << "\n";
  p << "command: " << cmd << "\n";
  p << "config_file: " << (argc > 1 ? argv[1] : "") << "\n";
  // TODO: also dump the fully-resolved config as run.
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

  // Phase 2b: fold the CLI-flag-backed config sections (solver / dev / boundaries / equilibrium_stop /
  // transmissivity background / parallel.threads_per_rank) into the PETSc options DB, so the existing
  // -wtm_*/-snes_* parsing reads them. Must run after PetscInitialize; explicit CLI flags still override.
  apply_config_petsc_options(argv[1]);

  // output.directory: resolve the per-run output subdirectory (rank 0 only; rewrites params.outfile_prefix /
  // textfilename to live inside it). Must precede any output writing.
  {
    PetscMPIInt out_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &out_rank);
    if (out_rank == 0) {
      const std::string run_dir = resolve_output_directory(params);
      if (!run_dir.empty()) write_provenance(run_dir, argc, argv);
    }
  }

  initialise(params, arp, user_context);

  // Echo the RESOLVED configuration (rank 0 only) so the run log records what the run actually used --
  // defaults applied, geometry derived from the geotransform, output paths rewritten by output.directory.
  // Placed after initialise() because report_steps / total_reports / the grid geometry are set there.
  {
    PetscMPIInt cfg_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &cfg_rank);
    if (cfg_rank == 0 && params.verbosity != "quiet") params.print();
  }

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
