std::tuple<PetscInt, PetscInt, PetscInt, PetscInt> get_corners(const DM da) {
  PetscInt xs, ys, xm, ym;
  DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr);
  return {xs, ys, xm, ym};
}

// Populate the static global vecs from arp. mask and porosity are scattered from rank-0 arp (float)
// so those arp arrays need not exist on non-root ranks; cellsize_EW_squared is derived from the 1-D
// Class-C array cellsize_e_w_metres (present on all ranks) by an owned-range copy. Writes the global
// vecs directly, so this must be called BEFORE DMDA_Array_Pack holds them.
void populate_DMDA_array_pack(AppCtx& user_context, ArrayPack& arp) {
  user_context.full_grid_gather->scatterFromZero(arp.land_mask.data(), user_context.mask);
  user_context.full_grid_gather->scatterFromZero(arp.porosity.data(), user_context.porosity_vec);

  // Distributed forcing fields for the (soon-to-be-distributed) recharge computation.
  // Scattered from rank-0 arp so recharge need not be computed serially on rank 0.
  user_context.full_grid_gather->scatterFromZero(arp.precip.data(), user_context.precip_vec);
  user_context.full_grid_gather->scatterFromZero(arp.evap.data(), user_context.evap_vec);
  user_context.full_grid_gather->scatterFromZero(arp.open_water_evap.data(), user_context.open_water_evap_vec);
  user_context.full_grid_gather->scatterFromZero(arp.runoff_ratio.data(), user_context.runoff_ratio_vec);

  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** cellsize;
  DMDAVecGetArray(user_context.da, user_context.cellsize_EW_squared, &cellsize);
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      cellsize[j][i] = arp.cellsize_e_w_metres[j] * arp.cellsize_e_w_metres[j];
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.cellsize_EW_squared, &cellsize);
}

// Populate the global topo/fdepth/ksat vecs from rank-0 arp and scatter to local ghost vectors.
// Sourcing from rank 0 (via the natural-ordering scatter) rather than an owned copy of a replicated
// arp is what allows arp.topo/fdepth/ksat to be dropped on non-root ranks. topo/ksat are float,
// fdepth is double; scatterFromZero converts to PetscScalar. Must be called while these global vecs
// are NOT under DMDAVecGetArray (DMDA_Array_Pack does not hold them, by design).
void scatter_static_fields(AppCtx& user_context, ArrayPack& arp) {
  user_context.full_grid_gather->scatterFromZero(arp.topo.data(), user_context.topo_vec);
  user_context.full_grid_gather->scatterFromZero(arp.fdepth.data(), user_context.fdepth_vec);
  user_context.full_grid_gather->scatterFromZero(arp.ksat.data(), user_context.ksat_vec);

  // The DMDA's internal PetscSF is shared across all GlobalToLocal operations on the same DM.
  // Overlapping Begin calls (Begin A, Begin B, End A, End B) confuse the SF state machine;
  // each pair must be completed sequentially.
  DMGlobalToLocalBegin(user_context.da, user_context.topo_vec,   INSERT_VALUES, user_context.topo_local);
  DMGlobalToLocalEnd(user_context.da, user_context.topo_vec,     INSERT_VALUES, user_context.topo_local);
  DMGlobalToLocalBegin(user_context.da, user_context.fdepth_vec, INSERT_VALUES, user_context.fdepth_local);
  DMGlobalToLocalEnd(user_context.da, user_context.fdepth_vec,   INSERT_VALUES, user_context.fdepth_local);
  DMGlobalToLocalBegin(user_context.da, user_context.ksat_vec,   INSERT_VALUES, user_context.ksat_local);
  DMGlobalToLocalEnd(user_context.da, user_context.ksat_vec,     INSERT_VALUES, user_context.ksat_local);
}
