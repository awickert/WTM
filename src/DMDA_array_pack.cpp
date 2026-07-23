std::tuple<PetscInt, PetscInt, PetscInt, PetscInt> get_corners(const DM da) {
  PetscInt xs, ys, xm, ym;
  DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr);
  return {xs, ys, xm, ym};
}

void populate_DMDA_array_pack(AppCtx& user_context, ArrayPack& arp, DMDA_Array_Pack& dmdapack) {
  // Get local array bounds
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);

  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      dmdapack.cellsize_EW_squared[j][i] = arp.cellsize_e_w_metres[j] * arp.cellsize_e_w_metres[j];
      dmdapack.mask[j][i]                = arp.land_mask(i, j);
      dmdapack.porosity_vec[j][i]        = arp.porosity(i, j);
    }
  }
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
