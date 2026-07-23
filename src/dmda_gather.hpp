#pragma once

#include <petscdm.h>
#include <petscdmda.h>

#include <vector>

// Reusable gather of a distributed 2D DMDA global vector into a full,
// row-major buffer with layout  index = j * Mx + i  (x fastest) -- the same
// layout used by richdem::Array2D and by the water-table reassembly in
// FanDarcyGroundwater::update.
//
// The natural-ordering vector and the scatter-to-rank-0 context are created
// once and reused on every call (they depend only on the DM's decomposition).
// The row-major ordering (index = j*Mx + i, x fastest) was verified to match
// the DMDA natural ordering independent of decomposition, for 1, 2, and 4 ranks
// on a non-square grid, and reproduces the previous zero-and-Allreduce layout
// bit-for-bit.
//
// This is the Phase-1 communication primitive for distributing ArrayPack: it
// gathers a distributed field to the full grid needed by FillSpillMerge and by
// GDAL output. See benchmark/DISTRIBUTED_ARP_DESIGN.md.
class DMDAFullGridGather {
 public:
  explicit DMDAFullGridGather(DM da) : da_(da) {
    DMDACreateNaturalVector(da_, &natural_);
    VecScatterCreateToZero(natural_, &to_zero_, &seq_on_zero_);
    DMDAGetInfo(da_, nullptr, &Mx_, &My_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr);
  }

  ~DMDAFullGridGather() {
    VecScatterDestroy(&to_zero_);
    VecDestroy(&seq_on_zero_);
    VecDestroy(&natural_);
  }

  DMDAFullGridGather(const DMDAFullGridGather&)            = delete;
  DMDAFullGridGather& operator=(const DMDAFullGridGather&) = delete;

  PetscInt width() const { return Mx_; }
  PetscInt height() const { return My_; }

  // Gather `global` (a DMDA-owned global vector) so that EVERY rank ends up with
  // the complete field in `full`, row-major (index = j*Mx + i). Preserves the
  // current fully-replicated model; Phase 2+ will switch callers that only need
  // the field on rank 0 to gatherToZero (below) to shed the broadcast.
  void gatherToAll(Vec global, std::vector<double>& full) {
    const PetscInt total = Mx_ * My_;
    full.assign(total, 0.0);
    gatherToZeroBuffer(global, full);
    MPI_Bcast(full.data(), total, MPI_DOUBLE, 0, PetscObjectComm((PetscObject)da_));
  }

  // Gather `global` onto rank 0 only. `full` is resized to Mx*My on rank 0 and
  // filled row-major; on other ranks `full` is left empty.
  void gatherToZero(Vec global, std::vector<double>& full) {
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)da_), &rank);
    if (rank == 0) {
      full.assign(Mx_ * My_, 0.0);
    } else {
      full.clear();
    }
    gatherToZeroBuffer(global, full);
  }

 private:
  // Core scatter: distributed global -> natural -> sequential-on-rank-0.
  // On rank 0, copies the sequential values into `full` (which the caller has
  // already sized to Mx*My). On other ranks does nothing to `full`.
  void gatherToZeroBuffer(Vec global, std::vector<double>& full) {
    DMDAGlobalToNaturalBegin(da_, global, INSERT_VALUES, natural_);
    DMDAGlobalToNaturalEnd(da_, global, INSERT_VALUES, natural_);
    VecScatterBegin(to_zero_, natural_, seq_on_zero_, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(to_zero_, natural_, seq_on_zero_, INSERT_VALUES, SCATTER_FORWARD);

    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)da_), &rank);
    if (rank == 0) {
      const PetscScalar* s;
      VecGetArrayRead(seq_on_zero_, &s);
      const PetscInt total = Mx_ * My_;
      for (PetscInt k = 0; k < total; k++) {
        full[k] = s[k];
      }
      VecRestoreArrayRead(seq_on_zero_, &s);
    }
  }

  DM da_;
  Vec natural_     = nullptr;
  Vec seq_on_zero_ = nullptr;
  VecScatter to_zero_ = nullptr;
  PetscInt Mx_ = 0, My_ = 0;
};
