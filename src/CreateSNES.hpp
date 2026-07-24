#include "dmda_gather.hpp"
#include "parameters.hpp"

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscsnes.h>

struct AppCtx {
  PetscReal cellsize_NS_squared;
  PetscReal deltat;
  SNES snes               = nullptr;
  DM da                   = nullptr;
  Vec x                   = nullptr;  // Solution vector
  Vec b                   = nullptr;  // RHS vector
  Vec cellsize_EW_squared = nullptr;
  Vec fdepth_vec          = nullptr;
  Vec ksat_vec            = nullptr;
  Vec mask                = nullptr;
  Vec topo_vec            = nullptr;
  Vec rech_vec            = nullptr;
  Vec porosity_vec        = nullptr;
  Vec starting_wtd        = nullptr;

  // Distributed forcing fields for the recharge computation. Scattered from
  // rank-0 arp at init (populate_DMDA_array_pack) so recharge can be computed over
  // each rank's owned cells rather than serially on rank 0. See DISTRIBUTED_ARP_DESIGN.md.
  Vec precip_vec          = nullptr;
  Vec evap_vec            = nullptr;
  Vec open_water_evap_vec = nullptr;
  Vec runoff_ratio_vec    = nullptr;

  // Local ghost vectors for fields accessed at neighbor indices in FormFunctionLocal
  Vec topo_local   = nullptr;
  Vec fdepth_local = nullptr;
  Vec ksat_local   = nullptr;
  Vec T_local      = nullptr;  // scratch: 1/T, computed over ghost range each F eval

  // Scratch global vector + reusable gather for assembling the full wtd field
  // from the distributed solve (see FanDarcyGroundwater::update). Owned by the
  // context; destroyed in finalise() before PetscFinalize.
  Vec wtd_global                       = nullptr;
  DMDAFullGridGather* full_grid_gather = nullptr;

  // Distributed per-cycle recharge source (populated from arp.rech each cycle),
  // so the solve loop reads recharge from DMDA-owned data rather than arp.rech.
  Vec rech_source = nullptr;

  // Extract global vectors from DM; then duplicate for remaining
  // vectors that are the same types
  void make_global_vectors() {
    DMCreateGlobalVector(da, &x);
    VecDuplicate(x, &b);
    VecDuplicate(x, &cellsize_EW_squared);
    VecDuplicate(x, &fdepth_vec);
    VecDuplicate(x, &ksat_vec);
    VecDuplicate(x, &mask);
    VecDuplicate(x, &topo_vec);
    VecDuplicate(x, &rech_vec);
    VecDuplicate(x, &porosity_vec);
    VecDuplicate(x, &starting_wtd);
    VecDuplicate(x, &wtd_global);
    VecDuplicate(x, &rech_source);
    VecDuplicate(x, &precip_vec);
    VecDuplicate(x, &evap_vec);
    VecDuplicate(x, &open_water_evap_vec);
    VecDuplicate(x, &runoff_ratio_vec);
  }

  void make_local_vectors() {
    DMCreateLocalVector(da, &topo_local);
    DMCreateLocalVector(da, &fdepth_local);
    DMCreateLocalVector(da, &ksat_local);
    DMCreateLocalVector(da, &T_local);
  }
};

void InitialiseSNES(AppCtx& user_context, Parameters& params);
