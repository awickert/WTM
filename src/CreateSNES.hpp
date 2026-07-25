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

  // --- Semi-implicit Picard path (gated behind -wtm_picard; default off) ---
  // The row-scaled operator A(x) uses centre-cell storativity (so porosity and
  // starting_wtd are read owned-only, no ghosts); only the harmonic-mean T needs
  // neighbor heads, which come from ghost-scattering the iterate x each assembly.
  // See PICARD_MATH.md.
  bool use_picard = false;
  Mat picard_A    = nullptr;  // assembled SPD operator A(x) (also the GAMG preconditioner)
  Vec picard_r    = nullptr;  // residual work vector for SNESSetPicard

  // --- BDF2 time integration (gated behind -wtm_bdf2; implies the Picard path) ---
  // Second-order backward differentiation: (3h^{n+1} - 4h^n + h^{n-1})/(2dt) = RHS.
  // vs backward Euler this doubles the diffusion coefficient (dt->2dt) and the storage
  // diagonal (S_c->3S_c), and the RHS becomes S_c*(4h^n - h^{n-1} + 2*rech). Needs the
  // previous-previous head h^{n-1} = starting_wtd_prev + topo (centre only, no ghosts);
  // step 0 has no h^{n-1} so it bootstraps with backward Euler. See BDF2_ADAPTIVE_DESIGN.md.
  bool   use_bdf2          = false;
  bool   bdf2_have_history = false;  // false until the first step has produced an h^{n-1}
  double bdf2_prev_dt      = 0.0;    // Δt_{n-1}: previous step size, for the variable-step
                                     // ratio ω = Δt_n/Δt_{n-1} (ω=1 when Δt is constant, i.e.
                                     // fixed-step BDF2). Set to the initial deltat at init.
  Vec    starting_wtd_prev = nullptr;  // h^{n-1} carrier (wtd), owned-range

  // --- Adaptive time stepping (gated behind -wtm_dt_adaptive; implies BDF2 -> Picard) ---
  // Forward (no-reject) controller: after each step the local error is estimated from the
  // deviation of the solution from a linear extrapolation of the history (~O(dt^2)), and the
  // NEXT step size is set to hold that near dt_tol (metres). No accept/reject retry (which
  // would double-count the per-step recharge/ocean accumulators); a too-large step is simply
  // followed by a smaller one. See BDF2_ADAPTIVE_DESIGN.md.
  bool   use_dt_adaptive = false;
  double dt_tol          = 0.1;  // target max |h - linear-extrapolation| per step, metres

  // Scratch global vector + reusable gather for assembling the full wtd field
  // from the distributed solve (see FanDarcyGroundwater::update). Owned by the
  // context; destroyed in finalise() before PetscFinalize.
  Vec wtd_global                       = nullptr;
  DMDAFullGridGather* full_grid_gather = nullptr;

  // Distributed per-cycle recharge source (populated from arp.rech each cycle),
  // so the solve loop reads recharge from DMDA-owned data rather than arp.rech.
  Vec rech_source = nullptr;

  // Distributed per-cycle runoff (runoff_ratio * rech), computed alongside the
  // distributed recharge and gathered to rank-0 arp.runoff for the next FillSpillMerge
  // when runoff_ratio_on. Unused when runoff is off. See DISTRIBUTED_ARP_DESIGN.md (2c).
  Vec runoff_dist_vec = nullptr;

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
    VecDuplicate(x, &runoff_dist_vec);
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
