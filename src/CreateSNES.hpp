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
  // Conservative-FV per-row flux geometry factors (owned range; see GRID_CONVENTION.md).
  // cell area A_j = cellsize_NS_squared / geom_ew. E-W conductance = e * geom_ew; N/S = e * geom_{n,s}.
  Vec geom_ew_vec         = nullptr;
  Vec geom_n_vec          = nullptr;
  Vec geom_s_vec          = nullptr;
  Vec fdepth_vec          = nullptr;
  Vec ksat_vec            = nullptr;
  Vec mask                = nullptr;
  Vec topo_vec            = nullptr;
  Vec rech_vec            = nullptr;
  Vec porosity_vec        = nullptr;
  Vec fringe_width_vec    = nullptr;  // per-cell sink band width (capillary fringe); see -wtm_fringe_source
  Vec prev_cycle_wtd      = nullptr;  // post-FSM water table at the previous cycle (for the per-CYCLE convergence metric)
  double last_cycle_dw    = 1e30;     // max|wtd_cycleN - wtd_cycleN-1| over the domain (the honest steady-state metric)
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
  Vec mask_local   = nullptr;  // ghost mask, so ocean-outflow accounting can find land->ocean faces
                               // at rank boundaries. Scattered once at init (static within a run).
  Vec starting_wtd_local = nullptr;  // ghost w^n (previous-step wtd), for the -wtm_Tbar time-averaged
                                     // interblock transmissivity T̄ = (Φ(w^{n+1})-Φ(w^n))/(w^{n+1}-w^n):
                                     // a neighbour's T̄ needs its w^n, so w^n must be ghosted. Re-
                                     // scattered each solve in update() (w^n changes per step). Only
                                     // used on the -wtm_Tbar path.

  // --- Semi-implicit Picard path (gated behind -wtm_picard; default off) ---
  // The row-scaled operator A(x) uses centre-cell storativity (so porosity and
  // starting_wtd are read owned-only, no ghosts); only the harmonic-mean T needs
  // neighbor heads, which come from ghost-scattering the iterate x each assembly.
  // See PICARD_MATH.md.
  bool use_picard = false;
  Mat picard_A    = nullptr;  // assembled SPD operator A(x) (also the GAMG preconditioner)
  Vec picard_r    = nullptr;  // residual work vector for SNESSetPicard

  // --- Newton-Krylov path (gated behind -wtm_newton; default off) ---
  // Opt-in true Newton on the matrix-free residual (FormFunctionLocal): registers the analytic
  // Jacobian FormJacobianLocal (∂F/∂x of the conservative-FV flux + secant storativity + sink/evap
  // tangents). NON-symmetric (dT/dw) → GMRES, not CG. For cold-start equilibrium where the frozen-
  // operator solvers (Anderson/Picard) diverge from far; see benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  bool use_newton = false;

  // --- Newton dt-continuation (gated behind -wtm_dt_continuation; needs -wtm_newton) ---
  // Pseudo-transient continuation for EQUILIBRIUM from a far guess: start deltat small (so the storage
  // term S/deltat keeps the Jacobian diagonally dominant -- a large step from far overshoots into a
  // SINGULAR Jacobian), and grow it geometrically after each converged step so it reaches a near-steady
  // large dt as the state warms. deltat persists across cycles, so it ramps toward equilibrium. GROW-
  // ONLY (no reject/retry yet), so it must be ramped gently enough not to overshoot. The per-step
  // recharge is rescaled to rate*deltat in update() (rech_dist is baked at rate*params.deltat), so the
  // steady state is correct at ANY ramped dt (recharge and flux both scale with dt -> dt cancels at the
  // fixed point). See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  bool   use_newton_continuation = false;
  double dtc_grow                = 1.5;   // dt growth when a step converges EASILY (-wtm_dtc_grow)
  double dtc_shrink              = 0.25;  // dt shrink on a REJECTED (non-converged) step (-wtm_dtc_shrink)
  double dtc_dt_max              = 0.0;   // cap on deltat [s]; 0 => set from params in InitialiseSNES
  int    dtc_easy_iters          = 8;     // grow dt only if the step converged in <= this many Newton iters
                                          // (-wtm_dtc_easy_iters); otherwise hold dt (near the safe ceiling)
  int    dtc_max_retries         = 15;    // consecutive rejects before giving up (-wtm_dtc_max_retries)
  // Equilibrium detector: set by update() to the global max |w^{n+1}-w^n| this step; the continuation
  // reports it per cycle (→ 0 at steady state). A residual/state-change SER dt controller was tried and
  // is worse than growing on solve-ease (see WTM.cpp), so this is a diagnostic, not a step controller.
  double last_dh_max             = 0.0;
  int    last_dh_i               = -1;  // argmax (i,j) of last_dh_max: which land cell moves most (diagnostic)
  int    last_dh_j               = -1;
  int    last_dh_nflicker        = 0;   // # land cells with per-sub-step |Δw| > 1mm (within-cycle flicker diagnostic)
  // Opt-in convergence-based early stop (-wtm_eq_tol, metres; 0 = off): stop the cycle loop once the
  // PER-CYCLE water-table change (last_cycle_dw = max|wtd_N - wtd_{N-1}| over land, the honest steady-state
  // measure) stays below eq_tol for two consecutive cycles, instead of always running total_cycles. The
  // per-SUB-STEP max|Δw| (last_dh_max) is NOT used for the stop: at lake/shore free boundaries it carries a
  // cosmetic within-cycle flicker that returns to the same value each cycle, so gating on it would never
  // settle. Recommended value ~1e-2 m (the worst cell moves <~1 cm per ~1-yr cycle); the achievable floor
  // is set by that cosmetic lakeshore wiggle, so tighter tolerances need it excluded. Off by default ->
  // existing behaviour unchanged (auto-set to 1e-2 only under -wtm_stiff).
  double eq_tol                  = 0.0;
  int    settled_count           = 0;

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

  // Modeling options (metres, default 0): round the two C0 kinks in the depth-integrated
  // transmissivity, each independently -- -wtm_ksat_soilbottom_smoothing_width at -1.5 m (the
  // conductivity profile's constant->exponential-decay transition) and -wtm_ksat_surface_smoothing_width
  // at 0 m (the land surface). Both 0 keeps the exact production piecewise (C0) Fan S4/S6 form; a
  // positive width uses the smooth (C-inf) form in the Picard operator at that boundary. Physically
  // a sub-grid conductivity smoothing. Read directly from the options DB in FormPicardOperator (no
  // AppCtx flag). Note: smoothing does NOT by itself restore BDF2 order 2 -- the order-1 cause was
  // the storativity treatment; see BDF2_ADAPTIVE_DESIGN.md.

  // --- TR-BDF2 (-wtm_tr_bdf2; matrix-free Anderson only) ---
  // L-stable, strongly (monotone) damped 2nd-order-in-time alternative to BDF2-on-V, for the matrix-free
  // Anderson residual. One step = TWO staged implicit solves: (1) a trapezoidal sub-step to t+gamma*dt
  // giving the intermediate Y_gamma, then (2) a BDF2 sub-step from (w^n, Y_gamma) to w^{n+1}. gamma=2-sqrt2;
  // both stages share the diagonal (SDIRK). Self-starting (no w^{n-1}). Unlike plain BDF2 its stiff-mode
  // damping is monotone (not oscillatory), so it is the safe choice if BDF2 rings on Anderson near the
  // step ceiling. tr_stage selects the residual branch; tr_ygamma carries Y_gamma between the two solves;
  // tr_expl caches the explicit old-state (w^n) flux+removal that the trapezoidal stage needs. See
  // benchmark/TBAR_TIME_AVERAGING.md.
  bool use_tr_bdf2 = false;
  int  tr_stage    = 0;        // 0 = n/a; 1 = trapezoidal stage; 2 = BDF2 stage

  // --- Anderson-accelerated GAMG-Picard via nonlinear preconditioning (-wtm_aa_picard) ---
  // Outer solver = Anderson on the head-form residual (FormFunctionLocal, well-scaled -> Anderson-safe).
  // The GAMG-preconditioned Picard solve (FormPicardOperator/RHS) is attached as the OUTER's nonlinear
  // PRECONDITIONER (SNESGetNPC + SNESSetPicard), so each Anderson step is a full multigrid Picard update
  // that Anderson then mixes/damps. Combines GAMG's large-step power with Anderson's oscillation-damping,
  // avoiding the volume-form residual that plain Anderson diverges on. Uses the Picard operator (picard_A/
  // picard_r) for the NPC, but the MAIN residual is the matrix-free head form (use_picard stays false).
  // Experimental (Skunkworks). See benchmark/AA_PICARD.md.
  bool use_aa_picard = false;

  // --- Predictor-seeded initial guess (-wtm_predict_guess) ---
  // Seed the SNES initial guess (hence the iteration-1 T̄ coefficient) with the 2nd-order history
  // extrapolation w^{n+1} ≈ w^n + ω(w^n − w^{n-1}) (ω = Δt/Δt_{n-1}) instead of w^n. Without it, the
  // iteration-1 T̄ collapses to the instantaneous T(w^n) (Δwtd→0) -- the frozen start-of-step value we
  // are trying to beat -- so T̄'s before-and-after advantage is unrealized on the first evaluation.
  // Guarded (no surface crossing in the guess; capped magnitude). Needs the w^{n-1} history carrier
  // (starting_wtd_prev), so it also switches that on. Speed/robustness only (the root is unchanged).
  bool use_predict_guess = false;
  Vec  tr_ygamma   = nullptr;  // intermediate Y_gamma (owned range; storage is centre-only)
  Vec  tr_expl     = nullptr;  // dt*(N(w^n)/A_j + removal(w^n)) per owned land cell (explicit TR half)

  // BDF2-on-V (-wtm_bdf2_on_V; implies BDF2 -> Picard): discretize the nonlinear storage with the
  // 3-level BDF2 difference of the stored volume V ((3V^{n+1}-4V^n+V^{n-1})/2dt = flux), using the
  // TANGENT dV/dh on the operator diagonal -- instead of the 2-level backward-Euler secant
  // storativity, which caps the achieved order at 1. Restores genuine 2nd order, physics-preserving
  // (no fixed-point shift). See BDF2_ADAPTIVE_DESIGN.md.
  bool use_bdf2_on_V = false;

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

  // Distributed per-cycle water removed by the implicit sub-surface sink (taper 1):
  // depth (m) accumulated per owned cell across the cycle's sub-steps, gathered to
  // rank-0 arp.runoff so this cycle's FillSpillMerge routes it. This is the smooth,
  // order-preserving replacement for FSM's hard wtd>0 -> runoff handoff: the sink holds
  // wtd<=0 in the solve, so the exfiltrated water leaves *here* instead. Zero (a no-op
  // gather) when the sink is off. See SURFACE_SINK_DESIGN.md and issue #4.
  Vec sink_removed_dist_vec = nullptr;

  // Extract global vectors from DM; then duplicate for remaining
  // vectors that are the same types
  void make_global_vectors() {
    DMCreateGlobalVector(da, &x);
    VecDuplicate(x, &b);
    VecDuplicate(x, &cellsize_EW_squared);
    VecDuplicate(x, &geom_ew_vec);
    VecDuplicate(x, &geom_n_vec);
    VecDuplicate(x, &geom_s_vec);
    VecDuplicate(x, &fdepth_vec);
    VecDuplicate(x, &ksat_vec);
    VecDuplicate(x, &mask);
    VecDuplicate(x, &topo_vec);
    VecDuplicate(x, &rech_vec);
    VecDuplicate(x, &porosity_vec);
    VecDuplicate(x, &fringe_width_vec);
    VecDuplicate(x, &prev_cycle_wtd);
    VecSet(prev_cycle_wtd, 0.0);
    VecDuplicate(x, &starting_wtd);
    VecDuplicate(x, &wtd_global);
    VecDuplicate(x, &rech_source);
    VecDuplicate(x, &runoff_dist_vec);
    VecDuplicate(x, &sink_removed_dist_vec);
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
    DMCreateLocalVector(da, &mask_local);
    DMCreateLocalVector(da, &starting_wtd_local);  // ghost w^n for -wtm_Tbar (see field comment)
  }
};

void InitialiseSNES(AppCtx& user_context, Parameters& params);
