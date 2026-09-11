#include "transient_groundwater.hpp"
#include "add_recharge.hpp"
#include "tr_bdf2_coefficients.hpp"
#include "update_effective_storativity.hpp"

#include <omp.h>
#include <array>
#include <chrono>
#include <cctype>   // std::isspace for the coverage tag
#include <cstdlib>  // std::getenv for the coverage fingerprint
#include <fstream>  // coverage fingerprint append
#include <experimental/source_location>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscerror.h>
#include <petscsnes.h>

///////////////////////
// PRIVATE FUNCTIONS //
///////////////////////

namespace FanDarcyGroundwater {

void PETSC_CHECK(
    const PetscErrorCode err,
    const std::experimental::source_location location = std::experimental::source_location::current()) {
  if (err) {
    throw std::runtime_error(
        "Petsc exception: " + std::to_string(err) + " at " + location.file_name() + ":" +
        std::to_string(location.line()));
  }
}

// get corners of arrays for individual processors
std::tuple<PetscInt, PetscInt, PetscInt, PetscInt> get_corners(const DM da) {
  PetscInt xs, ys, xm, ym;
  PETSC_CHECK(DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  return {xs, ys, xm, ym};
}

// declare functions
static PetscErrorCode FormRHS(AppCtx*, DM, Vec);
static PetscErrorCode FormInitialGuess(AppCtx*, DM, Vec);
static PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscScalar**, PetscScalar**, AppCtx*);
// Retained but no longer dispatched (the Newton-Krylov path is disabled in update()); kept so a
// Newton solver can be rebuilt. [[maybe_unused]] silences the now-uncalled-function warning.
[[maybe_unused]] static PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscScalar**, Mat, Mat, AppCtx*);

// Semi-implicit Picard path (experimental; PICARD_MATH.md). Global SNES callbacks
// for SNESSetPicard: FormPicardRHS computes b(x), FormPicardOperator computes the
// SPD operator A(x). Gated behind solver.method: picard; default Anderson path unaffected.
static PetscErrorCode FormPicardRHS(SNES, Vec, Vec, void*);
static PetscErrorCode FormPicardOperator(SNES, Vec, Mat, Mat, void*);

// --- Background (bedrock) transmissivity floor (-wtm_T_bedrock) -----------------------------------
// The Fan et al. (2013) transmissivity T = fdepth·ksat·exp((wtd+1.5)/fdepth) is a fit to the WEATHERED,
// fractured shallow zone. Extrapolated to depth with the steep-terrain e-folding length (fdepth ~ 2.5 m)
// it predicts T dropping ~69 orders of magnitude over ~400 m and underflowing to EXACTLY 0 below ~-1864 m
// -- physically absurd (no crust loses 69 orders over 400 m) and the numerical root of the deep-cell
// operator singularity: cells below ~-90 m have conductance below machine-epsilon relative to the surface,
// so their operator rows go numerically zero -> the matrix is rank-deficient -> every inversion-based
// solver (Picard/GAMG, Newton, MUMPS) fails at large dt (matrix-free Anderson is immune but still
// stiffness-throttled). See benchmark/PICARD_STIFFNESS_POSTMORTEM.md, memory finding-operator-singularity.
//
// Real crust does NOT go to zero: it retains a small background permeability (Manning & Ingebritsen 1999,
// Rev. Geophys.: mean crustal k ~ 1e-14 m² at 1 km falling to ~1e-16..-17 at depth; K = k·ρg/μ ~ 1e-7 m/s
// down to ~1e-9..-10 for competent bedrock) down to the base of the active flow system (ultimately the
// brittle-ductile transition, ~10-15 km). Integrating that background layer gives a constant ADDITIVE
// transmissivity T_bedrock = K_bedrock · d_active. Adding it recovers the correct physics AND caps T's
// dynamic range: with T_bedrock = 1e-8 m²/s the range collapses from ~69 orders to ~3.7 (surface ~4.6e-5),
// crossover at ~-20 m (just below the weathered zone) -- singular operator becomes non-singular, well
// within double precision's ~13-order headroom. Because it is a CONSTANT additive term:
//   * T(wtd)         gains + T_bedrock                     (depthIntegratedTransmissivity[Smooth])
//   * Φ(wtd) = ∫T    gains + T_bedrock·wtd                 (dischargePotential; keeps T̄ = ΔΦ/Δwtd EXACT)
//   * dT/dwtd        is UNCHANGED (derivative of a constant is 0; dDepthIntegratedTransmissivityDwtd)
//   * d(1/T)/dwtd    changes only via the T in the denominator (dTransmissivityInverseDwtd)
// So it composes with T̄ and every solver as a residual-level change. NOTE: this alters the DEEP
// equilibrium water table (deep cells now drain slowly instead of being frozen) -- the most sensitive
// region vs v2.0.1 -- so it is OFF by default and wants a sensitivity sweep + sign-off before adoption.
static double g_T_bedrock = 0.0;  // -wtm_T_bedrock: additive background transmissivity floor [m²/s], 0 = off

//////////////////////
// PUBLIC FUNCTIONS //
//////////////////////

// Depth-integrated transmissivity. Two forms are kept side by side:
//
//   * depthIntegratedTransmissivity       -- the published Fan et al. (2013) S4/S6
//     PIECEWISE form; the PRODUCTION choice, used by the Anderson residual. Cheap:
//     an exp only for deep cells (wtd < -shallow); shallow and above-surface cells
//     are a single multiply. Profiling showed it is ~20% faster per core than the
//     smooth form below at identical iteration counts (~30% stacked with
//     -snes_anderson_m 5); see benchmark/SOLVER_NOTES.md.
//   * depthIntegratedTransmissivitySmooth -- a smooth (C-inf) blend of the same,
//     differentiable everywhere. Its analytic derivative (dTransmissivityInverseDwtd) is the
//     Newton-Krylov Jacobian's T term (FormJacobianLocal) -- a differentiable, INEXACT-Newton
//     approximation of the residual, since the Newton residual (FormFunctionLocal) itself uses
//     the PIECEWISE T. Also used by the Picard operator when a -wtm_ksat_*_smoothing_width is
//     set. NOT used by the Anderson production residual.
double depthIntegratedTransmissivity(const double wtd_T, const double fdepth, const double ksat) {
  constexpr double shallow = 1.5;
  // Global soil datasets include information for shallow soils.
  // if the water table is deeper than this, the permeability
  // of the soil sees an exponential decay with depth.
  if (fdepth <= 0) {
    // If the fdepth is zero, there is no water transmission below the surface
    // soil layer.
    // If it is less than zero, it is incorrect -- but no water transmission
    // also seems an okay thing to do in this case.
    return 0;
  } else if (wtd_T < -shallow) {  // Equation S6 from the Fan paper
    return std::max(0.0, fdepth * ksat * std::exp((wtd_T + shallow) / fdepth)) + g_T_bedrock;
  } else if (wtd_T > 0 && !g_extended_soil) {
    // If wtd_T is greater than 0, max out rate of groundwater movement
    // as though wtd_T were 0. The surface water will get to move in
    // FillSpillMerge. (Extended-soil skips this clamp: the S4 form continues past the surface.)
    return std::max(0.0, ksat * (0 + shallow + fdepth)) + g_T_bedrock;
  } else {                                                    // Equation S4 from the Fan paper (extended: also wtd>0)
    return std::max(0.0, ksat * (wtd_T + shallow + fdepth)) + g_T_bedrock;  // max: no negative transmissivity.
  }
}

// --- Discharge potential Φ = ∫T dwtd (the piecewise Kirchhoff antiderivative) ----------------------
// KEPT AS A COEFFICIENT, NOT AS A SOLVE VARIABLE. Φ's one live consumer is T̄ (solver.t_bar), which needs
// the exact wtd-average of T over a step and gets it as ΔΦ/Δwtd (see interblockTransmissivity). Solving
// IN Φ was tried and retired 2026-09-09; the finding that retired it is kept below, because it is the
// reason Φ earns its place here and nowhere else.
// The steady groundwater problem is a nonlinear diffusion ∇·(T(wtd)∇h)+R=0 whose transmissivity spans
// MANY orders of magnitude with depth (T = fdepth·ksat·exp(wtd/fdepth)); that huge dynamic range is the
// dominant driver of the Jacobian ill-conditioning that caps the usable time step. The classic remedy
// for nonlinear diffusion is the KIRCHHOFF transform: solve for the discharge potential
//   Φ(wtd) = ∫_{-∞}^{wtd} T(s) ds
// instead of the head. Then dF/dΦ = (dF/dh)·(dh/dΦ) = (dF/dh)/T divides the transmissivity back out of
// the conditioning, and the residual is far more linear so Newton converges from farther / at larger dt.
// Φ is the antiderivative of the PIECEWISE Fan T above (so it matches the flux exactly), monotonic (T>0)
// hence invertible. dΦ/dwtd = T by construction. Requires the piecewise T (no ksat smoothing) and the
// standard surface physics (not -wtm_extended_soil).
//
// WHY THE VARIABLE CHANGE IS GONE (2026-08 measurement, acted on 2026-09-09): as a change of variable
// on the HEAD-FORM residual this reaches the identical
// equilibrium (verified to 8.7e-8 m) but does NOT raise the dt ceiling -- it worsens conditioning. The
// exact chain-rule Jacobian is dF/dΦ = (dF/dh)/T (column scaling by 1/T); the head-form storage term
// (h - rech) contributes dh/dΦ = 1/T to the DIAGONAL, and for deep cells T ~ 1e-11 so 1/T ~ 1e11 blows
// the diagonal up (shallow cells get near-zero columns), and MUMPS fails as cells drain deep. The
// continuous Kirchhoff benefit (operator -> constant-coefficient Laplacian) does not transfer to the
// discrete harmonic-mean CONSERVATIVE scheme under a mere change of variable. The 1/T blow-up is
// specific to the head form, so a VOLUME-form residual may transform more gracefully -- THAT VARIANT IS
// UNTESTED AND STILL OPEN; retiring the head-form implementation does not close it. See
// benchmark/EQUILIBRIUM_ROBUSTNESS.md and PORT_TO_UPSTREAM.md.
static double dischargePotential(const double wtd, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  const double fd = fdepth, k = ksat;
  // + g_T_bedrock·wtd throughout: the antiderivative of the constant background floor, so ∂Φ/∂wtd = T
  // (floored) and T̄ = ΔΦ/Δwtd stays exact (only ΔΦ is ever used, so the -∞ reference is immaterial).
  if (wtd < -shallow) return fd * fd * k * std::exp((wtd + shallow) / fd) + g_T_bedrock * wtd;  // exp: Φ = fdepth·T
  const double Phi0 = k * (0.5 * (shallow + fd) * (shallow + fd) + 0.5 * fd * fd);  // Φ at wtd = 0
  if (wtd > 0.0) return Phi0 + k * (shallow + fd) * wtd + g_T_bedrock * wtd;  // surface: T const → Φ linear
  const double u = wtd + shallow + fd;                                     // linear regime (-1.5 ≤ wtd ≤ 0)
  return k * (0.5 * u * u + 0.5 * fd * fd) + g_T_bedrock * wtd;
}

// --- Land-edge boundary condition (boundaries.land) -----------------------------------------------
// Ocean edges are ALWAYS Dirichlet h=0 (sea level; a fixed-head boundary -- not a choice). LAND edges are
// selectable. Two mechanisms, both applied at the off-map ghost node one cell outside the true edge:
//   neumann_toposlope (DEFAULT): ghost head = h_edge + (topo_edge - topo_inland) -> zero groundwater flux
//     RELATIVE TO the sloping land surface (terrain-following no-flow), the physical regional boundary.
//   dirichlet: ghost head = 0 (sea level) -> the modern ghost-node equivalent of the legacy sea-level
//     padding, imposed at a LAND edge without converting the cell to ocean. "For now just sea level."
// This selector is the general-framework hook; more values (plain zero-gradient, specified flux/head) can
// be added later. See BOUNDARY_CONDITIONS.md.
static bool g_land_boundary_dirichlet = false;  // boundaries.land: dirichlet_sea_level (default: neumann_toposlope)
// The collector actually in force after every override and solver downgrade -- what the run DID, not
// what the config asked for. Read only by the coverage fingerprint (emit_coverage_fingerprint below).
static std::string g_collector_resolved = "active_set";

// --- Time-averaged interblock transmissivity (-wtm_Tbar) -----------------------------------------
// The exponential T(wtd) is the dominant nonlinearity: the frozen-coefficient solvers freeze T at the
// CURRENT iterate (≈ start-of-step), which lags the true within-step transmissivity and makes the outer
// iteration oscillate/overshoot on stiff steps (the Kerry cold-start hang). Remedy: for the flux
// coefficient use each cell's TIME-AVERAGED T over the step wtd^n → wtd^{n+1}, not the instantaneous T.
// Because ∂Φ/∂wtd = T (Φ = dischargePotential, the piecewise Kirchhoff potential), the exact wtd-average
// of T between the two states is the Kirchhoff-potential difference
//     T̄ = (Φ(wtd^{n+1}) − Φ(wtd^n)) / (wtd^{n+1} − wtd^n)
// which is the LOG-MEAN in the deep exponential regime (where ln T is linear in wtd), the ARITHMETIC
// mean in the shallow-soil affine regime, and the constant surface T -- one continuous (C1) expression
// across all regimes. This changes ONLY the per-cell T that feeds the (unchanged) harmonic interblock
// mean: same physics, same equilibrium (at steady state wtd^{n+1}=wtd^n so T̄ → T), better-conditioned
// transient steps. It composes with any solver (residual-level change): Anderson evaluates it directly,
// Picard/Newton use it in the operator/Jacobian. Requires the piecewise Fan T (Φ is its antiderivative),
// so it is incompatible with ksat smoothing, extended soil, and the Kirchhoff variable change (enforced
// in update()). See benchmark/TBAR_TIME_AVERAGING.md.
static bool g_Tbar = false;  // -wtm_Tbar: use the step-time-averaged T̄ as the per-cell interblock T

// Defined below; forward-declared so interblockTransmissivity can fall back to the smooth form.
static double depthIntegratedTransmissivitySmooth(double wtd_T, double fdepth, double ksat);

// Per-cell interblock transmissivity for the harmonic face mean. With -wtm_Tbar, the step-time-averaged
// T̄ via the Kirchhoff-potential difference (small |Δwtd| → the instantaneous piecewise T, the exact
// Δ→0 limit); otherwise the instantaneous T (smooth form if a ksat smoothing width is set, else
// piecewise). wtd_old (= wtd^n) is ignored off the -wtm_Tbar path.
static double interblockTransmissivity(
    const double wtd_new, const double wtd_old, const double fdepth, const double ksat, const bool smooth_T) {
  if (g_Tbar) {
    const double dwtd = wtd_new - wtd_old;
    if (std::abs(dwtd) > 1e-9) {
      const double Tbar =
          (dischargePotential(wtd_new, fdepth, ksat) - dischargePotential(wtd_old, fdepth, ksat)) / dwtd;
      if (Tbar > 0.0) return Tbar;
    }
    return depthIntegratedTransmissivity(wtd_new, fdepth, ksat);
  }
  return smooth_T ? depthIntegratedTransmissivitySmooth(wtd_new, fdepth, ksat)
                  : depthIntegratedTransmissivity(wtd_new, fdepth, ksat);
}

// Exact wtd-derivative of the PIECEWISE Fan transmissivity (deep: T/fdepth = ksat·exp((wtd+1.5)/fdepth);
// soil: ksat; surface: 0). Used for the -wtm_Tbar Newton Jacobian tangent (the Δ→0 limit dT̄/dwtd_new →
// T'(wtd_new)/2, and the finite-Δ tangent [T(wtd_new) − T̄]/Δwtd needs the piecewise T at wtd_new).
[[maybe_unused]] static double dDepthIntegratedTransmissivityDwtd(
    const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  if (wtd_T < -shallow) return ksat * std::exp((wtd_T + shallow) / fdepth);  // d/dwtd of fdepth·ksat·exp(...)
  if (wtd_T > 0 && !g_extended_soil) return 0.0;                             // surface clamp: T constant
  return ksat;                                                              // soil affine: d/dwtd of ksat·(wtd+1.5+fdepth)
}

// Conductivity smoothing widths (metres) for the two kinks in the piecewise (C0) Fan
// transmissivity, each independent and each defaulting to 0 => sharp at that boundary:
//   * g_ksat_soilbottom_smoothing_width (-wtm_ksat_soilbottom_smoothing_width): the -1.5 m
//     soil-bottom transition, where conductivity switches from constant (shallow soil) to
//     exponential decay with depth.
//   * g_ksat_surface_smoothing_width (-wtm_ksat_surface_smoothing_width): the 0 m land-surface
//     clamp, where the water table reaches the surface and transmissivity is capped.
// The Picard operator uses the exact piecewise Fan T when both are 0 (production); if either is
// positive it uses the smooth (C-inf) form below with the respective bands (a positive width
// shifts the fixed point further from the piecewise Fan form).
static double g_ksat_soilbottom_smoothing_width = 0.0;  // eps1: -1.5 m conductivity transition
static double g_ksat_surface_smoothing_width    = 0.0;  // eps0: 0 m surface clamp

// --- Sub-surface surface-water sink (-wtm_surface_sink; taper 1, on by default) -------------------
// A smooth, compact-support removal in a band just BELOW the land surface that holds the water
// table strictly sub-surface (wtd < 0) while shunting the removed water on (to FSM, or discarded).
// Because no cell crosses wtd = 0, the model never engages the storativity jump / T-clamp free
// boundary, so BDF2-on-V stays 2nd order (the "no-crossing" regime) WITHOUT needing extended soil,
// and no above-surface water exists for open-water evaporation to act on. The ramp also crudely
// emulates near-surface evapotranspiration drawdown. Removal rate Q(wtd) = Qmax * g_w(wtd), with
// g_w a C2 quintic smoothstep rising 0 -> 1 across wtd in [-w, 0] (0 below the band, saturating at
// Qmax by the surface). Qmax must exceed the peak recharge rate to guarantee no breach.
// See benchmark/SURFACE_SINK_DESIGN.md sec 11.
static constexpr double SECONDS_IN_A_YEAR  = 31536000.0;
// EXFIL_EST note (adaptive-dt error estimate with the implicit exfiltration constraint): the head is constrained to
// wtd<=0 (h<=topo), excess routed to runoff. A linear step-predictor that overshoots ABOVE the surface would
// spike the per-cell error |x-h_pred| at that kink -- a projection artifact that does NOT shrink with dt, so
// the controller would reject to the floor (crash). Fix: CLAMP the predictor into the feasible set
// (h_pred = min(h_pred, topo)) in the error norm, so the estimate measures real truncation error -- a cell
// rising to the surface still contributes its true rise (bounding dt in Anderson's stable range), a pinned
// cell contributes ~0. Keeps ALL cells in the norm (excluding them would unbound dt -> Anderson can't take the
// huge step -> water piles). See benchmark/SURFACE_WATER_ROUTING.md / BDF2_ADAPTIVE_DESIGN.md.
static bool             g_volume_storage              = true;  // dev.storage_form: volume (DEFAULT) -- BE storage folded into f, RHS b=0
static bool             g_direct_to_runoff            = false; // -wtm_direct_to_runoff: in-residual exfiltration removal
// surface_water.routing: continuous (THE DEFAULT, #43) | impulse | off.
// CONTINUOUS feeds FSM's per-step water-table change into the NEXT step's recharge source. IMPULSE is
// the alternative: it overwrites the step baseline with the post-FSM table. Andy decided continuous on
// the PHYSICAL argument alone (#43) -- FSM moved that water, so the next step should see it as a source
// rather than as a rewritten initial condition.
// Assigned from params every step; this initialiser is never the value a run uses.
// History, so it is not re-litigated: continuous was BUILT to remove the between-step FSM shock (a
// Lie-split jump that breaks 2nd-order accuracy), and active_set turned out to remove that shock by
// itself (0.985 -> 3.6e-13). That did NOT demote it -- and the ~11% evaporation evidence originally
// cited for it did not survive re-measurement either (#51). The decision rests on the physics.
// It COMPOSES with active_set as of #40 (the obstacle reads the carried lake_stage, not the overwritten
// table); the old hard error is gone. Covered by tests/budget_closure and tests/fsm_conservation.
static bool             g_fsm_continuous            = true;
static bool             g_active_set                  = false; // -wtm_active_set [EXPERIMENTAL]: semismooth exfiltration pinned wtd=0 INSIDE the solve
static double           g_relax                       = 1.0;   // -wtm_relax: sub-step under-relaxation (1=off); damps free-boundary flicker

// -wtm_surface_exfiltration_to_runoff: post-solve surface exfiltration-to-runoff collection. Standard clamped-T physics; water is allowed to
// mound during the solve, then clamped to wtd=0 with the exact above-surface storage routed to FSM (via
// the sink accumulator). A robust, tuning-free "collect" alternative to the implicit sink; needs the sink
// off to do anything. See the truncation site in update().
static bool             g_surface_exfiltration_to_runoff_array    = false;


// Taper 2 -- demand-identity evaporation: the atmospheric loss transitions SMOOTHLY from the
// land-surface ET grid (deep) to the open-water rate owe (at/above the surface), as a logistic in
// wtd, treated IMPLICITLY in the solver (like the sink). Replaces the hard wtd>0 ? owe : ET recharge
// switch that sat on the wtd=0 knife-edge. E_eff(wtd) = ET + (owe-ET)*sigma((wtd - wtd_c)/s), a
// removal rate. Accessibility/extinction-depth (the max(0,P-ET) clamp) is deferred -> awickert/WTM#4.
// See SURFACE_SINK_DESIGN.md sec 14. ET/owe are per-cell (m/yr); the helpers return m/s.
static bool             g_evap_taper         = false;
static double           g_evap_taper_wtdc    = 0.05;  // wtd_c: half-rate depth [m] (small +, pond->exposed)
static double           g_evap_taper_s       = 0.1;   // s: transition width [m]

// Taper 3 -- accessibility / extinction-depth clamp (awickert/WTM#4). Gates the sub-surface part of the
// evaporative demand by an accessibility taper A(wtd): 1 at/above the surface, smoothly -> 0 at the
// extinction depth d_ext, below which the water table is too deep for evaporation/roots to reach. It
// converts taper 2's net removal into a DEFICIT-gated form so that in arid cells (E_eff > precip) the
// unmet demand draws down only a SHALLOW table (phreatic ET) and vanishes below d_ext -- without it,
// taper 2 alone draws an arid table down without bound (no equilibrium). Default d_ext = 8 m sits in the
// phreatic-transpiration band between sclerophyllous shrubland (5.2 m) and desert (9.5 m) rooting depths
// (Canadell et al. 1996); bare-soil direct evaporation is shallower (0.5-4.2 m by texture, Shah et al.
// 2007, whose exponential ET-vs-depth decay motivates the front-loaded smootherstep). See
// SURFACE_SINK_DESIGN.md sec 14f. Composes with taper 2 (inert when off: A == 1 everywhere).
static bool             g_extinction         = false;
static double           g_extinction_depth   = 8.0;   // d_ext: accessibility extinction depth [m]

// Compact-support C2 quintic smoothstep ramp: 0 for wtd <= -w, smoothly rising to 1 at wtd = 0
// (p(u) = u^3(6u^2 - 15u + 10), p'(0)=p'(1)=0). Argument is wtd = h - topo (centre cell).
// Exfiltration (-wtm_direct_to_runoff): remove exactly the ABOVE-surface excess to runoff each step.
// removal RATE [m/s] = max(0,wtd)/dt, so dt*removal = the excess depth. Pins wtd<=0 (no rate cap -> no
// pile) and removes nothing below the surface (no depression). The Anderson solve tolerates the hard
// max fine -- an earlier softplus smoothing was inert (convergence was eps-invariant), so it is gone.
static double directToRunoffRemoval(const double wtd, const double dt) { return std::max(0.0, wtd) / dt; }
// The exfiltration tangent dR/dwtd. Wired into the Picard operator + RHS as a FROZEN ACTIVE-SET diagonal
// (FormPicardOperator/FormPicardRHS): each Picard sweep fixes which cells exfil, so the discontinuous step at
// wtd=0 is handled by set membership, not a derivative -- and Picard+implicit then matches Anderson to ~1e-11.
// It is NOT yet in the Newton Jacobian (FormJacobianLocal): a Newton line search would overshoot the same kink,
// which needs a semismooth / active-set Newton (see the fork enhancement issue).
static double directToRunoffTangent(const double wtd, const double dt) {
  return (wtd > 0.0 ? 1.0 : 0.0) / dt;
}

// Taper 2 helpers. sigma is the logistic 1/(1+e^{-u}); u = (wtd - wtd_c)/s. E_eff(wtd) transitions
// ET -> owe as the table rises. Returns m/s (ET/owe supplied in m/yr). The tangent dE/dwtd is
// (owe-ET)*sigma*(1-sigma)/s; CLAMPED to >= 0 so it only ever strengthens the SPD storage diagonal
// (when ET > owe the raw tangent is negative). The clamp is applied identically in the operator and
// the RHS, so the Picard fixed point -- storage + dt*E_eff(w^{n+1}) = recharge -- is unchanged; only
// the linearization softens (a fixed-point step on that term). See SURFACE_SINK_DESIGN.md sec 14.
static double evapTaperSigma(const double wtd) {
  return 1.0 / (1.0 + std::exp(-(wtd - g_evap_taper_wtdc) / g_evap_taper_s));
}
static double evapTaper(const double wtd, const double et_yr, const double owe_yr) {
  return (et_yr + (owe_yr - et_yr) * evapTaperSigma(wtd)) / SECONDS_IN_A_YEAR;
}
// Raw (unclamped) dE_eff/dwtd; can be negative when ET > owe. Split out so taper 3 can build its own
// tangent from it before the shared SPD clamp is applied.
static double evapTaperTangentRaw(const double wtd, const double et_yr, const double owe_yr) {
  const double sig = evapTaperSigma(wtd);
  return (owe_yr - et_yr) * sig * (1.0 - sig) / g_evap_taper_s / SECONDS_IN_A_YEAR;
}
static double evapTaperTangent(const double wtd, const double et_yr, const double owe_yr) {
  const double raw = evapTaperTangentRaw(wtd, et_yr, owe_yr);
  return raw > 0.0 ? raw : 0.0;  // SPD-preserving clamp; see note above
}

// Taper 3 accessibility A(wtd): compact-support C2 quintic smootherstep, 1 at/above the surface,
// smoothly -> 0 at the extinction depth wtd = -d_ext (same ramp shape as surfaceSinkRamp). Inaccessible
// below d_ext, so phreatic ET there is zero. Only consulted when g_extinction is on.
static double accessTaper(const double wtd) {
  const double d = g_extinction_depth;
  if (wtd >= 0.0) return 1.0;   // at/above surface: fully accessible
  if (wtd <= -d) return 0.0;    // below extinction depth: inaccessible
  const double u = (wtd + d) / d;  // in (0,1): 0 at -d_ext, 1 at surface
  return u * u * u * (u * (6.0 * u - 15.0) + 10.0);
}
static double accessTaperTangent(const double wtd) {
  const double d = g_extinction_depth;
  if (wtd >= 0.0 || wtd <= -d) return 0.0;
  const double u = (wtd + d) / d;
  return 30.0 * u * u * (1.0 - u) * (1.0 - u) / d;  // p'(u)/d, p'(u)=30 u^2 (1-u)^2
}

// Net evaporative removal to the atmosphere with taper 3 folded in:
//   R(wtd) = min(E_eff, P) + (E_eff - P)_+ * A(wtd)
// = the demand met by precip, plus the ACCESSIBLE part of the sub-surface deficit. Equivalently the
// net atmospheric source is N = (P - E_eff)_+ - (E_eff - P)_+ * A. When taper 3 is off this returns
// E_eff exactly (byte-identical taper-2 fast path). p_rate is precipitation in m/s. See sec 14 / #4.
static double evapRemoval(const double wtd, const double et_yr, const double owe_yr, const double p_rate) {
  const double e = evapTaper(wtd, et_yr, owe_yr);
  if (!g_extinction) return e;  // taper 2 unchanged
  const double base    = (e < p_rate) ? e : p_rate;       // min(E_eff, P)
  const double deficit = e - p_rate;
  return base + (deficit > 0.0 ? deficit * accessTaper(wtd) : 0.0);
}
// dR/dwtd, SPD-clamped to >= 0 (same clamp discipline as evapTaperTangent). When taper 3 is off this
// returns evapTaperTangent exactly.
static double evapRemovalTangent(const double wtd, const double et_yr, const double owe_yr,
                                 const double p_rate) {
  if (!g_extinction) return evapTaperTangent(wtd, et_yr, owe_yr);  // taper 2 unchanged
  const double e      = evapTaper(wtd, et_yr, owe_yr);
  const double eprime = evapTaperTangentRaw(wtd, et_yr, owe_yr);
  double rprime;
  if (e <= p_rate) {
    rprime = eprime;  // wet: R = E_eff
  } else {
    rprime = eprime * accessTaper(wtd) + (e - p_rate) * accessTaperTangent(wtd);
  }
  return rprime > 0.0 ? rprime : 0.0;  // SPD clamp
}
// Unclamped dR/dwtd -- the EXACT derivative of evapRemoval (no SPD clamp). The Picard operator uses
// the clamped tangent above to keep its linearization SPD, but the true Newton Jacobian must
// differentiate the residual as written (evapRemoval is unclamped), so it uses this. Identical to
// evapRemovalTangent except the final max(.,0) is dropped; can be negative where ET > owe.
static double evapRemovalTangentRaw(const double wtd, const double et_yr, const double owe_yr,
                                   const double p_rate) {
  if (!g_extinction) return evapTaperTangentRaw(wtd, et_yr, owe_yr);  // taper 2 unchanged
  const double e      = evapTaper(wtd, et_yr, owe_yr);
  const double eprime = evapTaperTangentRaw(wtd, et_yr, owe_yr);
  if (e <= p_rate) return eprime;  // wet: R = E_eff
  return eprime * accessTaper(wtd) + (e - p_rate) * accessTaperTangent(wtd);
}

// Smooth (C-inf) depth-integrated transmissivity: a differentiable blend of the
// piecewise production form above. Kept for a future Newton path; its analytic
// derivative is dTransmissivityInverseDwtd, and FormJacobianLocal uses this
// version. NOT used by the Anderson production residual.
static double depthIntegratedTransmissivitySmooth(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0;
  constexpr double shallow = 1.5;
  const double eps0        = g_ksat_surface_smoothing_width;     // smooth clamping at WTD=0 boundary
  const double eps1        = g_ksat_soilbottom_smoothing_width;  // smooth blend at WTD=-shallow boundary

  const double wtd_eff = (wtd_T - std::sqrt(wtd_T * wtd_T + eps0 * eps0)) * 0.5;
  const double u       = wtd_T + shallow;
  // eps1 == 0 => the sigmoid degrades to a step (sharp -1.5 m switch); eps0 == 0 is naturally sharp
  // (sqrt(wtd^2) = |wtd| in wtd_eff), so either boundary can be sharp independently.
  const double sigma_1 = (eps1 > 0.0) ? 1.0 / (1.0 + std::exp(u / eps1)) : (u < 0.0 ? 1.0 : 0.0);

  const double T_linear = ksat * (wtd_eff + shallow + fdepth);
  const double T_exp    = fdepth * ksat * std::exp(u / fdepth);

  return std::max(0.0, (1.0 - sigma_1) * T_linear + sigma_1 * T_exp) + g_T_bedrock;
}

// Analytic derivative of (1/T) with respect to wtd_T for the Newton-Krylov Jacobian
// (FormJacobianLocal): the derivative of the SMOOTH T. When a ksat smoothing width is set the
// residual (FormFunctionLocal) uses the smooth T with that width, so track it here to stay the
// exact derivative; when a width is 0 the residual uses the piecewise T, and we fall back to a
// fixed 0.01 m regularization so the Jacobian stays differentiable (a standard inexact-Newton
// approximation) and never divides by zero.
static double dTransmissivityInverseDwtd(const double wtd_T, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  const double eps0 = (g_ksat_surface_smoothing_width  > 0.0) ? g_ksat_surface_smoothing_width  : 0.01;
  const double eps1 = (g_ksat_soilbottom_smoothing_width > 0.0) ? g_ksat_soilbottom_smoothing_width : 0.01;

  const double sq0      = std::sqrt(wtd_T * wtd_T + eps0 * eps0);
  const double wtd_eff  = (wtd_T - sq0) * 0.5;
  const double dwtd_eff = (1.0 - wtd_T / sq0) * 0.5;

  const double u       = wtd_T + shallow;
  const double sigma_1 = 1.0 / (1.0 + std::exp(u / eps1));
  const double dsigma1 = -sigma_1 * (1.0 - sigma_1) / eps1;

  const double T_linear = ksat * (wtd_eff + shallow + fdepth);
  const double T_exp    = fdepth * ksat * std::exp(u / fdepth);
  const double T        = std::max(0.0, (1.0 - sigma_1) * T_linear + sigma_1 * T_exp) + g_T_bedrock;
  if (T <= 0.0) return 0.0;  // with g_T_bedrock>0, T is bounded away from 0 (no dead-cell division)

  // dT/dwtd is unchanged by the constant floor; only the T in the denominator carries it.
  const double dT = dsigma1 * (T_exp - T_linear)
                  + (1.0 - sigma_1) * ksat * dwtd_eff
                  + sigma_1 * ksat * std::exp(u / fdepth);
  return -dT / (T * T);
}

// Analytic derivative of S_eff with respect to my_new_wtd: the EXACT derivative of
// updateEffectiveStorativity, so it must use the same storativity smoothing width (not a
// hardcoded constant) to stay the true Jacobian for any -wtm_storativity_surface_smoothing_width.
// Uses the same V(w) = [w(1+p) + sqrt(w²+eps²)(1-p)] / 2 construction as storedVolume/specificYield.
static double dEffectiveStorativityDnew(
    const double my_original_wtd, const double my_new_wtd, const double my_porosity) {
  const double eps = g_storativity_surface_smoothing_width;
  const double dwtd    = my_new_wtd - my_original_wtd;

  const auto V = [&](double w) {
    return (w * (1.0 + my_porosity) + std::sqrt(w * w + eps * eps) * (1.0 - my_porosity)) * 0.5;
  };
  const auto Vprime = [&](double w) {
    return ((1.0 + my_porosity) + w * (1.0 - my_porosity) / std::sqrt(w * w + eps * eps)) * 0.5;
  };

  if (std::abs(dwtd) > 1e-10) {
    const double S = (V(my_new_wtd) - V(my_original_wtd)) / dwtd;
    return (Vprime(my_new_wtd) - S) / dwtd;
  }
  // Near convergence (new ≈ old): dS/d(new) → V''(old)/2. With V''(w) = (1-p)·eps²/(w²+eps²)^1.5,
  // that limit is (1-p)·eps² / (4·(w²+eps²)^1.5) -- the ½ from the Taylor limit times the ½ in V''.
  const double w = my_original_wtd;
  return (1.0 - my_porosity) * eps * eps / (4.0 * std::pow(w * w + eps * eps, 1.5));
}

// The solve inputs are read from distributed DMDA arrays (indexed [y][x] over
// the owned range) rather than from full-grid arp arrays, so those arp arrays
// need not exist on non-root ranks: wtd from starting_wtd, recharge from
// rech_dist, land mask from mask, porosity from porosity_vec. cell_area is 1-D
// (Class-C) and stays replicated on all ranks. See DISTRIBUTED_ARP_DESIGN.md (2f-C).
void set_starting_values(
    ArrayPack& arp,
    PetscScalar** starting_wtd,
    PetscScalar** rech_dist,
    PetscScalar** mask,
    PetscScalar** porosity,
    PetscInt xs,
    PetscInt ys,
    PetscInt xm,
    PetscInt ym,
    double rech_dt_scale) {
  // no pragma because we're editing arp accumulators
  // Accumulate over this rank's OWNED cells only (DMDA owned range, which is
  // non-overlapping across ranks), so under MPI each ocean/recharge cell is
  // counted exactly once by its owner. total_loss_to_ocean_gw and
  // total_added_recharge are therefore per-rank partials; PrintValues reduces
  // them to global totals for reporting.
  // check to see if there is any non-zero water table in ocean
  // cells, and if so, record these values as changes to the ocean.
  for (int y = ys; y < ys + ym; y++) {
    for (int x = xs; x < xs + xm; x++) {
      if (mask[y][x] == 0) {
        if (starting_wtd[y][x] > 0)
          arp.total_loss_to_ocean_gw += starting_wtd[y][x] * arp.cell_area[y];
        else
          arp.total_loss_to_ocean_gw += starting_wtd[y][x] * arp.cell_area[y] * porosity[y][x];
        starting_wtd[y][x] = 0.;
      } else {
        // rech_dist holds a NOMINAL step's depth (rate * params.deltat); the source the solve actually
        // integrates is rech_dist * rech_dt_scale (see the note at the rech_vec assembly). Book the
        // SCALED depth, or a sub-stepped cycle credits a whole cycle's recharge once per sub-step --
        // measured at 2.55x inflation under adaptive dt, and it is only correct at all because
        // rech_dt_scale is exactly 1 on every fixed-dt path. The clip below mirrors add_recharge:
        // evaporation can remove surface water down to the land surface and no further.
        const double rech_step = rech_dist[y][x] * rech_dt_scale;
        double       rech_count = rech_step;
        if (starting_wtd[y][x] >= 0 && starting_wtd[y][x] + rech_step < 0)
          rech_count = -starting_wtd[y][x];

        arp.total_recharge_direct += rech_count * arp.cell_area[y];
      }
    }
  }
}

// Accumulate the water leaving through land->ocean faces this solve into arp.total_ocean_outflow_gw
// (a per-rank owned-cell partial; PrintValues reduces the partials to a global total). Ocean cells
// are Dirichlet h=0, so the crossing water is absorbed at the boundary and never appears as
// ocean-cell content -- the correct measure is the Darcy interface flux. It uses the SAME
// harmonic-mean conductance e = 2/(1/T_c + 1/T_nbr) the Picard operator assembles (mirroring its T
// construction), evaluated at the converged head, so the discrete budget closes exactly:
// recharge = d(storage) + ocean_outflow + surface_removed. Per land->ocean face the outflow volume
// is e * dt/(cell size)^2 * (h_land - 0) * cell_area, matching the operator's flux term (depth) times
// the cell area (volume). Needs ghost heads (x) and the ghost mask (mask_local); mirrors
// FormPicardOperator's ghost setup.
// `head` is the state to evaluate at and `weight` scales what it contributes. Single-stage schemes pass
// (user_context.x, 1.0). TR-BDF2 integrates the land->ocean flux over the step with the three-point
// quadrature its two stages define -- (w^n, Y_gamma, w^{n+1}) weighted (W_OLD, W_YGAMMA, W_NEW) -- so it
// calls this three times; see src/tr_bdf2_coefficients.hpp. Evaluating once at w^{n+1} with the full step,
// as a single-stage scheme correctly does, would attribute the whole step's outflow to the final state and
// the exact budget could not close.
// COVERAGE FINGERPRINT. One machine-readable line per run, appended to the file named by the
// WTM_COVERAGE_LOG environment variable (silent when unset, so ordinary runs are unaffected).
//
// WHY THE MODEL EMITS THIS RATHER THAN A SCRIPT PARSING CONFIGS. It records what the run ACTUALLY
// resolved to -- after CLI overrides, after the solver-dependent collector downgrade, after
// auto-enables -- not what a config file appears to say. Twice during this work a `sed` meant to
// switch a collector silently did nothing (a flat key applied to a nested-YAML file) and a whole
// measurement was made on the wrong configuration before anyone noticed. A fingerprint written by the
// model itself makes that impossible to sustain.
//
// tests/coverage_matrix.py aggregates these into tests/COVERAGE.md: what each test covers, and which
// CROSSINGS nothing covers. Every defect found on 2026-08-26 lived in an untested PAIR, not an
// untested axis, which is what the crossing view is for.
static void emit_coverage_fingerprint(const Parameters& params, const AppCtx& uc) {
  const char* path = std::getenv("WTM_COVERAGE_LOG");
  if (!path) return;
  PetscMPIInt rank = 0, size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (rank != 0) return;  // one line per run, from rank 0

  const char* solver = uc.use_picard    ? "picard"
                     : uc.use_newton    ? "newton"
                                        : "anderson";
  const char* integ  = uc.use_tr_bdf2    ? "tr_bdf2"
                     : uc.use_bdf2       ? "bdf2"
                     : g_volume_storage  ? "be_volume"
                                         : "be_secant";
  const char* dtctl  = uc.use_dt_adaptive          ? "adaptive"
                     : uc.use_newton_continuation  ? "continuation"
                                                   : "fixed";
  // The line is whitespace-delimited, so a tag with spaces would be truncated at the first word --
  // which silently mislabelled every multi-word test as its first token. Collapse to underscores.
  std::string tag = "unknown";
  if (const char* t = std::getenv("WTM_COVERAGE_TAG")) {
    tag = t;
    for (char& c : tag)
      if (std::isspace(static_cast<unsigned char>(c))) c = '_';
  }

  std::ofstream f(path, std::ios_base::app);
  if (!f) return;
  f << "coverage"
    << " test=" << tag
    << " run_type=" << params.run_type
    << " solver=" << solver
    << " integrator=" << integ
    << " dtctl=" << dtctl
    << " collector=" << g_collector_resolved
    << " fsm=" << (params.fsm_on ? 1 : 0)
    << " runoff_ratio=" << (params.runoff_ratio_on ? 1 : 0)
    << " infiltration=" << (params.infiltration_on ? 1 : 0)
    << " recharge_path=" << ((!params.fsm_on || !params.infiltration_on) ? "distributed" : "serial")
    << " coupling=" << (g_fsm_continuous ? "continuous" : "impulse")
    << " boundary=" << (g_land_boundary_dirichlet ? "dirichlet" : "neumann")
    << " ranks=" << size
    << "\n";
}

static void accumulate_ocean_outflow(AppCtx& user_context, ArrayPack& arp, Vec head, double weight) {
  DM  da = user_context.da;
  Vec xloc;
  DMGetLocalVector(da, &xloc);
  DMGlobalToLocalBegin(da, head, INSERT_VALUES, xloc);
  DMGlobalToLocalEnd(da, head, INSERT_VALUES, xloc);

  PetscScalar **xx, **my_topo, **my_fdepth, **my_ksat, **my_mask, **my_T, **gew, **gn, **gs;
  DMDAVecGetArray(da, xloc, &xx);
  DMDAVecGetArray(da, user_context.topo_local, &my_topo);
  DMDAVecGetArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecGetArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecGetArray(da, user_context.mask_local, &my_mask);
  DMDAVecGetArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecGetArray(da, user_context.geom_n_vec, &gn);
  DMDAVecGetArray(da, user_context.geom_s_vec, &gs);
  DMDAVecGetArray(da, user_context.T_local, &my_T);
  PetscScalar** my_starting_wtd_local = nullptr;  // -wtm_Tbar: ghosted w^n, so the accounted ocean-face
  if (g_Tbar) DMDAVecGetArray(da, user_context.starting_wtd_local, &my_starting_wtd_local);  // T̄ matches the solve

  DMDALocalInfo info;
  DMDAGetLocalInfo(da, &info);
  // Rebuild the SAME per-cell face T the solve used at the converged head, so the accounted land->ocean
  // flux closes the budget: instantaneous T, or (with -wtm_Tbar) the step-time-averaged T̄.
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  for (auto j = info.gys; j < info.gys + info.gym; j++)
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++) {
      const double wtd_T   = xx[j][i] - my_topo[j][i];
      const double wtd_old = g_Tbar ? my_starting_wtd_local[j][i] : 0.0;  // w^n; unused off -wtm_Tbar
      my_T[j][i] = 1.0 / interblockTransmissivity(wtd_T, wtd_old, my_fdepth[j][i], my_ksat[j][i], smooth_T);
    }

  const double dt = user_context.deltat;
  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      if (my_mask[j][i] == 0) continue;  // only LAND cells drain to ocean
      const double h_c = xx[j][i];
      // Volume flux through each land->ocean face = dt * G * h_c, with the SAME face conductance
      // G = e * (L_wall/d_centre) the operator assembles (E-W uses geom_ew, N/S the face geom_n/s).
      // Off-map faces (global domain edge) are the ghost boundary, not land->ocean -- guard the neighbour
      // reads so an edge LAND cell (real when -wtm_ghost_boundary skips setEdges) doesn't read out of bounds.
      if (i + 1 < info.mx && my_mask[j][i + 1] == 0) arp.total_ocean_outflow_gw += weight * dt * 2.0 / (my_T[j][i] + my_T[j][i + 1]) * gew[j][i] * h_c;
      if (i - 1 >= 0      && my_mask[j][i - 1] == 0) arp.total_ocean_outflow_gw += weight * dt * 2.0 / (my_T[j][i] + my_T[j][i - 1]) * gew[j][i] * h_c;
      if (j + 1 < info.my && my_mask[j + 1][i] == 0) arp.total_ocean_outflow_gw += weight * dt * 2.0 / (my_T[j][i] + my_T[j + 1][i]) * gn[j][i] * h_c;
      if (j - 1 >= 0      && my_mask[j - 1][i] == 0) arp.total_ocean_outflow_gw += weight * dt * 2.0 / (my_T[j][i] + my_T[j - 1][i]) * gs[j][i] * h_c;
      // Under land-edge Dirichlet, an off-map edge face also drains to the sea-level ghost (surface T);
      // count it so the water budget closes (neumann_toposlope off-map faces are no-flow -> nothing to add).
      if (g_land_boundary_dirichlet) {
        const double e_s = 2.0 / (my_T[j][i] + 1.0 / interblockTransmissivity(0.0, 0.0, my_fdepth[j][i], my_ksat[j][i], smooth_T));
        if (i + 1 >= info.mx) arp.total_ocean_outflow_gw += weight * dt * e_s * gew[j][i] * h_c;
        if (i - 1 < 0)        arp.total_ocean_outflow_gw += weight * dt * e_s * gew[j][i] * h_c;
        if (j + 1 >= info.my) arp.total_ocean_outflow_gw += weight * dt * e_s * gn[j][i] * h_c;
        if (j - 1 < 0)        arp.total_ocean_outflow_gw += weight * dt * e_s * gs[j][i] * h_c;
      }
    }
  }

  DMDAVecRestoreArray(da, xloc, &xx);
  DMDAVecRestoreArray(da, user_context.topo_local, &my_topo);
  DMDAVecRestoreArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecRestoreArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecRestoreArray(da, user_context.mask_local, &my_mask);
  DMDAVecRestoreArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecRestoreArray(da, user_context.geom_n_vec, &gn);
  DMDAVecRestoreArray(da, user_context.geom_s_vec, &gs);
  DMDAVecRestoreArray(da, user_context.T_local, &my_T);
  if (g_Tbar) DMDAVecRestoreArray(da, user_context.starting_wtd_local, &my_starting_wtd_local);
  DMRestoreLocalVector(da, &xloc);
}

// TR-BDF2 only: accumulate the step's FLUX and REMOVAL budget terms with the three-point quadrature
// the two stages define. Every such term enters the step balance as
//   W_OLD*Q(w^n) + W_YGAMMA*Q(Y_gamma) + W_NEW*Q(w^{n+1}),   the weights summing to 1,
// rather than as the single end-of-step evaluation a one-stage scheme correctly uses (derivation and
// the second-order condition: src/tr_bdf2_coefficients.hpp). Storage and recharge are NOT here: they
// telescope exactly to V(w^{n+1}) - V(w^n) and to the step's recharge, so the shared code below covers
// them unchanged.
//
// TIMING IS LOAD-BEARING. This must run while w^n (dmdapack.starting_wtd) and Y_gamma
// (user_context.tr_ygamma) are both still live -- update()'s copy-back overwrites w^n with w^{n+1}
// shortly afterwards, which is why the ocean-outflow call for every other scheme sits further down and
// this one does not.
static void accumulate_tr_bdf2_step_fluxes(AppCtx& user_context, ArrayPack& arp, DMDA_Array_Pack& dmdapack,
                                           bool evap_active, bool sink_active) {
  DM da                       = user_context.da;
  const auto [xs, ys, xm, ym] = get_corners(da);
  const double dt             = user_context.deltat;

  // w^n as a HEAD. starting_wtd stores 0 at ocean cells as a MARKER, not as a head, so w^n + topo there
  // would be the topography -- the same trap compute_tr_explicit documents. Ocean cells are Dirichlet
  // head 0, matching the converged x the other two evaluations use.
  if (!user_context.tr_head_old) VecDuplicate(user_context.x, &user_context.tr_head_old);
  {
    PetscScalar **ho, **topo_h;
    DMDAVecGetArray(da, user_context.tr_head_old, &ho);
    DMDAVecGetArray(da, user_context.topo_vec, &topo_h);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        ho[j][i] = (dmdapack.mask[j][i] == 0) ? 0.0 : dmdapack.starting_wtd[j][i] + topo_h[j][i];
    DMDAVecRestoreArray(da, user_context.tr_head_old, &ho);
    DMDAVecRestoreArray(da, user_context.topo_vec, &topo_h);
  }

  // Land->ocean Darcy flux at the three step states.
  accumulate_ocean_outflow(user_context, arp, user_context.tr_head_old, trbdf2::W_OLD);
  accumulate_ocean_outflow(user_context, arp, user_context.tr_ygamma, trbdf2::W_YGAMMA);
  accumulate_ocean_outflow(user_context, arp, user_context.x, trbdf2::W_NEW);

  // The two IN-RESIDUAL removals, same quadrature. Both must be taken here rather than in update()'s
  // commit loop, because both are evaluated at states the copy-back is about to destroy.
  //
  //   taper 2/3 evaporation -- to the ATMOSPHERE. Dominant in practice: on tests/multilake it carries
  //       91% of the water leaving the domain, so its weighting alone was worth 0.55% of recharge.
  //   taper 1 sink / direct-to-runoff -- to FILLSPILLMERGE, so the quadrature depth is also what gets
  //       handed over in sink_removed_dist. Weighting only the budget and not the handoff would remove
  //       one amount from the aquifer and deliver a different one, which is a leak rather than a
  //       mis-report. Measured on tests/fsm_consistency under the `implicit` collector: the exact
  //       residual was 191% of recharge with this term left at backward-Euler weighting, against
  //       6.5e-09 for backward Euler itself, so this is TR-specific and not a pre-existing gap.
  if (!evap_active && !sink_active) return;
  PetscScalar **topo, **yg;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;
  DMDAVecGetArray(da, user_context.topo_vec, &topo);
  DMDAVecGetArray(da, user_context.tr_ygamma, &yg);
  if (evap_active) {
    DMDAVecGetArray(da, user_context.evap_vec, &my_evap);
    DMDAVecGetArray(da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecGetArray(da, user_context.precip_vec, &my_precip);
  }
  // Quadrature of any per-cell removal rate R(wtd) over the step, in depth units.
  const auto quad = [&](auto&& R, int j, int i) {
    return dt * (trbdf2::W_OLD * R(static_cast<double>(dmdapack.starting_wtd[j][i]))
                 + trbdf2::W_YGAMMA * R(yg[j][i] - topo[j][i])
                 + trbdf2::W_NEW * R(dmdapack.x[j][i] - topo[j][i]));
  };
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++) {
      if (dmdapack.mask[j][i] == 0) continue;
      if (evap_active) {
        const double P = my_precip[j][i] / SECONDS_IN_A_YEAR;
        const double depth =
            quad([&](double w) { return evapRemoval(w, my_evap[j][i], my_owe[j][i], P); }, j, i);
        arp.total_evap_removed += depth * arp.cell_area[j];
      }
      if (sink_active) {
        // Mirrors FormFunctionLocal's cascade: direct-to-runoff supersedes the band sink.
        const double depth =
            g_direct_to_runoff ? quad([&](double w) { return directToRunoffRemoval(w, dt); }, j, i)
                               : 0.0;
        arp.total_surface_removed += depth * arp.cell_area[j];  // budget
        dmdapack.sink_removed_dist[j][i] += depth;              // and the water itself, to FSM
      }
    }
  DMDAVecRestoreArray(da, user_context.topo_vec, &topo);
  DMDAVecRestoreArray(da, user_context.tr_ygamma, &yg);
  if (evap_active) {
    DMDAVecRestoreArray(da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecRestoreArray(da, user_context.precip_vec, &my_precip);
  }
}

// Accumulate the solver's EXACT per-step discrete storage and specific-yield recharge over owned
// LAND cells, so the water budget closes to the SNES tolerance rather than the ~1% of the physical
// snapshot (whose gap is the BDF2-startup term, not a leak). The per-cell discrete balance the solve
// satisfies is  storage(w^{n+1},w^n,w^{n-1}) + dt*lateral_flux + dt*Q_sink = recharge_term, so summed
// over land cells (interior lateral fluxes cancel; the land->ocean flux is total_ocean_outflow):
//   total_storage_change = total_solver_recharge - total_ocean_outflow - total_surface_removed
// to SNES tolerance.
//
// SOLVER-AGNOSTIC (was Picard-only). The storage form must be the one the residual ACTUALLY used, so
// this mirrors the branch structure of both FormPicardRHS and the matrix-free FormFunctionLocal
// (transient_groundwater.cpp, the `tr_stage` / `bdf2v` / `vol_storage` / secant cascade) -- in VOLUME
// units, i.e. without the 1/Sy head-scaling the Anderson residual applies (a positive per-cell scale
// that leaves the root unchanged but would corrupt a budget).
//
// This term is what makes the budget agree with the SCHEME rather than with an external-water
// bookkeeping convention -- which matters directly for -wtm_fsm_continuous (the continuous coupling): once FSM's
// delivery is a source inside the step, the scheme's own conservation law counts it as an input, and
// `rech_vec` (which this reads) is exactly that full source term. See benchmark/WATER_BUDGET.md.
//
// TR-BDF2 IS covered. Its two stages each satisfy their own discrete balance, and the step's identity
// is the combination C1*(stage 1) + (stage 2), which telescopes because C1 - C2 == 1 and
// C1*gamma + C3 == 1 (src/tr_bdf2_coefficients.hpp). Storage and recharge come out as the
// backward-Euler forms below and need no special case; the flux and removal terms become a three-point
// quadrature over (w^n, Y_gamma, w^{n+1}) and are accumulated by accumulate_tr_bdf2_step_fluxes above.
// This used to bail and clear arp.exact_budget_valid, which was the honest thing to do while the
// combination was unbuilt -- but it also meant TR-BDF2 was the one scheme whose conservation nothing
// could check, and it was leaking 9.5% of recharge through the active-set exfiltration transfer.
//
// Called after the solve, BEFORE the BDF2 history overwrites w^{n-1}.
static void accumulate_budget_terms(AppCtx& user_context, ArrayPack& arp, DMDA_Array_Pack& dmdapack,
                                    bool evap_active, bool sink_active) {
  // TR-BDF2: the two stages DO telescope into one per-step identity -- C1*(stage 1) + (stage 2), with
  // C1 - C2 == 1 and C1*gamma + C3 == 1 (src/tr_bdf2_coefficients.hpp). Storage and recharge come out
  // unchanged from the backward-Euler forms below, so they fall through to the shared code; the flux
  // and removal terms become a three-point quadrature and are taken here, while w^n and Y_gamma are
  // both still live.
  if (user_context.use_tr_bdf2)
    accumulate_tr_bdf2_step_fluxes(user_context, arp, dmdapack, evap_active, sink_active);

  // TR-BDF2's telescoped storage is V(w^{n+1}) - V(w^n), i.e. the backward-Euler weights, whatever
  // -wtm_bdf2 may also be asking for -- the stage combination has already consumed the multi-level
  // structure. Without this guard the pair solver.time_integration: tr-bdf2 -wtm_bdf2 would silently take 3-level weights.
  const bool bdf2 = user_context.use_bdf2 && user_context.bdf2_have_history && !user_context.use_tr_bdf2;
  // Mirror the residual's branch choice -- but note that only BDF2 actually needs a separate volume
  // form. dev.storage_form: volume is a backward Euler whose storage is the exact volume change
  // V(w^{n+1}) - V(w^n), and the secant branch below already computes exactly that: by definition
  // updateEffectiveStorativity(w^n, w^{n+1}) IS the secant (V(w^{n+1}) - V(w^n)) / (w^{n+1} - w^n)
  // (pinned in src/test_storage_math.cpp), and h^{n+1} - h^n = w^{n+1} - w^n, so
  // S_c*(h^{n+1} - h^n) == V(w^{n+1}) - V(w^n) identically. The two forms separate only once the BDF2
  // weights are not (1,1,0), because S_c*(a_c h^{n+1} - b_c h^n + c_c h^{n-1}) is then NOT the
  // weighted volume difference. So the single volume branch is the BDF2-on-V one.
  double a_c = 1.0, b_c = 1.0, c_c = 0.0;  // backward-Euler weights (recharge form S_c*(h^{n+1}-h^n))
  if (bdf2) {
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    a_c                = (1.0 + 2.0 * omega) / (1.0 + omega);
    b_c                = 1.0 + omega;
    c_c                = omega * omega / (1.0 + omega);
  }

  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar **my_topo, **my_prev = nullptr;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  if (user_context.use_bdf2) DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &my_prev);
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      if (dmdapack.mask[j][i] == 0) continue;  // ocean: Dirichlet, no storage/recharge (flux counted separately)
      const double poro = dmdapack.porosity_vec[j][i];
      const double w1   = dmdapack.x[j][i] - my_topo[j][i];  // w^{n+1}
      const double w0   = dmdapack.starting_wtd[j][i];       // w^n (not yet overwritten by the copy-back)
      const double wm1  = my_prev ? my_prev[j][i] : 0.0;     // w^{n-1} (unused when c_c==0)
      const double rech = dmdapack.rech_vec[j][i];
      double storage, recharge;
      if (bdf2) {
        storage  = a_c * storedVolume(w1, poro) - b_c * storedVolume(w0, poro) + c_c * storedVolume(wm1, poro);
      } else {
        const double S_c = updateEffectiveStorativity(w0, w1, poro);  // secant storativity (matches the RHS)
        const double h1 = dmdapack.x[j][i], h0 = w0 + my_topo[j][i], hm1 = wm1 + my_topo[j][i];
        storage  = S_c * (a_c * h1 - b_c * h0 + c_c * hm1);
      }
      recharge = rech;  // recharge is now a fixed VOLUME (depth); the storativity scaling has moved out.
      arp.total_storage_change  += storage * arp.cell_area[j];
      arp.total_solver_recharge += recharge * arp.cell_area[j];
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);
  if (my_prev) DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &my_prev);
}

// Precompute the explicit old-state (w^n) term the TR-BDF2 trapezoidal stage needs:
//   tr_expl = dt*( N(w^n)/A_j + removal(w^n) )   per owned LAND cell,
// with N the conservative-FV net outflow at w^n. Uses the ghosted w^n (starting_wtd_local, scattered in
// update()) for neighbour heads and instantaneous T(w^n) (scratched into T_local). Called once per step
// before the trapezoidal solve; mirrors FormFunctionLocal's flux, evaluated at w^n instead of the iterate.
static void compute_tr_explicit(AppCtx& user_context) {
  DM da = user_context.da;
  PetscScalar **wn, **my_topo, **my_fdepth, **my_ksat, **my_mask, **my_T, **gew, **gn, **gs, **expl;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;
  DMDAVecGetArray(da, user_context.starting_wtd_local, &wn);
  DMDAVecGetArray(da, user_context.topo_local, &my_topo);
  DMDAVecGetArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecGetArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecGetArray(da, user_context.mask_local, &my_mask);  // GHOSTED: the nhead lambda reads neighbour mask
  DMDAVecGetArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecGetArray(da, user_context.geom_n_vec, &gn);
  DMDAVecGetArray(da, user_context.geom_s_vec, &gs);
  DMDAVecGetArray(da, user_context.T_local, &my_T);
  DMDAVecGetArray(da, user_context.tr_expl, &expl);
  if (g_evap_taper) {
    DMDAVecGetArray(da, user_context.evap_vec, &my_evap);
    DMDAVecGetArray(da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecGetArray(da, user_context.precip_vec, &my_precip);
  }
  DMDALocalInfo info;
  DMDAGetLocalInfo(da, &info);
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  // 1/T(w^n) over the ghost range (instantaneous; the explicit trapezoidal flux uses T at w^n).
  for (auto j = info.gys; j < info.gys + info.gym; j++)
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++)
      my_T[j][i] = 1.0 / interblockTransmissivity(wn[j][i], wn[j][i], my_fdepth[j][i], my_ksat[j][i], smooth_T);
  const double dt = user_context.deltat;
  for (auto j = info.ys; j < info.ys + info.ym; j++)
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      if (my_mask[j][i] == 0) { expl[j][i] = 0.0; continue; }
      const double h_c = wn[j][i] + my_topo[j][i];
      // Ocean neighbours are Dirichlet head = 0 (matching FormFunctionLocal, where x[ocean] is driven to
      // 0). starting_wtd stores 0 at ocean cells (a marker, NOT the head), so wn[ocean]+topo would be the
      // topography, not the boundary head -- for a truncation edge cutting high land that is a huge, and
      // scheme-inconsistent, error. Use the Dirichlet head here so the explicit trapezoidal flux sees the
      // SAME boundary as the implicit stages.
      // Per-face old-state outflow, matching FormFunctionLocal. Off-map faces (global domain edge; reached
      // only for a real LAND edge cell when -wtm_ghost_boundary skips setEdges) use the land-slope ghost
      // (head parallels topo via the INWARD reflection, 1/T = centre) -> no out-of-bounds under
      // DM_BOUNDARY_NONE. Ocean neighbours (mask==0) are Dirichlet head 0 (starting_wtd[ocean]=0 is a
      // marker, not a head -- see the committed compute_tr_explicit fix).
      const auto face = [&](int nj, int ni, double G) -> double {
        double h_nbr, Tinv_nbr;
        if (nj < 0 || nj >= info.my || ni < 0 || ni >= info.mx) {  // off-map land edge: ghost node
          if (g_land_boundary_dirichlet) {  // dirichlet: ghost = ocean neighbour (head 0, surface T)
            h_nbr    = 0.0;
            Tinv_nbr = 1.0 / interblockTransmissivity(0.0, 0.0, my_fdepth[j][i], my_ksat[j][i], smooth_T);
          } else {                          // neumann_toposlope (default): terrain-following no-flow
            const double topo_inland = my_topo[2 * j - nj][2 * i - ni];
            h_nbr    = h_c + (my_topo[j][i] - topo_inland);
            Tinv_nbr = my_T[j][i];
          }
        } else if (my_mask[nj][ni] == 0) {
          h_nbr    = 0.0;
          Tinv_nbr = my_T[nj][ni];
        } else {
          h_nbr    = wn[nj][ni] + my_topo[nj][ni];
          Tinv_nbr = my_T[nj][ni];
        }
        return (2.0 / (my_T[j][i] + Tinv_nbr)) * G * (h_c - h_nbr);
      };
      const double N = face(j, i + 1, gew[j][i]) + face(j, i - 1, gew[j][i])
                     + face(j + 1, i, gn[j][i]) + face(j - 1, i, gs[j][i]);
      const double A_j = user_context.cellsize_NS_squared / gew[j][i];
      double removal = 0.0;
      if (g_direct_to_runoff) removal += directToRunoffRemoval(wn[j][i], dt);
      if (g_evap_taper)
        removal += evapRemoval(wn[j][i], my_evap[j][i], my_owe[j][i], my_precip[j][i] / SECONDS_IN_A_YEAR);
      expl[j][i] = dt * N / A_j + dt * removal;
    }
  DMDAVecRestoreArray(da, user_context.starting_wtd_local, &wn);
  DMDAVecRestoreArray(da, user_context.topo_local, &my_topo);
  DMDAVecRestoreArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecRestoreArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecRestoreArray(da, user_context.mask_local, &my_mask);
  DMDAVecRestoreArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecRestoreArray(da, user_context.geom_n_vec, &gn);
  DMDAVecRestoreArray(da, user_context.geom_s_vec, &gs);
  DMDAVecRestoreArray(da, user_context.T_local, &my_T);
  DMDAVecRestoreArray(da, user_context.tr_expl, &expl);
  if (g_evap_taper) {
    DMDAVecRestoreArray(da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecRestoreArray(da, user_context.precip_vec, &my_precip);
  }
}

// Volume-weighted per-solve convergence (#127). Judges the SNES step in WATER (|S*Δwtd|) rather than head, so
// the per-solve gate speaks the same units as eq_tol (equilibrium) and dt_tol (adaptive) -- a deep low-storativity
// cell (big head step, ~zero water moved) can no longer hold the SOLVE unconverged. The water step is computed
// EXACTLY as |storedVolume(wtd_new) - storedVolume(wtd_old)| (slope 1 above the surface, porosity below -- the same
// V(wtd) the storage residual + eq metric use), with the step Δx from SNESGetSolutionUpdate.
//   DIAGNOSTIC (govern=false): print the head-vs-water step each iteration, and cross-check the reconstructed head
//   step ||Δx|| against PETSc's snorm to CONFIRM the update vector on the matrix-free Anderson path; then DEFER the
//   verdict to SNESConvergedDefault -- behaviour is UNCHANGED.
//   GOVERN (true): swap the head relative-step (stol) test for the water one; atol/rtol/maxit stay with the default.
// The relative WATER step |ΔV|/|V| between this iterate and the previous accepted one, with the
// diagnostic pieces alongside. Factored out of VolumeStepConverged (#62) so the ORDINARY path and the
// ADAPTIVE-RESTART path judge a step the same way: the restart phase test used to compare head snorm
// against ar_stol, which is the very head-vs-water mismatch #61 removed from the ordinary path.
// Maintains uc->vol_prev_x. At it == 0 there is no step yet: it seeds the reference and returns false.
static bool waterStep(SNES snes, AppCtx* uc, PetscInt it, double* water_rel, double* water_max,
                      double* water_L2, double* head_L2) {
  // The step is taken vs the PREVIOUS accepted iterate we store ourselves. SNESGetSolutionUpdate does NOT return
  // Anderson's accepted (mixed) step here -- measured ~10x larger and near-constant, so it is some internal
  // update vector; the exact semantics were not chased since the stored-iterate diff is authoritative (== snorm).
  Vec x;
  SNESGetSolution(snes, &x);
  if (uc->vol_prev_x == nullptr) VecDuplicate(x, &uc->vol_prev_x);
  if (it == 0) { VecCopy(x, uc->vol_prev_x); return false; }  // reset the reference at the start of each solve
  const auto [xs, ys, xm, ym] = get_corners(uc->da);
  PetscScalar **xa, **xpa, **topo, **poro, **msk;
  DMDAVecGetArray(uc->da, x, &xa);
  DMDAVecGetArray(uc->da, uc->vol_prev_x, &xpa);
  DMDAVecGetArray(uc->da, uc->topo_vec, &topo);
  DMDAVecGetArray(uc->da, uc->porosity_vec, &poro);
  DMDAVecGetArray(uc->da, uc->mask, &msk);
  double vmax = 0.0, vsq = 0.0, vnsq = 0.0, hsq = 0.0;  // water max, water L2^2, |V|^2, head-step L2^2 (check)
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      if (msk[j][i] != 0) {
        const double p     = poro[j][i];
        const double wtd_n = xa[j][i] - topo[j][i];
        const double wtd_o = xpa[j][i] - topo[j][i];
        const double dv    = std::abs(storedVolume(wtd_n, p) - storedVolume(wtd_o, p));
        const double vn    = storedVolume(wtd_n, p);
        const double dh    = static_cast<double>(xa[j][i]) - static_cast<double>(xpa[j][i]);
        if (dv > vmax) vmax = dv;
        vsq  += dv * dv;
        vnsq += vn * vn;
        hsq  += dh * dh;
      }
  DMDAVecRestoreArray(uc->da, x, &xa);
  DMDAVecRestoreArray(uc->da, uc->vol_prev_x, &xpa);
  DMDAVecRestoreArray(uc->da, uc->topo_vec, &topo);
  DMDAVecRestoreArray(uc->da, uc->porosity_vec, &poro);
  DMDAVecRestoreArray(uc->da, uc->mask, &msk);
  VecCopy(x, uc->vol_prev_x);  // this iterate becomes the reference for the next step
  double loc[4] = {vmax, vsq, vnsq, hsq}, g[4];
  MPI_Allreduce(&loc[0], &g[0], 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);        // water max
  MPI_Allreduce(&loc[1], &g[1], 3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);        // sums: water L2^2, |V|^2, head L2^2
  *water_max = g[0];
  *water_L2  = std::sqrt(g[1]);
  *water_rel = (g[2] > 0.0) ? std::sqrt(g[1] / g[2]) : 0.0;  // solution-relative water step
  *head_L2   = std::sqrt(g[3]);                              // reconstructed ||Δx||; should ≈ snorm
  return true;
}

// Opt-in via -wtm_snes_volume_conv[_govern]; this function is the full criterion, gated by the govern switch.
static PetscErrorCode VolumeStepConverged(SNES snes, PetscInt it, PetscReal xnorm, PetscReal snorm,
                                          PetscReal fnorm, SNESConvergedReason* reason, void* ctx) {
  AppCtx* uc = static_cast<AppCtx*>(ctx);
  if (it == 0) uc->snes_fnorm0 = fnorm;   // this solve's reference residual; the gate below is relative to it
  // Standard verdict first: atol/rtol/maxit + the head-step stol. Keep all of it except, when governing, the stol.
  SNESConvergedDefault(snes, it, xnorm, snorm, fnorm, reason, nullptr);
  double water_rel = 0.0, water_max = 0.0, water_L2 = 0.0, head_L2 = 0.0;
  if (!waterStep(snes, uc, it, &water_rel, &water_max, &water_L2, &head_L2)) return 0;

  if (uc->vol_step_trace)  // printing is now independent of the verdict: both, either, or neither
    PetscPrintf(PETSC_COMM_WORLD,
                "  [vol-conv diag] it=%d  head snorm=%.3e (recon %.3e) rel=%.3e | water max=%.3e L2=%.3e rel=%.3e\n",
                (int)it, (double)snorm, head_L2, (double)(xnorm > 0.0 ? snorm / xnorm : 0.0),
                water_max, water_L2, water_rel);

  // ---------------------------------------------------------------------------------------------
  // THIS TEST CAN DECLARE CONVERGENCE ON A STALLED SEMISMOOTH SOLVE. KNOWN, MEASURED, OPEN (#104).
  //
  // water_rel is a RELATIVE STEP: ||Δwater|| / ||water||. #61 fixed the QUANTITY -- it moved the
  // per-solve criterion off the head step, which was a stagnation test letting 88-95% of solves exit
  // early. It did not change the FORM. A relative-step criterion cannot distinguish "converged" from
  // "stalled", whatever quantity it measures, and on the semismooth active-set path (the min-NCP
  // obstacle, surface_water.collection.method: active_set -- the SHIPPED DEFAULT) the solve does stall.
  //
  // MEASURED on the first step of a `saturated` cold start (also the shipped default for equilibrium
  // spin-up), tests/variable_porosity fixture, backward-euler, routing off. Exit reason and iteration
  // count at step 0:
  //     affected arms        CONVERGED_SNORM_RELATIVE at  4-7 iterations
  //     unaffected active_set                            17-62
  //     collection.method: explicit                      24-69   (never single digits)
  // SINGLE-DIGIT ITERATION COUNTS ON THIS PATH ARE THE TELL. #61 recorded the same signature for the
  // head-metric version of this defect ("signature of a bad step: iters=3"); it still holds.
  //
  // The committed answer is not slightly off, it is somewhere else. Tightening
  // solver.convergence.water_volume_tol removes it, and the result then STOPS MOVING -- converged,
  // not drifting -- and lands on the collection.method: explicit answer EXACTLY:
  //     dt (wk)   resid @1e-08   @1e-10       @1e-12     | min wtd @1e-12   explicit
  //     2.5000    +9.530e+07     +6.220e-02   +6.220e-02 |   -30.3551       -30.3551
  //     1.1875    -8.266e+07     -1.017e-01   -1.017e-01 |   -19.6073       -19.6073
  //     1.0000    +1.112e+07     +1.112e+07   -3.342e-02 |   -17.4760       -17.4760
  //
  // IT APPEARS IN BANDS OF dt, NOT AS A TREND, and that is diagnostic rather than curious: 1.0 wk needs
  // 1e-12 where 2.5 wk is fixed by 1e-10, so whether the shipped 1e-8 is tight enough depends on the
  // solve TRAJECTORY, which varies discretely with dt. It also flips sign between bands. A diffusive
  // operator cannot produce that, which is what pointed at a switching mechanism in the first place.
  // Under collection.method: explicit or off the same sweep is clean at every dt (0 of 16 arms), and
  // the step-0 answer varies smoothly with dt; under active_set it does not.
  //
  // IT IS NOT PETSc's snes_stol, and that was checked rather than assumed: running with `-snes_stol 0`
  // verifiably takes (the banner prints `tolerance(snes_stol)=0.`) and changes NOTHING -- identical
  // residuals and identical iteration counts. The verdict below is ours.
  //
  // DO NOT "fix" this by loosening a downstream tolerance or re-golding a reference. Reproduction and
  // every table: benchmark/n78_first_step/. Options under consideration in #104 -- an absolute
  // residual/complementarity test instead of a relative step, requiring the ACTIVE SET to be unchanged
  // for k iterations, tightening the default (costed: the iteration count roughly quadruples), or at
  // minimum refusing to report CONVERGED at single-digit iterations on this path.
  // ---------------------------------------------------------------------------------------------
  // THE GATE (#104). The step verdict is a FALLBACK for a residual that cannot reach rtol -- a real
  // possibility on this semismooth path, where the residual can floor on the constraint kink. It is NOT
  // a second opinion that may overrule a residual still far from converged. So it is refused until the
  // residual has demonstrably come down.
  //
  // RELATIVE TO THIS SOLVE'S FIRST RESIDUAL, deliberately: an absolute bound measured on a 96-cell
  // fixture would not carry to a 384k-cell domain. MEASURED on a 20-point dt sweep, fnorm_exit/fnorm_0
  // at the moment of exit, classified by whether the committed answer matched a converged one:
  //     10 arms that agreed   worst ratio 5.5870e-08
  //     10 arms that did not  best  ratio 2.1219e-02
  // A separation of 3.8e+05, with fnorm_0 itself spanning 5.09 to 162.9 across the sweep -- so the RATIO
  // discriminates where an absolute number could not. The default 1e-5 is log-centred in that window:
  // ~179x above the worst agreeing arm and ~2100x below the best disagreeing one. It is a choice, and
  // solver.convergence.residual_gate exposes it; it is not a tuned number, because nothing lives in the
  // five orders either side of it.
  const bool residual_has_come_down =
      (uc->snes_fnorm0 <= 0.0) || (fnorm <= uc->snes_residual_gate * uc->snes_fnorm0);
  if (uc->snes_volume_conv_govern) {
    if (*reason == SNES_CONVERGED_SNORM_RELATIVE) *reason = SNES_CONVERGED_ITERATING;  // drop the head stol verdict
    if (water_rel < uc->snes_volume_conv_tol && residual_has_come_down)
      *reason = SNES_CONVERGED_SNORM_RELATIVE;  // ...use the water one, but only once the residual agrees
  }
  return 0;
}

// Custom SNES convergence test for a -wtm_adaptive_restart Anderson phase. Tracks the GLOBAL best
// (lowest-residual) iterate across restarts (ar_best_x) and STOPS the phase, recording WHY in
// ar_stop_kind, so update()'s outer loop can decide converge-vs-restart:
//   1 = true convergence (relative WATER step < ar_stol; see #62)
//   2 = RATE precursor: rho = |F_k|/|F_{k-1}| > ar_rho_threshold for ar_rho_patience iters -> restart
//   3 = phase cap ar_max_it reached -> restart
// A stopped phase always returns a POSITIVE reason so SNESSolve does not report a spurious divergence.
static PetscErrorCode AdaptiveRestartTest(SNES snes, PetscInt it, PetscReal xnorm, PetscReal snorm,
                                          PetscReal fnorm, SNESConvergedReason* reason, void* ctx) {
  AppCtx* uc       = static_cast<AppCtx*>(ctx);
  *reason          = SNES_CONVERGED_ITERATING;
  uc->ar_stop_kind = 0;
  if (!uc->ar_best_valid || fnorm < uc->ar_best_norm) {  // global best across all restart phases
    uc->ar_best_norm = fnorm;
    Vec x;
    SNESGetSolution(snes, &x);
    VecCopy(x, uc->ar_best_x);
    uc->ar_best_valid = PETSC_TRUE;
  }
  // Measure this phase's step in WATER, exactly as the ordinary path does (#62). At it == 0 there is no
  // step yet and this only seeds the reference, alongside the rate history seeded just below.
  (void)snorm; (void)xnorm;  // the head step is no longer what decides; kept in the signature by PETSc
  double water_rel = 0.0, w_max = 0.0, w_L2 = 0.0, h_L2 = 0.0;
  const bool have_step = waterStep(snes, uc, it, &water_rel, &w_max, &w_L2, &h_L2);
  if (it == 0) {  // start of a phase: seed the rate history, no rho/step test yet
    uc->ar_prev_norm = fnorm;
    uc->ar_rho_bad   = 0;
    return 0;
  }
  // TRUE CONVERGENCE, judged in water. This was `snorm < ar_stol * xnorm` -- a HEAD step against a
  // head-relative bound, the same mismatch #61 removed from the ordinary solve path. It let a phase call
  // itself converged once the iterate stopped MOVING in head, which on a warm start happens well before
  // the water it still owes has been driven out. water_rel is already solution-relative, so ar_stol
  // carries over unchanged -- only the metric moves, not the tolerance.
  // SAME FORM, SO THE SAME GATE (#104). This is a relative water step against a relative bound, exactly
  // as VolumeStepConverged above, and it inherited the same inability to tell "converged" from "stalled"
  // on the semismooth active-set path. It carries the gate so the two paths cannot disagree about what
  // convergence means. NOTE the gate is measured on the ORDINARY solve path, not this one -- applying it
  // here is consistency, not a second measurement. ar_best_norm above is this phase's reference: it is
  // the lowest residual seen, which is the right thing to compare against when restarts may have reset
  // the iterate, and it is already maintained for the restart logic.
  const bool ar_residual_has_come_down =
      (uc->snes_fnorm0 <= 0.0) || (fnorm <= uc->snes_residual_gate * uc->snes_fnorm0);
  if (have_step && water_rel < uc->ar_stol && ar_residual_has_come_down) {  // true convergence
    *reason          = SNES_CONVERGED_SNORM_RELATIVE;
    uc->ar_stop_kind = 1;
    return 0;
  }
  const PetscReal rho = (uc->ar_prev_norm > 0.0) ? fnorm / uc->ar_prev_norm : 0.0;
  uc->ar_prev_norm    = fnorm;
  if (rho > uc->ar_rho_threshold) uc->ar_rho_bad++;
  else uc->ar_rho_bad = 0;
  if (uc->ar_rho_bad >= uc->ar_rho_patience) {  // rate precursor -> restart
    *reason          = SNES_CONVERGED_ITS;
    uc->ar_stop_kind = 2;
    return 0;
  }
  if (it >= uc->ar_max_it) {  // phase cap -> restart
    *reason          = SNES_CONVERGED_ITS;
    uc->ar_stop_kind = 3;
    return 0;
  }
  return 0;
}

int update(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  PetscInt its;                // iterations for convergence
  SNESConvergedReason reason;  // Check convergence

  // --- diagnostic: profile the non-PETSc O(N) overhead; appears in -log_view ---
  static PetscLogEvent EVENT_SETSTART = 0, EVENT_FULLREDUCE = 0;
  static bool events_registered = false;
  if (!events_registered) {
    PetscLogEventRegister("SetStartVals", 0, &EVENT_SETSTART);
    PetscLogEventRegister("FullGridReduce", 0, &EVENT_FULLREDUCE);
    events_registered = true;
  }

  // Get local array bounds
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);

  // compute any starting values needed for arrays (owned cells only).
  // wtd is carried in dmdapack.starting_wtd (populated once per cycle before the
  // per-report step loop, then maintained by the copy-back below), not in arp.wtd.
  // Recharge is a per-step AMOUNT (a depth) = rate*dt, but rech_dist is baked ONCE as
  // rate*params.deltat (irf.cpp / WTM.cpp). The residual adds my_rech directly and scales only the
  // flux by user_context.deltat, so on a VARIABLE-dt path (adaptive / Newton dt-continuation) an
  // unscaled source over-recharges when dt shrinks below params.deltat -- the "source term grows as
  // the step shrinks" instability that broke earlier adaptive stepping. Rescale to rate*(actual dt) so
  // recharge and drainage scale together; the steady state is then dt-independent (rate = drainage at
  // the fixed point, dt cancels). Exactly 1.0 on every fixed-dt path, so those are byte-identical.
  // See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  const double rech_dt_scale = user_context.deltat / params.deltat;

  // compute any starting values needed for arrays (owned cells only).
  // wtd is carried in dmdapack.starting_wtd (populated once per cycle before the
  // per-report step loop, then maintained by the copy-back below), not in arp.wtd.
  // Hoisted BELOW rech_dt_scale (it used to sit above) because it books the direct-recharge input and
  // must book the SCALED depth -- the same depth the rech_vec assembly just below hands the solver.
  PetscLogEventBegin(EVENT_SETSTART, 0, 0, 0, 0);
  set_starting_values(arp, dmdapack.starting_wtd, dmdapack.rech_dist, dmdapack.mask, dmdapack.porosity_vec,
                      xs, ys, xm, ym, rech_dt_scale);
  PetscLogEventEnd(EVENT_SETSTART, 0, 0, 0, 0);

//  values for storativity are reset each time; and recharge changes from one timestep to the next, so set these here
#pragma omp parallel for default(none) shared(arp, ys, ym, xs, xm, dmdapack, params, rech_dt_scale) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      // The solver's source is external recharge PLUS the FSM delta; set_starting_values above books only
      // rech_dist, so the delta reaches the solve without being counted as external input. Summed BEFORE
      // add_recharge (not clipped separately) and scaled by the same rech_dt_scale, which is exactly what
      // the single-carrier version did -- so this is bit-identical wherever fsm_delta_dist is zero, i.e.
      // everywhere except fsm_continuous runs. OPEN: whether the delta SHOULD carry rech_dt_scale is a
      // separate question -- it is a volume FSM already moved for one specific step. Preserved, not
      // silently changed; rech_dt_scale is 1 on every fixed-dt path.
      dmdapack.rech_vec[j][i] =
          add_recharge(dmdapack.rech_dist[j][i] * rech_dt_scale + dmdapack.fsm_delta_dist[j][i],
                       dmdapack.starting_wtd[j][i], dmdapack.porosity_vec[j][i]);
    }
  }

  // Smoothing widths are physics modeling options and apply on ALL solver paths (Anderson,
  // Newton, Picard), so read them here -- before the solver branch -- rather than gating them
  // behind use_picard. Storativity land-surface transition (sub-grid roughness); default 0.01 m,
  // always on. The two ksat/transmissivity widths default to 0 (=> exact piecewise Fan T); any
  // positive width rounds that boundary in every path that evaluates T (residual and operator).
  g_storativity_surface_smoothing_width = params.storativity_surface_smoothing;
  g_ksat_soilbottom_smoothing_width     = params.ksat_soilbottom_smoothing;
  g_ksat_surface_smoothing_width        = params.ksat_surface_smoothing;
  // -wtm_extended_soil is RETIRED: `surface_water.collection.method: extended_soil` is the one way in.
  // g_extended_soil is now set ONLY by the selector below, where the mode is resolved and where its
  // NONPHYSICAL banner is printed -- warn where the mode is in force, not where a request for it is parsed.

  const bool anderson_path = !user_context.use_picard && !user_context.use_newton;

  // Surface-water CLAMP (Fan & Miguez-Macho) -- DEFAULT ON (all solver paths) so physical runs pin wtd<=0
  // and never flicker. "Route vs discard" is the fsm_on choice ON TOP of this (fsm_on routes the exfiltrated
  // water as lakes; fsm off discards it to runoff -- both keep the clamp). This is a POST-SOLVE clamp: it
  // truncates any residual wtd>0 to the surface and routes the excess, and fires only when a cell ends above
  // the surface. On every path the in-residual taper-1 sink (-wtm_surface_sink, on by default) is the primary
  // manager that already holds wtd<=0, so this is a safety net (a no-op when the sink holds; the corrective
  // truncation when a step overshoots). The Picard/Newton IN-RESIDUAL exfiltration (-wtm_direct_to_runoff) remains
  // the operator-consistent alternative (its Jacobian tangent is task #100). The route to the
  // nonphysical unmanaged-free-boundary regime is surface_water.collection.method: off.
  (void)anderson_path;
  // The post-solve clamp. -wtm_surface_exfiltration_to_runoff is RETIRED: it was the interface of the
  // `legacy` mode, which is gone, and surface_water.collection.method: explicit is the documented route
  // (verified byte-identical to the flag, max|d| = 0.000e+00).
  //
  // SET UNCONDITIONALLY BY THE SELECTOR BELOW, which is why dev.allow_aboveground_water_columns was
  // removed (#35): that key was read HERE and then overwritten by every branch of the selector, so it
  // could not affect any run. `collection.method: off` is the route to an unmanaged free boundary, and
  // it is the only one -- one setting, one key.
  g_surface_exfiltration_to_runoff_array = false;

  // -wtm_Tbar: use the step-time-averaged interblock transmissivity T̄ (Kirchhoff-potential difference;
  // see interblockTransmissivity). Composes with any solver. Requires the piecewise Fan T (Φ is its
  // antiderivative), so it is incompatible with ksat smoothing and extended soil. Applies on the Anderson
  // residual, the Picard operator, and the Newton Jacobian.
  // config-owned (solver.t_bar); the -wtm_Tbar flag is retired
  g_Tbar = params.t_bar;
  if (g_Tbar && (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0 ||
                 g_extended_soil))
    throw std::runtime_error("solver.t_bar requires the piecewise Fan transmissivity: remove "
                             "-wtm_ksat_*_smoothing_width and collection.method: extended_soil.");

  // -wtm_T_bedrock: additive background (bedrock) transmissivity floor [m²/s]; default 0 = v2.0.1 (no
  // floor). A constant added to T everywhere, representing the deep crust's small nonzero conductance
  // integrated over the active flow thickness; it removes the deep-cell operator singularity by capping
  // T's dynamic range (e.g. 1e-8 -> ~3.7 orders vs surface). See the block above depthIntegratedTransmissivity.
  // Config-owned (transmissivity.additive_background_transmissivity); the -wtm_T_bedrock flag is GONE.
  g_T_bedrock = params.t_bedrock;
  if (g_T_bedrock < 0.0)
    throw std::runtime_error("-wtm_T_bedrock must be >= 0 (it is an additive transmissivity floor in m^2/s).");

  // boundaries.land: select the LAND-edge boundary condition (ocean is always Dirichlet h=0). Accepts
  // "neumann_toposlope" (default; terrain-following no-flow) or "dirichlet_sea_level" (ghost head = sea level, the
  // modern ghost-node equivalent of the legacy sea-level padding, imposed at land edges without turning them
  // to ocean). Currently wired into the matrix-free residual (Anderson path); Picard/Newton are guarded off
  // below until their off-map operator/Jacobian tangents are extended.
  g_land_boundary_dirichlet = params.land_boundary_dirichlet;  // config-owned (boundaries.land)
  // Land Dirichlet is wired into all three solver paths: the matrix-free residual (Anderson/TR-BDF2), the
  // Newton analytic Jacobian (FD-verified), and the Picard operator+RHS (diagonal absorbing conductance).

  // -wtm_direct_to_runoff: in-residual exfiltration removal (supersedes the qmax sink where on). Removes the
  // above-surface excess (max(0,wtd)) to runoff each step, holding the table AT the surface with no rate cap
  // and no below-surface band -> no pile, no depression. OPT-IN (default off): its removal tangent is NOT
  // yet wired into the Picard operator / Newton Jacobian (directToRunoffTangent is unused), so defaulting it
  // on for those paths makes their solve inconsistent (Newton diverges/aborts). The Picard/Newton default
  // surface-water clamp remains the -wtm_surface_sink taper (which DOES carry tangents); wiring the
  // direct_to_runoff tangent so it can default on for those paths is future work.
  // The in-residual siphon. -wtm_direct_to_runoff is RETIRED alongside the `legacy` mode it belonged to;
  // surface_water.collection.method: implicit is the route (verified byte-identical). Set by the selector.
  g_direct_to_runoff = false;

  // surface_water.routing, DEFAULT continuous (Andy, 2026-09-04, reaffirmed after review).
  //
  // THE PHYSICAL ARGUMENT IS THE ARGUMENT. FSM is instantaneous by construction, so under `impulse` the
  // state always carries a FULLY EQUILIBRATED lake -- a depression is full from the instant there is
  // water to fill it, and evaporates at the open-water rate for the whole step. Real water flows in over
  // the interval and evaporates as it arrives. `continuous` encodes that; `impulse` cannot.
  //
  // WHAT THE MEASUREMENTS SAY, including the ones that do NOT support this choice, so the next reader
  // gets the whole picture rather than the case for the verdict:
  //   * the two couplings CONVERGE, first order, to the same continuum (tests/coupling_convergence):
  //         stored_volume  4.477e-02 -> 1.733e-02 -> 6.706e-03   (gap, /recharge)
  //         evap_removed   5.428e-02 -> 2.529e-02 -> 1.427e-02
  //     so they are two discretisations of one physics, and neither is a different model.
  //   * ACCURACY at usable dt is a WASH. Richardson-extrapolating to the limit and asking which sits
  //     closer at dt = 1 yr, same verdict on both the 8 yr and 20 yr fixtures:
  //         stored_volume  continuous closer     evap_removed     impulse closer
  //         ocean_outflow  continuous closer     surface_removed  impulse closer
  //     Two columns each. There is no accuracy case for either coupling.
  //   * the ~11% evaporation figure that ORIGINALLY justified this default did NOT survive. It was a
  //     defect -- the active-set obstacle destroying water the FSM delta was separately moving (19ee097)
  //     -- and the sign has not been reproduced. See #51, still open.
  //
  // WHY NOT `impulse`, given it is the tidier default. A case was made for it on robustness and was
  // REJECTED as tidiness rather than correctness, which is right:
  //   - "composes with every collector", "no estimator blind spot", "matches v2.0.1" are properties of
  //     the SOFTWARE and the tooling, not evidence about which model is closer to the world.
  //   - "resets cross-rank drift" is actively misleading: impulse does not HAVE less drift, it
  //     re-broadcasts the rank-0 table every step and hides it. See task #39, which measured exactly
  //     that -- impulse's cross-rank agreement is an artefact of the broadcast.
  //   - "fate-invariant to the solve count" is not independent evidence: impulse earns that invariance
  //     BY re-equilibrating the lake every step, which is the very thing being disputed.
  //
  // THE COSTS ARE REAL AND ARE NOT HIDDEN. `continuous` works only with collection.method: active_set
  // (refused with explicit, #44; budget does not close with implicit, xfail in tests/budget_closure);
  // its budget is not fate-invariant to the solve count, which is why tests/dt_invariance is scoped to
  // impulse and tests/coupling_convergence carries continuous's conservation instead; and it costs the
  // adaptive error estimator a blind spot over the FSM-delta cells (e8d568b). Those are the price of the
  // physics, paid deliberately.
  //
  // What did NOT decide it: source does not restore 2nd order (1.16/1.25/1.60 against overwrite's
  // 1.13/1.23/1.59), and its flicker benefit is already spent by active_set, which is the default collector.
  // Both couplings reach the same equilibrium, so this matters for TRANSIENTS far more than equilibrium.
  // Resolved AFTER the collector below, because `auto` has to know it. Read the request here.
  const PetscBool fsm_cont_set = params.fsm_coupling_set ? PETSC_TRUE : PETSC_FALSE;
  g_fsm_continuous = params.fsm_coupling_continuous;

  // Runoff-collection selector (config key `surface_water.collection.method`, optional; the internal
  // variable is still called runoff_collector). When set it OVERRIDES the
  // legacy -wtm_ flags above with one coherent choice; unset ("") keeps the legacy defaults
  // (behaviour-preserving). See benchmark/SURFACE_WATER_ROUTING.md. All modes share one destination
  // (above-surface excess -> total_surface_removed + arp.runoff -> FSM); they differ in HOW the wtd<=0
  // exfiltration constraint is enforced:
  //   implicit : in-residual exfiltration (-wtm_direct_to_runoff) ALONE, pins wtd=0, dt-INDEPENDENT and exact. Its
  //              exfiltration kink is matrix-free only for now (Anderson); on Picard/Newton it is inconsistent
  //              (needs active-set Newton), so WARN there. NO explicit backstop: mixing the two would let the
  //              post-solve clamp silently mop up any implicit overshoot and HIDE an implicit bug, so the modes
  //              are mutually exclusive -- if implicit misbehaves the water visibly piles.
  //   explicit : post-solve clamp only (-wtm_surface_exfiltration_to_runoff). Robust on every solver; the
  //              lateral flow does not see the pin during the solve, so it is a lower-order (dt-lagged) form
  //              (~1 cm from implicit here, converging as dt->0).
  //   off      : no collection -- above-surface water piles up. NONPHYSICAL; warn loudly.
  //   extended_soil : also lets water pile, but continues the AQUIFER above the surface (storativity stays
  //              porosity, T never clamps, recharge always fills pore space), so the wtd=0 free boundary is
  //              removed rather than merely unenforced. That is what restores BDF2's 2nd order. NONPHYSICAL
  //              and [WIP]: the production half (truncate the mound at the FSM handoff, NOT per GW step) is
  //              unimplemented. (`-wtm_extended_soil`, the legacy alias, is RETIRED.)
  // NOTE the sub-surface band sink (taper 1, -wtm_surface_sink) is a SEPARATE strategy (keep wtd<0, dodge
  // the free boundary, stay 2nd-order); the selector turns it OFF in every mode. Retiring it fully (and making
  // a mode the default) is a later, regold-bearing step.
  std::string rc = params.runoff_collector;
  if (rc.empty()) rc = "active_set";  // "" = the default (see parameters.hpp for why active_set)
  // SOLVER-DEPENDENT DEFAULT RESOLUTION. The active-set pin lives in the matrix-free (Anderson)
  // residual only -- the Picard operator and Newton Jacobian carry no tangent for it, and this block
  // also switches every collector removal off, so those solvers would run with the constraint
  // effectively UNENFORCED. That is not a degradation but a hard failure: Newton ABORTS (verified on
  // tests/boundary_consistency, which core-dumped the moment the default flipped). A default must not
  // crash a supported path, so the PICARD DEFAULT is `explicit` -- the post-solve clamp, which is
  // robust on every solver. This is a solver-dependent DEFAULT, not a downgrade: an omitted key means
  // "the default", and which default applies follows from the solver, exactly as it does for
  // solver.time_integration and solver.time_step.mode. The full set is tabulated in config.yaml.
  // An EXPLICIT surface_water.collection.method is always honoured, so this only ever decides what an
  // unspecified config does.
  if (rc == "active_set" && !params.runoff_collector_set && user_context.use_picard) {
    rc = "explicit";
    static bool noted_downgrade = false;
    if (!noted_downgrade) {
      noted_downgrade = true;
      // NOT a downgrade: this IS the Picard default. Defaults are solver-dependent by design -- an
      // omitted key always means "the default", and which default applies follows from the solver,
      // because the solvers do not support the same machinery. See the default-set table in
      // config.yaml. Saying "downgrade" made an ordinary default read like a silent substitution.
      PetscPrintf(PETSC_COMM_WORLD,
                  "NOTE: surface_water.collection.method defaults to `explicit` on the Picard solver "
                  "(the post-solve clamp), because the active_set pin is absent from the Picard "
                  "operator/RHS. Defaults are solver-dependent; set the key explicitly to override.\n");
    }
  }
  bool collector_wants_active_set = false;  // set by rc == "active_set"; enabling happens below
  // RETIRED (fork issue #7): the taper-1 band sink is now OFF unconditionally, in `legacy` too. Its band
  // width is w = 2*qmax*dt and that dt-scaling is INTRINSIC -- a fixed width overshoots for a rate-capped
  // smooth sink -- so its equilibrium water table is dt-DEPENDENT (measured in #7: a plateau interior at
  // -1.56 m at dt = 1 yr vs -0.79 m at dt = 0.25 yr, against a dt-independent pure groundwater solve).
  // Its purpose was to dodge the wtd=0 free boundary rather than solve it, giving Picard/Newton a
  // differentiable tangent. #7 prescribed the replacement -- "a primal-dual active-set / semismooth
  // Newton for the complementarity wtd <= 0 _|_ seepage >= 0 ... no smoothing, dt-independent" -- and
  // that is -wtm_active_set, now the DEFAULT and carrying the pin in the Newton Jacobian
  // (FormJacobianLocal). The niche is gone. With the sink off, `legacy` collapses exactly onto
  // explicit/implicit (verified byte-identical, max|d| = 0.000e+00).
  {  // one mode is always in force: `legacy` (hand control to the -wtm_ surface flags) is RETIRED
    // The selector OWNS extended soil now, exactly as it owns the three removals: exactly one mode is
    // in force, so a config that names a different method turns extended soil off rather than leaving
    // two contradictory mechanisms running and letting whichever clamps last win.
    g_extended_soil = (rc == "extended_soil");
    if (rc == "implicit") {
      g_direct_to_runoff                     = true;
      g_surface_exfiltration_to_runoff_array = false;  // exclusive: no clamp backstop (keep implicit's bugs visible)
      if (user_context.use_newton)
        PetscPrintf(PETSC_COMM_WORLD, "WARNING [surface_water.collection.method=implicit]: the exfiltration tangent is wired into "
                    "the Anderson residual and the Picard operator, but NOT the Newton Jacobian (its kink needs "
                    "active-set Newton). On the Newton path the solve is inconsistent (may diverge); use "
                    "surface_water.collection.method: explicit there until active-set Newton lands.\n");
    } else if (rc == "explicit") {
      g_direct_to_runoff                     = false;
      g_surface_exfiltration_to_runoff_array = true;
    } else if (rc == "active_set") {
      // The semismooth constraint solved INSIDE the residual (see the block below, which does the
      // actual enabling -- this only records the request, since g_active_set is assigned there).
      // It supersedes all three collector removals; that happens below too. Measured to be the only
      // enforcement whose equilibrium does not carry a spurious dt-dependence: the `implicit` siphon
      // leaves a head ~ LINEAR in dt, which FSM routes into a different set of lakes (the lake COUNT
      // itself moves with dt -- tests/multilake). Anderson residual only; Picard/Newton have no
      // tangent for the pin, warned below.
      collector_wants_active_set = true;
    } else if (rc == "extended_soil") {
      // Sibling of `off`: neither enforces wtd<=0, and above-surface water piles up. They differ in the
      // PHYSICS above the surface. `off` keeps standard physics (storativity jumps porosity->1, T clamps
      // at wtd=0), so the pile is surface water sitting on a kinked coefficient. `extended_soil`
      // continues the aquifer upward (storativity stays porosity, T never clamps, recharge always fills
      // pore space), which removes the wtd=0 free boundary entirely and is why it restores BDF2's 2nd
      // order -- 2.07/2.01/2.00 against ~1 for a pinned surface (see BDF2_RECHARGE_ORDER.md section 15).
      // NONPHYSICAL and [WIP]: the mound is real storage the model then owes to FSM, and the production
      // half -- truncating it back to topography AT THE FSM HANDOFF (not per GW step) -- is not
      // implemented. Do not use it for model runs; it is a diagnostic that establishes the order ceiling.
      g_direct_to_runoff                     = false;
      g_surface_exfiltration_to_runoff_array = false;
      PetscPrintf(PETSC_COMM_WORLD, "WARNING [surface_water.collection.method=extended_soil]: NONPHYSICAL developer mode -- the "
                  "aquifer continues above the land surface (no free boundary, no surface water). The above-surface "
                  "mound is real storage the model owes to FSM, and the production half (truncate it at the FSM "
                  "handoff) is NOT implemented. Testing/experiments only, not for model runs.\n");
    } else {  // "off"
      g_direct_to_runoff                     = false;
      g_surface_exfiltration_to_runoff_array = false;
      PetscPrintf(PETSC_COMM_WORLD, "WARNING [surface_water.collection.method=off]: NONPHYSICAL -- above-surface water is NOT "
                  "collected; it piles up and the free surface will limit-cycle. Testing/diagnostics only.\n");
    }
  }

  // dev.storage_form -- which ASSEMBLY the backward-Euler storage term uses. NOT an accuracy choice: the
  // two forms are the SAME EQUATION, because S is the EXACT SECANT
  // S = (V(w^{n+1}) - V(w^n)) / (w^{n+1} - w^n), so S·Δh ≡ ΔV identically -- even across the surface,
  // where dV/dh jumps porosity→~1. tests/storage_equivalence pins that at max|Δwtd| = 0.000e+00 m.
  //
  // THIS COMMENT PREVIOUSLY CLAIMED THE OPPOSITE, with numbers ("at a surface CROSSING the secant
  // S·Δh ≠ ΔV ... Esquibel: mean ~0.11 m, tails ~19 m"). That claim is RETRACTED -- see
  // tests/storage_equivalence, which exists to disprove it -- and the stale text was believed and
  // repeated twice while designing the config schema before the test was found. A retracted result left
  // in a comment is worse than no comment: it is indistinguishable from a measurement.
  //
  // What DOES differ is only the residual assembly: `volume` folds the storage into f and leaves RHS
  // b = 0; `secant` puts the previous-step storage in b = h^n. The active-set constraint needs a b=0
  // path, which is the ONLY reason the choice is load-bearing. It is therefore a DEV key: its one
  // legitimate consumer is tests/storage_equivalence, which bites if updateEffectiveStorativity ever
  // stops being the exact secant (a tangent or endpoint storativity would break the identity and make
  // the default BE silently inconsistent with the volume schemes).
  g_volume_storage = params.volume_storage;  // dev.storage_form; default volume

  // collection.method: active_set -- enforce the wtd<=0 exfiltration constraint as an ACTIVE-SET /
  // semismooth constraint INSIDE the solve: a cell whose iterate rises above the land surface is pinned at
  // wtd=0 by overriding its residual with f = w_c (mirroring the ocean Dirichlet), instead of a post-solve
  // clamp (`explicit`) or an in-residual siphon (`implicit`). It is the DEFAULT, and the only enforcement
  // measured to leave no spurious dt-dependence.
  //
  // -wtm_active_set and -wtm_dev_active_set are both RETIRED. The flag used to be an ORTHOGONAL switch that
  // superseded whatever collector was configured; as a member of the collection.method enumeration it is
  // simply one mode among six, mutually exclusive with the others. That is the point -- "active_set AND a
  // collector" is now unrepresentable rather than resolved by precedence -- but it does mean the old
  // supersession behaviour is gone rather than renamed.
  g_active_set         = collector_wants_active_set;
  g_collector_resolved = rc;

  // STATE THE ENFORCEMENT, ONCE PER RUN. This is the single most consequential surface-water choice --
  // it moves the equilibrium head and, through FSM, the LAKE COUNT (tests/multilake) -- and until now a
  // run recorded it NOWHERE that a user sees: g_collector_resolved went only to the coverage file. A run
  // whose log cannot say which boundary condition produced it is not reproducible from its own output.
  // Naming the SOURCE as well as the value is the point: "active_set (default)" and "active_set (config)"
  // are different runs to anyone auditing a result later.
  {
    static bool announced = false;
    if (!announced) {
      announced = true;
      const char* src = params.runoff_collector_set ? "surface_water.collection.method"
                                                    : "default -- no method configured";
      PetscPrintf(PETSC_COMM_WORLD, "surface-water exfiltration enforcement: %s  [%s]\n",
                  g_collector_resolved.c_str(), src);
    }
  }
  // Active-set needs a b=0 residual path (the SNES RHS = 0), so the residual f driven to zero IS the mass
  // balance -- the semismooth max(w_c, f) and the captured exfiltration f*Sy are only meaningful then. The default
  // secant backward-Euler uses RHS b = h^n (f != residual). If no b=0 scheme is already selected, auto-enable
  // the exact volume-storage BE (b=0, 1st-order, same limit as TR-BDF2/BDF2-on-V; see finding_cc_secant_...).
  // dev.storage_form: secant + active_set is REFUSED, not resolved. It used to auto-enable volume and
  // print a NOTE -- i.e. silently override an explicit user setting, the same shape as the dev.active_set
  // defect (#28), differing only in that it overrode the USER rather than another key.
  //
  // The default is volume, so !g_volume_storage here can only be an EXPLICIT dev.storage_form: secant.
  // The check does not exclude the b=0 integrators (bdf2 / tr-bdf2): on those the storage branch is never
  // reached, so an explicit `secant` would be silently void rather than honoured, which is the same
  // failure by a quieter route.
  // surface_water.routing, resolved when the key is ABSENT (the enum itself is continuous|impulse|off;
  // there is no `auto` value). `continuous` is what we want everywhere it is valid, but it is
  // REFUSED with collection.method: explicit (below), and `explicit` is what solver.method: picard
  // resolves to when the collector is unset. A constant `continuous` default therefore made plain
  // `solver.method: picard` ABORT out of the box -- and Picard is the independent oracle that certifies
  // matrix-free Anderson, so that is not a corner. An absent key yields to `impulse` exactly where
  // continuous cannot run, the same shape as an absent solver.time_integration or solver.adaptive_dt.
  // An EXPLICIT fsm_coupling is never overridden: it falls through to the refusal and the user is told.
  if (fsm_cont_set == PETSC_FALSE && rc == "explicit") {
    g_fsm_continuous = false;
    static bool noted_auto_impulse = false;  // update() runs EVERY step; announce the resolution once
    if (!noted_auto_impulse) {
      noted_auto_impulse = true;
      PetscPrintf(PETSC_COMM_WORLD,
                  "surface_water.routing: absent -> impulse (collection.method: explicit cannot take "
                  "the continuous coupling).\n");
    }
  }

  // fsm_coupling: continuous x infiltration_during_flow: true. The continuous coupling delivers FSM's
  // per-cell volume change through the DISTRIBUTED recharge carrier (fsm_delta_dist), and that carrier
  // only exists on the distributed path: WTM.cpp sets
  //     distribute_recharge = !fsm_on || !infiltration_on
  // so turning infiltration on WITH FSM routes recharge through the serial rank-0 loop instead, where the
  // delta has nowhere to go. The coupling was previously switched off silently at the point of use
  // (WTM.cpp, `fsm_continuous_on() && fsm_on && distribute_recharge`), so `fsm_coupling: continuous` was
  // accepted, ignored, and never mentioned -- MEASURED as byte-identical output to `impulse`, max
  // difference 0.000000e+00 across every budget column. That is the defect class #27 and #28 already
  // ruled against: a key that reads as honoured and is not.
  //
  // Resolve it here instead, where it can be SAID. An EXPLICIT request is refused by name; the default
  // yields to impulse and announces, exactly as it does for the explicit collector above.
  if (g_fsm_continuous && params.fsm_on && params.infiltration_on) {
    if (fsm_cont_set == PETSC_TRUE)
      throw std::runtime_error(
          "config: surface_water.routing: continuous cannot be used with "
          "surface_water.infiltration_during_flow: true. The continuous coupling hands FillSpillMerge's "
          "per-cell volume change to the next step through the DISTRIBUTED recharge carrier, but "
          "infiltration_during_flow routes recharge through the serial rank-0 loop, which has no such "
          "carrier -- so the coupling would be silently inert. Use routing: impulse, or "
          "infiltration_during_flow: false.");
    g_fsm_continuous = false;
    static bool noted_infil_impulse = false;
    if (!noted_infil_impulse) {
      noted_infil_impulse = true;
      PetscPrintf(PETSC_COMM_WORLD,
                  "surface_water.routing: absent -> impulse (infiltration_during_flow: true routes "
                  "recharge serially, which carries no FSM-delta source).\n");
    }
  }

  // fsm_coupling: continuous x collection.method: explicit is REFUSED, because it does not converge. Solution
  // convergence at a fixed 8 yr on tests/fsm_consistency: the other three combinations refine cleanly
  // (observed order 1.4-1.6), while this one stalls at ~1.1 m and its observed order goes NEGATIVE
  // (1.12, -0.10, 0.24) -- refining dt stops helping after dt/2. `explicit` is a POST-SOLVE CLAMP, so under
  // source the above-surface water is neither pinned in the residual nor written into the state between
  // steps: the clamp keeps removing what the source keeps re-adding. Refused by name rather than left
  // reachable now that source is the default. See task #44.
  if (g_fsm_continuous && rc == "explicit")
    throw std::runtime_error(
        "config: surface_water.routing: continuous cannot be used with "
        "surface_water.collection.method: explicit -- the pair does not converge (measured: observed order "
        "goes negative under dt refinement, error stalls at ~1.1 m). `explicit` clamps above-surface water "
        "AFTER the solve, while `continuous` feeds it back in as a source term, so the two fight and the run "
        "never settles to a dt-independent state. Use collection.method: active_set (the default), or "
        "fsm_coupling: impulse.");

  if (g_active_set && !g_volume_storage)
    throw std::runtime_error(
        "config: dev.storage_form: secant cannot be used with surface_water.collection.method: active_set. "
        "The semismooth exfiltration constraint is enforced inside the residual and needs a b=0 residual "
        "path; the secant form puts the previous-step storage in the RHS (b = h^n) instead, so the "
        "constraint would not be enforced. Use dev.storage_form: volume (the default), or a different "
        "surface_water.collection.method. NOTE the two forms are mathematically identical -- S is the exact "
        "secant, so S*dh == dV (tests/storage_equivalence) -- so this is a constraint on the ASSEMBLY, not "
        "a difference in the answer.");
  // Active-set IS the exfiltration enforcement, so it SUPERSEDES the runoff_collector removals -- otherwise the
  // in-residual siphon (implicit) or post-solve clamp (explicit) stack on top of the pin and the result is no
  // longer enforcement-independent. Disable all collector removals when active-set is on; the pinned-cell
  // exfiltration is conserved via the captured-exfiltration transfer instead.
  if (g_active_set) {
    g_direct_to_runoff                     = false;
    g_surface_exfiltration_to_runoff_array = false;
    // HARD ERROR on the solvers that cannot carry the pin. It lives in the matrix-free (Anderson)
    // residual only; the Picard operator and Newton Jacobian have no tangent for it, and this block also
    // switches every collector removal OFF. So selecting active_set there NEVER actually enforces the
    // constraint -- it silently becomes one of two other things, and measurement says neither is what
    // the user asked for:
    //   FSM on  -> FillSpillMerge's between-step overwrite (runoff += wtd; wtd = 0, then re-level) does
    //              the job instead. That is a post-solve projection, i.e. functionally `explicit`, and
    //              it is NOT luck -- FSM duplicates that function. Measured on tests/multilake: Picard
    //              under active_set and under explicit both give 6 lakes (stages differ in detail).
    //   FSM off -> nothing does the job. Measured on the same fixture with FSM off: Picard piles water
    //              over 1440 of ~1444 land cells (ponded 37.48 m) where explicit holds max wtd at 0.000.
    // Erroring rather than warning is the honest option: silently resolving an EXPLICIT choice to
    // something else would violate "an explicit choice is always honoured", and a warning in front of
    // the nonphysical `off` mode is a warning users will scroll past. (The DEFAULT never lands here --
    // it resolves to `explicit` on these solvers further up.)
    // TRIED AND REVERTED (2026-08-26) -- record so the next attempt starts here rather than repeating it.
    // Adding the active-set ROW to the Picard operator/RHS is the easy half and it does work: assemble the
    // pinned row as a single diagonal entry with b = topo + surface_water_depth, AREA-SCALED (every other
    // row is volume-form, ~1e6-1e8 m^2; an unscaled identity row against a ~90 m head puts 8 orders of
    // magnitude of row scaling in the matrix and CG/GAMG dies -- that scaling was the whole reason the
    // first three attempts failed identically at 12 iterations). With it scaled, Picard CONVERGES (607
    // iterations, 1.8 s) and reproduces Anderson's lake TOPOLOGY (4 lakes) on tests/multilake.
    //
    // It is still WRONG, and the missing piece is not a tangent. Pinning discards the cell's mass balance,
    // so the water the constraint removes must be recovered and handed to FSM. On the Anderson path that
    // is the multiplier read straight off the residual (my_exfiltration = max(0, -f*Sy) in
    // FormFunctionLocal). The Picard formulation never forms a residual, so there is nothing to read:
    // measured total_surface_removed = 0.0 (Anderson: 2.0e12) and an exact budget residual of 93% of
    // recharge, with lake stages 97.5-99.3 m against Anderson's 90.9-97.2 m. Recovering the multiplier
    // post-solve -- evaluating the free row that was discarded, for pinned cells only -- is the real work,
    // and it is a new mechanism rather than a derivative. Two further caveats found on the way: the
    // operator becomes NONSYMMETRIC (neighbours keep entries in the pinned column, and it cannot be
    // symmetrised as ocean cells are, because MatZeroRowsColumns with x=b=NULL is valid only for an
    // imposed value of zero), so the CG+GAMG default is no longer justified; and CG vs GMRES made no
    // difference to any of the failures, so symmetry was never the binding constraint.
    if (user_context.use_picard)
      throw std::runtime_error(
          "surface_water.collection.method: active_set is not supported on the Picard solver. The pin is "
          "absent from the Picard operator and RHS (a separate formulation from the residual), so the "
          "exfiltration constraint would never be enforced -- it would silently fall through to "
          "FillSpillMerge's overwrite (with FSM on) or to no enforcement at all (with FSM off; measured: "
          "water piles over 1440 of ~1444 land cells). Use surface_water.collection.method: explicit on "
          "Picard, or run the Anderson solver.");
    // Newton is DIFFERENT from Picard here, and the distinction is easy to lose: the Newton RESIDUAL is
    // FormFunctionLocal, the same function that carries the pin, so Newton has always ENFORCED the
    // constraint -- it merely differentiated a different function. FormJacobianLocal now carries the
    // matching semismooth tangent, so the pair is consistent and active_set is supported there.
  }

  // fsm_continuous x active_set: the hard error here is GONE (2026-09-03, #40). It existed because the
  // active-set obstacle was INFERRED as max(0, starting_wtd) -- the table FSM overwrites -- and
  // fsm_continuous exists to suppress that overwrite, so the obstacle collapsed to the land surface and
  // every lake drained (5.6986 m -> 0.0000 m). The obstacle now reads the CARRIED lake_stage, which FSM
  // writes under both couplings, so suppressing the overwrite no longer blinds the pin.
  //
  // MEASURED on tests/fsm_consistency inputs, anderson + active_set + FSM, 600 yr, fixed dt:
  //   lakes SURVIVE          max wtd 3.3049 m (source) vs 3.4362 m (overwrite) at 120 yr
  //   both SETTLE            per-cycle |S.dwtd| max decays monotonically; source is smaller at every
  //                          sample (0.02397 vs 0.02511 at 600 yr), consistent with the Lie-split jump
  //                          being gone -- the skim/return path does NOT cycle water indefinitely
  //   both CONSERVE          exact_budget_residual / recharge = 5.656e-08 (source) vs 7.926e-08
  //
  // IT IS NOT ANSWER-NEUTRAL, and the difference has the shape Andy predicted. Lake stage over time:
  //   during rapid drawdown source runs 1-8% LOWER (returned water arrives a step late, and some is
  //   skimmed straight back out -- it cycles); near the residual stage it settles ~21% HIGHER
  //   (0.183 m vs 0.151 m -- the gradual return slows the lowering). Which coupling is right is a
  //   MODELLING judgement, not a numerical one; overwrite remains the default.

  // -wtm_relax: sub-step under-relaxation of the water table (w <- a*w_solve + (1-a)*w_prev). a=1 is off
  // (byte-identical). a<1 damps the period-2 flicker at pinned free boundaries (lakeshore / exfiltration). At
  // steady state w_solve=w_prev so the fixed point (equilibrium) is unchanged; only the transient is damped.
  g_relax = params.under_relaxation;

  // Everything above has resolved; record what this run actually is. Once per process.
  {
    static bool coverage_emitted = false;
    if (!coverage_emitted) { coverage_emitted = true; emit_coverage_fingerprint(params, user_context); }
  }

  // Taper 2 (on by default): implicit demand-identity evaporation (ET -> owe). Read here AND early in
  // WTM.cpp::initialise() (before the initial recharge) via the same call, so the explicit-recharge
  // sites -- including irf.cpp's initial pass -- all see a consistent flag. See SURFACE_SINK_DESIGN.md 14.
  read_evap_taper_options(params);

  // Whether the sink was actually applied THIS solve (it lives only in the BDF2-on-V branch, which
  // needs an established history -- the BE bootstrap step has no sink). Captured before the solve,
  // since the copy-back below sets bdf2_have_history for the NEXT step. Used to account the removed
  // water in the same step it was removed.
  // Where the removals actually act, so we account exactly what the solve removed:
  //  * matrix-free path (Anderson/Newton, !use_picard): FormFunctionLocal applies them EVERY solve.
  //  * Picard path: only in the BDF2-on-V branch, once a history exists (the BE bootstrap has none).
  const bool matrix_free   = !user_context.use_picard;
  const bool picard_bdf2_V = user_context.use_bdf2 && user_context.bdf2_have_history && user_context.use_bdf2;
  const bool sink_active_this_step = g_direct_to_runoff && (matrix_free || picard_bdf2_V);
  const bool evap_active_this_step = g_evap_taper && (matrix_free || picard_bdf2_V);

  // -wtm_Tbar: ghost-scatter w^n (starting_wtd) so a neighbour's time-averaged T̄ can read its w^n
  // under MPI. w^n is fixed for this step (set above by set_starting_values), so scatter ONCE here
  // rather than per SNES iteration. starting_wtd is checked out read-write by DMDA_Array_Pack, but
  // this scatter only READS it (global → local), which is safe alongside the outstanding pointer.
  if (g_Tbar || user_context.use_tr_bdf2) {  // ghosted w^n: T̄, TR-BDF2 old
    // flux, and the predictor's first-step forward-Euler bootstrap (compute_tr_explicit) all read it.
    DMGlobalToLocalBegin(user_context.da, user_context.starting_wtd, INSERT_VALUES, user_context.starting_wtd_local);
    DMGlobalToLocalEnd(user_context.da, user_context.starting_wtd, INSERT_VALUES, user_context.starting_wtd_local);
  }

  if (user_context.use_picard) {
    // Semi-implicit Picard path (PICARD_MATH.md).
    // PETSc solves A(x) x = b(x); FormPicardRHS supplies b(x) (so SNESSolve is
    // called with a NULL rhs), FormPicardOperator supplies the SPD A(x). A is its
    // own preconditioner (GAMG). Inner solve defaults to CG+GAMG (CreateSNES).
    SNESSetPicard(
        user_context.snes,
        user_context.picard_r,
        FormPicardRHS,
        user_context.picard_A,
        user_context.picard_A,
        FormPicardOperator,
        &user_context);

    FormInitialGuess(&user_context, user_context.da, user_context.x);
    SNESSolve(user_context.snes, nullptr, user_context.x);
  } else {
    // Set local function evaluation routine (always needed).
    DMDASNESSetFunctionLocal(
        user_context.da,
        INSERT_VALUES,
        (PetscErrorCode(*)(DMDALocalInfo*, void*, void*, void*))FormFunctionLocal,
        &user_context);

    // Newton-Krylov path (solver.method: newton): register the analytic Jacobian of FormFunctionLocal. The
    // Jacobian (FormJacobianLocal) is the exact ∂F/∂x of the conservative-FV residual including the
    // sink/evap-taper tangents; verify it against FD with -snes_test_jacobian (see FormJacobianLocal).
    // Anderson (snes_type == SNESANDERSON) is matrix-free and skips this. Any OTHER non-Anderson
    // type reaching here WITHOUT solver.method: newton is refused: it would drive a Newton solve with no
    // registered Jacobian (PETSc would fall back to a full FD Jacobian -- prohibitively slow).
    SNESType snes_type;
    SNESGetType(user_context.snes, &snes_type);
    const bool is_anderson = (std::string(snes_type) == std::string(SNESANDERSON));
    if (!is_anderson) {
      if (!user_context.use_newton) {
        throw std::runtime_error(
            std::string("The Newton-Krylov solver (-snes_type ") + snes_type +
            ") needs solver.method: newton to register its analytic Jacobian. Use the default Anderson solver, "
            "solver.method: picard for the semi-implicit (BDF2-on-V) path, or solver.method: newton.");
      }
      DMDASNESSetJacobianLocal(
          user_context.da,
          (PetscErrorCode(*)(DMDALocalInfo*, void*, Mat, Mat, void*))FormJacobianLocal,
          &user_context);
    }

    // Evaluate initial guess
    FormInitialGuess(&user_context, user_context.da, user_context.x);

    // Volume-weighted per-solve convergence (#127), opt-in. Registered here so it covers the plain + TR-BDF2
    // paths; the adaptive-restart branch below installs ITS own test (so volume-conv applies only to the
    // ordinary production solve). Registered if EITHER half is wanted -- output.trace: [water_step] to
    // print, solver.convergence.metric: volume to decide -- because the test computes the water step that
    // both need. The two are independent; see VolumeStepConverged and the AppCtx note.
    if ((user_context.vol_step_trace || user_context.snes_volume_conv_govern) && !user_context.use_adaptive_restart)
      SNESSetConvergenceTest(user_context.snes, VolumeStepConverged, &user_context, nullptr);

    // set the RHS (b = h^n for backward Euler; b = 0 for the self-contained BDF2-on-V / TR-BDF2 residuals)
    FormRHS(&user_context, user_context.da, user_context.b);
    if (user_context.use_tr_bdf2) {
      // TR-BDF2: two staged implicit solves per step. Precompute the explicit old-state flux the
      // trapezoidal stage needs (from the ghosted w^n scattered above), then solve stage 1 for the
      // intermediate Y_gamma, store it, and solve stage 2 (BDF2 from w^n and Y_gamma) for w^{n+1}.
      compute_tr_explicit(user_context);
      user_context.tr_stage = 1;  // trapezoidal → Y_gamma (initial guess = FormInitialGuess above)
      SNESSolve(user_context.snes, user_context.b, user_context.x);
      SNESConvergedReason stage1_reason;
      SNESGetConvergedReason(user_context.snes, &stage1_reason);
      if (stage1_reason < 0) {
        user_context.tr_stage = 0;
        // -wtm_dt_adaptive: a non-converged stage is a REJECT (shrink dt, retry from the unchanged
        // state) not a fatal error. State is not committed until below, so w^n is preserved for the retry.
        if (user_context.use_dt_adaptive) {
          user_context.deltat *= user_context.dtc_shrink;
          return -1;
        }
        throw std::runtime_error("TR-BDF2 trapezoidal stage (1) did not converge.");
      }
      VecCopy(user_context.x, user_context.tr_ygamma);  // Y_gamma carried into stage 2
      // -wtm_active_set: BOTH stages pin, and BOTH discard water into the pin. The step's exfiltration is
      // E = C1*E1 + E2 (derived in src/tr_bdf2_coefficients.hpp; C1*gamma = 70.71% of the step is carried
      // by stage 1). exfiltration_vec is rewritten by every residual evaluation, so E1 has to be taken now,
      // before stage 2 overwrites it -- and taken by an EXPLICIT residual evaluation at the accepted
      // Y_gamma, because the solver's last evaluation may have been at a rejected trial iterate.
      if (g_active_set) {
        if (!user_context.tr_exfil_stage1) VecDuplicate(user_context.x, &user_context.tr_exfil_stage1);
        if (!user_context.tr_fwork) VecDuplicate(user_context.x, &user_context.tr_fwork);
        SNESComputeFunction(user_context.snes, user_context.x, user_context.tr_fwork);  // tr_stage still 1
        VecCopy(user_context.exfiltration_vec, user_context.tr_exfil_stage1);
      }
      user_context.tr_stage = 2;  // BDF2 → w^{n+1} (initial guess = Y_gamma, already in x)
      SNESSolve(user_context.snes, user_context.b, user_context.x);
      if (g_active_set) {
        // E2 at the accepted w^{n+1}, then combine into the STEP multiplier the commit block transfers to
        // FillSpillMerge. Both stage multipliers are >= 0 (each is a max(0, .)), so the sum is too.
        SNESComputeFunction(user_context.snes, user_context.x, user_context.tr_fwork);  // tr_stage still 2
        VecAXPY(user_context.exfiltration_vec, trbdf2::WE_STAGE1, user_context.tr_exfil_stage1);
      }
      user_context.tr_stage = 0;
    } else if (user_context.use_adaptive_restart) {
      // rho-triggered proactive restart: run Anderson in phases, restarting the history (each fresh
      // SNESSolve resets it) from the BEST iterate whenever the RATE degrades (rho -> 1, the flail
      // precursor) or the phase cap is hit. Converges where a single Anderson phase flails, and -- unlike
      // the fixed-period default -- adapts to a flail that arrives at an unknown iteration (global). #87.
      SNESSetConvergenceTest(user_context.snes, AdaptiveRestartTest, &user_context, nullptr);
      user_context.ar_best_valid = PETSC_FALSE;
      user_context.ar_best_norm  = 0.0;
      SNESConvergedReason r = SNES_CONVERGED_ITERATING;
      PetscReal prev_best = PETSC_MAX_REAL;
      bool true_conv = false, hard_fail = false;
      for (int phase = 0; phase <= user_context.ar_max_restarts; phase++) {
        SNESSolve(user_context.snes, user_context.b, user_context.x);
        SNESGetConvergedReason(user_context.snes, &r);
        if (user_context.ar_stop_kind == 1) { true_conv = true; break; }   // step-relative true convergence
        if (r < 0 && !user_context.ar_best_valid) { hard_fail = true; break; }  // failed before any usable iterate
        // A restart re-solves from the SAME best iterate with cleared history, so once a restart no longer
        // lowers the best residual, further restarts are deterministic repeats -- stop thrashing. The
        // rho/phase-cap tests already keep a stuck phase short; this stops the OUTER loop from spinning
        // through all ar_max_restarts (and, before, then throwing) once we are at the achievable floor.
        const bool improved = user_context.ar_best_norm < prev_best * (1.0 - 1e-6);
        prev_best = user_context.ar_best_norm;
        if (r < 0) break;                              // a phase diverged, but we HAVE a good earlier iterate
        if (!improved && phase > 0) break;             // stagnated at the residual floor
        if (phase == user_context.ar_max_restarts) break;  // out of restarts
        VecCopy(user_context.ar_best_x, user_context.x);   // restart from the best iterate
      }
      // Robust finish. On true convergence, leave x and the reason exactly as the solver set them
      // (byte-identical to a single solve). Otherwise -- unless the solve failed before producing ANY
      // usable iterate (hard_fail) -- return the BEST iterate found and mark the step converged.
      // Stagnating at the residual floor or exhausting restarts near equilibrium (where the Anderson step
      // floors just ABOVE the relative step tolerance, so true convergence is never formally declared) is a
      // NORMAL outcome, not a fatal error: the point of tracking ar_best_x is to hand it back, and the
      // per-cycle equilibrium test ends the run. A NaN/line-search failure that still left a good earlier
      // iterate falls back to it (with a warning) rather than aborting a long spin-up. Only a failure with
      // no usable iterate at all propagates to the throw/reject below.
      if (!true_conv && !hard_fail && user_context.ar_best_valid) {
        VecCopy(user_context.ar_best_x, user_context.x);
        if (r < 0)
          PetscPrintf(PETSC_COMM_WORLD,
                      "solver.anderson.restart: a phase diverged (%s); fell back to the best iterate.\n",
                      SNESConvergedReasons[r]);
        SNESSetConvergedReason(user_context.snes, SNES_CONVERGED_FNORM_RELATIVE);
      }
      PetscPrintf(PETSC_COMM_WORLD, "solver.anderson.restart: best residual %g\n", (double)user_context.ar_best_norm);
    } else {
      // Solve nonlinear system (single implicit solve)
      SNESSolve(user_context.snes, user_context.b, user_context.x);
    }
  }

  SNESGetIterationNumber(user_context.snes, &its);
  SNESGetConvergedReason(user_context.snes, &reason);

  PetscPrintf(
      PETSC_COMM_WORLD, "%s Number of nonlinear iterations = %" PetscInt_FMT "\n", SNESConvergedReasons[reason], its);

  if (reason != 2 && reason != 3 && reason != 4) {
    // Newton dt-continuation drives the step; a non-converged step is a REJECT (the caller shrinks dt
    // and retries from the unchanged state), not a fatal error. Return a negative sentinel WITHOUT
    // committing (the state commit is below, after this check, so starting_wtd is preserved for the
    // retry). Every other path still throws -- their callers do not handle a failure return.
    if (user_context.use_newton_continuation) return -1;
    // -wtm_dt_adaptive: same reject/retry contract -- shrink dt here and return without committing;
    // the caller rolls back the step's accumulators and retries the same step at the smaller dt.
    if (user_context.use_dt_adaptive) {
      user_context.deltat *= user_context.dtc_shrink;
      return -1;
    }
    throw std::runtime_error("The SNES solver has not converged.");
  }

  // -wtm_dt_adaptive + TR-BDF2: embedded local-error estimate from the two stages (no history needed;
  // valid on step 1). A linear extrapolation through (t_n, h^n) and (t_n+gamma*dt, Y_gamma) to t_n+dt is
  //   h_pred = [Y_gamma - (1-gamma) h^n] / gamma   -- EXACT for linear-in-time, O(dt^2) for curvature,
  // so |h^{n+1} - h_pred| is the local truncation error. The error norm covers ALL land cells INCLUDING
  // the free surface: stability is SET at the free surface, so excluding it blinds the controller to the
  // very overshoot that rings -- it then grows dt into the ring (measured: island cold start rang forever
  // at 14843 iters with the surface excluded; including it settles monotonically in 1547). A SETTLED
  // clamped cell has h_pred ~= h^{n+1} => deviation ~= 0, so it costs nothing on a warm transient (measured
  // byte-identical to the excluded norm at dt_tol 0.5/5/20); only a TRANSITIONING/ringing surface cell
  // spikes, which correctly shrinks dt. est > dt_tol => accuracy REJECT (shrink, retry, no commit); else
  // ACCEPT and grow toward the tolerance (capped by the step's convergence headroom and dtc_dt_max).
  // ACCURACY (option 2) on top of the reject/retry CONVERGENCE floor (option 1) above. See BDF2_ADAPTIVE_DESIGN.md.
  // ESTIMATE (method-specific) -> a single scalar `est`; the CONTROLLER below is method-AGNOSTIC. This is
  // the detachment: the integrator (cc / TR-BDF2 / BDF2-on-V) is chosen by its own flags and only supplies
  // the local-error estimate; the grow/shrink/reject logic is identical for all of them.
  double est      = 0.0;
  // The two components kept SEPARATE (#63), because only one of them answers to dt. est_int is the
  // integrator's local truncation error, O(dt^2) -- shrinking dt reduces it. est_cpl is the FSM COUPLING
  // deviation, and the FSM delta is delivered WHOLE regardless of dt, so est_cpl is O(1) in dt: no amount
  // of shrinking touches it. A controller may only STEER on error it can control.
  double est_int  = 0.0;
  double est_cpl  = 0.0;
  bool   have_est  = false;
  bool   est_valid = false;  // the estimate EXISTS -- at least one cell informed it (see the note below)
  long   est_n     = 0;      // how many cells did
  long   est_cn    = 0;      // of those, the COUPLING population (FSM moved water there)
  // The NEXT step's dt, held back until every accumulator below has finished accounting THIS one.
  // See the DEFERRED note in the controller block.
  double dt_next      = 0.0;
  bool   have_dt_next = false;
  if (user_context.use_dt_adaptive && user_context.use_tr_bdf2) {
    // TR-BDF2 embedded estimate from the two stages (no history needed; valid on step 1):
    //   h_pred = [Y_gamma - (1-gamma) h^n]/gamma  -- EXACT for linear-in-time, O(dt^2) for curvature.
    const double TR_G = trbdf2::GAMMA;
    PetscScalar **yg, **topo_e;
    DMDAVecGetArray(user_context.da, user_context.tr_ygamma, &yg);
    DMDAVecGetArray(user_context.da, user_context.topo_vec, &topo_e);
    double local_max = 0.0, local_sq = 0.0;    // INTEGRATOR population: cells FSM did not touch
    long   local_n = 0;
    double local_cmax = 0.0, local_csq = 0.0;  // COUPLING population: cells FSM moved water in
    long   local_cn = 0;
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (dmdapack.mask[j][i] != 0) {  // ALL land cells; predictor clamped to the feasible set (see below)
          const double h_n = dmdapack.starting_wtd[j][i] + topo_e[j][i];
          double       h_pred = (yg[j][i] - (1.0 - TR_G) * h_n) / TR_G;
          // Implicit exfiltration constraint: the head is constrained to wtd<=0 (h<=topo). A linear predictor that
          // overshoots ABOVE the surface would spike |x-h_pred| at the kink (a projection artifact that does
          // NOT shrink with dt -> the controller would reject to the floor). Clamp the predictor into the
          // feasible set so the estimate measures real truncation error: a cell RISING to the surface still
          // contributes its true rise (bounding dt), a cell already pinned contributes ~0. See EXFIL_EST note.
          if (g_direct_to_runoff) h_pred = std::min(h_pred, topo_e[j][i]);
          // WATER (volume) local error: the step tolerance lives in the same water units as the equilibrium
          // stop (|S·Δwtd|). |storedVolume(wtd_actual) - storedVolume(wtd_pred)| IS that water moved (secant
          // S·Δwtd), reusing the exact V(wtd) the storage residual + eq metric use; slope 1 above the surface
          // (ponded), porosity below. Symmetric with convergence: integrate to the accuracy we detect.
          const double poro = dmdapack.porosity_vec[j][i];
          // COUNT THE ERROR AGAINST THE ACTUAL INPUTS, NOT THE EXPECTED ONES (Andy, 2026-09-04).
          // h_pred extrapolates from HISTORY, so it cannot know about the FSM delta -- FillSpillMerge's
          // per-cell volume change, handed to THIS step as a recharge adjustment. The gap it opens is not
          // truncation error; it is an input the prediction omitted. Subtract it, and what remains is the
          // part of the move the integrator is actually responsible for. Both terms are volumes (the delta
          // is storedVolume(post) - storedVolume(pre)), so this is dimensionally the same quantity.
          // Without it, refining dt CANNOT reduce the estimate: the delta is sized for the step that
          // PRODUCED it, so a shorter next step makes it relatively larger, and the controller rejects to
          // its floor and dies with "step failed after max retries" (measured on tests/xrank_growth:
          // est fell only as ~dt^0.35 while dt fell four orders of magnitude).
          // Zero under fsm_coupling: impulse (the carrier is never written there), so that path is
          // bit-identical. SUBTRACTING the delta was tried first and is WRONG: the injected water
          // redistributes LATERALLY within the step, so V(x) - V(h_pred) != delta at the cell, and the
          // correction overshoots -- it made MORE arms abort, not fewer. Excluding the cell is the honest
          // form: where the forcing is discontinuous there is no truncation ORDER to control, so the cell
          // cannot inform a step-size decision either way. Measured coverage cost: 17-22% of land cells
          // excluded on tests/dt_invariance, so ~80% still bound the step.
          // CLASSIFY ON A MEANINGFUL DELTA, NOT A NONZERO ONE. `!= 0.0` was an exact float test on a
          // value FSM computes as a difference of stored volumes, so a cell FSM did not meaningfully
          // touch could still carry ULP-level noise and be excluded. That made the excluded set
          // DECOMPOSITION-DEPENDENT: measured n=1 vs n=2 on tests/xrank_adaptive, the count differed by
          // 4 and 11 cells of 196 at two steps. The separation is enormous and not a delicate choice --
          // noise is ~1e-14 m of water and the smallest genuine FSM delta measured is 6.7e-03, so any
          // bound in 1e-12..1e-6 behaves identically. 1e-9 m, a nanometre of water over a cell, sits
          // ~5 orders above the noise and ~6 below the signal.
          // THE SAME MEASUREMENT, PARTITIONED -- not two different quantities. `dev` is ONE expression,
          // |V(x - topo) - V(h_pred - topo)|, in metres of water, evaluated identically on every land
          // cell. Only WHICH cells it lands on differs, and so which ERROR SOURCE it reports there:
          //   est_integrator -- cells FSM did not touch. dev there IS the integrator's time-
          //                     discretisation error; measured order 2.00 over dt 0.25..3 yr.
          //   est_coupling   -- cells FSM moved water in. dev there is dominated by the GW<->FSM handoff,
          //                     which the linear predictor cannot know about; measured order 0.97 over a
          //                     16x range in dt.
          // Same measurement, same units, DISJOINT cell sets -- so they are directly comparable and the
          // max() below is meaningful: take whichever error source is currently the bigger issue. The
          // DIFFERING ORDERS are why both are needed. The order-1 coupling term decays more slowly, so it
          // governs at coarse steps -- which is where the controller does its damage -- while the order-2
          // term governs once the step is fine, leaving that regime bit-unchanged.
          //
          // max() rather than a sum, quadrature, or a single RMS over all cells: because the two
          // components converge at DIFFERENT ORDERS, folding them into one average yields an estimate
          // whose own order drifts with the cell mix. Quadrature of two RMSs over disjoint sets is not
          // the norm of anything. max() keeps a meaning -- the worse of two populations, measured the
          // same way -- and all three were measured indistinguishable on err/est anyway.
          //
          // WHY THIS REPLACED A BARE EXCLUSION. The coupling cells used to be DISCARDED, which left the
          // estimate blind whenever FSM had touched every land cell. That is not incidental: it is the
          // step right after FSM first routes a domain that starts ponded, and it happened once per run
          // at every step count tested. Measured there: a cell plunged 10.9 m in a single 3.85 yr step
          // while est reported exactly 0.0, and the discarded cells were carrying rms 4.34e-01 against a
          // tolerance of 0.1 -- 4.3x over, and invisible.
          const double dev  = std::abs(storedVolume(dmdapack.x[j][i] - topo_e[j][i], poro)
                                     - storedVolume(h_pred - topo_e[j][i], poro));
          if (std::fabs(dmdapack.fsm_delta_dist[j][i]) > 1e-9) {  // COUPLING population
            if (dev > local_cmax) local_cmax = dev;
            local_csq += dev * dev;
            local_cn++;
          } else {                                                // INTEGRATOR population
            if (dev > local_max) local_max = dev;
            local_sq += dev * dev;
            local_n++;
          }
        }
    DMDAVecRestoreArray(user_context.da, user_context.tr_ygamma, &yg);
    DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &topo_e);
    // AN ESTIMATE MUST EXIST BEFORE IT CAN STEER ANYTHING. Cells carrying an FSM delta are excluded
    // above; when FSM has touched EVERY land cell there is nothing left to measure and gn = 0. The old
    // code set est = 0.0 there, which the controller read as "zero error -- grow maximally". That is the
    // opposite of the truth: it is NO DATA. Measured on tests/golden transient_test (4 cycles x 8 yr,
    // controller free): this happened exactly ONCE PER RUN at every tolerance from 0.02 to 0.5, and at
    // that step the error the estimator could not see was the LARGEST in the run -- rms 0.27-0.50,
    // max 0.86-1.23 m -- while est reported 0.0 and dt was grown by the maximum factor. Carry validity
    // separately so the controller can HOLD dt instead of guessing. Task #58.
    long gn = 0, gcn = 0;
    MPI_Allreduce(&local_n,  &gn,  1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&local_cn, &gcn, 1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);
    est_n     = gn + gcn;   // EVERY land cell informs the estimate now; see the note in the loop
    est_cn    = gcn;
    est_valid = (est_n > 0);   // now false only on a domain with no land at all -- kept as a backstop
    if (user_context.dt_norm_rms) {
      double gsq = 0.0, gcsq = 0.0;
      MPI_Allreduce(&local_sq,  &gsq,  1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      MPI_Allreduce(&local_csq, &gcsq, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      est_int = (gn  > 0) ? std::sqrt(gsq  / (double)gn)  : 0.0;
      est_cpl = (gcn > 0) ? std::sqrt(gcsq / (double)gcn) : 0.0;
      est     = std::max(est_int, est_cpl);   // REPORTED total; the controller reads the parts, not this
    } else {
      MPI_Allreduce(&local_max,  &est_int, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
      MPI_Allreduce(&local_cmax, &est_cpl, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
      est = std::max(est_int, est_cpl);       // same partition under the MAX norm
    }
    have_est = true;
  } else if (user_context.use_dt_adaptive && user_context.bdf2_have_history) {
    // Generic linear-history predictor (ANY non-TR integrator -- cc backward-Euler / BDF2-on-V):
    //   h_pred = h^n + omega*(h^n - h^{n-1}), omega = dt_n/dt_{n-1}  -- dev ~ O(dt^2). Needs the last two
    // accepted states (history save below, tracked whenever adaptive). Same surface-inclusive norm and the
    // same controller as TR-BDF2 -- only the estimate differs. Runs from the 2nd step (once history exists).
    PetscScalar **swp, **topo_e;
    DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecGetArray(user_context.da, user_context.topo_vec, &topo_e);
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    double local_max = 0.0, local_sq = 0.0;
    long   local_n = 0;
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (dmdapack.mask[j][i] != 0) {  // ALL land cells; predictor clamped to the feasible set (see EXFIL_EST note)
          const double h_n = dmdapack.starting_wtd[j][i] + topo_e[j][i];
          double       h_pred = h_n + omega * (dmdapack.starting_wtd[j][i] - swp[j][i]);
          if (g_direct_to_runoff) h_pred = std::min(h_pred, topo_e[j][i]);  // wtd<=0 feasible set (kink-free error)
          // WATER (volume) local error -- same water units as the equilibrium stop (see the TR-BDF2 branch above).
          const double poro = dmdapack.porosity_vec[j][i];
          // Same exclusion as the TR-BDF2 branch above, for the same reason. See the note there.
          if (dmdapack.fsm_delta_dist[j][i] != 0.0) continue;  // see the note above
          const double dev  = std::abs(storedVolume(dmdapack.x[j][i] - topo_e[j][i], poro)
                                     - storedVolume(h_pred - topo_e[j][i], poro));
          if (dev > local_max) local_max = dev;
          local_sq += dev * dev;
          local_n++;
        }
    DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &topo_e);
    // AN ESTIMATE MUST EXIST BEFORE IT CAN STEER ANYTHING. Cells carrying an FSM delta are excluded
    // above; when FSM has touched EVERY land cell there is nothing left to measure and gn = 0. The old
    // code set est = 0.0 there, which the controller read as "zero error -- grow maximally". That is the
    // opposite of the truth: it is NO DATA. Measured on tests/golden transient_test (4 cycles x 8 yr,
    // controller free): this happened exactly ONCE PER RUN at every tolerance from 0.02 to 0.5, and at
    // that step the error the estimator could not see was the LARGEST in the run -- rms 0.27-0.50,
    // max 0.86-1.23 m -- while est reported 0.0 and dt was grown by the maximum factor. Carry validity
    // separately so the controller can HOLD dt instead of guessing. Task #58.
    long gn = 0;
    MPI_Allreduce(&local_n, &gn, 1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);
    est_n     = gn;
    est_valid = (gn > 0);
    if (user_context.dt_norm_rms) {
      double gsq = 0.0;
      MPI_Allreduce(&local_sq, &gsq, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      est = est_valid ? std::sqrt(gsq / (double)gn) : 0.0;
    } else {
      MPI_Allreduce(&local_max, &est, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    }
    // This predictor EXCLUDES the FSM-touched cells (see the loop), so everything it measures is
    // integrator error and there is no coupling part to separate. est_cpl stays 0.
    est_int  = est;
    have_est = true;
  }
  // CONTROLLER (method-agnostic): a PI step-size controller -- the standard cure for the dt "hunting" that
  // made the plain I-controller (hard reject on any overshoot, grow on any undershoot) lock into limit
  // cycles at certain dt_tol (measured dead-band ~0.2-0.25 m, flanked by fine values). The PI term uses the
  // PREVIOUS accepted error to DAMP the oscillation (a tuned thermostat vs bang-bang), and the reject shrink
  // is PI-damped rather than a hard slam -- that is what removes the dead-bands (verified: the 0.2-0.25 band
  // went from 84-99 cycles to 11-21, no catastrophe anywhere in the 0.1-0.5 operating range). reject_margin
  // keeps the ring firmly REJECTED (est > tol); loosening it to accept mild overshoots let big steps ring at
  // loose tol, so it stays 1.0. Robustness over speed: the nominal point is ~28% slower than the old
  // (hunting-prone) I-controller, but adaptive is now trustworthy on unknown terrain. See BDF2_ADAPTIVE_DESIGN.md.
  if (user_context.use_dt_adaptive && have_est) {
    const double safety        = 0.9;
    const double reject_margin = 1.0;   // reject an overshoot (est > tol); the PI-damped shrink below is
                                        // gentle, so this keeps the ring rejected without the bang-bang hunt
    const double kI = 0.3, kP = 0.2;    // PI gains (elementary I-exponent ~0.5, split I+P for damping)
    const double dt_now = user_context.deltat;
    const double prev   = (user_context.dt_prev_est > 0.0) ? user_context.dt_prev_est : est;  // I-only on step 1
    // NO DATA -> DO NOT ACT. factor 1.0 holds dt exactly where it is: we cannot estimate the error
    // this step, so we neither grow into the unknown nor shrink without cause. Distinct from est == 0.0
    // WITH data, which is a genuine measurement of zero error and still earns the growth factor.
    double factor = !est_valid
                      ? 1.0
                      : (est_int > 0.0)
                          ? safety * std::pow(user_context.dt_tol / est_int, kI) * std::pow(prev / est_int, kP)
                          : user_context.dtc_grow;
    factor = std::min(user_context.dtc_grow, std::max(user_context.dtc_shrink, factor));
    // THE COUPLING PART MAY WITHHOLD GROWTH, NEVER FORCE A SHRINK (#63). est_cpl does not answer to dt --
    // MEASURED on tests/golden fsm_runoff_hi, where it sat at 0.6043593790 while dt was driven from
    // 4.4e+07 s down to 4.1e-02 s, NINE ORDERS OF MAGNITUDE, changing in the 9th significant figure. Feeding
    // it to the reject test (as `est = max(int, cpl)` did) therefore asks the controller to fix by shrinking
    // something shrinking cannot fix: dt collapses until dtc_max_retries and the run ABORTS. That is not a
    // conservative choice, it is an unsatisfiable one.
    // Withholding growth IS legitimate and is the whole of what the coupling signal can honestly say:
    // "FSM is moving a lot of water here, do not get greedy." The threshold is the user's own accuracy
    // target, so no new tunable is introduced.
    if (est_valid && est_cpl > user_context.dt_tol) factor = std::min(factor, 1.0);
    // -wtm_dt_trace: report the quantity that STEERS the integration. `est` was computed on every
    // adaptive step and reported nowhere, so nothing could tell whether it responded to dt at all --
    // which is how a generic-branch estimator of observed order p = 0.00 survived. One machine-readable
    // line per step. REJECTED steps are included deliberately: they are where a mis-scaled estimate does
    // its damage (grinding dt down against an error that will not shrink), and omitting them would hide
    // exactly the failure this exists to expose. Consumed by tests/estimator_order.
    const bool dt_accept = !(est_int > reject_margin * user_context.dt_tol);  // est_int, not est: see #63 above
    if (user_context.dt_trace)
      PetscPrintf(PETSC_COMM_WORLD,
                  "DTTRACE dt=%.9e est=%.9e eint=%.9e ecpl=%.9e tol=%.9e factor=%.6f iters=%d accepted=%d "
                  "nest=%ld ncpl=%ld\n",
                  dt_now, est, est_int, est_cpl, user_context.dt_tol, factor, its, dt_accept ? 1 : 0,
                  est_n, est_cn);
    if (!dt_accept) {  // LARGE overshoot: reject + retry (state NOT committed)
      user_context.deltat = dt_now * std::min(factor, 1.0);
      return -1;
    }
    // ACCEPT (a mild overshoot is tolerated): commit, remember this error for the PI term, size the next step.
    //
    // DEFERRED, and this ordering is load-bearing. The sized dt belongs to the NEXT step, but everything
    // below this point is still accounting the step just TAKEN and must see THAT step's dt: the BDF2
    // history ratio (bdf2_prev_dt), the taper-1 sink and taper-2/3 evaporation removal depths, the
    // land->ocean flux accumulation, and TR-BDF2's step quadrature. Writing it into user_context.deltat
    // here made all five read the next step's dt. Measured on tests/fsm_consistency, exact budget
    // residual as a fraction of recharge: TR-BDF2 + adaptive -1.603 and BDF2-on-V + adaptive -0.417,
    // against ~2e-07 for the same schemes at fixed dt. The REJECT branch above is unaffected -- it
    // returns before any of that accounting runs, so it still writes deltat directly.
    // The PI history must hold the SAME quantity the P term divides by, which is now est_int (#63);
    // storing the combined est would compare an integrator error against a coupling-inflated one.
    if (est_valid) user_context.dt_prev_est = est_int;  // a non-estimate must not enter the PI history
    if (its > user_context.dtc_easy_iters) factor = std::min(factor, 1.0);  // hard solve: hold, don't grow
    dt_next = dt_now * factor;
    if (user_context.dtc_dt_max > 0.0 && dt_next > user_context.dtc_dt_max) dt_next = user_context.dtc_dt_max;
    have_dt_next = true;
  }

  // Exact budget-closing accounting (Picard path): the solver's discrete storage + recharge terms,
  // read while starting_wtd still holds w^n and starting_wtd_prev still holds w^{n-1} (both below
  // overwrite these). Together with total_ocean_outflow and total_surface_removed the budget then
  // closes to the SNES tolerance. See benchmark/WATER_BUDGET.md.
  accumulate_budget_terms(user_context, arp, dmdapack, evap_active_this_step, sink_active_this_step);

  // BDF2 / predictor: before starting_wtd is overwritten with h^{n+1} below, save the current h^n
  // wtd as the next step's h^{n-1}. The first step captures h^0 and sets the history flag, so BDF2 /
  // the predictor engage from the second step on (the first bootstraps with backward Euler / w^n guess).
  // (fsm_off / Phase A: history is continuous; Phase B will reset the flag after FSM.)
  if (user_context.use_bdf2 || user_context.use_dt_adaptive) {
    PetscScalar** my_starting_wtd_prev;
    DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &my_starting_wtd_prev);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        my_starting_wtd_prev[j][i] = dmdapack.starting_wtd[j][i];
    DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &my_starting_wtd_prev);
    user_context.bdf2_have_history = true;
    user_context.bdf2_prev_dt      = user_context.deltat;  // this step's Δt becomes Δt_{n-1} next
  }

  // copy the result back into the distributed wtd carrier (starting_wtd), which
  // feeds the next solve in the per-report step loop and is assembled to arp.wtd once
  // per cycle by gather_wtd_to_all. Read topo/mask/porosity from DMDA arrays
  // (topo_vec is re-scattered each cycle in transient) so arp is not needed here.
  PetscScalar** my_topo;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  PetscScalar** my_exfiltration_post = nullptr;  // -wtm_active_set: captured exfiltration depth from the converged residual eval
  if (g_active_set) DMDAVecGetArray(user_context.da, user_context.exfiltration_vec, &my_exfiltration_post);
  if (evap_active_this_step) {
    DMDAVecGetArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecGetArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecGetArray(user_context.da, user_context.precip_vec, &my_precip);  // taper 3 deficit (E_eff - P)
  }
  double dh_max_local = 0.0;  // max |w^{n+1} - w^n| over owned land cells (for the PTC/SER dt controller)
  int    dh_i_local = -1, dh_j_local = -1;  // argmax cell (diagnostic: which land cell moves most this step)
  int    nflick_local = 0;                  // # owned land cells with |Δw| > 1mm (within-cycle flicker diagnostic)
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      // The SNES variable IS the head: wtd = x - topo.
      const double solved_wtd = dmdapack.x[j][i] - my_topo[j][i];
      // ACTIVE SET: PROJECT ONTO THE FEASIBLE SET. The semismooth constraint is w <= lake_stage, with
      // EQUALITY on the active set -- so on the active set the value is determined by the CONSTRAINT, not
      // by the iterate. The solve cannot deliver that equality: at a pinned cell the residual IS the water
      // table (f = pin = x - topo), while the SNES variable is the HEAD, of order topo, and snes_stol is a
      // RELATIVE STEP tolerance. At 1e-12 with topo = 100 m it permits a step of 1e-10, ~1760x larger than
      // the residual actually left, so the solver correctly stops with the constraint satisfied only to
      // within ULP(topo) -- and WHICH ulp depends on the arithmetic path, hence on the MPI decomposition.
      // Measured: the leftover wtd values are exactly 1, 3, 4 and 5 ULPs of 100 m (1.421e-14 .. 7.105e-14)
      // and are IDENTICAL at snes_stol 1e-8, 1e-10, 1e-12 and 1e-14 -- six orders of tightening move them
      // not at all, which is what proves this is not a convergence remainder. n=1 left 387 such cells,
      // n=2 left none. Downstream that flipped FSM's "did this cell change?" answer for 65 of 196 land
      // cells, which moved the adaptive estimate's RMS divisor and hence the time steps taken (task #56).
      // min() is the projection, not a tolerance: a free cell already satisfies w < stage and is untouched.
      const double new_wtd = (g_active_set && dmdapack.mask[j][i] != 0)
                                 ? std::min(solved_wtd, static_cast<double>(dmdapack.lake_stage[j][i]))
                                 : solved_wtd;
      // Under-relaxation (-wtm_relax a<1): damp the step to w <- a*w_solve + (1-a)*w_prev. a=1 -> byte-
      // identical. The metric measures the RELAXED change (the true state move), so it stays honest.
      const double relaxed = (g_relax >= 1.0) ? new_wtd
                                              : g_relax * new_wtd + (1.0 - g_relax) * dmdapack.starting_wtd[j][i];
      if (dmdapack.mask[j][i] != 0) {
        const double dh = std::abs(relaxed - dmdapack.starting_wtd[j][i]);
        if (dh > dh_max_local) { dh_max_local = dh; dh_i_local = i; dh_j_local = j; }
        if (dh > 1e-3) nflick_local++;
      }
      dmdapack.starting_wtd[j][i] = relaxed;
      // LAKE STAGE, write site A of two (site B is in couple_surface_and_recharge, WTM.cpp). Which site
      // is AUTHORITATIVE depends on the configuration, and all three cases are live -- do not assume this
      // write is redundant:
      //   FSM off                    -> B never runs (it is inside `if (fsm_on)`). A is the only per-step
      //                                 writer, and max(0, post-solve wtd) is the correct stage.
      //   FSM on, distributed        -> B overwrites this from the POST-FSM table in the same step, before
      //     (infiltration off)          anything reads it, so A's value is never seen. Under `impulse` the
      //                                 two agree exactly (starting_wtd is itself post-FSM there), which is
      //                                 why carrying the stage explicitly was bit-identical when introduced.
      //   FSM on, serial             -> distribute_recharge is false, so B is skipped. The cycle-top block
      //     (infiltration on)           in WTM.cpp reseeds the stage once per CYCLE; A is the writer that
      //                                 keeps it current between steps WITHIN a cycle.
      // Under fsm_coupling: continuous B deliberately writes something DIFFERENT from this -- max(post-FSM
      // stage, pre-FSM ponding) -- so that the obstacle yields to water the FSM delta is still moving.
      dmdapack.lake_stage[j][i] = std::max(0.0, relaxed);
      if (dmdapack.mask[j][i] == 0) {
        // Ocean cell: Dirichlet head h = 0 by definition. The matrix-free Anderson solve enforces this
        // exactly (post-solve wtd = 0), but the Picard CG/GAMG solve leaves a tiny, MPI-decomposition-
        // dependent residual head here. That residual is solver noise, not ocean loss -- accumulating
        // it into total_loss_to_ocean_gw made the diagnostic scale with the rank count (n1 0.047, n4
        // 0.141) and broke MPI consistency. Project to exact 0 and do NOT accumulate: the real
        // land->ocean groundwater loss is the Darcy interface flux (total_ocean_outflow_gw), counted in
        // accumulate_ocean_outflow. Initial ocean-cell water (from the input starting_wt) is still
        // captured once, at setup, by set_starting_values. Anderson is unaffected (it added 0 here).
        dmdapack.starting_wtd[j][i] = 0.;
        continue;
      }
      // -wtm_active_set: the semismooth constraint removed the exfiltration INSIDE the solve, so it left no above-
      // surface water for the collectors below to see. Transfer the captured per-cell exfiltration depth (from the
      // converged residual eval) to FSM (sink_removed_dist) and the budget (total_surface_removed). Serial loop.
      if (g_active_set && my_exfiltration_post) {
        const double exfil_depth = my_exfiltration_post[j][i];
        if (exfil_depth > 0.0) {
          arp.total_surface_removed        += exfil_depth * arp.cell_area[j];
          dmdapack.sink_removed_dist[j][i] += exfil_depth;
        }
      }
      // Land cells: the sink and the evaporation taper can both be active; account each in the same
      // sub-step it was removed, evaluated at the just-computed new head. Serial loop -> += race-free.
      // TR-BDF2 took this above, with the step quadrature, and handed the SAME depth to FSM. Repeating
      // it here at w^{n+1} alone would both double-count and use the wrong weight.
      if (sink_active_this_step && !user_context.use_tr_bdf2) {
        // Sink removed dt*Q(w^{n+1}) (Q is m/s -> dt*Q is a depth). To FSM (stays in domain).
        const double removed_depth =
            g_direct_to_runoff ? std::max(0.0, static_cast<double>(dmdapack.starting_wtd[j][i]))  // = dt*rate = the excess depth
                               : 0.0;
        arp.total_surface_removed += removed_depth * arp.cell_area[j];  // budget-closing (WATER_BUDGET.md)
        dmdapack.sink_removed_dist[j][i] += removed_depth;              // per-cycle FSM input (taper 1)
      }
      // TR-BDF2 accumulated its evaporation above, with the step quadrature over (w^n, Y_gamma,
      // w^{n+1}); doing it again here at w^{n+1} alone would both double-count and use the wrong weight.
      if (evap_active_this_step && !user_context.use_tr_bdf2) {
        // Taper 2 (+ taper 3) removed dt*R(w^{n+1}) to the ATMOSPHERE (leaves the domain) -> its own
        // budget channel, kept separate from the sink's exfiltration-to-FSM (different destination).
        // R = min(E_eff,P) + (E_eff-P)_+ * A(wtd): the accessible evaporative loss (== E_eff when taper 3
        // is off). The inaccessible deep deficit is correctly NOT counted.
        const double evap_depth =
            user_context.deltat * evapRemoval(dmdapack.starting_wtd[j][i], my_evap[j][i], my_owe[j][i],
                                              my_precip[j][i] / SECONDS_IN_A_YEAR);
        arp.total_evap_removed += evap_depth * arp.cell_area[j];
      }
      // Post-solve surface exfiltration-to-runoff collection / truncation. Two opt-in modes route above-surface water to FSM
      // BETWEEN steps -- an explicit "collect" that is cheaper and more robust than the implicit sink
      // (no qmax/width/ramp, mass-exact, cannot diverge):
      //   -wtm_surface_exfiltration_to_runoff : STANDARD physics (T stays CLAMPED above the surface, so parked water
      //       adds no extra transmissivity -> no lateral spreading). Water is allowed to mound during
      //       the solve, then clamped to the surface here = A_legacy but with a PER-STEP collect instead
      //       of waiting for FSM's cadence. The robust variant. (Turn the sink off to use it: else the
      //       sink holds wtd<=0 and this never fires.)
      // -wtm_extended_soil USED TO SELECT THIS TOO, AND MUST NOT. That pairing was added 2026-08-10 as a
      // second post-solve mode (EXTENDED-soil physics: no T-clamp, so T GROWS above the surface) and was
      // recorded in the same breath as a failure -- "Tested NEGATIVE: the un-clamped T conducts laterally
      // = a new stiffness (slower, diverges at large dt). Kept as a documented dead-end; the clamp variant
      // is the one to use." The dead end was kept, but it was left wired to -wtm_extended_soil, which
      // already meant something else and something CONTRADICTORY: remove the wtd=0 free boundary so the GW
      // step is smooth and BDF2 recovers 2nd order (a6a33f9, 2026-07-27; benchmark/BDF2_RECHARGE_ORDER.md
      // section 15, which states the truncation belongs at the FSM/cycle handoff, NOT per GW step).
      // Truncating the mound here every step reinstates exactly the temporal kink the flag exists to
      // remove, so the dead-end experiment silently defeated the validated fix. Measured on the design
      // note's own harness (benchmark/picard/recharge_free_boundary.py, arm E), order in dt:
      //     with extended_soil selecting this block : 1.22  1.08  1.02  0.92   err 1.647 mm at dt=1 yr
      //     without (this line)                     : 2.07  2.01  2.00  2.01   err 0.002031 mm
      // against the section-15 record of 2.07/2.07/2.00 and 0.0019 mm -- i.e. restored. The mound returns
      // with it: max wtd +23.392 m over 4294 cells, the "+23 m mound" section 15 describes.
      // NOTE the second, independent clamp: a runoff COLLECTOR also pins wtd<=0 and defeats extended soil
      // on its own, so both must be off to see the effect (that is why single-variable elimination reads
      // as "no change" here -- each masks the other). extended_soil + a collector is a contradictory
      // request the model does not currently refuse.
      // Excess = storedVolume(wtd) - storedVolume(0): the real above-surface storage under the ACTIVE
      // storativity (~wtd surface-water depth in standard physics; porosity*wtd in extended soil).
      if (g_surface_exfiltration_to_runoff_array && dmdapack.starting_wtd[j][i] > 0.0) {
        const double poro         = dmdapack.porosity_vec[j][i];
        const double excess_depth = storedVolume(dmdapack.starting_wtd[j][i], poro) - storedVolume(0.0, poro);
        arp.total_surface_removed += excess_depth * arp.cell_area[j];  // budget-closing (WATER_BUDGET.md)
        dmdapack.sink_removed_dist[j][i] += excess_depth;              // collect -> gather -> arp.runoff -> FSM
        // ...and correct the STORAGE term to the state we are about to commit. This removal is
        // POST-SOLVE: it is not a term in the residual, so the solve satisfied
        //     dV_pre = solver_recharge - ocean - evap
        // and accumulate_budget_terms (which ran earlier, before this loop) measured dV from
        // dmdapack.x -- the PRE-clamp w^{n+1}. Subtracting `excess_depth` from the budget while leaving
        // the storage term at its pre-clamp value describes a state the model does NOT carry forward,
        // and leaves residual = -total_surface_removed exactly. With FSM on, the same water is returned
        // and re-skimmed every step, so that gross flux accumulated to ~9x recharge:
        //     collector    Anderson      Picard
        //     explicit     -9.165e+00    -9.331e+00
        //     legacy       -7.998e+00    -8.865e+00
        // against ~1e-7..1e-10 for every collector whose removal IS in the residual. No water was ever
        // lost -- excess_depth is handed to FSM on the line above -- so this was a defect in the
        // conservation CHECK, not in conservation. But `explicit` is what the Picard solver resolves to
        // when the collection method is unset, so the budget was unusable in Picard's default
        // configuration. Substituting dV_post = dV_pre - excess_depth closes the identity exactly.
        arp.total_storage_change -= excess_depth * arp.cell_area[j];
        dmdapack.starting_wtd[j][i] = 0.0;                             // truncate to the real land surface
      }
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);
  if (g_active_set) DMDAVecRestoreArray(user_context.da, user_context.exfiltration_vec, &my_exfiltration_post);
  if (evap_active_this_step) {
    DMDAVecRestoreArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecRestoreArray(user_context.da, user_context.precip_vec, &my_precip);
  }

  // Global max |Δw| this step: the pseudo-transient/SER dt controller grows Δt as this shrinks toward
  // equilibrium (the discrete steady residual ~ S·Δw/Δt), so the ramp accelerates to Newton near steady
  // state. Reduced here (cheap) so WTM.cpp's continuation loop can read user_context.last_dh_max.
  MPI_Allreduce(&dh_max_local, &user_context.last_dh_max, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  MPI_Allreduce(&nflick_local, &user_context.last_dh_nflicker, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  user_context.last_dh_i = dh_i_local;  // argmax cell (exact at n=1; rank-local under MPI -- diagnostic only)
  user_context.last_dh_j = dh_j_local;

  // Account the water that left through land->ocean faces this solve (Darcy interface flux at the
  // converged head), the term that closes the water budget against the Dirichlet ocean boundary.
  // TR-BDF2 already accumulated its three-point step quadrature inside accumulate_budget_terms, back
  // when w^n and Y_gamma were both still live (the copy-back below has since overwritten w^n).
  if (!user_context.use_tr_bdf2) accumulate_ocean_outflow(user_context, arp, user_context.x, 1.0);

  // COMMIT THE INTERVAL. One place, written once, before `deltat` is advanced below to the NEXT step's
  // size. Everything that accounts for this step reads user_context.step rather than `deltat`, so the
  // controller cannot mutate the interval out from under its own consumers. Every reject path returns
  // ABOVE this point, so a retried step is committed once, at the dt it finally succeeded with.
  user_context.step.dt           = user_context.deltat;
  user_context.step.elapsed_from = params.elapsed_time_s;
  params.elapsed_time_s         += user_context.deltat;   // TRUE elapsed time: summed, never derived
  user_context.step.elapsed_to   = params.elapsed_time_s;

  // -wtm_dt_adaptive: NOW size the next step. Every accumulator above has finished accounting the step
  // just taken, so user_context.deltat is free to become the next step's. See the DEFERRED note in the
  // controller block for what went wrong when this happened up there.
  if (have_dt_next) user_context.deltat = dt_next;

  // The full wtd field is assembled once per cycle, after the per-report step loop, by
  // gather_wtd_to_all -- not here per solve (see benchmark/DISTRIBUTED_ARP_DESIGN.md).
  // Return the Newton iteration count (>=0) so the dt-continuation controller can grow dt after an
  // easy step; a non-converged continuation step returned -1 above.
  return static_cast<int>(its);
}

// Assemble the full wtd field on every rank from each rank's owned cells of the
// distributed carrier (starting_wtd).
void gather_wtd_to_all(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.starting_wtd[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  // Gather to rank 0 only: wtd is consumed by the serial sections (FSM, recharge,
  // diagnostics, output), which all run on rank 0, and re-scattered to the solve
  // next cycle. Non-root ranks do not need the full field, so arp.wtd can be
  // rank-0-only.
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++)
        arp.wtd(i, j) = full[j * params.ncells_x + i];
}

// Gather the distributed per-cycle runoff (runoff_dist = runoff_ratio*rech) and ADD it to the rank-0
// arp.runoff carrier (approach B: the carrier is zeroed once per step -- after FSM consumes -- and every
// contributor then accumulates into it), so the next FillSpillMerge routes the recharge's runoff. Called
// only when runoff_ratio_on; otherwise this contribution is simply absent. Reuses the un-held wtd_global as
// the gather scratch (after gather_wtd_to_all has finished with it -- the two run sequentially). See
// DISTRIBUTED_ARP_DESIGN.md (2c).
// `dt_scale` = (elapsed since the last handoff) / params.deltat. runoff_dist holds a NOMINAL step's depth
// (runoff_ratio * rate * params.deltat), baked when the recharge for the next step was prepared -- at
// which point the next step's dt is not yet final under adaptive stepping. So it is scaled HERE, at
// consumption, where the accepted dt is known: the same LAZY treatment the direct channel already gets
// via rech_dt_scale at the rech_vec assembly. Baking the scale in early instead would be wrong twice
// over -- the adaptive loop can still clamp dt to the cycle remainder afterwards, and a rejected step
// re-runs at a smaller dt after the water was already sized. Exactly 1.0 on every fixed-dt path.
void gather_runoff_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack,
                           double dt_scale, bool deliver) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.runoff_dist[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++) {
        const double routed = full[j * params.ncells_x + i] * dt_scale;
        if (deliver) {
          arp.runoff(i, j) += routed;         // FillSpillMerge will route it
        } else {
          // No surface-water model: the runoff LEAVES the domain. `fsm_on` is the route-vs-discard
          // switch (see the selector note above), and the discard must be BOOKED or the water simply
          // vanishes: measured, 30% of P-ET disappeared with nothing in the output saying so, because
          // col 20 read 0 -- indistinguishable from runoff_ratio being off. Booked to the surface
          // water-to-sea counter, which is what "runs off" means and is a term the physical budget
          // already reads, so cols 15/16 keep closing. total_loss_to_ocean is a REPLICATED global
          // (never MPI-reduced), and this runs on rank 0 alone, so adding here is consistent.
          arp.total_loss_to_ocean += routed * arp.cell_area[j];
        }
        // Book the routed INPUT CHANNEL (col 20) HERE, where the water is actually handed over and at
        // the same scale -- not at preparation, where the nominal amount is baked. Booking it there
        // made the diagnostic disagree with the physics the moment the delivery was scaled. rank 0
        // holds the whole grid, every other rank contributes 0, so the Allreduce in PrintValues still
        // yields the correct global total.
        arp.total_runoff_to_surface += routed * arp.cell_area[j];
      }
}

// Whether the implicit sub-surface sink is configured this run (taper 1). Lets the cycle loop
// decide whether to gather the sink accumulator into arp.runoff for FSM without reaching into the
// file-static flag. Set in update() from -wtm_surface_sink, so valid by the post-solve gather.
double ksat_surface_smoothing_width() { return g_ksat_surface_smoothing_width; }
double ksat_soilbottom_smoothing_width() { return g_ksat_soilbottom_smoothing_width; }

bool direct_to_runoff_on() { return g_direct_to_runoff; }
bool fsm_continuous_on() { return g_fsm_continuous; }
// Whether the lake-aware active-set skim is on. It captures the skimmed above-free-surface water into the
// same sink accumulator, so the post-solve gather must hand it to arp.runoff for FSM -- otherwise the
// skimmed water is removed from the aquifer and counted as surface_removed but never delivered to the lake,
// so lakes cannot fill (bug found via tests/fsm_fullness: skim drained a basin plain filled to its sill).
bool active_set_on() { return g_active_set; }

// Whether extended-soil surface truncation routes above-surface water to FSM (via the same sink
// accumulator). Lets the cycle loop gather the accumulator for FSM when extended soil is on, just as
// for the sink. Set in update() from collection.method: extended_soil.
bool extended_soil_on() { return g_extended_soil; }

// Whether post-solve surface exfiltration-to-runoff collection routes above-surface water to FSM (via the sink accumulator).
// Lets the cycle loop gather the accumulator for FSM, as for the sink. Set from -wtm_surface_exfiltration_to_runoff.
bool surface_exfiltration_to_runoff_on() { return g_surface_exfiltration_to_runoff_array; }

// Whether the demand-identity evaporation taper is on (taper 2). Lets the explicit-recharge sites
// (irf.cpp, WTM.cpp) drop their hard ET<->owe switch and feed just precip, because the smooth
// implicit E_eff now carries that ET->open-water transition. Set by read_evap_taper_options().
bool evap_taper_on() { return g_evap_taper; }

// Taper 3 (accessibility / extinction-depth) is on. Gates taper 2's sub-surface deficit; inert on its
// own. Set by read_evap_taper_options().
bool extinction_on() { return g_extinction; }

// Read the taper-2 options (-wtm_evap_taper, wtd_c, s) into the file-static flags. (It once also
// "enforced the evap_mode-1 requirement"; that stopped being true before evap_mode was removed, and
// the claim outlived the code by some margin.) Called BOTH early in WTM.cpp::initialise() (so irf.cpp's initial recharge
// sees the flag) AND in update() (so a standalone solve still parses it). Idempotent -- it just
// re-reads the same PETSc options -- so the double call is harmless.
void read_evap_taper_options(const Parameters& params) {
  g_evap_taper = params.taper_surface_transition;   // evaporation.tapers.surface_transition
  // Config-owned (evaporation.et_sigmoid). The -wtm_evap_taper_wtdc / -wtm_evap_taper_s flags are GONE:
  // they had no callers anywhere in the repo and existed only as transport for these two YAML keys, which
  // the bridge pushed into PETSc's options DB for this line to read back. Reading Parameters directly
  // means the value is stored, schema-checked, and appears in the resolved-config log -- none of which
  // was true while it lived only in the options database.
  g_evap_taper_wtdc = params.evap_taper_wtdc;
  g_evap_taper_s    = params.evap_taper_s;

  // Taper 3: accessibility / extinction-depth clamp (awickert/WTM#4). Own on/off toggle plus the depth.
  g_extinction = params.taper_depth_extinction;     // evaporation.tapers.depth_extinction
  g_extinction_depth = params.extinction_depth;  // config-owned; -wtm_extinction_depth retired

  // open_water_evap is supplied either way, so E_eff has the owe it needs, and the recharge paths
  // check the taper FIRST -- the smooth removal auto-zeroes standing water in place of the hard wtd=0
  // used when the taper is off. Configuration mismatches are surfaced as warnings, not errors -- see
  // the warn_taper_configuration() checks.
  (void)params;
}

// Emit configuration warnings for the surface-water evaporation model. The intended (blessed)
// configuration is the smooth transition with BOTH taper 2 (-wtm_evap_taper) and taper 3
// (-wtm_extinction) on; every other combination is arid-unsafe, inert, or the legacy hard-switch
// model, and is flagged here. Caller guards rank 0 so this prints once. See SURFACE_SINK_DESIGN.md 14.
void warn_taper_configuration(const Parameters& params) {
  (void)params;
  // evap_mode is GONE (2026-09-10). It had been frozen at 0 and unsettable, so the whole
  // `if (params.evap_mode)` half of this function -- three warnings about the mode-1 configurations --
  // was UNREACHABLE, and the mode-0 half fired on every run. One of those was pure noise in the
  // DEFAULT configuration: "evap_mode 0 with the taper on ... evap_mode 0 and 1 coincide" told a user
  // about a setting they could not set, on every single run.
  //
  // What survives is what was always the real content: the taper is what governs evaporation, and two
  // configurations of it are genuinely worth warning about.
  if (g_evap_taper) {
    if (!g_extinction)
      std::cerr << "WARNING: evaporation.tapers.surface_transition without depth_extinction: in arid "
                   "cells (ET > precip) the evaporation taper draws the water table down WITHOUT BOUND "
                   "(no equilibrium). Turn depth_extinction on -- the accessibility / extinction-depth "
                   "clamp -- unless you specifically want the surface taper alone for testing."
                << std::endl;
  } else {
    if (g_extinction)
      std::cerr << "WARNING: evaporation.tapers.depth_extinction without surface_transition has NO "
                   "EFFECT: the extinction-depth clamp gates the surface taper's evaporative deficit, "
                   "which is not active."
                << std::endl;
    std::cerr << "WARNING: with evaporation.tapers.surface_transition off, ALL surface water is removed "
                 "every step (GW-alone testing; Fan Reinfelder et al. 2013), and the hard wtd=0 "
                 "ET<->open-water switch makes FillSpillMerge lake formation rank-dependent "
                 "(NON-DETERMINISTIC across MPI rank counts) and applies no phreatic ET. The smooth "
                 "tapers are recommended."
              << std::endl;
  }
}

// Gather this cycle's distributed sink removal (sink_removed_dist, depth m) and ADD it to rank-0 arp.runoff,
// so this cycle's FillSpillMerge routes the exfiltrated water the implicit sink pulled out of the solve
// (taper 1). Because the sink holds wtd<=0, FSM's own wtd>0->runoff handoff never fires -- this is its
// smooth, order-preserving replacement. One of several ADDITIVE contributors to the arp.runoff carrier
// (approach B: the carrier is zeroed once per step after FSM consumes, then runoff-ratio + this sink + FSM's
// ponded `+= wtd` all accumulate into it before the one FSM). Reuses wtd_global as scratch, after
// gather_wtd_to_all has finished with it (the two run sequentially). See SURFACE_SINK_DESIGN.md sec 14.
void gather_sink_removed_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
  const auto [xs, ys, xm, ym] = get_corners(user_context.da);
  PetscScalar** wg;
  DMDAVecGetArray(user_context.da, user_context.wtd_global, &wg);
  for (int j = ys; j < ys + ym; j++)
    for (int i = xs; i < xs + xm; i++)
      wg[j][i] = dmdapack.sink_removed_dist[j][i];
  DMDAVecRestoreArray(user_context.da, user_context.wtd_global, &wg);

  std::vector<double> full;
  user_context.full_grid_gather->gatherToZero(user_context.wtd_global, full);

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0)
    for (int j = 0; j < params.ncells_y; j++)
      for (int i = 0; i < params.ncells_x; i++)
        arp.runoff(i, j) += full[j * params.ncells_x + i];
}

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
static PetscErrorCode FormInitialGuess(AppCtx* user_context, DM da, Vec X) {
  PetscScalar **x, **my_starting_wtd, **my_topo, **my_fdepth, **my_ksat;

  DMDAVecGetArray(da, X, &x);
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_vec, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_vec, &my_ksat));

  const auto [xs, ys, xm, ym] = get_corners(da);

  // Seed the SNES variable, which is the head: x = starting_wtd + topo. Ocean cells (starting_wtd =
  // topo = 0) seed 0.
#pragma omp parallel for default(none) \
    shared(my_starting_wtd, my_topo, ys, ym, xs, xm, x) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      x[j][i] = my_starting_wtd[j][i] + my_topo[j][i];
    }
  }

  DMDAVecRestoreArray(da, X, &x);
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_vec, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_vec, &my_ksat));
  return 0;
}

/*
   FormRHS - Forms constant RHS for the problem.

   Input Parameters:
   user - user-defined application context
   B - RHS vector

   Output Parameter:
   B - vector
 */
static PetscErrorCode FormRHS(AppCtx* user_context, DM da, Vec B) {
  PetscScalar **b, **my_starting_wtd, **my_topo;

  DMDAVecGetArray(da, B, &b);
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));

  const auto [xs, ys, xm, ym] = get_corners(da);

  // Anderson BE: the SNES RHS b = h^n carries the previous-step storage (residual = F(x) − b). The
  // matrix-free BDF2-on-V path instead folds the FULL 3-level storage (V^{n+1},V^n,V^{n-1}) into the
  // residual itself, so its RHS is zero. The bootstrap step (no history yet) still uses the BE RHS.
  const bool bdf2v = user_context->use_bdf2 && user_context->bdf2_have_history && !user_context->use_picard;
  // dev.storage_form: volume folds the FULL storage ΔV into the residual (like bdf2v), so its RHS is 0 too.
  const bool zero_rhs = bdf2v || user_context->use_tr_bdf2 || g_volume_storage;
#pragma omp parallel for default(none) shared(ys, ym, xs, xm, b, my_starting_wtd, my_topo, zero_rhs) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      b[j][i] = zero_rhs ? 0.0
                         : my_starting_wtd[j][i] + my_topo[j][i];  // land mask==0: topo and wtd already 0 elsewhere
    }
  }
  DMDAVecRestoreArray(da, B, &b);
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));

  return 0;
}

/* ------------------------------------------------------------------- */
// GRID INDEX CONVENTION (see benchmark/GRID_CONVENTION.md) -- fixed once for the whole file:
//   i = column = EAST-WEST (longitude); spacing cellsize_e_w_metres[j], SHRINKS poleward.
//   j = row    = NORTH-SOUTH (latitude); spacing cellsize_n_s_metres, CONSTANT.
// Arrays are field[j][i]. NOTE the WTM paper (Callaghan et al. 2025, App. B) uses the OPPOSITE
// letters (paper x = S-N, y = W-E), so paper-Delta_x = cellsize_n_s and paper-Delta_y =
// cellsize_e_w. Divide the E-W (i +/- 1) flux by cellsize_e_w^2 and the N-S (j +/- 1) flux by
// cellsize_n_s^2 (with the face-centred E-W wall length) -- NOT the reverse.
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo* info, PetscScalar** x, PetscScalar** f, AppCtx* user_context) {
  DM da = user_context->da;
  PetscScalar **my_mask, **my_fdepth, **my_ksat, **my_topo, **my_rech, **my_T, **my_starting_wtd, **my_porosity, **gew,
      **gn, **gs;

  /*
    Compute function over the locally owned part of the grid.
    topo/fdepth/ksat/T use local ghost vectors so neighbor accesses [j][i±1] are valid under MPI.
  */
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // conservative-FV flux geometry
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscScalar** my_exfiltration = nullptr;  // -wtm_active_set: per-cell captured exfiltration depth (m) -> FSM post-solve
  if (g_active_set) PetscCall(DMDAVecGetArray(da, user_context->exfiltration_vec, &my_exfiltration));
  PetscScalar** my_lake_stage = nullptr;  // -wtm_active_set: the obstacle, carried rather than inferred
  if (g_active_set) PetscCall(DMDAVecGetArray(da, user_context->lake_stage, &my_lake_stage));
  // Matrix-free 2nd-order-in-time (solver.method: anderson solver.time_integration: bdf2): once a history exists, the storage
  // term is the 3-level BDF2 difference of the stored VOLUME (genuine 2nd order), head-scaled by the
  // specific yield so the residual stays O(metres) for Anderson. Same fixed point as the Picard
  // BDF2-on-V operator (verified). The bootstrap step (no history) uses backward Euler. See
  // benchmark/TBAR_TIME_AVERAGING.md / BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2v = user_context->use_bdf2 && user_context->bdf2_have_history && !user_context->use_picard;
  double a_c = 1.0, b_c = 1.0, c_c = 0.0;  // BDF2-on-V weights (a_c V^{n+1} - b_c V^n + c_c V^{n-1})
  if (bdf2v) {
    const double omega = user_context->deltat / user_context->bdf2_prev_dt;
    a_c = (1.0 + 2.0 * omega) / (1.0 + omega);
    b_c = 1.0 + omega;
    c_c = omega * omega / (1.0 + omega);
  }
  PetscScalar** my_starting_wtd_prev = nullptr;  // w^{n-1} (owned; storage is centre-only) for BDF2-on-V
  if (bdf2v) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  // TR-BDF2 (matrix-free): tr_stage 1 = trapezoidal (needs the precomputed explicit old-state flux+removal
  // tr_expl); tr_stage 2 = BDF2 from (w^n, Y_gamma). gamma=2-sqrt2; recharge conserves via c1*gamma+c3=1.
  // Head-scaled by Sy like BDF2-on-V. Intended standalone (not combined with -wtm_Tbar).
  const int    tr_stage = user_context->use_tr_bdf2 ? user_context->tr_stage : 0;
  // Named in src/tr_bdf2_coefficients.hpp, which also derives the STEP weights the water budget
  // needs. Kept as locals here so the OpenMP clause below can name them.
  const double TR_G  = trbdf2::GAMMA;
  const double tr_c1 = trbdf2::C1;
  const double tr_c2 = trbdf2::C2;
  const double tr_c3 = trbdf2::C3;
  PetscScalar** my_tr_ygamma = nullptr;
  PetscScalar** my_tr_expl   = nullptr;
  if (tr_stage == 2) PetscCall(DMDAVecGetArray(da, user_context->tr_ygamma, &my_tr_ygamma));
  if (tr_stage == 1) PetscCall(DMDAVecGetArray(da, user_context->tr_expl, &my_tr_expl));
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3: ET, owe, precip (m/yr)
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecGetArray(da, user_context->precip_vec, &my_precip));  // taper 3 deficit (E_eff - P)
  }
  PetscScalar** my_starting_wtd_local = nullptr;  // -wtm_Tbar: ghosted w^n for the time-averaged T̄
  if (g_Tbar) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));

  // Use the smooth (C-inf) T when a ksat smoothing width is set (universal across solver paths);
  // otherwise the exact piecewise (C0) Fan form (production). Widths are read once in update().
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  // Compute 1/T over the full ghost range so neighbor lookups in the owned-range loop below are valid.
  // -wtm_Tbar swaps the instantaneous T for the step-time-averaged T̄ (Kirchhoff-potential difference
  // against the ghosted w^n); off it, this is byte-identical to the instantaneous form.
#pragma omp parallel for default(none)                                                                          \
    shared(info, my_T, x, my_topo, my_fdepth, my_ksat, smooth_T, g_Tbar, my_starting_wtd_local)   \
    collapse(2)
  for (auto j = info->gys; j < info->gys + info->gym; j++) {
    for (auto i = info->gxs; i < info->gxs + info->gxm; i++) {
      const double wtd_T = x[j][i] - my_topo[j][i];   // the SNES variable is the head
      const double wtd_old = g_Tbar ? my_starting_wtd_local[j][i] : 0.0;  // w^n (ghosted); unused off -wtm_Tbar
      my_T[j][i] = 1. / interblockTransmissivity(wtd_T, wtd_old, my_fdepth[j][i], my_ksat[j][i], smooth_T);
    }
  }

  const bool dtr_on  = g_direct_to_runoff;
  const bool taper_on = g_evap_taper;
  const bool vol_storage = g_volume_storage;  // BE with volume-form (ΔV) storage instead of secant S·Δh
  const bool as_on    = g_active_set;         // -wtm_active_set: pin exfiltrating land cells at wtd=0 in-solve
#pragma omp parallel for default(none)                                                                                \
    shared(info, gew, gn, gs, x, my_T, my_mask, my_rech, user_context, my_porosity, my_starting_wtd, my_topo, f,      \
           my_evap, my_owe, my_precip, my_fdepth, my_ksat, dtr_on, taper_on, \
           bdf2v, vol_storage, a_c, b_c, c_c, my_starting_wtd_prev, smooth_T, g_land_boundary_dirichlet,      \
           tr_stage, TR_G, tr_c1, tr_c2, tr_c3, my_tr_ygamma, my_tr_expl, as_on, my_exfiltration,   \
           my_lake_stage) collapse(2)
  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      // The SNES variable IS the head. Used for the flux (h_c - h_nbr) and the centre wtd (h_c - topo).
      const auto head = [&](int jj, int ii) { return x[jj][ii]; };
      if (my_mask[j][i] == 0) {
        // Dirichlet condition: ocean head h = 0. f = x forces x = 0, with a unit Jacobian diagonal.
        // With -wtm_ghost_boundary this is the sea-level BC (ghost outside = 0), applied at real ocean cells.
        f[j][i] = x[j][i];
      } else {
        // Conservative finite-volume flux, HEAD form. The volume balance is
        //   A_j*S*(h - my_rech) + dt*(net outflow) = 0; we divide by A_j*S so the residual stays in
        // head units (O(metres)) -- the matrix-free Anderson solver diverges (DIVERGED_DTOL) on the
        // area-scaled volume-form residual, and, having no matrix, gains nothing from it: the root
        // (hence the solution and its conservation) is IDENTICAL. Face conductances G =
        // e*(L_wall/d_centre): E-W uses geom_ew, N/S the FACE-centred geom_n/geom_s, so shared-face
        // fluxes cancel (mass conserving) and the E-W/N-S cell sizes are no longer swapped. The
        // Picard OPERATOR keeps the volume form (it needs the exact symmetry). See GRID_CONVENTION.md.
        const double this_x = head(j, i);  // centre head (Kirchhoff: Φ⁻¹(x)+topo)
        const double this_T = my_T[j][i];  // 1/T at the centre
        // Per-face outflow = harmonic-mean face T * geom * (h_c - h_nbr). An OFF-MAP face (a global domain
        // edge; only reached for a LAND edge cell when -wtm_ghost_boundary skips setEdges -- ocean edges are
        // mask==0 = Dirichlet h=0 above) uses a GHOST node whose water-table SURFACE slope equals the LAND
        // surface slope: ghost head = this_x + (topo_c - topo_inland), ghost 1/T = this_T. The inland cell is
        // the reflection (2j-nj, 2i-ni) -- one step toward the interior -- so this reads only inward and never
        // goes out of bounds under DM_BOUNDARY_NONE. Topo gradient is from FIXED data (stable). See task #96.
        const auto face_out = [&](int nj, int ni, double G) -> double {
          double h_nbr, Tinv_nbr;
          if (nj < 0 || nj >= info->my || ni < 0 || ni >= info->mx) {  // off-map land edge: ghost node
            if (g_land_boundary_dirichlet) {  // dirichlet: ghost = an ocean neighbour (head 0, surface T),
              h_nbr    = 0.0;                  // identical to how ocean cells impose Dirichlet -> reproduces
              Tinv_nbr = 1.0 / interblockTransmissivity(0.0, 0.0, my_fdepth[j][i], my_ksat[j][i], smooth_T);
            } else {                           // neumann_toposlope (default): terrain-following no-flow
              const double topo_inland = my_topo[2 * j - nj][2 * i - ni];
              h_nbr    = this_x + (my_topo[j][i] - topo_inland);
              Tinv_nbr = this_T;
            }
          } else {
            h_nbr    = head(nj, ni);
            Tinv_nbr = my_T[nj][ni];
          }
          return (2. / (this_T + Tinv_nbr)) * G * (this_x - h_nbr);
        };
        const double net_outflow = face_out(j, i + 1, gew[j][i]) + face_out(j, i - 1, gew[j][i])
                                 + face_out(j + 1, i, gn[j][i]) + face_out(j - 1, i, gs[j][i]);

        const double A_j = user_context->cellsize_NS_squared / gew[j][i];  // cell area
        const double w_c = this_x - my_topo[j][i];                         // centre-cell wtd
        const double S   = updateEffectiveStorativity(my_starting_wtd[j][i], w_c, my_porosity[j][i]);

        // Sub-surface sink (taper 1) and demand-identity evaporation (taper 2), as head-form removals
        // dt*Q/S: evaluated at the current iterate, so implicit at the Anderson root. Matrix-free ->
        // no tangent needed (unlike the Picard operator). Off unless their flags are set.
        double removal = 0.0;  // m/s
        if (dtr_on) removal += directToRunoffRemoval(w_c, user_context->deltat);
        if (taper_on) removal += evapRemoval(w_c, my_evap[j][i], my_owe[j][i], my_precip[j][i] / SECONDS_IN_A_YEAR);

        if (tr_stage == 1) {
          // TR-BDF2 stage 1 (trapezoidal to t+gamma*dt), head-scaled by Sy(Y_gamma). w_c is the iterate =
          // Y_gamma: implicit half at Y_gamma + explicit half at w^n (my_tr_expl, precomputed in update()).
          // Recharge over gamma*dt = gamma*my_rech (constant source, exact).
          const double poro = my_porosity[j][i];
          const double Sy   = specificYield(w_c, poro);
          const double storage = (storedVolume(w_c, poro) - storedVolume(my_starting_wtd[j][i], poro)) / Sy;
          const double impl    = user_context->deltat * net_outflow / A_j + user_context->deltat * removal;
          f[j][i] = storage + 0.5 * TR_G * (impl + my_tr_expl[j][i]) / Sy - TR_G * my_rech[j][i] / Sy;
        } else if (tr_stage == 2) {
          // TR-BDF2 stage 2 (BDF2 from w^n and Y_gamma to w^{n+1}), head-scaled by Sy(w^{n+1}). w_c is the
          // iterate = w^{n+1}. Recharge conserves over the whole step (c1*gamma + c3 = 1; Y_gamma already
          // carries gamma*my_rech via V(Y_gamma)).
          const double poro   = my_porosity[j][i];
          const double Sy     = specificYield(w_c, poro);
          const double wtd_Yg = my_tr_ygamma[j][i] - my_topo[j][i];  // tr_ygamma stores the HEAD; V needs wtd
          const double storage = (storedVolume(w_c, poro) - tr_c1 * storedVolume(wtd_Yg, poro)
                                  + tr_c2 * storedVolume(my_starting_wtd[j][i], poro)) / Sy;
          f[j][i] = storage
                    + tr_c3 * (user_context->deltat * net_outflow / A_j + user_context->deltat * removal) / Sy
                    - tr_c3 * my_rech[j][i] / Sy;
        } else if (bdf2v) {
          // 2nd-order-in-time: storage = 3-level BDF2 difference of the stored VOLUME, head-scaled by the
          // specific yield Sy = dV/dh (so the residual stays O(metres) for Anderson; scaling by a positive
          // per-cell factor leaves the root/accuracy unchanged). RHS b=0 on this path (FormRHS), so the
          // whole storage lives here. Matches the Picard BDF2-on-V fixed point:
          //   a_c V(w^{n+1}) - b_c V(w^n) + c_c V(w^{n-1}) - Sy·rech + dt·N/A_j + dt·removal = 0, ÷Sy.
          const double poro = my_porosity[j][i];
          const double Sy   = specificYield(w_c, poro);
          const double storage = (a_c * storedVolume(w_c, poro) - b_c * storedVolume(my_starting_wtd[j][i], poro)
                                  + c_c * storedVolume(my_starting_wtd_prev[j][i], poro)) / Sy;
          f[j][i] = storage - my_rech[j][i] / Sy + user_context->deltat * net_outflow / (A_j * Sy)
                    + user_context->deltat * removal / Sy;
        } else if (vol_storage) {
          // dev.storage_form: volume -- backward Euler (1st-order in time, NO BDF2 history) but with the EXACT
          // stored-volume change ΔV = V(w^{n+1}) − V(w^n) instead of the secant S·Δh below. Identical in
          // form to the bdf2v branch with (a_c,b_c,c_c)=(1,1,0). Head-scaled by Sy = dV/dh so the residual
          // stays O(metres) for Anderson (a positive per-cell scale leaves the root unchanged); RHS b=0
          // (FormRHS). Converges (dt→0) to the SAME limit as TR-BDF2 / BDF2-on-V, unlike the secant BE,
          // whose S·Δh ≠ ΔV at surface crossings. See finding_cc_secant_storage_inconsistency.
          const double poro = my_porosity[j][i];
          const double Sy   = specificYield(w_c, poro);
          const double storage = (storedVolume(w_c, poro) - storedVolume(my_starting_wtd[j][i], poro)) / Sy;
          f[j][i] = storage - my_rech[j][i] / Sy + user_context->deltat * net_outflow / (A_j * Sy)
                    + user_context->deltat * removal / Sy;
        } else {
          // Backward Euler (secant storativity): the SNES RHS b=h^n supplies the previous-step storage.
          // Recharge is a fixed VOLUME (depth) my_rech; dividing by the secant S gives its head-form
          // contribution so the realized volume is exactly my_rech at any storativity (below the surface
          // S=porosity and my_rech/S reduces to the old head, byte-identical).
          f[j][i] = (this_x - my_rech[j][i] / S) + user_context->deltat * net_outflow / (A_j * S)
                    + user_context->deltat * removal / S;
        }
        // my_rech is converted to appropriate recharge for this timestep and starting water
        // table outside of the solve.
        // -wtm_active_set [EXPERIMENTAL]: LAKE-AWARE semismooth exfiltration constraint, enforced INSIDE the solve
        // (enforcement-independent -- not a post-solve clamp `explicit` nor an in-residual siphon `implicit`).
        // THE OBSTACLE. The head may not exceed  topo + surface_water_depth  -- ONE single-valued surface
        // over the whole land domain, equal to the lake stage inside a depression, to sea level on land
        // below sea level, and to the land surface everywhere else. Writing it as a DEPTH above topo (rather
        // than storing the elevation) is deliberate: topo cancels out of the residual below, the quantity is
        // independent of topography (which transient runs re-interpolate every cycle, so a stored elevation
        // would go stale), and it costs no array -- just this max on a value already in registers.
        //
        // In wtd variables: wtd <= surface_water_depth, where surface_water_depth = max(0, starting_wtd) is
        // the ponded depth left by the PREVIOUS step's FillSpillMerge (0 off lakes; the one-step lag).
        // Verified flat: on a real lake the per-cell depths differ but topo + surface_water_depth is constant
        // to sigma = 0 across the lake, i.e. one free-surface ELEVATION, which is what a lake should have.
        //
        // NOTE (fragility): surface_water_depth is INFERRED from starting_wtd, so it silently depends on FSM
        // having written its result there. -wtm_fsm_continuous skips exactly that write, which collapses
        // this to 0 everywhere and drains every lake. The two are incompatible until the stage is carried
        // explicitly. See benchmark/scheme_bench/README.md.
        //
        // Complementarity 0 <= (surface_water_depth - w_c) ⊥ (exfiltration flux) >= 0, on the mass residual
        // R = f (head units): the semismooth min-NCP is f <- max(w_c - surface_water_depth, R). Below the
        // free surface the normal residual R drives mass balance (f=R->0); if the cell overshoots ABOVE it
        // the (w_c - surface_water_depth) branch skims the excess to runoff. Off lakes this reduces to the
        // wtd<=0 pin (skim at the land surface = hillslope discharge). On lakes the aquifer keeps water up to
        // the lake stage -- its head is felt DURING the solve -- and only the OVERFLOW above the stage is
        // skimmed; the lake is a finite reservoir whose level is free to fall (drying). Continuous at the
        // shore (surface_water_depth->0), so no discrete wet/dry switch. LAND ONLY: ocean cells take the
        // Dirichlet h=0 branch above and never reach here. The branches MEET at the free surface (both
        // 0) so f is CONTINUOUS -- a hard switch's jump makes matrix-free Anderson diverge. Pinned-cell exfiltration
        // flux (the residual R discarded into the max) is captured for mass conservation: depth-form residual
        // is f*Sy, so an over-supplied cell (f<0) sheds exfiltration_depth = max(0,-f*Sy) to sink_removed_dist ->
        // FSM and total_surface_removed (budget). At convergence only cells held AT the free surface carry a
        // nonzero residual, so freely-solving (incl. below-stage lake) cells shed ~0. Anderson residual only.
        if (as_on) {
          const double surface_water_depth = my_lake_stage[j][i];  // lagged FSM lake stage (0 off lakes)
          const double pin = w_c - surface_water_depth;
          // Capture the multiplier ONLY on the ACTIVE SET -- the cells where the pin branch actually
          // wins. Off the active set the multiplier is zero BY DEFINITION: there the free residual is
          // what the solve drives to zero, so nothing is being discarded. Capturing max(0, -f*Sy) on
          // every cell therefore does not read a near-zero quantity, it RECTIFIES the converged
          // residual noise -- clipping the negative half and keeping the positive -- into a
          // systematically positive bias that ACCUMULATES monotonically instead of averaging out.
          // And it is not only a reporting bias: the captured depth is added to sink_removed_dist and
          // handed to FillSpillMerge, while the solve removed nothing from that cell, so with FSM on
          // it is water CREATED. Measured on tests/local_ledger arm B, table 8-20 m below the surface
          // where nothing can exfiltrate: 31.27 m^3 of phantom removal over 20 cycles, exactly 0 after
          // this gate. Every active-set budget arm tightened (TR-BDF2 1.37e-07 -> 5.67e-08).
          if (my_exfiltration)
            my_exfiltration[j][i] =
                (pin >= f[j][i]) ? std::max(0.0, -f[j][i] * specificYield(w_c, my_porosity[j][i])) : 0.0;
          f[j][i] = std::max(pin, f[j][i]);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  if (g_active_set) PetscCall(DMDAVecRestoreArray(da, user_context->exfiltration_vec, &my_exfiltration));
  if (g_active_set) PetscCall(DMDAVecRestoreArray(da, user_context->lake_stage, &my_lake_stage));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecRestoreArray(da, user_context->precip_vec, &my_precip));
  }
  if (g_Tbar) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));
  if (bdf2v) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  if (tr_stage == 2) PetscCall(DMDAVecRestoreArray(da, user_context->tr_ygamma, &my_tr_ygamma));
  if (tr_stage == 1) PetscCall(DMDAVecRestoreArray(da, user_context->tr_expl, &my_tr_expl));

  PetscLogFlops(info->xm * info->ym * (72.0));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormJacobianLocal - Analytic 5-point Jacobian of FormFunctionLocal (the exact ∂F/∂x of the
   conservative-FV head-form residual). Registered on the opt-in Newton-Krylov path (solver.method: newton;
   see update()); the SNESSolve constant b = hⁿ is independent of x, so J(F − b) = J(F).

   Residual (land cells): f = (x_c − rech) + dt·N/(A_j·S) + dt·removal/S, with
     N       = Σ_X e_X·G_X·(x_c − x_X)      conservative-FV net outflow (X ∈ {E,W,N,S})
     e_X     = 2/(τ_c + τ_X),  τ = 1/T       harmonic-mean face conductance
     G_X     = geom_ew (E,W) / geom_n / geom_s   face geometry factor
     A_j     = cellsize_NS² / geom_ew        cell area
     S       = updateEffectiveStorativity(wⁿ_c, w_c, poro)   secant storativity (centre only)
     removal = surfaceSink(w_c) + evapRemoval(w_c, …)        tapers (centre only)
   with w = x − topo. Differentiating w.r.t. the centre head x_c and the four neighbour heads x_X:
     ∂f/∂x_X = B·G_X·[ −2·τ'_X/sum_X²·(x_c−x_X) − e_X ]                        (off-diagonal)
     ∂f/∂x_c = 1 + B·Σ_X G_X·[ e_X − 2·τ'_c/sum_X²·(x_c−x_X) ]
               − (S'/S)·(flux_term + removal_term) + D·removal'               (diagonal)
   where B = dt/(A_j·S), D = dt/S, τ' = dTransmissivityInverseDwtd (d(1/T)/dw of the SMOOTH T),
   S' = dEffectiveStorativityDnew, removal' = surfaceSinkTangent + evapRemovalTangentRaw (the
   UNCLAMPED evap tangent, so this is the exact derivative of the residual as written).

   For ocean cells (mask == 0): J = I (unit diagonal for the Dirichlet f = x). Ocean neighbours of a
   land cell ARE coupled (their column carries the true off-diagonal); their own row pins dx = 0.

   INEXACT-NEWTON note: τ' is always the SMOOTH-T derivative, while the residual uses the piecewise
   (C0) Fan T unless a -wtm_ksat_*_smoothing_width is set. So with no smoothing width this Jacobian
   is a differentiable inexact-Newton approximation; to VERIFY it against FD with -snes_test_jacobian,
   set positive -wtm_ksat_soilbottom_smoothing_width / -wtm_ksat_surface_smoothing_width (and a
   -wtm_storativity_surface_smoothing_width) so residual and derivative use the identical smooth forms.

   Uses the SAME local ghosted vectors (topo_local/fdepth_local/ksat_local) and neighbour-access
   pattern as FormFunctionLocal, so it is exactly as MPI-safe as the residual.
 */
static PetscErrorCode FormJacobianLocal(
    DMDALocalInfo* info, PetscScalar** x, Mat Jmat, Mat P, AppCtx* user_context) {
  DM           da = user_context->da;
  PetscScalar **my_mask, **my_fdepth, **my_ksat, **my_topo, **my_porosity, **my_starting_wtd, **gew, **gn, **gs;
  PetscScalar** my_lake_stage_J = nullptr;  // the active-set obstacle; see FormFunctionLocal

  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  if (g_active_set) PetscCall(DMDAVecGetArray(da, user_context->lake_stage, &my_lake_stage_J));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3 inputs (m/yr)
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecGetArray(da, user_context->precip_vec, &my_precip));
  }
  PetscScalar** my_starting_wtd_local = nullptr;  // -wtm_Tbar: ghosted w^n for the time-averaged T̄
  if (g_Tbar) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));
  PetscScalar** my_rech;  // fixed-volume recharge (depth); enters the residual as −my_rech/S (1/S-scaled)
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));

  const bool   smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  const bool   taper_on = g_evap_taper;
  const double dt       = user_context->deltat;
  const double cns2     = user_context->cellsize_NS_squared;

  // τ = 1/T and its wtd-derivative τ'. Off -wtm_Tbar: the instantaneous T (smooth if a ksat width is
  // set, else piecewise) and dTransmissivityInverseDwtd (the SMOOTH-T derivative -> exact only when
  // smooth_T; else a differentiable inexact-Newton approximation). With -wtm_Tbar: the step-time-
  // averaged T̄ = (Φ(w)−Φ(w^n))/(w−w^n) and its EXACT tangent d(1/T̄)/dw = −T̄'/T̄², with
  // T̄' = [T(w) − T̄]/(w−w^n) (Δ→0 limit T'(w)/2) built from the piecewise T (Φ's derivative) -- so on
  // the -wtm_Tbar path this is an exact analytic Jacobian of the T̄ residual (T̄ is C1 -> FD-verifiable).
  const auto Tinv = [&](double w_new, double w_old, double fd, double ks) {
    const double T = interblockTransmissivity(w_new, w_old, fd, ks, smooth_T);
    return T > 0.0 ? 1.0 / T : 1e30;
  };
  const auto tauPrime = [&](double w_new, double w_old, double fd, double ks) {
    if (!g_Tbar) return dTransmissivityInverseDwtd(w_new, fd, ks);
    const double Tb = interblockTransmissivity(w_new, w_old, fd, ks, smooth_T);
    if (Tb <= 0.0) return 0.0;
    const double dwtd  = w_new - w_old;
    const double dTbar = (std::abs(dwtd) > 1e-9)
                             ? (depthIntegratedTransmissivity(w_new, fd, ks) - Tb) / dwtd
                             : 0.5 * dDepthIntegratedTransmissivityDwtd(w_new, fd, ks);
    return -dTbar / (Tb * Tb);
  };

  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      MatStencil row;
      row.j = j; row.i = i; row.c = 0;

      if (my_mask[j][i] == 0) {
        const PetscScalar one = 1.0;
        MatStencil col;
        col.j = j; col.i = i; col.c = 0;
        MatSetValuesStencil(Jmat, 1, &row, 1, &col, &one, INSERT_VALUES);
        if (P != Jmat) MatSetValuesStencil(P, 1, &row, 1, &col, &one, INSERT_VALUES);
        continue;
      }

      // ACTIVE-SET (semismooth) TANGENT. The residual applies f <- max(w_c - surface_water_depth, f)
      // (see FormFunctionLocal). Differentiating that max: where the PIN branch wins, the residual IS
      // w_c - surface_water_depth = x - (topo + surface_water_depth), so the row is d/dx = 1 with NO
      // neighbour coupling -- structurally identical to the ocean Dirichlet row above, except that the
      // imposed set is DISCOVERED each iteration from the current iterate rather than fixed by the mask.
      // That is what makes this a semismooth / primal-dual active-set Newton.
      //
      // WHY THIS MATTERS: without it the Newton residual (which HAS the pin) and the Newton Jacobian
      // (which did not) describe different functions. An inconsistent Jacobian is the textbook cause of
      // losing quadratic convergence and of line-search failure, which is what plain Newton did here.
      //
      if (g_active_set) {
        const double w_c_pin = x[j][i] - my_topo[j][i];
        const double swd_pin = my_lake_stage_J[j][i];
        if (w_c_pin - swd_pin > 0.0) {  // the max() picks the pin branch: this cell is in the active set
          const PetscScalar one = 1.0;
          MatStencil col;
          col.j = j; col.i = i; col.c = 0;
          MatSetValuesStencil(Jmat, 1, &row, 1, &col, &one, INSERT_VALUES);
          if (P != Jmat) MatSetValuesStencil(P, 1, &row, 1, &col, &one, INSERT_VALUES);
          continue;
        }
      }

      // wtd at centre and 4 neighbours from the SNES variable (x is the head), matching the residual.
      const auto wtd_of = [&](int jj, int ii) { return x[jj][ii] - my_topo[jj][ii]; };
      const double w_c = wtd_of(j, i);

      // w^n (ghosted) at the centre -- the "before" state for the -wtm_Tbar time-average; ignored (0)
      // off the -wtm_Tbar path (Tinv/tauPrime do not read it there).
      const auto wold_of = [&](int jj, int ii) { return g_Tbar ? my_starting_wtd_local[jj][ii] : 0.0; };
      const double wo_c = wold_of(j, i);

      // τ = 1/T and its wtd-derivative τ' at the centre (T̄-aware; see the lambdas above)
      const double tau_c  = Tinv(w_c, wo_c, my_fdepth[j][i], my_ksat[j][i]);
      const double taup_c = tauPrime(w_c, wo_c, my_fdepth[j][i], my_ksat[j][i]);
      const double h_c    = w_c + my_topo[j][i];  // centre head, h = wtd + topo

      const double A_j = cns2 / gew[j][i];  // cell area (matches the residual)
      const double S   = updateEffectiveStorativity(my_starting_wtd[j][i], w_c, my_porosity[j][i]);
      const double Sp  = dEffectiveStorativityDnew(my_starting_wtd[j][i], w_c, my_porosity[j][i]);
      const double B   = dt / (A_j * S);  // flux prefactor
      const double D   = dt / S;          // removal prefactor

      // Per-face flux, its centre-derivative ∂N/∂x_c, and the off-diagonal ∂f/∂x_X. An OFF-MAP face (a
      // global domain edge; only reached for a LAND edge cell when -wtm_ghost_boundary skips setEdges)
      // uses the land-slope ghost of FormFunctionLocal: τ_nbr = τ_c and h_nbr = h_c + (topo_c − topo_inland),
      // both functions of the CENTRE variable only, so dX = h_c − h_nbr = topo_inland − topo_c is CONSTANT
      // in x, the face has NO off-diagonal column (τ_nbr = τ_c is not an independent unknown), and its only
      // Jacobian entry is d(e·G·dX)/dw_c = G·dX·(−τ'_c/τ_c²) with e = 1/τ_c. The inland cell is the inward
      // reflection (2j−nj, 2i−ni), so this reads only toward the interior -- never out of bounds. For an
      // in-bounds face the standard 5-point terms are recovered, in E,W,N,S order (FP-identical to before).
      struct FaceGeom { int dj, di; double G; };
      const FaceGeom faces[4] = {{0, 1, gew[j][i]}, {0, -1, gew[j][i]}, {1, 0, gn[j][i]}, {-1, 0, gs[j][i]}};
      double net_outflow = 0.0, dN_dc = 0.0;
      double J_nbr[4]   = {0.0, 0.0, 0.0, 0.0};                 // off-diagonals (unused for off-map: no column)
      int    nbr_j[4], nbr_i[4];                                // neighbour stencil (only used when in-bounds)
      bool   nbr_inb[4];                                        // true = emit an off-diagonal column for this face
      for (int fi = 0; fi < 4; ++fi) {
        const int    nj = j + faces[fi].dj, ni = i + faces[fi].di;
        const double G  = faces[fi].G;
        nbr_j[fi] = nj; nbr_i[fi] = ni;
        if (nj < 0 || nj >= info->my || ni < 0 || ni >= info->mx) {  // off-map land edge: ghost node
          nbr_inb[fi] = false;  // NO stencil column either way: the ghost is not an independent unknown
          if (g_land_boundary_dirichlet) {  // dirichlet: ghost = ocean neighbour (head 0, surface τ)
            // Structurally the in-bounds case with τ_nbr = τ_surf (constant in x, so taup_nbr = 0) and
            // h_nbr = 0 (dX = h_c). No column because the ghost is not an unknown. Mirrors the residual's
            // surface-T ghost, so the Jacobian matches finite differences.
            const double tau_s = Tinv(0.0, 0.0, my_fdepth[j][i], my_ksat[j][i]);  // surface 1/T(0), constant in x
            const double sumS  = tau_c + tau_s, e_S = 2.0 / sumS;
            const double dX    = h_c;  // h_c − 0
            net_outflow += e_S * G * dX;
            dN_dc       += G * (e_S - 2.0 * taup_c / (sumS * sumS) * dX);
          } else {  // neumann_toposlope (default): terrain-following no-flow (τ_nbr = τ_c, dX constant in x)
            const double topo_inland = my_topo[j - faces[fi].dj][i - faces[fi].di];
            const double dX = topo_inland - my_topo[j][i];  // h_c − h_nbr, constant in x
            net_outflow += (1.0 / tau_c) * G * dX;
            dN_dc       += G * dX * (-taup_c / (tau_c * tau_c));  // d[(1/τ_c)·G·dX]/dw_c
          }
        } else {
          nbr_inb[fi] = true;
          const double w_X    = wtd_of(nj, ni);
          const double tau_X  = Tinv(w_X, wold_of(nj, ni), my_fdepth[nj][ni], my_ksat[nj][ni]);
          const double taup_X = tauPrime(w_X, wold_of(nj, ni), my_fdepth[nj][ni], my_ksat[nj][ni]);
          const double sumX = tau_c + tau_X, e_X = 2.0 / sumX;
          const double dX   = h_c - (w_X + my_topo[nj][ni]);
          net_outflow += e_X * G * dX;
          dN_dc       += G * (e_X - 2.0 * taup_c / (sumX * sumX) * dX);
          J_nbr[fi]    = B * G * (-2.0 * taup_X / (sumX * sumX) * dX - e_X);
        }
      }

      double removal = 0.0, rho = 0.0;  // removal [m/s] and its exact (unclamped) wtd-derivative
      if (taper_on) {
        const double p_rate = my_precip[j][i] / SECONDS_IN_A_YEAR;
        removal += evapRemoval(w_c, my_evap[j][i], my_owe[j][i], p_rate);
        rho     += evapRemovalTangentRaw(w_c, my_evap[j][i], my_owe[j][i], p_rate);
      }

      const double flux_term    = B * net_outflow;  // dt·N/(A_j·S)   (part of the residual)
      const double removal_term = D * removal;       // dt·removal/S   (part of the residual)

      // Recharge is a fixed volume entering the residual as −my_rech/S (a 1/S-scaled term like the
      // flux/removal), so it joins the −(S'/S)·(…) chain-rule group: ∂(−my_rech/S)/∂x_c = +(S'/S)·(my_rech/S).
      const double recharge_term = my_rech[j][i] / S;  // dt-independent; head-form recharge magnitude
      const double J_center =
          1.0 + B * dN_dc - (Sp / S) * (flux_term + removal_term - recharge_term) + D * rho;

      // Assemble a variable-length stencil: one off-diagonal per IN-BOUNDS face (E,W,N,S order), then the
      // centre. An OFF-MAP face contributes NO column (its land-slope ghost is internal to this cell, so its
      // whole Jacobian entry already sits on the centre diagonal via dN_dc) -- emitting an out-of-range
      // stencil column makes MatSetValuesStencil error ("inserting a new nonzero"/out-of-range), it is NOT
      // silently dropped. With the flag off every land cell is interior (all 4 faces in-bounds -> nc = 5,
      // identical to the fixed 5-point stencil), so this is FP- and sparsity-identical there.
      MatStencil  cols[5];
      PetscScalar vals[5];
      int nc = 0;
      for (int fi = 0; fi < 4; ++fi) {
        if (!nbr_inb[fi]) continue;
        cols[nc].j = nbr_j[fi]; cols[nc].i = nbr_i[fi]; cols[nc].c = 0;
        vals[nc] = J_nbr[fi];
        ++nc;
      }
      cols[nc].j = j; cols[nc].i = i; cols[nc].c = 0;
      vals[nc] = J_center;
      ++nc;
      MatSetValuesStencil(Jmat, 1, &row, nc, cols, vals, INSERT_VALUES);
      if (P != Jmat) MatSetValuesStencil(P, 1, &row, nc, cols, vals, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(Jmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Jmat, MAT_FINAL_ASSEMBLY);
  if (P != Jmat) {
    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
  }

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  if (g_active_set) PetscCall(DMDAVecRestoreArray(da, user_context->lake_stage, &my_lake_stage_J));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecRestoreArray(da, user_context->precip_vec, &my_precip));
  }
  if (g_Tbar) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormPicardRHS - right-hand side b(x) of the Picard system A(x) x = b(x)
   (PICARD_MATH.md sec 4). The production residual divides the whole flux
   divergence by the CENTRE storativity S_c; that makes the natural operator
   nonsymmetric, so each row is scaled by S_c to symmetrize it (this leaves the
   solution unchanged -- row scaling by a positive constant). The RHS carries the
   same S_c factor: on land, b = S_c * (starting_wtd + topo + rech); on ocean,
   b = 0 (Dirichlet h = 0). S_c depends on the current head, so b(x) genuinely
   depends on x here (S is frozen at the outer iterate, PETSc's SNESFunctionFn).
 */
static PetscErrorCode FormPicardRHS(SNES snes, Vec x, Vec b, void* ctx) {
  (void)snes;
  AppCtx* user_context = static_cast<AppCtx*>(ctx);
  DM      da           = user_context->da;
  PetscScalar **bb, **xx, **my_starting_wtd, **my_topo, **my_rech, **my_porosity, **my_mask, **gew;
  PetscScalar **my_fdepth, **my_ksat, **gn, **gs, **my_topo_g;  // for the land-slope-Neumann off-map ghost flux (see below)
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3: ET, owe, precip (m/yr)
  const double  cns2 = user_context->cellsize_NS_squared;  // cell area A_j = cns2 / geom_ew (volume form)

  // Variable-step BDF2 (once an h^{n-1} exists): b = S_c*(b_c*h^n - c_c*h^{n-1} + rech) with
  // omega = dt_n/dt_{n-1}, b_c = 1+omega, c_c = omega^2/(1+omega) (b_c=2, c_c=1/2 when the
  // step is constant -> uniform BDF2). Backward Euler otherwise: b = S_c*(h^n + rech).
  // h^{n-1} = starting_wtd_prev + topo (centre only).
  const bool bdf2      = user_context->use_bdf2 && user_context->bdf2_have_history;
  double a_c = 1.0, b_c = 0.0, c_c = 0.0;
  if (bdf2) {
    const double omega = user_context->deltat / user_context->bdf2_prev_dt;
    a_c                = (1.0 + 2.0 * omega) / (1.0 + omega);
    b_c                = 1.0 + omega;
    c_c                = omega * omega / (1.0 + omega);
  }
  PetscScalar** my_starting_wtd_prev = nullptr;

  PetscCall(DMDAVecGetArray(da, b, &bb));
  PetscCall(DMDAVecGetArray(da, x, &xx));  // owned range: S_c is a centre-cell quantity
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecGetArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // for the cell area A_j
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));    // off-map ghost flux geometry
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));  // off-map ghost flux: centre T_c
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo_g));    // ghosted topo for the inward reflection
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(da, &info));  // info.mx/my are GLOBAL dims -> the off-map bounds test
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  if (bdf2) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecGetArray(da, user_context->precip_vec, &my_precip));  // taper 3 deficit (E_eff - P)
  }

  const auto [xs, ys, xm, ym] = get_corners(da);
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      if (my_mask[j][i] == 0) {
        bb[j][i] = 0.0;  // Dirichlet ocean cell: h = 0
      } else if (bdf2) {
        // BDF2-on-V: storage = a*V(w) - b*V(w^n) + c*V(w^{n-1}), Picard-linearized about x_k so
        // the diagonal a*Sy(w_k) (in the operator) cancels a*Sy(w_k)*x_k here, leaving a*V(w_k) at
        // the fixed point. Volume form: the whole storage+recharge+sink RHS scales by the cell area
        // A_j (matching the operator's a*Sy*A_j diagonal and dt*G face conductances).
        const double poro  = my_porosity[j][i];
        const double w_k   = xx[j][i] - my_topo[j][i];
        const double Sy    = specificYield(w_k, poro);
        const double A_j   = cns2 / gew[j][i];
        bb[j][i] = A_j * (a_c * Sy * xx[j][i] - a_c * storedVolume(w_k, poro)
                        + b_c * storedVolume(my_starting_wtd[j][i], poro)
                        - c_c * storedVolume(my_starting_wtd_prev[j][i], poro)
                        + my_rech[j][i]);  // fixed-volume recharge (depth); no storativity scaling
        if (g_direct_to_runoff) {
          // Exfiltration (runoff_collector=implicit): in-residual removal dt*max(0,w)/dt (the above-surface
          // excess), Picard-linearized about w_k in the SAME form as the sink. The removal is exactly linear
          // where active (rate w/dt for w>0), so this linearization is exact; it matches the operator's
          // dt*R'(w_k)*A_j = A_j exfiltrating-cell diagonal below (a frozen active set: the exfiltrating set is fixed
          // each Picard sweep). Mutually exclusive with the sink under the selector.
          const double dt = user_context->deltat;
          bb[j][i] += A_j * (dt * directToRunoffTangent(w_k, dt) * xx[j][i] - dt * directToRunoffRemoval(w_k, dt));
        }
        if (g_evap_taper) {
          // Taper 2: implicit demand-identity evaporation dt*E_eff(w^{n+1}) (ET -> owe), Picard-
          // linearized about w_k with the SPD-clamped tangent (matches the operator's evap diagonal).
          const double dt     = user_context->deltat;
          const double p_rate = my_precip[j][i] / SECONDS_IN_A_YEAR;  // taper 3: deficit (E_eff - P)
          bb[j][i] += A_j * (dt * evapRemovalTangent(w_k, my_evap[j][i], my_owe[j][i], p_rate) * xx[j][i]
                             - dt * evapRemoval(w_k, my_evap[j][i], my_owe[j][i], p_rate));
        }
      } else {
        const double S_c =
            updateEffectiveStorativity(my_starting_wtd[j][i], xx[j][i] - my_topo[j][i], my_porosity[j][i]);
        const double h_n = my_starting_wtd[j][i] + my_topo[j][i];
        const double A_j = cns2 / gew[j][i];  // volume form: scale the storage/recharge by the cell area
        if (bdf2) {
          const double h_nm1 = my_starting_wtd_prev[j][i] + my_topo[j][i];
          bb[j][i] = A_j * (S_c * (b_c * h_n - c_c * h_nm1) + my_rech[j][i]);  // recharge = fixed volume
        } else {
          bb[j][i] = A_j * (S_c * h_n + my_rech[j][i]);  // recharge = fixed volume (depth), out of S_c
        }
      }

      // Land-slope-Neumann off-map faces (-wtm_ghost_boundary): the ghost flux across a global-edge LAND
      // face is e·G·(topo_c − topo_inland) with e = 1/τ_c = T_c (harmonic mean of T_c with itself, since
      // τ_nbr = τ_c) -- CONSTANT in x, so FormPicardOperator omits it from A and it is supplied here on the
      // RHS. Only the centre cell's T_c is needed; the reflection reads the ghosted topo one step INWARD, so
      // it never goes OOB. Inert with the flag off (setEdges makes every edge cell ocean -> no land cell
      // reaches a global edge). Matches FormFunctionLocal's off-map face and FormPicardOperator's omission.
      if (my_mask[j][i] != 0) {
        const double w_c   = xx[j][i] - my_topo[j][i];
        const double w_old = g_Tbar ? my_starting_wtd[j][i] : 0.0;  // owned centre == starting_wtd_local[j][i]
        const double T_c   = interblockTransmissivity(w_c, w_old, my_fdepth[j][i], my_ksat[j][i], smooth_T);
        const auto add_offmap = [&](int nj, int ni, double G) {
          if (nj < 0 || nj >= info.my || ni < 0 || ni >= info.mx) {
            if (g_land_boundary_dirichlet) return;  // dirichlet ghost head = 0: absorbing term is on the
                                                    // operator diagonal (FormPicardOperator), nothing on the RHS
            const double topo_inland = my_topo_g[2 * j - nj][2 * i - ni];  // ghosted; inward reflection
            bb[j][i] += user_context->deltat * T_c * G * (my_topo[j][i] - topo_inland);  // neumann constant flux
          }
        };
        add_offmap(j, i + 1, gew[j][i]);
        add_offmap(j, i - 1, gew[j][i]);
        add_offmap(j + 1, i, gn[j][i]);
        add_offmap(j - 1, i, gs[j][i]);
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, b, &bb));
  PetscCall(DMDAVecRestoreArray(da, x, &xx));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo_g));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecRestoreArray(da, user_context->precip_vec, &my_precip));
  }
  if (bdf2) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_prev, &my_starting_wtd_prev));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormPicardOperator - the SPD operator A(x) of the Picard system A(x) x = b(x)
   (PICARD_MATH.md sec 4). It is the production backward-Euler operator

       (row c)   S_c*x_c + dt * sum_nbr e_{c,nbr} (x_c - x_nbr)

   i.e. the CENTRE-storativity discretization of the Anderson residual, ROW-SCALED
   by S_c so it is symmetric (the flux term dt*e is symmetric in the cell pair; the
   1/S_c that would otherwise multiply it -- and break symmetry -- is cleared by
   the scaling). The RHS carries the matching S_c factor (FormPicardRHS), so the
   scaling cancels and the fixed point is exactly the Anderson one. e is the
   harmonic mean of the PIECEWISE transmissivity. Diagonal = S_c + sum(dt*e) is
   strictly dominant -> SPD -> CG-compatible.

   Ocean (Dirichlet) rows/columns are eliminated symmetrically with
   MatZeroRowsColumnsStencil after assembly; h_ocean = 0, so no RHS correction is
   needed (x = b = NULL). This keeps each land cell's drain-to-ocean conductance in
   its diagonal while removing the asymmetric off-diagonal (PICARD_MATH.md 4.4).

   Only the harmonic-mean T needs neighbor values, so the iterate x is ghost-
   scattered here and read with topo/fdepth/ksat from their *_local ghost vectors;
   the centre-only S_c reads starting_wtd/porosity owned. A and P are the same
   matrix (A preconditions itself via GAMG).
 */
static PetscErrorCode FormPicardOperator(SNES snes, Vec x, Mat A, Mat P, void* ctx) {
  (void)snes;
  (void)P;  // A is its own preconditioner
  AppCtx* user_context = static_cast<AppCtx*>(ctx);
  DM      da           = user_context->da;

  // Ghost-scatter the current iterate so neighbor heads (for T) are valid under MPI.
  Vec xloc;
  PetscCall(DMGetLocalVector(da, &xloc));
  PetscCall(DMGlobalToLocalBegin(da, x, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(da, x, INSERT_VALUES, xloc));

  PetscScalar **xx, **my_topo, **my_fdepth, **my_ksat, **my_porosity, **my_starting_wtd, **my_mask, **cellsize_ew_sq,
      **my_T, **gew, **gn, **gs;
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3: ET, owe, precip (m/yr)
  PetscCall(DMDAVecGetArray(da, xloc, &xx));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));      // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));  // owned: centre S_c
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // conservative-FV flux geometry
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->T_local, &my_T));
  PetscScalar** my_starting_wtd_local = nullptr;  // -wtm_Tbar: ghosted w^n for the time-averaged T̄
  if (g_Tbar) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecGetArray(da, user_context->precip_vec, &my_precip));  // taper 3 deficit (E_eff - P)
  }

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(da, &info));

  // 1/T over the full ghost range so the neighbor harmonic means on the owned range are valid
  // (mirrors FormFunctionLocal). Production uses the piecewise (C0) Fan form; a positive
  // -wtm_ksat_soilbottom_smoothing_width (-1.5 m) and/or -wtm_ksat_surface_smoothing_width (0 m)
  // swaps in the smooth (C-inf) form, rounding that boundary. Both 0 (default) => piecewise. The
  // widths are read once per cycle in update() (universal across solver paths). -wtm_Tbar swaps the
  // instantaneous T for the step-time-averaged T̄ (against the ghosted w^n), matching the residual.
  const bool smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  for (auto j = info.gys; j < info.gys + info.gym; j++) {
    for (auto i = info.gxs; i < info.gxs + info.gxm; i++) {
      const double wtd_T   = xx[j][i] - my_topo[j][i];
      const double wtd_old = g_Tbar ? my_starting_wtd_local[j][i] : 0.0;  // w^n; unused off -wtm_Tbar
      my_T[j][i] = 1.0 / interblockTransmissivity(wtd_T, wtd_old, my_fdepth[j][i], my_ksat[j][i], smooth_T);
    }
  }

  const double dt   = user_context->deltat;
  const double cns2 = user_context->cellsize_NS_squared;

  // Variable-step BDF2 (once an h^{n-1} exists): a*h^{n+1} - b*h^n + c*h^{n-1} = dt*RHS,
  // with omega = dt_n/dt_{n-1}, a = (1+2w)/(1+w) [here], b,c on the RHS. The diffusion term
  // always carries the current dt; the storage diagonal carries a*S_c (a=3/2 when the step is
  // constant, i.e. w=1 -> uniform BDF2). Backward Euler is a=1. See BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2 = user_context->use_bdf2 && user_context->bdf2_have_history;
  double a_coeff  = 1.0;  // coefficient of h^{n+1} on the storativity diagonal (BE)
  if (bdf2) {
    const double omega = dt / user_context->bdf2_prev_dt;
    a_coeff            = (1.0 + 2.0 * omega) / (1.0 + omega);
  }
  // BDF2-on-V: use the TANGENT dV/dh on the diagonal (BDF2 applied to the volume), instead of the
  // backward-Euler secant storativity that caps the order at 1. Only once history exists (a BDF2
  // step); the BE bootstrap step keeps the secant. See BDF2_ADAPTIVE_DESIGN.md.

  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      const MatStencil row = {.k = 0, .j = j, .i = i, .c = 0};

      if (my_mask[j][i] == 0) {
        // Ocean: placeholder diagonal; MatZeroRowsColumnsStencil fixes it to identity.
        const PetscScalar one = 1.0;
        PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, &one, INSERT_VALUES));
      } else {
        // Conservative FINITE-VOLUME (volume-form) assembly: each row is the cell's VOLUME balance,
        // so off-diagonals are the shared face conductances dt*G (G = e * L_wall/d_centre) -- exactly
        // symmetric across every face and mass-conservative -- and the storage/sink diagonal carries
        // the cell area A_j. See benchmark/GRID_CONVENTION.md. (Was head-form, which divided the E-W
        // flux by cellsize_n_s^2 and the N-S flux by cellsize_e_w^2 -- the two swapped, off by
        // cos^2(lat), and non-conservative across N-S faces.)
        const double A_j = cns2 / gew[j][i];  // cell area = cellsize_n_s^2 / (cellsize_n_s/cellsize_e_w)

        // Storativity diagonal coefficient, frozen at the current x. BDF2-on-V uses the tangent
        // dV/dh (specificYield); otherwise the backward-Euler secant (matches FormFunctionLocal).
        const double w_k = xx[j][i] - my_topo[j][i];
        const double S_c =
            bdf2 ? specificYield(w_k, my_porosity[j][i])
                      : updateEffectiveStorativity(my_starting_wtd[j][i], w_k, my_porosity[j][i]);

        // Harmonic-mean interface transmissivities e = 2/(1/T_c + 1/T_nbr), times the face geometry
        // G = e * (L_wall/d_centre): E-W uses geom_ew (per row); N/S use the FACE-centred geom_n/geom_s,
        // so G_N(j) = G_S(j+1) exactly (shared face) -> conservative. An OFF-MAP face (global edge, only
        // for a LAND edge cell under -wtm_ghost_boundary) carries the land-slope-Neumann ghost flux
        // e·G·(topo_inland − topo_c), which is CONSTANT in x (the centre head cancels) -> it contributes
        // NOTHING to the SPD operator (no diagonal, no off-diagonal) and is placed on the RHS instead
        // (FormPicardRHS). The bounds test reads only in-bounds T (never OOB); inert with the flag off
        // (setEdges makes every edge cell ocean, so no land cell sits on the global boundary).
        // Storage scales with the cell area A_j (volume form).
        const double sink_diag = 0.0;  // taper-1 band sink retired (fork issue #7)
        // Taper 2 (+ taper 3) evaporation diagonal: dt*R'(w_k)*A_j, SPD-clamped >= 0 (matches the RHS
        // term). R' == E_eff' when taper 3 is off.
        const double evap_diag = (g_evap_taper && bdf2)
                                     ? dt * evapRemovalTangent(w_k, my_evap[j][i], my_owe[j][i],
                                                               my_precip[j][i] / SECONDS_IN_A_YEAR) * A_j
                                     : 0.0;
        // Exfiltration (runoff_collector=implicit): dt*R'(w_k)*A_j = A_j for a exfiltrating cell (w_k > 0), 0 below.
        // A frozen active-set diagonal (the exfiltrating set is fixed each Picard sweep); >= 0 -> SPD-preserving.
        // The matching RHS constant is added in FormPicardRHS. Mutually exclusive with the sink.
        const double dtr_diag = (g_direct_to_runoff && bdf2)
                                    ? dt * directToRunoffTangent(w_k, dt) * A_j : 0.0;

        // Variable-length stencil: one off-diagonal dt*G conductance per IN-BOUNDS face (E,W,N,S order),
        // then the centre. An OFF-MAP face emits NO column (its constant ghost flux is on the RHS) -- an
        // out-of-range stencil column makes MatSetValuesStencil error, it is NOT dropped. With the flag off
        // every land cell is interior (nc = 5) -> FP- and sparsity-identical to the fixed 5-point stencil.
        const int    fdj[4] = {0, 0, 1, -1};
        const int    fdi[4] = {1, -1, 0, 0};
        const double fG[4]  = {gew[j][i], gew[j][i], gn[j][i], gs[j][i]};
        MatStencil  cols[5];
        PetscScalar vals[5];
        int    nc       = 0;
        double face_sum = 0.0;  // Σ off-diagonals (= −Σ conductances), for the diagonal
        for (int fi = 0; fi < 4; ++fi) {
          const int nj = j + fdj[fi], ni = i + fdi[fi];
          if (nj < 0 || nj >= info.my || ni < 0 || ni >= info.mx) {  // off-map land edge
            if (g_land_boundary_dirichlet) {
              // Dirichlet ghost = an ocean neighbour (head 0, surface T): its flux e_S·G·h_c is LINEAR in
              // the centre head, so it goes on the DIAGONAL as an absorbing conductance (+dt·e_S·G, strictly
              // positive -> SPD-preserving) with NO column and NO RHS term (ghost head = 0). Mirrors an
              // in-bounds ocean neighbour, whose off-diagonal column is zeroed to the RHS at head 0 anyway.
              const double tau_s = 1.0 / interblockTransmissivity(0.0, 0.0, my_fdepth[j][i], my_ksat[j][i], smooth_T);
              face_sum += -dt * (2.0 / (my_T[j][i] + tau_s)) * fG[fi];
            }
            continue;  // neumann_toposlope: constant flux -> RHS (FormPicardRHS); dirichlet: handled above
          }
          const double A_x = -dt * (2.0 / (my_T[j][i] + my_T[nj][ni])) * fG[fi];
          cols[nc] = {.k = 0, .j = nj, .i = ni, .c = 0};
          vals[nc] = A_x;
          face_sum += A_x;
          ++nc;
        }
        cols[nc] = {.k = 0, .j = j, .i = i, .c = 0};                          // centre
        vals[nc] = a_coeff * S_c * A_j + sink_diag + dtr_diag + evap_diag - face_sum;  // diagonal (strictly dominant)
        ++nc;
        PetscCall(MatSetValuesStencil(A, 1, &row, nc, cols, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Symmetric Dirichlet elimination on ocean cells (see doc comment above).
  std::vector<MatStencil> ocean_rows;
  for (auto j = info.ys; j < info.ys + info.ym; j++) {
    for (auto i = info.xs; i < info.xs + info.xm; i++) {
      if (my_mask[j][i] == 0) {
        ocean_rows.push_back({.k = 0, .j = j, .i = i, .c = 0});
      }
    }
  }
  PetscCall(MatZeroRowsColumnsStencil(
      A, static_cast<PetscInt>(ocean_rows.size()), ocean_rows.data(), 1.0, nullptr, nullptr));

  PetscCall(DMDAVecRestoreArray(da, xloc, &xx));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->cellsize_EW_squared, &cellsize_ew_sq));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->T_local, &my_T));
  if (g_Tbar) PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));
  if (g_evap_taper) {
    PetscCall(DMDAVecRestoreArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecRestoreArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecRestoreArray(da, user_context->precip_vec, &my_precip));
  }
  PetscCall(DMRestoreLocalVector(da, &xloc));
  return 0;
}

}  // namespace FanDarcyGroundwater
