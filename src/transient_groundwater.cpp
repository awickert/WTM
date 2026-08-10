#include "transient_groundwater.hpp"
#include "add_recharge.hpp"
#include "update_effective_storativity.hpp"

#include <omp.h>
#include <array>
#include <chrono>
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
// SPD operator A(x). Gated behind -wtm_picard; default Anderson path unaffected.
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
// Incompatible with the -wtm_kirchhoff variable change (Φ + T_bedrock·wtd is not analytically invertible);
// enforced in update().
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

// --- Kirchhoff / discharge-potential transform (-wtm_kirchhoff) -----------------------------------
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
// FINDING (2026-08): as a change of variable on the HEAD-FORM residual this reaches the identical
// equilibrium (verified to 8.7e-8 m) but does NOT raise the dt ceiling -- it worsens conditioning. The
// exact chain-rule Jacobian is dF/dΦ = (dF/dh)/T (column scaling by 1/T); the head-form storage term
// (h - rech) contributes dh/dΦ = 1/T to the DIAGONAL, and for deep cells T ~ 1e-11 so 1/T ~ 1e11 blows
// the diagonal up (shallow cells get near-zero columns), and MUMPS fails as cells drain deep. The
// continuous Kirchhoff benefit (operator -> constant-coefficient Laplacian) does not transfer to the
// discrete harmonic-mean CONSERVATIVE scheme under a mere change of variable. Kept opt-in as a documented
// alternative; the 1/T blow-up is specific to the head form, so a VOLUME-form residual may transform more
// gracefully (WIP). See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
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
// Inverse Φ → wtd (piecewise; continuous, matches dischargePotential's branch boundaries).
static double dischargePotentialInverse(const double Phi, const double fdepth, const double ksat) {
  if (fdepth <= 0) return 0.0;
  constexpr double shallow = 1.5;
  const double fd = fdepth, k = ksat;
  const double Phi_sb = fd * fd * k;                                                        // Φ at wtd=-1.5
  const double Phi_0  = k * (0.5 * (shallow + fd) * (shallow + fd) + 0.5 * fd * fd);        // Φ at wtd= 0
  if (Phi < Phi_sb) return fd * std::log(std::max(Phi, 1e-300) / (fd * fd * k)) - shallow;  // exp
  if (Phi > Phi_0)  return (Phi - Phi_0) / (k * (shallow + fd));                            // surface
  return std::sqrt(std::max(2.0 * (Phi / k - 0.5 * fd * fd), 0.0)) - shallow - fd;          // linear
}
static bool g_kirchhoff = false;  // -wtm_kirchhoff: solve in the discharge potential Φ (Newton path)

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

// --- Sub-surface surface-water sink (-wtm_surface_sink; WIP prototype) ---------------------------
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
static bool             g_surface_sink       = false;
static bool             g_direct_to_runoff            = false; // -wtm_direct_to_runoff: excess-to-runoff seepage face
static double           g_relax                       = 1.0;   // -wtm_relax: sub-step under-relaxation (1=off); damps free-boundary flicker
static double           g_surface_sink_qmax  = 0.0;  // Qmax: peak removal rate [m/s]
static double           g_surface_sink_width = 1.0;  // w: band width below the surface [m]

// -wtm_surface_exfiltration_to_runoff: post-solve surface exfiltration-to-runoff collection. Standard clamped-T physics; water is allowed to
// mound during the solve, then clamped to wtd=0 with the exact above-surface storage routed to FSM (via
// the sink accumulator). A robust, tuning-free "collect" alternative to the implicit sink; needs the sink
// off to do anything. See the truncation site in update().
static bool             g_surface_exfiltration_to_runoff_array    = false;

// -- Fringe size (sink band width) source (-wtm_fringe_source). The sink band width is the physical
// capillary fringe height psi_a per cell, content-matched to the sink spline as w = psi_a / KAPPA_SINK
// (see benchmark/capillary_taper_math.tex). Modes: none (default) = today's numerical width
// g_surface_sink_width (byte-identical); fixed = uniform -wtm_fringe_length; ksat = pedotransfer
// psi_a = C*sqrt(n/ksat), capped at g_fringe_cap; file (not yet) = per-cell raster. Populated per cell
// into user_context.fringe_width_vec in update() and read at every sink call site.
enum FringeSource { FRINGE_NONE = 0, FRINGE_FIXED, FRINGE_KSAT };
static int              g_fringe_source    = FRINGE_NONE;
static double           g_fringe_length    = 0.1;    // -wtm_fringe_length [m]: fixed fringe height psi_a (FIXED)
static double           g_fringe_ksat_coef = 5e-4;   // -wtm_fringe_ksat_coef C [SI]: psi_a = C*sqrt(n/ksat)
static double           g_fringe_cap       = 2.0;    // -wtm_fringe_cap [m]: max psi_a
static constexpr double KAPPA_SINK         = 0.5;    // sink spline shape-factor (quintic) -> w = psi_a/KAPPA_SINK

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
// The band width w is now a PER-CELL argument (the fringe width, from the fringe-size field); callers pass
// it in. See the fringe-size field populated in update() and the taper-design principle (size = physical).
static double surfaceSinkRamp(const double wtd, const double w) {
  if (wtd <= -w) return 0.0;
  if (wtd >= 0.0) return 1.0;
  const double u = (wtd + w) / w;  // in (0,1)
  return u * u * u * (u * (6.0 * u - 15.0) + 10.0);
}
// d(ramp)/d(wtd) = p'(u)/w, with p'(u) = 30 u^2 (1-u)^2.
static double surfaceSinkRampTangent(const double wtd, const double w) {
  if (wtd <= -w || wtd >= 0.0) return 0.0;
  const double u = (wtd + w) / w;
  return 30.0 * u * u * (1.0 - u) * (1.0 - u) / w;
}
static double surfaceSink(const double wtd, const double w) { return g_surface_sink_qmax * surfaceSinkRamp(wtd, w); }
static double surfaceSinkTangent(const double wtd, const double w) {
  return g_surface_sink_qmax * surfaceSinkRampTangent(wtd, w);
}

// Seepage face (-wtm_direct_to_runoff): remove exactly the ABOVE-surface excess to runoff each step.
// removal RATE [m/s] = max(0,wtd)/dt, so dt*removal = the excess depth. Pins wtd<=0 (no rate cap -> no
// pile) and removes nothing below the surface (no depression). The Anderson solve tolerates the hard
// max fine -- an earlier softplus smoothing was inert (convergence was eps-invariant), so it is gone.
// (The tangent is the step function, for a future Newton path.)
static double directToRunoffRemoval(const double wtd, const double dt) { return std::max(0.0, wtd) / dt; }
static double directToRunoffTangent(const double wtd, const double dt) { return (wtd > 0.0 ? 1.0 : 0.0) / dt; }

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
    PetscInt ym) {
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
        double rech_count = rech_dist[y][x];
        if (starting_wtd[y][x] >= 0 && starting_wtd[y][x] + rech_dist[y][x] < 0)
          rech_count = -starting_wtd[y][x];

        arp.total_added_recharge += rech_count * arp.cell_area[y];
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
static void accumulate_ocean_outflow(AppCtx& user_context, ArrayPack& arp) {
  DM  da = user_context.da;
  Vec xloc;
  DMGetLocalVector(da, &xloc);
  DMGlobalToLocalBegin(da, user_context.x, INSERT_VALUES, xloc);
  DMGlobalToLocalEnd(da, user_context.x, INSERT_VALUES, xloc);

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
      if (my_mask[j][i + 1] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j][i + 1]) * gew[j][i] * h_c;
      if (my_mask[j][i - 1] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j][i - 1]) * gew[j][i] * h_c;
      if (my_mask[j + 1][i] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j + 1][i]) * gn[j][i] * h_c;
      if (my_mask[j - 1][i] == 0) arp.total_ocean_outflow_gw += dt * 2.0 / (my_T[j][i] + my_T[j - 1][i]) * gs[j][i] * h_c;
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

// Accumulate the solver's EXACT per-step discrete storage and specific-yield recharge over owned
// LAND cells, so the water budget closes to the SNES tolerance rather than the ~1% of the physical
// snapshot (whose gap is the BDF2-startup term, not a leak). The per-cell discrete balance the solve
// satisfies is  storage(w^{n+1},w^n,w^{n-1}) + dt*lateral_flux + dt*Q_sink = recharge_term, so summed
// over land cells (interior lateral fluxes cancel; the land->ocean flux is total_ocean_outflow):
//   total_storage_change = total_solver_recharge - total_ocean_outflow - total_surface_removed
// to SNES tolerance. The storage/recharge forms mirror FormPicardRHS exactly across its paths
// (BDF2-on-V uses V and specific yield; the secant paths use the effective storativity and heads).
// Called after the solve, BEFORE the BDF2 history overwrites w^{n-1}. Picard path only (the exact
// residual is not defined for the matrix-free Anderson default). See benchmark/WATER_BUDGET.md.
static void accumulate_budget_terms(AppCtx& user_context, ArrayPack& arp, DMDA_Array_Pack& dmdapack) {
  const bool bdf2      = user_context.use_bdf2 && user_context.bdf2_have_history;
  const bool bdf2_on_V = bdf2 && user_context.use_bdf2_on_V;
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
      if (bdf2_on_V) {
        storage  = a_c * storedVolume(w1, poro) - b_c * storedVolume(w0, poro) + c_c * storedVolume(wm1, poro);
        recharge = specificYield(w1, poro) * rech;
      } else {
        const double S_c = updateEffectiveStorativity(w0, w1, poro);  // secant storativity (matches the RHS)
        const double h1 = dmdapack.x[j][i], h0 = w0 + my_topo[j][i], hm1 = wm1 + my_topo[j][i];
        storage  = S_c * (a_c * h1 - b_c * h0 + c_c * hm1);
        recharge = S_c * rech;
      }
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
  DMDAVecGetArray(da, user_context.mask, &my_mask);
  DMDAVecGetArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecGetArray(da, user_context.geom_n_vec, &gn);
  DMDAVecGetArray(da, user_context.geom_s_vec, &gs);
  DMDAVecGetArray(da, user_context.T_local, &my_T);
  DMDAVecGetArray(da, user_context.tr_expl, &expl);
  PetscScalar **my_fringe;
  DMDAVecGetArray(da, user_context.fringe_width_vec, &my_fringe);
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
      const double e_E = 2.0 / (my_T[j][i] + my_T[j][i + 1]);
      const double e_W = 2.0 / (my_T[j][i] + my_T[j][i - 1]);
      const double e_N = 2.0 / (my_T[j][i] + my_T[j + 1][i]);
      const double e_S = 2.0 / (my_T[j][i] + my_T[j - 1][i]);
      const double N   = e_E * gew[j][i] * (h_c - (wn[j][i + 1] + my_topo[j][i + 1]))
                       + e_W * gew[j][i] * (h_c - (wn[j][i - 1] + my_topo[j][i - 1]))
                       + e_N * gn[j][i] * (h_c - (wn[j + 1][i] + my_topo[j + 1][i]))
                       + e_S * gs[j][i] * (h_c - (wn[j - 1][i] + my_topo[j - 1][i]));
      const double A_j = user_context.cellsize_NS_squared / gew[j][i];
      double removal = 0.0;
      if (g_direct_to_runoff)           removal += directToRunoffRemoval(wn[j][i], dt);
      else if (g_surface_sink) removal += surfaceSink(wn[j][i], my_fringe[j][i]);
      if (g_evap_taper)
        removal += evapRemoval(wn[j][i], my_evap[j][i], my_owe[j][i], my_precip[j][i] / SECONDS_IN_A_YEAR);
      expl[j][i] = dt * N / A_j + dt * removal;
    }
  DMDAVecRestoreArray(da, user_context.starting_wtd_local, &wn);
  DMDAVecRestoreArray(da, user_context.topo_local, &my_topo);
  DMDAVecRestoreArray(da, user_context.fdepth_local, &my_fdepth);
  DMDAVecRestoreArray(da, user_context.ksat_local, &my_ksat);
  DMDAVecRestoreArray(da, user_context.mask, &my_mask);
  DMDAVecRestoreArray(da, user_context.geom_ew_vec, &gew);
  DMDAVecRestoreArray(da, user_context.geom_n_vec, &gn);
  DMDAVecRestoreArray(da, user_context.geom_s_vec, &gs);
  DMDAVecRestoreArray(da, user_context.T_local, &my_T);
  DMDAVecRestoreArray(da, user_context.tr_expl, &expl);
  DMDAVecRestoreArray(da, user_context.fringe_width_vec, &my_fringe);
  if (g_evap_taper) {
    DMDAVecRestoreArray(da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecRestoreArray(da, user_context.precip_vec, &my_precip);
  }
}

// Overwrite the SNES initial guess x with the guarded 2nd-order history predictor
//   w^{n+1} ≈ w^n + ω(w^n − w^{n-1}),  ω = Δt/Δt_{n-1},   x = (w^n + step) + topo   (the head).
// This makes the iteration-1 T̄ a genuine step time-average instead of the instantaneous T(w^n), and
// starts every solver closer to the root. Guarded so a non-smooth trajectory (shock, GW↔SW surface
// crossing) cannot produce a wild guess: the predicted step is capped in magnitude and may not cross the
// land surface (wtd=0). Land cells only (ocean keeps its Dirichlet x=0). Call only once a history exists.
static void apply_predictor_guess(AppCtx& user_context) {
  DM da = user_context.da;
  const auto [xs, ys, xm, ym] = get_corners(da);
  constexpr double STEP_CAP = 50.0;  // m: absolute cap on the extrapolated step (guards a wild predictor)

  // Guard + write helper: clamp the step, forbid a land-surface (wtd=0) crossing in the GUESS (the
  // trajectory is non-smooth there), and store the head x = (w^n + step) + topo.
  const auto guarded_write = [&](PetscScalar** x, PetscScalar** wn, PetscScalar** topo, int j, int i,
                                 double step) {
    if (step > STEP_CAP) step = STEP_CAP;
    if (step < -STEP_CAP) step = -STEP_CAP;
    double pred = wn[j][i] + step;
    if (wn[j][i] < 0.0 && pred > 0.0) pred = 0.0;
    if (wn[j][i] > 0.0 && pred < 0.0) pred = 0.0;
    x[j][i] = pred + topo[j][i];  // SNES variable is the head
  };

  PetscScalar **x, **wn, **topo, **mask;
  DMDAVecGetArray(da, user_context.x, &x);
  DMDAVecGetArray(da, user_context.starting_wtd, &wn);
  DMDAVecGetArray(da, user_context.topo_vec, &topo);
  DMDAVecGetArray(da, user_context.mask, &mask);

  if (user_context.bdf2_have_history) {
    // Steps 2+: 2nd-order history extrapolation  w^{n+1} ≈ w^n + ω(w^n − w^{n-1}),  ω = Δt/Δt_{n-1}.
    PetscScalar** wnm1;
    DMDAVecGetArray(da, user_context.starting_wtd_prev, &wnm1);
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (mask[j][i] != 0) guarded_write(x, wn, topo, j, i, omega * (wn[j][i] - wnm1[j][i]));
    DMDAVecRestoreArray(da, user_context.starting_wtd_prev, &wnm1);
  } else {
    // First step (no history): forward-Euler bootstrap  w^{n+1} ≈ w^n + Δt·f(w^n).  The wtd step is
    //   Δw = my_rech − E/Sy(w^n),   E = dt·(N(w^n)/A_j + removal(w^n))  (compute_tr_explicit),
    // i.e. Δt·dwtd/dt at w^n. This is the Δt_tiny→0 limit of "run a tiny step, then extrapolate": the
    // trajectory tangent. For a warm transient start (transient runs supply a starting table) f(w^n) is
    // moderate, so the forward-Euler direction is stable; the guard caps any overshoot.
    compute_tr_explicit(user_context);  // fills user_context.tr_expl with E per owned land cell
    PetscScalar **expl, **rech, **poro;
    DMDAVecGetArray(da, user_context.tr_expl, &expl);
    DMDAVecGetArray(da, user_context.rech_vec, &rech);
    DMDAVecGetArray(da, user_context.porosity_vec, &poro);
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (mask[j][i] != 0)
          guarded_write(x, wn, topo, j, i, rech[j][i] - expl[j][i] / specificYield(wn[j][i], poro[j][i]));
    DMDAVecRestoreArray(da, user_context.tr_expl, &expl);
    DMDAVecRestoreArray(da, user_context.rech_vec, &rech);
    DMDAVecRestoreArray(da, user_context.porosity_vec, &poro);
  }

  DMDAVecRestoreArray(da, user_context.x, &x);
  DMDAVecRestoreArray(da, user_context.starting_wtd, &wn);
  DMDAVecRestoreArray(da, user_context.topo_vec, &topo);
  DMDAVecRestoreArray(da, user_context.mask, &mask);
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
  // maxiter loop, then maintained by the copy-back below), not in arp.wtd.
  PetscLogEventBegin(EVENT_SETSTART, 0, 0, 0, 0);
  set_starting_values(
      arp, dmdapack.starting_wtd, dmdapack.rech_dist, dmdapack.mask, dmdapack.porosity_vec, xs, ys, xm, ym);
  PetscLogEventEnd(EVENT_SETSTART, 0, 0, 0, 0);

  // Recharge is a per-step AMOUNT (a depth) = rate*dt, but rech_dist is baked ONCE as
  // rate*params.deltat (irf.cpp / WTM.cpp). The residual adds my_rech directly and scales only the
  // flux by user_context.deltat, so on a VARIABLE-dt path (adaptive / Newton dt-continuation) an
  // unscaled source over-recharges when dt shrinks below params.deltat -- the "source term grows as
  // the step shrinks" instability that broke earlier adaptive stepping. Rescale to rate*(actual dt) so
  // recharge and drainage scale together; the steady state is then dt-independent (rate = drainage at
  // the fixed point, dt cancels). Exactly 1.0 on every fixed-dt path, so those are byte-identical.
  // See benchmark/EQUILIBRIUM_ROBUSTNESS.md.
  const double rech_dt_scale = user_context.deltat / params.deltat;
//  values for storativity are reset each time; and recharge changes from one timestep to the next, so set these here
#pragma omp parallel for default(none) shared(arp, ys, ym, xs, xm, dmdapack, params, rech_dt_scale) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      dmdapack.rech_vec[j][i] = add_recharge(
          dmdapack.rech_dist[j][i] * rech_dt_scale, dmdapack.starting_wtd[j][i], dmdapack.porosity_vec[j][i]);
    }
  }

  // Smoothing widths are physics modeling options and apply on ALL solver paths (Anderson,
  // Newton, Picard), so read them here -- before the solver branch -- rather than gating them
  // behind use_picard. Storativity land-surface transition (sub-grid roughness); default 0.01 m,
  // always on. The two ksat/transmissivity widths default to 0 (=> exact piecewise Fan T); any
  // positive width rounds that boundary in every path that evaluates T (residual and operator).
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_storativity_surface_smoothing_width", &g_storativity_surface_smoothing_width, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_ksat_soilbottom_smoothing_width", &g_ksat_soilbottom_smoothing_width, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_ksat_surface_smoothing_width", &g_ksat_surface_smoothing_width, nullptr);
  PetscBool extsoil = PETSC_FALSE;  // [WIP] -wtm_extended_soil: aquifer continues above surface (smooth GW step)
  PetscOptionsHasName(nullptr, nullptr, "-wtm_extended_soil", &extsoil);
  g_extended_soil = (extsoil == PETSC_TRUE);

  // -wtm_surface_exfiltration_to_runoff: post-solve clamp-to-surface + route the exact excess to FSM (standard clamped
  // T; robust "collect" alternative to the implicit sink). Composes with -wtm_Tbar (unlike extended_soil,
  // it keeps the piecewise Fan T). Use with -wtm_surface_sink 0 (else the sink holds wtd<=0 first).
  PetscBool surfexfil = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_surface_exfiltration_to_runoff", &surfexfil);
  g_surface_exfiltration_to_runoff_array = (surfexfil == PETSC_TRUE);

  // -wtm_kirchhoff: solve the Newton path in the discharge potential Φ = ∫T dwtd (compresses T's dynamic
  // range out of the Jacobian conditioning; see the transform helpers above). The Φ transform is the
  // antiderivative of the PIECEWISE Fan T, so it requires the piecewise T (no ksat smoothing widths) and
  // the standard surface physics (not extended-soil). Only meaningful on the Newton path (use_newton).
  PetscBool kirchhoff = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_kirchhoff", &kirchhoff);
  g_kirchhoff = (kirchhoff == PETSC_TRUE) && user_context.use_newton;
  if (g_kirchhoff && (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0 || g_extended_soil))
    throw std::runtime_error("-wtm_kirchhoff requires the piecewise Fan transmissivity: remove "
                             "-wtm_ksat_*_smoothing_width and -wtm_extended_soil.");
  if (kirchhoff == PETSC_TRUE && !user_context.use_newton)
    throw std::runtime_error("-wtm_kirchhoff is a Newton-path option; also pass -wtm_newton.");

  // -wtm_Tbar: use the step-time-averaged interblock transmissivity T̄ (Kirchhoff-potential difference;
  // see interblockTransmissivity). Composes with any solver. Requires the piecewise Fan T (Φ is its
  // antiderivative), so it is incompatible with ksat smoothing, extended soil, and the Kirchhoff change
  // of variable (which redefines the solve variable). Applies on the Anderson residual, the Picard
  // operator, and the Newton Jacobian.
  PetscBool tbar = PETSC_FALSE;
  PetscOptionsHasName(nullptr, nullptr, "-wtm_Tbar", &tbar);
  g_Tbar = (tbar == PETSC_TRUE);
  if (g_Tbar && (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0 ||
                 g_extended_soil || g_kirchhoff))
    throw std::runtime_error("-wtm_Tbar requires the piecewise Fan transmissivity: remove "
                             "-wtm_ksat_*_smoothing_width, -wtm_extended_soil, and -wtm_kirchhoff.");

  // -wtm_T_bedrock: additive background (bedrock) transmissivity floor [m²/s]; default 0 = v2.0.1 (no
  // floor). A constant added to T everywhere, representing the deep crust's small nonzero conductance
  // integrated over the active flow thickness; it removes the deep-cell operator singularity by capping
  // T's dynamic range (e.g. 1e-8 -> ~3.7 orders vs surface). See the block above depthIntegratedTransmissivity.
  // Incompatible with -wtm_kirchhoff (Φ + T_bedrock·wtd is not analytically invertible for the Φ variable).
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_T_bedrock", &g_T_bedrock, nullptr);
  if (g_T_bedrock < 0.0)
    throw std::runtime_error("-wtm_T_bedrock must be >= 0 (it is an additive transmissivity floor in m^2/s).");
  if (g_T_bedrock > 0.0 && g_kirchhoff)
    throw std::runtime_error("-wtm_T_bedrock is incompatible with -wtm_kirchhoff: Phi + T_bedrock*wtd has no "
                             "closed-form inverse for the discharge-potential variable.");

  // Taper 1 -- sub-surface sink: a smooth, order-preserving near-surface removal that holds the water
  // table at/below the land surface and hands the exfiltrated water to FillSpillMerge (it stays in the
  // domain, unlike taper 2's evaporation). Applied on the Anderson default path (FormFunctionLocal,
  // every solve) and the Picard BDF2-on-V path; it smooths the wtd=0 exfiltration->runoff handoff that
  // otherwise breaks 2nd-order accuracy. Qmax supplied in m/yr (intuitive), stored as m/s.
  PetscBool sink = PETSC_TRUE;  // taper 1 default ON (off-switch: -wtm_surface_sink 0 / false)
  PetscOptionsGetBool(nullptr, nullptr, "-wtm_surface_sink", &sink, nullptr);
  g_surface_sink         = (sink == PETSC_TRUE);
  double sink_qmax_yr    = 1.0;  // default peak removal 1 m/yr (~ precip/evap scale; supplied m/yr, stored m/s)
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_surface_sink_qmax", &sink_qmax_yr, nullptr);
  g_surface_sink_qmax = sink_qmax_yr / SECONDS_IN_A_YEAR;
  // Default sink width SCALES WITH the per-timestep removal depth: width = C * qmax * dt. The implicit
  // near-surface removal is a near-clamp, so its stable width tracks qmax*dt: if width < qmax*dt the
  // solve diverges (DIVERGED_MAX_IT on both paths), and if width is much larger the table is held too
  // far below the surface. C=2 gives stability headroom while keeping the table tight -- mm-cm at the
  // small dt of the 2nd-order transient regime, only necessarily wider at a large equilibrium dt. A
  // -wtm_surface_sink_width overrides. (Adaptive dt: uses the base deltat, conservative -- errs wide =
  // stable.) NOTE (exfiltration): a tight width routes MORE water to FSM as exfiltration; if that
  // over-exfiltrates, revisit C or qmax. See SURFACE_SINK_DESIGN.md sec 11/14.
  constexpr double C_sink = 2.0;
  g_surface_sink_width = C_sink * g_surface_sink_qmax * params.deltat;
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_surface_sink_width", &g_surface_sink_width, nullptr);

  // -wtm_direct_to_runoff: seepage-face removal (supersedes the qmax sink where on). Removes the above-
  // surface excess (max(0,wtd)) to runoff each step, holding the table AT the surface with no rate cap
  // and no below-surface band -> no pile, no depression.
  PetscBool seep = PETSC_FALSE;
  PetscOptionsGetBool(nullptr, nullptr, "-wtm_direct_to_runoff", &seep, nullptr);
  g_direct_to_runoff = (seep == PETSC_TRUE);

  // -wtm_relax: sub-step under-relaxation of the water table (w <- a*w_solve + (1-a)*w_prev). a=1 is off
  // (byte-identical). a<1 damps the period-2 flicker at pinned free boundaries (lakeshore / seepage). At
  // steady state w_solve=w_prev so the fixed point (equilibrium) is unchanged; only the transient is damped.
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_relax", &g_relax, nullptr);

  // -- Fringe-size source: set the per-cell sink band width (the physical capillary fringe), populated
  // into user_context.fringe_width_vec. Default none = today's numerical width (byte-identical). See the
  // FringeSource enum above and benchmark/capillary_taper_math.tex.
  {
    const char *fringe_modes[] = {"none", "fixed", "ksat", "file"};
    PetscInt    fmode          = 0;
    PetscOptionsGetEList(nullptr, nullptr, "-wtm_fringe_source", fringe_modes, 4, &fmode, nullptr);
    if (fmode == 3)
      throw std::runtime_error("-wtm_fringe_source file: not yet implemented (use none|fixed|ksat).");
    g_fringe_source = static_cast<int>(fmode);  // 0 none, 1 fixed, 2 ksat (matches FringeSource)
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_fringe_length", &g_fringe_length, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_fringe_ksat_coef", &g_fringe_ksat_coef, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-wtm_fringe_cap", &g_fringe_cap, nullptr);

    // Populate w = psi_a / KAPPA_SINK per cell (content-match). none -> uniform g_surface_sink_width
    // (byte-identical); fixed -> uniform; ksat -> psi_a = C*sqrt(n/ksat), capped at g_fringe_cap.
    const auto [fxs, fys, fxm, fym] = get_corners(user_context.da);
    PetscScalar **fw, **fks, **fpo;
    DMDAVecGetArray(user_context.da, user_context.fringe_width_vec, &fw);
    DMDAVecGetArray(user_context.da, user_context.ksat_vec, &fks);
    DMDAVecGetArray(user_context.da, user_context.porosity_vec, &fpo);
    for (int j = fys; j < fys + fym; j++)
      for (int i = fxs; i < fxs + fxm; i++) {
        double w;
        if (g_fringe_source == FRINGE_KSAT) {
          const double ks  = std::max(static_cast<double>(fks[j][i]), 1e-30);
          const double psi = std::min(g_fringe_cap, g_fringe_ksat_coef * std::sqrt(fpo[j][i] / ks));
          w                = psi / KAPPA_SINK;
        } else if (g_fringe_source == FRINGE_FIXED) {
          w = g_fringe_length / KAPPA_SINK;
        } else {  // FRINGE_NONE
          w = g_surface_sink_width;  // today's numerical width -> byte-identical default
        }
        fw[j][i] = w;
      }
    DMDAVecRestoreArray(user_context.da, user_context.fringe_width_vec, &fw);
    DMDAVecRestoreArray(user_context.da, user_context.ksat_vec, &fks);
    DMDAVecRestoreArray(user_context.da, user_context.porosity_vec, &fpo);
  }

  // Taper 2 [WIP]: implicit demand-identity evaporation (ET -> owe). Read here AND early in
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
  const bool picard_bdf2_V = user_context.use_bdf2 && user_context.bdf2_have_history && user_context.use_bdf2_on_V;
  const bool sink_active_this_step = (g_surface_sink || g_direct_to_runoff) && (matrix_free || picard_bdf2_V);
  const bool evap_active_this_step = g_evap_taper && (matrix_free || picard_bdf2_V);

  // -wtm_Tbar: ghost-scatter w^n (starting_wtd) so a neighbour's time-averaged T̄ can read its w^n
  // under MPI. w^n is fixed for this step (set above by set_starting_values), so scatter ONCE here
  // rather than per SNES iteration. starting_wtd is checked out read-write by DMDA_Array_Pack, but
  // this scatter only READS it (global → local), which is safe alongside the outstanding pointer.
  if (g_Tbar || user_context.use_tr_bdf2 || user_context.use_predict_guess) {  // ghosted w^n: T̄, TR-BDF2 old
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
    if (user_context.use_predict_guess) apply_predictor_guess(user_context);
    SNESSolve(user_context.snes, nullptr, user_context.x);
  } else {
    // Set local function evaluation routine (always needed).
    DMDASNESSetFunctionLocal(
        user_context.da,
        INSERT_VALUES,
        (PetscErrorCode(*)(DMDALocalInfo*, void*, void*, void*))FormFunctionLocal,
        &user_context);

    // -wtm_aa_picard: register the GAMG-Picard solve as the OUTER Anderson's nonlinear preconditioner.
    // The outer keeps the head-form FormFunctionLocal (above); the NPC solves A(x)x = b(x) (volume form,
    // CG+GAMG) and the outer Anderson mixes the preconditioned iterates.
    if (user_context.use_aa_picard) {
      SNES npc;
      SNESGetNPC(user_context.snes, &npc);
      SNESSetPicard(npc, user_context.picard_r, FormPicardRHS, user_context.picard_A, user_context.picard_A,
                    FormPicardOperator, &user_context);
    }

    // Newton-Krylov path (-wtm_newton): register the analytic Jacobian of FormFunctionLocal. The
    // Jacobian (FormJacobianLocal) is the exact ∂F/∂x of the conservative-FV residual including the
    // sink/evap-taper tangents; verify it against FD with -snes_test_jacobian (see FormJacobianLocal).
    // Anderson (snes_type == SNESANDERSON) is matrix-free and skips this. Any OTHER non-Anderson
    // type reaching here WITHOUT -wtm_newton is refused: it would drive a Newton solve with no
    // registered Jacobian (PETSc would fall back to a full FD Jacobian -- prohibitively slow).
    SNESType snes_type;
    SNESGetType(user_context.snes, &snes_type);
    const bool is_anderson = (std::string(snes_type) == std::string(SNESANDERSON));
    if (!is_anderson) {
      if (!user_context.use_newton) {
        throw std::runtime_error(
            std::string("The Newton-Krylov solver (-snes_type ") + snes_type +
            ") needs -wtm_newton to register its analytic Jacobian. Use the default Anderson solver, "
            "-wtm_picard for the semi-implicit (BDF2-on-V) path, or add -wtm_newton for true Newton.");
      }
      DMDASNESSetJacobianLocal(
          user_context.da,
          (PetscErrorCode(*)(DMDALocalInfo*, void*, Mat, Mat, void*))FormJacobianLocal,
          &user_context);
    }

    // Evaluate initial guess
    FormInitialGuess(&user_context, user_context.da, user_context.x);
    if (user_context.use_predict_guess) apply_predictor_guess(user_context);

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
      if (stage1_reason < 0)
        throw std::runtime_error("TR-BDF2 trapezoidal stage (1) did not converge.");
      VecCopy(user_context.x, user_context.tr_ygamma);  // Y_gamma carried into stage 2
      user_context.tr_stage = 2;  // BDF2 → w^{n+1} (initial guess = Y_gamma, already in x)
      SNESSolve(user_context.snes, user_context.b, user_context.x);
      user_context.tr_stage = 0;
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
    throw std::runtime_error("The SNES solver has not converged.");
  }

  // Adaptive dt (forward, no-reject): estimate the local error as the max deviation of the
  // new head h^{n+1} from a LINEAR extrapolation of the history (h_pred = h^n + w(h^n -
  // h^{n-1}), w = dt_n/dt_{n-1}); this deviation ~ O(dt^2). Set the NEXT step to hold it near
  // dt_tol via dt_new = dt * clamp(safety*sqrt(tol/est), shrink, grow). Runs once history
  // exists, and BEFORE starting_wtd_prev is overwritten below. See BDF2_ADAPTIVE_DESIGN.md.
  if (user_context.use_dt_adaptive && user_context.bdf2_have_history) {
    PetscScalar **swp, **topo_e;
    DMDAVecGetArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecGetArray(user_context.da, user_context.topo_vec, &topo_e);
    const double omega = user_context.deltat / user_context.bdf2_prev_dt;
    double local_max   = 0.0;
    for (int j = ys; j < ys + ym; j++)
      for (int i = xs; i < xs + xm; i++)
        if (dmdapack.mask[j][i] != 0) {  // land only
          const double h_n    = dmdapack.starting_wtd[j][i] + topo_e[j][i];
          const double h_pred = h_n + omega * (dmdapack.starting_wtd[j][i] - swp[j][i]);
          const double dev    = std::abs(dmdapack.x[j][i] - h_pred);
          if (dev > local_max) local_max = dev;
        }
    DMDAVecRestoreArray(user_context.da, user_context.starting_wtd_prev, &swp);
    DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &topo_e);

    double est = 0.0;
    MPI_Allreduce(&local_max, &est, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

    const double safety = 0.9, grow = 1.5, shrink = 0.5;
    double factor = (est > 0.0) ? safety * std::sqrt(user_context.dt_tol / est) : grow;  // est~O(dt^2)
    factor        = std::min(grow, std::max(shrink, factor));
    user_context.deltat *= factor;
  }

  // Exact budget-closing accounting (Picard path): the solver's discrete storage + recharge terms,
  // read while starting_wtd still holds w^n and starting_wtd_prev still holds w^{n-1} (both below
  // overwrite these). Together with total_ocean_outflow and total_surface_removed the budget then
  // closes to the SNES tolerance. See benchmark/WATER_BUDGET.md.
  if (user_context.use_picard) accumulate_budget_terms(user_context, arp, dmdapack);

  // BDF2 / predictor: before starting_wtd is overwritten with h^{n+1} below, save the current h^n
  // wtd as the next step's h^{n-1}. The first step captures h^0 and sets the history flag, so BDF2 /
  // the predictor engage from the second step on (the first bootstraps with backward Euler / w^n guess).
  // (fsm_off / Phase A: history is continuous; Phase B will reset the flag after FSM.)
  if (user_context.use_bdf2 || user_context.use_predict_guess) {
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
  // feeds the next solve in the maxiter loop and is assembled to arp.wtd once
  // per cycle by gather_wtd_to_all. Read topo/mask/porosity from DMDA arrays
  // (topo_vec is re-scattered each cycle in transient) so arp is not needed here.
  PetscScalar** my_topo;
  PetscScalar **my_fdepth_cb = nullptr, **my_ksat_cb = nullptr;  // for the Kirchhoff Φ⁻¹ back-transform
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;
  DMDAVecGetArray(user_context.da, user_context.topo_vec, &my_topo);
  PetscScalar **my_fringe;
  DMDAVecGetArray(user_context.da, user_context.fringe_width_vec, &my_fringe);
  if (g_kirchhoff) {
    DMDAVecGetArray(user_context.da, user_context.fdepth_vec, &my_fdepth_cb);
    DMDAVecGetArray(user_context.da, user_context.ksat_vec, &my_ksat_cb);
  }
  if (evap_active_this_step) {
    DMDAVecGetArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecGetArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecGetArray(user_context.da, user_context.precip_vec, &my_precip);  // taper 3 deficit (E_eff - P)
  }
  double dh_max_local = 0.0;  // max |w^{n+1} - w^n| over owned land cells (for the PTC/SER dt controller)
  for (int j = ys; j < ys + ym; j++) {
    for (int i = xs; i < xs + xm; i++) {
      // Back-transform the SNES variable to wtd: Kirchhoff x=Φ → wtd=Φ⁻¹(x); else head x → wtd=x−topo.
      const double new_wtd = g_kirchhoff
                                 ? dischargePotentialInverse(dmdapack.x[j][i], my_fdepth_cb[j][i], my_ksat_cb[j][i])
                                 : dmdapack.x[j][i] - my_topo[j][i];
      // Under-relaxation (-wtm_relax a<1): damp the step to w <- a*w_solve + (1-a)*w_prev. a=1 -> byte-
      // identical. The metric measures the RELAXED change (the true state move), so it stays honest.
      const double relaxed = (g_relax >= 1.0) ? new_wtd
                                              : g_relax * new_wtd + (1.0 - g_relax) * dmdapack.starting_wtd[j][i];
      if (dmdapack.mask[j][i] != 0)
        dh_max_local = std::max(dh_max_local, std::abs(relaxed - dmdapack.starting_wtd[j][i]));
      dmdapack.starting_wtd[j][i] = relaxed;
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
      // Land cells: the sink and the evaporation taper can both be active; account each in the same
      // sub-step it was removed, evaluated at the just-computed new head. Serial loop -> += race-free.
      if (sink_active_this_step) {
        // Sink removed dt*Q(w^{n+1}) (Q is m/s -> dt*Q is a depth). To FSM (stays in domain).
        const double removed_depth =
            g_direct_to_runoff ? std::max(0.0, static_cast<double>(dmdapack.starting_wtd[j][i]))  // = dt*rate = the excess depth
                      : user_context.deltat * surfaceSink(dmdapack.starting_wtd[j][i], my_fringe[j][i]);
        arp.total_surface_removed += removed_depth * arp.cell_area[j];  // budget-closing (WATER_BUDGET.md)
        dmdapack.sink_removed_dist[j][i] += removed_depth;              // per-cycle FSM input (taper 1)
      }
      if (evap_active_this_step) {
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
      //   -wtm_extended_soil   : EXTENDED-soil physics (no T-clamp -> T GROWS above the surface). Tested
      //       NEGATIVE (2026-08-10): the un-clamped T conducts laterally = a new stiffness (slower,
      //       diverges at large dt). Kept as a documented dead-end; the clamp variant is the one to use.
      // Excess = storedVolume(wtd) - storedVolume(0): the real above-surface storage under the ACTIVE
      // storativity (~wtd surface-water depth in standard physics; porosity*wtd in extended soil).
      if ((g_surface_exfiltration_to_runoff_array || g_extended_soil) && dmdapack.starting_wtd[j][i] > 0.0) {
        const double poro         = dmdapack.porosity_vec[j][i];
        const double excess_depth = storedVolume(dmdapack.starting_wtd[j][i], poro) - storedVolume(0.0, poro);
        arp.total_surface_removed += excess_depth * arp.cell_area[j];  // budget-closing (WATER_BUDGET.md)
        dmdapack.sink_removed_dist[j][i] += excess_depth;              // collect -> gather -> arp.runoff -> FSM
        dmdapack.starting_wtd[j][i] = 0.0;                             // truncate to the real land surface
      }
    }
  }
  DMDAVecRestoreArray(user_context.da, user_context.topo_vec, &my_topo);
  DMDAVecRestoreArray(user_context.da, user_context.fringe_width_vec, &my_fringe);
  if (g_kirchhoff) {
    DMDAVecRestoreArray(user_context.da, user_context.fdepth_vec, &my_fdepth_cb);
    DMDAVecRestoreArray(user_context.da, user_context.ksat_vec, &my_ksat_cb);
  }
  if (evap_active_this_step) {
    DMDAVecRestoreArray(user_context.da, user_context.evap_vec, &my_evap);
    DMDAVecRestoreArray(user_context.da, user_context.open_water_evap_vec, &my_owe);
    DMDAVecRestoreArray(user_context.da, user_context.precip_vec, &my_precip);
  }

  // Global max |Δw| this step: the pseudo-transient/SER dt controller grows Δt as this shrinks toward
  // equilibrium (the discrete steady residual ~ S·Δw/Δt), so the ramp accelerates to Newton near steady
  // state. Reduced here (cheap) so WTM.cpp's continuation loop can read user_context.last_dh_max.
  MPI_Allreduce(&dh_max_local, &user_context.last_dh_max, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

  // Account the water that left through land->ocean faces this solve (Darcy interface flux at the
  // converged head), the term that closes the water budget against the Dirichlet ocean boundary.
  accumulate_ocean_outflow(user_context, arp);

  // The full wtd field is assembled once per cycle, after the maxiter loop, by
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

// Gather the distributed per-cycle runoff (runoff_dist = runoff_ratio*rech) to rank-0
// arp.runoff, so the NEXT FillSpillMerge (rank 0) sees the recharge's runoff. Called only
// when runoff_ratio_on; otherwise the runoff is 0 and arp.runoff stays at FSM's own 0.
// Reuses the un-held wtd_global as the gather scratch (after gather_wtd_to_all has
// finished with it -- the two run sequentially). See DISTRIBUTED_ARP_DESIGN.md (2c).
void gather_runoff_to_zero(Parameters& params, ArrayPack& arp, AppCtx& user_context, DMDA_Array_Pack& dmdapack) {
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
      for (int i = 0; i < params.ncells_x; i++)
        arp.runoff(i, j) = full[j * params.ncells_x + i];
}

// Whether the implicit sub-surface sink is configured this run (taper 1). Lets the cycle loop
// decide whether to gather the sink accumulator into arp.runoff for FSM without reaching into the
// file-static flag. Set in update() from -wtm_surface_sink, so valid by the post-solve gather.
bool surface_sink_on() { return g_surface_sink; }
bool direct_to_runoff_on() { return g_direct_to_runoff; }

// Whether extended-soil surface truncation routes above-surface water to FSM (via the same sink
// accumulator). Lets the cycle loop gather the accumulator for FSM when extended soil is on, just as
// for the sink. Set in update() from -wtm_extended_soil.
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

// Read the taper-2 options (-wtm_evap_taper, wtd_c, s) into the file-static flags and enforce the
// evap_mode-1 requirement. Called BOTH early in WTM.cpp::initialise() (so irf.cpp's initial recharge
// sees the flag) AND in update() (so a standalone solve still parses it). Idempotent -- it just
// re-reads the same PETSc options -- so the double call is harmless.
void read_evap_taper_options(const Parameters& params) {
  PetscBool evap_taper = PETSC_TRUE;  // taper 2 default ON (off-switch: -wtm_evap_taper 0 / false)
  PetscOptionsGetBool(nullptr, nullptr, "-wtm_evap_taper", &evap_taper, nullptr);
  g_evap_taper = (evap_taper == PETSC_TRUE);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_evap_taper_wtdc", &g_evap_taper_wtdc, nullptr);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_evap_taper_s", &g_evap_taper_s, nullptr);

  // Taper 3: accessibility / extinction-depth clamp (awickert/WTM#4). Own on/off toggle plus the depth.
  PetscBool extinction = PETSC_TRUE;  // taper 3 default ON (off-switch: -wtm_extinction 0 / false)
  PetscOptionsGetBool(nullptr, nullptr, "-wtm_extinction", &extinction, nullptr);
  g_extinction = (extinction == PETSC_TRUE);
  PetscOptionsGetReal(nullptr, nullptr, "-wtm_extinction_depth", &g_extinction_depth, nullptr);

  // The taper works in BOTH evap_modes: evap_mode 0 also supplies open_water_evap (used for surface
  // recharge), so E_eff has the owe it needs, and the recharge paths check the taper first so it
  // governs evaporation mode-independently (the smooth removal auto-zeroes standing water in place of
  // mode 0's hard wtd=0). Configuration mismatches are surfaced as warnings, not errors -- see the
  // warn_taper_configuration() checks. (params retained for that call site.)
  (void)params;
}

// Emit configuration warnings for the surface-water evaporation model. The intended (blessed)
// configuration is the smooth transition with BOTH taper 2 (-wtm_evap_taper) and taper 3
// (-wtm_extinction) on; every other combination is arid-unsafe, inert, or the legacy hard-switch
// model, and is flagged here. Caller guards rank 0 so this prints once. See SURFACE_SINK_DESIGN.md 14.
void warn_taper_configuration(const Parameters& params) {
  if (params.evap_mode) {
    // evap_mode 1: the smooth ET->open-water transition is the intended model.
    if (g_evap_taper && !g_extinction)
      std::cerr << "WARNING: -wtm_evap_taper without -wtm_extinction: in arid cells (ET > precip) the "
                   "evaporation taper draws the water table down WITHOUT BOUND (no equilibrium). Add "
                   "-wtm_extinction (accessibility / extinction-depth clamp) unless you specifically want "
                   "taper 2 alone for testing."
                << std::endl;
    else if (!g_evap_taper && g_extinction)
      std::cerr << "WARNING: -wtm_extinction without -wtm_evap_taper has NO EFFECT: the extinction-depth "
                   "clamp gates taper 2's evaporative deficit, which is not active."
                << std::endl;
    else if (!g_evap_taper && !g_extinction)
      std::cerr << "WARNING: running the LEGACY hard-switch evaporation model (neither -wtm_evap_taper nor "
                   "-wtm_extinction). The hard wtd=0 ET<->open-water switch makes FillSpillMerge lake "
                   "formation rank-dependent (NON-DETERMINISTIC across MPI rank counts) and applies no "
                   "phreatic ET. The smooth tapers (-wtm_evap_taper -wtm_extinction) are recommended."
                << std::endl;
  } else {
    // evap_mode 0: remove all surface water.
    if (g_evap_taper) {
      std::cerr << "WARNING: evap_mode 0 (remove all surface water) with the taper on: the smooth taper "
                   "governs evaporation, so surface water is evaporated smoothly (not hard-removed) and "
                   "evap_mode 0 and 1 coincide."
                << (g_extinction ? "" : " Also, without -wtm_extinction, arid drawdown is unbounded.")
                << std::endl;
    } else {
      std::cerr << "WARNING: evap_mode 0 removes ALL surface water every step (GW-alone testing; Fan "
                   "Reinfelder et al. 2013)."
                << std::endl;
      std::cerr << "WARNING: running the LEGACY hard-switch evaporation model. The hard wtd=0 switch is "
                   "rank-dependent (non-deterministic FSM lakes) and applies no phreatic ET; the smooth "
                   "tapers (-wtm_evap_taper -wtm_extinction) are recommended."
                << std::endl;
    }
  }
}

// Gather this cycle's distributed sink removal (sink_removed_dist, depth m) to rank-0 arp.runoff,
// ADDING to it, so this cycle's FillSpillMerge routes the exfiltrated water the implicit sink pulled
// out of the solve (taper 1). Because the sink holds wtd<=0, FSM's own wtd>0->runoff handoff never
// fires -- this is its smooth, order-preserving replacement. It composes with the runoff_ratio
// channel (gather_runoff_to_zero OVERWRITES arp.runoff a cycle earlier, for the *next* FSM; this ADDS
// after that FSM has consumed the prior value). Reuses wtd_global as scratch, after gather_wtd_to_all
// has finished with it (the two run sequentially). See SURFACE_SINK_DESIGN.md sec 14 (taper 1).
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

  // Kirchhoff: the SNES variable is the discharge potential Φ, so seed x = Φ(starting_wtd); else x is the
  // head starting_wtd+topo. Ocean cells (starting_wtd = topo = 0) seed Φ(0) / 0 respectively.
#pragma omp parallel for default(none) \
    shared(my_starting_wtd, my_topo, my_fdepth, my_ksat, ys, ym, xs, xm, x, g_kirchhoff) collapse(2)
  for (auto j = ys; j < ys + ym; j++) {
    for (auto i = xs; i < xs + xm; i++) {
      x[j][i] = g_kirchhoff ? dischargePotential(my_starting_wtd[j][i], my_fdepth[j][i], my_ksat[j][i])
                            : my_starting_wtd[j][i] + my_topo[j][i];
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
  const bool bdf2v = user_context->use_bdf2_on_V && user_context->bdf2_have_history && !user_context->use_picard;
  const bool zero_rhs = bdf2v || user_context->use_tr_bdf2;  // TR-BDF2 stages are also self-contained (b=0)
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
  PetscScalar **my_fringe;
  PetscCall(DMDAVecGetArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  // Matrix-free 2nd-order-in-time (-wtm_anderson -wtm_bdf2_on_V): once a history exists, the storage
  // term is the 3-level BDF2 difference of the stored VOLUME (genuine 2nd order), head-scaled by the
  // specific yield so the residual stays O(metres) for Anderson. Same fixed point as the Picard
  // BDF2-on-V operator (verified). The bootstrap step (no history) uses backward Euler. See
  // benchmark/TBAR_TIME_AVERAGING.md / BDF2_ADAPTIVE_DESIGN.md.
  const bool bdf2v = user_context->use_bdf2_on_V && user_context->bdf2_have_history && !user_context->use_picard;
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
  const double TR_G  = 2.0 - std::sqrt(2.0);
  const double tr_c1 = 1.0 / (TR_G * (2.0 - TR_G));
  const double tr_c2 = (1.0 - TR_G) * (1.0 - TR_G) / (TR_G * (2.0 - TR_G));
  const double tr_c3 = (1.0 - TR_G) / (2.0 - TR_G);
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
    shared(info, my_T, x, my_topo, my_fdepth, my_ksat, smooth_T, g_kirchhoff, g_Tbar, my_starting_wtd_local)   \
    collapse(2)
  for (auto j = info->gys; j < info->gys + info->gym; j++) {
    for (auto i = info->gxs; i < info->gxs + info->gxm; i++) {
      // Kirchhoff: the SNES variable x is the discharge potential Φ, so wtd = Φ⁻¹(x); else x is the head.
      const double wtd_T = g_kirchhoff ? dischargePotentialInverse(x[j][i], my_fdepth[j][i], my_ksat[j][i])
                                       : x[j][i] - my_topo[j][i];
      const double wtd_old = g_Tbar ? my_starting_wtd_local[j][i] : 0.0;  // w^n (ghosted); unused off -wtm_Tbar
      my_T[j][i] = 1. / interblockTransmissivity(wtd_T, wtd_old, my_fdepth[j][i], my_ksat[j][i], smooth_T);
    }
  }

  const bool sink_on  = g_surface_sink;  // hoisted for the omp default(none) clause below
  const bool dtr_on  = g_direct_to_runoff;
  const bool taper_on = g_evap_taper;
#pragma omp parallel for default(none)                                                                                \
    shared(info, gew, gn, gs, x, my_T, my_mask, my_rech, user_context, my_porosity, my_starting_wtd, my_topo, f,      \
           my_evap, my_owe, my_precip, my_fdepth, my_ksat, sink_on, dtr_on, taper_on, g_kirchhoff, my_fringe, \
           bdf2v, a_c, b_c, c_c, my_starting_wtd_prev,                                                        \
           tr_stage, TR_G, tr_c1, tr_c2, tr_c3, my_tr_ygamma, my_tr_expl) collapse(2)
  for (auto j = info->ys; j < info->ys + info->ym; j++) {
    for (auto i = info->xs; i < info->xs + info->xm; i++) {
      // Head from the SNES variable: Kirchhoff x=Φ → h = Φ⁻¹(x)+topo; else x IS the head. Used for the
      // flux (h_c - h_nbr) and the centre wtd (h_c - topo). Cheap per-cell pointwise map.
      const auto head = [&](int jj, int ii) {
        return g_kirchhoff ? dischargePotentialInverse(x[jj][ii], my_fdepth[jj][ii], my_ksat[jj][ii]) + my_topo[jj][ii]
                           : x[jj][ii];
      };
      if (my_mask[j][i] == 0) {
        // Dirichlet condition: ocean head h = 0. In head variables f = x forces x=0 (h=0). In Kirchhoff
        // variables f = x - Φ(wtd=0) forces Φ = Φ(0) i.e. wtd=0; both give a unit Jacobian diagonal.
        f[j][i] = g_kirchhoff ? (x[j][i] - dischargePotential(0.0, my_fdepth[j][i], my_ksat[j][i])) : x[j][i];
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
        const double this_T = my_T[j][i];
        const double e_E    = 2. / (this_T + my_T[j][i + 1]);  // harmonic-mean interface transmissivities
        const double e_W    = 2. / (this_T + my_T[j][i - 1]);
        const double e_N    = 2. / (this_T + my_T[j + 1][i]);
        const double e_S    = 2. / (this_T + my_T[j - 1][i]);

        // Net outflow volume-rate = sum of face conductances * (h_c - h_nbr).
        const double net_outflow = e_E * gew[j][i] * (this_x - head(j, i + 1)) + e_W * gew[j][i] * (this_x - head(j, i - 1))
                                 + e_N * gn[j][i] * (this_x - head(j + 1, i)) + e_S * gs[j][i] * (this_x - head(j - 1, i));

        const double A_j = user_context->cellsize_NS_squared / gew[j][i];  // cell area
        const double w_c = this_x - my_topo[j][i];                         // centre-cell wtd
        const double S   = updateEffectiveStorativity(my_starting_wtd[j][i], w_c, my_porosity[j][i]);

        // Sub-surface sink (taper 1) and demand-identity evaporation (taper 2), as head-form removals
        // dt*Q/S: evaluated at the current iterate, so implicit at the Anderson root. Matrix-free ->
        // no tangent needed (unlike the Picard operator). Off unless their flags are set.
        double removal = 0.0;  // m/s
        if (dtr_on)      removal += directToRunoffRemoval(w_c, user_context->deltat);
        else if (sink_on) removal += surfaceSink(w_c, my_fringe[j][i]);
        if (taper_on) removal += evapRemoval(w_c, my_evap[j][i], my_owe[j][i], my_precip[j][i] / SECONDS_IN_A_YEAR);

        if (tr_stage == 1) {
          // TR-BDF2 stage 1 (trapezoidal to t+gamma*dt), head-scaled by Sy(Y_gamma). w_c is the iterate =
          // Y_gamma: implicit half at Y_gamma + explicit half at w^n (my_tr_expl, precomputed in update()).
          // Recharge over gamma*dt = gamma*my_rech (constant source, exact).
          const double poro = my_porosity[j][i];
          const double Sy   = specificYield(w_c, poro);
          const double storage = (storedVolume(w_c, poro) - storedVolume(my_starting_wtd[j][i], poro)) / Sy;
          const double impl    = user_context->deltat * net_outflow / A_j + user_context->deltat * removal;
          f[j][i] = storage + 0.5 * TR_G * (impl + my_tr_expl[j][i]) / Sy - TR_G * my_rech[j][i];
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
                    - tr_c3 * my_rech[j][i];
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
          f[j][i] = storage - my_rech[j][i] + user_context->deltat * net_outflow / (A_j * Sy)
                    + user_context->deltat * removal / Sy;
        } else {
          // Backward Euler (secant storativity): the SNES RHS b=h^n supplies the previous-step storage.
          f[j][i] = (this_x - my_rech[j][i]) + user_context->deltat * net_outflow / (A_j * S)
                    + user_context->deltat * removal / S;
        }
        // my_rech is converted to appropriate recharge for this timestep and starting water
        // table outside of the solve.
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
  PetscCall(DMDAVecRestoreArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
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
   conservative-FV head-form residual). Registered on the opt-in Newton-Krylov path (-wtm_newton;
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

  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecGetArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecGetArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecGetArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecGetArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecGetArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecGetArray(da, user_context->porosity_vec, &my_porosity));
  PetscScalar **my_fringe;
  PetscCall(DMDAVecGetArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecGetArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3 inputs (m/yr)
  if (g_evap_taper) {
    PetscCall(DMDAVecGetArray(da, user_context->evap_vec, &my_evap));
    PetscCall(DMDAVecGetArray(da, user_context->open_water_evap_vec, &my_owe));
    PetscCall(DMDAVecGetArray(da, user_context->precip_vec, &my_precip));
  }
  PetscScalar** my_starting_wtd_local = nullptr;  // -wtm_Tbar: ghosted w^n for the time-averaged T̄
  if (g_Tbar) PetscCall(DMDAVecGetArray(da, user_context->starting_wtd_local, &my_starting_wtd_local));

  const bool   smooth_T = (g_ksat_soilbottom_smoothing_width > 0.0 || g_ksat_surface_smoothing_width > 0.0);
  const bool   sink_on  = g_surface_sink;
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

      // wtd at centre and 4 neighbours from the SNES variable (Kirchhoff x=Φ → wtd=Φ⁻¹(x); else x−topo),
      // matching the residual.
      const auto wtd_of = [&](int jj, int ii) {
        return g_kirchhoff ? dischargePotentialInverse(x[jj][ii], my_fdepth[jj][ii], my_ksat[jj][ii])
                           : x[jj][ii] - my_topo[jj][ii];
      };
      const double w_c = wtd_of(j, i);
      const double w_E = wtd_of(j, i + 1);
      const double w_W = wtd_of(j, i - 1);
      const double w_N = wtd_of(j + 1, i);
      const double w_S = wtd_of(j - 1, i);

      // w^n (ghosted) at centre and neighbours -- the "before" state for the -wtm_Tbar time-average;
      // ignored (0) off the -wtm_Tbar path (Tinv/tauPrime do not read it there).
      const auto wold_of = [&](int jj, int ii) { return g_Tbar ? my_starting_wtd_local[jj][ii] : 0.0; };
      const double wo_c = wold_of(j, i);
      const double wo_E = wold_of(j, i + 1);
      const double wo_W = wold_of(j, i - 1);
      const double wo_N = wold_of(j + 1, i);
      const double wo_S = wold_of(j - 1, i);

      // τ = 1/T and its wtd-derivative τ' at centre and neighbours (T̄-aware; see the lambdas above)
      const double tau_c  = Tinv(w_c, wo_c, my_fdepth[j][i],     my_ksat[j][i]);
      const double tau_E  = Tinv(w_E, wo_E, my_fdepth[j][i + 1], my_ksat[j][i + 1]);
      const double tau_W  = Tinv(w_W, wo_W, my_fdepth[j][i - 1], my_ksat[j][i - 1]);
      const double tau_N  = Tinv(w_N, wo_N, my_fdepth[j + 1][i], my_ksat[j + 1][i]);
      const double tau_S  = Tinv(w_S, wo_S, my_fdepth[j - 1][i], my_ksat[j - 1][i]);
      const double taup_c = tauPrime(w_c, wo_c, my_fdepth[j][i],     my_ksat[j][i]);
      const double taup_E = tauPrime(w_E, wo_E, my_fdepth[j][i + 1], my_ksat[j][i + 1]);
      const double taup_W = tauPrime(w_W, wo_W, my_fdepth[j][i - 1], my_ksat[j][i - 1]);
      const double taup_N = tauPrime(w_N, wo_N, my_fdepth[j + 1][i], my_ksat[j + 1][i]);
      const double taup_S = tauPrime(w_S, wo_S, my_fdepth[j - 1][i], my_ksat[j - 1][i]);

      // Harmonic-mean face conductances e_X = 2/(τ_c+τ_X) and their geometry factors G_X.
      const double sumE = tau_c + tau_E, e_E = 2.0 / sumE, G_E = gew[j][i];
      const double sumW = tau_c + tau_W, e_W = 2.0 / sumW, G_W = gew[j][i];
      const double sumN = tau_c + tau_N, e_N = 2.0 / sumN, G_N = gn[j][i];
      const double sumS = tau_c + tau_S, e_S = 2.0 / sumS, G_S = gs[j][i];

      // Head differences h_c − h_nbr (outflow-positive), h = wtd + topo (matches the residual's flux)
      const double h_c = w_c + my_topo[j][i];
      const double dE = h_c - (w_E + my_topo[j][i + 1]);
      const double dW = h_c - (w_W + my_topo[j][i - 1]);
      const double dN = h_c - (w_N + my_topo[j + 1][i]);
      const double dS = h_c - (w_S + my_topo[j - 1][i]);

      const double net_outflow = e_E * G_E * dE + e_W * G_W * dW + e_N * G_N * dN + e_S * G_S * dS;

      const double A_j = cns2 / gew[j][i];  // cell area (matches the residual)
      const double S   = updateEffectiveStorativity(my_starting_wtd[j][i], w_c, my_porosity[j][i]);
      const double Sp  = dEffectiveStorativityDnew(my_starting_wtd[j][i], w_c, my_porosity[j][i]);
      const double B   = dt / (A_j * S);  // flux prefactor
      const double D   = dt / S;          // removal prefactor

      double removal = 0.0, rho = 0.0;  // removal [m/s] and its exact (unclamped) wtd-derivative
      if (sink_on) {
        removal += surfaceSink(w_c, my_fringe[j][i]);
        rho     += surfaceSinkTangent(w_c, my_fringe[j][i]);
      }
      if (taper_on) {
        const double p_rate = my_precip[j][i] / SECONDS_IN_A_YEAR;
        removal += evapRemoval(w_c, my_evap[j][i], my_owe[j][i], p_rate);
        rho     += evapRemovalTangentRaw(w_c, my_evap[j][i], my_owe[j][i], p_rate);
      }

      const double flux_term    = B * net_outflow;  // dt·N/(A_j·S)   (part of the residual)
      const double removal_term = D * removal;       // dt·removal/S   (part of the residual)

      // Off-diagonals: ∂f/∂x_X = B·G_X·[ −2·τ'_X/sum_X²·dX − e_X ]
      const double J_east  = B * G_E * (-2.0 * taup_E / (sumE * sumE) * dE - e_E);
      const double J_west  = B * G_W * (-2.0 * taup_W / (sumW * sumW) * dW - e_W);
      const double J_north = B * G_N * (-2.0 * taup_N / (sumN * sumN) * dN - e_N);
      const double J_south = B * G_S * (-2.0 * taup_S / (sumS * sumS) * dS - e_S);

      // ∂N/∂x_c: each face contributes G_X·(e_X − 2·τ'_c/sum_X²·dX)
      const double dN_dc = G_E * (e_E - 2.0 * taup_c / (sumE * sumE) * dE)
                         + G_W * (e_W - 2.0 * taup_c / (sumW * sumW) * dW)
                         + G_N * (e_N - 2.0 * taup_c / (sumN * sumN) * dN)
                         + G_S * (e_S - 2.0 * taup_c / (sumS * sumS) * dS);

      const double J_center = 1.0 + B * dN_dc - (Sp / S) * (flux_term + removal_term) + D * rho;

      MatStencil  cols[5];
      PetscScalar vals[5];
      cols[0].j = j;     cols[0].i = i + 1; cols[0].c = 0;
      cols[1].j = j;     cols[1].i = i - 1; cols[1].c = 0;
      cols[2].j = j + 1; cols[2].i = i;     cols[2].c = 0;
      cols[3].j = j - 1; cols[3].i = i;     cols[3].c = 0;
      cols[4].j = j;     cols[4].i = i;     cols[4].c = 0;
      vals[0] = J_east; vals[1] = J_west; vals[2] = J_north; vals[3] = J_south; vals[4] = J_center;
      // Kirchhoff chain rule: dF/dΦ = (dF/dh)·(dh/dΦ) = (dF/dh)/T. The entries above are dF/dh (h form);
      // column-scale each by dwtd_k/dΦ_k = 1/T_k = τ_k to get dF/dΦ (divides the transmissivity range out
      // of the conditioning). τ_k = 1/T at that column's cell.
      if (g_kirchhoff) {
        vals[0] *= tau_E; vals[1] *= tau_W; vals[2] *= tau_N; vals[3] *= tau_S; vals[4] *= tau_c;
      }
      MatSetValuesStencil(Jmat, 1, &row, 5, cols, vals, INSERT_VALUES);
      if (P != Jmat) MatSetValuesStencil(P, 1, &row, 5, cols, vals, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(Jmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Jmat, MAT_FINAL_ASSEMBLY);
  if (P != Jmat) {
    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
  }

  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_n_vec, &gn));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_s_vec, &gs));
  PetscCall(DMDAVecRestoreArray(da, user_context->fdepth_local, &my_fdepth));
  PetscCall(DMDAVecRestoreArray(da, user_context->ksat_local, &my_ksat));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_local, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
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
  PetscScalar **my_evap = nullptr, **my_owe = nullptr, **my_precip = nullptr;  // taper 2/3: ET, owe, precip (m/yr)
  const double  cns2 = user_context->cellsize_NS_squared;  // cell area A_j = cns2 / geom_ew (volume form)

  // Variable-step BDF2 (once an h^{n-1} exists): b = S_c*(b_c*h^n - c_c*h^{n-1} + rech) with
  // omega = dt_n/dt_{n-1}, b_c = 1+omega, c_c = omega^2/(1+omega) (b_c=2, c_c=1/2 when the
  // step is constant -> uniform BDF2). Backward Euler otherwise: b = S_c*(h^n + rech).
  // h^{n-1} = starting_wtd_prev + topo (centre only).
  const bool bdf2      = user_context->use_bdf2 && user_context->bdf2_have_history;
  const bool bdf2_on_V = bdf2 && user_context->use_bdf2_on_V;
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
  PetscScalar **my_fringe;
  PetscCall(DMDAVecGetArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecGetArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecGetArray(da, user_context->geom_ew_vec, &gew));  // for the cell area A_j
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
      } else if (bdf2_on_V) {
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
                        + Sy * my_rech[j][i]);
        if (g_surface_sink) {
          // Implicit sub-surface removal dt*Q(w^{n+1}), Picard-linearized about w_k like the storage
          // term; scaled by A_j to match the operator's dt*Q'(w_k)*A_j diagonal (volume form).
          const double dt = user_context->deltat;
          bb[j][i] += A_j * (dt * surfaceSinkTangent(w_k, my_fringe[j][i]) * xx[j][i] - dt * surfaceSink(w_k, my_fringe[j][i]));
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
          bb[j][i] = A_j * S_c * (b_c * h_n - c_c * h_nm1 + my_rech[j][i]);
        } else {
          bb[j][i] = A_j * S_c * (h_n + my_rech[j][i]);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, b, &bb));
  PetscCall(DMDAVecRestoreArray(da, x, &xx));
  PetscCall(DMDAVecRestoreArray(da, user_context->starting_wtd, &my_starting_wtd));
  PetscCall(DMDAVecRestoreArray(da, user_context->topo_vec, &my_topo));
  PetscCall(DMDAVecRestoreArray(da, user_context->rech_vec, &my_rech));
  PetscCall(DMDAVecRestoreArray(da, user_context->porosity_vec, &my_porosity));
  PetscCall(DMDAVecRestoreArray(da, user_context->fringe_width_vec, &my_fringe));
  PetscCall(DMDAVecRestoreArray(da, user_context->mask, &my_mask));
  PetscCall(DMDAVecRestoreArray(da, user_context->geom_ew_vec, &gew));
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
  PetscScalar **my_fringe;
  PetscCall(DMDAVecGetArray(da, user_context->fringe_width_vec, &my_fringe));
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
  const bool bdf2_on_V = bdf2 && user_context->use_bdf2_on_V;

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
            bdf2_on_V ? specificYield(w_k, my_porosity[j][i])
                      : updateEffectiveStorativity(my_starting_wtd[j][i], w_k, my_porosity[j][i]);

        // Harmonic-mean interface transmissivities: 2 / (1/T_c + 1/T_nbr).
        const double e_E = 2.0 / (my_T[j][i] + my_T[j][i + 1]);
        const double e_W = 2.0 / (my_T[j][i] + my_T[j][i - 1]);
        const double e_N = 2.0 / (my_T[j][i] + my_T[j + 1][i]);
        const double e_S = 2.0 / (my_T[j][i] + my_T[j - 1][i]);

        // Face conductances G = e * (L_wall/d_centre): E-W uses geom_ew (per row); N/S use the
        // FACE-centred geom_n/geom_s, so G_N(j) = G_S(j+1) exactly (shared face) -> conservative.
        const double A_east   = -dt * e_E * gew[j][i];
        const double A_west   = -dt * e_W * gew[j][i];
        const double A_north  = -dt * e_N * gn[j][i];
        const double A_south  = -dt * e_S * gs[j][i];
        // Storage and sub-surface sink now scale with the cell area A_j (volume form). The sink
        // tangent dt*Q'(w_k)*A_j is >= 0, so the diagonal stays dominant -> SPD-preserving.
        const double sink_diag = (g_surface_sink && bdf2_on_V) ? dt * surfaceSinkTangent(w_k, my_fringe[j][i]) * A_j : 0.0;
        // Taper 2 (+ taper 3) evaporation diagonal: dt*R'(w_k)*A_j, SPD-clamped >= 0 (matches the RHS
        // term). R' == E_eff' when taper 3 is off.
        const double evap_diag = (g_evap_taper && bdf2_on_V)
                                     ? dt * evapRemovalTangent(w_k, my_evap[j][i], my_owe[j][i],
                                                               my_precip[j][i] / SECONDS_IN_A_YEAR) * A_j
                                     : 0.0;
        const double A_center =
            a_coeff * S_c * A_j + sink_diag + evap_diag - (A_east + A_west + A_north + A_south);

        // 5-point stencil: east, west, north, south, centre.
        const MatStencil cols[5] = {
            {.k = 0, .j = j,     .i = i + 1, .c = 0},  // east
            {.k = 0, .j = j,     .i = i - 1, .c = 0},  // west
            {.k = 0, .j = j + 1, .i = i,     .c = 0},  // north
            {.k = 0, .j = j - 1, .i = i,     .c = 0},  // south
            {.k = 0, .j = j,     .i = i,     .c = 0},  // centre
        };
        const PetscScalar vals[5] = {
            A_east,
            A_west,
            A_north,
            A_south,
            A_center,
        };
        PetscCall(MatSetValuesStencil(A, 1, &row, 5, cols, vals, INSERT_VALUES));
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
  PetscCall(DMDAVecRestoreArray(da, user_context->fringe_width_vec, &my_fringe));
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
