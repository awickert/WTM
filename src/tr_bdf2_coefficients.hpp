#pragma once
///////////////////////////////////////////////////////////////////////////////////////////////////
// TR-BDF2 stage coefficients AND the step-quadrature weights the water budget needs.
//
// WHY THIS HEADER EXISTS. The stage coefficients used to be bare local constants, redeclared in each
// place that needed them (the residual, the embedded error estimate). That duplication is what let a
// STAGE weight be used where a STEP weight belonged: the active-set exfiltration multiplier was read
// off the stage-2 residual alone and handed to FillSpillMerge as if it were the step's exfiltrated
// water. Measured on tests/multilake (active_set, dt = 0.25 yr): 5.97e11 m^3 delivered against
// 2.00e12 m^3 under backward Euler, a factor of 3.36 low against the predicted 1/C3 = 3.414, and a
// physical budget residual of 9.5% of recharge (BDF2-on-V, also multi-level and also second order,
// closes at 0.2%, so this is the STAGING and not a multi-level startup gap). Naming the two kinds of
// weight separately is the structural fix; the tests in src/test_tr_bdf2_balance.cpp pin the
// identities that make them work.
//
// THE SCHEME. One step from t_n to t_n + dt, in stored-volume (depth) units per cell, with
// gamma = 2 - sqrt(2):
//
//   stage 1  (trapezoidal, t_n -> t_n + gamma*dt), solved for Y_gamma:
//     V(Y_gamma) - V(w^n) + 0.5*gamma*( F(Y_gamma) + F(w^n) ) - gamma*R - E1 = 0
//
//   stage 2  (BDF2 through w^n and Y_gamma, -> t_n + dt), solved for w^{n+1}:
//     V(w^{n+1}) - C1*V(Y_gamma) + C2*V(w^n) + C3*F(w^{n+1}) - C3*R - E2 = 0
//
// where F(w) = dt*( net outflow(w)/A + removal(w) ) is the step-scaled flux-plus-removal, R is the
// step's recharge (a fixed volume), and E1, E2 >= 0 are the active-set exfiltration multipliers each
// stage discards into its pin (max(0, -f*Sy); zero when the constraint is inactive).
//
// THE STEP BALANCE. Take C1 * (stage 1) + (stage 2). The storage terms telescope, because
// C1 - C2 == 1 exactly:
//
//     C1*[V(Y_gamma) - V(w^n)] + V(w^{n+1}) - C1*V(Y_gamma) + C2*V(w^n)
//   = V(w^{n+1}) - (C1 - C2)*V(w^n)  =  V(w^{n+1}) - V(w^n)
//
// and the recharge collapses to the step total, because C1*gamma + C3 == 1. What is left is
//
//   V(w^{n+1}) - V(w^n)  =  R  -  [ W_OLD*F(w^n) + W_YGAMMA*F(Y_gamma) + W_NEW*F(w^{n+1}) ]
//                              -  [ C1*E1 + E2 ]
//
// with W_OLD = W_YGAMMA = C1*gamma/2 and W_NEW = C3. So TR-BDF2's per-step balance is the backward-
// Euler balance with two substitutions, and NOTHING else:
//
//   (1) every FLUX and REMOVAL term becomes a three-point quadrature over the step, at the states
//       (w^n, Y_gamma, w^{n+1}) with weights (C1*gamma/2, C1*gamma/2, C3) that sum to 1;
//   (2) the exfiltration multiplier becomes E = C1*E1 + E2.
//
// The storage difference and the recharge are UNCHANGED from backward Euler -- they telescope
// exactly. That is why the storage and recharge halves of the budget were always right under TR-BDF2
// and only the flux/removal/exfiltration halves were wrong.
//
// WHY THE WEIGHTS ARE WHAT THEY ARE. (W_OLD, W_YGAMMA, W_NEW) is a quadrature rule on the nodes
// (0, gamma, 1) of the unit step. Summing to 1 makes it consistent (exact for a constant integrand);
// W_YGAMMA*gamma + W_NEW == 1/2 exactly makes it exact for a LINEAR integrand, which is the second-
// order condition and the reason TR-BDF2's budget is second-order accurate rather than merely
// conservative. It is NOT exact on a quadratic (it gives 0.414214 against 1/3), which is the
// expected order barrier. All three statements are pinned as tests.
//
// SHARE OF THE STEP. C1*gamma = 0.7071 of the step is carried by stage 1 and C3 = 0.2929 by stage 2.
// Using stage 2 alone therefore recovers 29.29% of the step, i.e. it understates by 1/C3 = 3.4142 --
// the factor measured above.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>

namespace trbdf2 {

// gamma = 2 - sqrt(2): the standard TR-BDF2 stage point, chosen so the two stages share a Jacobian
// and the method is L-stable.
inline const double GAMMA = 2.0 - std::sqrt(2.0);

// Stage-2 (BDF2) coefficients on the nodes (0, gamma, 1):
//   V(w^{n+1}) - C1*V(Y_gamma) + C2*V(w^n) = C3*dt*V'(t_n + dt) + O(dt^3)
// exact for V quadratic in t (pinned in test_tr_bdf2_balance.cpp).
inline const double C1 = 1.0 / (GAMMA * (2.0 - GAMMA));
inline const double C2 = (1.0 - GAMMA) * (1.0 - GAMMA) / (GAMMA * (2.0 - GAMMA));
inline const double C3 = (1.0 - GAMMA) / (2.0 - GAMMA);

// STEP weights for a per-stage FLUX or REMOVAL quantity evaluated at the three step states. Use these
// -- never a bare stage coefficient -- for anything that must total over the whole step: the ocean
// outflow, the taper removals, and any diagnostic that has to close against the storage change.
inline const double W_OLD    = 0.5 * C1 * GAMMA;  // at w^n
inline const double W_YGAMMA = 0.5 * C1 * GAMMA;  // at Y_gamma
inline const double W_NEW    = C3;                // at w^{n+1}

// STEP weights for the active-set exfiltration multiplier discarded by each stage's pin. Different
// from the flux weights above because E1 is already a gamma*dt-scaled quantity (it comes out of the
// stage-1 equation, which is itself scaled by gamma), whereas F is always dt-scaled.
inline const double WE_STAGE1 = C1;
inline const double WE_STAGE2 = 1.0;

}  // namespace trbdf2
