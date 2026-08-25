// Unit tests for the TR-BDF2 per-STEP water balance (src/tr_bdf2_coefficients.hpp).
//
// WHAT THESE PIN. TR-BDF2 takes two implicit stages per step, and each stage satisfies its own
// discrete balance. Neither stage balance is the step's. The header derives how the two combine --
// C1*(stage 1) + (stage 2) -- and these tests pin every identity that derivation leans on, so the
// combination cannot silently stop telescoping:
//
//   C1 - C2 == 1              the storage difference collapses to V(w^{n+1}) - V(w^n)
//   C1*gamma + C3 == 1        the recharge collapses to the step total
//   W_OLD + W_YGAMMA + W_NEW == 1      the flux quadrature is consistent (exact on a constant)
//   W_YGAMMA*gamma + W_NEW == 1/2      and second-order (exact on a linear integrand)
//
// WHY THEY EXIST. The exfiltration multiplier the active-set constraint hands to FillSpillMerge was
// read off the stage-2 residual alone, which recovers only C3 = 29.29% of the step. Measured on
// tests/multilake: 5.97e11 m^3 delivered against backward Euler's 2.00e12 m^3, and 9.5% of recharge
// lost from the physical budget. A bare stage coefficient was standing in for a step weight, and
// nothing in the build could see it. The BITES case below fails if that substitution is ever made
// again.
//
// Pure math (no MPI); compiled into test_dmda.x alongside the other unit tests and run by
// tests/run_unit_tests.sh. doctest's main + PetscInitialize live in test_dmda_gather.cpp; this TU
// only registers TEST_CASEs.

#include "doctest.h"
#include "tr_bdf2_coefficients.hpp"

#include <cmath>

using namespace trbdf2;

namespace {

// A manufactured step: arbitrary states and fluxes, with the two stage multipliers SOLVED from the
// stage equations rather than assumed. Nothing here is physical -- the point is that the algebra must
// hold for any values at all, so a coefficient typo cannot hide behind a plausible-looking fixture.
struct ManufacturedStep {
  // Stored volumes at the three step states, and the step-scaled flux F = dt*(net outflow/A + removal).
  // Chosen so both multipliers come out POSITIVE, i.e. both stages are genuinely pinned and shedding
  // water -- the case the active-set constraint exists for.
  double V0 = 3.25, Vg = 3.55, V1 = 3.7;
  double F0 = 0.9, Fg = 1.3, F1 = 1.8;
  double R  = 2.4;  // the step's recharge (a fixed volume)

  // SIGN. In the code the stage residual is f = storage + flux - recharge, and the pin discards it as
  // the multiplier E = max(0, -f*Sy). So E is MINUS the free residual: a cell the equation over-supplies
  // (f < 0) sheds E > 0. Each stage equation therefore closes as
  //   storage + flux - recharge + E = 0.
  //
  // stage 1:  V(Y_gamma) - V(w^n) + 0.5*gamma*(F(Y_gamma) + F(w^n)) - gamma*R + E1 = 0
  double E1() const { return -((Vg - V0) + 0.5 * GAMMA * (Fg + F0) - GAMMA * R); }

  // stage 2:  V(w^{n+1}) - C1*V(Y_gamma) + C2*V(w^n) + C3*F(w^{n+1}) - C3*R + E2 = 0
  double E2() const { return -((V1 - C1 * Vg + C2 * V0) + C3 * F1 - C3 * R); }

  // The step's flux quadrature, which is what the ocean-outflow and removal accumulators must total.
  double flux_over_step() const { return W_OLD * F0 + W_YGAMMA * Fg + W_NEW * F1; }
};

}  // namespace

TEST_CASE("TR-BDF2 coefficients: the identities the step balance rests on") {
  CHECK(GAMMA == doctest::Approx(2.0 - std::sqrt(2.0)));

  // Storage telescoping. Without this the step's storage term is not V(w^{n+1}) - V(w^n) and the
  // budget cannot be written against the stored volume at all.
  CHECK(C1 - C2 == doctest::Approx(1.0).epsilon(1e-14));

  // Recharge conservation over the step: stage 1 carries gamma*R weighted by C1, stage 2 carries C3*R.
  CHECK(C1 * GAMMA + C3 == doctest::Approx(1.0).epsilon(1e-14));

  // Flux quadrature consistency: exact for a constant integrand.
  CHECK(W_OLD + W_YGAMMA + W_NEW == doctest::Approx(1.0).epsilon(1e-14));

  // The two stages' shares of the step, quoted in the header and in WATER_BUDGET.md.
  CHECK(C1 * GAMMA == doctest::Approx(0.70710678).epsilon(1e-7));
  CHECK(C3 == doctest::Approx(0.29289322).epsilon(1e-7));
}

TEST_CASE("TR-BDF2 flux quadrature: second order, and no better") {
  // Nodes (0, gamma, 1) with weights (W_OLD, W_YGAMMA, W_NEW) as a rule for the integral over the
  // unit step. Exact on t^0 (checked above) and on t^1 -- the second-order condition.
  const double moment1 = W_YGAMMA * GAMMA + W_NEW * 1.0;  // W_OLD sits at t = 0
  CHECK(moment1 == doctest::Approx(0.5).epsilon(1e-14));

  // NOT exact on t^2. TR-BDF2 is a second-order method and this is the order barrier; pinning the
  // failure documents the order rather than leaving it to be assumed.
  const double moment2 = W_YGAMMA * GAMMA * GAMMA + W_NEW * 1.0;
  CHECK(moment2 != doctest::Approx(1.0 / 3.0).epsilon(1e-6));
  CHECK(moment2 == doctest::Approx(0.41421356).epsilon(1e-7));
}

TEST_CASE("TR-BDF2 stage 2: the BDF2 difference is exact through quadratics") {
  // V(w^{n+1}) - C1*V(Y_gamma) + C2*V(w^n) must equal C3*dt*V'(t=1) for V polynomial in t up to
  // degree 2, with t the fraction of the step and the nodes at (0, gamma, 1).
  const auto bdf2_diff = [](double at0, double at_g, double at_1) {
    return at_1 - C1 * at_g + C2 * at0;
  };

  // V = 1 (derivative 0)
  CHECK(bdf2_diff(1.0, 1.0, 1.0) == doctest::Approx(0.0).epsilon(1e-13).scale(1.0));
  // V = t (derivative 1)
  CHECK(bdf2_diff(0.0, GAMMA, 1.0) == doctest::Approx(C3 * 1.0).epsilon(1e-13));
  // V = t^2 (derivative 2t -> 2 at t = 1)
  CHECK(bdf2_diff(0.0, GAMMA * GAMMA, 1.0) == doctest::Approx(C3 * 2.0).epsilon(1e-13));
  // V = t^3 is NOT reproduced -- the order barrier again.
  CHECK(bdf2_diff(0.0, GAMMA * GAMMA * GAMMA, 1.0) != doctest::Approx(C3 * 3.0).epsilon(1e-6));
}

TEST_CASE("TR-BDF2 stage 1: the trapezoidal difference is exact through quadratics") {
  // V(Y_gamma) - V(w^n) must equal 0.5*gamma*(V'(gamma) + V'(0)) for V up to degree 2.
  const auto trap_gap = [](double v0, double vg) { return vg - v0; };
  const auto trap_rhs = [](double dv0, double dvg) { return 0.5 * GAMMA * (dvg + dv0); };

  // V = t
  CHECK(trap_gap(0.0, GAMMA) == doctest::Approx(trap_rhs(1.0, 1.0)).epsilon(1e-13));
  // V = t^2
  CHECK(trap_gap(0.0, GAMMA * GAMMA) == doctest::Approx(trap_rhs(0.0, 2.0 * GAMMA)).epsilon(1e-13));
  // V = t^3 is not reproduced (trapezoid is exact for a LINEAR integrand, i.e. quadratic V).
  CHECK(trap_gap(0.0, std::pow(GAMMA, 3)) != doctest::Approx(trap_rhs(0.0, 3.0 * GAMMA * GAMMA)).epsilon(1e-6));
}

TEST_CASE("TR-BDF2 step balance: C1*(stage 1) + (stage 2) telescopes exactly") {
  const ManufacturedStep s;

  // The fixture must actually exercise the constraint on BOTH stages, or the telescoping of the
  // multiplier terms is untested and the case passes for the wrong reason.
  CHECK(s.E1() > 0.0);
  CHECK(s.E2() > 0.0);

  // The step balance the water budget asserts:
  //   V(w^{n+1}) - V(w^n) = R - [step flux quadrature] - [C1*E1 + E2]
  const double storage_change = s.V1 - s.V0;
  const double exfiltration   = WE_STAGE1 * s.E1() + WE_STAGE2 * s.E2();
  const double residual       = storage_change - (s.R - s.flux_over_step() - exfiltration);

  CHECK(residual == doctest::Approx(0.0).epsilon(1e-13).scale(std::abs(s.R)));
}

TEST_CASE("TR-BDF2 step balance BITES: the stage-2 multiplier alone does not close") {
  // This is the defect the header describes, held in place so it cannot return. Using E2 as if it
  // were the step's exfiltration leaves exactly C1*E1 of water unaccounted for.
  const ManufacturedStep s;

  const double storage_change = s.V1 - s.V0;
  const double correct        = storage_change - (s.R - s.flux_over_step() - (WE_STAGE1 * s.E1() + s.E2()));
  const double stage2_only    = storage_change - (s.R - s.flux_over_step() - s.E2());

  CHECK(correct == doctest::Approx(0.0).epsilon(1e-13).scale(std::abs(s.R)));
  // Dropping C1*E1 makes the budget believe LESS water left than actually did, so it predicts a
  // LARGER remaining storage change than the true one and the residual is exactly -C1*E1. That sign
  // is the signature to look for in the field: an apparent SURPLUS of stored water, not a deficit.
  CHECK(stage2_only == doctest::Approx(-WE_STAGE1 * s.E1()).epsilon(1e-12));
  CHECK(std::abs(stage2_only) > 1e-6);  // the manufactured step really does expose the gap
}

TEST_CASE("TR-BDF2 exfiltration: a steady rate is recovered at C3 by stage 2 alone") {
  // The quantitative form of the same statement, and the one that predicts the measured factor. If a
  // cell exfiltrates at a constant rate through the step, each stage's multiplier is its own share of
  // the step, so the stage-2 multiplier alone recovers C3 = 29.29% and understates by 1/C3 = 3.4142.
  // Measured on tests/multilake: 2.00453e12 / 5.97221e11 = 3.356 against this predicted 3.414.
  const double rate = 1.0;                 // exfiltrated volume per unit of step
  const double E1   = GAMMA * rate;        // stage 1 spans gamma of the step
  const double E2   = C3 * rate;           // stage 2 contributes C3 of the step directly
  const double step = WE_STAGE1 * E1 + WE_STAGE2 * E2;

  CHECK(step == doctest::Approx(rate).epsilon(1e-13));
  CHECK(E2 / step == doctest::Approx(C3).epsilon(1e-13));
  CHECK(step / E2 == doctest::Approx(3.41421356).epsilon(1e-7));
}
