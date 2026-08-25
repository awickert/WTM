// Unit tests for the effective-storativity math (src/update_effective_storativity.{hpp,cpp}):
// storedVolume V(wtd), its tangent specificYield = dV/dwtd, and updateEffectiveStorativity (the
// SECANT of V used by the backward-Euler storage term). These are the functions at the heart of the
// water (volume) formulation -- and the surface transition in V is what drives the adaptive MAX-norm
// cold-start spike investigated in GH awickert/WTM#13, so pinning the contract here is well-motivated.
//
// Pure math (no MPI); compiled into test_dmda.x alongside the DMDA tests and run by
// tests/run_unit_tests.sh. doctest's main + PetscInitialize live in test_dmda_gather.cpp; this TU only
// registers TEST_CASEs.
//
// Contract (smoothed absolute-value form, smoothing width eps = g_storativity_surface_smoothing_width):
//   V(wtd)  = 0.5*( wtd*(1+phi) + sqrt(wtd^2 + eps^2)*(1-phi) )
//   Sy(wtd) = dV/dwtd = 0.5*( (1+phi) + wtd*(1-phi)/sqrt(wtd^2 + eps^2) )
//   -> slope phi far below the surface (wtd << 0), slope 1 far above (wtd >> 0, ponded water),
//      a smooth transition of half-width ~eps across wtd = 0.

#include "doctest.h"
#include "update_effective_storativity.hpp"

#include <cmath>
#include <initializer_list>

namespace {

// Central finite difference of storedVolume -- the independent check that specificYield is its tangent.
double fd_dV(double wtd, double phi, double h = 1e-6) {
  return (storedVolume(wtd + h, phi) - storedVolume(wtd - h, phi)) / (2.0 * h);
}

// Restore the smoothing-width global on scope exit so a case that perturbs it cannot leak into later cases.
struct SmoothingWidthGuard {
  double saved;
  SmoothingWidthGuard() : saved(g_storativity_surface_smoothing_width) {}
  ~SmoothingWidthGuard() { g_storativity_surface_smoothing_width = saved; }
};

}  // namespace

TEST_CASE("storedVolume: asymptotic slopes, surface reference, monotonicity") {
  const double eps = g_storativity_surface_smoothing_width;  // default 0.01
  for (double phi : {0.05, 0.25, 0.40}) {
    // Far below the surface: V ~ phi*wtd (specific-yield storage). Far above: V ~ wtd (ponded, slope 1).
    CHECK(storedVolume(-100.0, phi) == doctest::Approx(phi * -100.0).epsilon(1e-6));
    CHECK(storedVolume(100.0, phi) == doctest::Approx(100.0).epsilon(1e-6));
    // Surface reference: V(0) = 0.5*eps*(1-phi) (small positive, the smoothed corner).
    CHECK(storedVolume(0.0, phi) == doctest::Approx(0.5 * eps * (1.0 - phi)));
    // Strictly increasing in wtd (physical: adding head never removes stored water).
    double prev = storedVolume(-50.0, phi);
    for (double w = -50.0 + 0.5; w <= 50.0; w += 0.5) {
      const double cur = storedVolume(w, phi);
      CHECK(cur > prev);
      prev = cur;
    }
  }
}

TEST_CASE("specificYield equals dV/dwtd (analytic tangent == finite difference of V)") {
  for (double phi : {0.05, 0.25, 0.40}) {
    for (double w : {-100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0})
      CHECK(specificYield(w, phi) == doctest::Approx(fd_dV(w, phi)).epsilon(1e-6));
    // Asymptotes and the exact surface midpoint.
    CHECK(specificYield(-1e4, phi) == doctest::Approx(phi).epsilon(1e-6));       // -> specific yield
    CHECK(specificYield(1e4, phi) == doctest::Approx(1.0).epsilon(1e-6));        // -> ponded (slope 1)
    CHECK(specificYield(0.0, phi) == doctest::Approx(0.5 * (1.0 + phi)));        // exact midpoint
  }
}

TEST_CASE("updateEffectiveStorativity is the secant of V with a tangent Delta->0 limit") {
  for (double phi : {0.05, 0.25, 0.40}) {
    // Finite step: the secant is exactly (V(w1) - V(w0)) / (w1 - w0).
    const double w0 = -2.3, w1 = 0.7;
    CHECK(updateEffectiveStorativity(w0, w1, phi)
          == doctest::Approx((storedVolume(w1, phi) - storedVolume(w0, phi)) / (w1 - w0)));
    // Straddling the surface (|w| >> eps): the secant is the average of the two slopes, 0.5*(1+phi).
    CHECK(updateEffectiveStorativity(-0.5, 0.5, phi) == doctest::Approx(0.5 * (1.0 + phi)).epsilon(1e-4));
    // Delta->0 limit (|dwtd| <= 1e-10) must return the TANGENT specificYield(w0), NOT 0/0. This guards the
    // historically-fixed factor-2 error in the small-step limit (see finding_analytic_jacobian_newton).
    for (double w : {-1.0, 0.0, 0.3}) {
      CHECK(updateEffectiveStorativity(w, w, phi) == doctest::Approx(specificYield(w, phi)));
      // Continuity across the 1e-10 branch: a tiny finite step matches the tangent it falls back to.
      CHECK(updateEffectiveStorativity(w, w + 1e-6, phi) == doctest::Approx(specificYield(w, phi)).epsilon(1e-4));
    }
  }
}

TEST_CASE("surface smoothing width sets the sharpness of the V transition (GH #13 kink)") {
  SmoothingWidthGuard guard;  // restore the default on exit
  const double phi = 0.25, w = 0.05;  // a point just above the surface, inside a wide transition
  g_storativity_surface_smoothing_width = 0.001;  // sharp: Sy near the ponded asymptote (1)
  const double sy_sharp = specificYield(w, phi);
  g_storativity_surface_smoothing_width = 0.5;     // wide: Sy pulled toward the midpoint 0.5*(1+phi)
  const double sy_wide = specificYield(w, phi);
  CHECK(sy_sharp > sy_wide);                                   // wider smoothing => gentler transition
  CHECK(sy_sharp == doctest::Approx(1.0).epsilon(0.05));       // sharp corner: essentially ponded at w>>eps
  CHECK(sy_wide < 0.5 * (1.0 + phi) + 0.05);                   // wide corner: still climbing out of the midpoint
}
