#pragma once

// Smoothing width (metres) of the specific-yield -> surface-water transition at the land
// surface in the effective-storativity V(w). Default 0.01 m (1 cm, sub-grid roughness).
// Settable via -wtm_storativity_eps for the BDF2-order experiment: physically defensible up to
// ~10 cm (surface roughness, esp. with large cells). See BDF2_ADAPTIVE_DESIGN.md.
extern double g_storativity_eps;

// Experiment (-wtm_const_storativity): force S == porosity (constant), removing the secant, the
// surface corner, and all head-dependence at once. The decisive discriminator for whether the
// BDF2 order-1 comes from the storativity treatment or the time-integration structure.
extern bool g_const_storativity;

double updateEffectiveStorativity(const double my_original_wtd, const double my_wtd_T, const double my_porosity);

// Stored water per unit area V(wtd) (smooth C-inf), and its derivative dV/dwtd = the TANGENT
// specific yield. updateEffectiveStorativity is the SECANT of V (a 2-level backward-Euler
// construction); these expose V and its tangent for the BDF2-on-V storage discretization, which
// applies the 3-level BDF2 difference to V directly and uses the tangent as the operator diagonal.
double storedVolume(const double wtd, const double porosity);
double specificYield(const double wtd, const double porosity);
