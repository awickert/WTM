#pragma once

// Smoothing width (metres) of the specific-yield -> surface-water transition at the land surface
// in the effective-storativity V(w). Default 0.01 m (1 cm); represents sub-grid land-surface
// roughness, physically defensible up to ~10 cm (esp. with large cells). Always on (no piecewise
// V form is wired); settable via -wtm_storativity_surface_smoothing_width.
extern double g_storativity_surface_smoothing_width;

double updateEffectiveStorativity(const double my_original_wtd, const double my_wtd_T, const double my_porosity);

// Stored water per unit area V(wtd) (smooth C-inf), and its derivative dV/dwtd = the TANGENT
// specific yield. updateEffectiveStorativity is the SECANT of V (a 2-level backward-Euler
// construction); these expose V and its tangent for the BDF2-on-V storage discretization, which
// applies the 3-level BDF2 difference to V directly and uses the tangent as the operator diagonal.
double storedVolume(const double wtd, const double porosity);
double specificYield(const double wtd, const double porosity);
