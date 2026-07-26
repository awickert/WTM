#pragma once

// Smoothing width (metres) of the specific-yield -> surface-water transition at the land
// surface in the effective-storativity V(w). Default 0.01 m (1 cm, sub-grid roughness).
// Settable via -wtm_storativity_eps for the BDF2-order experiment: physically defensible up to
// ~10 cm (surface roughness, esp. with large cells). See BDF2_ADAPTIVE_DESIGN.md.
extern double g_storativity_eps;

double updateEffectiveStorativity(const double my_original_wtd, const double my_wtd_T, const double my_porosity);
