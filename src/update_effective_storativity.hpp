#pragma once

// Smoothing width (metres) of the specific-yield -> surface-water transition at the land surface
// in the effective-storativity V(w). Default 0.01 m (1 cm); represents sub-grid land-surface
// roughness, physically defensible up to ~10 cm (esp. with large cells). Always on (no piecewise
// V form is wired); settable via -wtm_storativity_surface_smoothing_width.
//
// RECOMMENDATION (cold-start conditioning; see benchmark/SURFACE_SMOOTHING_STABILIZATION.md): the
// wtd=0 storativity jump is a driver of the frozen-coefficient contraction failure ("disease 2").
// A WIDER width (~0.1-0.5 m), composed with -wtm_Tbar, cuts cold-start iterations ~16-24% for BOTH
// Anderson and Picard (island AND 384k Esquibel) and lifts Picard's island ceiling 1->2 wk -- a real
// conditioning/speed win. It does NOT raise Anderson's step ceiling, and it SHIFTS the shallow-cell
// equilibrium ~0.2-0.35% (0.5 m; larger widths distort more -- 0.5 m is the sweet spot). Kept at the
// 0.01 m DEFAULT to preserve v2.0.1 numbers; raise it per-run when you want the speedup and can accept
// the small equilibrium shift.
extern double g_storativity_surface_smoothing_width;

// [WIP experiment -wtm_extended_soil] Treat the aquifer as continuing infinitely ABOVE the land
// surface: storativity = porosity everywhere (no jump to 1 for surface water), transmissivity
// continues past wtd=0 (no clamp), recharge always partitions as rech/porosity. This removes the
// wtd=0 FREE BOUNDARY from the GW time-stepping so the solve stays smooth (2nd order); the real
// surface is truncated later at the FSM handoff (excess -> depressions/off-map). See
// BDF2_RECHARGE_ORDER.md. Default off; when off, all functions use the standard surface physics.
extern bool g_extended_soil;

double updateEffectiveStorativity(const double my_original_wtd, const double my_wtd_T, const double my_porosity);

// Stored water per unit area V(wtd) (smooth C-inf), and its derivative dV/dwtd = the TANGENT
// specific yield. updateEffectiveStorativity is the SECANT of V (a 2-level backward-Euler
// construction); these expose V and its tangent for the BDF2-on-V storage discretization, which
// applies the 3-level BDF2 difference to V directly and uses the tangent as the operator diagonal.
double storedVolume(const double wtd, const double porosity);
double specificYield(const double wtd, const double porosity);
