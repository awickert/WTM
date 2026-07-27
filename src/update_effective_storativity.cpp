#include "update_effective_storativity.hpp"

#include <cmath>

// Smoothing width of the surface transition; default 1 cm, overridable via
// -wtm_storativity_surface_smoothing_width (see the header / BDF2_ADAPTIVE_DESIGN.md).
double g_storativity_surface_smoothing_width = 0.01;
bool   g_extended_soil                        = false;  // [WIP] see header / BDF2_RECHARGE_ORDER.md

// Stored water per unit area V(wtd). Smooth (C-inf) blend: slope ~1 above the surface (surface
// water), ~porosity below (specific yield), over a g_storativity_surface_smoothing_width transition.
// Extended-soil: slope porosity everywhere (aquifer continues above the surface, no jump to 1).
double storedVolume(const double wtd, const double porosity) {
  if (g_extended_soil) return porosity * wtd;
  const double eps = g_storativity_surface_smoothing_width;
  return 0.5 * (wtd * (1.0 + porosity) + std::sqrt(wtd * wtd + eps * eps) * (1.0 - porosity));
}

// Tangent specific yield dV/dwtd (the head-consistent coefficient for BDF2-on-V).
double specificYield(const double wtd, const double porosity) {
  if (g_extended_soil) return porosity;
  const double eps = g_storativity_surface_smoothing_width;
  return 0.5 * ((1.0 + porosity) + wtd * (1.0 - porosity) / std::sqrt(wtd * wtd + eps * eps));
}

// Effective storativity = SECANT of V between my_original_wtd and my_new_wtd (a 2-level
// backward-Euler construction; exact volume change for BE). Near-equal wtds fall back to the
// tangent dV/dwtd to avoid 0/0.
double updateEffectiveStorativity(const double my_original_wtd, const double my_new_wtd, const double my_porosity) {
  if (g_extended_soil) return my_porosity;
  const double dwtd = my_new_wtd - my_original_wtd;
  if (std::abs(dwtd) > 1e-10) {
    return (storedVolume(my_new_wtd, my_porosity) - storedVolume(my_original_wtd, my_porosity)) / dwtd;
  }
  return specificYield(my_original_wtd, my_porosity);
}

// double updateEffectiveStorativity(
//     const double my_original_wtd,
//     const double my_new_wtd,
//     const double my_porosity,
//     const double starting_effective_storativity) {
//   if (my_original_wtd <= 0. && my_new_wtd <= 0.) {  // both are below ground, so we can use the original porosity
//     return my_porosity;
//   } else if (my_original_wtd > 0. && my_new_wtd > 0.) {  // both are above ground, so the porosity is 1
//     return 1.;                                           // 1.;
//   } else {
//     const double change_in_water_column_thickness = std::abs(my_new_wtd - my_original_wtd);
//     if (my_original_wtd <= 0. && my_new_wtd > 0.) {  // started below ground and ended above ground
//       // First, scale the change in wtd as if the whole column has the porosity of the belowground area
//       const double scaled_change_in_water_column_thickness =
//           change_in_water_column_thickness * starting_effective_storativity / my_porosity;
//       // Then get just that portion that is >0 (aboveground), and scale it back down to the equivalent thickness with
//       // 100% porosity (surface water)
//       const double aboveground_water_column_thickness =
//           (scaled_change_in_water_column_thickness + my_original_wtd) *
//           starting_effective_storativity;  // Above-ground effective porosity is ==1, so no need to actually divide
//           by
//                                            // 1 here (save on computation).
//       // belowground water column thickness is equal to -original_wtd, so no need to assign it to a new variable.
//       return (aboveground_water_column_thickness + my_porosity * -my_original_wtd) /
//              (aboveground_water_column_thickness - my_original_wtd);
//     } else {
//       // This means that (my_original_wtd > 0. && my_new_wtd < 0.). started above ground and ended below
//       // ground.
//       const double scaled_change_in_water_column_thickness =
//           change_in_water_column_thickness *
//           starting_effective_storativity;  // divided by 1 for above-ground porosity
//                                            //  if (scaled_change_in_water_column_thickness < my_new_wtd) {
//       // when rescaling the water according to the porosity values, there is no longer enough to reach below ground;
//       // so the scaled new porosity will be = 1
//       //    return 1;
//       // } else {
//       // Get the belowground water thickness and expand it to a deeper depth in order to account for changing
//       porosity const double belowground_water_column_thickness =
//           (scaled_change_in_water_column_thickness - my_original_wtd) * starting_effective_storativity / my_porosity;
//       return (my_original_wtd + my_porosity * belowground_water_column_thickness) /
//              (my_original_wtd + belowground_water_column_thickness);
//     }
//   }
// }

//  if (my_original_wtd <= 0. && my_new_wtd <= 0.) {  // both are below ground, so we can use the original porosity
//    return my_porosity;
//  } else if (my_original_wtd > 0. && my_new_wtd > 0.) {  // both are above ground, so the porosity is 1
//    return 1.;                                           // 1.;
//  } else {
//    //    if (my_original_wtd > 0 && my_new_wtd > 0) {
//    //      float cutoff        = 0.1;
//    //      float porosity_diff = 1 - my_porosity;
//    //
//    //      return porosity_diff / (cutoff / my_new_wtd) + my_porosity;
//    //    } else {
//    const double change_in_water_column_thickness = std::abs(my_new_wtd - my_original_wtd);
//    if (my_original_wtd <= 0. && my_new_wtd > 0.1) {  // started below ground and ended above ground
//      // First, scale the change in wtd as if the whole column has the porosity of the belowground area
//      const double scaled_change_in_water_column_thickness =
//          change_in_water_column_thickness * starting_effective_storativity / my_porosity;
//      // Then get just that portion that is >0 (aboveground), and scale it back down to the equivalent thickness with
//      // 100% porosity (surface water)
//      const double aboveground_water_column_thickness =
//          (scaled_change_in_water_column_thickness + my_original_wtd) *
//          starting_effective_storativity;  // Above-ground effective porosity is ==1, so no need to actually divide by
//                                           // 1 here (save on computation).
//      // belowground water column thickness is equal to -original_wtd, so no need to assign it to a new variable.
//      return (aboveground_water_column_thickness + my_porosity * -my_original_wtd) /
//             (aboveground_water_column_thickness - my_original_wtd);
//    } else if (my_new_wtd <= 0. && my_original_wtd > 0.1) {
//      // This means that (my_original_wtd > 0. && my_new_wtd < 0.). started above ground and ended below
//      // ground.
//      const double scaled_change_in_water_column_thickness =
//          change_in_water_column_thickness *
//          starting_effective_storativity;  // divided by 1 for above-ground porosity
//                                           //  if (scaled_change_in_water_column_thickness < my_new_wtd) {
//      // when rescaling the water according to the porosity values, there is no longer enough to reach below ground;
//      // so the scaled new porosity will be = 1
//      //    return 1;
//      // } else {
//      // Get the belowground water thickness and expand it to a deeper depth in order to account for changing porosity
//      const double belowground_water_column_thickness =
//          (scaled_change_in_water_column_thickness - my_original_wtd) * starting_effective_storativity / my_porosity;
//      return (my_original_wtd + my_porosity * belowground_water_column_thickness) /
//             (my_original_wtd + belowground_water_column_thickness);
//
//    } else if (my_original_wtd <= 0.) {  // this is the case where my_new_wtd is between 0 and 0.1.
//      float porosity_diff = 1 - my_porosity;
//      return porosity_diff / (0.1 / my_new_wtd) + my_porosity;
//      // return starting_effective_storativity;
//    } else {  // this is the case where my_new_wtd <= 0 and my_original_Wtd is between 0 and 0.1.
//      float porosity_diff = 1 - my_porosity;
//      return porosity_diff / (0.1 / my_original_wtd) + my_porosity;
//      // return starting_effective_storativity;
//    }
//  }
//}
