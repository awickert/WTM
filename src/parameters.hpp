#ifndef _parameters_hpp_
#define _parameters_hpp_

#include <stdint.h>
#include <cmath>
#include <limits>
#include <string>

struct Parameters {
  Parameters() = default;
  Parameters(const std::string& config_file);
  void check() const;
  std::string get_path(const std::string& time, const std::string& layer_name) const;
  std::string get_path(const std::string& layer_name) const;

  static constexpr auto UNINIT_STR = "uninitialized";

  // DEPRECATED (replaced by report_interval). Parsed for back-compat: if set and report_interval is not, it
  // seeds report_steps and warns. FSM no longer batches over maxiter -- it runs EVERY timestep.
  int32_t maxiter = -1;

  // report_interval: the number of timesteps (or a simulated time, e.g. "50yr") between equilibrium checks +
  // log lines. Parsed into report_steps (the fixed-dt loop count) and report_seconds (the report duration, the
  // span the adaptive loop covers and the unit of the output year). Default 100 steps + a LOUD warning if the
  // user omits it. This is NOT a coupling interval: FillSpillMerge runs every timestep regardless.
  int32_t report_steps   = -1;
  double  report_seconds = std::numeric_limits<double>::signaling_NaN();
  bool    report_interval_is_time = false;   // true if the user gave a time (so report_steps is derived via deltat)
  double  report_interval_time    = std::numeric_limits<double>::signaling_NaN();  // the user's time value (s), if any

  std::string outfile_prefix = UNINIT_STR;
  std::string region         = UNINIT_STR;
  std::string run_type       = UNINIT_STR;
  std::string surfdatadir    = UNINIT_STR;
  std::string textfilename   = UNINIT_STR;
  std::string time_start     = UNINIT_STR;
  std::string time_end       = UNINIT_STR;
  // Surface-water routing selector. DEFAULT "implicit": the exact, dt-independent in-residual seepage face
  // (wired into the Anderson residual and the Picard operator; adaptive-dt handles it via the feasible-set
  // predictor clamp in the error estimate). Alternatives: "explicit" (post-solve clamp -- robust everywhere,
  // ~1 cm from implicit), "off" (no collection -- NONPHYSICAL, warns), "legacy" (the old -wtm_ surface-flag
  // band-sink defaults). "" is accepted as a synonym for the default. See README / SURFACE_WATER_ROUTING.md.
  std::string runoff_collector = "implicit";

  double cells_per_degree = -1;

  double UNDEF = -1.0e7;

  int32_t infiltration_on = -1;
  int32_t supplied_wt     = -1;
  int32_t evap_mode       = -1;
  int32_t fsm_on          = -1;
  int32_t runoff_ratio_on = -1;

  double deltat          = std::numeric_limits<double>::signaling_NaN();
  double fdepth_a        = -1.;
  double fdepth_b        = -1.;
  double fdepth_fmin     = -1.;
  double southern_edge   = std::numeric_limits<double>::signaling_NaN();
  int32_t total_cycles   = -1;
  int32_t cycles_to_save = -1;      // DEPRECATED (replaced by save_nreport_interval); seeds it + warns if set.
  int32_t save_nreport_interval = -1;  // save a raster every K reports. Default 1 + a LOUD warning if omitted.

  double cellsize_n_s_metres = std::numeric_limits<double>::signaling_NaN();
  int32_t cycles_done        = 0;
  double infiltration_change = 0.;
  // Exact stored water volume at t=0, captured on the first PrintValues call, so the budget-closing
  // diagnostic can report the change in stored volume (see benchmark/WATER_BUDGET.md).
  double stored_volume_initial      = 0.;
  bool   have_stored_volume_initial = false;

  // Set for convenience within the code
  int32_t ncells_x = -1;
  int32_t ncells_y = -1;

  void print() const;
};

#endif
