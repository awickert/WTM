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
  // Surface-water routing selector. DEFAULT "implicit": the exact, dt-independent in-residual exfiltration constraint
  // (wired into the Anderson residual and the Picard operator; adaptive-dt handles it via the feasible-set
  // predictor clamp in the error estimate). Alternatives: "explicit" (post-solve clamp -- robust everywhere,
  // ~1 cm from implicit), "off" (no collection -- NONPHYSICAL, warns), "legacy" (the old -wtm_ surface-flag
  // band-sink defaults). "" is accepted as a synonym for the default. See README / SURFACE_WATER_ROUTING.md.
  std::string runoff_collector = "implicit";

  // Grid geometry. cells_per_degree / southern_edge are DEPRECATED config inputs (the `grid:` block),
  // kept only as an override for inputs that lack georeferencing. By default the geometry is derived from
  // the input topography's GDAL geotransform (#124): ns_deg_per_cell / ew_deg_per_cell are the true
  // (possibly non-square) degree spacings, and southern_edge the domain's southern-edge latitude.
  double cells_per_degree = -1;
  double ns_deg_per_cell  = std::numeric_limits<double>::signaling_NaN();  // N-S degrees per cell (|dy|)
  double ew_deg_per_cell  = std::numeric_limits<double>::signaling_NaN();  // E-W degrees per cell (dx)

  double UNDEF = -1.0e7;

  // Defaults adopted from the config_flags_prototype.yaml schema (Phase 2 hard cutover): an omitted key takes
  // the prototype default rather than erroring (the old parser used -1 sentinels + required them all).
  int32_t infiltration_on = 0;   // surface_water.infiltration_during_flow: false
  int32_t supplied_wt     = 0;   // run.initial_water_table: omit -> saturated (wtd = 0)   [TODO: folder auto-detect]
  int32_t evap_mode       = 0;   // dropped from the config (vestigial when the ET sigmoid is on = default); 0 = remove
  int32_t fsm_on          = 1;   // surface_water.mode: routed
  int32_t runoff_ratio_on = 0;   // surface_water.runoff_ratio: omit -> 0 (off)
  double  runoff_ratio_uniform = -1.0;  // >=0: uniform runoff ratio everywhere; <0: read the runoff_ratio raster
  std::string initial_wt_path;          // run.initial_water_table: <path> -> load the starting WT from this file
  std::string verbosity = "normal";     // output.verbosity: quiet | normal | verbose (console/log chatter level)
  std::string output_directory;         // output.directory: parent dir; each run gets its own subdir below.
                                        //   Empty = legacy (outfile_prefix / run_log used as literal paths).
  std::string if_exists = "increment";  // output.if_exists: increment (run<NNN>_<ts>/) | overwrite | error

  double deltat          = std::numeric_limits<double>::signaling_NaN();
  double fdepth_a        = -1.;
  double fdepth_b        = -1.;
  double fdepth_fmin     = -1.;
  double southern_edge   = std::numeric_limits<double>::signaling_NaN();
  // total_time: the total simulated time to run. Parsed from an explicit unit ("500yr" or "1000s"); a bare
  // number is REJECTED (unlike report_interval), to avoid a steps-vs-seconds ambiguity. Must be an integer
  // multiple of report_seconds (the report span = report_steps*deltat), because the loop advances one whole
  // report at a time -- resolved to total_reports below. Replaces the old user-facing total_cycles concept.
  double  total_time    = std::numeric_limits<double>::signaling_NaN();  // seconds
  int32_t total_reports = -1;  // derived from total_time / report_seconds (validated to be an exact integer)
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

// Parse a simulated-time value ("500yr" / "1000s"; a bare number = years, with a warning) into seconds.
double parse_time_seconds(const std::string& v, const char* key);

// Phase 2b: translate the CLI-flag-backed config sections (solver / dev / boundaries / equilibrium_stop /
// transmissivity background / parallel.threads_per_rank) into PETSc options + omp_set_num_threads. Call once,
// AFTER PetscInitialize and before the SNES is built. CLI flags override the config.
void apply_config_petsc_options(const std::string& config_file);

#endif
