#ifndef _parameters_hpp_
#define _parameters_hpp_

#include <stdint.h>
#include <cmath>
#include <initializer_list>
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
  // Surface-water routing selector: how the wtd <= topo + surface_water_depth exfiltration constraint is
  // ENFORCED. DEFAULT "active_set": the semismooth constraint solved INSIDE the residual. It is the only
  // enforcement measured to leave no SPURIOUS dt-dependence -- `implicit`'s in-residual siphon removes at
  // rate max(0,wtd)/dt, so its retained head is ~LINEAR in dt (1.97 / 0.68 / 0.34 m at dt = 1, 1/3, 1/6
  // week), and FSM routes that dt-dependent excess into a different set of lakes: the lake COUNT itself
  // moves with dt (tests/multilake). Active-set also eliminates the between-step FSM shock (ratio 0.985 ->
  // 3.6e-13) and is 2-100x cheaper across solvers (benchmark/scheme_bench).
  // Alternatives: "implicit" (in-residual siphon -- the former default, dt-dependent), "explicit"
  // (post-solve clamp -- robust on every solver, dt-lagged), "off" (no collection -- NONPHYSICAL, warns),
  // "legacy" (the old -wtm_ surface-flag band-sink defaults). "" is a synonym for the default.
  // CAVEAT: the active-set pin lives in the Anderson residual only; the Picard operator and Newton Jacobian
  // carry no tangent for it, so those paths warn and should use "explicit" until that lands.
  // See README / SURFACE_WATER_ROUTING.md.
  std::string runoff_collector = "active_set";
  // True only if surface_water.collection.method was PRESENT in the config. Lets the solver-dependent
  // default resolution below distinguish "the user chose active_set" from "active_set is the default",
  // so an explicit choice is always honoured (with a warning) and never silently downgraded.
  bool runoff_collector_set = false;

  // Evaporation taper 3 (evaporation.extinction_depth): depth below which ET cannot reach the table [m].
  double extinction_depth = 8.0;
  // evaporation.tapers -- the two surface-transition tapers, BOTH default ON. They are SWITCHES, not
  // values: what each uses lives beside it (et_sigmoid.* for the first, extinction_depth above for the
  // second). Named for what each DOES rather than by the internal numbering ("taper 2", "taper 3"),
  // which meant nothing outside transient_groundwater.cpp and had already lost taper 1 to retirement.
  bool taper_surface_transition = true;  // blend phreatic ET to open-water ET across wtd = 0
  bool taper_depth_extinction   = true;  // no phreatic ET below evaporation.extinction_depth

  // run.equilibrium_stop: tol is a WATER depth [m] (0 = the stop is off). Its DEFAULT is run-type
  // dependent -- 0.001 for equilibrium, 0 for transient (a time-evolution run must play out in full) --
  // so the was-it-set flag is required: an absent key must reach that per-run-type default, not a
  // constant. metric selects how the per-cycle change is aggregated (max|rms|frac).
  double      eq_tol     = 0.0;
  bool        eq_tol_set = false;
  std::string eq_metric  = "frac";

  // boundaries.land: the LAND-edge condition (ocean is always Dirichlet h = 0, not configurable).
  // `dirichlet_sea_level` -> true (ghost head at sea level); `neumann_toposlope` -> false (default,
  // terrain-following no-flow). Stored resolved: the consumer only ever asks which of the two it is.
  bool land_boundary_dirichlet = false;

  // dev.storage_form: which ASSEMBLY the backward-Euler storage term uses -- `volume` (exact dV, RHS b=0)
  // or `secant` (S*dh, RHS b=h^n). NOT an accuracy choice: S is the exact secant, so the two are the same
  // equation and tests/storage_equivalence pins them bit-identical. DEFAULT volume, which is what the
  // active-set constraint requires; `secant` exists so that equivalence test has something to compare.
  // Since the default is volume, `false` here can only mean an EXPLICIT dev.storage_form: secant.
  bool volume_storage = true;

  // solver.method: the solver path. "" = unset, which resolves to the matrix-free Anderson default.
  // -wtm_picard and -wtm_newton are RETIRED; solver.method: anderson remains a flag for now (it FORCES the
  // matrix-free path and is the last of the three), so CreateSNES still ORs this member with it.
  std::string solver_method;

  // solver.time_integration: backward-euler (default) | bdf2 | tr-bdf2. "" = unset = backward-euler.
  // solver.time_integration: bdf2 and solver.time_integration: tr-bdf2 are retired one at a time; while either remains, CreateSNES ORs this
  // member with the surviving flag rather than replacing it.
  std::string time_integration;
  // True when the key was ABSENT rather than explicitly written, so the run can report what it resolved
  // to instead of leaving the user to infer it. (Named _auto from when `auto` was a writable value; the
  // word is refused now -- omission is the only way to ask for the default.)
  bool time_integration_auto = false;

  // solver.newton.dt_continuation: PSEUDO-TRANSIENT CONTINUATION (PTC) -- the standard method for
  // globalising Newton toward a STEADY STATE (Kelley & Keyes 1998 -- CITATION UNVERIFIED, written from
  // memory; confirm the reference before this ships). A
  // pseudo-time term S/dt is added to the Jacobian, which makes it diagonally dominant and keeps a far
  // or cold guess inside the basin; dt is then RAMPED from small toward large as the state warms, so
  // the term fades and the iteration approaches the true steady Newton step. "Continuation" is the
  // literature's word for that deformation from an easy problem to the hard one -- it does NOT mean
  // continuing or resuming a run, which is tests/snapshot_restart's subject.
  //
  // Its control law is SOLVE-EASE, not error: grow when the step converged in <= grow_if_niter_leq
  // iterations, hold when it was hard, reject-shrink-retry when it did not converge. There is no error
  // estimate on this path, and simulated time is not respected -- it marches a COUNT of accepted steps.
  //
  // RESOLVED here, because its default is not constant:
  // solver.method: newton IMPLIES it, since plain Newton does not converge from a cold start
  // (DIVERGED_LINE_SEARCH). solver.newton.dt_continuation: false opts out -- legitimate for a warm finish --
  // and CreateSNES warns. The `_set` flag distinguishes "the user declined" from "nobody asked", which
  // is what makes the newton default overridable rather than sticky.
  bool t_bar = false;

  // solver.smoothing.* and dev.under_relaxation: read STRAIGHT from here by the consumers in
  // transient_groundwater.cpp. They used to travel YAML -> string -> PETSc options DB -> re-parsed,
  // which was symmetrical back when the -wtm_ FLAG was the interface and is pure overhead now that the
  // config is. Defaults below are the consumers' own compile-time values.
  double ksat_surface_smoothing        = 0.0;   // eps0: surface conductivity clamp width [m]
  double ksat_soilbottom_smoothing     = 0.0;   // eps1: -1.5 m conductivity transition width [m]
  double storativity_surface_smoothing = 0.01;  // sub-grid roughness blend width [m]; always on
  double under_relaxation              = 1.0;   // 1 = off
  // dev.allow_aboveground_water_columns [DEVELOPER, NONPHYSICAL]: disable the surface-water clamp.
  bool   allow_aboveground_water_columns = false;

  // solver.anderson.restart.*: the outer rho-driven restart loop. Defaults are AppCtx's own.
  bool   ar_enabled      = false;
  double ar_rho          = 0.9;   // restart when rho exceeds this...
  int    ar_patience     = 2;     // ...for this many consecutive iterations
  int    ar_max_it       = 40;    // cap per Anderson phase before a forced restart
  int    ar_max_restarts = 30;    // outer restart cap

  // solver.time_step.mode: WHO SIZES THE STEP -- fixed | adaptive | ramp. ONE key, because these are
  // three answers to ONE question. They used to be two independent booleans, solver.adaptive_dt and
  // solver.newton.dt_continuation, which could BOTH be true; adaptive then silently won and the ramp
  // never ran. An enum makes that contradiction unrepresentable rather than something to abort on.
  // See the resolution block in parameters.cpp for what an ABSENT key resolves to, and why adaptive is
  // not a superset of ramp.
  std::string time_step_mode;              // always concrete after parse: fixed | adaptive | ramp
  bool        time_step_mode_set = false;  // the user WROTE it, rather than it being resolved

  // DERIVED from time_step_mode, in ONE place (parameters.cpp), so they cannot disagree with the mode
  // or with each other. The solver paths read these; nothing else sets them.
  bool adaptive_dt     = false;
  bool dt_continuation = false;

  // solver.time_step.error_tol: per-step local-error target in WATER volume. An ABSENT key leaves it
  // unset so the consumer's own default (which tracks eq_tol) applies. Named
  // solver.water_volume_timestep_error_tol until it joined the rest of the controller.
  double dt_tol     = 0.1;
  bool   dt_tol_set = false;

  // solver.dt_max: cap on the adaptive/continuation step [s]. An ABSENT key leaves this UNSET, and each consumer keeps its own default -- they differ deliberately: the continuation ramp
  // caps at 1000*deltat, while the adaptive controller treats 0 as "no cap". A single shared default
  // would silently change one of them, so the was-it-set flag carries that distinction.
  double dtc_dt_max     = 0.0;
  bool   dtc_dt_max_set = false;

  // Equilibrium stop: fraction of land cells allowed above eq_tol for the `frac` metric.
  double eq_frac = 0.001;

  // Background (bedrock) transmissivity floor [m^2/s], 0 = off (v2.0.1 behaviour).
  double t_bedrock = 0.0;

  // parallel.threads_per_rank as RESOLVED. Applied via omp_set_num_threads rather than a PETSc option,
  // so it is retained here for full_config.yaml.
  int threads_per_rank = 1;

  // Evaporation: the always-on soil<->open-water ET sigmoid (evaporation.et_sigmoid). Config-owned --
  // these were reached only through -wtm_evap_taper_wtdc / -wtm_evap_taper_s, which the YAML bridge set
  // from these very keys, so the flags were pure transport with no CLI callers anywhere in the repo.
  // Held here instead: the value is stored, schema-checked, and printed in the resolved-config log.
  double evap_taper_wtdc = 0.05;  // wtd_c: half-rate depth [m] (small +, pond->exposed)
  double evap_taper_s    = 0.1;   // s: logistic transition width [m]

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
  // Cumulative solve accounting, reported alongside the budget so the run log carries the DENOMINATORS
  // a reader needs to turn any cumulative volume into a rate -- and, more importantly, so this holds
  // where it can be checked:
  //     NO cumulative quantity may be proportional to the SOLVE COUNT.
  //     Every one must be proportional to ELAPSED TIME, or be a difference of states.
  // Three separate bugs violated that (the adaptive controller resizing dt before the step was
  // accounted; column 9's missing rech_dt_scale; the runoff-ratio channel delivered at nominal-step
  // size per sub-step), and each showed up as a column tracking solves instead of time. With both
  // denominators in the file the violation is visible by inspection. Deliberately NOT accompanied by
  // derived rate columns: a rate computed from a wrong amount is wrong in the same proportion, so it
  // adds no checking power -- only the denominators do.
  // TRUE elapsed simulated time, accumulated from the ACCEPTED steps themselves. It must not be
  // derived as cycles_done * report_seconds: that assumes every cycle covers one report span, which
  // holds for the fixed-dt and adaptive loops but NOT for -wtm_dt_continuation, whose loop runs
  // report_steps STEPS at a dt it is free to grow. Measured: a 20-cycle continuation run at
  // deltat 9.09e+06 s covers 5.77 yr while the derived form claimed 20.000 yr.
  double  elapsed_time_s     = 0.0;
  // Simulated time at which the runoff-ratio share was last handed over. The amount to book is the
  // rate times the interval SINCE THEN -- not one step's worth -- because the handoff cadence differs
  // by configuration: with FillSpillMerge on it happens every accepted step, with FSM off only once
  // per report. Scaling by a single step's dt under-counted by the steps-per-report factor.
  double  runoff_booked_upto_s = 0.0;
  int64_t solves_done        = 0;  // accepted groundwater solves
  int64_t rejects_done       = 0;  // rejected + retried steps (adaptive / dt-continuation only)
  // Per-cycle change in WATER VOLUME (|S*Dwtd|, water per unit area -- a depth in metres, not m^3),
  // carried here purely so PrintValues can write it to the run log. The authoritative copies live in
  // AppCtx (last_cycle_dw_water / last_cycle_rms_water) and drive the equilibrium stop; these are set
  // from them in WTM.cpp immediately before PrintValues, in the same unconditional block, so they
  // cannot go one cycle stale the way the budget baseline once did.
  double last_cycle_dw_volume  = 0.0;   // max over land cells
  double last_cycle_rms_volume = 0.0;   // rms over land cells
  double infiltration_change = 0.;
  // Exact stored water volume at t=0, captured by CaptureInitialStoredVolume at the END of
  // initialise() -- before any stepping -- so the budget-closing diagnostic differences a storage
  // change over the SAME cycles the flux accumulators cover. It used to be taken on the first
  // PrintValues call, which is the end of cycle 0, so the first cycle's storage change was missing
  // from d_stored while its fluxes were present. See benchmark/WATER_BUDGET.md.
  double stored_volume_initial      = 0.;
  bool   have_stored_volume_initial = false;

  // Set for convenience within the code
  int32_t ncells_x = -1;
  int32_t ncells_y = -1;

  void print() const;
};

// Parse a simulated-time value ("500yr" / "1000s"; a bare number = years, with a warning) into seconds.
double parse_time_seconds(const std::string& v, const char* key);

// Validate a config ENUM. The schema check in parameters.cpp validates KEYS; this validates VALUES, which
// is the same defect one level down: `solver.method: pickard` used to fall through the bridge's if/else
// chain to the DEFAULT and report success, so a sweep over a misspelled solver silently compared Anderson
// with Anderson. Throws naming the key, the allowed values, and what was actually given.
const std::string& require_enum(const std::string& value, const char* key,
                                std::initializer_list<const char*> allowed);

// Phase 2b: translate the CLI-flag-backed config sections (solver / dev / boundaries / equilibrium_stop /
// transmissivity background / parallel.threads_per_rank) into PETSc options + omp_set_num_threads. Call once,
// AFTER PetscInitialize and before the SNES is built. CLI flags override the config.
void apply_config_petsc_options(const std::string& config_file);

#endif
