#include "parameters.hpp"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <map>
#include <set>
#include <string>
#include <vector>

// Parse a simulated-time value with an explicit unit ("500yr" / "1000s"); a bare number defaults to YEARS
// with a loud warning so the assumption is never silent. Returns seconds. (Exposed for the Phase-2b config
// bridge in WTM.cpp.)
double parse_time_seconds(const std::string& v, const char* key) {
  if (v.size() > 2 && v.substr(v.size() - 2) == "yr")
    return std::stod(v.substr(0, v.size() - 2)) * 31536000.0;
  if (v.size() > 1 && v.back() == 's' && std::isdigit(static_cast<unsigned char>(v[v.size() - 2])))
    return std::stod(v.substr(0, v.size() - 1));
  std::cerr << "WARNING [" << key << "]: no unit for '" << v << "' -- ASSUMING YEARS (" << v
            << "yr). Set '" << v << "yr' or '<seconds>s' to silence.\n";
  return std::stod(v) * 31536000.0;
}

namespace {

// THE CONFIG DICTIONARY: every key the model understands, indexed by its parent path ("" = top level).
// A key absent from here is REJECTED -- see validate_config_keys below for why that is worth an abort.
//
// Keeping this in step with the readers is the maintenance cost, and it is deliberately paid in ONE
// place. Two readers consume this file: Parameters (member-backed keys, this file) and
// apply_config_config_petsc_options (the YAML->PetscOptions bridge in WTM.cpp, which owns solver / dev /
// parallel / boundaries / evaporation / run.equilibrium_stop / surface_water.collection.sink /
// output.verbosity). Both are covered here, so a key that only the bridge reads still validates.
// tests/config_schema asserts that this table and the reference config.yaml agree in both directions,
// which is what catches a key added to one and not the other.
const std::map<std::string, std::set<std::string>>& config_schema() {
  static const std::map<std::string, std::set<std::string>> schema = {
      {"", {"run", "time", "transmissivity", "surface_water", "evaporation", "boundaries", "solver",
            "dev", "parallel", "io", "output", "derived"}},
      {"run", {"type", "initial_water_table", "equilibrium_stop"}},
      {"run.equilibrium_stop", {"tol", "metric", "frac"}},
      {"time", {"total", "report_interval", "save_every_n_reports"}},
      // OUTPUT-ONLY. `derived` records what the run READ FROM THE DATA rather than what anyone chose:
      // grid geometry comes from the input raster's GDAL geotransform, not from the config. It is
      // written into full_config.yaml so a run's provenance is complete, and ACCEPTED-AND-IGNORED here
      // so that file stays loadable -- full_config.yaml promises to be re-runnable as-is, and a config
      // the model refuses to read would break that promise on its own output.
      //
      // The distinction is load-bearing for the explicit-config rule (tests/config_identity.py): a test
      // config must state every SETTING the run resolved to, but it cannot state a DERIVED fact, since
      // that comes from the raster. Without somewhere to put them, derived values would make the rule
      // unsatisfiable; with this section, the comparator simply excludes it.
      {"derived", {"ns_deg_per_cell", "ew_deg_per_cell", "southern_edge", "cells_per_degree",
                   "ncells_x", "ncells_y"}},
      {"transmissivity", {"fdepth", "additive_background_transmissivity"}},
      {"transmissivity.fdepth", {"a", "b", "fmin"}},
      {"surface_water", {"mode", "runoff_ratio", "infiltration_during_flow", "collection", "fsm_coupling"}},
      {"surface_water.collection", {"method"}},
      {"evaporation", {"et_sigmoid", "extinction_depth", "tapers"}},
      {"evaporation.tapers", {"surface_transition", "depth_extinction"}},
      {"evaporation.et_sigmoid", {"wtd_center", "logistic_width"}},
      {"boundaries", {"land"}},
      {"solver", {"method", "tolerance", "max_iterations", "time_integration", "adaptive_dt",
                  "t_bar",
                  "time_step", "smoothing", "anderson", "newton", "convergence"}},
      // solver.convergence: what the PER-SOLVE step test judges. `metric: volume` swaps the head
      // relative-step test for |S*Δwtd|, so all three "close enough" gates (this, run.equilibrium_stop
      // and solver.time_step.error_tol) finally speak the same units. water_volume_tol is read ONLY when
      // metric: volume, and sits beside it for that reason.
      {"solver.convergence", {"metric", "water_volume_tol"}},
      // solver.time_step: ONE step-size controller, deliberately not nested under adaptive_dt --
      // Newton's dt_continuation ramp reads the same dials, so an `adaptive_`-prefixed home would
      // misdescribe them.
      {"solver.time_step", {"dt", "grow", "shrink", "grow_if_niter_leq", "max_retries", "norm",
                               "dt_max", "error_tol"}},
      // solver.smoothing: widths that ROUND a kink in the coefficients. The two ksat_* default to 0
      // (sharp) and exist so a Jacobian finite-difference check has a smooth tangent; they are off in a
      // normal run. storativity_surface is different in kind -- 0.01 m, always on, sub-grid roughness --
      // and sits here only because the three share a mechanism.
      {"solver.smoothing", {"ksat_surface", "ksat_soilbottom", "storativity_surface"}},
      // solver.anderson: settings only the matrix-free Anderson path reads -- hence nested, unlike the
      // shared blocks above. restart is its own mapping because it is one switch plus four constants.
      {"solver.anderson", {"restart"}},
      {"solver.anderson.restart", {"enabled", "rho", "patience", "max_it", "max_restarts"}},
      // solver.newton: read only on the Newton path.
      {"solver.newton", {"dt_continuation", "dt0"}},
      // dev.active_set was REMOVED 2026-09-01: it was a SECOND YAML route to the same enforcement as
      // surface_water.collection.method: active_set, and it silently OVERRODE an explicit method (measured:
      // 54/256 cells, max 0.127 m, with no log line). One setting, one key. Removing it from this schema is
      // what makes an old config say so instead of drifting.
      {"dev", {"allow_aboveground_water_columns", "storage_form", "under_relaxation"}},
      {"parallel", {"threads_per_rank"}},
      {"io", {"source", "region", "time_start", "time_end"}},
      {"output", {"outfile_prefix", "run_log", "directory", "if_exists", "verbosity", "trace"}},
  };
  return schema;
}

// Levenshtein, for "did you mean". Small strings; the naive two-row version is plenty.
int edit_distance(const std::string& a, const std::string& b) {
  std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
  for (size_t j = 0; j <= b.size(); j++) prev[j] = static_cast<int>(j);
  for (size_t i = 1; i <= a.size(); i++) {
    cur[0] = static_cast<int>(i);
    for (size_t j = 1; j <= b.size(); j++)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    prev = cur;
  }
  return prev[b.size()];
}

// Walk the tree and collect EVERY unrecognised key, with its full dotted path.
void collect_unknown_keys(const YAML::Node& node, const std::string& path, std::vector<std::string>& errs) {
  const auto& schema = config_schema();
  const auto it = schema.find(path);
  if (it == schema.end()) {
    // A mapping we have no dictionary for. This is a bug in the table above, not in the user's file, and
    // saying so is better than silently accepting anything nested under it.
    errs.push_back("  " + path + ": (internal) no schema entry for this section; the dictionary in "
                                 "src/parameters.cpp is incomplete");
    return;
  }
  for (const auto& kv : node) {
    const std::string key  = kv.first.as<std::string>();
    const std::string full = path.empty() ? key : path + "." + key;
    if (!it->second.count(key)) {
      // Suggest the nearest sibling if it is close enough to be a plausible typo.
      std::string best;
      int best_d = 1000;
      for (const auto& cand : it->second) {
        const int d = edit_distance(key, cand);
        if (d < best_d) { best_d = d; best = cand; }
      }
      std::string msg = "  unknown key '" + full + "'";
      if (best_d <= 3 && best_d < static_cast<int>(key.size()))
        msg += "  -- did you mean '" + (path.empty() ? best : path + "." + best) + "'?";
      msg += "\n      known keys in " + (path.empty() ? std::string("<top level>") : "'" + path + "'") + ": ";
      bool first = true;
      for (const auto& cand : it->second) { msg += (first ? "" : ", ") + cand; first = false; }
      errs.push_back(msg);
      continue;  // do not descend into an unknown section; its children would all be noise
    }
    if (kv.second.IsMap()) collect_unknown_keys(kv.second, full, errs);
  }
}

// REJECT unrecognised keys instead of ignoring them. yaml-cpp reads by lookup, so anything not looked up
// is simply never seen: a typo, a key retired by a schema migration, or a setting a user believes is in
// force all behave identically to not writing them at all, and the run proceeds and reports success.
//
// That is worse than a lost setting, because it makes NEGATIVE results untrustworthy. A parameter sweep
// over a key nothing reads returns "no effect" for a reason that has nothing to do with the model, and it
// is indistinguishable from a real finding. Two of those happened here: `total_cycles` (retired when the
// schema went nested) sat in ten benchmark scripts doing nothing, and a controller sweep reported
// byte-identical results because the flag it varied was parsed on a different code path.
//
// Failing loudly costs a user one clear error message; accepting silently costs whoever has to work out
// later why a documented experiment cannot be reproduced.
void validate_config_keys(const YAML::Node& root, const std::string& config_file) {
  std::vector<std::string> errs;
  collect_unknown_keys(root, "", errs);
  if (errs.empty()) return;
  std::string msg = "config file '" + config_file + "' has " + std::to_string(errs.size())
                    + (errs.size() == 1 ? " unrecognised key:\n" : " unrecognised keys:\n");
  for (const auto& e : errs) msg += e + "\n";
  msg += "\nEvery key is checked against the schema in src/parameters.cpp; see config.yaml for the full\n"
         "documented set. If a key was recently renamed, the old spelling is gone rather than ignored --\n"
         "this abort exists so that a setting you wrote can never be silently doing nothing.";
  throw std::runtime_error(msg);
}

}  // namespace

const std::string& require_enum(const std::string& value, const char* key,
                                std::initializer_list<const char*> allowed) {
  for (const char* a : allowed)
    if (value == a) return value;
  std::string list;
  for (const char* a : allowed) { if (!list.empty()) list += " | "; list += a; }
  throw std::runtime_error("config: " + std::string(key) + " must be " + list + ", got '" + value + "'");
}

// Real initializer
Parameters::Parameters(const std::string& config_file) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_file);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to read config file '" + config_file + "': " + e.what());
  }
  if (!root.IsMap()) {
    throw std::runtime_error("config file '" + config_file + "' is not a YAML mapping. The config format is "
                             "nested YAML with sections (run / time / io / output / transmissivity / "
                             "surface_water / solver / ...) -- see config.yaml. (A legacy 'key value' .cfg "
                             "will trip this.)");
  }
  // Before reading anything: reject keys the model does not understand, so a typo or a retired key can
  // never sit in a config quietly doing nothing. See validate_config_keys.
  validate_config_keys(root, config_file);

  // Each key is read only if present; an absent key keeps the member's default, and check() below enforces
  // the ones that must be set (matching the previous parser's behavior). Chained operator[] on an absent
  // parent yields an undefined node (no mutation), so root["a"]["b"] is safe even when "a" is missing.

  // NOTE (Phase 2 hard cutover to the config_flags_prototype.yaml schema). This handles the MEMBER-backed
  // keys (parsed straight into Parameters). The CLI-flag-backed sections -- solver, parallel, dev, boundaries,
  // transmissivity.additive_background_transmissivity, evaporation (et_sigmoid/extinction_depth),
  // surface_water.collection.sink, run.equilibrium_stop, output.verbosity/if_exists/directory -- are not read
  // here yet; they remain -wtm_* / PETSc CLI flags until the YAML->PetscOptions bridge lands (Phase 2b).

  if (auto n = root["run"]["equilibrium_stop"]["tol"])    { eq_tol = n.as<double>(); eq_tol_set = true; }
  // max | rms | frac, plus three RETIRED spellings that the consumer maps with a NOTE (all metrics judge
  // water now). Without this, an unrecognised metric fell through to "frac" and the run reported success.
  if (auto n = root["run"]["equilibrium_stop"]["metric"])
    eq_metric = require_enum(n.as<std::string>(), "run.equilibrium_stop.metric",
                             {"max", "rms", "frac", "water", "water-max", "water-rms"});
  if (auto n = root["boundaries"]["land"])
    land_boundary_dirichlet = (require_enum(n.as<std::string>(), "boundaries.land",
                                            {"neumann_toposlope", "dirichlet_sea_level"})
                               == "dirichlet_sea_level");
  // dev.storage_form: DEFAULT volume. Because the default is volume, `volume_storage == false` can only
  // mean the user explicitly asked for secant -- which is what lets the active-set check below abort on an
  // EXPLICIT request without needing a companion _set boolean.
  // `auto` IS NOT A VALUE (2026-09-07). Omitting a key already means "take the default"; a WORD that
  // means the same thing is a second way to say one thing, and it is the one that leaked into the
  // resolved config -- where it recorded the QUESTION instead of the answer, so a run could not be
  // reproduced from its own record if the policy moved, and no test could DECLARE it to satisfy the
  // declared==resolved rule. Refused by name here, with the migration in the message, rather than
  // failing later as an unparseable number.
  const auto refuse_auto = [](const YAML::Node& n, const std::string& key) {
    if (n.as<std::string>() == "auto")
      throw std::runtime_error(
          "config: " + key + ": auto is no longer accepted. OMIT the key to take the default -- absence "
          "already means auto, and the resolved value is recorded in full_config.yaml. (A test config "
          "should instead state the value it wants; see tests/config_identity.py.)");
  };

  if (auto n = root["dev"]["storage_form"])
    volume_storage = (require_enum(n.as<std::string>(), "dev.storage_form", {"volume", "secant"}) == "volume");
  if (auto n = root["solver"]["method"])
    solver_method = require_enum(n.as<std::string>(), "solver.method", {"anderson", "picard", "newton"});
  if (auto n = root["solver"]["time_integration"]) {
    refuse_auto(n, "solver.time_integration");
    time_integration = require_enum(n.as<std::string>(), "solver.time_integration",
                                    {"backward-euler", "bdf2", "tr-bdf2"});
  }
  if (auto n = root["solver"]["newton"]["dt_continuation"]) { dt_continuation = n.as<bool>(); dt_continuation_set = true; }
  // solver.method: newton implies dt-continuation unless the user explicitly declined it. Read the method
  // here rather than depending on the flag bridge, so the implication holds however the method arrives.
  // solver.time_integration: auto -- RESOLVED here, like dt_continuation, because the answer depends on
  // another key. An unset key means `auto`. It resolves PER METHOD rather than to a constant, because
  // tr-bdf2 runs only on the matrix-free Anderson path: a constant tr-bdf2 default would make a config
  // that says only `solver.method: picard` abort on a key the user never wrote, and an abort must never
  // fire on a defaulted value.
  //
  // THE TABLE.
  //   anderson -> tr-bdf2         the production path. 2nd-order, L-stable with strong damping. At
  //                               EQUILIBRIUM it cannot change the answer at all (the transient term
  //                               vanishes: three integrators agree to 1.8e-15 m on flicker_evap), so
  //                               this is a cost/robustness choice there, and an ACCURACY choice for
  //                               transient runs, where it is the 2nd-order option.
  //   picard   -> backward-euler  today's behaviour. bdf2 (BDF2-on-V) is Picard's designed 2nd-order
  //                               mode and is the alternative, but Picard is a verification oracle, not
  //                               a production path, so minimum surprise wins.
  //   newton   -> backward-euler  today's behaviour; tr-bdf2 is unavailable off the Anderson path.
  if (time_integration.empty()) {   // ABSENT means auto-resolve; the word itself is refused above
    const std::string m   = solver_method.empty() ? "anderson" : solver_method;
    time_integration      = (m == "anderson") ? "tr-bdf2" : "backward-euler";
    time_integration_auto = true;
  }
  if (!dt_continuation_set)
    if (auto n = root["solver"]["method"])
      if (n.as<std::string>() == "newton") dt_continuation = true;
  if (auto n = root["solver"]["t_bar"])       t_bar       = n.as<bool>();
  if (auto n = root["solver"]["adaptive_dt"]) {
    refuse_auto(n, "solver.adaptive_dt");
    adaptive_dt = n.as<bool>();
    adaptive_dt_set = true;
  } else {
    adaptive_dt_auto = true;  // an absent key means auto
  }
  if (auto n = root["solver"]["time_step"]["error_tol"]) {
    refuse_auto(n, "solver.time_step.error_tol");
    dt_tol = std::stod(n.as<std::string>());
    dt_tol_set = true;
  }
  if (auto n = root["solver"]["time_step"]["dt_max"]) {
    refuse_auto(n, "solver.time_step.dt_max");
    dtc_dt_max = parse_time_seconds(n.as<std::string>(), "solver.time_step.dt_max");
    dtc_dt_max_set = true;
  }
  if (auto n = root["evaporation"]["extinction_depth"]) extinction_depth = n.as<double>();
  if (auto n = root["evaporation"]["tapers"]["surface_transition"]) taper_surface_transition = n.as<bool>();
  if (auto n = root["evaporation"]["tapers"]["depth_extinction"])   taper_depth_extinction   = n.as<bool>();
  if (auto n = root["run"]["equilibrium_stop"]["frac"])  eq_frac          = n.as<double>();

  // -------- transmissivity (config-owned; formerly -wtm_ transport only) --------
  if (auto n = root["transmissivity"]["additive_background_transmissivity"]) t_bedrock = n.as<double>();

  // -------- evaporation --------
  if (auto n = root["evaporation"]["et_sigmoid"]["wtd_center"])     evap_taper_wtdc = n.as<double>();
  if (auto n = root["evaporation"]["et_sigmoid"]["logistic_width"]) evap_taper_s    = n.as<double>();

  // -------- run --------
  if (auto n = root["run"]["type"]) run_type = n.as<std::string>();
  // initial_water_table: "saturated" -> start at the surface (wtd = 0); any other value names a supplied
  // starting water table to load. TODO: a literal <path> should load that file, and omitting the key should
  // auto-detect a starting_wt layer in io.source; for now a non-"saturated" value selects the supplied-WT layer.
  if (auto n = root["run"]["initial_water_table"]) {
    const std::string v = n.as<std::string>();
    if (v == "saturated") {
      supplied_wt = 0;  // start at the surface (wtd = 0)
    } else if (v == "supplied") {
      supplied_wt = 1;  // read the standard starting_wt layer from io.source (initial_wt_path stays empty)
    } else {
      supplied_wt     = 1;
      initial_wt_path = v;  // <path>: load the starting water table from this file directly
    }
  }

  // -------- time --------
  // solver.time_step.deltat -- the step itself, moved out of time: because it is a SOLVER property
  // (Andy, 2026-09-03) and belongs with the dials that adjust it. time: keeps the run's clock -- how long,
  // how often to report -- which is quantised BY the step but does not choose it.
  if (auto n = root["solver"]["time_step"]["dt"]) deltat = n.as<double>();
  if (auto n = root["time"]["total"])  total_time = parse_time_seconds(n.as<std::string>(), "time.total");
  if (auto n = root["time"]["report_interval"]) {
    // A bare integer = timesteps, or a simulated time ("50yr"/"1000s"). Resolved to report_steps /
    // report_seconds below, once deltat is known.
    const std::string v = n.as<std::string>();
    if (v.size() > 2 && v.substr(v.size() - 2) == "yr") {
      report_interval_is_time = true;
      report_interval_time    = std::stod(v.substr(0, v.size() - 2)) * 31536000.0;
    } else if (v.size() > 1 && v.back() == 's'
               && std::isdigit(static_cast<unsigned char>(v[v.size() - 2]))) {
      report_interval_is_time = true;
      report_interval_time    = std::stod(v.substr(0, v.size() - 1));
    } else {
      report_steps = std::stoi(v);
    }
  }
  if (auto n = root["time"]["save_every_n_reports"]) save_nreport_interval = n.as<int32_t>();

  // -------- grid: DEPRECATED (override only; geometry derives from the GDAL geotransform, #124) --------

  // -------- transmissivity (was physics.fdepth) --------
  if (auto n = root["transmissivity"]["fdepth"]["a"])    fdepth_a    = n.as<double>();
  if (auto n = root["transmissivity"]["fdepth"]["b"])    fdepth_b    = n.as<double>();
  if (auto n = root["transmissivity"]["fdepth"]["fmin"]) fdepth_fmin = n.as<double>();

  // -------- surface_water --------
  // mode: routed = FillSpillMerge routes above-ground water; ponded/removed do not route it. (The ponded-vs-
  // removed distinction is a dev-flag detail -- TODO.) Replaces the old fsm bool.
  if (auto n = root["surface_water"]["mode"]) {
    const std::string m = n.as<std::string>();
    if (m == "routed")                        fsm_on = 1;
    else if (m == "ponded" || m == "removed") fsm_on = 0;
    else throw std::runtime_error("config: surface_water.mode must be 'routed', 'ponded', or 'removed', got '" + m + "'");
  }
  // runoff_ratio: a number in [0,1] = a uniform ratio everywhere; the string "raster" = require the
  // runoff_ratio raster from io.source; omitted = off. (TODO: omit -> auto-detect the raster if present.)
  if (auto n = root["surface_water"]["runoff_ratio"]) {
    const std::string v = n.as<std::string>();
    if (v == "raster") {
      runoff_ratio_on = 1;  // require the raster (runoff_ratio_uniform stays < 0)
    } else {
      double r;
      try {
        r = n.as<double>();
      } catch (...) {
        throw std::runtime_error(
            "config: surface_water.runoff_ratio must be a number in [0,1], 'raster', or omitted (got '" + v + "')");
      }
      if (r < 0.0 || r > 1.0)
        throw std::runtime_error("config: surface_water.runoff_ratio must be in [0,1], got " + v);
      runoff_ratio_on      = (r > 0.0) ? 1 : 0;
      runoff_ratio_uniform = r;
    }
  }
  if (auto n = root["surface_water"]["infiltration_during_flow"]) infiltration_on = n.as<bool>() ? 1 : 0;
  if (auto n = root["surface_water"]["collection"]["method"]) {
    runoff_collector     = n.as<std::string>();
    runoff_collector_set = true;
  }

  // Resolve adaptive_dt against dt_continuation. They are two controllers for ONE question -- who sizes
  // the step -- and WTM.cpp:593 is `if (use_dt_adaptive) ... else if (use_newton_continuation)`, so
  // adaptive silently WINS and the ramp never runs, after InitialiseSNES has already printed its banner.
  // Measured on tests/dt_sensitivity inputs with solver.method: newton: "adaptive dt: 30 steps" instead
  // of "dt-continuation: deltat now 5.24e+08 s". Newton needs that ramp to converge from cold, which
  // tests/newton_solver's CONTRACT arm asserts, so this must never resolve silently.
  if (adaptive_dt_set && adaptive_dt && dt_continuation && dt_continuation_set)
    throw std::runtime_error(
        "config: solver.adaptive_dt: true and solver.newton.dt_continuation: true both control the step "
        "size, and only one can. The adaptive loop takes precedence and the continuation ramp would never "
        "run -- silently, before this check. Choose one: solver.adaptive_dt: false to keep Newton's ramp "
        "(the recipe it needs to converge from cold), or solver.newton.dt_continuation: false to let the "
        "adaptive controller size the step for plain Newton.");
  // ...and against the COLLECTOR. `implicit` siphons above-surface water at rate max(0,wtd)/dt, so its
  // per-step error GROWS as the controller shrinks dt: the founding assumption of error-controlled
  // stepping -- refine dt, reduce error -- is false for it, and the reject/retry loop cannot converge.
  // It does not misbehave subtly, it DIES: "adaptive dt: step failed after max retries; -wtm_dt_tol too
  // tight or the local stability ceiling is below the smallest tried dt." (tests/active_set, imp_plain.)
  // Andy, 2026-09-03: implicit + adaptive_dt is not allowed; active_set + adaptive_dt is the pairing that
  // works and is the production default.
  const bool implicit_collector = (runoff_collector == "implicit");
  if (adaptive_dt_set && adaptive_dt && implicit_collector)
    throw std::runtime_error(
        "config: solver.adaptive_dt: true cannot be used with surface_water.collection.method: implicit. "
        "The implicit siphon removes above-surface water at rate max(0,wtd)/dt, so its per-step error "
        "GROWS as the controller shrinks dt -- no step can ever be accepted, and the run dies with "
        "'step failed after max retries'. Use collection.method: active_set (the default, and the "
        "enforcement adaptive stepping is built for), or set solver.adaptive_dt: false.");
  if (adaptive_dt_auto) {
    // auto YIELDS twice over: to Newton's ramp where that owns the step size, and to fixed stepping
    // under the implicit collector, which adaptive cannot drive at all.
    adaptive_dt = !dt_continuation && !implicit_collector;
  } else if (adaptive_dt && dt_continuation) {
    // Explicit adaptive against an IMPLIED ramp (solver.method: newton implies it). The explicit value
    // wins over the defaulted one -- but it is not allowed to do so silently, because it strips Newton of
    // the ramp. CreateSNES announces it and the existing plain-Newton divergence warning still fires.
    dt_continuation = false;
    adaptive_dt_disabled_continuation = true;
  }

  // -------- io (source was surfdatadir; outfile/log moved to output) --------
  if (auto n = root["io"]["source"])     surfdatadir = n.as<std::string>();
  if (auto n = root["io"]["region"])     region      = n.as<std::string>();
  if (auto n = root["io"]["time_start"]) time_start  = n.as<std::string>();
  if (auto n = root["io"]["time_end"])   time_end    = n.as<std::string>();

  // -------- parallel --------
  // Read here as well as in apply_config_petsc_options (which calls omp_set_num_threads and is #ifdef
  // _OPENMP): the RESOLVED value has to be reportable in full_config.yaml on every build.
  if (auto n = root["parallel"]["threads_per_rank"]) threads_per_rank = n.as<int>();

  // -------- output (was io.outfile_prefix / io.textfilename) --------
  if (auto n = root["output"]["outfile_prefix"]) outfile_prefix = n.as<std::string>();
  if (auto n = root["output"]["run_log"])        textfilename   = n.as<std::string>();
  if (auto n = root["output"]["verbosity"]) {
    verbosity = n.as<std::string>();
    if (verbosity != "quiet" && verbosity != "normal" && verbosity != "verbose")
      throw std::runtime_error("config: output.verbosity must be quiet | normal | verbose, got '" + verbosity + "'");
  }
  if (auto n = root["output"]["directory"]) output_directory = n.as<std::string>();
  if (auto n = root["output"]["if_exists"]) {
    if_exists = n.as<std::string>();
    if (if_exists != "increment" && if_exists != "overwrite" && if_exists != "error")
      throw std::runtime_error("config: output.if_exists must be increment | overwrite | error, got '" + if_exists + "'");
  }

  // Resolve the report cadence now that deltat is parsed. FSM runs EVERY timestep; report_interval is ONLY the
  // equilibrium-check + log/output cadence. Explicit report_interval (steps or time), else default 100 steps
  // (with a loud warning).
  if (report_interval_is_time) {
    report_seconds = report_interval_time;
    report_steps   = std::max<int32_t>(1, static_cast<int32_t>(std::llround(report_seconds / deltat)));
  } else if (report_steps > 0) {
    report_seconds = report_steps * deltat;
  } else {
    report_steps   = 100;
    report_seconds = report_steps * deltat;
    std::cerr << "WARNING [report_interval]: NOT SET -- defaulting to 100 steps between equilibrium checks / "
                 "reports. Set it explicitly, as steps (e.g. 'report_interval 100') or a time (e.g. "
                 "'report_interval 50yr').\n";
  }
  // Resolve the total run length. total_time is the canonical user input; the loop advances one report at a
  // time, so total_time must be an exact integer number of reports (report_seconds each). Error otherwise --
  // this is the "integer multiple of the time step" guard, tightened to the report span the loop actually
  // takes (report_seconds = report_steps*deltat, so an integer number of reports is also an integer number of
  // timesteps).
  if (!(total_time > 0.0)) {
    throw std::runtime_error("total_time must be set to a positive simulated time, e.g. 'total_time 500yr'.");
  }
  {
    const double reports_exact = total_time / report_seconds;
    const double reports_round = std::round(reports_exact);
    if (std::abs(reports_exact - reports_round) > 1e-6 * std::max(1.0, reports_round)) {
      throw std::runtime_error(fmt::format(
          "total_time ({} s) is not an integer multiple of the report interval ({} s = {} timesteps of {} s): "
          "{} reports. Adjust total_time, report_interval, or deltat so they divide evenly.",
          total_time, report_seconds, report_steps, deltat, reports_exact));
    }
    total_reports = static_cast<int32_t>(reports_round);
  }
  // Resolve the raster-save cadence (every K reports); default 1 (with a loud warning).
  if (save_nreport_interval <= 0) {
    save_nreport_interval = 1;
    std::cerr << "WARNING [save_nreport_interval]: NOT SET -- defaulting to 1 (save a raster every report). "
                 "Set it explicitly.\n";
  }

  check();
}

void Parameters::check() const {
  const auto check_positive = [](const std::string name, const auto val) {
    if (std::isnan(val) || val < 0) {
      throw std::runtime_error("Please enter a positive value for " + name);
    }
  };

  const auto check_string_init = [&](const std::string name, const std::string& val) {
    if (val == UNINIT_STR) {
      throw std::runtime_error("Please provide a value for " + name);
    }
  };

  const auto check_binary = [](const auto val, const std::string& msg) {
    if (val != 0 && val != 1) {
      throw std::runtime_error(msg);
    }
  };

  check_positive("save_nreport_interval", save_nreport_interval);
  check_positive("deltat", deltat);
  // Grid geometry is NOT validated here any more: cells_per_degree and southern_edge are no longer
  // inputs. They are DERIVED from the input raster's geotransform (grid_geometry.cpp), which validates
  // what it reads -- north-up, geographic CRS, a geotransform present at all -- at the point of reading.
  // Validating a derived value here would be checking our own arithmetic against the user's mistake.
  // evap_mode is no longer a config key (dropped in the Phase-2 schema; vestigial when the ET sigmoid is on,
  // which is the default). It keeps its member default and is not validated here.
  check_positive("fdepth_a", fdepth_a);
  check_positive("fdepth_b", fdepth_b);
  check_positive("fdepth_fmin", fdepth_fmin);
  check_binary(
      fsm_on, "set fsm_on to 1 to allow Fill-Spill-Merge to move surface water, or 0 to disable Fill-Spill-Merge.");
  check_binary(
      infiltration_on,
      "set infiltration_on to 1 to allow water to infiltrate as it flows downslope, or 0 to neglect infiltration and "
      "assume impermeable substrates while flowing downslope.");
  check_binary(
      runoff_ratio_on,
      "set runoff_ratio_on to 1 to supply a runoff ratio array, or 0 to assume all P-ET infiltrates in the cell "
      "where it falls.");
  check_binary(
      supplied_wt,
      "set supplied_wt to 1 to supply a starting water table, or 0 to set starting water table == 0 (only available "
      "for equilibrium runs).");
  check_positive("report_steps", report_steps);
  check_string_init("outfile_prefix", outfile_prefix);
  check_string_init("region", region);
  check_string_init("run_type", run_type);
  check_string_init("surfdatadir", surfdatadir);
  check_string_init("textfilename", textfilename);
  check_string_init("time_start", time_start);
  check_string_init("time_end", time_end);
  check_positive("total_time", total_time);
  check_positive("total_reports", total_reports);
  // `extended_soil` is a member of this enumeration rather than a separate flag ON PURPOSE. It is a
  // choice about what happens to water at and above the land surface, which is exactly what this
  // selector decides, and it is a sibling of `off`: both let water pile up instead of enforcing
  // wtd<=0, and they differ in the physics ABOVE the surface (`off` keeps the standard jump to
  // storativity 1 and the T clamp; `extended_soil` continues the aquifer, storativity stays at
  // porosity and T never clamps). Held as one enum value, "extended soil AND a collector" is
  // unrepresentable rather than merely detected -- previously they were independent switches, both
  // clamped, and the collector silently won, so -wtm_extended_soil printed its mode banner while
  // doing nothing. See the resolution block in transient_groundwater.cpp.
  if (!(runoff_collector == "" || runoff_collector == "implicit" || runoff_collector == "explicit"
        || runoff_collector == "active_set" || runoff_collector == "off"
        || runoff_collector == "extended_soil")) {
    throw std::runtime_error(
        "runoff_collector must be one of: active_set, implicit, explicit, off, extended_soil. Got: '"
        + runoff_collector + "'");
  }
}

std::string Parameters::get_path(const std::string& time, const std::string& layer_name) const {
  constexpr auto SURF_DATA_PATH_FORMAT = "{}/{}_{}_{}.tif";
  return fmt::format(SURF_DATA_PATH_FORMAT, surfdatadir, region, time, layer_name);
}

std::string Parameters::get_path(const std::string& layer_name) const {
  constexpr auto SURF_DATA_PATH_FORMAT = "{}/{}_{}.tif";
  return fmt::format(SURF_DATA_PATH_FORMAT, surfdatadir, region, layer_name);
}

// RESOLVED-CONFIG ECHO. Every `c <key> = <value>` line reports the value the run is ACTUALLY using --
// after config parsing, after defaults are applied, and after the geometry is derived from the input
// geotransform -- so a log can answer "what did this run do?" without re-deriving it from the source.
// Call once on rank 0, after initialise() (report_steps / total_reports / the grid geometry are set
// there, and output.directory has by then rewritten outfile_prefix and textfilename).
//
// KEEP THIS IN SYNC with the fields of Parameters. It went dead once -- nothing called it -- and drifted
// behind the config walk's new keys while runs quietly logged nothing; that gap turned a silently
// overridden setting into a wrong conclusion. A new config key belongs here in the same commit.
void Parameters::print() const {
  std::cout << "c --- resolved configuration ---" << std::endl;
  std::cout << "c run_type               = " << run_type << std::endl;
  std::cout << "c region                 = " << region << std::endl;
  std::cout << "c surfdatadir            = " << surfdatadir << std::endl;
  std::cout << "c time_start             = " << time_start << std::endl;
  std::cout << "c time_end               = " << time_end << std::endl;
  std::cout << "c initial_water_table    = "
            << (initial_wt_path.empty() ? (supplied_wt ? "supplied (from file)" : "saturated (wtd = 0)")
                                        : initial_wt_path)
            << std::endl;
  // Time stepping.
  std::cout << "c deltat                 = " << deltat << std::endl;
  std::cout << "c report_steps           = " << report_steps << std::endl;
  std::cout << "c report_seconds         = " << report_seconds << std::endl;
  std::cout << "c total_time (s)         = " << total_time << std::endl;
  std::cout << "c total_reports          = " << total_reports << std::endl;
  std::cout << "c save_nreport_interval  = " << save_nreport_interval << std::endl;
  // Grid geometry, ALL DERIVED from the topography's geotransform (#124) -- none of it is a config key.
  // ns/ew_deg_per_cell are what the run actually uses; cells_per_degree is the nominal 1/ns_deg_per_cell,
  // kept because the run log and older scripts refer to it.
  std::cout << "c ncells_x, ncells_y     = " << ncells_x << ", " << ncells_y << std::endl;
  std::cout << "c ns_deg_per_cell        = " << ns_deg_per_cell << std::endl;
  std::cout << "c ew_deg_per_cell        = " << ew_deg_per_cell << std::endl;
  std::cout << "c cells_per_degree       = " << cells_per_degree << " (nominal = 1/ns_deg_per_cell)" << std::endl;
  std::cout << "c southern_edge          = " << southern_edge << std::endl;
  std::cout << "c cellsize_n_s_metres    = " << cellsize_n_s_metres << std::endl;
  // Transmissivity.
  std::cout << "c fdepth_a               = " << fdepth_a << std::endl;
  std::cout << "c fdepth_b               = " << fdepth_b << std::endl;
  std::cout << "c fdepth_fmin            = " << fdepth_fmin << std::endl;
  // Surface water. runoff_collector is the SELECTOR: it supersedes the
  // legacy -wtm_ surface flags, so this line -- not the command line -- says which exfiltration
  // enforcement ran. See transient_groundwater.cpp (the selector block) and SURFACE_WATER_ROUTING.md.
  std::cout << "c fsm_on                 = " << fsm_on << std::endl;
  std::cout << "c runoff_collector       = " << (runoff_collector.empty() ? "implicit (default)" : runoff_collector)
            << std::endl;
  std::cout << "c runoff_ratio_on        = " << runoff_ratio_on << std::endl;
  if (runoff_ratio_uniform >= 0.0)
    std::cout << "c runoff_ratio_uniform   = " << runoff_ratio_uniform << std::endl;
  std::cout << "c infiltration_on        = " << infiltration_on << std::endl;
  std::cout << "c evap_mode              = " << evap_mode << std::endl;
  // Output.
  std::cout << "c output_directory       = " << (output_directory.empty() ? "(legacy: literal paths)" : output_directory)
            << std::endl;
  std::cout << "c if_exists              = " << if_exists << std::endl;
  std::cout << "c outfile_prefix         = " << outfile_prefix << std::endl;
  std::cout << "c textfilename           = " << textfilename << std::endl;
  std::cout << "c verbosity              = " << verbosity << std::endl;
  std::cout << "c --- end configuration ---" << std::endl;
}
