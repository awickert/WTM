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
      {"", {"run", "time", "grid", "transmissivity", "surface_water", "evaporation", "boundaries", "solver",
            "dev", "parallel", "io", "output"}},
      {"run", {"type", "initial_water_table", "equilibrium_stop"}},
      {"run.equilibrium_stop", {"tol", "metric", "frac"}},
      {"time", {"deltat", "total", "report_interval", "save_every_n_reports"}},
      {"grid", {"cells_per_degree", "southern_edge"}},
      {"transmissivity", {"fdepth", "additive_background_transmissivity"}},
      {"transmissivity.fdepth", {"a", "b", "fmin"}},
      {"surface_water", {"mode", "runoff_ratio", "infiltration_during_flow", "collection"}},
      {"surface_water.collection", {"method", "sink"}},
      {"surface_water.collection.sink",
       {"qmax", "width", "fringe_source", "fringe_cap", "fringe_ksat_coef", "fringe_length"}},
      {"evaporation", {"et_sigmoid", "extinction_depth"}},
      {"evaporation.et_sigmoid", {"wtd_center", "logistic_width"}},
      {"boundaries", {"land"}},
      {"solver", {"method", "tolerance", "max_iterations", "time_integration", "adaptive_dt", "dt_max",
                  "water_volume_timestep_error_tol", "t_bar", "storage"}},
      {"dev", {"active_set", "allow_aboveground_water_columns", "padded_dirichlet"}},
      {"parallel", {"threads_per_rank"}},
      {"io", {"source", "region", "time_start", "time_end"}},
      {"output", {"outfile_prefix", "run_log", "directory", "if_exists", "verbosity"}},
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
  if (auto n = root["run"]["equilibrium_stop"]["metric"]) eq_metric = n.as<std::string>();
  if (auto n = root["surface_water"]["collection"]["sink"]["fringe_source"])
    fringe_source = n.as<std::string>();
  if (auto n = root["solver"]["t_bar"])       t_bar       = n.as<bool>();
  if (auto n = root["solver"]["adaptive_dt"]) adaptive_dt = n.as<bool>();
  if (auto n = root["solver"]["water_volume_timestep_error_tol"]) {
    const std::string v = n.as<std::string>();
    if (v != "auto") { dt_tol = std::stod(v); dt_tol_set = true; }
  }
  if (auto n = root["solver"]["dt_max"]) {
    const std::string v = n.as<std::string>();
    if (v != "auto") { dtc_dt_max = parse_time_seconds(v, "solver.dt_max"); dtc_dt_max_set = true; }
  }
  if (auto n = root["evaporation"]["extinction_depth"]) extinction_depth = n.as<double>();
  if (auto n = root["run"]["equilibrium_stop"]["frac"])  eq_frac          = n.as<double>();
  if (auto n = root["surface_water"]["collection"]["sink"]["qmax"])  surface_sink_qmax = n.as<double>();
  if (auto n = root["surface_water"]["collection"]["sink"]["width"])
    { surface_sink_width = n.as<double>(); surface_sink_width_set = true; }

  // -------- transmissivity / surface-water sink (config-owned; formerly -wtm_ transport only) --------
  if (auto n = root["transmissivity"]["additive_background_transmissivity"]) t_bedrock = n.as<double>();
  if (auto n = root["surface_water"]["collection"]["sink"]["fringe_length"])
    { if (!n.IsNull()) fringe_length = n.as<double>(); }
  if (auto n = root["surface_water"]["collection"]["sink"]["fringe_ksat_coef"])
    { if (!n.IsNull()) fringe_ksat_coef = n.as<double>(); }
  if (auto n = root["surface_water"]["collection"]["sink"]["fringe_cap"])
    { if (!n.IsNull()) fringe_cap = n.as<double>(); }

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
  if (auto n = root["time"]["deltat"]) deltat = n.as<double>();
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
  if (auto n = root["grid"]["cells_per_degree"]) cells_per_degree = n.as<double>();
  if (auto n = root["grid"]["southern_edge"])    southern_edge    = n.as<double>();

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

  // -------- io (source was surfdatadir; outfile/log moved to output) --------
  if (auto n = root["io"]["source"])     surfdatadir = n.as<std::string>();
  if (auto n = root["io"]["region"])     region      = n.as<std::string>();
  if (auto n = root["io"]["time_start"]) time_start  = n.as<std::string>();
  if (auto n = root["io"]["time_end"])   time_end    = n.as<std::string>();

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
  // Grid geometry (cells_per_degree / southern_edge) is now derived from the geotransform (#124), so it is
  // no longer required here. Validate only when supplied as the deprecated override.
  if (cells_per_degree != -1) {
    check_positive("cells_per_degree", cells_per_degree);
  }
  if (!std::isnan(southern_edge) && (southern_edge < -90 || southern_edge > 90)) {
    throw std::runtime_error("please enter a value between -90 and 90 degrees for the southern_edge!");
  }
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
        || runoff_collector == "active_set" || runoff_collector == "off" || runoff_collector == "legacy"
        || runoff_collector == "extended_soil")) {
    throw std::runtime_error(
        "runoff_collector must be one of: active_set, implicit, explicit, off, legacy, extended_soil. Got: '"
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
  // Grid geometry: cells_per_degree is the DEPRECATED override; ns/ew_deg_per_cell are what the run uses
  // when the geometry comes from the topography's geotransform (#124).
  std::cout << "c ncells_x, ncells_y     = " << ncells_x << ", " << ncells_y << std::endl;
  std::cout << "c ns_deg_per_cell        = " << ns_deg_per_cell << std::endl;
  std::cout << "c ew_deg_per_cell        = " << ew_deg_per_cell << std::endl;
  std::cout << "c cells_per_degree       = " << cells_per_degree << " (deprecated override)" << std::endl;
  std::cout << "c southern_edge          = " << southern_edge << std::endl;
  std::cout << "c cellsize_n_s_metres    = " << cellsize_n_s_metres << std::endl;
  // Transmissivity.
  std::cout << "c fdepth_a               = " << fdepth_a << std::endl;
  std::cout << "c fdepth_b               = " << fdepth_b << std::endl;
  std::cout << "c fdepth_fmin            = " << fdepth_fmin << std::endl;
  // Surface water. runoff_collector is the SELECTOR: when it is anything but "legacy" it supersedes the
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
