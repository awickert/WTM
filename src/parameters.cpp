#include "parameters.hpp"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

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
                             "surface_water / solver / ...) -- see Config_file.yaml. (A legacy 'key value' .cfg "
                             "will trip this.)");
  }

  // Each key is read only if present; an absent key keeps the member's default, and check() below enforces
  // the ones that must be set (matching the previous parser's behavior). Chained operator[] on an absent
  // parent yields an undefined node (no mutation), so root["a"]["b"] is safe even when "a" is missing.

  // NOTE (Phase 2 hard cutover to the config_flags_prototype.yaml schema). This handles the MEMBER-backed
  // keys (parsed straight into Parameters). The CLI-flag-backed sections -- solver, parallel, dev, boundaries,
  // transmissivity.additive_background_transmissivity, evaporation (et_sigmoid/extinction_depth),
  // surface_water.collection.sink, run.equilibrium_stop, output.verbosity/if_exists/directory -- are not read
  // here yet; they remain -wtm_* / PETSc CLI flags until the YAML->PetscOptions bridge lands (Phase 2b).

  // -------- run --------
  if (auto n = root["run"]["type"]) run_type = n.as<std::string>();
  // initial_water_table: "saturated" -> start at the surface (wtd = 0); any other value names a supplied
  // starting water table to load. TODO: a literal <path> should load that file, and omitting the key should
  // auto-detect a starting_wt layer in io.source; for now a non-"saturated" value selects the supplied-WT layer.
  if (auto n = root["run"]["initial_water_table"]) {
    const std::string v = n.as<std::string>();
    if (v == "saturated") {
      supplied_wt = 0;  // start at the surface (wtd = 0)
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
  if (auto n = root["surface_water"]["collection"]["method"])     runoff_collector = n.as<std::string>();

  // -------- io (source was surfdatadir; outfile/log moved to output) --------
  if (auto n = root["io"]["source"])     surfdatadir = n.as<std::string>();
  if (auto n = root["io"]["region"])     region      = n.as<std::string>();
  if (auto n = root["io"]["time_start"]) time_start  = n.as<std::string>();
  if (auto n = root["io"]["time_end"])   time_end    = n.as<std::string>();

  // -------- output (was io.outfile_prefix / io.textfilename) --------
  if (auto n = root["output"]["outfile_prefix"]) outfile_prefix = n.as<std::string>();
  if (auto n = root["output"]["run_log"])        textfilename   = n.as<std::string>();

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
  if (!(runoff_collector == "" || runoff_collector == "implicit" || runoff_collector == "explicit"
        || runoff_collector == "off" || runoff_collector == "legacy")) {
    throw std::runtime_error("runoff_collector must be one of: implicit, explicit, off, legacy. Got: '"
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

void Parameters::print() const {
  std::cout << "c cells_per_degree       = " << cells_per_degree << std::endl;
  std::cout << "c deltat                 = " << deltat << std::endl;
  std::cout << "c evap_mode              = " << evap_mode << std::endl;
  std::cout << "c fdepth_a               = " << fdepth_a << std::endl;
  std::cout << "c fdepth_b               = " << fdepth_b << std::endl;
  std::cout << "c fdepth_fmin            = " << fdepth_fmin << std::endl;
  std::cout << "c fsm_on                 = " << fsm_on << std::endl;
  std::cout << "c infiltration_on        = " << infiltration_on << std::endl;
  std::cout << "c report_steps           = " << report_steps << std::endl;
  std::cout << "c report_seconds         = " << report_seconds << std::endl;
  std::cout << "c save_nreport_interval  = " << save_nreport_interval << std::endl;
  std::cout << "c outfile_prefix         = " << outfile_prefix << std::endl;
  std::cout << "c region                 = " << region << std::endl;
  std::cout << "c run_type               = " << run_type << std::endl;
  std::cout << "c runoff_ratio_on        = " << runoff_ratio_on << std::endl;
  std::cout << "c runoff_collector       = " << (runoff_collector.empty() ? "(unset: legacy flags)" : runoff_collector) << std::endl;
  std::cout << "c southern_edge          = " << southern_edge << std::endl;
  std::cout << "c supplied_wt            = " << supplied_wt << std::endl;
  std::cout << "c surfdatadir            = " << surfdatadir << std::endl;
  std::cout << "c textfilename           = " << textfilename << std::endl;
  std::cout << "c time_end               = " << time_end << std::endl;
  std::cout << "c time_start             = " << time_start << std::endl;
  std::cout << "c total_time (s)         = " << total_time << std::endl;
  std::cout << "c total_reports          = " << total_reports << std::endl;
}
