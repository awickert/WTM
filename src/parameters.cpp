#include "parameters.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Real initializer
Parameters::Parameters(const std::string& config_file) {
  std::ifstream fin(config_file);

  if (!fin.good()) {
    throw std::runtime_error("Failed to read config file!");
  }

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string key;
    ss >> key;

    // Skip blank lines and '#' comment lines (so a self-documenting config like Config_file.cfg parses).
    // Dummy key to make it easier to alphabetize list below.
    if (key.empty() || key[0] == '#') {
    } else if (key == "cells_per_degree") {
      ss >> cells_per_degree;
    } else if (key == "deltat") {
      ss >> deltat;
    } else if (key == "evap_mode") {
      ss >> evap_mode;
    } else if (key == "fdepth_a") {
      ss >> fdepth_a;
    } else if (key == "fdepth_b") {
      ss >> fdepth_b;
    } else if (key == "fdepth_fmin") {
      ss >> fdepth_fmin;
    } else if (key == "fsm_on") {
      ss >> fsm_on;
    } else if (key == "infiltration_on") {
      ss >> infiltration_on;
    } else if (key == "outfile_prefix") {
      ss >> outfile_prefix;
    } else if (key == "region") {
      ss >> region;
    } else if (key == "run_type") {
      ss >> run_type;
    } else if (key == "runoff_ratio_on") {
      ss >> runoff_ratio_on;
    } else if (key == "runoff_collector") {
      ss >> runoff_collector;
    } else if (key == "southern_edge") {
      ss >> southern_edge;
    } else if (key == "supplied_wt") {
      ss >> supplied_wt;
    } else if (key == "surfdatadir") {
      ss >> surfdatadir;
    } else if (key == "textfilename") {
      ss >> textfilename;
    } else if (key == "time_end") {
      ss >> time_end;
    } else if (key == "time_start") {
      ss >> time_start;
    } else if (key == "total_cycles") {
      ss >> total_cycles;
    } else if (key == "report_interval") {
      std::string v;
      ss >> v;
      // Steps (a bare integer) OR a simulated time (value + unit: "50yr" or "1000s"). Resolved to
      // report_steps / report_seconds at the end of the constructor, once deltat is known.
      if (v.size() > 2 && v.substr(v.size() - 2) == "yr") {
        report_interval_is_time = true;
        report_interval_time    = std::stod(v.substr(0, v.size() - 2)) * 31536000.0;  // years -> seconds
      } else if (v.size() > 1 && v.back() == 's'
                 && std::isdigit(static_cast<unsigned char>(v[v.size() - 2]))) {
        report_interval_is_time = true;
        report_interval_time    = std::stod(v.substr(0, v.size() - 1));  // seconds
      } else {
        report_steps = std::stoi(v);  // bare integer = number of timesteps
      }
    } else if (key == "save_nreport_interval") {
      ss >> save_nreport_interval;
    } else {
      throw std::runtime_error("Unrecognised key: " + key);
    }
  }
  std::cout << infiltration_on << std::endl;

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

  check_positive("cells_per_degree", cells_per_degree);
  check_positive("save_nreport_interval", save_nreport_interval);
  check_positive("deltat", deltat);
  if (std::isnan(southern_edge) || southern_edge < -90 || southern_edge > 90) {
    throw std::runtime_error("please enter a value between -90 and 90 degrees for the southern_edge!");
  }
  check_binary(
      evap_mode,
      "set evap_mode to 0 to remove all surface water, or 1 to use a grid of potential evaporation for lakes.");
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
  check_positive("total_cycles", total_cycles);
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
  std::cout << "c total_cycles           = " << total_cycles << std::endl;
}
