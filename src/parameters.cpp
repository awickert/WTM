#include "parameters.hpp"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
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
      // surface_water.routing REPLACED surface_water.mode + surface_water.fsm_coupling (2026-09-10).
      // They were two keys answering ONE question -- what happens to above-ground water -- and the split
      // made a contradiction representable: `mode: ponded` with `fsm_coupling: continuous` asked for a
      // coupling that cannot happen, and the model resolved it silently. Worse, full_config.yaml then
      // RECORDED `impulse` for a run in which no coupling ran at all, so an FSM-off config had to declare
      // a mechanism it never used. One key, three values, and the contradiction is unrepresentable rather
      // than merely refused -- the same move as extended_soil joining collection.method (#26).
      {"surface_water", {"routing", "runoff_ratio", "infiltration_during_flow", "collection"}},
      {"surface_water.collection", {"method"}},
      {"evaporation", {"et_sigmoid", "extinction_depth", "tapers"}},
      {"evaporation.tapers", {"surface_transition", "depth_extinction"}},
      {"evaporation.et_sigmoid", {"wtd_center", "logistic_width"}},
      {"boundaries", {"land"}},
      {"solver", {"method", "tolerance", "max_iterations", "time_integration",
                  "t_bar",
                  "time_step", "smoothing", "anderson", "newton", "convergence"}},
      // solver.convergence: what the PER-SOLVE step test judges. `metric: volume` swaps the head
      // relative-step test for |S*Δwtd|, so all three "close enough" gates (this, run.equilibrium_stop
      // and solver.time_step.error_tol) finally speak the same units. water_volume_tol is read ONLY when
      // metric: volume, and sits beside it for that reason.
      {"solver.convergence", {"metric", "water_volume_tol", "residual_gate"}},
      // solver.time_step: ONE step-size controller, deliberately not nested under adaptive_dt --
      // Newton's dt_continuation ramp reads the same dials, so an `adaptive_`-prefixed home would
      // misdescribe them.
      {"solver.time_step", {"dt", "mode", "grow", "shrink", "grow_if_niter_leq", "max_retries", "norm",
                               "dt_max", "dt_min", "error_tol"}},
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
      {"solver.newton", {"dt0"}},
      // dev.active_set was REMOVED 2026-09-01: it was a SECOND YAML route to the same enforcement as
      // surface_water.collection.method: active_set, and it silently OVERRODE an explicit method (measured:
      // 54/256 cells, max 0.127 m, with no log line). One setting, one key. Removing it from this schema is
      // what makes an old config say so instead of drifting.
      {"dev", {"storage_form", "under_relaxation"}},
      {"parallel", {"threads_per_rank"}},
      {"io", {"source", "region", "time_start", "time_end"}},
      {"output", {"outfile_prefix", "run_log", "directory", "if_exists", "verbosity", "trace",
                  "extra_rasters"}},
      // Both are MAPS of name -> bool rather than lists of names, so the vocabulary ships in the
      // config file instead of having to be known. Every channel/raster is listed here, which is
      // what makes a typo an abort rather than a silently-ignored line.
      {"output.trace", {"dt", "water_step", "budget", "fsm"}},
      {"output.extra_rasters", {"post_groundwater"}},
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

// REFUSE A DUPLICATE KEY. yaml-cpp keeps ONE of them and discards the other WITHOUT A WORD, so a
// setting can be written down and silently dropped -- the exact defect class the config work exists to
// remove, arriving through the parser instead of through a second route.
//
// It is not hypothetical. Twice on 2026-09-10: tests/route_equality emitted `time_step:` twice under
// `solver:` and its `mode: fixed` vanished (the run reported `absent -> ramp` while the file said
// otherwise), and tests/flicker_evap got a second `runoff_collector` line whose last-wins assignment
// overrode the first. Both were invisible until the resulting BEHAVIOUR looked wrong.
//
// SCOPE, stated because it is not total: this scans BLOCK style -- `  key:` at a fixed indent under a
// section -- which is what every generated and hand-written config here uses. A duplicate inside a
// one-line flow map (`{a: 1, a: 2}`) is not caught. Better to catch the realistic case loudly than to
// catch nothing while appearing thorough.
static void refuse_duplicate_keys(const std::string& config_file) {
  std::ifstream in(config_file);
  if (!in) return;  // the loader below reports an unreadable file with a better message
  std::string line, section;
  std::map<std::string, std::vector<int>> seen;   // "section.key" -> line numbers
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const size_t indent = line.find_first_not_of(' ');
    if (indent == std::string::npos || indent > colon) continue;
    const std::string key = line.substr(indent, colon - indent);
    if (key.empty() || key.find_first_of(" \t{}[]\"'") != std::string::npos) continue;
    if (indent == 0) { section = key; continue; }     // a new top-level section
    if (indent != 2) continue;                        // only the section's own keys; nested maps are theirs
    seen[section + "." + key].push_back(lineno);
  }
  std::string msg;
  for (const auto& kv : seen)
    if (kv.second.size() > 1) {
      msg += "  " + kv.first + "  (lines";
      for (int l : kv.second) msg += " " + std::to_string(l);
      msg += ")\n";
    }
  if (!msg.empty())
    throw std::runtime_error("config file '" + config_file + "' sets the same key more than once:\n" + msg +
                             "\nYAML keeps only one of them, silently, so one of your settings would be "
                             "discarded without a word. Delete the duplicate and state the value once.");
}

// Real initializer
Parameters::Parameters(const std::string& config_file) {
  refuse_duplicate_keys(config_file);
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
  // The two keys surface_water.routing replaced are refused BY NAME, before the generic unknown-key
  // check below -- a merge that also remaps the VALUES is not something "did you mean routing?" can
  // explain. Without this an old config gets a suggestion that would produce a different run.
  if (root["surface_water"]["mode"] || root["surface_water"]["fsm_coupling"]) {
    throw std::runtime_error(
        "config: surface_water.mode and surface_water.fsm_coupling were REPLACED by the single key "
        "surface_water.routing (2026-09-10), because they answered one question between them: what "
        "happens to above-ground water. Translate:\n"
        "    mode: routed  + fsm_coupling: continuous  (or absent)  ->  routing: continuous\n"
        "    mode: routed  + fsm_coupling: impulse                  ->  routing: impulse\n"
        "    mode: ponded  (or removed)                             ->  routing: off\n"
        "`continuous` stays the default, so an old config that set neither key becomes routing: continuous. "
        "`ponded` and `removed` both become `off`: they were already indistinguishable from any config, "
        "because whether surface water was ponded or removed was decided by the evaporation taper, not by "
        "this key.");
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
          "config: " + key + ": `auto` is not a value. OMIT the key to take the default -- that is what "
          "absence means -- and the CONCRETE value it resolved to is recorded in full_config.yaml. (A "
          "test config should instead state the value it wants; see tests/config_identity.py.)");
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
  if (auto n = root["solver"]["time_step"]["mode"]) {
    refuse_auto(n, "solver.time_step.mode");
    time_step_mode     = require_enum(n.as<std::string>(), "solver.time_step.mode", {"fixed", "adaptive", "ramp"});
    time_step_mode_set = true;
  }
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
    time_integration_absent = true;
  }
  if (auto n = root["solver"]["t_bar"])       t_bar       = n.as<bool>();
  if (auto sm = root["solver"]["smoothing"]) {
    if (auto n = sm["ksat_surface"])        ksat_surface_smoothing        = n.as<double>();
    if (auto n = sm["ksat_soilbottom"])     ksat_soilbottom_smoothing     = n.as<double>();
    if (auto n = sm["storativity_surface"]) storativity_surface_smoothing = n.as<double>();
  }
  if (auto n = root["dev"]["under_relaxation"]) under_relaxation = n.as<double>();
  // output.trace is a MAP of channel -> bool, not a list of names (Andy, 2026-09-16: "I would rather
  // have this than have the user need to know what to type"). A list requires the reader to already know
  // the vocabulary; a map ships the vocabulary in the file, with every channel visible at its default.
  // Same reasoning as output.extra_rasters below, and the two are deliberately the same shape.
  if (auto tr = root["output"]["trace"]) {
    if (!tr.IsMap())
      throw std::runtime_error(
          "config: output.trace must be a map of channel -> true|false, e.g.\n"
          "  trace:\n    dt: true\n    fsm: false\n"
          "The list form ([dt]) was retired 2026-09-16: it required knowing the channel names to discover "
          "them. The channels are dt, water_step, budget, fsm.");
    for (const auto& kv : tr) {
      const std::string k = require_enum(kv.first.as<std::string>(), "output.trace",
                                         {"dt", "water_step", "budget", "fsm"});
      const bool on = kv.second.as<bool>();
      if (k == "dt")         trace_dt         = on;
      if (k == "water_step") trace_water_step = on;
      if (k == "budget")     trace_budget     = on;
      if (k == "fsm")        trace_fsm        = on;
    }
  }
  // output.extra_rasters -- same MAP shape as output.trace, and for the same reason: the raster names
  // ship in the config rather than having to be known. Adding one here is a new entry in the map and a
  // new line in the schema dictionary, not a new key.
  if (auto er = root["output"]["extra_rasters"]) {
    if (!er.IsMap())
      throw std::runtime_error(
          "config: output.extra_rasters must be a map of raster -> true|false, e.g.\n"
          "  extra_rasters:\n    post_groundwater: true");
    for (const auto& kv : er) {
      const std::string k = require_enum(kv.first.as<std::string>(), "output.extra_rasters",
                                         {"post_groundwater"});
      if (k == "post_groundwater") write_post_groundwater = kv.second.as<bool>();
    }
  }

  // surface_water.routing: continuous | impulse | off -- ONE key for whether FillSpillMerge routes
  // above-ground water AND, when it does, how its result reaches the groundwater. `off` is a real state
  // (nothing is routed), not an absence, which is what lets an FSM-off run RECORD what it did instead of
  // naming a coupling that never happened. Absent -> continuous, the default (#43).
  // fsm_coupling_set means "the user stated this", and the two absent -> impulse resolutions in
  // transient_groundwater.cpp still turn on it: stating `routing` at all is the explicit case.
  if (auto n = root["surface_water"]["routing"]) {
    const std::string r =
        require_enum(n.as<std::string>(), "surface_water.routing", {"continuous", "impulse", "off"});
    fsm_on                  = (r == "off") ? 0 : 1;
    fsm_coupling_continuous = (r == "continuous");
    fsm_coupling_set        = true;
  }
  if (auto n = root["solver"]["convergence"]["metric"])
    convergence_metric_head =
        (require_enum(n.as<std::string>(), "solver.convergence.metric", {"head", "volume"}) == "head");
  if (auto n = root["solver"]["convergence"]["water_volume_tol"]) water_volume_tol = std::stod(n.as<std::string>());
  if (auto n = root["solver"]["convergence"]["residual_gate"]) residual_gate = std::stod(n.as<std::string>());
  if (auto sc = root["solver"]["time_step"]) {
    if (auto n = sc["norm"])              dt_norm_rms     =
        (require_enum(n.as<std::string>(), "solver.time_step.norm", {"rms", "max"}) == "rms");
    if (auto n = sc["grow"])              dtc_grow        = n.as<double>();
    if (auto n = sc["shrink"])            dtc_shrink      = n.as<double>();
    if (auto n = sc["grow_if_niter_leq"]) dtc_easy_iters  = n.as<int>();
    if (auto n = sc["max_retries"])       dtc_max_retries = n.as<int>();
  }
  if (auto n = root["solver"]["newton"]["dt0"]) {
    refuse_auto(n, "solver.newton.dt0");
    dtc_dt0     = parse_time_seconds(n.as<std::string>(), "solver.newton.dt0");
    dtc_dt0_set = true;
  }
  if (auto ar = root["solver"]["anderson"]["restart"]) {
    if (auto n = ar["enabled"])      ar_enabled      = n.as<bool>();
    if (auto n = ar["rho"])          ar_rho          = n.as<double>();
    if (auto n = ar["patience"])     ar_patience     = n.as<int>();
    if (auto n = ar["max_it"])       ar_max_it       = n.as<int>();
    if (auto n = ar["max_restarts"]) ar_max_restarts = n.as<int>();
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
  if (auto n = root["solver"]["time_step"]["dt_min"]) {
    refuse_auto(n, "solver.time_step.dt_min");
    dtc_dt_min = parse_time_seconds(n.as<std::string>(), "solver.time_step.dt_min");
    dtc_dt_min_set = true;
    if (dtc_dt_min < 0.0)
      throw std::runtime_error("config: solver.time_step.dt_min must be >= 0 (0 disables the floor).");
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
  // mode was folded into surface_water.routing above; see the schema note and the migration refusal.
  // The ponded-vs-removed TODO that lived here is CLOSED rather than carried: the two were already
  // indistinguishable from any config, so `off` loses nothing.
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

  // WHO SIZES THE STEP -- one question, one key (solver.time_step.mode).
  //
  // This used to be TWO booleans, solver.adaptive_dt and solver.newton.dt_continuation, and they could
  // both be true. They are two controllers for ONE question, and WTM.cpp resolved the clash as
  // `if (use_dt_adaptive) ... else if (use_newton_continuation)` -- so adaptive silently WON and the ramp
  // never ran, after InitialiseSNES had already printed its banner. Measured on tests/dt_sensitivity
  // inputs with solver.method: newton: "adaptive dt: 30 steps" instead of "dt-continuation: deltat now
  // 5.24e+08 s". That earned an abort, which an enum makes UNNECESSARY: the contradiction can no longer
  // be written down. The abort is gone with it.
  //
  // The two were also named on different principles -- one for what it RESPONDS to (error), one for its
  // METHOD (continuation) -- which is why neither name suggested they competed.
  //
  //   adaptive  error-controlled, clamped to the report span
  //   ramp      pseudo-transient continuation: solve-ease growth, unclamped
  //   fixed     dt exactly as given
  //   ABSENT    resolved below; full_config.yaml records the concrete mode, never a sentinel
  //
  // WHY NOT MAKE adaptive A SUPERSET OF ramp (asked 2026-09-03): an error-driven controller cannot ramp
  // to steady state. During a long drainage transient the error IS large, and error control reads that
  // as "shrink" -- WTM.cpp records the experiment: a residual/state-change SER controller "is WORSE
  // here ... growing on solve-EASE advances far better". The thing continuation must do is what error
  // control forbids. They stay distinct modes.
  // ...and against the COLLECTOR. `implicit` siphons above-surface water at rate max(0,wtd)/dt, so its
  // per-step error GROWS as the controller shrinks dt: the founding assumption of error-controlled
  // stepping -- refine dt, reduce error -- is false for it, and the reject/retry loop cannot converge.
  // It does not misbehave subtly, it DIES: "adaptive dt: step failed after max retries; -wtm_dt_tol too
  // tight or the local stability ceiling is below the smallest tried dt." (tests/active_set, imp_plain.)
  // Andy, 2026-09-03: implicit + adaptive_dt is not allowed; active_set + adaptive_dt is the pairing that
  // works and is the production default.
  const bool implicit_collector = (runoff_collector == "implicit");
  if (time_step_mode_set && time_step_mode == "adaptive" && implicit_collector)
    throw std::runtime_error(
        "config: solver.time_step.mode: adaptive cannot be used with surface_water.collection.method: implicit. "
        "The implicit siphon removes above-surface water at rate max(0,wtd)/dt, so its per-step error "
        "GROWS as the controller shrinks dt -- no step can ever be accepted, and the run dies with "
        "'step failed after max retries'. Use collection.method: active_set (the default, and the "
        "enforcement adaptive stepping is built for), or solver.time_step.mode: fixed.");

  // ...and `explicit` against the same axis, as a WARNING rather than a refusal. The free surface under
  // `explicit` is a post-solve clamp, so above-surface water is removed AFTER the step rather than
  // constrained inside it, and the clamped set is free to change between steps. Andy, 2026-09-11: that
  // will generally permit free-surface oscillations, it is expected, and it is bad behaviour -- but
  // `explicit` is not the production collector, so this is stated, not fixed.
  //
  // MEASURED (benchmark/lc103_knob_sweep, tests/storage_equivalence fixture at 64 cells/degree, 460
  // simulated years, 56 of 88 land cells sitting at wtd == 0):
  //     explicit + adaptive   6.4379e-03 m still moving at the end, 32 of 88 cells, 74509 solves
  //     explicit + fixed      7.1054e-14 m                          0 of 88 cells,  6000 solves
  //     active_set + adaptive 4.3109e-07 m                          0 of 88 cells,    118 solves
  // PERMITS vs DOES, and the difference matters (Andy, 2026-09-11). A post-solve clamp ALWAYS PERMITS
  // oscillation: nothing in the formulation prevents the clamped set from changing between steps, so
  // the possibility is a property of the method and no run can retire it. What the numbers above show
  // is only that it does not always HAPPEN -- at fixed dt, on this fixture, the dynamics do not drive
  // it there. Those are different questions, and a passing run answers the weaker one. The warning is
  // therefore phrased as "may", and says nothing about when it will not.
  //
  // NOT a refusal: this pairing runs, converges its solves, and closes its budget -- and suites in the
  // tree use it deliberately. Refusing would break them to prevent a behaviour the user may want.
  if (time_step_mode_set && time_step_mode == "adaptive" && runoff_collector == "explicit")
    std::cerr << "WARNING [surface_water.collection.method=explicit + solver.time_step.mode=adaptive]: the "
                 "explicit collector clamps the free surface AFTER each step, so the clamped set can change "
                 "from step to step and the surface may LIMIT-CYCLE rather than settle. Measured on "
                 "tests/storage_equivalence at 64 cells/degree over 460 yr: 6.4e-03 m of motion still "
                 "present at the end across 32 of 88 cells, at 74509 nonlinear solves -- against 7.1e-14 m, "
                 "0 cells and 6000 solves for the SAME collector at solver.time_step.mode: fixed, and "
                 "4.3e-07 m, 0 cells and 118 solves for collection.method: active_set. If the run is meant "
                 "to reach equilibrium, prefer active_set (the production collector); if you want explicit, "
                 "prefer a fixed step. See benchmark/lc103_knob_sweep.\n";
  // `ramp` is the NEWTON path's pseudo-transient continuation, and CreateSNES gates it on use_newton
  // (`dtc_on && use_newton`). Asked for anywhere else it would simply not run, and the run would step
  // at a fixed dt while its config said `ramp` -- a key that reads as a choice and is not one, which is
  // the defect class of #27/#28/#49. Say so instead.
  if (time_step_mode_set && time_step_mode == "ramp" && solver_method != "newton")
    throw std::runtime_error(
        "config: solver.time_step.mode: ramp is the Newton path's continuation ramp and only runs with "
        "solver.method: newton (this run asks for " + (solver_method.empty() ? std::string("anderson") : solver_method) +
        "). Use solver.time_step.mode: adaptive or fixed, or set solver.method: newton.");
  // RESOLVE AN ABSENT KEY. It yields twice over: to Newton's ramp, which owns the step size on that
  // path, and to fixed stepping under the implicit collector, which an error controller cannot drive
  // at all (see the refusal just above).
  if (!time_step_mode_set) {
    const bool newton = (solver_method == "newton");
    time_step_mode = newton ? "ramp" : implicit_collector ? "fixed" : "adaptive";
  }
  // The two internal booleans the solver paths read. They are DERIVED here and nowhere else, so the
  // pair can never disagree with the mode or with each other.
  adaptive_dt     = (time_step_mode == "adaptive");
  dt_continuation = (time_step_mode == "ramp");

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
  // Output.
  std::cout << "c output_directory       = " << (output_directory.empty() ? "(legacy: literal paths)" : output_directory)
            << std::endl;
  std::cout << "c if_exists              = " << if_exists << std::endl;
  std::cout << "c outfile_prefix         = " << outfile_prefix << std::endl;
  std::cout << "c textfilename           = " << textfilename << std::endl;
  std::cout << "c verbosity              = " << verbosity << std::endl;
  std::cout << "c --- end configuration ---" << std::endl;
}
