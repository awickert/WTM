#ifndef RESOLVED_CONFIG_HPP
#define RESOLVED_CONFIG_HPP
// THE RESOLUTION LEDGER (Andy, 2026-09-10): "full config should be set by the code as it executes to
// ensure that it is accurate to what actually happened. this means it is built after or while going
// through the flow control."
//
// WHY THIS EXISTS. full_config.yaml used to be RECONSTRUCTED by a writer that re-derived each
// resolution from Parameters. That writer was a SECOND implementation of the model's own resolution
// policy -- the identical defect we deleted from the test-config shim, living in WTM.cpp -- and it got
// three different keys wrong in one day:
//   surface_water.fsm_coupling   recorded `impulse` for runs in which NO coupling ran (#89)
//   solver.newton.dt0            recorded seconds as a bare number, which re-reads as YEARS
//   collection.method            recorded the REQUEST; a Picard run enforcing `explicit` said active_set
// The last is the dangerous one: a suite declaring the request would MATCH the record while the run
// did something else, so the declared-config check (#83) would pass on a false statement.
//
// THE RULE THIS ENFORCES. A resolved value is recorded AT THE POINT OF RESOLUTION, by the code that
// makes the decision, as control flow passes through it. The writer then only PRINTS what was
// recorded -- resolved_or_die() throws if a key was never recorded, so the writer cannot invent or
// re-derive a value even by accident.
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace resolved_config {

inline std::map<std::string, std::string>& ledger() {
  static std::map<std::string, std::string> m;
  return m;
}

// Record what this run actually resolved for `key`. Called from the decision site.
// Idempotent for the same value: resolution sites can run per-step (update() does), and re-recording
// the same answer is fine. A DIFFERENT answer for a key already recorded is a real bug -- two places
// deciding one setting -- so it throws rather than letting the last writer win.
inline void record(const std::string& key, const std::string& value) {
  auto& m  = ledger();
  auto  it = m.find(key);
  if (it != m.end() && it->second != value)
    throw std::runtime_error("resolved_config: " + key + " was resolved twice, to '" + it->second +
                             "' and then '" + value +
                             "'. One setting, one decision site -- two disagreeing resolutions is the "
                             "defect this ledger exists to make impossible.");
  m[key] = value;
}

// A const char* MUST have its own overload. Without one it binds to the bool overload in preference
// to std::string -- so record(key, "off") called the bool version, which called record(key, "true"),
// which was ALSO a const char*, and the function recursed until the stack died. That is what hung the
// model on its first two integration attempts, and it looked like a solve-path hang because it fired
// from inside initialise(). It is a C++ overload-resolution trap, nothing to do with WTM.
inline void record(const std::string& key, const char* value) { record(key, std::string(value)); }
inline void record(const std::string& key, bool value) { record(key, std::string(value ? "true" : "false")); }

inline void record_num(const std::string& key, double value) {
  std::ostringstream os;
  os << value;
  record(key, os.str());
}

inline bool recorded(const std::string& key) { return ledger().count(key) != 0; }

// Print-time accessor. THROWS if nothing recorded this key, which is the point: the writer must not be
// able to fall back on re-deriving it.
inline const std::string& resolved_or_die(const std::string& key) {
  auto& m  = ledger();
  auto  it = m.find(key);
  if (it == m.end())
    throw std::runtime_error(
        "resolved_config: full_config.yaml asked for '" + key +
        "' but no code path recorded it. Record it AT THE POINT OF RESOLUTION rather than re-deriving "
        "it here -- re-derivation in the writer is what made full_config describe runs that did not "
        "happen.");
  return it->second;
}

}  // namespace resolved_config
#endif
