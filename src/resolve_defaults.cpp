#include "resolve_defaults.hpp"

#include <petscsys.h>

#include <stdexcept>
#include <string>

#include "resolved_config.hpp"

// See resolve_defaults.hpp for what belongs here and what does not.
//
// use_picard is derived from params.solver_method and nothing else (CreateSNES.cpp: picard_flag =
// (params.solver_method == "picard")), so this module needs no solve context at all -- which is what
// lets it run before anything is built.
// THE ORDER IS THE CONTRACT (Andy, 2026-09-10):
//   1. everything the user SET is already in Parameters -- the parser did that, and nothing here
//      overwrites a stated value;
//   2. then the defaults, IN DEPENDENCY ORDER: the ones with a single unconditional answer first, the
//      ones needing flow control after, so every conditional reads inputs that are already final.
// Read top to bottom it is a sequence rather than a web, and a cycle cannot be written by accident.
void resolve_defaults(Parameters& params) {
  const bool use_picard = (params.solver_method == "picard");

  // ---- surface_water.collection.method -------------------------------------------------------
  // An empty string is the ABSENT key. The default is active_set, except on Picard, where the
  // active-set pin is absent from the operator and RHS: selecting it there would leave the constraint
  // UNENFORCED (measured: water piles over 1440 of ~1444 land cells with FSM off), so a default must
  // not choose it. That is a solver-dependent DEFAULT, not a downgrade -- an omitted key always means
  // "the default", and which default applies follows from the solver. The table is in config.yaml.
  // An EXPLICIT method is always honoured; this only ever decides what an unspecified config does.
  if (params.runoff_collector.empty()) params.runoff_collector = "active_set";
  if (params.runoff_collector == "active_set" && !params.runoff_collector_set && use_picard) {
    params.runoff_collector = "explicit";
    PetscPrintf(PETSC_COMM_WORLD,
                "NOTE: surface_water.collection.method defaults to `explicit` on the Picard solver "
                "(the post-solve clamp), because the active_set pin is absent from the Picard "
                "operator/RHS. Defaults are solver-dependent; set the key explicitly to override.\n");
  }

  // ---- surface_water.routing (the coupling half) ---------------------------------------------
  // With FSM off there is no coupling at all, so nothing to resolve -- `off` is a real state, not an
  // absence (#89). With FSM on, `continuous` yields to `impulse` in two cases, and an EXPLICIT request
  // is REFUSED by name in both rather than silently yielding: a key that reads as honoured and is not
  // is the defect class of #27, #28 and #49.
  if (params.fsm_on) {
    // (1) The explicit collector clamps above-surface water AFTER the solve while `continuous` feeds
    //     it back in as a source, so the two fight and the run never settles to a dt-independent
    //     state (measured: observed order goes negative under dt refinement, error stalls at ~1.1 m).
    if (params.runoff_collector == "explicit") {
      if (params.fsm_coupling_set && params.fsm_coupling_continuous)
        throw std::runtime_error(
            "config: surface_water.routing: continuous cannot be used with "
            "surface_water.collection.method: explicit -- the pair does not converge (measured: observed "
            "order goes negative under dt refinement, error stalls at ~1.1 m). `explicit` clamps "
            "above-surface water AFTER the solve, while `continuous` feeds it back in as a source term, "
            "so the two fight and the run never settles to a dt-independent state. Use "
            "collection.method: active_set (the default), or routing: impulse.");
      if (params.fsm_coupling_continuous) {
        params.fsm_coupling_continuous = false;
        PetscPrintf(PETSC_COMM_WORLD,
                    "surface_water.routing: absent -> impulse (collection.method: explicit cannot take "
                    "the continuous coupling).\n");
      }
    }
    // (2) `continuous` hands FSM's per-cell volume change to the next step through the DISTRIBUTED
    //     recharge carrier, and infiltration_during_flow routes recharge through the serial rank-0
    //     loop, which has no such carrier -- so the coupling would be silently inert.
    if (params.infiltration_on) {
      if (params.fsm_coupling_set && params.fsm_coupling_continuous)
        throw std::runtime_error(
            "config: surface_water.routing: continuous cannot be used with "
            "surface_water.infiltration_during_flow: true. The continuous coupling hands FillSpillMerge's "
            "per-cell volume change to the next step through the DISTRIBUTED recharge carrier, but "
            "infiltration_during_flow routes recharge through the serial rank-0 loop, which has no such "
            "carrier -- so the coupling would be silently inert. Use routing: impulse, or "
            "infiltration_during_flow: false.");
      if (params.fsm_coupling_continuous) {
        params.fsm_coupling_continuous = false;
        PetscPrintf(PETSC_COMM_WORLD,
                    "surface_water.routing: absent -> impulse (infiltration_during_flow: true routes "
                    "recharge serially, which carries no FSM-delta source).\n");
      }
    }
  }

  // ---- RECORD, at the decision site, what this run resolved -----------------------------------
  // full_config.yaml PRINTS these rather than re-deriving them; resolved_or_die() throws if a key was
  // never recorded, so the writer cannot fall back on a second implementation of this policy.
  resolved_config::record("surface_water.collection.method", params.runoff_collector);
  resolved_config::record("surface_water.routing",
                          !params.fsm_on ? "off" : (params.fsm_coupling_continuous ? "continuous" : "impulse"));
  resolved_config::record("solver.time_integration", params.time_integration);
  resolved_config::record("solver.time_step.mode", params.time_step_mode);
}
