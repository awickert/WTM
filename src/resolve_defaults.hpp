#ifndef RESOLVE_DEFAULTS_HPP
#define RESOLVE_DEFAULTS_HPP
#include "parameters.hpp"

// RESOLVE THIS RUN'S DEFAULTS, IN ONE PLACE, BEFORE THE RUN STARTS.
//
// Andy, 2026-09-10: "full config should be set by the code as it executes to ensure that it is
// accurate to what actually happened ... deciding on the full routing in initialize makes a lot of
// sense ... there should be a reasonable and minimalistic way of doing this: a section or module that
// is entirely dedicated to resolving defaults."
//
// WHAT BELONGS HERE: decisions that turn an ABSENT key into a concrete value, and the refusals that
// gate them. Pure -- Parameters in, Parameters out. No solve state, no arrays, no PETSc objects.
//
// WHAT DOES NOT: per-step setup. That distinction is the whole lesson of the first attempt, which
// hoisted 351 lines of update() wholesale and hung the model: those lines were resolution AND the
// per-step g_* switch setting braided together, and moving both starved the second.
//
// WHY IT MUST RUN EARLY: full_config.yaml is written before the first solve. While these two
// resolutions still happened inside update(), the writer could not see them and RE-DERIVED them
// instead -- a second implementation of the policy, which got the collector wrong (recording the
// REQUEST, so a Picard run enforcing `explicit` recorded `active_set`) and the coupling wrong (#89).
void resolve_defaults(Parameters& params);
#endif
