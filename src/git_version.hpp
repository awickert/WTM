#pragma once

// Git provenance of the WTM binary, baked in at build time (see cmake/GitStamp.cmake). Used for the run
// provenance record and printed at start-up so every run is traceable to the exact code that produced it.

const char* wtm_git_commit();  // full commit hash the binary was built from, or "unknown" / "no-git"
bool        wtm_git_dirty();   // true if the build tree had uncommitted tracked changes (not reproducible)
const char* wtm_git_state();   // "clean" | "dirty" | "no-git"
