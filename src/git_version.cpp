#include "git_version.hpp"

// Generated at build time into ${CMAKE_BINARY_DIR}/generated (on wtm's include path). Including it HERE, in
// this single small translation unit, is what makes the compiler emit the loud dirty-tree #warning without
// forcing the rest of the project to recompile.
#include "wtm_git_version.hpp"

const char* wtm_git_commit() {
  return WTM_GIT_COMMIT;
}

bool wtm_git_dirty() {
  return WTM_GIT_DIRTY != 0;
}

const char* wtm_git_state() {
  return WTM_GIT_STATE;
}
