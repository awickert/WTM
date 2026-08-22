# Stamp the current git commit hash + clean/dirty state into a header at BUILD time (not configure time),
# so the value is fresh on every build and correct even where the source tree is absent/different at run
# time (e.g. a binary rsynced to MSI). Run in script mode:
#   cmake -DSRC_DIR=<repo> -DIN_FILE=<template> -DOUT_FILE=<generated header> -P GitStamp.cmake
#
# "dirty" = uncommitted TRACKED changes (matches `git describe --dirty`); untracked scratch files do not
# count, so a stray config or note does not flag an otherwise-clean build.

find_package(Git QUIET)

set(WTM_GIT_COMMIT "unknown")
set(WTM_GIT_DIRTY 0)
set(WTM_GIT_STATE "no-git")

if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SRC_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE _hash OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc ERROR_QUIET)
  if(_rc EQUAL 0)
    set(WTM_GIT_COMMIT "${_hash}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${SRC_DIR}" status --porcelain --untracked-files=no
      OUTPUT_VARIABLE _status OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if("${_status}" STREQUAL "")
      set(WTM_GIT_STATE "clean")
    else()
      set(WTM_GIT_DIRTY 1)
      set(WTM_GIT_STATE "dirty")
    endif()
  endif()
endif()

# configure_file only rewrites when the content changes, so clean incremental builds don't churn.
configure_file("${IN_FILE}" "${OUT_FILE}" @ONLY)

# Loud build-time banner every build when dirty (the compiler #warning in the header fires when the header
# is (re)compiled; this message fires on every build so a dirty tree is never silent).
if(WTM_GIT_DIRTY)
  message(WARNING
    "WTM build from a DIRTY git tree (uncommitted tracked changes): commit ${WTM_GIT_COMMIT} does not "
    "fully describe this binary. Commit for a reproducible, git-consistent build.")
endif()
