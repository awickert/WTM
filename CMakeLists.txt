cmake_minimum_required(VERSION 3.9)

project(WTM LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Default to an optimized build. Without this, a plain `cmake ..` leaves
# CMAKE_BUILD_TYPE empty -> no optimization flags -> an -O0 binary that runs the
# groundwater solve several times slower. Override with -DCMAKE_BUILD_TYPE=Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, RelTesting)" FORCE)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release (-O3).")
endif()

# use -fsanitize=address to check for memory errors
set(CMAKE_CXX_FLAGS_RELTESTING "-O3 -g -Wall -Wextra -pedantic -Wshadow")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wall -Wextra -pedantic -Wshadow")

# RichDEM requires GDAL
set(USE_GDAL ON)

# # TODO: Enable to perform many linting checks
# set(CMAKE_CXX_CLANG_TIDY
#         clang-tidy;
#             # -header-filter=.*;
#             -checks=*,-llvmlibc-restrict-system-libc-headers,-modernize-use-trailing-return-type,-llvmlibc-callee-namespace,-bugprone-branch-clone,-google-readability-namespace-comments,-llvm-namespace-comment,-llvmlibc-implementation-in-namespace,-readability-uppercase-literal-suffix,-readability-function-cognitive-complexity,-fuchsia-default-arguments-calls,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-hicpp-uppercase-literal-suffix,-hicpp-no-array-decay,-misc-unused-parameters;
#             -warnings-as-errors=*;)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

find_package(GDAL REQUIRED)
#find_package(MPI REQUIRED)
find_package(OpenMP)
find_package(PkgConfig REQUIRED)
find_package(Sanitizers)

# NOTE: You may need to update this path to identify PETSc's location
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/home/kerry/petsc/arch-linux-cxx-opt/lib/pkgconfig/")
pkg_check_modules(PETSC PETSc>=3.17.1 IMPORTED_TARGET REQUIRED)
message(STATUS "Found PETSc ${PETSC_VERSION}")

add_subdirectory(common/richdem EXCLUDE_FROM_ALL)
add_subdirectory(common/fmt EXCLUDE_FROM_ALL)
add_subdirectory(common/yaml-cpp EXCLUDE_FROM_ALL)

# --- Build-time git stamp -------------------------------------------------------------------------
# Bake the git commit hash + clean/dirty state into the binary (provenance), and make the compiler warn
# loudly when building from a dirty tree. GitStamp.cmake runs on EVERY build (not cached at configure
# time) so the hash is always current -- and correct even when the source tree is absent/different at run
# time (e.g. a binary rsynced to MSI). configure_file inside it only rewrites on change, so clean
# incremental builds do not churn.
set(WTM_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${WTM_GENERATED_DIR})
add_custom_target(wtm_git_stamp ALL
  BYPRODUCTS ${WTM_GENERATED_DIR}/wtm_git_version.hpp
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DIN_FILE=${CMAKE_SOURCE_DIR}/cmake/git_version.hpp.in
          -DOUT_FILE=${WTM_GENERATED_DIR}/wtm_git_version.hpp
          -P ${CMAKE_SOURCE_DIR}/cmake/GitStamp.cmake
  COMMENT "Stamping git commit + clean/dirty state into wtm_git_version.hpp"
  VERBATIM)

# Define a library aggregating all the model componenets that we can use for
# both running simulations and running tests
add_library(wtm
  src/ArrayPack.cpp
  src/git_version.cpp
  src/irf.cpp
  src/parameters.cpp
  src/transient_groundwater.cpp
  src/update_effective_storativity.cpp
)
target_compile_options(wtm PRIVATE -Wfloat-conversion -Wall -Wextra -pedantic -Wshadow)
target_link_libraries(wtm PUBLIC richdem OpenMP::OpenMP_CXX fmt::fmt PkgConfig::PETSC yaml-cpp::yaml-cpp)
target_compile_features(wtm PUBLIC cxx_std_20)
# git_version.cpp includes the generated wtm_git_version.hpp; regenerate it (and re-fire the dirty warning)
# before wtm builds.
target_include_directories(wtm PRIVATE ${WTM_GENERATED_DIR})
add_dependencies(wtm wtm_git_stamp)

add_executable(wtm.x
  src/WTM.cpp
)
target_compile_options(wtm.x PRIVATE -Wfloat-conversion -Wall -Wextra -pedantic -Wshadow)
target_link_libraries(wtm.x PRIVATE wtm)
target_compile_features(wtm.x PRIVATE cxx_std_20)
# RICHDEM_NO_PROGRESS: silence richdem's FillSpillMerge progress bar. FSM now runs every timestep
# (tight coupling), so the per-call bar would spam stderr and slow runs; the standalone dephier.x keeps it.
target_compile_definitions(wtm PRIVATE DOCTEST_CONFIG_DISABLE RICHDEM_NO_PROGRESS)

add_sanitizers(wtm.x)

add_executable(dephier.x
  src/run_dephier.cpp
  src/parameters.cpp
  src/ArrayPack.cpp
)
target_compile_options(dephier.x PRIVATE -Wfloat-conversion -Wall -Wextra -pedantic -Wshadow)
target_link_libraries(dephier.x PRIVATE richdem OpenMP::OpenMP_CXX fmt::fmt yaml-cpp::yaml-cpp)
target_compile_features(dephier.x PRIVATE cxx_std_20)
add_sanitizers(dephier.x)

# MPI unit tests for the DMDA gather/scatter primitives. Run under mpirun at
# several rank counts (see tests/run_unit_tests.sh); the row-major layout must
# hold independent of decomposition, which only n>1 exercises.
add_executable(test_dmda.x
  src/test_dmda_gather.cpp
  src/test_storage_math.cpp
  src/test_geometry.cpp
  src/test_fsm_water_ledger.cpp
  src/test_tr_bdf2_balance.cpp
)
target_link_libraries(test_dmda.x PRIVATE wtm)
target_compile_features(test_dmda.x PRIVATE cxx_std_20)
