#!/bin/bash -l
# build_msi.sh -- configure and build WTM on MSI (Agate) in one command.
#
# Self-contained: it loads the verified toolchain by sourcing msi_env.sh (the
# single source of truth for the petsc/gdal/gcc module set), then runs cmake +
# make with the MPI compiler wrappers. Run it ON an MSI compute node -- grab one
# first, e.g.
#     srun -N 1 --ntasks-per-node=8 --mem-per-cpu=4gb -t 2:00:00 -p interactive --pty bash
# -- do NOT build on the login node.
#
# Usage:
#   ./build_msi.sh                  configure + build (build/, make -j nproc)
#   ./build_msi.sh --fresh          wipe the build dir first (clean reconfigure)
#   ./build_msi.sh --test           also run the full test suite (uses the wtmtest conda env)
#   ./build_msi.sh -j 16            override the parallel job count
#   ./build_msi.sh --build-dir DIR  build in DIR instead of ./build
#
# See BUILD_HPC.md for the full procedure, prerequisites, and troubleshooting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
JOBS="$(nproc)"
FRESH=0
RUN_TESTS=0

usage() { sed -n '11,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --fresh)     FRESH=1; shift ;;
    --test)      RUN_TESTS=1; shift ;;
    -j)          JOBS="${2:?-j needs a number}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:?--build-dir needs a path}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1  (try --help)" >&2; exit 2 ;;
  esac
done

# --- toolchain (modules) -----------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: the 'module' command is not available." >&2
  echo "       Run this on an MSI node (login or compute), not on your laptop." >&2
  echo "       See BUILD_HPC.md for the local (conda) build instead." >&2
  exit 1
fi
# msi_env.sh does 'module purge' then loads petsc/gdal/cmake (+ prints versions).
# shellcheck source=msi_env.sh
source "$ROOT/msi_env.sh"

# --- submodules present? (common/richdem, common/fmt) ------------------------
if [ ! -e "$ROOT/common/richdem/CMakeLists.txt" ] || [ ! -e "$ROOT/common/fmt/CMakeLists.txt" ]; then
  echo "Submodules missing -- initialising (common/richdem, common/fmt) ..."
  git -C "$ROOT" submodule update --init --recursive
fi

# --- configure + build -------------------------------------------------------
if [ "$FRESH" = 1 ]; then echo "Removing $BUILD_DIR (--fresh)"; rm -rf "$BUILD_DIR"; fi
mkdir -p "$BUILD_DIR"

echo "==> Configuring (RelWithDebInfo, GDAL on, MPI wrappers) in $BUILD_DIR"
CXX=mpicxx CC=mpicc cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON

echo "==> Building with -j $JOBS"
cmake --build "$BUILD_DIR" -j "$JOBS"

if [ ! -x "$BUILD_DIR/wtm.x" ]; then
  echo "ERROR: build finished but $BUILD_DIR/wtm.x is missing." >&2
  exit 1
fi
echo
echo "Build complete: $BUILD_DIR/wtm.x"

# --- optional test suite (needs rasterio; build stays conda-off above) -------
if [ "$RUN_TESTS" = 1 ]; then
  echo "==> Activating the wtmtest conda env (rasterio) and running the suite"
  source "$ROOT/msi_env.sh" test          # keeps the build modules, adds rasterio
  ( cd "$ROOT/tests" && ./run_all.sh )
fi
