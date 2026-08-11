# run_env.sh -- prepare the environment to RUN an already-built WTM on MSI (Agate).
#
# No build. It loads the toolchain modules (the runtime PETSc/GDAL libraries the
# binary is linked against, plus PETSc's mpiexec) and the `wtmtest` conda env
# (rasterio, needed by scaling_study.py and the test suite), sets the pure-MPI
# default, guards against conda shadowing PETSc's launcher, and checks the binary.
# msi_env.sh stays the single source of truth for the module set.
#
# SOURCE it (so the module loads / conda activate / exports affect your shell):
#     source run_env.sh
#
# Then launch, e.g.:
#     mpiexec -n 8 build/wtm.x <config.cfg>              # or: srun --mpi=pmi2 -n 8 build/wtm.x <cfg>
#     ( cd benchmark/scaling && python3 scaling_study.py --strong 2000 --ranks 8 4 2 1 --builds after )
#     ( cd tests && ./run_all.sh )
#
# One-time prerequisites (see BUILD_HPC.md): the wtmtest env must exist, created via
# the ANACONDA module (NOT the plain python/*-gcc-* ones, which swap gcc/11.3.0 and
# break wtm.x at runtime), with `conda config --set auto_activate_base false`.

_wtm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Toolchain modules + wtmtest (rasterio). 'test' is what pulls in the conda env.
# shellcheck source=msi_env.sh
source "$_wtm_root/msi_env.sh" test

# Pure MPI: one rank per core, no OpenMP threads (OpenMP x MPI on the same cores
# oversubscribes and can hang at >=4 ranks). Override if you intend hybrid runs.
export OMP_NUM_THREADS=1

# Conda-shadow guard (BUILD_HPC.md): activating wtmtest prepends conda's bin to PATH,
# which can shadow PETSc's MPI launcher/wrappers. Re-assert $PETSC_DIR/bin so
# mpiexec/mpicc stay PETSc's MPICH; conda's python3 (rasterio) still wins for python
# (there is no python3 in $PETSC_DIR/bin).
if [ -n "${PETSC_DIR:-}" ]; then export PATH="$PETSC_DIR/bin:$PATH"; fi

# Verify nothing critical got shadowed by conda.
_mpi="$(command -v mpiexec || command -v mpirun || true)"
case "${_mpi:-}" in
  "${PETSC_DIR:-/nonexistent}"/*) echo "  mpiexec $_mpi  (PETSc MPICH -- good)" ;;
  "") echo "  WARNING: no mpiexec/mpirun on PATH -- are the modules loaded?" >&2 ;;
  *)  echo "  WARNING: mpiexec is '$_mpi', NOT under \$PETSC_DIR/bin ($PETSC_DIR/bin) -- possible conda shadow." >&2 ;;
esac
if python3 -c "import rasterio" 2>/dev/null; then
  echo "  python3 $(command -v python3)  (rasterio ok)"
else
  echo "  WARNING: rasterio not importable -- is wtmtest active? scaling_study.py / the tests need it." >&2
fi

# You said it's already built -- confirm the binary is actually there.
if [ -x "$_wtm_root/build/wtm.x" ]; then
  echo "  wtm.x   $_wtm_root/build/wtm.x  (ready; OMP_NUM_THREADS=1)"
  echo "  launch  mpiexec -n N $_wtm_root/build/wtm.x <config.cfg>   (or: srun --mpi=pmi2 -n N ...)"
else
  echo "  wtm.x   NOT FOUND at $_wtm_root/build/wtm.x -- build it first (./build_msi.sh)." >&2
fi

unset _wtm_root _mpi
