# msi_env.sh -- set up the WTM build/run environment on MSI (Agate, Rocky 8).
#
# SOURCE this (do not execute), so `module load` / `conda activate` affect your
# shell:
#     source msi_env.sh          # toolchain only -- safe for building AND running wtm.x
#     source msi_env.sh test     # ALSO activate the conda env with rasterio (for run_all.sh)
#
# The 'test' form activates a conda env named 'wtmtest' that must already exist:
#     module load python/3.10.9_anaconda2023.03_libmamba
#     conda create -n wtmtest -c conda-forge rasterio numpy
# (one-time; see BUILD_HPC.md).

# --- MPI / PETSc / GDAL / CMake: the verified consistent toolchain --------------
module purge
module load petsc/3.24.5-gnu-rocky8              # auto-loads gcc/11.3.0 + its MPICH ($PETSC_DIR/bin)
module load gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3  # gcc-11.3.0-matched; shares PETSc's HDF5
module load cmake/3.29.2-rocky8
module load git

echo "WTM toolchain:"
echo "  PETSc   $(pkg-config --modversion PETSc 2>/dev/null)   ($PETSC_DIR)"
echo "  mpicxx  $(command -v mpicxx)"
echo "  gdal    $(gdal-config --version 2>/dev/null)"
echo "  gcc     $(gcc -dumpfullversion 2>/dev/null)"

# --- Optional: Python + rasterio for the test suite (source with 'test') --------
if [ "${1:-}" = "test" ]; then
  module load python/3.10.9_anaconda2023.03_libmamba
  # make `conda activate` available in this shell, then activate the rasterio env
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda activate wtmtest 2>/dev/null; then
    echo "  conda   wtmtest active -- rasterio $(python3 -c 'import rasterio; print(rasterio.__version__)' 2>/dev/null || echo MISSING)"
    echo "  NOTE:   wtmtest carries a conda GDAL. Run 'conda deactivate' before re-running cmake,"
    echo "          so the build doesn't pick up conda's GDAL. It is fine for running the model/tests."
  else
    echo "  conda   env 'wtmtest' not found -- create it (see BUILD_HPC.md), then re-source with 'test'."
  fi
fi
