# Building WTM on HPC clusters

A likely path to a working build on a shared cluster, with the specific pitfalls
that bit us during development. Written to work for **any** module- or
conda-based cluster; the **Minnesota Supercomputing Institute (MSI)** is used as
the concrete worked example throughout (module names, Slurm launch), since that
is our production target. On another cluster, substitute your site's module
names and job launcher and the rest carries over.

The golden rule up front:

> **Use ONE consistent toolchain for compiler + MPI + PETSc + GDAL.** Mixing
> sources (e.g. conda GDAL with system PETSc/MPI) causes link failures. Every
> build problem we hit locally was a mixed-toolchain problem.

## Dependencies

External (must be provided by modules or conda):

| Dep | Constraint | Notes |
|-----|-----------|-------|
| C++ compiler | **C++20** (`gcc`/`g++` ≥ 11; 13 ideal) | `cxx_std_20` in CMakeLists |
| MPI | OpenMPI or Intel MPI | **must be the same MPI PETSc was built against** |
| PETSc | **≥ 3.17.1**, real scalars, MPI | found via `pkg-config` (`PETSc.pc`) |
| GDAL | any recent | `find_package(GDAL REQUIRED)` |
| CMake | ≥ 3.16 | |
| pkg-config | | used to locate PETSc |

Bundled (git submodules — no module needed, but you MUST clone them):

- `common/richdem`, `common/fmt` → clone with `--recurse-submodules`.

The sanitizer CMake modules are vendored in `cmake/`, so `find_package(Sanitizers)`
needs nothing external.

## Step 0 — clone with submodules

```sh
git clone --recurse-submodules <your fork URL> WTM
cd WTM
# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

## Path A — cluster modules (try this first; MSI shown)

Find what's actually available (module names/versions vary by site and change over
time; always verify on the node):

```sh
module avail                      # full list
module spider petsc               # or: gdal, openmpi, cmake, gcc
```

Load a C++20 gcc, an OpenMPI, PETSc ≥ 3.17.1 (built with that OpenMPI), GDAL, CMake:

```sh
module load gcc/<11-or-newer> openmpi/<ver> petsc/<>=3.17> gdal cmake
```

**Verify the toolchain is consistent BEFORE building:**

```sh
pkg-config --modversion PETSc     # must print >= 3.17.1
pkg-config --variable=ccompiler PETSc   # note which MPI/compiler PETSc used
mpicxx --version                  # should be the same gcc family
gdal-config --version
```

If `pkg-config --modversion PETSc` fails, add PETSc's pkgconfig dir:

```sh
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH
# (or the petsc module's lib/pkgconfig)
```

Build — **use the MPI compiler wrappers** (this is what fixed our local build; a
plain g++ misses the MPI headers/libs that our direct `MPI_*` calls need):

```sh
mkdir build && cd build
CXX=mpicxx CC=mpicc cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON ..
make -j
```

If the PETSc module is older than 3.17.1, or was built against a different MPI
than the `openmpi` you loaded, use Path B instead.

## Path B — conda / mamba (robust fallback; matches the local dev workflow)

conda-forge gives a fully self-consistent petsc+openmpi+gdal+compiler stack, which
sidesteps module-version mismatches. This is the most reliable path if the modules
fight you.

```sh
mamba create -n wtm -c conda-forge \
    "petsc>=3.17" gdal openmpi "gxx_linux-64>=11" gcc_linux-64 cmake pkg-config make
mamba activate wtm

# make sure cmake finds conda's PETSc/GDAL, not a stray system copy:
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
```

Build with conda's MPI wrappers:

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON \
      -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/mpicxx \
      -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/mpicc ..
make -j
```

**Do NOT mix conda and system libraries.** Locally, a conda `libgdal` pulled in a
conda `libcurl` that lacked `CURL_OPENSSL_4` and broke linking against the system
`libhdf5`. With *everything* from conda-forge (or *everything* from modules), it's
consistent. If you see `undefined reference to curl_*@CURL_OPENSSL_4` or an
`mpi.h: No such file`, it's a mixed-toolchain symptom — pick one source.

## Running (Slurm)

Launch with the same modules/env loaded in the batch script. The memory win only
shows with `>1` rank:

```sh
# interactive example
srun -n 8 ./wtm.x <config.cfg>
# or mpirun -n 8 ./wtm.x <config.cfg>   (per MSI's MPI-launch guidance)
```

For production at 141M cells, the point of the distributed-ArrayPack work is that
non-root ranks no longer hold the full grid — so you can use many more ranks per
node than before (e.g. all 32 on msilong, formerly ~4).

## Verify the build is correct

The test suite confirms the build (including the MPI flip) end-to-end. It needs
`python3` with `rasterio` (`pip install rasterio` or `mamba install -c conda-forge rasterio`):

```sh
cd tests
./run_all.sh              # unit + ghost-cell + mass-balance + consistency + golden
```

All six suites should report PASS at n = 1..8. `tests/run_unit_tests.sh` alone
(the DMDA gather/scatter unit tests) is a fast smoke test that only needs the
built `test_dmda.x`.

## Quick troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mpi.h: No such file` | compiler isn't the MPI wrapper | `CXX=mpicxx CC=mpicc` |
| `undefined reference ... MPI_*` at link | linked without MPI | use `mpicxx`, or ensure PkgConfig::PETSC pulls MPI |
| `curl_*@CURL_OPENSSL_4` link error | conda vs system lib mix | one toolchain (all conda or all modules) |
| CMake can't find PETSc | pkg-config path | `export PKG_CONFIG_PATH=.../lib/pkgconfig` |
| `PETSc>=3.17.1 ... not found` | module too old | Path B (conda petsc>=3.17) |
| C++20 errors | gcc too old | load newer gcc module / conda `gxx_linux-64` |
