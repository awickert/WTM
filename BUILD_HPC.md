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

## Quickstart — MSI (verified module set, 2026-07)

The exact modules that give a consistent toolchain on MSI (Agate, Rocky 8). See
"Path A" below for how these were found and what to substitute on another cluster.

```sh
# 0. Clone the fork + branch + submodules (git may itself be a module)
module load git
git clone --recurse-submodules -b solver-optimization-2 \
    https://github.com/awickert/WTM.git WTM

# 1. Get an interactive compute node (do NOT build on the login node)
srun -N 1 --ntasks-per-node=8 --mem-per-cpu=4gb -t 2:00:00 -p interactive --pty bash

# 2. Load the toolchain. Load PETSc FIRST -- it auto-loads gcc/11.3.0 and a
#    serial HDF5, and provides its own MPICH wrappers (mpicc/mpicxx/mpiexec in
#    $PETSC_DIR/bin). Then add the gcc-11.3.0-matched GDAL (shares that HDF5) and CMake.
module load petsc/3.24.5-gnu-rocky8
module load gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3
module load cmake/3.29.2-rocky8
module list                       # sanity: gcc/11.3.0 + petsc + gdal all present

# 3. Verify the toolchain BEFORE building
pkg-config --modversion PETSc     # 3.24.5  (>= 3.17.1 required)
which mpicc mpicxx                # $PETSC_DIR/bin/... (PETSc's MPICH)
gdal-config --version             # 3.12.1

# 4. Build -- use PETSc's MPI compiler wrappers (the code makes direct MPI_* calls)
cd WTM && mkdir build && cd build
CXX=mpicxx CC=mpicc cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON ..
make -j 8

# 5. Confirm the build (incl. the MPI flip); needs python3 + rasterio
cd ../tests && ./run_all.sh
```

Module names change over time; if these exact versions are gone, use the
discovery steps in Path A to find the current equivalents. If no PETSc >= 3.17.1
exists, use the conda path (Path B).

**Per-session shortcut:** after the first successful setup, `msi_env.sh` (in the
repo root) loads this whole toolchain in one line — `source msi_env.sh` to build
or run, or `source msi_env.sh test` to also activate the `wtmtest` conda env for
the Python test suite.

## Dependencies

External (must be provided by modules or conda):

| Dep | Constraint | Notes |
|-----|-----------|-------|
| C++ compiler | **C++20** (`gcc`/`g++` ≥ 11; 13 ideal) | `cxx_std_20` in CMakeLists |
| MPI | any (OpenMPI, MPICH, Intel) | **must be the same MPI PETSc was built against** — often bundled *inside* the PETSc install (MSI: MPICH, wrappers in `$PETSC_DIR/bin`) |
| PETSc | **≥ 3.17.1**, real scalars, MPI | found via `pkg-config` (`PETSc.pc`) |
| GDAL | any recent | `find_package(GDAL REQUIRED)` |
| CMake | ≥ 3.16 | |
| pkg-config | | used to locate PETSc |

Bundled (git submodules — no module needed, but you MUST clone them):

- `common/richdem`, `common/fmt` → clone with `--recurse-submodules`.

The sanitizer CMake modules are vendored in `cmake/`, so `find_package(Sanitizers)`
needs nothing external.

## Step 0 — clone the fork + branch (with submodules)

`git` itself may be a module on the cluster (the system git can be old or absent):

```sh
module -t avail 2>&1 | grep -i git      # find the exact git module name
module load git         # if a git module is listed
```

Cloning from GitHub needs auth — either an SSH key registered with your GitHub
account (then use the `git@github.com:` URL) or a personal access token (for the
`https://` URL). Clone the fork, the branch, and the submodules in one go:

```sh
# WTM fork + working branch:
git clone --recurse-submodules -b solver-optimization-2 \
    https://github.com/awickert/WTM.git WTM
cd WTM

# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

Make sure the branch was **pushed to the fork first** (`git push` from wherever
the commits live) — a clone only sees what is on the remote.

## Step 1 — get onto a compute node (MSI: interactive session)

Do **not** build or run on the login node. Grab an interactive session on a
compute node first. MSI uses Slurm; the `interactive` partition is for exactly
this (the job ends when the shell exits).

```sh
# from your laptop: ssh to an MSI login host
ssh <username>@agate.msi.umn.edu          # or the current login host per MSI docs

# request an interactive shell: 1 node, 8 cores, 4 GB/core, 2 hours, interactive partition
srun -N 1 --ntasks-per-node=8 --mem-per-cpu=4gb -t 2:00:00 -p interactive --pty bash
```

You are now on a compute node with 8 cores allocated — enough to build
(`make -j`) and smoke-test with a handful of MPI ranks. Load modules and build
(Path A or B below) **inside this session** so the build sees the compute node's
environment. To run the model on the allocated cores from the interactive shell:

```sh
srun -n 8 ./wtm.x <config.cfg>            # Slurm launches 8 MPI ranks
# or, within the allocation:
mpirun -n 8 ./wtm.x <config.cfg>
```

(On another cluster, substitute your site's login host, partition name, and
interactive-job command — `salloc`/`srun --pty` are the usual variants.)

## Path A — cluster modules (try this first; MSI shown)

**MSI uses classic Environment Modules (Tmod), not Lmod** — there is **no
`module spider`** (it errors with `Invalid command 'spider'`), and the tree is
flat, so everything shows at once. Two consequences:

- `module avail` dumps the *entire* (huge) list and can flood the terminal.
  Always **filter**, and note that `avail` prints to **stderr** (so `2>&1`):

  ```sh
  module -t avail 2>&1 | grep -iE 'petsc|openmpi|ompi|mpi|gdal|gcc|cmake'
  # ( -t = terse, one per line; add "| less" if still long )
  ```

- `module load openmpi` failing with `Unable to locate a modulefile` just means
  that exact name doesn't exist — the real name has a version or path suffix
  (e.g. `ompi/4.1.5/gnu`). Use the names the grep above prints.

**Load PETSc first — it pulls its own toolchain.** On MSI, `petsc/3.24.5-gnu-rocky8`
auto-loads `gcc/11.3.0`, a serial HDF5, and — crucially — **bundles its own MPICH**
(wrappers at `$PETSC_DIR/bin/{mpicc,mpicxx,mpiexec}`); there is **no standalone
`openmpi` module** on MSI, and you don't need one. So the discovery order is:
find PETSc, load it, and read what it brought:

```sh
module load petsc/<version>
module list                       # shows the gcc + (maybe) MPI it auto-loaded
which mpicc mpicxx                 # PETSc's wrappers should be on PATH
```

**Confirm the PETSc is *parallel*, not a serial MPIUNI stub** (a serial build
would compile WTM but run on 1 rank, defeating the point). The `which mpicc mpicxx`
above succeeding is the first sign; confirm with:

```sh
grep -iE 'MPIUNI|HAVE_MPI(CH|_)' $PETSC_DIR/include/petscconf.h
# good: shows PETSC_HAVE_MPICH 1 (or another real MPI) and NO PETSC_HAVE_MPIUNI 1
```

Then add a GDAL and CMake. **Match GDAL to PETSc's compiler** so they share one
HDF5/netcdf lineage (avoids a mixed-HDF5 link clash): on MSI,
`gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3` is built with the same `gcc-11.3.0` as PETSc
and reuses PETSc's `hdf5-gcc-11.3.0-serial`. CMake is only the build driver, so
any recent one works (`cmake/3.29.2-rocky8`).

```sh
module load gdal/<gcc-matched-version> cmake/<recent>
module list                       # verify gcc did NOT get bumped to a different version
```

If the grep shows **no PETSc, or only a version < 3.17.1, or only serial/MPIUNI
builds**, use Path B (conda) — its `petsc>=3.17` is MPI-parallel by default.

**Verify the toolchain is consistent BEFORE building:**

```sh
pkg-config --modversion PETSc     # must print >= 3.17.1  (MSI: 3.24.5)
which mpicc mpicxx                # PETSc's wrappers (MSI: $PETSC_DIR/bin/...)
gdal-config --version             # MSI: 3.12.1
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

## Running

The memory win only shows with `>1` rank. Interactive testing (inside the
Step-1 session) is just:

```sh
mpiexec -n 8 ./wtm.x <config.cfg>      # PETSc's own launcher ($PETSC_DIR/bin/mpiexec)
# or with Slurm:  srun --mpi=pmi2 -n 8 ./wtm.x <config.cfg>
```

**Launcher note (MSI):** the MPI is **MPICH** (bundled with PETSc), so use PETSc's
`$PETSC_DIR/bin/mpiexec`, or `srun --mpi=pmi2` — MPICH under Slurm typically needs
the `pmi2` PMI. A plain `srun ./wtm.x` may fail to wire up the ranks; `mpiexec`
inside an interactive allocation is the reliable smoke test.

Production runs go through `sbatch` on a production partition (msilong for long
single-node runs, msismall for shorter single-node, msilarge for multi-node).
Load the **same** modules/env you built with. Example `run_wtm.sbatch`:

```bash
#!/bin/bash -l
#SBATCH --job-name=wtm
#SBATCH --partition=msilong        # long single-node; or msismall / msilarge
#SBATCH --nodes=1
#SBATCH --ntasks=32                # MPI ranks (msilong caps at 32 cores / 128 GB)
#SBATCH --mem=120gb
#SBATCH --time=7-00:00:00          # D-HH:MM:SS; msilong allows up to 37 days
#SBATCH --output=wtm-%j.out

# the SAME modules used to build (MSI):
module load petsc/3.24.5-gnu-rocky8 gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3 cmake/3.29.2-rocky8
# (or: source activate wtm, if you built via the conda path)

cd /path/to/WTM/build
srun --mpi=pmi2 ./wtm.x /path/to/config.cfg   # MPICH under Slurm; uses the 32 ranks
```

Submit and monitor:

```sh
sbatch run_wtm.sbatch
squeue --me
```

For production at 141M cells, the point of the distributed-ArrayPack work is that
non-root ranks no longer hold the full grid — so you can use many more ranks per
node than before (e.g. all 32 on msilong, formerly ~4). Set `--ntasks` to the
cores you want; on a bandwidth-bound stencil the useful count may plateau below
the node maximum, so a quick scaling sweep (8, 16, 32) is worth doing once.

## Verify the build is correct

The test suite confirms the build (including the MPI flip) end-to-end. It needs
`python3` with `rasterio` + `numpy` (to compare output rasters). This is
**decoupled from the C++ build** — rasterio just reads the output GeoTIFFs — so it
does not need to match the build toolchain.

**On MSI, use the anaconda Python module, not the plain `python/*-gcc-*` ones.**
The plain Python modules are each tied to a *different* gcc (13.1.0, 8.2.0, …) and
loading one can swap your `gcc/11.3.0`, breaking `wtm.x` at runtime. The anaconda
module is self-contained and won't swap gcc:

```sh
module load python/3.10.9_anaconda2023.03_libmamba
conda init bash                              # once; then: source ~/.bashrc
conda config --set auto_activate_base false  # so base never auto-activates (would shadow future builds)
conda create -n wtmtest -c conda-forge rasterio numpy
conda activate wtmtest

# confirm nothing got shadowed:
module list         # WTM modules (petsc/gdal/gcc-11.3.0) still loaded
which mpirun        # must be $PETSC_DIR/bin/mpirun (PETSc's MPICH), NOT a conda one
python3 -c "import rasterio, numpy; print('ok', rasterio.__version__)"
```

Then run the suite (keep the WTM modules loaded; `wtmtest` supplies only rasterio):

```sh
cd tests
./run_all.sh              # unit + ghost-cell + mass-balance + consistency + golden
```

All six suites should report PASS at n = 1..8. `tests/run_unit_tests.sh` alone
(the DMDA gather/scatter unit tests) is a fast smoke test that only needs the
built `test_dmda.x` — no Python.

(Off MSI, a plain-Python `venv` + `pip install rasterio numpy` is cleaner than
conda, since a venv doesn't touch `LD_LIBRARY_PATH` and so can't shadow the
compiled `wtm.x`. On MSI the anaconda module wins only because the plain-Python
modules would swap gcc.)

## Quick troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `mpi.h: No such file` | compiler isn't the MPI wrapper | `CXX=mpicxx CC=mpicc` |
| `undefined reference ... MPI_*` at link | linked without MPI | use `mpicxx`, or ensure PkgConfig::PETSC pulls MPI |
| `curl_*@CURL_OPENSSL_4` link error | conda vs system lib mix | one toolchain (all conda or all modules) |
| CMake can't find PETSc | pkg-config path | `export PKG_CONFIG_PATH=.../lib/pkgconfig` |
| `PETSc>=3.17.1 ... not found` | module too old | Path B (conda petsc>=3.17) |
| C++20 errors | gcc too old | load newer gcc module / conda `gxx_linux-64` |
