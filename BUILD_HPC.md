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

## Step 0 — clone the fork + branch (with submodules)

`git` itself may be a module on the cluster (the system git can be old or absent):

```sh
module avail git        # or: module spider git
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

**MSI (and most modern clusters) use a *hierarchical* module tree (Lmod):** MPI
and libraries built on top of a compiler are **hidden until you load that
compiler**. So `module load openmpi` on a fresh shell fails with
`Unable to locate a modulefile for 'openmpi'` — openmpi does not exist until a
gcc is loaded. Load in dependency order: **compiler → MPI → PETSc/GDAL**.

`module spider` is the tool that cuts through this: it searches the *entire*
tree (including hidden modules) and prints the exact prerequisites to load first.

```sh
module avail                 # what's directly loadable right now (compilers, cmake, …)
module spider openmpi        # finds openmpi even when hidden; prints "load gcc/X first"
module spider petsc          # same for PETSc, and shows its version (need >= 3.17.1)
```

Then load in order (re-run `module avail` after each step to see what unlocked):

```sh
module load gcc/<11-or-newer>   # a C++20-capable gcc (e.g. gcc/13.1.0)
module avail                    # openmpi now appears, built for that gcc
module load openmpi
module load petsc gdal cmake    # petsc typically appears only after gcc + openmpi
```

If `module spider petsc` finds nothing, or only a version < 3.17.1, use Path B
(conda) — it avoids the module hierarchy entirely.

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

## Running

The memory win only shows with `>1` rank. Interactive testing (inside the
Step-1 session) is just:

```sh
srun -n 8 ./wtm.x <config.cfg>
# or mpirun -n 8 ./wtm.x <config.cfg>
```

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

module load gcc openmpi petsc gdal cmake   # the SAME modules used to build
# (or: source activate wtm, if you built via the conda path)

cd /path/to/WTM/build
srun ./wtm.x /path/to/config.cfg           # uses the allocation's 32 ranks
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
