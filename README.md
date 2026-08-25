[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4265369.svg)](https://doi.org/10.5281/zenodo.4265369)

# The Water Table Model (WTM)

***This model combines groundwater flow and dynamic lake simulation to output the elevation of the water table relative to the land surface at a given time.***

The model is intended for determining the depth or elevation of the water table, given a certain topography and set of climate inputs. Water table can be below ground (groundwater) or above ground (lake surfaces).

The model works by coupling groundwater and dynamic lake components. The groundwater component moves water cell-to-cell by solving the 2D groundwater flow equation for a heterogeneous, horizontally isotropic medium. It uses PETSc's SNES component to do so. The model is structured with a single layer of vertically integrated hydraulic conductivity.

The dynamic lake component was collaboratively written by R Barnes and KL Callaghan. It works by creating a hierarchy of depressions for the topography, and then allowing water to move across the land surface, filling depressions and spilling from one depression into another. For more details on the depression hierarchy, see:

**Barnes, R, Callaghan, KL, and Wickert, AD, (2020), [Computing water flow through complex landscapes, Part 2: Finding hierarchies in depressions and morphological segmentations](https://esurf.copernicus.org/articles/8/431/2020/), *Earth Surf. Dynam.*, doi:10.5194/esurf-8-431-2020**

More details on the surface-water component, Fill-Spill-Merge, are available at:
**Barnes, R, Callaghan, KL, and Wickert, AD, (2020), [Computing water flow through complex landscapes, Part 3: Fill-Spill-Merge: Flow routing in depression hierarchies](https://esurf.copernicus.org/preprints/esurf-2020-31/), *Earth Surf. Dynam. Discuss.*, doi:10.5194/esurf-2020-31 **

This code has not been tested on Windows and may only work on Unix-based systems.

Please contact us if you have questions or suggestions!

## Required data inputs

Data inputs are all in a Geotiff (.tif) format. 

The following files are required:
* Topography - elevation in metres
* Slope - determined from your topography
* Mask - indicating the location of land (1) and ocean (0)
* Precipitation - in metres per year
* Evapotranspiration - in metres per year
* Winter air temperature - in degrees Celsius
* Hydraulic conductivity - in metres per second
* Porosity - unitless
* Open-water evaporation - in metres per year

## Dependencies

* The C++ compiler g++
* GDAL
* RichDEM

## Downloading with dependencies

The best way to obtain this code is by cloning the repository.

Before starting, note that in order to include RichDEM from GitHub, you will need a Public Key associated with your account. Instructions to do so can be found here.
https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

Clone with submodule dependencies (RichDEM) included:
```sh
git clone --recurse-submodules https://github.com/KCallaghan/WTM
```

If you forget to do this and just run a normal `git clone`, you can still pull the submodules:
```
git submodule update --init --recursive
```

In either case, use the following to update the submodules:
```
git pull --recurse-submodules
```

## Compilation
To build with `cmake` use:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_GDAL=ON ..
make
```
Use `-DSANITIZE_ADDRESS=On` to enable addressing sanitizing.

Alternatively, to build with `ninja`, use:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DRICHDEM_LOGGING=ON  -GNinja ..
ninja
```

For building on HPC clusters (e.g. MSI) with module- or conda-based toolchains, and for running with MPI under Slurm, see [BUILD_HPC.md](BUILD_HPC.md).

## Running the code
Ensure that all of the data files are located appropriately in a folder together, then edit `config.yaml`. **The configuration is nested YAML** — settings are grouped into sections (`run`, `time`, `io`, `output`, `boundaries`, `transmissivity`, `evaporation`, `surface_water`, `solver`, `parallel`), and the annotated `config.yaml` is the authoritative reference: copy and edit it. Grid geometry is derived from the input GDAL geotransform. NOTE: the flat variable notes in this section describe the earlier `key value` format — several keys have moved into sections or been removed (`maxiter` / `total_cycles` are gone; `cells_per_degree` / `southern_edge` are now automatic; input naming is `io.source` / `io.region` / `io.time_start`) — so use `config.yaml` as the source of truth. The main settings:

* textfilename       {The name of your output text file.txt}
* outfile_prefix     {The name of your output depth-to-water-table file in geoTiff format. The code will append the time passed and .tif extension.}
* cells_per_degree   {How many cells per degree in your data. E.g. One-degree resolution will be 1. 5 arcsecond resolution will be 12.}

time_start, surfdatadir, and region are all to help you name your input files or place them in a specific folder. The code will look for data files in the following format:
surfdatadir + region + time_start + "\_suffix.tif",
where the suffix refers to the specific file (topography, mask, precipitation, evaporation, winter_temperature, slope, open_water_evaporation, porosity, or ksat).
An example of a file path would be: "surfdata/North_America_10000_topography.tif".
In this case, you would set:

* surfdatadir        surfdata/
* region             North_America_
* time_start         10000

Two run types are possible: equilibrium, and transient. An equilibrium run assumes that the topography and climate are not changing and runs for many iterations until the equilibrium condition for the water table is found. Set the run_type parameter to 'equilibrium'.
A transient run requires a starting depth to water table as an additional input. The algorithm will then run for a set number of iterations, to represent a number of years passing, and output the new water table under a changing set of climatic and topographic conditions. In this case, both start and end states are required for all file inputs. Set the time_end parameter to lead to the files at the end time of the transient run, while time_start leads to the files at the initial time of the transient run. The run_type parameter should be set to 'transient'.
Other parameters include:

* deltat             {Number of seconds per time step, e.g. 315360000 for a 10-year time step}
* southern_edge      {Southern-most latitude of your domain in decimal degrees, e.g. 5}
* maxiter            500                  {How many times GW should run before FSM runs}
* total_cycles       1000                 {how many times FSM should run before completion}
* infiltration_on    0                    {true is 1, false is 0. Only recommend true for high-resolution input data.}
* fdepth_a           200                  {e-folding depth coefficients}
* fdepth_b           150                  {e-folding depth coefficients}
* fdepth_fmin        2                    {e-folding depth coefficients}

Once the configuration file has been set up appropriately, simply open a terminal and type
```
# Optionally set the number of CPU threads for the parallel groundwater solve:
# export OMP_NUM_THREADS=N
./build/wtm.x config.yaml
```
Here, N is the number of CPU threads you want the parallel processing for the groundwater-flow step to use. In the above line, you are setting an environment variable that will define this until you exit the terminal window.

The model chooses sensible solver defaults, so no PETSc solver flags are required on the command line.
The **default solver is the matrix-free Anderson** path, for both run types: it is robust across regimes,
bit-exact across MPI ranks, and carries the exact in-residual exfiltration constraint (`runoff_collector` implicit).
It is 1st-order-in-time (backward-Euler cc) — the right choice for equilibrium, where a 2nd-order step
oscillates at the free surface.

For **large, stable time steps** (fast equilibrium spin-up) and **2nd-order-in-time transients**, opt into
the semi-implicit **BDF2-on-V (Picard)** solver with **`-wtm_bdf2_on_V`**: its Newton + algebraic-multigrid
solve has a nearly step-size-independent cost, so `deltat` can be raised by orders of magnitude. (Picard is
also cross-rank deterministic to ~1e-9 on FSM-routing-threshold cases, so it is the grounding reference the
golden tests hold Anderson against.) `-wtm_anderson -wtm_bdf2_on_V` gives the 2nd-order BDF2-on-V
discretization on the matrix-free residual (time discretization is decoupled from the solver). Any explicit
PETSc `-snes_*` option (e.g. `-snes_stol`, `-snes_anderson_beta`) still overrides the defaults. For a more strongly damped (L-stable, non-ringing) 2nd-order
option, **`-wtm_tr_bdf2`** runs TR-BDF2 (two staged solves per step); in testing it took twice the stable
time step of BDF2-on-V with fewer iterations near the limit, at a modest per-step cost.

For hard equilibrium **cold starts on stiff terrain** — a deep, far-from-equilibrium initial water table
(for example a cold start of `wtd = 0`) on steep, heterogeneous topography, where the solver struggles to
take the first large steps — add **`-wtm_stiff`**. It bundles the analytic-Jacobian **Newton** solver with
*dt-continuation*: `deltat` starts small (keeping the far initial guess within the Newton basin) and ramps
up automatically as the table settles, stopping at equilibrium on its own. It is shorthand for
`-wtm_newton -wtm_dt_continuation -wtm_eq_tol 0.01`, and each piece can be overridden individually.

An experimental conditioning option, **`-wtm_Tbar`**, addresses the same stiffness from a different angle:
it uses each cell's *step-time-averaged* transmissivity (the Kirchhoff-potential difference over the step)
for the interblock flux instead of the instantaneous start-of-step value. The exponential transmissivity
is what makes the frozen-coefficient iteration oscillate on stiff steps; averaging it over the step damps
that oscillation. It composes with any solver, changes nothing at equilibrium (the time-average collapses
to the instantaneous value as the table settles), and requires the piecewise transmissivity (it is refused
with the smoothing / extended-soil / Kirchhoff options). See `benchmark/TBAR_TIME_AVERAGING.md`.

There will be some on-screen outputs to indicate the first steps through the code, after which values of interest will be output to the text file and an updated geoTiff output file will be saved every X iterations (X is set in the configuration file).

## Example of a full config file:
```
textfilename       my_model_run.txt     #The name of the output textfile, which will include printed values describing change in the water table.
outfile_prefix     my_model_run_        #Output rasters are named "<prefix><cycle:09>_<year>yr.tif", e.g. my_model_run_000000010_10yr.tif.
cells_per_degree   60                   #how many cells in one degree. This example for 1 arcminute cells.
#the below parameters are used in import file names. The code searches for files in the format:
#surfdatadir + region + time_start + input_type.tif,
#where the input_type is each of the input files discussed above (e.g. topography, precipitation, etc).
#For equilibrium runs, only time_start is used. For transient runs, time_start and time_end indicate
#the two sets of files with data at the beginning and at the end of the time period for the transient run.
time_start         021000               #used in filenames for import
time_end           020000               #used in filenames for import
surfdatadir        surfdata/            #used in filenames for import
region             my_area_             #used in filenames for import
deltat             31536000             #seconds in your timestep. This example for 1 year.
run_type           equilibrium          #test, equilibrium or transient
southern_edge      -52                  #Southern-most latitude of your import files
maxiter            1                    #how many times GW should run before FSM runs. Optionally, the groundwater can move multiple times before running FSM for
surface water. This is to save time on computation.
total_cycles       5000                 #how many times FSM should run before completion.
fdepth_a           100                  #e-folding depth coefficients
fdepth_b           150                  #e-folding depth coefficients
fdepth_fmin        2                    #e-folding depth coefficients
#should water be allowed to infiltrate during overland flow in FSM?
infiltration_on    0                    #true is 1, false is 0. Only recommend true for high-resolution input data.
#Are you supplying a starting water table? Note that you MUST supply a starting water table for transient runs.
#For equilibrium runs, you can optionally supply a starting water table; if you do not, water table will initialise = 0 everywhere.
supplied_wt        1                    #1 if you are supplying a starting water table, 0 if not
#Should surface water be moved using FSM? Only recommend turning this off for testing purposes.
fsm_on             1                    # 1 to enable Fill-Spill-Merge for routing surface water is enabled; 0 otherwise.
#Is water allowed to gather in lakes, with lake evaporation removing some portion of it?
#If this is set to 0, all surface water will be removed from the domain.
evap_mode          1                    # 1 to use a grid of potential evaporation for lakes; 0 to remove all surface water.
#How is above-surface water routed to runoff (the exfiltration constraint)? Optional; default is
#active_set on Anderson, explicit on Picard/Newton. See "Solution modes" and "Surface-water routing" below.
#See "Surface-water routing" below.
runoff_collector   implicit              # implicit (in-residual exfiltration, exact; default) | explicit (post-solve clamp) | off (nonphysical) | legacy (old band sink)
```

## Surface-water transition (smooth tapers, on by default)
At the land surface (water-table depth `wtd = 0`) WTM smooths the transition between groundwater and
surface water with implicit, order-preserving **tapers**, which replace the old hard `wtd = 0` switch.
The **evaporation** tapers (2 & 3) are **on by default** and are controlled by command-line `-wtm_*` flags
(each disabled with `<flag> 0`). The **exfiltration** at the surface — taper 1's old job — is now the
`runoff_collector` config-file selector (see "Surface-water routing" below; default is the exact in-residual
face). The legacy sub-surface band sink is reached with `runoff_collector legacy`:

- **Taper 1 — sub-surface band sink** (`-wtm_surface_sink`; the legacy exfiltration, off unless
  `runoff_collector legacy`): a smooth removal in a band that holds the table at/below the surface and hands
  exfiltrated water to Fill-Spill-Merge. Preserves 2nd-order time accuracy across the surface, but its band
  width scales with `deltat` (so the equilibrium is dt-dependent — the reason `runoff_collector` replaced it).
  Peak removal `-wtm_surface_sink_qmax` (default 1 m/yr); band width `-wtm_surface_sink_width`.
- **Taper 2 — demand-identity evaporation** (`-wtm_evap_taper`): a single smooth transition from
  land-surface evapotranspiration (below the surface) to open-water evaporation (at/above it),
  replacing the hard ET↔open-water switch. This is what makes lake formation identical regardless of
  MPI rank count.
- **Taper 3 — accessibility / extinction depth** (`-wtm_extinction`, depth `-wtm_extinction_depth`,
  default 8 m): lets an arid water table (evaporative demand > precipitation) draw down via phreatic ET
  only within the extinction depth, rather than without bound.

Running with any combination other than all three on prints a warning explaining the consequence
(arid-unsafe, no-effect, or the legacy hard-switch model). The tapers work with either `evap_mode`; in
`evap_mode 0` the taper governs evaporation in place of the hard "remove all surface water" step. See
`benchmark/SURFACE_SINK_DESIGN.md` for the full derivation.

## Solution modes: each solver plus the mechanisms it needs to work

A *solution mode* is not just a solver. Each one comes with the machinery that makes it work — the
exfiltration enforcement it can support, and the cold-start treatment it needs. Choosing a solver
selects the bundle; you only override a piece if you mean to.

| | **Anderson** (default) | **Picard** (BDF2-on-V) | **Newton** (analytic Jacobian) |
|---|---|---|---|
| select with | *(default)* | `-wtm_picard -wtm_bdf2_on_V` | `-wtm_newton` |
| solve | matrix-free | assembled operator + linear solve | assembled Jacobian + linear solve |
| exfiltration enforcement | **`active_set`** | **`explicit`** | **`explicit`** |
| cold start from far | works as-is | needs **`-wtm_Tbar`** (log-mean transmissivity) | needs **dt-continuation** (`-wtm_stiff`) |
| order in time | 1st (BE); 2nd via `-wtm_tr_bdf2` / `-wtm_bdf2_on_V` | 2nd | 1st |

**Why each gets the enforcement it does.** The semismooth `active_set` pin is wired into the
matrix-free residual only; the Picard operator and Newton Jacobian carry no tangent for it, and
selecting it also switches every collector removal off — so on those solvers the constraint would be
*unenforced* and Newton aborts. They fall back to `explicit`, not to `implicit`, and that is a measured
choice: on the multi-lake fixture, halving `dt` leaves `active_set` and `explicit` topologically stable
(4 → 4 and 6 → 6 lakes) while **`implicit` changes the lake count, 6 → 5**. Note that `explicit` and
`active_set` still disagree with each other on the answer (6 lakes vs 4); `explicit` is the best
*available* enforcement on those solvers, not an equivalent one.

**Cold starts are a property of the mode, not of the problem.** Plain Picard and plain Newton both fail
from a cold start at production `dt` on every enforcement tested — Picard hits the iteration cap,
Newton aborts. Those failures are the frozen-coefficient contraction and large-step overshoot
respectively, *separate diseases* from the exfiltration constraint, and `-wtm_Tbar` and dt-continuation
are their respective cures. Neither is optional if you start far from equilibrium.

## Surface-water routing (`runoff_collector`)
Above-surface water leaves the subsurface where the water table reaches the land surface — it
**exfiltrates** — and is routed to runoff / Fill-Spill-Merge. The constraint is that head may not exceed
`topo + surface_water_depth`: one surface, equal to the lake stage inside a depression, sea level on
land below sea level, and the land surface elsewhere. The config key
`surface_water.collection.method` selects **how** that one boundary condition is enforced.

**Default is `active_set`, resolved per solver** (see the table above): `active_set` on Anderson,
`explicit` on Picard/Newton. Set the key explicitly to override; an explicit choice is always honoured.

- **`active_set`** (default on Anderson) — the constraint solved *inside* the residual as a semismooth
  complementarity condition, rather than approximated. No `dt` appears in it, so it carries no spurious
  `dt`-dependence; it also eliminates the between-step Fill-Spill-Merge shock and is 2–100× cheaper
  across solvers. Matrix-free (Anderson) only.
- **`implicit`** (the former default) — an in-residual siphon removing at rate `max(0,wtd)/dt`. Because
  that is a *rate*, the retained head is ~**linear in `dt`** (1.97 / 0.68 / 0.34 m at `dt` = 1, 1/3, 1/6
  week), and Fill-Spill-Merge routes that `dt`-dependent excess into a different set of lakes. Wired
  into the Anderson residual and the Picard operator; **not** into the Newton Jacobian.
- **`explicit`** (the robust clamp) — a post-solve clamp: works on **every** solver, and is `dt`-stable
  in lake topology. Lower-order (the flow field never feels the pin during the solve).
- **`off`** — no collection; above-surface water piles up. **Nonphysical**, testing only (warns loudly).
- **`legacy`** — the pre-selector `-wtm_surface_sink` band-sink defaults (dt-scaled; kept for the taper tests).

The modes are mutually exclusive (no hidden backstop), so a misbehaving enforcement shows visibly
rather than being masked. See `benchmark/SURFACE_WATER_ROUTING.md` for the measurements.

## Command-line flag reference

Every runtime option below is a PETSc-style flag passed **after** the config file, not a config-file key:

```
./build/wtm.x config.yaml -wtm_anderson -wtm_tr_bdf2 -wtm_eq_tol 0.001
```

The model runs correctly with **no flags at all** (the default column marks what is active out of the box).
Standard PETSc `-snes_*` / `-ksp_*` / `-pc_*` options are also accepted and override the WTM defaults. Boolean
tapers that are on by default are disabled by passing the flag with a `0` argument (e.g. `-wtm_surface_sink 0`).

The **Status** column is a guide to intended audience:
*default* = active unless switched off · *opt-in* = production-supported, off by default · *tuning* = a numeric
knob for another flag · *experimental* = works but not validated for production · *developer* = diagnostic or
deliberately nonphysical (prints a runtime warning).

### Solver selection
Exactly one solver runs. If none is named, the default (matrix-free **Anderson**, 1st-order cc) is used; an
explicit path flag wins, and Newton is mutually exclusive with Picard/Anderson.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_anderson` | **on** | default | Matrix-free Anderson mixing. Robust across regimes, bit-exact across MPI ranks, carries the exact in-residual exfiltration constraint. 1st-order-in-time unless `-wtm_bdf2_on_V` is added. |
| `-wtm_bdf2_on_V` | off | opt-in | Semi-implicit, volume-form BDF2 solved by Picard (Newton + algebraic multigrid). Large stable steps; 2nd-order in time; cross-rank deterministic (the golden reference). |
| `-wtm_newton` | off | opt-in | True Newton–Krylov on the analytic Jacobian (GMRES + multigrid). For cold starts from far, usually with `-wtm_dt_continuation`. |
| `-wtm_picard` | off | opt-in | Force the frozen-coefficient backward-Euler Picard operator explicitly (it is also the operator behind the default). |
| `-wtm_bdf2` | off | opt-in | Bare backward-looking BDF2 in head form (secant storativity), Picard operator. |
| `-wtm_tr_bdf2` | off | opt-in | L-stable TR-BDF2 (two staged solves per step) on the matrix-free residual; larger stable step, no ringing. |
| `-wtm_stiff` | off | opt-in | Convenience bundle for hard cold starts: shorthand for `-wtm_newton -wtm_dt_continuation -wtm_eq_tol 0.01`. |
| `-wtm_predict_guess` | off | experimental | Seed each step's initial guess by 2nd-order history extrapolation. |
| `-wtm_aa_picard` | off | experimental | Anderson-accelerated Picard via nonlinear preconditioning. |

### Time integration and step control
The dt controller is detached from the integrator: `-wtm_dt_adaptive` sizes `deltat` for whichever solver is
active. The `-wtm_dtc_*` knobs are read only when continuation is on, but also parameterize the adaptive controller.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_dt_adaptive` | off | opt-in | Error-controlled variable substeps (accept / reject / grow / shrink) around each step. |
| `-wtm_dt_continuation` | off | opt-in | Newton pseudo-transient ramp: start `deltat` small, grow after easy steps (requires `-wtm_newton`). |
| `-wtm_dt_tol` | 0.1 m (or `min(50·eq_tol, 0.5)`) | tuning | Target per-step error the adaptive controller holds `deltat` to. |
| `-wtm_dt_norm_rms` | off (MAX norm) | tuning | Use the RMS rather than max-cell norm for the adaptive error estimate. |
| `-wtm_dtc_dt0` | `deltat/200` | tuning | Starting step for the continuation ramp. |
| `-wtm_dtc_dt_max` | `1000·deltat` | tuning | Cap on `deltat` in the ramp and the adaptive controller. |
| `-wtm_dtc_grow` | 1.5 | tuning | Growth factor after an easy / accepted step. |
| `-wtm_dtc_shrink` | 0.25 | tuning | Shrink factor after a rejected step. |
| `-wtm_dtc_easy_iters` | 8 | tuning | "Easy step" threshold (converged in ≤ this many iterations) that permits growth. |
| `-wtm_dtc_max_retries` | 15 | tuning | Consecutive rejects allowed before the step is a hard failure. |

### Equilibrium detection (equilibrium runs)
| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_eq_tol` | 0.01 m (equilibrium); 0 (transient) | default | Per-cycle change threshold; the run stops after two consecutive settled cycles. `0` disables early stop. |
| `-wtm_eq_metric` | `frac` | default | How the per-cycle change is judged: `frac`, `max`, `rms`, or the pure-water-depth `water`/`water-max`/`water-rms` (weights head change by storativity: `\|S·Δwtd\|`, so deep low-storativity cells cannot dominate). |
| `-wtm_eq_frac` | 0.001 | tuning | For `-wtm_eq_metric frac`: allowed fraction of land cells still changing by more than `eq_tol`. |

### At-scale Anderson robustness
Aids for very large / stiff Anderson solves, where the mixing least-squares can go ill-conditioned near
convergence (the "flail"). All force the Anderson path.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_adaptive_restart` | off | opt-in | Restart Anderson's history proactively when the convergence *rate* degrades (ρ → 1), re-running from the best iterate; returns the best iterate near equilibrium rather than aborting. |
| `-wtm_ar_rho` | 0.9 | tuning | ρ = ‖F_k‖/‖F_{k-1}‖ threshold that flags rate degradation. |
| `-wtm_ar_patience` | 2 | tuning | Consecutive high-ρ iterations before a restart. |
| `-wtm_ar_max_it` | 40 | tuning | Iteration cap per Anderson phase before a forced restart. |
| `-wtm_ar_max_restarts` | 30 | tuning | Cap on the number of restart phases. |
| `-wtm_handoff` | off | opt-in | Run Anderson until it stalls, then hand the best iterate to a Newton finisher (nonlinear preconditioning). |
| `-wtm_handoff_picard` | off | opt-in | Use a Picard (CG + multigrid) finisher instead of Newton; implies `-wtm_handoff`. |
| `-wtm_handoff_patience` | 3 | tuning | Stalled Anderson iterations before the hand-off fires. |
| `-wtm_handoff_max_it` | 60 | tuning | Hard cap on phase-1 Anderson iterations before hand-off. |

### Transmissivity and storativity conditioning
| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_storativity_surface_smoothing_width` | 0.01 m | default | Rounds the storativity jump at the land surface (sub-grid roughness); always on. |
| `-wtm_ksat_surface_smoothing_width` | 0 (sharp) | opt-in | Rounds the **transmissivity** kink at the surface (`wtd = 0`). *(Named "ksat" for history; it smooths T, not raw conductivity.)* |
| `-wtm_ksat_soilbottom_smoothing_width` | 0 (sharp) | opt-in | Rounds the **transmissivity** kink at the soil bottom (−1.5 m). *(Same naming note.)* |
| `-wtm_Tbar` | off | experimental | Use each cell's step-time-averaged transmissivity (Kirchhoff-potential difference over the step) for interblock flux; damps stiff-step oscillation. Requires piecewise T (refused with the smoothing / extended-soil / Kirchhoff options). |
| `-wtm_T_bedrock` | 0 | opt-in | Additive background transmissivity floor (Manning–Ingebritsen); collapses the deep exponential-T range. |
| `-wtm_kirchhoff` | off | experimental | Solve in the discharge-potential variable Φ = ∫T dwtd on the Newton path. Retained for study; it worsens conditioning in practice. |
| `-wtm_volume_storage` | off | experimental | Anderson-only: use the exact volume change ΔV for backward-Euler storage instead of secant `S·Δh`. Identical below the surface; differs at a surface crossing. |
| `-wtm_relax` | 1.0 (off) | tuning | Post-solve under-relaxation `w ← a·w_solved + (1−a)·w_prev` (all solver paths); damps free-surface flicker. |

### Surface-water handling
Above-surface water is managed by the three default tapers (see the "Surface-water transition" section above)
plus optional routing modes. `-wtm_direct_to_runoff` supersedes the taper-1 sink;
`-wtm_dev_allow_aboveground_water_columns` switches both off.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_surface_sink` | **on** | default | Taper 1: smoothly holds the table at/below the surface, handing exfiltrated water to Fill-Spill-Merge. |
| `-wtm_surface_sink_qmax` | 1 m/yr | tuning | Peak removal rate of the taper-1 sink (also sets its default band width). |
| `-wtm_surface_sink_width` | auto (`2·qmax·deltat`) | tuning | Override the sink's band width below the surface. |
| `-wtm_evap_taper` | **on** | default | Taper 2: single smooth land-ET ↔ open-water-evaporation transition (makes lakes rank-count-independent). |
| `-wtm_evap_taper_wtdc` | 0.05 m | tuning | Half-rate depth of the ET transition. |
| `-wtm_evap_taper_s` | 0.1 m | tuning | Width of the ET transition. |
| `-wtm_extinction` | **on** | default | Taper 3: limits arid draw-down to within the extinction depth (requires taper 2). |
| `-wtm_extinction_depth` | 8 m | tuning | Depth below which phreatic ET is inaccessible. |
| `-wtm_direct_to_runoff` | off | opt-in | In-residual exfiltration constraint: route above-surface excess `max(0,wtd)/dt` straight to runoff (supersedes the taper-1 sink). |
| `-wtm_surface_exfiltration_to_runoff` | **on** (all paths) | default | Post-solve clamp: pin the table at/below the surface and route exact above-surface water to the runoff accumulator, keeping T clamped. Disable with `-wtm_surface_exfiltration_to_runoff false`. |
| `-wtm_dev_allow_aboveground_water_columns` | off | developer | Leave above-surface water unmanaged as nonphysical vertical columns (limit-cycles). Switches the two runoff clamps off; prints a warning. |

### Capillary fringe (taper-1 sink band width)
The `-wtm_fringe_*` knobs only take effect when `-wtm_fringe_source` is set to `fixed` or `ksat`.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_fringe_source` | `none` | opt-in | Per-cell sink band width: `none` (uniform `surface_sink_width`), `fixed` (uniform `fringe_length`), or `ksat` (capillary height from a pedotransfer estimate). |
| `-wtm_fringe_length` | 0.1 m | tuning | Uniform fringe height for `fringe_source fixed`. |
| `-wtm_fringe_ksat_coef` | 5e-4 | tuning | Coefficient in the `ksat` capillary-height estimate ψ_a = C·√(n/ksat). |
| `-wtm_fringe_cap` | 2 m | tuning | Upper cap on the `ksat` capillary height. |

### Boundary conditions and developer modes
Domain edges use the **mask-aware ghost-node boundary** by default (no flag needed): ocean edges are always
Dirichlet `h = 0` (sea level), and land edges default to terrain-following no-flow (Neumann). The land-edge
condition is selectable; the legacy sea-level-padding method is retained only as a verification tool.

| Flag | Default | Status | Effect |
|---|---|---|---|
| `-wtm_land_boundary` | `neumann_toposlope` | opt-in | Land-edge boundary condition: `neumann_toposlope` (terrain-following no-flow) or `dirichlet` (sea-level `h = 0` via ghost nodes — a land edge behaves as ocean). Ocean edges are always Dirichlet regardless. Works on all solver paths; not compatible with `-wtm_kirchhoff`. |
| `-wtm_dev_padded_dirichlet` | off | developer | Verification only: reproduce the legacy "1-cell sea-level padding" boundary (force every domain edge to ocean `h = 0`). **Requires an all-ocean domain boundary** and fails otherwise (it would discard edge land). On an ocean-ringed domain it coincides with the default mask-aware boundary — that equivalence is what it verifies. |
| `-wtm_extended_soil` | off | developer | Let transmissivity keep growing above the surface (skips the `wtd > 0` clamp). Testing only; prints a warning; refused with `-wtm_Tbar` / `-wtm_kirchhoff`. |

## Outputs
The program outputs a text file that provides information on the current minimum and maximum water table elevation, the changes in surface water and groundwater within the past iteration, and the number of iterations passed.
The main output is a geoTiff file that supplies the depth to/elevation of the water table. Negative values indicate a water table below the surface, while positive values indicate a water table above the surface (i.e. a lake).

## Completing a model run
A run always stops after `total_cycles` cycles. Equilibrium runs also stop *early*, on their own, once the
water table settles: after each cycle WTM measures the per-cycle change and, when it stays below a tolerance
for two consecutive cycles, declares equilibrium and stops. The tolerance is `-wtm_eq_tol` (default 0.01 m for
equilibrium runs; set to 0 to disable and run the full `total_cycles`), and `-wtm_eq_metric` selects how the
change is judged (default `frac`; see the flag reference below). To watch the settling without stopping early,
set `-wtm_eq_tol 0` and read the per-cycle change reported in the text output. Snapshots are written every
`cycles_to_save` cycles as `<outfile_prefix><cycle:09>_<year>yr.tif`, and any of them can seed a restart via
`supplied_wt 1` (point `region`/`time_start` at the snapshot).

## Development status and upstream porting

This is a research branch that explores solver, time-integration, and surface-physics improvements over
the upstream release (Callaghan's WTM, v2.0.1). Most additions are opt-in runtime flags and off by default,
so the production path is unchanged. The maintained checklist of which improvements are intended for
upstream — and *why* each (accuracy, speed, correctness, or other, quantified where measured) — is in
[`PORT_TO_UPSTREAM.md`](PORT_TO_UPSTREAM.md).
