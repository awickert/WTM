#!/usr/bin/env python3
"""Island equilibrium demo: implicit (2nd-order Picard) groundwater, serial == parallel.

Runs WTM to equilibrium on an island (ocean along every side) with the 2nd-order-in-time
implicit solver (`solver.method: picard` + `solver.time_integration: bdf2`, the BDF2-on-V path),
in serial and on N MPI ranks, and shows that the result is cross-rank consistent (identical to
floating-point-reduction noise) while producing the expected surface hydrology: lakes ponded in
closed depressions, rivers draining to the coast, and the ocean boundary.

MEASURED 2026-09-17, with slope derived from the DEM (see terrain_slope), against the 1e-6 m
threshold:

    region    solver/mode         n=4 max|dwtd|   n=8 max|dwtd|   lakes   max lake
    spectral  picard/fixed          1.851e-10 m     1.127e-10 m     37      8.3 m
    spectral  anderson/fixed        4.880e-11 m     1.174e-11 m     37      8.3 m
    spectral  anderson/adaptive     4.880e-11 m           --        37      8.3 m
    spectral  newton/ramp           1.386e-10 m           --        30      5.6 m
    corsica   picard/fixed        DID NOT COMPLETE -- DIVERGED_MAX_IT at 10000 iterations
    corsica   anderson/fixed        6.139e-12 m     6.594e-12 m    254     79.0 m
    corsica   anderson/adaptive   MISMATCH 1.331e-03 m     --      254     79.0 m
    corsica   newton/ramp           6.174e-08 m           --       165     10.7 m

TWO THINGS IN THAT TABLE ARE WORTH MORE THAN THE REST.

(1) ADAPTIVE IS NOT CROSS-RANK DETERMINISTIC ON THE HARD PROBLEM, and that is why this demo pins
    mode: fixed rather than taking the shipped default. The error estimate is an MPI reduction, so
    its last bits depend on how many ranks summed it; where the estimate sits near a decision
    boundary the two rank counts make DIFFERENT accept/grow choices, take different step sequences
    from there on, and end up 1.3 mm apart -- a thousand times the 1e-6 threshold. It is not a
    defect: the two runs agree on the physics (same 254 lakes, same 79.0 m maximum), they just are
    not the same arithmetic. On spectral, where the estimate never sits near a boundary, adaptive
    and fixed return the identical 4.880e-11, which suggests the controller never moved dt there at
    all -- stated as a reading of the number, not verified with a dt trace.

(2) THE SOLVERS DISAGREE ON THE ANSWER, AND THE REASON IS THAT NOBODY HAS CONVERGED. At cycle 14 of
    the corsica Newton run the log still reports abs_change_volume_max = 0.856 m, falling only
    0.859 -> 0.857 -> 0.856 over the last three cycles. 15 years from a saturated start on 2453 m of
    relief is nowhere near equilibrium, so each solver is showing a DIFFERENT POINT ON A MOVING
    TRAJECTORY, reached by a different step sequence. That is why newton/ramp reports 165 lakes and
    anderson 254.
    SO: the cross-RANK comparison in this demo is sound -- same solver, same steps, one variable.
    A cross-SOLVER comparison of these numbers is NOT, and nothing here should be read as one. To
    compare solvers you need them stopped by the same physical criterion rather than the same clock,
    which means running to a real equilibrium_stop.tol and accepting that the rank counts may then
    stop at different cycles.

AND THAT LAST PAIR IS THE POINT. Nothing was tuned to make Picard fail: the demo was made more
REALISTIC, by giving it the slope field a real DEM implies instead of slope = 0, and the split
appeared on its own. The two topographies now sit either side of the difficulty:

    spectral   median slope 0.0010, max 0.0025  ->  fdepth 145 .. 200 m
               A continental-scale gentle dome (1/10 degree cells, ~11 km). Barely any T contrast,
               and BOTH solvers handle it. This is the "it works, and it is deterministic" case.
    corsica    median slope 0.1341, max 0.7660  ->  fdepth 2 .. 200 m, median 9.5
               Real 30-arcsecond terrain over 2453 m of relief. fdepth now spans two orders, and
               T = fdepth * ksat * exp((wtd + 1.5) / fdepth) spans ~9.5e+05 at wtd = -20 m. Only 5
               of 14064 land cells reach the fdepth_fmin floor, so the floor is not what makes this
               hard -- the exponential is.

Picard freezes its coefficients within a solve, so a steep, spatially varying exp-T is exactly what
it cannot chase; this is the limiter recorded in the solver notes, reproduced here on real terrain
rather than argued. Anderson completes the same fixture and stays cross-rank consistent to 6e-12 m.

WHAT ADAPTIVE STEPPING IS FOR, DEMONSTRATED (--dt-weeks). The model documents adaptive stepping as
INTENDED to get through runs a fixed step cannot, and until 2026-09-17 that was design intent rather
than a measurement -- no synthetic fixture here had ever produced the case. Real terrain does.
Corsica, Anderson, serial, walking the step up in powers of two:

    dt        mode: fixed                                    mode: adaptive
    48 wk     completes                                      completes
    64 wk     completes                                      --
    128 wk    completes                                      --
    256 wk    completes                                      --
    512 wk    DIES: "The SNES solver has not converged"       completes (253 lakes, 79.0 m)
    1024 wk   DIES: "TR-BDF2 trapezoidal stage (1) did not    completes (253 lakes, 79.0 m)
                     converge"

At 512 weeks and beyond the fixed stepper is handed a step the solver cannot take and the run ends.
The adaptive controller is handed the SAME step as its starting point, finds it too large, subdivides
-- and finishes. That is robustness in the only sense that matters here: computes versus does not
compute, not a speed comparison at matched precision.

Note it took 512 weeks, not the 50 the idea was first sketched with. Corsica at 48 weeks is simply
not hard enough, which is worth knowing: the case exists but it is not near the working regime.

Note the lake count moved (211 -> 254 for Anderson) when slope became real: shallower fdepth means
less transmissivity, so water backs up into more depressions. Expected, and a physics change rather
than a numerical one.

CORSICA NEVER REACHES EQUILIBRIUM, AND THAT IS THE RIGHT ANSWER (measured 2026-09-17).
Run with --equilibrium and corsica reports "did not settle" however long you give it: over a full
20000 yr run the stop metric bottoms out at frac = 0.001351 against its 0.001 threshold. The domain
genuinely has no steady state, for a reason visible in the numbers at the top of _fields():

    precipitation 0.22    evaporation 0.10    open_water_evaporation 0.30

Open-water evaporation is ABOVE precipitation -- deliberately, to cap lakes -- so the water balance
changes sign at the ground surface: +0.060 m/yr below it, -0.080 m/yr at it.

THAT IS NOT THE DRIVER, and it was written here as though it were until an ablation said otherwise.
HALVING open_water_evaporation to 0.15 (below P, so a saturated cell stays in surplus) leaves the
oscillation UNCHANGED: cell (120,63) span 24.4915 m -> 24.5777 m. The ET contrast is a real feature of
this forcing and it is not what makes the cell leave the surface.

WHAT IS ESTABLISHED, by ablation rather than argument:
  - REMOVING SURFACE WATER ENTIRELY (routing: off, collection: off) STOPS the driver cell
    oscillating: (120,63) rises to the surface and STAYS, reaching +0.02 m. So surface-water removal
    is NECESSARY for that cell's cycle.
  - WHICH removal mechanism does not matter: active_set, explicit and implicit give spans of 24.4966,
    24.4085 and 24.4877 m, the same 8 of 41 reports at the surface, the same phase.
  - Cell (98,75) is NOT the same story: its span is 9.8932 m with routing on and 9.8741 m with routing
    off. Something drives that one which does not need surface water at all.
  - The FILL limb is recharge-limited and predicted to 7%: 24.5 m of table x phi 0.25 = 6.12 m of
    water at +0.060 m/yr = 102 yr, against 110 yr measured. That prediction survives -- it depends on
    the BELOW-surface balance, which the E_ow ablation did not change.

SO THE MECHANISM IS NOT SETTLED. At least two distinct behaviours are present, one needing surface
removal and one not, and no single explanation has survived a test yet. Two have already failed here
(the ET contrast; and earlier, "active_set fails to cure flicker" -- see #111). Treat anything more
specific than the four bullets above as unproven.

It shows up in 32 of 14064 land cells -- 2-3 that actually reach the surface, plus neighbours. They
are the HIGH, STEEP cells (median topo 1172 m against 428 domain-wide, median slope 0.264 against
0.134) because slope sets fdepth, and fdepth ~5 m makes lateral drainage slow enough for local
recharge to stack up 24 m of water table.

NOT a numerical artifact, and five candidates were eliminated by measurement before concluding that:
the metric reading pre-FSM state (real defect, fixed separately, but here pre- and post-FSM agree to
1e-12); FSM outlet switching (lake volume bit-identical every cycle); time discretisation (4x dt
refinement moves the amplitude 0.5%); operator splitting (FSM runs once per STEP, so that same sweep
refined the coupling too); and the surface removal law (active_set, explicit and implicit all give
the same cycle to 0.4%). See task #111 for the full record.

If you want corsica to settle, lower open_water_evaporation below precipitation. That is a change to
the PHYSICS you are asking for, not a fix.

Two topographies:
  * `spectral` -- a synthetic island (radial dome + Fourier roughness + two carved basins).
    Deterministic, self-contained.
  * `corsica`  -- a real DEM: a 240x156 window of GEBCO_08 over Corsica (bundled as
    `corsica_gebco.tif`, so no GEBCO download is needed). Steep real terrain, and for a long time
    the case where the default Anderson solver FAILED to converge while the implicit Picard solver
    did. THAT IS NO LONGER TRUE, measured 2026-09-17: under the current default set (tr-bdf2 +
    active_set + continuous routing) Anderson completes corsica and is cross-rank consistent to
    4.8e-12 m. The old claim is kept here, corrected rather than deleted, because it is why the
    Picard path was the demo's default in the first place.

Usage:
    demo.py spectral [--ranks 4 8] [--map]
    demo.py corsica  [--ranks 4 8] [--map]

Requires ../../build/wtm.x, rasterio, numpy (matplotlib only for --map).
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
WTM = os.path.abspath(os.path.join(HERE, "..", "..", "build", "wtm.x"))

# The ONE shared geotransform writer (tests/wtm_testgrid.py). WTM derives its grid geometry -- degree
# spacing, southern-edge latitude, and the cos-latitude cell-size scaling -- from each input raster's
# GDAL geotransform (#124). This demo used to stamp `from_bounds(0, 0, W, H, W, H)`, i.e. ONE DEGREE
# PER CELL starting at the equator, from the era when WTM ignored georeferencing entirely. That is
# 111 km cells against the 11 km (spectral) and 0.9 km (corsica) the demo intends, so the run was
# solving a different problem than the one it describes. Routed through the shared helper rather than
# hand-rolled here, for the reason that file gives: a hand-rolled transform drifts from its grid.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tests")))
from wtm_testgrid import write_tif  # noqa: E402


def terrain_slope(topo, mask, cpd, south):
    """|grad z|, dimensionless rise/run, by central differences on the DEM.

    WHY THIS EXISTS. WTM's e-folding depth is a function of slope (src/irf.cpp setup_fdepth,
    the Fan/Ying form):

        fdepth = max(fdepth_a / (1 + fdepth_b * slope), fdepth_fmin)

    and transmissivity is T = fdepth * ksat * exp((wtd + 1.5) / fdepth). This demo used to supply
    slope = 0 EVERYWHERE, so with a = 200 and b = 150 every cell got the same fdepth = 200 m, and
    with a uniform ksat the domain had NO spatial transmissivity contrast whatsoever. That is the
    numerically easiest case there is, and it is not what a real DEM gives you: the same parameters
    over real Corsica span fdepth from the 2 m floor on the steep faces to 200 m in the flats, and T
    with it. Spatially varying exp-T is the thing the solver notes name as Picard's real limiter, so
    a demo with slope = 0 was not exercising the hard part of this model.

    Cell spacing is computed in METRES from the geotransform the fixture is written with, and the
    east-west spacing carries the cos(latitude) factor -- at Corsica's 41.2 deg N a 1/120 deg cell is
    ~927 m north-south but only ~697 m east-west, and ignoring that would tilt every slope.

    Ocean cells are excluded from the gradient stencil by filling them with their land neighbour's
    elevation: a land cell at the coast should not read a slope from the 0 m ocean beside it, which
    would manufacture the steepest gradients in the domain exactly where the boundary already is.
    """
    deg = 1.0 / float(cpd)
    H, W = topo.shape
    lat = south + (np.arange(H - 1, -1, -1) + 0.5) * deg        # row 0 = north
    m_per_deg = 111320.0
    dy = deg * m_per_deg                                         # metres, constant
    dx = deg * m_per_deg * np.cos(np.radians(lat))               # metres, per row
    land = mask > 0
    z = np.where(land, topo, np.nan)
    # fill ocean with the nearest land value along each axis so the coast reads a land-side gradient
    filled = np.array(z)
    for _ in range(2):
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            nb = np.roll(filled, sh, axis=ax)
            filled = np.where(np.isnan(filled), nb, filled)
    filled = np.where(np.isnan(filled), 0.0, filled)
    gy, gx = np.gradient(filled, dy, 1.0)                        # dz/dy in m/m; dz/dx still per CELL
    gx = gx / dx[:, None]                                        # now m/m, with the cos(lat) spacing
    slope = np.sqrt(gx ** 2 + gy ** 2)
    return np.where(land, slope, 0.0).astype("float32")


def _fields(H, W, topo, mask, slope):
    """Uniform forcing that yields a surplus (lakes + rivers) without saturating the whole island.

    `slope` is the ONE field derived from the terrain rather than set to a constant. ksat and
    porosity stay uniform: WTM takes those from soil data, and there is no soil raster bundled with
    this demo, so anything spatial here would be invented rather than measured.
    """
    return {
        "topography": (topo, "float32"),
        "slope": (slope, "float32"),
        "mask": (mask, "float32"),
        "precipitation": (np.full((H, W), 0.22), "float32"),        # m/yr, > evap+... : net surplus
        "evaporation": (np.full((H, W), 0.10), "float32"),
        "open_water_evaporation": (np.full((H, W), 0.30), "float32"),  # caps lakes
        "winter_temperature": (np.zeros((H, W)), "float32"),
        "horizontal_ksat": (np.full((H, W), 1e-4), "float32"),
        "porosity": (np.full((H, W), 0.25), "float32"),
        "runoff_ratio": (np.full((H, W), 0.5), "float32"),          # overland flow -> rivers
        "starting_wt": (np.full((H, W), -20.0), "float64"),
    }


def _write(d, region, H, W, topo, mask, cpd, south):
    """Write the input rasters with the geotransform the demo actually intends (cpd, south)."""
    slope = terrain_slope(topo, mask, cpd, south)
    land = mask > 0
    fd = np.maximum(200.0 / (1 + 150 * slope[land]), 2.0)   # setup_fdepth at this demo's a/b/fmin
    print(f"  slope (land): median {np.median(slope[land]):.4f}  max {slope[land].max():.4f}"
          f"  ->  fdepth min {fd.min():.2f}  median {np.median(fd):.2f}  max {fd.max():.2f} m")
    for name, (arr, dt) in _fields(H, W, topo, mask, slope).items():
        # ksat/porosity are time-independent (no _t0); the rest carry the time tag.
        fn = f"{region}_{name}.tif" if name in ("horizontal_ksat", "porosity") else f"{region}_t0_{name}.tif"
        write_tif(os.path.join(d, fn), np.asarray(arr), cpd, south, dtype=dt)


def make_spectral(d, N=96):
    y, x = np.mgrid[0:N, 0:N]
    cx = cy = (N - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    R = 0.40 * N
    dome = 280.0 * (1 - (r / R) ** 2)                                   # crosses sea level -> coast
    rough = np.zeros((N, N))
    for kx, ky, amp, px, py in [(3, 2, 28, 0.3, 0.7), (5, 4, 16, 1.1, 0.2), (7, 3, 11, 2.0, 1.5),
                                (2, 6, 18, 0.5, 2.3), (9, 7, 6, 1.7, 0.9)]:
        rough += amp * np.sin(2 * np.pi * kx * x / N + px) * np.sin(2 * np.pi * ky * y / N + py)
    topo = dome + rough
    for bx, by, br, dep in [(0.36, 0.44, 0.09, 70), (0.60, 0.60, 0.08, 55)]:  # carved basins -> lakes
        dd = np.sqrt((x - bx * N) ** 2 + (y - by * N) ** 2)
        topo -= dep * np.exp(-(dd / (br * N)) ** 2)
    mask = (topo > 0.0).astype("float32")
    mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0                    # ocean edge ring
    topo = np.where(mask > 0, np.maximum(topo, 0.5), 0.0).astype("float32")
    _write(d, "spectral", N, N, topo, mask, 10, -30.0)
    return "spectral", 10, -30.0                                        # region, cells/deg, southern_edge


def make_corsica(d):
    with rasterio.open(os.path.join(HERE, "corsica_gebco.tif")) as s:
        dem = s.read(1).astype("float32")
    H, W = dem.shape
    mask = (dem > 0).astype("float32")
    mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0                    # force ocean boundary
    topo = np.where(mask > 0, np.maximum(dem, 0.5), 0.0).astype("float32")
    _write(d, "corsica", H, W, topo, mask, 120, 41.2)
    return "corsica", 120, 41.2                                         # GEBCO 30" ; Corsica latitude


# THE TWO SOLVER SETS THIS DEMO CAN RUN. Both are stated in full rather than one being expressed as
# a diff of the other: which keys a solver resolves to is exactly what a reader comes here to learn,
# and a diff hides it. They are NOT free to vary independently -- each set is internally forced:
#
#   picard   explicit collector (active_set is refused on the Picard path), so routing must be
#            `impulse` (continuous x explicit is refused), and time_integration bdf2 = BDF2-on-V.
#            This is what the demo has always run, via the retired flag -wtm_bdf2_on_V.
#   anderson THE SHIPPED DEFAULT SET: active_set collector, routing continuous, tr-bdf2. What a new
#            user actually gets by omitting all three keys. Stated explicitly anyway, per the
#            declared-config rule.
#
# time_step.mode is `fixed` for BOTH, and that is a DEMO choice rather than either solver's default
# (both resolve to `adaptive`). Serial and parallel must take the SAME steps for the cross-rank
# comparison to be measuring cross-rank behaviour and not two different step sequences.
SOLVERS = {
    "picard": dict(method="picard", time_integration="bdf2",
                   collection="explicit", routing="impulse", mode="fixed"),
    "anderson": dict(method="anderson", time_integration="tr-bdf2",
                     collection="active_set", routing="continuous", mode="fixed"),
    # Newton means the WORKING recipe: the analytic Jacobian WITH pseudo-transient continuation.
    # Its natural mode is `ramp`, not `fixed` -- #50 measured plain Newton failing at 5 of 6 step
    # sizes from a cold start, and the single success was a single-dt artefact. `ramp` grows dt on
    # solve-EASE rather than on an error estimate, starting from newton.dt0 (default time.deltat/200),
    # which keeps the storage term S/dt diagonally dominant while the guess is still far away.
    "newton": dict(method="newton", time_integration="backward-euler",
                   collection="active_set", routing="continuous", mode="ramp"),
}

# WHO MAY SIZE THE STEP. `ramp` is the Newton path's continuation and is REFUSED elsewhere; the other
# two run anywhere. Stated here so an impossible pairing is rejected by this script with an
# explanation rather than by the model a minute into a run.
MODES = {
    "fixed":    "dt exactly as configured; serial and parallel take identical steps",
    "adaptive": "error-controlled, clamped to the report span -- the SHIPPED default",
    "ramp":     "pseudo-transient continuation, grows on solve-ease. Newton only",
}


CYCLES = 15          # reports per run. Held fixed across step sizes, so time.total follows dt
YEAR_S = 31536000    # seconds; the demo's historical step
WEEK_S = 7 * 86400


def time_step_block(mode, dt):
    """The solver.time_step keys THIS mode actually reads.

    Deliberately minimal per mode. The model refuses a -wtm_ flag nothing consumed, and the same
    principle applies to config keys: a dial written down beside a controller that never reads it
    looks like a setting and is not one.
    """
    if mode == "fixed":
        return f"    mode: fixed\n    dt: {dt}      # exactly as given; nothing may change it\n"
    if mode == "adaptive":
        return (f"    mode: adaptive\n"
                f"    dt: {dt}      # the STARTING step; the controller moves it from here,\n"
                f"                      # and may SUBDIVIDE it -- clamped to the report span\n"
                "    error_tol: 0.01   # metres of WATER per step. Stated rather than inherited:\n"
                "                      # it would otherwise track equilibrium_stop.tol, which this\n"
                "                      # demo sets to 0 to keep the two rank counts in lockstep\n"
                "    grow: 1.5\n    shrink: 0.25\n    grow_if_niter_leq: 8\n"
                "    max_retries: 15\n    norm: rms\n")
    if mode == "ramp":
        return (f"    mode: ramp\n"
                f"    dt: {dt}      # the TARGET step the continuation climbs towards\n")
    raise ValueError(mode)


# RUNNING TO EQUILIBRIUM RATHER THAN TO A CLOCK. The two are different experiments and this demo now
# does both, because each answers a question the other cannot:
#
#   CYCLES (default)  every run takes the SAME 15 steps of the SAME size and stops. One variable, so
#                     serial-vs-parallel is a clean comparison -- this is what the cross-rank claim
#                     rests on, and why equilibrium_stop.tol is 0 there.
#   EQUILIBRIUM       every run stops when the water table stops moving, however long that takes.
#                     Now the solvers are compared at the SAME PHYSICAL STATE instead of at the same
#                     clock reading, which is the only way a cross-SOLVER comparison of lake counts
#                     means anything (at 15 years none of them had converged -- corsica/newton was
#                     still moving 0.856 m of water per cycle at cycle 14).
#                     The cross-RANK comparison is NOT valid here and the script does not make it:
#                     two rank counts may satisfy the stop test at different cycles, which is a
#                     different amount of simulated time, not a disagreement.
#
# The stop tolerance is the SHIPPED DEFAULT from config.yaml (0.001 m of water, metric frac, 0.001 of
# cells) rather than a number invented for this demo. TOTAL is a CAP, not a target: if a run reaches
# it, it did not converge, and that is reported rather than passed off as an equilibrium.
EQ_TOL   = 0.001     # config.yaml's shipped default: metres of WATER moved in a whole cycle
EQ_FRAC  = 0.001     # ...at this fraction of cells
EQ_YEARS = 20000     # the CAP. Reaching it means the run did not settle
EQ_REPORT = 10       # steps per equilibrium check


def cfg(d, region, txt, pfx, outdir, solver, mode, dt, equilibrium):
    """The run's configuration, in the nested-YAML schema (see config.yaml at the repo root).

    EVERY setting this run resolves to is stated here. That is the house rule for configs in this
    repo, and it is what makes a run reproducible from the file alone rather than from knowing which
    defaults applied on the day. Four of them are LOAD-BEARING for what this demo claims:

      solver.method + time_integration + collection.method + routing
          One of the two sets in SOLVERS above, chosen by --solver. They are internally forced: each
          solver supports a different collector, and the collector then decides what routing is
          representable.
      time_step.mode: fixed
          The legacy behaviour: dt is exactly `dt` below. Picard would otherwise resolve to
          `adaptive`. Held fixed so the serial and parallel runs take the SAME steps, and any
          difference between them is the thing being measured rather than a different step sequence.
      equilibrium_stop.tol: 0
          Run the clock out; never stop early. A determinism comparison that is free to auto-stop can
          have its two runs halt at different cycles and then differ for a reason that is not a
          defect -- exactly the trap #95 caught in the taper study.
      surface_water.routing
          `impulse` on Picard (FSM's routed table replaces the step baseline -- what the model has
          always done, and what v2.0.1 does; `continuous` is REFUSED with the `explicit` collector
          Picard resolves to). `continuous` on Anderson, the shipped default.
    """
    sv = SOLVERS[solver]
    if equilibrium:
        report_interval = EQ_REPORT
        reports = max(1, int(round(EQ_YEARS * YEAR_S / (dt * report_interval))))
        eq_block = (f"    tol: {EQ_TOL}        # config.yaml's shipped default, in metres of WATER\n"
                    f"    metric: frac\n"
                    f"    frac: {EQ_FRAC}       # stop once fewer than this fraction of cells still move\n")
    else:
        report_interval = 1
        reports = CYCLES
        eq_block = ("    tol: 0            # LOAD-BEARING: never auto-stop -- see the docstring\n"
                    "    metric: frac      # INERT with tol 0\n"
                    "    frac: 0.001       # INERT, as above\n")
    total_s = reports * report_interval * dt
    if mode == "ramp" and solver != "newton":
        sys.exit(f"--mode ramp is the Newton continuation and is refused on the {solver} path.")
    return f"""run:
  type: equilibrium
  initial_water_table: saturated   # was `supplied_wt 0`
  equilibrium_stop:
{eq_block}
time:
  total: "{total_s}s"   # reports of exactly dt each. Stated in seconds, not years, because
                    # total must be an integer number of report spans and a non-annual dt
                    # does not divide a year
  report_interval: {report_interval}
  save_every_n_reports: 1
io:
  source: '{d}'
  region: '{region}'
  time_start: 't0'
  time_end: 't0'
output:
  directory: '{outdir}'
  outfile_prefix: '{pfx}'
  run_log: '{txt}'
  verbosity: normal
  if_exists: overwrite
boundaries:
  land: neumann_toposlope
transmissivity:
  fdepth:
    a: 200
    b: 150
    fmin: 2
  additive_background_transmissivity: 0
evaporation:
  et_sigmoid:
    wtd_center: 0.05
    logistic_width: 0.1
  extinction_depth: 8
  tapers:
    surface_transition: true
    depth_extinction: true
surface_water:
  routing: {sv['routing']}    # LOAD-BEARING -- see the docstring
  runoff_ratio: raster   # was `runoff_ratio_on 1`: use the runoff_ratio layer this demo writes
  infiltration_during_flow: false
  collection:
    method: {sv['collection']}
solver:
  method: {sv['method']}
  time_integration: {sv['time_integration']}
  convergence:
    metric: volume         # the shipped default (#61): the per-solve step judged in WATER
    water_volume_tol: 1.0e-8
  smoothing:
    ksat_surface: 0
    ksat_soilbottom: 0
    storativity_surface: 0.01
  time_step:
{time_step_block(mode, dt)}parallel:
  threads_per_rank: 1
"""


class DidNotComplete(RuntimeError):
    """The model ran and did not reach an answer. An outcome of the experiment, not a bug in it."""


def run(d, region, cpd, south, n, solver, mode, dt, equilibrium=False):
    # The stem carries the SOLVER and the MODE as well as the rank count. Without it a picard run and an anderson
    # run at the same n write the same prefix, and the glob below silently reads the other solver's
    # raster -- the failure mode that #74 exists to prevent in the test suites.
    stem = f"{solver}_{mode}_dt{dt}{'_eq' if equilibrium else ''}_n{n}"
    c = os.path.join(d, f"cfg_{stem}.yaml")
    with open(c, "w") as f:
        # Absolute paths, as the test suites do: `directory` is the run's provenance dir, while the
        # raster prefix and the log are written where this script then looks for them.
        f.write(cfg(d, region, os.path.join(d, f"out_{stem}.txt"),
                    os.path.join(d, f"{stem}_"), os.path.join(d, f"prov_{stem}"), solver, mode, dt, equilibrium))
    # A SOLVER THAT CANNOT FINISH IS A RESULT, NOT A CRASH. On the realistic corsica fixture Picard
    # does not converge, and that is the single most informative thing this demo has to say -- so it
    # is reported as an outcome with the model's own reason, not as a Python traceback. Robustness
    # here is computes-vs-does-not-compute, and the run log is where the model says which.
    log = os.path.join(d, f"stdout_{stem}.log")
    with open(log, "w") as lf:
        rc = subprocess.run(["mpirun", "-n", str(n), WTM, c], cwd=d,
                            env={**os.environ, "OMP_NUM_THREADS": "1"},
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        reason = "(no reason found in the log)"
        with open(log, errors="replace") as lf:
            for line in lf:
                if "DIVERGED" in line or line.startswith("ERROR"):
                    reason = line.strip()
        raise DidNotComplete(f"{solver}/{mode} n={n}: exit {rc} -- {reason}\n      log: {log}")
    with rasterio.open(sorted(glob.glob(os.path.join(d, f"{stem}_*.tif")))[-1]) as s:
        return s.read(1)


def render_map(d, region, w_serial, w_par, npar, out):
    """Left: the feature map (ocean / rivers / lakes). Right: |serial - parallel| -- the visual
    serial==parallel proof (machine-zero)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(os.path.join(d, f"{region}_t0_topography.tif")) as s:
        topo = s.read(1)
    with rasterio.open(os.path.join(d, f"{region}_t0_mask.tif")) as s:
        land = s.read(1) > 0
    NY, NX = topo.shape
    acc = np.ones_like(topo)                                            # D8 flow accumulation -> rivers
    for i in np.argsort(-np.where(land, topo, -1e9).ravel()):
        yy, xx = i // NX, i % NX
        if not land[yy, xx]:
            continue
        bz, best = topo[yy, xx], None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < NY and 0 <= nx < NX and (dy or dx) and topo[ny, nx] < bz:
                    bz, best = topo[ny, nx], (ny, nx)
        if best:
            acc[best] += acc[yy, xx]
    rivers = (acc > 0.008 * land.sum()) & land

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 7))
    # -- left: features --
    ax.imshow(np.where(land, topo, np.nan), cmap="terrain", vmin=0, vmax=topo.max())
    ax.imshow(np.where(~land, 0, np.nan), cmap="Blues", vmin=-1, vmax=1)     # ocean (blue)
    ax.imshow(np.where(rivers, 1, np.nan), cmap="winter", vmin=0, vmax=1)    # rivers (cyan)
    lk = np.where(land & (w_serial > 0.05), w_serial, np.nan)               # lakes (magenta)
    if np.isfinite(np.nanmax(lk)):
        ax.imshow(lk, cmap="cool", vmin=0, vmax=np.nanmax(lk))
    ax.set_title(f"{region}: equilibrium (2nd-order Picard, serial)\nterrain, ocean (blue), "
                 "rivers (cyan), lakes (magenta)")
    ax.axis("off")
    # -- right: serial vs parallel, in nanometres --
    diff = np.where(land, np.abs(w_serial - w_par) * 1e9, np.nan)           # m -> nm
    im = ax2.imshow(diff, cmap="magma")
    ax2.set_title(f"|serial - parallel(n={npar})|   (nanometres)\n"
                  f"max = {np.nanmax(diff):.2f} nm  =>  serial == parallel")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="nm")
    plt.tight_layout()
    plt.savefig(out, dpi=95)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=["spectral", "corsica"])
    ap.add_argument("--ranks", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--equilibrium", action="store_true",
                    help="run until the water table stops moving (config.yaml's shipped stop: 0.001 m "
                         "of water at 0.001 of cells) instead of for a fixed 15 cycles. Compares "
                         "solvers at the same PHYSICAL STATE. The cross-rank check is not meaningful "
                         "here -- two rank counts may satisfy the stop at different cycles -- so run "
                         "it with --ranks and nothing after it.")
    ap.add_argument("--dt-weeks", type=float, default=None,
                    help="time step in WEEKS (default: 1 year). The #60 case is --dt-weeks 48: a step "
                         "big enough that a fixed stepper may not get through realistic terrain, "
                         "while adaptive can subdivide it. time.total follows, so the number of "
                         "reporting cycles is what stays fixed across step sizes.")
    ap.add_argument("--mode", choices=sorted(MODES), default=None,
                    help="who sizes the time step. Default = the solver's own: "
                         + ", ".join(f"{k}->{v['mode']}" for k, v in sorted(SOLVERS.items())))
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="picard",
                    help="picard = the 2nd-order BDF2-on-V path this demo has always run "
                         "(cross-rank deterministic); anderson = the SHIPPED DEFAULT set "
                         "(tr-bdf2 + active_set + continuous routing), i.e. what a new user gets.")
    a = ap.parse_args()
    mode = a.mode or SOLVERS[a.solver]["mode"]
    dt = int(round(a.dt_weeks * WEEK_S)) if a.dt_weeks else YEAR_S
    if not os.access(WTM, os.X_OK):
        sys.exit(f"build wtm.x first (looked for {WTM})")
    d = os.path.join(HERE, f"_work_{a.region}")
    os.makedirs(d, exist_ok=True)
    if a.region == "spectral":
        region, cpd, south = make_spectral(d)
    else:
        region, cpd, south = make_corsica(d)
    try:
        w1 = run(d, region, cpd, south, 1, a.solver, mode, dt, a.equilibrium)
    except DidNotComplete as e:
        print(f"{region} [{a.solver}/{mode}, dt={dt/WEEK_S:.3g} wk]: DID NOT COMPLETE\n      {e}")
        print(f"      This is the result. Try --solver "
              f"{'anderson' if a.solver == 'picard' else 'picard'} on the same terrain.")
        sys.exit(2)
    with rasterio.open(os.path.join(d, f"{region}_t0_mask.tif")) as s:
        land = s.read(1) > 0
    lakes = int((land & (w1 > 0.05)).sum())
    if a.equilibrium and a.ranks:
        print("NOTE: --equilibrium compares solvers at one physical state; the cross-rank numbers\n"
              "      below are NOT a determinism check, because two rank counts may stop at\n"
              "      different cycles. Use the default (15-cycle) mode for that.")
    print(f"{region} [{a.solver}/{mode}, dt={dt/WEEK_S:.3g} wk"
          f"{', to equilibrium' if a.equilibrium else ''}]: serial (n=1) done -- {lakes} lake cells (max {w1[land].max():.1f} m), "
          f"{int((~land).sum())} ocean cells")
    fail = 0
    w_par, npar = None, None
    for n in a.ranks:
        try:
            wn = run(d, region, cpd, south, n, a.solver, mode, dt, a.equilibrium)
        except DidNotComplete as e:
            print(f"  serial vs n={n}: DID NOT COMPLETE -- {e}")
            fail += 1
            continue
        md = float(np.abs(w1 - wn)[land].max())
        ok = md < 1e-6
        fail += not ok
        w_par, npar = wn, n                                             # keep the last (largest) for the map
        print(f"  serial vs n={n}: max|dwtd| = {md:.3e} m  {'CONSISTENT' if ok else 'MISMATCH'}")
    if a.map and w_par is not None:
        render_map(d, region, w1, w_par, npar, os.path.join(HERE, f"{region}_{a.solver}_{mode}_map.png"))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
