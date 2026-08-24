#!/usr/bin/env python3
"""Surface-transition taper tests (the SURFACE_SINK_DESIGN sec 14d experiment sequence).

These exercise the smooth surface-water transition -- the sub-surface sink (-wtm_surface_sink,
taper 1) plus the demand-identity evaporation taper (-wtm_evap_taper, taper 2) -- on the matrix-free
Anderson solver (forced with -wtm_anderson; the default is now Picard). This validates the tapers on
the matrix-free path specifically (the Picard-path tapers are exercised by the golden suite). It is
the path whose hard wtd=0 switch used to make FillSpillMerge lake formation flip with the MPI rank
count near the evaporation threshold.

Study A -- the triggering case (plateau with an off-centre depression). Sweep the open-water
evaporation rate `owe` through the critical value owe = precip. The smooth taper must give:
  (1) CROSS-RANK DETERMINISM -- n=1 and n=N produce the same equilibrium at every owe, INCLUDING at
      the knife-edge owe = precip (the hard model is rank-sensitive here); and
  (2) A SMOOTH, SINGLE-VALUED RESPONSE -- the summed water table varies monotonically with owe (no
      jump / bifurcation through the threshold).

Study B -- a gently-sloping central depression. Water collects into a pond with a real shoreline;
check that the pond forms and that its extent and shoreline are cross-rank deterministic (the whole
water-table field is identical on 1 vs N ranks -- a shoreline flip would move it by metres).

Run as:  taper_test.py <wtm.x> [nrank ...]        (default ranks: 4)
Exits non-zero if any assertion fails.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wtm_testgrid import write_tif as _grid_write_tif  # noqa: E402

NX = NY = 16
REGION, TIME = "taper_test", "t0"
PRECIP = 0.1                      # m/yr; the sweep crosses owe = PRECIP
OWE_SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
DET_RTOL = 1e-9                   # cross-rank agreement: tight (a routing flip would be macroscopic)

# Intended grid: WTM derives geometry from the geotransform (#124), which the shared writer encodes.
CELLS_PER_DEGREE = 10.0
SOUTHERN_EDGE    = -45.0

# Emit the nested-YAML config (config.yaml schema) from the legacy key-value bodies below.
EMIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emit_config.sh")


def _write_tif(path, data, dtype):
    _grid_write_tif(path, np.asarray(data), CELLS_PER_DEGREE, SOUTHERN_EDGE, dtype=dtype)


def _write_cfg(path, legacy_text):
    with open(path, "w") as f:
        subprocess.run([EMIT], input=legacy_text, text=True, stdout=f, check=True)


def write_fixture(d, owe, topo):
    """Write the full input set for a plateau/depression fixture with a given owe and topography."""
    os.makedirs(d, exist_ok=True)
    mask = np.ones((NY, NX), dtype=np.float32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0          # ocean edge ring
    fields = {
        f"{REGION}_{TIME}_topography.tif":            (topo, "float32"),
        f"{REGION}_{TIME}_slope.tif":                 (np.zeros((NY, NX)), "float32"),
        f"{REGION}_{TIME}_mask.tif":                  (mask, "float32"),
        f"{REGION}_{TIME}_precipitation.tif":         (np.full((NY, NX), PRECIP), "float32"),
        f"{REGION}_{TIME}_evaporation.tif":           (np.zeros((NY, NX)), "float32"),  # ET=0 (see design 14)
        f"{REGION}_{TIME}_open_water_evaporation.tif": (np.full((NY, NX), owe), "float32"),
        f"{REGION}_{TIME}_winter_temperature.tif":    (np.zeros((NY, NX)), "float32"),
        f"{REGION}_horizontal_ksat.tif":              (np.full((NY, NX), 1e-4), "float32"),
        f"{REGION}_porosity.tif":                     (np.full((NY, NX), 0.25), "float32"),
        f"{REGION}_{TIME}_starting_wt.tif":           (np.full((NY, NX), 5.0), "float64"),
    }
    for fname, (arr, dt) in fields.items():
        _write_tif(os.path.join(d, fname), np.asarray(arr), dt)


def _cfg(d, txt, prefix):
    return f"""run_type equilibrium
fsm_on 1
evap_mode 1
infiltration_on 0
runoff_ratio_on 0
cells_per_degree 10
southern_edge -45
deltat 31536000
total_time 10yr
report_interval 2
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start t0
time_end t0
surfdatadir {d}
region {REGION}
supplied_wt 1
save_nreport_interval 9999
runoff_collector legacy
textfilename {txt}
outfile_prefix {prefix}
"""


# Matrix-free Anderson path (forced; default is now Picard) with both tapers on. The width 1.0
# (= qmax*dt, the marginal-stability point) is a deliberate Anderson-path stress; Picard would need
# the dt-scaled default. Anderson is the opt-in path this test also keeps covered.
# -wtm_eq_tol 0: run the full fixed cycle count (cross-rank determinism check; do not auto-stop).
TAPER_FLAGS = ["-wtm_anderson", "-wtm_surface_sink", "-wtm_surface_sink_qmax", "1.0",
               "-wtm_surface_sink_width", "1.0", "-wtm_evap_taper", "-snes_stol", "1e-8", "-wtm_eq_tol", "0"]


def _run(wtm, d, tag, n):
    """Run wtm on n ranks (leaves outputs in d); return the final-cycle summed water table (col 11)."""
    txt = os.path.join(d, f"{tag}_n{n}.txt")
    cfg = os.path.join(d, f"cfg_{tag}_n{n}.yaml")
    _write_cfg(cfg, _cfg(d, txt, os.path.join(d, f"{tag}_n{n}_")))
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(["mpirun", "-n", str(n), wtm, cfg] + TAPER_FLAGS,
                   cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rows = [ln.split() for ln in open(txt) if ln[:1].isdigit()]
    if not rows:
        raise RuntimeError(f"{tag} n={n}: no output rows (solver failed?)")
    return float(rows[-1][10])  # sum_of_water_tables


def _final_wtd(d, tag, n):
    """Read the last saved water-table raster for a run (the full spatial field)."""
    tifs = sorted(f for f in os.listdir(d) if f.startswith(f"{tag}_n{n}_") and f.endswith(".tif"))
    with rasterio.open(os.path.join(d, tifs[-1])) as s:
        return s.read(1)


def study_a(wtm, ranks):
    # Off-centre depression in a plateau (the fsm_test triggering fixture): FSM has a routing
    # threshold to flip on, so the HARD wtd=0 switch is rank-sensitive here (this is the fsm_evap1
    # rank-dependence that forces a cm tolerance in the golden suite). The tapers must remove it.
    # Verified to bite: without the taper flags the hard switch diverges cross-rank (~1e-5 at
    # owe=0.20), which this rtol=1e-9 determinism check would fail.
    print("Study A -- plateau with off-centre depression, owe-sweep through owe = precip = %.3g" % PRECIP)
    topo = np.full((NY, NX), 100.0, dtype=np.float32)
    topo[9:13, 9:13] = 90.0
    wt_n1 = []
    fails = 0
    for owe in OWE_SWEEP:
        with tempfile.TemporaryDirectory(prefix=f"taperA_owe{owe:.3f}_") as d:
            write_fixture(d, owe, topo)
            base = _run(wtm, d, "A", 1)
            wt_n1.append(base)
            for n in ranks:
                v = _run(wtm, d, "A", n)
                rel = abs(v - base) / (abs(base) + 1e-30)
                ok = rel < DET_RTOL
                fails += not ok
                print(f"  owe={owe:5.3f}  wt_sum(n1)={base: .6e}  n={n}: |Δ|/|n1|={rel:.2e}  "
                      f"{'OK' if ok else 'FAIL (cross-rank flip)'}")
    # Smoothness: the summed water table must be monotone in owe (single-valued, no jump/bifurcation).
    diffs = np.diff(wt_n1)
    monotone = np.all(diffs <= 0) or np.all(diffs >= 0)
    print(f"  smoothness: wt_sum vs owe monotone = {monotone}  (deltas: "
          f"{', '.join('%.2e' % x for x in diffs)})")
    fails += not monotone
    return fails


def study_b(wtm, ranks):
    # A gently-sloping central depression (a smooth bowl, 100 m rim -> 92 m centre) with a supply
    # surplus (owe < precip), so water collects into a pond with a real SHORELINE (the wet/dry margin
    # where the table crosses the surface). Checks that the pond forms and that its extent + shoreline
    # are CROSS-RANK DETERMINISTIC (the full water-table field is identical whether run on 1 or N ranks
    # -- a routing/shoreline flip would move it by metres, not the ~1e-10 m of FP-reduction noise).
    print("Study B -- gently-sloping central depression (pond + shoreline)")
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = ((xx - (NX - 1) / 2.0) ** 2 + (yy - (NY - 1) / 2.0) ** 2) / ((NX / 2.0) ** 2)
    topo = (100.0 - 8.0 * np.exp(-r2)).astype(np.float32)
    fails = 0
    with tempfile.TemporaryDirectory(prefix="taperB_") as d:
        write_fixture(d, 0.05, topo)                      # owe = 0.05 < precip = 0.1 -> pond fills
        _run(wtm, d, "B", 1)
        w1 = _final_wtd(d, "B", 1)
        pond_cells = int((w1 > 0).sum())
        pond_ok = pond_cells > 0
        print(f"  pond forms: {pond_cells} cells with standing water, max depth {w1.max():.2f} m  "
              f"{'OK' if pond_ok else 'FAIL (no pond formed)'}")
        fails += not pond_ok
        for n in ranks:
            _run(wtm, d, "B", n)
            wn = _final_wtd(d, "B", n)
            md = float(np.abs(w1 - wn).max())
            ok = md < 1e-6
            print(f"  n={n}: max|wtd(n1) - wtd(n{n})| = {md:.2e} m  "
                  f"{'OK' if ok else 'FAIL (shoreline/pond flips with rank count)'}")
            fails += not ok
    return fails


def _arid_fixture(d, ksat=1e-9):
    """Flat plateau where ET (0.5 m/yr) > precip (0.2 m/yr): the ARID regime in which taper 2 alone has
    no equilibrium. Tiny ksat -> negligible lateral flow -> each interior cell is ~an isolated vertical
    ET/recharge balance, so the taper-3 clamp shows cleanly at ~ -d_ext (lateral drainage to the ocean
    ring would otherwise mask it)."""
    os.makedirs(d, exist_ok=True)
    mask = np.ones((NY, NX), dtype=np.float32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields = {
        f"{REGION}_{TIME}_topography.tif":             (np.full((NY, NX), 100.0), "float32"),
        f"{REGION}_{TIME}_slope.tif":                  (np.zeros((NY, NX)), "float32"),
        f"{REGION}_{TIME}_mask.tif":                   (mask, "float32"),
        f"{REGION}_{TIME}_precipitation.tif":          (np.full((NY, NX), 0.2), "float32"),   # P
        f"{REGION}_{TIME}_evaporation.tif":            (np.full((NY, NX), 0.5), "float32"),   # ET > P
        f"{REGION}_{TIME}_open_water_evaporation.tif": (np.full((NY, NX), 1.0), "float32"),
        f"{REGION}_{TIME}_winter_temperature.tif":     (np.zeros((NY, NX)), "float32"),
        f"{REGION}_horizontal_ksat.tif":               (np.full((NY, NX), ksat), "float32"),
        f"{REGION}_porosity.tif":                      (np.full((NY, NX), 0.25), "float32"),
        f"{REGION}_{TIME}_starting_wt.tif":            (np.full((NY, NX), -1.0), "float64"),
    }
    for fname, (arr, dt) in fields.items():
        _write_tif(os.path.join(d, fname), np.asarray(arr), dt)


def _arid_cfg(d, txt, prefix):
    # fsm_on 0: a pure groundwater drawdown test (no lakes). 180 yr to equilibrium (60 reports x 3 yr).
    return (f"run_type equilibrium\nfsm_on 0\nevap_mode 1\ninfiltration_on 0\nrunoff_ratio_on 0\n"
            f"cells_per_degree 10\nsouthern_edge -45\ndeltat 31536000\ntotal_time 180yr\nreport_interval 3\n"
            f"fdepth_a 200\nfdepth_b 150\nfdepth_fmin 2\ntime_start t0\ntime_end t0\n"
            f"surfdatadir {d}\nregion {REGION}\nsupplied_wt 1\nsave_nreport_interval 9999\n"
            f"runoff_collector legacy\n"
            f"textfilename {txt}\noutfile_prefix {prefix}\n")


def _arid_run(wtm, d, tag, flags):
    cfg = os.path.join(d, f"cfg_{tag}.yaml")
    _write_cfg(cfg, _arid_cfg(d, os.path.join(d, f"{tag}.txt"), os.path.join(d, f"{tag}_")))
    subprocess.run([wtm, cfg] + flags, cwd=d, env={**os.environ, "OMP_NUM_THREADS": "1"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    tifs = sorted(f for f in os.listdir(d) if f.startswith(f"{tag}_") and f.endswith(".tif"))
    with rasterio.open(os.path.join(d, tifs[-1])) as s:
        return s.read(1)


def study_c(wtm):
    """Taper 3 (accessibility / extinction-depth, -wtm_extinction) clamps the arid drawdown. This is the
    regression that BITES: without taper 3, taper 2 alone has NO equilibrium in an arid cell (E_eff >
    precip) and the table runs away; with it, drawdown halts at ~ -d_ext. The taper-2-alone run below IS
    the pre-taper-3 behavior -- the test asserts it runs away while the extinction runs clamp, so it
    fails if taper 3 stops clamping. Also checks the clamp depth scales with d_ext."""
    print("Study C -- arid extinction-depth clamp (ET=0.5 > precip=0.2; taper 3 = -wtm_extinction)")
    E = ["-wtm_anderson", "-wtm_evap_taper", "-snes_stol", "1e-8", "-wtm_eq_tol", "0"]  # full 60-cycle clamp run
    c = (NY // 2, NX // 2)  # interior cell, farthest from the ocean ring
    fails = 0
    with tempfile.TemporaryDirectory(prefix="taperC_") as d:
        _arid_fixture(d)
        # taper 3 is default-on, so "taper 2 alone" must explicitly disable it (-wtm_extinction 0).
        w2 = float(_arid_run(wtm, d, "C2", E + ["-wtm_extinction", "0"])[c])
        w8 = float(_arid_run(wtm, d, "C8", E + ["-wtm_extinction", "-wtm_extinction_depth", "8"])[c])
        w4 = float(_arid_run(wtm, d, "C4", E + ["-wtm_extinction", "-wtm_extinction_depth", "4"])[c])
        runaway = w2 < -50.0                                          # no equilibrium without taper 3
        ok8 = -8.0 <= w8 <= -6.5                                      # clamped just inside d_ext = 8
        ok4 = -4.0 <= w4 <= -3.0                                      # clamped just inside d_ext = 4
        scales = w4 > w8                                             # shallower d_ext -> shallower clamp
        print(f"  taper 2 alone      : interior wtd = {w2:8.2f} m  "
              f"{'OK (runs away, no equilibrium)' if runaway else 'FAIL (expected runaway)'}")
        print(f"  taper 2 + ext(8 m) : interior wtd = {w8:8.2f} m  "
              f"{'OK (clamped ~ -d_ext)' if ok8 else 'FAIL (not clamped -- taper 3 broken?)'}")
        print(f"  taper 2 + ext(4 m) : interior wtd = {w4:8.2f} m  {'OK (clamped ~ -d_ext)' if ok4 else 'FAIL'}")
        print(f"  clamp scales with d_ext (ext4 shallower than ext8) = {scales}")
        fails += (not runaway) + (not ok8) + (not ok4) + (not scales)
    return fails


def main():
    if len(sys.argv) < 2:
        print("usage: taper_test.py <wtm.x> [nrank ...]", file=sys.stderr)
        return 2
    wtm = os.path.abspath(sys.argv[1])
    ranks = [int(a) for a in sys.argv[2:]] or [4]
    if not os.access(wtm, os.X_OK):
        print(f"ERROR: WTM binary not executable: {wtm}", file=sys.stderr)
        return 1
    fails = study_a(wtm, ranks)
    print()
    fails += study_b(wtm, ranks)
    print()
    fails += study_c(wtm)
    print()
    if fails:
        print(f"TAPER TESTS FAILED ({fails} assertion(s))", file=sys.stderr)
        return 1
    print("TAPER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
