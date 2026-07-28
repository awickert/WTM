#!/usr/bin/env python3
"""Surface-transition taper tests (the SURFACE_SINK_DESIGN sec 14d experiment sequence).

These exercise the smooth surface-water transition -- the sub-surface sink (-wtm_surface_sink,
taper 1) plus the demand-identity evaporation taper (-wtm_evap_taper, taper 2) -- on the Anderson
DEFAULT solver, i.e. the production path whose hard wtd=0 switch used to make FillSpillMerge lake
formation flip with the MPI rank count near the evaporation threshold.

Study A -- the flat plateau (the triggering case). Sweep the open-water evaporation rate `owe`
through the critical value owe = precip. The smooth taper must give:
  (1) CROSS-RANK DETERMINISM -- n=1 and n=N produce the same equilibrium at every owe, INCLUDING at
      the knife-edge owe = precip (the hard model rank-flips here); and
  (2) A SMOOTH, SINGLE-VALUED RESPONSE -- the summed water table varies monotonically with owe (no
      jump / bifurcation through the threshold).

Run as:  taper_test.py <wtm.x> [nrank ...]        (default ranks: 4)
Exits non-zero if any assertion fails.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_bounds

NX = NY = 16
REGION, TIME = "taper_test", "t0"
PRECIP = 0.1                      # m/yr; the sweep crosses owe = PRECIP
OWE_SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
DET_RTOL = 1e-9                   # cross-rank agreement: tight (a routing flip would be macroscopic)

_TRANSFORM = from_bounds(0, 0, NX, NY, NX, NY)


def _write_tif(path, data, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dtype, crs="EPSG:4326", transform=_TRANSFORM) as dst:
        dst.write(data.astype(dtype), 1)


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
total_cycles 5
maxiter 2
fdepth_a 200
fdepth_b 150
fdepth_fmin 2
time_start t0
time_end t0
surfdatadir {d}
region {REGION}
supplied_wt 1
cycles_to_save 9999
textfilename {txt}
outfile_prefix {prefix}
"""


# Anderson DEFAULT path (no -wtm_bdf2_on_V) with both tapers on.
TAPER_FLAGS = ["-wtm_surface_sink", "-wtm_surface_sink_qmax", "1.0",
               "-wtm_surface_sink_width", "1.0", "-wtm_evap_taper", "-snes_stol", "1e-8"]


def _run(wtm, d, tag, n):
    """Run wtm on n ranks; return the final-cycle summed water table (col 11 of the text file)."""
    txt = os.path.join(d, f"{tag}_n{n}.txt")
    cfg = os.path.join(d, f"cfg_{tag}_n{n}")
    with open(cfg, "w") as f:
        f.write(_cfg(d, txt, os.path.join(d, f"{tag}_n{n}_")))
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(["mpirun", "-n", str(n), wtm, cfg] + TAPER_FLAGS,
                   cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rows = [ln.split() for ln in open(txt) if ln[:1].isdigit()]
    if not rows:
        raise RuntimeError(f"{tag} n={n}: no output rows (solver failed?)")
    return float(rows[-1][10])  # sum_of_water_tables


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
    if fails:
        print(f"TAPER TESTS FAILED ({fails} assertion(s))", file=sys.stderr)
        return 1
    print("TAPER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
