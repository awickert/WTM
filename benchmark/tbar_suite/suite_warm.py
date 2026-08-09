#!/usr/bin/env python3
"""Suite 2: warm-start (equilibrium already reached) response to a recharge perturbation, sweeping the
time step in weeks up to failure/stalling. For each solver x Tbar we find the largest dt that still
converges from the warm state under the perturbation -- the step ceiling -- and the iteration cost.

Setup: builds a 'warm domain' = the source island inputs with (1) precipitation scaled by `precip_scale`
(the perturbation) and (2) a supplied starting water table = a converged equilibrium raster. Then runs
supplied_wt=1 equilibrium for a few cycles at each dt; OK => dt within ceiling, FAIL/TIMEOUT => beyond.

Usage: suite_warm.py <src_domain> <eq_raster.tif> <out_tag> [precip_scale]"""
import os, sys, glob, json, shutil, re
import numpy as np, rasterio
import suite  # run_one, WEEK, SOLVERS

def setup_warm_domain(src, warm, eq_raster, precip_scale):
    os.makedirs(warm, exist_ok=True)
    region_time = None
    for f in glob.glob(os.path.join(src, "*.tif")):
        b = os.path.basename(f)
        m = re.match(r"(.+_\d+)_precipitation\.tif", b)
        if m:
            region_time = m.group(1)
        if "_precipitation.tif" in b:
            continue  # scaled below
        if re.search(r"_\d{9}\.tif$", b):
            continue  # an OUTPUT raster (prefix_NNNNNNNNN.tif), not an input layer
        # everything else is an input layer: time-stamped (Esquibel_010000_*) or not
        # (Esquibel_horizontal_ksat, Esquibel_porosity). Symlink it.
        dst = os.path.join(warm, b)
        if not os.path.exists(dst):
            os.symlink(f, dst)
    assert region_time, "could not find precipitation raster to infer region_time"
    # scaled precipitation (the perturbation)
    with rasterio.open(os.path.join(src, region_time + "_precipitation.tif")) as r:
        prof = r.profile; p = r.read(1)
    with rasterio.open(os.path.join(warm, region_time + "_precipitation.tif"), "w", **prof) as w:
        w.write((p * precip_scale).astype(prof["dtype"]), 1)
    # supplied starting water table = the equilibrium
    with rasterio.open(eq_raster) as r:
        prof = r.profile; wt = r.read(1)
    with rasterio.open(os.path.join(warm, region_time + "_starting_wt.tif"), "w", **prof) as w:
        w.write(wt.astype(prof["dtype"]), 1)
    # template cfg + southern edge
    for aux in ("eq_anderson.cfg", "island.cfg", "_se.txt"):
        s = os.path.join(src, aux)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(warm, aux))
    return region_time

def suite_warm(warm, tag, weeks=(1, 2, 4, 8, 16, 32, 64, 128), cycles=3, maxiter=6, timeout=200):
    results = []
    for sname, flags in suite.SOLVERS:
        ceiling = None
        for wk in weeks:
            name = f"{tag}_warm_{sname}_{wk}wk"
            res = suite.run_one(warm, name, flags, wk * suite.WEEK, cycles, maxiter,
                                "equilibrium", 1, timeout, settle_thresh=0.5)
            res["solver"] = sname; res["weeks"] = wk
            results.append(res)
            ok = res["status"] == "OK"
            if ok:
                ceiling = wk
            print(f"  {sname:14s} {wk:3d}wk: {res['status']:7s} wall={res['wall']:6.1f}s "
                  f"iters={res['tot_iters']:6d}", flush=True)
            if not ok:
                break  # first failure => ceiling found; stop increasing dt for this solver
        print(f"  -> {sname:14s} step ceiling = {ceiling} wk", flush=True)
    return results

if __name__ == "__main__":
    src = sys.argv[1]
    eq_raster = sys.argv[2]
    tag = sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    warm = os.path.join(os.path.dirname(src), f"warm_{tag}")
    print(f"WARM SUITE src={src} eq={eq_raster} scale={scale} -> {warm}", flush=True)
    rt = setup_warm_domain(src, warm, eq_raster, scale)
    print("warm domain ready, region_time =", rt, flush=True)
    res = suite_warm(warm, tag)
    outdir = os.path.dirname(os.path.abspath(__file__))
    json.dump(res, open(os.path.join(outdir, f"{tag}_warm.json"), "w"), indent=1)
    print("DONE ->", os.path.join(outdir, f"{tag}_warm.json"), flush=True)
