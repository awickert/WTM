import glob, os, re, sys
import numpy as np, rasterio
sys.path.insert(0, "tests")
import wtm_volume as VOL
W = "examples/island_equilibrium/_work_corsica"
STEM = "anderson_fixed_dt31536000_eq_n1"
phi   = VOL.read_band(os.path.join(W, "corsica_porosity.tif"))
mask  = rasterio.open(os.path.join(W, "corsica_t0_mask.tif")).read(1) > 0
topo  = rasterio.open(os.path.join(W, "corsica_t0_topography.tif")).read(1).astype(float)
slope = rasterio.open(os.path.join(W, "corsica_t0_slope.tif")).read(1).astype(float)
TOL = 1e-3
fs = sorted(glob.glob(os.path.join(W, f"{STEM}_*.tif")),
            key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
N = 200
fs = fs[-(N+1):]
count = np.zeros(mask.shape, dtype=int)
wtd_last = None
prev = rasterio.open(fs[0]).read(1).astype(float)
for p in fs[1:]:
    cur = rasterio.open(p).read(1).astype(float)
    dv = np.where(mask, VOL.volume_diff(cur, prev, phi), 0.0)
    count += (dv > TOL)
    prev = cur
wtd_last = prev
land = mask
off = count > 0
print(f"over the last {N} cycles, {int(off.sum())} distinct land cells EVER exceed tol "
      f"({100*off.sum()/land.sum():.3f}% of {int(land.sum())} land cells)")
print(f"cells offending in EVERY cycle: {int((count == N).sum())}")
print(f"cells offending in >90% of cycles: {int((count > 0.9*N).sum())}")
print()
ys, xs = np.where(off)
order = np.argsort(-count[off])
print("%5s %5s %7s %9s %8s %9s %10s" % ("row", "col", "hits", "topo", "slope", "wtd", "fdepth"))
for k in order[:12]:
    y, x = ys[k], xs[k]
    fd = max(200.0 / (1 + 150 * slope[y, x]), 2.0)
    print("%5d %5d %7d %9.1f %8.4f %9.3f %10.2f" % (y, x, count[y, x], topo[y, x], slope[y, x], wtd_last[y, x], fd))
print()
o = off & land
print("OFFENDERS vs ALL LAND, medians:")
for nm, arr in (("topo", topo), ("slope", slope), ("wtd", wtd_last)):
    print("  %-6s offenders %10.4f      all land %10.4f" % (nm, np.median(arr[o]), np.median(arr[land])))
print(f"  wtd >= -0.01 m (at/near surface): offenders {100*np.mean(wtd_last[o] >= -0.01):.1f}%"
      f"   all land {100*np.mean(wtd_last[land] >= -0.01):.1f}%")
np.save("/tmp/claude-1000/offenders.npy", count)
