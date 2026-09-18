import glob, os, re, sys
import numpy as np, rasterio
sys.path.insert(0, "tests")
import wtm_volume as VOL
W = "examples/island_equilibrium/_work_corsica"
STEM = "anderson_fixed_dt31536000_eq_n1"
phi  = VOL.read_band(os.path.join(W, "corsica_porosity.tif"))
mask = rasterio.open(os.path.join(W, "corsica_t0_mask.tif")).read(1) > 0
nland = int(mask.sum())
TOL, FRAC = 1e-3, 1e-3

fs = sorted(glob.glob(os.path.join(W, f"{STEM}_*.tif")),
            key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
print(f"{len(fs)} post-FSM snapshots, {nland} land cells; stop = frac(dv>{TOL}) < {FRAC}")
prev = rasterio.open(fs[0]).read(1).astype(float)
out = []
for k, p in enumerate(fs[1:], start=1):
    cur = rasterio.open(p).read(1).astype(float)
    dv = np.where(mask, VOL.volume_diff(cur, prev, phi), np.nan)
    mx = float(np.nanmax(dv)); rms = float(np.sqrt(np.nanmean(dv**2)))
    fr = float(np.nansum(dv > TOL)) / nland
    out.append((k, mx, rms, fr))
    prev = cur
np.save("/tmp/claude-1000/remeasure.npy", np.array(out))
print("%7s %12s %12s %10s" % ("cycle", "dvol_max", "dvol_rms", "frac>tol"))
for k, mx, rms, fr in out[::max(1, len(out)//14)]:
    print("%7d %12.4e %12.4e %10.5f" % (k, mx, rms, fr))
stop = next((k for k, mx, rms, fr in out if fr < FRAC), None)
print()
if stop:
    k, mx, rms, fr = out[stop-1]
    print(f"WOULD HAVE STOPPED at cycle {k} ({k*10} yr): frac={fr:.6f} < {FRAC}, dvol_max={mx:.4e}")
else:
    print(f"NEVER satisfies the stop: min frac over the run = {min(f for *_, f in out):.6f}")
last = out[-500:]
print(f"last 500 cycles: max mean {np.mean([m for _,m,_,_ in last]):.4e}   "
      f"rms mean {np.mean([r for _,_,r,_ in last]):.4e}   frac mean {np.mean([f for *_,f in last]):.6f}")
