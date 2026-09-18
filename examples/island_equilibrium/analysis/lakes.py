import glob, os, re, sys
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"
STEM = "anderson_fixed_dt31536000_eq_n1"
mask = rasterio.open(os.path.join(W, "corsica_t0_mask.tif")).read(1) > 0
fs = sorted(glob.glob(os.path.join(W, f"{STEM}_*.tif")),
            key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))[-61:]
print("%6s %10s %14s %12s %12s" % ("cycle", "lakecells", "lakevol(m)", "wtd(119,64)", "wtd(121,63)"))
n_l, v_l, c1, c2 = [], [], [], []
for i, p in enumerate(fs):
    a = rasterio.open(p).read(1).astype(float)
    lake = mask & (a > 0.05)
    n_l.append(int(lake.sum())); v_l.append(float(a[lake].sum()))
    c1.append(a[119, 64]); c2.append(a[121, 63])
    if i < 24:
        print("%6d %10d %14.4f %12.4f %12.4f" % (i, n_l[-1], v_l[-1], c1[-1], c2[-1]))
n_l, v_l, c1, c2 = map(np.array, (n_l, v_l, c1, c2))
print()
print(f"lake CELL COUNT over 60 cycles: min {n_l.min()} max {n_l.max()} unique {len(set(n_l.tolist()))}")
print(f"lake VOLUME  : min {v_l.min():.3f} max {v_l.max():.3f}  span {v_l.max()-v_l.min():.3f} m")
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/d) if d else float("nan")
print(f"\ncorrelation lake VOLUME vs wtd(119,64): {corr(v_l, c1):+.3f}")
print(f"correlation lake VOLUME vs wtd(121,63): {corr(v_l, c2):+.3f}")
print(f"correlation lake COUNT  vs wtd(121,63): {corr(n_l.astype(float), c2):+.3f}")
