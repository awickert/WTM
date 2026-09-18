import glob, os, re, sys
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"
STEM = "anderson_fixed_dt31536000_eq_n1"
fs = sorted(glob.glob(os.path.join(W, f"{STEM}_*.tif")),
            key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))[-61:]
cells = [(119, 64), (98, 77), (121, 63), (119, 65)]
ser = {c: [] for c in cells}
for p in fs:
    a = rasterio.open(p).read(1).astype(float)
    for c in cells:
        ser[c].append(a[c])
print("water table at the worst offenders, last 60 cycles (m)")
print("%6s" % "cycle", "".join("%14s" % str(c) for c in cells))
base = len(fs) - 1
for i in range(0, 24):
    print("%6d" % i, "".join("%14.5f" % ser[c][i] for c in cells))
print("...")
for c in cells:
    v = np.array(ser[c])
    d = np.diff(v)
    sign_flips = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    print(f"\ncell {c}: range {v.min():.4f} .. {v.max():.4f}  (span {v.max()-v.min():.4f} m)")
    print(f"   mean {v.mean():.4f}   drift over 60 cycles = {v[-1]-v[0]:+.5f} m")
    print(f"   consecutive-step sign flips: {sign_flips} of {len(d)-1}"
          f"   -> {'ALTERNATING (period-2)' if sign_flips > 0.8*(len(d)-1) else 'not a clean period-2'}")
