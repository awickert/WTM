import glob, os, re
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"
def band(n): return rasterio.open(os.path.join(W, n)).read(1).astype(float)
topo, slope, ks = band("corsica_t0_topography.tif"), band("corsica_t0_slope.tif"), band("corsica_horizontal_ksat.tif")
B = (121, 63)
def fd(c): return max(200.0/(1+150*slope[c]), 2.0)
def T(c, w):
    f, k, sh = fd(c), ks[c], 1.5
    w = float(w)
    return f*k*np.exp((w+sh)/f) if w < -sh else (k*(sh+f) if w > 0 else k*(w+sh+f))
def load(d, stem):
    fs = sorted(glob.glob(os.path.join(d, f"{stem}_0*.tif")),
                key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
    return np.stack([rasterio.open(p).read(1).astype(float) for p in fs])
lat = 41.2 + (156 - B[0])/120.0
dy = (1/120.)*111320.; dx = dy*np.cos(np.radians(lat)); Ar = dx*dy
N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
runs = {"removal ON ": load("/tmp/claude-1000/dtsweep", "a_dt1"),
        "removal OFF": load("/tmp/claude-1000/calm", "nofsm")}
print("WHERE DOES CELL B (121,63, topo 874) GET ITS WATER?  mean over 41 reports, m/yr into B.\n")
hdr = "%-12s" % "run"
for d in N8: hdr += "%11s" % f"{topo[B[0]+d[0], B[1]+d[1]]:.0f}"
print(hdr + "%11s" % "NET")
for nm, st in runs.items():
    tot = np.zeros(len(N8)); net = 0.0
    for i in range(len(st)):
        w0 = st[i][B]
        for j, d in enumerate(N8):
            n = (B[0]+d[0], B[1]+d[1]); wn = st[i][n]
            Th = 2*T(B,w0)*T(n,wn)/(T(B,w0)+T(n,wn))
            diag = (d[0] != 0 and d[1] != 0)
            L = (dx if d[0] else dy)*(0.5 if diag else 1.0)
            dist = np.hypot(dx if d[1] else 0, dy if d[0] else 0)
            tot[j] += Th*((topo[n]+wn)-(topo[B]+w0))/dist*L/Ar*31536000.0
    tot /= len(st)
    print("%-12s" % nm + "".join("%11.4f" % v for v in tot) + "%11.4f" % tot.sum())
print("\n  (1084) is cell A, the driver, directly uphill.  Positive = flow INTO B.")
print("\nB's OWN TRAJECTORY:")
for nm, st in runs.items():
    wb = st[:, B[0], B[1]]
    print(f"  {nm}: " + " ".join(f"{x:7.2f}" for x in wb[:14]))
