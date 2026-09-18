import glob, os, re, sys
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"; S = "/tmp/claude-1000/dtsweep"
def band(n): return rasterio.open(os.path.join(W, n)).read(1).astype(float)
topo, slope, ks, phi = (band("corsica_t0_topography.tif"), band("corsica_t0_slope.tif"),
                        band("corsica_horizontal_ksat.tif"), band("corsica_porosity.tif"))
fs = sorted(glob.glob(os.path.join(S, "a_dt1_0*.tif")), key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
st = np.stack([rasterio.open(p).read(1).astype(float) for p in fs])
C = (120, 63)
def fd(c): return max(200.0/(1+150*slope[c]), 2.0)
def T(c, w): return fd(c)*ks[c]*np.exp((w+1.5)/fd(c))
lat = 41.2 + (156 - C[0])/120.0
dy = (1/120.)*111320.; dx = dy*np.cos(np.radians(lat)); A = dx*dy
N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
print("FLUX ACROSS THE CYCLE. Positive = INTO the driver. m/yr of water depth over the driver's area.")
print("Driver topo 1084 m. Neighbour topo in brackets.\n")
hdr = "%6s %8s" % ("report", "wtd")
for d in N8: hdr += "%12s" % f"{topo[C[0]+d[0], C[1]+d[1]]:.0f}"
hdr += "%10s %10s" % ("NET in", "recharge")
print(hdr)
print("%6s %8s" % ("", "") + "".join("%12s" % f"({d[0]:+d},{d[1]:+d})" for d in N8) + "%10s %10s" % ("m/yr", "m/yr"))
for i in list(range(0, 24)):
    w0 = st[i][C]; row = "%6d %8.3f" % (i, w0); net = 0.0
    for d in N8:
        n = (C[0]+d[0], C[1]+d[1])
        wn = st[i][n]
        h0, hn = topo[C]+w0, topo[n]+wn
        Th = 2*T(C,w0)*T(n,wn)/(T(C,w0)+T(n,wn))
        diag = (d[0] != 0 and d[1] != 0)
        L = (dx if d[0]!=0 else dy) * (0.5 if diag else 1.0)     # diagonal faces: half weight
        dist = np.hypot(dx if d[1] else 0, dy if d[0] else 0)
        q = Th*(hn-h0)/dist*L/A*31536000.0                        # + = inflow
        net += q; row += "%12.4f" % q
    row += "%10.4f %10.4f" % (net, 0.060)
    print(row)
