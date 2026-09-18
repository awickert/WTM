"""5x5 block of the REAL corsica terrain, outer ring held fixed, interior 3x3 integrated.

Everything is the model's own: piecewise T, harmonic-mean interfaces, 5-point stencil, specificYield
storage, surface cap as the collector. Real topography, real fdepth from real slopes, real cell sizes.
The outer ring is Dirichlet at each cell's TIME-MEAN wtd from the shipped run -- so the uphill supply
and the downhill sink are both present and at their measured strength.
"""
import glob, os, re, sys
import numpy as np, rasterio
W = "examples/island_equilibrium/_work_corsica"
def band(n): return rasterio.open(os.path.join(W, n)).read(1).astype(float)
topo, slope, ks = band("corsica_t0_topography.tif"), band("corsica_t0_slope.tif"), band("corsica_horizontal_ksat.tif")
fs = sorted(glob.glob(os.path.join("/tmp/claude-1000/dtsweep", "a_dt1_0*.tif")),
            key=lambda p: int(re.search(r"_(\d{9})_", p).group(1)))
st = np.stack([rasterio.open(p).read(1).astype(float) for p in fs])
R0, C0 = 118, 61                          # top-left of the 5x5
Z  = topo[R0:R0+5, C0:C0+5]
FD = np.maximum(200.0/(1+150*slope[R0:R0+5, C0:C0+5]), 2.0)
K  = ks[R0:R0+5, C0:C0+5]
Hbar = st[:, R0:R0+5, C0:C0+5].mean(axis=0)     # time-mean, used for the fixed ring
H0   = st[0,  R0:R0+5, C0:C0+5].copy()          # initial condition = the real restart state
YR, phi, eps = 31536000.0, 0.25, 0.01
lat = 41.2 + (156 - 120)/120.0
dy = (1/120.)*111320.; dx = dy*np.cos(np.radians(lat))
R = 0.060/YR
def T(h, f, k):
    return np.where(h < -1.5, f*k*np.exp((h+1.5)/f), np.where(h > 0, k*(1.5+f), k*(h+1.5+f)))
def S(h): return 0.5*((1+phi) + h*(1-phi)/np.sqrt(h*h + eps*eps))
INT = [(i, j) for i in range(1, 4) for j in range(1, 4)]
def rhs(h):
    Tm = T(h, FD, K); d = np.zeros_like(h)
    for (i, j) in INT:
        net = 0.0
        for (di, dj, L) in ((-1,0,dy), (1,0,dy), (0,-1,dx), (0,1,dx)):
            ii, jj = i+di, j+dj
            Th = 2*Tm[i,j]*Tm[ii,jj]/(Tm[i,j]+Tm[ii,jj])
            net += Th*((Z[ii,jj]+h[ii,jj]) - (Z[i,j]+h[i,j]))/L**2
        d[i,j] = (R + net)/S(h[i,j])
    return d
def run(years, dt_yr, removal, h_init):
    h = h_init.copy(); dt = dt_yr*YR; out = []
    for n in range(int(years/dt_yr)):
        k1 = rhs(h); k2 = rhs(h+0.5*dt*k1); k3 = rhs(h+0.5*dt*k2); k4 = rhs(h+dt*k3)
        h = h + dt/6*(k1+2*k2+2*k3+k4)
        h[0,:] = H0[0,:]*0 + Hbar[0,:]; h[4,:] = Hbar[4,:]      # ring stays fixed
        h[:,0] = Hbar[:,0]; h[:,4] = Hbar[:,4]
        if removal: h = np.minimum(h, 0.0)
        out.append((n*dt_yr, h[2,2], h[3,2]))                    # A=(120,63)->(2,2)  B=(121,63)->(3,2)
    return np.array(out)
for removal in (True, False):
    r = run(900, 0.02, removal, H0)
    tail = r[len(r)//2:]
    print(f"=== 5x5 block, removal {'ON ' if removal else 'OFF'} ===")
    print("    t(yr)      h_A       h_B")
    for row in r[::int(len(r)/16)][:16]: print("  %7.0f %9.3f %9.3f" % tuple(row))
    print("  second half: A span %.3f m   B span %.3f m\n"
          % (tail[:,1].max()-tail[:,1].min(), tail[:,2].max()-tail[:,2].min()))
