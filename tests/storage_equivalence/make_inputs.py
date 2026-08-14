#!/usr/bin/env python3
"""Fixture for the SECANT ≡ VOLUME backward-Euler storage equivalence test.

WTM's default (Callaghan) backward Euler writes the storage term as S·Δh with the EXACT secant
effective storativity S = (V(wⁿ⁺¹) − V(wⁿ))/(wⁿ⁺¹ − wⁿ); the -wtm_volume_storage / BDF2-on-V schemes use
the stored-volume change ΔV directly. Because S is the exact secant, S·Δh ≡ ΔV *identically* -- even across
the surface where the specific yield jumps from porosity to ~1. So on a WELL-BEHAVED (non-oscillating)
domain the two forms must give the identical water table to machine precision. (They diverge only in a
surface limit-cycle, where the ÷S vs ÷Sy residual scaling makes Anderson's last digits differ and the
flicker amplifies them -- a numerical artifact, not a real difference; see
finding_cc_secant_storage_inconsistency.)

A coastal wedge that drains to an ocean strip (west) and reaches a smooth steady state -- coastal cells
cross wtd=0 (so S ≠ Sy pointwise, the meaningful case) but the table does not flicker. Regenerate with:
    python3 make_inputs.py
"""
import numpy as np, os, rasterio
from rasterio.transform import from_bounds

NX, NY = 12, 8
REGION = "storeq"
OUT = os.path.join(os.path.dirname(__file__), "inputs")
os.makedirs(OUT, exist_ok=True)
tr = from_bounds(0, 0, NX, NY, NX, NY)

def w(name, data, dt="float32"):
    with rasterio.open(os.path.join(OUT, name), "w", driver="GTiff", height=NY, width=NX, count=1,
                       dtype=dt, crs="EPSG:4326", transform=tr) as d:
        d.write(data.astype(dt), 1)

topo = np.full((NY, NX), 100.0, np.float32)  # high plateau
topo[:, 0] = 0.0                             # ocean strip (west) at sea level -> a drainage gradient
mask = np.ones((NY, NX), np.float32); mask[:, 0] = 0.0
zero = np.zeros((NY, NX), np.float32)
# precip tuned (like tests/recharge_consistency) so the interior mound rises ACROSS the land surface within
# the run -- the cells then sit in the storativity transition (S != Sy pointwise), the case that MATTERS:
# there the exact-secant identity S*Δh ≡ ΔV is nontrivial. A short run keeps it pre-flicker.
for lay, a in {"topography":topo,"slope":zero,"mask":mask,"precipitation":np.full((NY,NX),1.5,np.float32),
               "evaporation":zero,"open_water_evaporation":zero,"winter_temperature":zero}.items():
    w(f"{REGION}_ta_{lay}.tif", a); w(f"{REGION}_tb_{lay}.tif", a)
w(f"{REGION}_horizontal_ksat.tif", np.full((NY, NX), 1e-4, np.float32))
w(f"{REGION}_porosity.tif", np.full((NY, NX), 0.25, np.float32))
w(f"{REGION}_ta_wtd.tif", np.full((NY, NX), -2.0, np.float64), "float64")  # starts below -> crosses up
print("wrote", OUT, "region", REGION)
