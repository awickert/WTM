#!/usr/bin/env python3
"""Band-limited synthetic terrain from a sum of 2D Fourier (sine) modes.

Shared terrain generator for WTM test fixtures (and, later, example terrain). The
surface is a sum of separable sinusoidal modes over the domain interior:

    topo(x, y) = base + Σ_m  amp_m · sin(2π·kx_m·x'/span_x + φ_m)
                                    · sin(2π·ky_m·y'/span_y + φ_m)

where (x', y') are interior coordinates and (kx, ky) are integer wavenumbers (cycles
across the interior span). Low wavenumbers give a few large closed basins; higher
"overtones" give more, smaller basins. Phased to vanish at the interior edge, the
product of sines yields *interior closed depressions* -- exactly what FillSpillMerge
ponds surface water into. The single mode (1, 1) is two hills and two depressions
(the topography Kerry & Andy use for WTM tests).

Design constraints (learned the hard way from the golden fixtures):

  * BAND-LIMIT. Keep max(kx, ky) well below the Nyquist wavenumber (~span/2) so the
    gradient stays smooth. Smooth gradients make FillSpillMerge flow routing
    unambiguous, hence cross-rank reproducible. Sharp or flat terrain puts spill
    points on knife-edges where a sub-1e-6 per-rank solver-rounding difference flips
    lake extent -- which is how a runoff+FSM fixture became cross-rank sensitive.

  * BE DETERMINISTIC. Explicit modes, no RNG, so committed golden references
    regenerate bit-for-bit. (A random 1/|k|^β spectral-synthesis variant for
    naturalistic *example* terrain is a natural extension via numpy.fft.ifft2 -- keep
    it OUT of golden references, where numpy's RNG stream drift would break them.)

Returns (topo, mask): float32 elevation and a float32 land/ocean mask (1 land, 0 ocean)
with a one-cell ocean ring, matching the raster conventions of the WTM test fixtures.
"""
import numpy as np


def spectral_terrain(shape, modes, base=0.0, ocean_ring=True, dtype=np.float32):
    """Sum of 2D sine modes -> (topo, mask).

    shape       : (ny, nx)
    modes       : iterable of (kx, ky, amp[, phase]); kx, ky integer wavenumbers,
                  amp in metres, phase in radians (default 0). Keep max(kx, ky) below
                  ~min(span)/2 to stay band-limited (smooth, cross-rank stable).
    base        : elevation datum added to the mode sum (m).
    ocean_ring  : if True, the outer one-cell ring is ocean (mask 0) and the interior
                  span (nx-2, ny-2) sets the fundamental wavelength.
    """
    ny, nx = shape
    off = 1 if ocean_ring else 0
    span_x = (nx - 2) if ocean_ring else nx
    span_y = (ny - 2) if ocean_ring else ny

    yy, xx = np.mgrid[0:ny, 0:nx]
    field = np.zeros((ny, nx), dtype=np.float64)
    for mode in modes:
        kx, ky, amp = mode[0], mode[1], mode[2]
        phase = mode[3] if len(mode) > 3 else 0.0
        field += amp * (
            np.sin(2.0 * np.pi * kx * (xx - off) / span_x + phase)
            * np.sin(2.0 * np.pi * ky * (yy - off) / span_y + phase)
        )
    topo = (base + field).astype(dtype)

    mask = np.ones((ny, nx), dtype=dtype)
    if ocean_ring:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    return topo, mask
