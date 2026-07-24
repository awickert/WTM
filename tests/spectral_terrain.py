#!/usr/bin/env python3
"""Synthetic terrain from Fourier modes -- for WTM test fixtures and example terrain.

Two generators, both band-limited-friendly and sharing the ocean-ring convention of the
WTM raster fixtures:

  * spectral_terrain -- DETERMINISTIC sum of a few explicit 2D sine modes,
        topo = base + Σ amp·sin(2π·kx·x'/span)·sin(2π·ky·y'/span).
    Low wavenumbers give a few large closed depressions; higher "overtones" give more,
    smaller ones. The single mode (1, 1) is two hills and two depressions (the topography
    Kerry & Andy use for WTM tests). Use this for TESTS: explicit modes + no RNG means
    committed golden references regenerate bit-for-bit.

  * fractal_terrain -- NATURALISTIC terrain by spectral synthesis: a 1/|k|^β power-law
    amplitude spectrum with seeded random phases, inverse-FFT'd to real space. Use this
    for EXAMPLE/demo terrain, NOT golden references -- numpy's RNG stream can drift across
    versions, and the structure is uncontrolled (no guaranteed depression count).

Design constraint learned from the golden fixtures: BAND-LIMIT for anything fed to
FillSpillMerge. Smooth gradients make FSM flow routing unambiguous, hence cross-rank
reproducible; sharp or flat terrain puts spill points on knife-edges where a sub-1e-6
per-rank solver-rounding difference flips lake extent. spectral_terrain is band-limited by
construction (few low modes); fractal_terrain takes a kmax to cap the highest wavenumber.

Both return (topo, mask): float32 elevation and a float32 land/ocean mask (1 land, 0 ocean)
with a one-cell ocean ring.
"""
import numpy as np


def _ocean_mask(shape, ocean_ring, dtype):
    """Land/ocean mask (1 land, 0 ocean) with an optional one-cell ocean ring."""
    ny, nx = shape
    mask = np.ones((ny, nx), dtype=dtype)
    if ocean_ring:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    return mask


def spectral_terrain(shape, modes, base=0.0, ocean_ring=True, dtype=np.float32):
    """Sum of 2D sine modes -> (topo, mask). Deterministic; for test fixtures.

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
    return topo, _ocean_mask((ny, nx), ocean_ring, dtype)


def fractal_terrain(shape, beta=2.0, relief=1000.0, base=0.0, seed=0,
                    kmax=None, ocean_ring=True, dtype=np.float32):
    """Naturalistic terrain by spectral synthesis -> (topo, mask). For EXAMPLES, not
    golden references (seeded RNG can drift across numpy versions).

    A power-law amplitude spectrum A(k) ∝ |k|^(-beta) is given uniformly-random phases
    (seeded) and inverse-FFT'd; the real part is a random field with that spectrum.

    shape   : (ny, nx)
    beta     : amplitude-spectrum exponent (power exponent = 2*beta). ~2 gives smooth,
               correlated, Brownian-surface-like terrain; lower is rougher.
    relief   : peak-to-peak elevation range (m); the field is rescaled to [base, base+relief].
    base     : elevation of the lowest point (m).
    seed     : RNG seed (numpy default_rng / PCG64) -> reproducible for a given numpy.
    kmax     : if set, zero all modes with |k| > kmax (cycles per domain) -- band-limits the
               terrain so FillSpillMerge routing stays smooth/deterministic. None = full band.
    ocean_ring : one-cell ocean ring in the mask.
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)

    # Radial wavenumber |k| in cycles per domain (fftfreq * N gives integer wavenumbers).
    ky = (np.fft.fftfreq(ny) * ny)[:, None]
    kx = (np.fft.fftfreq(nx) * nx)[None, :]
    k = np.sqrt(kx * kx + ky * ky)

    with np.errstate(divide="ignore"):
        amp = np.where(k > 0, k ** (-beta), 0.0)
    amp[0, 0] = 0.0                       # drop DC; the datum is set by `base` below
    if kmax is not None:
        amp[k > kmax] = 0.0

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    field = np.fft.ifft2(amp * np.exp(1j * phase)).real

    field -= field.min()
    if field.max() > 0:
        field *= relief / field.max()     # -> [0, relief]
    topo = (base + field).astype(dtype)
    return topo, _ocean_mask((ny, nx), ocean_ring, dtype)


if __name__ == "__main__":
    # Smoke demo: a fractal terrain and its basic stats.
    topo, mask = fractal_terrain((64, 64), beta=2.0, relief=800.0, base=100.0, seed=1, kmax=16)
    print(f"fractal_terrain 64x64: elev [{topo.min():.1f}, {topo.max():.1f}] m, "
          f"land cells {int(mask.sum())}")
