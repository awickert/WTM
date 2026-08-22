"""Shared GeoTIFF writer for WTM test fixtures.

WTM derives its grid geometry (cells-per-degree spacing, southern-edge latitude, and the cos-latitude
cell-size scaling) from each input raster's GDAL geotransform (#124). Test fixtures must therefore carry a
CORRECT geotransform, not the arbitrary placeholder the generators used when WTM ignored georeferencing.

Rather than hand-roll a geotransform in every generator (brittle, and easy to drift from the intended grid),
all generators route through this ONE helper. It builds a north-up, square lat-lon geotransform directly from
the intended (cells_per_degree, southern_edge) so WTM re-derives exactly those values.

Geotransform (GDAL affine): pixel = 1 / cells_per_degree degrees; origin is the top-left (north-west) corner
at (western_edge, north_edge) where north_edge = southern_edge + n_rows * pixel; dy = -pixel (north-up).
"""

import numpy as np
import rasterio
from rasterio.transform import Affine

DEFAULT_CRS = "EPSG:4326"


def make_transform(cells_per_degree, southern_edge, n_rows, western_edge=0.0):
    """Return a rasterio Affine for a north-up, square lat-lon grid of n_rows rows."""
    pixel = 1.0 / float(cells_per_degree)
    north_edge = southern_edge + n_rows * pixel  # top edge = south edge + height
    # Affine(a, b, c, d, e, f) = (dx, 0, x0, 0, dy, y0); dy < 0 for north-up.
    return Affine(pixel, 0.0, western_edge, 0.0, -pixel, north_edge)


def write_tif(path, data, cells_per_degree, southern_edge, western_edge=0.0, crs=DEFAULT_CRS, dtype=None):
    """Write a single-band GeoTIFF with a correct geotransform for the given grid.

    data: 2-D array (n_rows, n_cols), row 0 = north (standard GDAL orientation).
    """
    data = np.asarray(data)
    n_rows, n_cols = data.shape
    transform = make_transform(cells_per_degree, southern_edge, n_rows, western_edge)
    if dtype is None:
        dtype = "float32" if data.dtype.kind == "f" else str(data.dtype)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=n_rows, width=n_cols, count=1,
        dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data.astype(dtype), 1)
