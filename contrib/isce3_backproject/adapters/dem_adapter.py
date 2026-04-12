"""
DEM adapter for isce3_backproject.

Loads raster DEM files (GeoTIFF, etc.) via rasterio and wraps them
as DEMInterpolator objects that the C++ backprojection code can use
through the HeightFunc callback interface.

Requirements: rasterio, scipy, numpy
"""

import numpy as np


def dem_from_file(dem_path, band=1, interp_method="bilinear"):
    """
    Load a raster DEM file and return a DEMInterpolator.

    Parameters
    ----------
    dem_path : str
        Path to DEM raster file (GeoTIFF, VRT, etc.).
    band : int
        Raster band number (1-indexed, default 1).
    interp_method : str
        Interpolation method: 'bilinear' (default) or 'nearest'.

    Returns
    -------
    DEMInterpolator
        C++ DEMInterpolator backed by the loaded raster.

    Examples
    --------
    >>> from isce3_backproject.adapters.dem_adapter import dem_from_file
    >>> dem = dem_from_file("srtm_30m.tif")
    >>> dem.refHeight()
    500.0
    >>> dem.interpolateLonLat(120.5, 31.2)
    523.4
    """
    import rasterio
    from scipy.interpolate import RegularGridInterpolator

    from ..backproject import DEMInterpolator

    with rasterio.open(dem_path) as ds:
        data = ds.read(band).astype(np.float64)
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata
        height, width = data.shape

    epsg = 4326
    if crs and crs.to_epsg():
        epsg = crs.to_epsg()

    if nodata is not None:
        valid = data != nodata
        if valid.any():
            min_h = float(np.nanmin(data[valid]))
            max_h = float(np.nanmax(data[valid]))
            mean_h = float(np.nanmean(data[valid]))
        else:
            min_h = max_h = mean_h = 0.0
        data[~valid] = mean_h
    else:
        min_h = float(np.nanmin(data))
        max_h = float(np.nanmax(data))
        mean_h = float(np.nanmean(data))

    # Build coordinate arrays (pixel centers)
    # transform maps (col, row) → (x, y)
    row_coords = np.array(
        [transform.f + (r + 0.5) * transform.e for r in range(height)]
    )
    col_coords = np.array([transform.c + (c + 0.5) * transform.a for c in range(width)])

    method = "nearest" if interp_method == "nearest" else "linear"
    interp = RegularGridInterpolator(
        (row_coords, col_coords),
        data,
        method=method,
        bounds_error=False,
        fill_value=mean_h,
    )

    def height_func(lon, lat):
        return float(interp((lat, lon)))

    dem = DEMInterpolator(height_func, mean_h, epsg)
    dem.setStats(min_h, mean_h, max_h)
    return dem


def dem_from_array(data, transform, epsg=4326, nodata=None, interp_method="bilinear"):
    """
    Create a DEMInterpolator from a numpy array + affine transform.

    Parameters
    ----------
    data : np.ndarray
        2D DEM height array (rows=lat, cols=lon).
    transform : affine.Affine or tuple
        Affine transform (rasterio-style 6-element).
        If tuple: (x_origin, x_res, 0, y_origin, 0, y_res).
    epsg : int
        EPSG code (default 4326).
    nodata : float or None
        No-data value to mask.
    interp_method : str
        'bilinear' (default) or 'nearest'.

    Returns
    -------
    DEMInterpolator
    """
    from scipy.interpolate import RegularGridInterpolator

    from ..backproject import DEMInterpolator

    data = np.asarray(data, dtype=np.float64)
    height, width = data.shape

    if hasattr(transform, "a"):
        c, a, _, f, _, e = (
            transform.c,
            transform.a,
            transform.b,
            transform.f,
            transform.d,
            transform.e,
        )
    else:
        c, a, _, f, _, e = transform

    if nodata is not None:
        valid = data != nodata
        if valid.any():
            mean_h = float(np.nanmean(data[valid]))
        else:
            mean_h = 0.0
        data = data.copy()
        data[~valid] = mean_h
    else:
        mean_h = float(np.nanmean(data))

    min_h = float(np.nanmin(data))
    max_h = float(np.nanmax(data))

    row_coords = np.array([f + (r + 0.5) * e for r in range(height)])
    col_coords = np.array([c + (col + 0.5) * a for col in range(width)])

    method = "nearest" if interp_method == "nearest" else "linear"
    interp = RegularGridInterpolator(
        (row_coords, col_coords),
        data,
        method=method,
        bounds_error=False,
        fill_value=mean_h,
    )

    def height_func(lon, lat):
        return float(interp((lat, lon)))

    dem = DEMInterpolator(height_func, mean_h, epsg)
    dem.setStats(min_h, mean_h, max_h)
    return dem
