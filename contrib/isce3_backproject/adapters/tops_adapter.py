"""
Converters for TOPS burst SLC data to ISCE3 backprojection parameters.

Handles the burst-level geometry: each burst has its own sensing window,
Doppler variation, and valid sample range.
"""

import numpy as np
from .isce2_adapter import orbit_from_isce2, lookside_from_isce2, _datetime_to_isce3
from ..backproject import (
    LUT2d,
    RadarGridParameters,
    RadarGeometry,
    DEMInterpolator,
    KnabKernel,
    TabulatedKernel,
    ErrorCode,
    backproject,
)


def burst_to_radar_geometry(burst, orbit_isce3, look_side):
    """
    Convert an ISCE2 BurstSLC to an ISCE3 RadarGeometry.

    Parameters
    ----------
    burst : BurstSLC
        ISCE2 burst object with sensingStart, numberOfLines, numberOfSamples,
        startingRange, rangePixelSize, azimuthTimeInterval (1/PRF).
    orbit_isce3 : Orbit
        Pre-converted ISCE3 orbit.
    look_side : LookSide
        ISCE3 LookSide enum.

    Returns
    -------
    RadarGeometry
    """
    sensing_start = _datetime_to_isce3(burst.sensingStart)
    prf = 1.0 / burst.azimuthTimeInterval
    wavelength = burst.radarWavelength
    starting_range = burst.startingRange
    range_pxl_spacing = burst.rangePixelSize

    grid = RadarGridParameters(
        0.0,
        wavelength,
        prf,
        starting_range,
        range_pxl_spacing,
        look_side,
        burst.numberOfLines,
        burst.numberOfSamples,
        sensing_start,
    )

    lut = LUT2d(0.0)
    return RadarGeometry(grid, orbit_isce3, lut)


def read_burst_slc(burst):
    """
    Read burst SLC data as a complex64 numpy array.

    Parameters
    ----------
    burst : BurstSLC
        ISCE2 burst with image attribute pointing to SLC file.

    Returns
    -------
    np.ndarray
        Complex64 array of shape (numberOfLines, numberOfSamples).
    """
    img = burst.image
    filename = img.filename
    width = burst.numberOfSamples
    length = burst.numberOfLines
    return np.fromfile(filename, dtype=np.complex64).reshape(length, width)


def refocus_burst(burst, orbit_isce3, look_side, dem=None, num_threads=None):
    """
    Refocus a single burst SLC using backprojection.

    Parameters
    ----------
    burst : BurstSLC
        ISCE2 burst object.
    orbit_isce3 : Orbit
        Pre-converted ISCE3 orbit.
    look_side : LookSide
        ISCE3 LookSide enum.
    dem : DEMInterpolator, optional
        DEM for terrain correction. Defaults to flat Earth (h=0).
    num_threads : int, optional
        Number of OpenMP threads.

    Returns
    -------
    np.ndarray
        Refocused complex64 SLC data.
    """
    from ..backproject import set_num_threads

    if num_threads is not None:
        set_num_threads(num_threads)

    in_geom = burst_to_radar_geometry(burst, orbit_isce3, look_side)
    out_geom = burst_to_radar_geometry(burst, orbit_isce3, look_side)

    slc_data = read_burst_slc(burst)

    if dem is None:
        dem = DEMInterpolator(0.0)

    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)
    c = 299792458.0
    fc = c / burst.radarWavelength
    ds = burst.rangePixelSize

    result, height, ec = backproject(
        slc_data,
        in_geom,
        out_geom,
        dem,
        fc=fc,
        ds=ds,
        kernel=kernel,
    )

    if ec != ErrorCode.Success:
        raise RuntimeError(f"backproject() returned {ec} for burst")

    return result
