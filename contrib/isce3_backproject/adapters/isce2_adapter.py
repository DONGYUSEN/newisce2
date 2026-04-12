"""
Converters from ISCE2 data objects to ISCE3 backprojection parameters.

Converts:
  - isceobj.Orbit.Orbit → isce3 Orbit (list of StateVectors)
  - isceobj.Scene.Frame → isce3 RadarGridParameters
  - pointingDirection → LookSide enum
"""

import datetime
import numpy as np

from ..backproject import (
    DateTime as Isce3DateTime,
    TimeDelta,
    Vec3,
    StateVector as Isce3StateVector,
    Orbit as Isce3Orbit,
    LookSide,
    LUT2d,
    RadarGridParameters,
    RadarGeometry,
    DEMInterpolator,
)


def _datetime_to_isce3(dt):
    """Convert Python datetime.datetime to ISCE3 DateTime ISO string constructor."""
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
    usec = dt.microsecond
    nsec_str = f"{usec:06d}000"
    return Isce3DateTime(f"{iso}.{nsec_str}")


def orbit_from_isce2(isce2_orbit):
    """Convert an ISCE2 Orbit (iterable of StateVectors) to an ISCE3 Orbit."""
    svecs = []
    for sv in isce2_orbit:
        t = sv.getTime() if callable(getattr(sv, "getTime", None)) else sv.time
        pos = (
            sv.getPosition()
            if callable(getattr(sv, "getPosition", None))
            else sv.position
        )
        vel = (
            sv.getVelocity()
            if callable(getattr(sv, "getVelocity", None))
            else sv.velocity
        )

        isv = Isce3StateVector()
        isv.datetime = _datetime_to_isce3(t)
        isv.position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        isv.velocity = Vec3(float(vel[0]), float(vel[1]), float(vel[2]))
        svecs.append(isv)

    return Isce3Orbit(svecs)


def lookside_from_isce2(pointing_direction):
    """Convert ISCE2 pointingDirection (-1=right, 1=left) to ISCE3 LookSide."""
    if int(pointing_direction) == -1:
        return LookSide.Right
    elif int(pointing_direction) == 1:
        return LookSide.Left
    else:
        raise ValueError(f"Unknown pointingDirection: {pointing_direction}")


def radargrid_from_isce2(frame):
    """
    Build an ISCE3 RadarGridParameters from an ISCE2 Frame.

    Uses frame.sensingStart, frame.PRF, frame.radarWavelength,
    frame.startingRange, frame.rangeSamplingRate, frame.numberOfLines,
    frame.numberOfSamples, frame.instrument.platform.pointingDirection.
    """
    try:
        from isceobj.Constants import SPEED_OF_LIGHT
    except ImportError:
        SPEED_OF_LIGHT = 299792458.0

    sensing_start_isce3 = _datetime_to_isce3(frame.sensingStart)
    wavelength = frame.radarWavelength
    prf = frame.PRF
    starting_range = frame.startingRange
    range_sampling_rate = frame.rangeSamplingRate
    range_pxl_spacing = SPEED_OF_LIGHT / (2.0 * range_sampling_rate)
    look_side = lookside_from_isce2(frame.instrument.platform.pointingDirection)
    n_az = frame.numberOfLines
    n_rg = frame.numberOfSamples

    return RadarGridParameters(
        0.0,  # sensingStart as epoch offset (seconds)
        wavelength,
        prf,
        starting_range,
        range_pxl_spacing,
        look_side,
        n_az,
        n_rg,
        sensing_start_isce3,
    )


def frame_metadata_from_isce2(frame):
    """
    Extract all metadata needed for backprojection from an ISCE2 Frame.

    Returns a dict with keys: orbit, grid, doppler_coeffs, chirp_slope,
    pulse_length, range_sampling_rate, iq_bias, wavelength, fc.
    """
    instr = (
        frame.getInstrument() if hasattr(frame, "getInstrument") else frame.instrument
    )
    try:
        from isceobj.Constants import SPEED_OF_LIGHT
    except ImportError:
        SPEED_OF_LIGHT = 299792458.0

    return {
        "orbit": orbit_from_isce2(frame.orbit),
        "grid": radargrid_from_isce2(frame),
        "doppler_coeffs": list(frame._dopplerVsPixel),
        "chirp_slope": instr.chirpSlope,
        "pulse_length": instr.pulseLength,
        "range_sampling_rate": instr.rangeSamplingRate,
        "iq_bias": (instr.inPhaseValue, instr.quadratureValue),
        "wavelength": instr.radarWavelength,
        "fc": SPEED_OF_LIGHT / instr.radarWavelength,
        "prf": instr.PRF,
        "look_side": lookside_from_isce2(instr.platform.pointingDirection),
    }
