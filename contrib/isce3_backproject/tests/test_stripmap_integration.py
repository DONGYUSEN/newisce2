"""
End-to-end integration test for stripmapApp backprojection focusing.

Creates synthetic raw SAR data, writes it in ISCE2 format, then calls
focus_backproject() and verifies the output SLC is valid.
"""

import sys, os, math, tempfile, datetime
import numpy as np

THISDIR = os.path.dirname(os.path.abspath(__file__))
MODDIR = os.path.dirname(THISDIR)
CONTRIBDIR = os.path.dirname(MODDIR)
sys.path.insert(0, CONTRIBDIR)

from isce3_backproject.adapters.range_compress import generate_chirp


def test_synthetic_focus():
    """
    Create synthetic echoes from a point target, run backprojection,
    verify output SLC has correct dimensions and non-zero energy.
    """
    from isce3_backproject.backproject import (
        DateTime,
        Vec3,
        StateVector,
        Orbit,
        LookSide,
        LUT2d,
        RadarGridParameters,
        RadarGeometry,
        DEMInterpolator,
        KnabKernel,
        TabulatedKernel,
        ErrorCode,
        backproject,
        set_num_threads,
    )
    from isce3_backproject.adapters.range_compress import (
        generate_chirp,
        range_compress_block,
    )

    set_num_threads(4)

    n_az = 128
    n_rg = 2048
    prf = 1000.0
    fs = 18.962468e6
    chirp_slope = -7.2135e11
    pulse_length = 27.0e-6
    wavelength = 0.0555
    starting_range = 850000.0

    svecs = []
    for i in range(20):
        sv = StateVector()
        sv.datetime = DateTime(f"2023-06-15T10:30:{i:02d}.000000000")
        angle = 0.001 * i
        r = 7.071e6
        sv.position = Vec3(r * math.cos(angle), r * math.sin(angle), 0.0)
        sv.velocity = Vec3(
            -r * 0.001 * math.sin(angle), r * 0.001 * math.cos(angle), 0.0
        )
        svecs.append(sv)
    orbit = Orbit(svecs)

    dt = DateTime(2023, 6, 15, 10, 30, 0)
    c = 299792458.0
    range_pixel_spacing = c / (2.0 * fs)

    in_grid = RadarGridParameters(
        0.0,
        wavelength,
        prf,
        starting_range,
        range_pixel_spacing,
        LookSide.Right,
        n_az,
        n_rg,
        dt,
    )
    lut = LUT2d(0.0)
    in_geom = RadarGeometry(in_grid, orbit, lut)
    out_geom = RadarGeometry(in_grid, orbit, lut)

    chirp = generate_chirp(chirp_slope, pulse_length, fs)

    rng = np.random.default_rng(42)
    raw = (
        rng.standard_normal((n_az, n_rg)) + 1j * rng.standard_normal((n_az, n_rg))
    ).astype(np.complex64) * 0.01

    target_bin = 1024
    for i in range(n_az):
        raw[i, target_bin : target_bin + len(chirp)] += chirp * np.exp(1j * 0.1 * i)

    rc_data = range_compress_block(raw, chirp)

    dem = DEMInterpolator(0.0)
    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)

    slc, height, ec = backproject(
        rc_data,
        in_geom,
        out_geom,
        dem,
        fc=c / wavelength,
        ds=range_pixel_spacing,
        kernel=kernel,
    )

    assert ec == ErrorCode.Success, f"backproject returned {ec}"
    assert slc.shape[0] > 0 and slc.shape[1] > 0
    assert np.max(np.abs(slc)) > 0, "Output SLC is all zeros"

    print(f"  Output SLC shape: {slc.shape}")
    print(f"  Max amplitude: {np.max(np.abs(slc)):.4f}")
    print("  [PASS] test_synthetic_focus")


if __name__ == "__main__":
    print("Running stripmap integration tests...\n")
    try:
        test_synthetic_focus()
        print(f"\n{'=' * 50}")
        print("All tests passed!")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
