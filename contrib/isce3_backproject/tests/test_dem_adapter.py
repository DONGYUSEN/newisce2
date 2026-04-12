#!/usr/bin/env python3
"""
Tests for the DEM adapter module.

Validates:
  1. dem_from_array() — synthetic numpy array → DEMInterpolator
  2. dem_from_file()  — synthetic GeoTIFF → DEMInterpolator
  3. DEM callback integration with backproject()

Run:
    python tests/test_dem_adapter.py
"""

import math
import os
import sys
import tempfile

import numpy as np

THISDIR = os.path.dirname(os.path.abspath(__file__))
MODDIR = os.path.dirname(THISDIR)
CONTRIBDIR = os.path.dirname(MODDIR)
sys.path.insert(0, CONTRIBDIR)

from isce3_backproject.backproject import (
    DateTime,
    TimeDelta,
    LookSide,
    Vec3,
    StateVector,
    OrbitInterpMethod,
    Orbit,
    LUT2d,
    RadarGridParameters,
    RadarGeometry,
    DEMInterpolator,
    DryTroposphereModel,
    KnabKernel,
    TabulatedKernel,
    ErrorCode,
    backproject,
)


def _make_orbit(n=20):
    svecs = []
    for i in range(n):
        sv = StateVector()
        sv.datetime = DateTime(f"2023-06-15T10:30:{i:02d}.000000000")
        angle = 0.001 * i
        r = 7.071e6
        sv.position = Vec3(r * math.cos(angle), r * math.sin(angle), 0.0)
        sv.velocity = Vec3(
            -r * 0.001 * math.sin(angle), r * 0.001 * math.cos(angle), 0.0
        )
        svecs.append(sv)
    return Orbit(svecs)


def _make_geometries(orbit):
    dt = DateTime(2023, 6, 15, 10, 30, 0)
    lut = LUT2d(0.0)
    in_grid = RadarGridParameters(
        0.0,
        0.0556,
        1000.0,
        850000.0,
        7.5,
        LookSide.Right,
        64,
        128,
        dt,
    )
    out_grid = RadarGridParameters(
        0.01,
        0.0556,
        500.0,
        855000.0,
        15.0,
        LookSide.Right,
        8,
        16,
        dt,
    )
    in_geom = RadarGeometry(in_grid, orbit, lut)
    out_geom = RadarGeometry(out_grid, orbit, lut)
    return in_geom, out_geom


def test_dem_from_array():
    from isce3_backproject.adapters.dem_adapter import dem_from_array

    nrows, ncols = 10, 10
    lon_origin, lat_origin = 120.0, 31.0
    dx, dy = 0.1, -0.1

    # height(lon,lat) = 100 + 10*(lon-120) + 5*(31-lat)
    data = np.zeros((nrows, ncols), dtype=np.float64)
    for r in range(nrows):
        for c in range(ncols):
            lon = lon_origin + (c + 0.5) * dx
            lat = lat_origin + (r + 0.5) * dy
            data[r, c] = 100.0 + 10.0 * (lon - 120.0) + 5.0 * (31.0 - lat)

    transform = (lon_origin, dx, 0.0, lat_origin, 0.0, dy)

    dem = dem_from_array(data, transform, epsg=4326)

    assert abs(dem.refHeight() - float(np.mean(data))) < 0.01, (
        f"refHeight mismatch: {dem.refHeight()} vs {float(np.mean(data))}"
    )
    assert abs(dem.minHeight() - float(np.min(data))) < 0.01
    assert abs(dem.maxHeight() - float(np.max(data))) < 0.01
    assert abs(dem.meanHeight() - float(np.mean(data))) < 0.01
    assert dem.epsgCode() == 4326

    lon_mid = lon_origin + 0.5 * ncols * dx
    lat_mid = lat_origin + 0.5 * nrows * dy
    expected = 100.0 + 10.0 * (lon_mid - 120.0) + 5.0 * (31.0 - lat_mid)
    result = dem.interpolateLonLat(lon_mid, lat_mid)
    assert abs(result - expected) < 1.0, (
        f"interpolation at ({lon_mid}, {lat_mid}): got {result}, expected ~{expected}"
    )

    result_oob = dem.interpolateLonLat(0.0, 0.0)
    assert abs(result_oob - float(np.mean(data))) < 0.01, (
        f"out-of-bounds should return mean, got {result_oob}"
    )

    print("  [PASS] test_dem_from_array")


def test_dem_from_file():
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        print("  [SKIP] test_dem_from_file — rasterio not installed")
        return

    from isce3_backproject.adapters.dem_adapter import dem_from_file

    nrows, ncols = 20, 20
    west, east = 120.0, 121.0
    south, north = 30.0, 31.0

    # parabolic DEM: peak=500 at center (120.5, 30.5)
    lons = np.linspace(west, east, ncols, endpoint=False) + 0.5 * (east - west) / ncols
    lats = (
        np.linspace(north, south, nrows, endpoint=False) + 0.5 * (south - north) / nrows
    )
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = 500.0 - 100.0 * ((lon_grid - 120.5) ** 2 + (lat_grid - 30.5) ** 2)
    data = data.astype(np.float32)

    transform = from_bounds(west, south, east, north, ncols, nrows)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as ds:
            ds.write(data, 1)

        dem = dem_from_file(tmp_path)

        assert abs(dem.minHeight() - float(np.min(data))) < 1.0
        assert abs(dem.maxHeight() - float(np.max(data))) < 1.0
        assert abs(dem.meanHeight() - float(np.mean(data))) < 1.0
        assert dem.epsgCode() == 4326

        result = dem.interpolateLonLat(120.5, 30.5)
        assert abs(result - 500.0) < 5.0, (
            f"center interpolation: got {result}, expected ~500.0"
        )

        result_corner = dem.interpolateLonLat(120.05, 30.95)
        assert result_corner < result, (
            f"corner ({result_corner}) should be less than peak ({result})"
        )

        print("  [PASS] test_dem_from_file")
    finally:
        os.unlink(tmp_path)


def test_dem_backproject_integration():
    from isce3_backproject.adapters.dem_adapter import dem_from_array

    data = np.full((10, 10), 500.0, dtype=np.float64)
    transform = (-180.0, 36.0, 0.0, 90.0, 0.0, -18.0)

    dem = dem_from_array(data, transform, epsg=4326)

    assert abs(dem.refHeight() - 500.0) < 0.01
    assert abs(dem.interpolateLonLat(0.0, 0.0) - 500.0) < 0.01

    orbit = _make_orbit()
    in_geom, out_geom = _make_geometries(orbit)
    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)

    rng = np.random.default_rng(42)
    in_data = (
        rng.standard_normal((64, 128)) + 1j * rng.standard_normal((64, 128))
    ).astype(np.complex64)

    out, height, ec = backproject(
        in_data,
        in_geom,
        out_geom,
        dem,
        fc=5.405e9,
        ds=5.0,
        kernel=kernel,
    )

    assert ec == ErrorCode.Success, f"backproject returned {ec}"
    assert out.shape == (8, 16), f"unexpected output shape {out.shape}"

    dem_const = DEMInterpolator(500.0)
    out_const, _, ec_const = backproject(
        in_data,
        in_geom,
        out_geom,
        dem_const,
        fc=5.405e9,
        ds=5.0,
        kernel=kernel,
    )
    assert ec_const == ErrorCode.Success

    max_diff = np.max(np.abs(out - out_const))
    assert max_diff < 1e-3, (
        f"callback DEM vs constant DEM max diff = {max_diff}, expected < 1e-3"
    )

    print("  [PASS] test_dem_backproject_integration")


def test_dem_from_array_nodata():
    from isce3_backproject.adapters.dem_adapter import dem_from_array

    data = np.array(
        [
            [100.0, 200.0, -9999.0],
            [150.0, -9999.0, 250.0],
            [180.0, 220.0, 300.0],
        ]
    )
    transform = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)

    dem = dem_from_array(data, transform, nodata=-9999.0)

    valid_heights = [100, 200, 150, 250, 180, 220, 300]
    expected_mean = float(np.mean(valid_heights))
    assert abs(dem.meanHeight() - expected_mean) < 1.0, (
        f"mean {dem.meanHeight()} vs expected {expected_mean}"
    )

    # nodata cells replaced with mean, so minHeight >= min(valid_heights)
    assert dem.minHeight() >= 90.0, f"minHeight {dem.minHeight()} unexpectedly low"

    print("  [PASS] test_dem_from_array_nodata")


if __name__ == "__main__":
    print("Running DEM adapter tests...\n")
    passed = 0
    failed = 0
    skipped = 0

    for test_fn in [
        test_dem_from_array,
        test_dem_from_file,
        test_dem_backproject_integration,
        test_dem_from_array_nodata,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"  [ERROR] {test_fn.__name__}: {e}")
                import traceback

                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)
    print("All tests passed!")
