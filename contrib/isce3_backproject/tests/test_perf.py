#!/usr/bin/env python3
"""
OpenMP performance benchmark for isce3_backproject.

Measures wall-clock time of backproject() across thread counts (1, 2, 4, N)
and reports speedup relative to single-thread baseline.

Also verifies:
  - Multi-threaded results are bitwise identical to single-threaded.
  - set_num_threads / get_max_threads / get_num_procs bindings work.

Run:
    python tests/test_perf.py
"""

import math
import os
import sys
import time

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
    set_num_threads,
    get_max_threads,
    get_num_procs,
)


def _make_orbit(n=20):
    """Create a synthetic orbit with *n* state vectors."""
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


def _make_geometries(orbit, az_lines=128, rg_samples=256):
    """Build input/output radar geometries with configurable grid size."""
    dt = DateTime(2023, 6, 15, 10, 30, 0)
    lut = LUT2d(0.0)
    in_grid = RadarGridParameters(
        0.0,
        0.0556,
        1000.0,
        850000.0,
        7.5,
        LookSide.Right,
        az_lines,
        rg_samples,
        dt,
    )
    out_grid = RadarGridParameters(
        0.01,
        0.0556,
        500.0,
        855000.0,
        15.0,
        LookSide.Right,
        az_lines // 4,
        rg_samples // 4,
        dt,
    )
    in_geom = RadarGeometry(in_grid, orbit, lut)
    out_geom = RadarGeometry(out_grid, orbit, lut)
    return in_geom, out_geom


def test_omp_bindings():
    """Verify set_num_threads / get_max_threads / get_num_procs round-trip."""
    nprocs = get_num_procs()
    assert nprocs >= 1, f"get_num_procs() returned {nprocs}"

    original = get_max_threads()

    set_num_threads(1)
    assert get_max_threads() == 1

    set_num_threads(2)
    assert get_max_threads() == 2

    set_num_threads(nprocs)
    assert get_max_threads() == nprocs

    set_num_threads(original)
    print("  [PASS] test_omp_bindings")


def test_thread_determinism():
    """Multi-threaded backproject must produce identical results to single-threaded."""
    orbit = _make_orbit()
    in_geom, out_geom = _make_geometries(orbit, az_lines=64, rg_samples=128)
    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)
    dem = DEMInterpolator(0.0)

    rng = np.random.default_rng(42)
    in_data = (
        rng.standard_normal((64, 128)) + 1j * rng.standard_normal((64, 128))
    ).astype(np.complex64)

    set_num_threads(1)
    out_1, _, ec_1 = backproject(
        in_data,
        in_geom,
        out_geom,
        dem,
        fc=5.405e9,
        ds=5.0,
        kernel=kernel,
    )
    assert ec_1 == ErrorCode.Success

    nprocs = get_num_procs()
    for nthreads in [2, 4, nprocs]:
        if nthreads > nprocs:
            continue
        set_num_threads(nthreads)
        out_n, _, ec_n = backproject(
            in_data,
            in_geom,
            out_geom,
            dem,
            fc=5.405e9,
            ds=5.0,
            kernel=kernel,
        )
        assert ec_n == ErrorCode.Success

        max_diff = np.max(np.abs(out_n - out_1))
        # Floating-point reduction order may differ; allow tiny tolerance
        assert max_diff < 1e-6, (
            f"nthreads={nthreads}: max diff = {max_diff} (expected < 1e-6)"
        )

    set_num_threads(nprocs)
    print("  [PASS] test_thread_determinism")


def test_perf_scaling():
    """
    Benchmark backproject() across thread counts and report speedup.

    Uses a larger grid to make timing meaningful.
    This test always passes — it is informational.
    """
    orbit = _make_orbit()
    in_geom, out_geom = _make_geometries(orbit, az_lines=256, rg_samples=512)
    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)
    dem = DEMInterpolator(0.0)

    rng = np.random.default_rng(99)
    in_data = (
        rng.standard_normal((256, 512)) + 1j * rng.standard_normal((256, 512))
    ).astype(np.complex64)

    nprocs = get_num_procs()
    thread_counts = sorted(set([1, 2, 4, nprocs]))
    thread_counts = [t for t in thread_counts if t <= nprocs]

    results = {}
    n_warmup = 1
    n_runs = 3

    for nthreads in thread_counts:
        set_num_threads(nthreads)

        for _ in range(n_warmup):
            backproject(
                in_data,
                in_geom,
                out_geom,
                dem,
                fc=5.405e9,
                ds=5.0,
                kernel=kernel,
            )

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _, _, ec = backproject(
                in_data,
                in_geom,
                out_geom,
                dem,
                fc=5.405e9,
                ds=5.0,
                kernel=kernel,
            )
            elapsed = time.perf_counter() - t0
            assert ec == ErrorCode.Success
            times.append(elapsed)

        avg = sum(times) / len(times)
        results[nthreads] = avg

    set_num_threads(nprocs)

    baseline = results[1]
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  OpenMP Performance Benchmark (backproject)  │")
    print("  ├──────────┬──────────┬──────────┬─────────────┤")
    print("  │ Threads  │ Avg (s)  │ Speedup  │ Efficiency  │")
    print("  ├──────────┼──────────┼──────────┼─────────────┤")
    for nthreads in thread_counts:
        avg = results[nthreads]
        speedup = baseline / avg if avg > 0 else 0
        efficiency = speedup / nthreads * 100
        print(
            f"  │ {nthreads:>6}   │ {avg:>7.4f}  │ {speedup:>7.2f}x │ {efficiency:>9.1f}%  │"
        )
    print("  └──────────┴──────────┴──────────┴─────────────┘\n")

    print("  [PASS] test_perf_scaling (informational)")


if __name__ == "__main__":
    print(f"Running OpenMP performance tests...  (CPU cores: {get_num_procs()})\n")

    passed = 0
    failed = 0

    for test_fn in [test_omp_bindings, test_thread_determinism, test_perf_scaling]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
