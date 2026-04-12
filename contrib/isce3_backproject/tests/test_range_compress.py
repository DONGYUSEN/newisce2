# tests/test_range_compress.py
"""Tests for FFT-based range compression."""

import sys, os
import numpy as np

THISDIR = os.path.dirname(os.path.abspath(__file__))
MODDIR = os.path.dirname(THISDIR)
CONTRIBDIR = os.path.dirname(MODDIR)
sys.path.insert(0, CONTRIBDIR)

from isce3_backproject.adapters.range_compress import (
    generate_chirp,
    range_compress_pulse,
    range_compress_block,
    remove_iq_bias,
)


def test_chirp_generation():
    """Chirp should be unit-amplitude complex with linearly increasing phase."""
    fs = 18.962468e6
    chirp_slope = -7.2135e11
    pulse_length = 27.0e-6
    n_samples = int(np.round(pulse_length * fs))

    chirp = generate_chirp(chirp_slope, pulse_length, fs)

    assert chirp.dtype == np.complex64
    assert len(chirp) == n_samples
    assert np.allclose(np.abs(chirp), 1.0, atol=1e-5)
    print("  [PASS] test_chirp_generation")


def test_range_compress_point_target():
    """Range compression of single point target should peak at correct bin."""
    fs = 18.962468e6
    chirp_slope = -7.2135e11
    pulse_length = 27.0e-6
    n_range = 2048

    chirp = generate_chirp(chirp_slope, pulse_length, fs)

    target_bin = 1000
    echo = np.zeros(n_range, dtype=np.complex64)
    chirp_len = len(chirp)
    echo[target_bin : target_bin + chirp_len] = chirp

    compressed = range_compress_pulse(echo, chirp)

    assert compressed.dtype == np.complex64
    assert len(compressed) == n_range

    peak_bin = np.argmax(np.abs(compressed))
    assert abs(peak_bin - target_bin) <= 2, (
        f"Peak at {peak_bin}, expected near {target_bin}"
    )
    print("  [PASS] test_range_compress_point_target")


def test_range_compress_block():
    """Block processing: range-compress multiple azimuth lines at once."""
    fs = 18.962468e6
    chirp_slope = -7.2135e11
    pulse_length = 27.0e-6
    n_az = 64
    n_range = 2048

    chirp = generate_chirp(chirp_slope, pulse_length, fs)
    chirp_len = len(chirp)
    target_bin = 800

    raw = np.zeros((n_az, n_range), dtype=np.complex64)
    for i in range(n_az):
        raw[i, target_bin : target_bin + chirp_len] = chirp * np.exp(1j * 0.1 * i)

    compressed = range_compress_block(raw, chirp)

    assert compressed.shape == (n_az, n_range)
    assert compressed.dtype == np.complex64

    for i in range(n_az):
        peak = np.argmax(np.abs(compressed[i]))
        assert abs(peak - target_bin) <= 2
    print("  [PASS] test_range_compress_block")


def test_remove_iq_bias():
    """IQ bias removal should subtract (I_bias + j*Q_bias) from each sample."""
    raw = np.array(
        [15.5 + 1j * 15.5, 16.5 + 1j * 14.5, 15.0 + 1j * 16.0], dtype=np.complex64
    )
    corrected = remove_iq_bias(raw, 15.5, 15.5)
    expected = np.array([0.0 + 0j, 1.0 - 1j, -0.5 + 0.5j], dtype=np.complex64)
    assert np.allclose(corrected, expected, atol=1e-5)
    print("  [PASS] test_remove_iq_bias")


if __name__ == "__main__":
    print("Running range compression tests...\n")
    passed = 0
    failed = 0
    for fn in [
        test_chirp_generation,
        test_range_compress_point_target,
        test_range_compress_block,
        test_remove_iq_bias,
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
