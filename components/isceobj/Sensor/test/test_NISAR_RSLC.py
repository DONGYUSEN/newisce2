#!/usr/bin/env python3

"""
Unit tests for the NISAR_RSLC sensor plugin.

Tests are split into two groups:
  1. Standalone tests — exercise helper functions and HDF5 parsing logic
     using a synthetic mock HDF5 file.  These require only h5py + numpy
     and always run.
  2. Integration tests — exercise the full Sensor workflow (parse,
     extractImage, extractDoppler) using ISCE2 objects.  These are
     skipped when ISCE2 is not importable.

Run:
    python -m unittest test_NISAR_RSLC               (from this directory)
    python -m pytest   test_NISAR_RSLC.py -v          (if pytest is available)
"""

import datetime
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

_ISCE2_AVAILABLE = False
try:
    import isce  # noqa: F401

    _ISCE2_AVAILABLE = True
except ImportError:
    pass

if _ISCE2_AVAILABLE:
    from isceobj.Sensor.NISAR_RSLC import (
        NISAR_RSLC,
        _get_nisar_root_path,
        _get_product_group,
    )
else:
    import importlib.util
    import sys
    import types

    _mod_path = os.path.join(os.path.dirname(__file__), os.pardir, "NISAR_RSLC.py")

    class _SensorStubBase:
        """Stand-in base class for Sensor so NISAR_RSLC can subclass it without ISCE2."""

        parameter_list = ()

        def __init__(self, *a, **kw):
            pass

        def __init_subclass__(cls, **kw):
            pass

    class _StubParameter:
        def __init__(self, *a, **kw):
            pass

    class _StubComponent:
        Parameter = _StubParameter
        parameter_list = ()

        def __getattr__(self, name):
            if name == "Parameter":
                return _StubParameter
            return _StubFallback()

    class _StubFallback:
        def __getattr__(self, name):
            return self

        def __call__(self, *a, **kw):
            return self

    _fallback = _StubFallback()
    for mod in [
        "isce",
        "isceobj",
        "isceobj.Scene",
        "isceobj.Scene.Frame",
        "isceobj.Orbit",
        "isceobj.Orbit.Orbit",
        "isceobj.Planet",
        "isceobj.Planet.Planet",
        "isceobj.Planet.AstronomicalHandbook",
        "isceobj.Constants",
        "iscesys",
        "iscesys.Component",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = _fallback

    _sensor_mod = types.ModuleType("isceobj.Sensor.Sensor")
    _sensor_mod.Sensor = _SensorStubBase
    sys.modules["isceobj.Sensor.Sensor"] = _sensor_mod

    _sensor_pkg = types.ModuleType("isceobj.Sensor")
    _sensor_pkg.Sensor = _sensor_mod
    if "isceobj.Sensor" not in sys.modules:
        sys.modules["isceobj.Sensor"] = _sensor_pkg

    _comp_mod = types.ModuleType("iscesys.Component.Component")
    _comp_mod.Component = _StubComponent
    sys.modules["iscesys.Component.Component"] = _comp_mod

    spec = importlib.util.spec_from_file_location(
        "isceobj.Sensor.NISAR_RSLC", _mod_path
    )
    _nisar_mod = importlib.util.module_from_spec(spec)
    _nisar_mod.__package__ = "isceobj.Sensor"
    try:
        spec.loader.exec_module(_nisar_mod)
    except Exception:
        _nisar_mod = None

    if _nisar_mod is not None:
        _get_nisar_root_path = _nisar_mod._get_nisar_root_path
        _get_product_group = _nisar_mod._get_product_group
        NISAR_RSLC = getattr(_nisar_mod, "NISAR_RSLC", None)
    else:
        _get_nisar_root_path = None
        _get_product_group = None
        NISAR_RSLC = None


_EPOCH = datetime.datetime(2025, 6, 15, 12, 0, 0)
_EPOCH_STR = "seconds since 2025-06-15 12:00:00"

_N_LINES = 64
_N_SAMPLES = 128
_N_ORBIT_PTS = 10


def _create_mock_nisar_h5(
    path,
    band="LSAR",
    product="RSLC",
    frequency="A",
    polarization="HH",
    use_complex32=False,
):
    """Create a minimal but structurally correct NISAR RSLC HDF5 file.

    Parameters
    ----------
    path : str
        File path for the HDF5 file.
    band : str
        Sensor band: 'LSAR' or 'SSAR'.
    product : str
        Product group: 'RSLC' or 'SLC'.
    frequency : str
        Frequency letter: 'A' or 'B'.
    polarization : str
        Polarization code: 'HH', 'HV', etc.
    use_complex32 : bool
        If True, store SLC as structured float16 pairs ('r', 'i').
        Otherwise store as complex64.
    """
    with h5py.File(path, "w") as fp:
        root = f"/science/{band}"

        ident = fp.create_group(f"{root}/identification")
        ident.create_dataset("missionId", data=np.bytes_("NISAR"))
        ident.create_dataset("productType", data=np.bytes_(product))
        ident.create_dataset("lookDirection", data=np.bytes_("right"))
        ident.create_dataset("orbitPassDirection", data=np.bytes_("Ascending"))
        ident.create_dataset("absoluteOrbitNumber", data=np.int32(12345))

        prod_path = f"{root}/{product}"
        swath_path = f"{prod_path}/swaths"
        freq_key = f"frequency{frequency}"
        freq_path = f"{swath_path}/{freq_key}"

        zd_times = np.linspace(0.0, (_N_LINES - 1) * 0.002, _N_LINES)
        zd_ds = fp.create_dataset(f"{swath_path}/zeroDopplerTime", data=zd_times)
        zd_ds.attrs["units"] = np.bytes_(_EPOCH_STR)

        fp.create_dataset(
            f"{swath_path}/zeroDopplerTimeSpacing", data=np.float64(0.002)
        )

        r0 = 800000.0
        dr = 7.5
        slant_ranges = r0 + np.arange(_N_SAMPLES) * dr
        fp.create_dataset(f"{freq_path}/slantRange", data=slant_ranges)
        fp.create_dataset(f"{freq_path}/slantRangeSpacing", data=np.float64(dr))

        fp.create_dataset(
            f"{freq_path}/processedCenterFrequency", data=np.float64(1.257e9)
        )
        fp.create_dataset(
            f"{freq_path}/processedRangeBandwidth", data=np.float64(40.0e6)
        )

        fp.create_dataset(
            f"{freq_path}/listOfPolarizations",
            data=np.array([np.bytes_(polarization)]),
        )

        rng = np.random.RandomState(42)
        if use_complex32:
            dt = np.dtype([("r", np.float16), ("i", np.float16)])
            raw = np.empty((_N_LINES, _N_SAMPLES), dtype=dt)
            raw["r"] = rng.randn(_N_LINES, _N_SAMPLES).astype(np.float16)
            raw["i"] = rng.randn(_N_LINES, _N_SAMPLES).astype(np.float16)
            fp.create_dataset(f"{freq_path}/{polarization}", data=raw)
        else:
            real = rng.randn(_N_LINES, _N_SAMPLES).astype(np.float32)
            imag = rng.randn(_N_LINES, _N_SAMPLES).astype(np.float32)
            slc = (real + 1j * imag).astype(np.complex64)
            fp.create_dataset(f"{freq_path}/{polarization}", data=slc)

        orbit_path = f"{prod_path}/metadata/orbit"
        orbit_t = np.linspace(-10.0, 10.0, _N_ORBIT_PTS)
        orbit_time_ds = fp.create_dataset(f"{orbit_path}/time", data=orbit_t)
        orbit_time_ds.attrs["units"] = np.bytes_(_EPOCH_STR)

        positions = np.zeros((_N_ORBIT_PTS, 3))
        velocities = np.zeros((_N_ORBIT_PTS, 3))
        for i in range(_N_ORBIT_PTS):
            positions[i] = [
                -2.4e6 + 1000.0 * orbit_t[i],
                -4.7e6,
                4.5e6 + 500.0 * orbit_t[i],
            ]
            velocities[i] = [7500.0, 0.0, 500.0]
        fp.create_dataset(f"{orbit_path}/position", data=positions)
        fp.create_dataset(f"{orbit_path}/velocity", data=velocities)

        proc_freq_path = (
            f"{prod_path}/metadata/processingInformation/parameters/{freq_key}"
        )
        proc_params_path = f"{prod_path}/metadata/processingInformation/parameters"

        n_dop_az = 8
        n_dop_rng = 32
        dop_rng = np.linspace(r0, r0 + (_N_SAMPLES - 1) * dr, n_dop_rng)
        dop_1d = np.linspace(50.0, 20.0, n_dop_rng)
        dop_2d = np.tile(dop_1d, (n_dop_az, 1))
        dop_2d += rng.randn(n_dop_az, n_dop_rng) * 0.5

        fp.create_dataset(f"{proc_freq_path}/dopplerCentroid", data=dop_2d)
        fp.create_dataset(f"{proc_params_path}/slantRange", data=dop_rng)


class TestGetNisarRootPath(unittest.TestCase):
    """Test _get_nisar_root_path helper."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipIf(_get_nisar_root_path is None, "Could not import helpers")
    def test_lsar_found(self):
        path = os.path.join(self.tmpdir, "lsar.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/LSAR")
        with h5py.File(path, "r") as fp:
            self.assertEqual(_get_nisar_root_path(fp), "/science/LSAR")

    @unittest.skipIf(_get_nisar_root_path is None, "Could not import helpers")
    def test_ssar_found(self):
        path = os.path.join(self.tmpdir, "ssar.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/SSAR")
        with h5py.File(path, "r") as fp:
            self.assertEqual(_get_nisar_root_path(fp), "/science/SSAR")

    @unittest.skipIf(_get_nisar_root_path is None, "Could not import helpers")
    def test_lsar_preferred_over_ssar(self):
        """When both bands exist, LSAR is found first (list order)."""
        path = os.path.join(self.tmpdir, "both.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/LSAR")
            fp.create_group("/science/SSAR")
        with h5py.File(path, "r") as fp:
            self.assertEqual(_get_nisar_root_path(fp), "/science/LSAR")

    @unittest.skipIf(_get_nisar_root_path is None, "Could not import helpers")
    def test_missing_band_raises(self):
        path = os.path.join(self.tmpdir, "empty.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/OTHER")
        with h5py.File(path, "r") as fp:
            with self.assertRaises(RuntimeError):
                _get_nisar_root_path(fp)


class TestGetProductGroup(unittest.TestCase):
    """Test _get_product_group helper."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipIf(_get_product_group is None, "Could not import helpers")
    def test_rslc_preferred(self):
        path = os.path.join(self.tmpdir, "rslc.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/LSAR/RSLC")
            fp.create_group("/science/LSAR/SLC")
        with h5py.File(path, "r") as fp:
            self.assertEqual(_get_product_group(fp, "/science/LSAR"), "RSLC")

    @unittest.skipIf(_get_product_group is None, "Could not import helpers")
    def test_slc_fallback(self):
        path = os.path.join(self.tmpdir, "slc.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/LSAR/SLC")
        with h5py.File(path, "r") as fp:
            self.assertEqual(_get_product_group(fp, "/science/LSAR"), "SLC")

    @unittest.skipIf(_get_product_group is None, "Could not import helpers")
    def test_missing_product_raises(self):
        path = os.path.join(self.tmpdir, "empty.h5")
        with h5py.File(path, "w") as fp:
            fp.create_group("/science/LSAR")
        with h5py.File(path, "r") as fp:
            with self.assertRaises(RuntimeError):
                _get_product_group(fp, "/science/LSAR")


class TestParseEpochFromUnits(unittest.TestCase):
    """Test _parseEpochFromUnits static method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "epoch.h5")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _get_parse_func(self):
        """Get the epoch parser regardless of how the module was imported."""
        if _ISCE2_AVAILABLE:
            return NISAR_RSLC._parseEpochFromUnits
        elif _nisar_mod is not None:
            return _nisar_mod.NISAR_RSLC._parseEpochFromUnits
        else:
            return None

    def test_standard_format(self):
        parse_fn = self._get_parse_func()
        if parse_fn is None:
            self.skipTest("Could not import NISAR_RSLC")

        with h5py.File(self.h5path, "w") as fp:
            ds = fp.create_dataset("t", data=[0.0])
            ds.attrs["units"] = b"seconds since 2025-06-15 12:00:00"
        with h5py.File(self.h5path, "r") as fp:
            result = parse_fn(fp["t"])
        self.assertEqual(result, datetime.datetime(2025, 6, 15, 12, 0, 0))

    def test_fractional_seconds(self):
        parse_fn = self._get_parse_func()
        if parse_fn is None:
            self.skipTest("Could not import NISAR_RSLC")

        with h5py.File(self.h5path, "w") as fp:
            ds = fp.create_dataset("t", data=[0.0])
            ds.attrs["units"] = b"seconds since 2025-06-15 12:00:00.500000"
        with h5py.File(self.h5path, "r") as fp:
            result = parse_fn(fp["t"])
        self.assertEqual(result, datetime.datetime(2025, 6, 15, 12, 0, 0, 500000))

    def test_iso8601_T_separator(self):
        parse_fn = self._get_parse_func()
        if parse_fn is None:
            self.skipTest("Could not import NISAR_RSLC")

        with h5py.File(self.h5path, "w") as fp:
            ds = fp.create_dataset("t", data=[0.0])
            ds.attrs["units"] = b"seconds since 2025-06-15T12:00:00"
        with h5py.File(self.h5path, "r") as fp:
            result = parse_fn(fp["t"])
        self.assertEqual(result, datetime.datetime(2025, 6, 15, 12, 0, 0))

    def test_string_units_not_bytes(self):
        """Units stored as str rather than bytes."""
        parse_fn = self._get_parse_func()
        if parse_fn is None:
            self.skipTest("Could not import NISAR_RSLC")

        with h5py.File(self.h5path, "w") as fp:
            ds = fp.create_dataset("t", data=[0.0])
            ds.attrs["units"] = "seconds since 2025-06-15 12:00:00"
        with h5py.File(self.h5path, "r") as fp:
            result = parse_fn(fp["t"])
        self.assertEqual(result, datetime.datetime(2025, 6, 15, 12, 0, 0))


class TestMockHDF5Structure(unittest.TestCase):
    """Validate the mock HDF5 builder itself to ensure tests are valid."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "mock.h5")
        _create_mock_nisar_h5(self.h5path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_structure_exists(self):
        with h5py.File(self.h5path, "r") as fp:
            self.assertIn("/science/LSAR", fp)
            self.assertIn("/science/LSAR/RSLC", fp)
            self.assertIn("/science/LSAR/RSLC/swaths", fp)
            self.assertIn("/science/LSAR/RSLC/swaths/frequencyA", fp)
            self.assertIn("/science/LSAR/RSLC/swaths/frequencyA/HH", fp)
            self.assertIn("/science/LSAR/identification", fp)
            self.assertIn("/science/LSAR/RSLC/metadata/orbit", fp)

    def test_slc_shape(self):
        with h5py.File(self.h5path, "r") as fp:
            ds = fp["/science/LSAR/RSLC/swaths/frequencyA/HH"]
            self.assertEqual(ds.shape, (_N_LINES, _N_SAMPLES))

    def test_orbit_shape(self):
        with h5py.File(self.h5path, "r") as fp:
            pos = fp["/science/LSAR/RSLC/metadata/orbit/position"]
            vel = fp["/science/LSAR/RSLC/metadata/orbit/velocity"]
            self.assertEqual(pos.shape, (_N_ORBIT_PTS, 3))
            self.assertEqual(vel.shape, (_N_ORBIT_PTS, 3))

    def test_doppler_2d_shape(self):
        with h5py.File(self.h5path, "r") as fp:
            dop = fp[
                "/science/LSAR/RSLC/metadata/processingInformation"
                "/parameters/frequencyA/dopplerCentroid"
            ]
            self.assertEqual(dop.ndim, 2)

    def test_complex32_variant(self):
        path2 = os.path.join(self.tmpdir, "c32.h5")
        _create_mock_nisar_h5(path2, use_complex32=True)
        with h5py.File(path2, "r") as fp:
            ds = fp["/science/LSAR/RSLC/swaths/frequencyA/HH"]
            self.assertIn("r", ds.dtype.names)
            self.assertIn("i", ds.dtype.names)

    def test_ssar_band(self):
        path2 = os.path.join(self.tmpdir, "ssar.h5")
        _create_mock_nisar_h5(path2, band="SSAR")
        with h5py.File(path2, "r") as fp:
            self.assertIn("/science/SSAR", fp)
            self.assertNotIn("/science/LSAR", fp)


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_Parse(unittest.TestCase):
    """Integration tests: parse() and metadata population."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "nisar.h5")
        _create_mock_nisar_h5(self.h5path)

        self.sensor = NISAR_RSLC()
        self.sensor.configure()
        self.sensor.hdf5file = self.h5path
        self.sensor.frequency = "A"
        self.sensor.polarization = "HH"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_parse_succeeds(self):
        self.sensor.parse()

    def test_paths_discovered(self):
        self.sensor.parse()
        self.assertEqual(self.sensor._root_path, "/science/LSAR")
        self.assertEqual(self.sensor._product_name, "RSLC")

    def test_platform_populated(self):
        self.sensor.parse()
        platform = self.sensor.frame.getInstrument().getPlatform()
        self.assertEqual(platform.getMission(), "NISAR")
        self.assertEqual(platform.getPointingDirection(), -1)

    def test_instrument_populated(self):
        self.sensor.parse()
        instrument = self.sensor.frame.getInstrument()
        wavelength = instrument.getRadarWavelength()
        self.assertAlmostEqual(wavelength, 0.2385, places=3)

        prf = instrument.getPulseRepetitionFrequency()
        self.assertAlmostEqual(prf, 500.0, places=0)

    def test_frame_populated(self):
        self.sensor.parse()
        frame = self.sensor.frame
        self.assertEqual(frame.getNumberOfLines(), _N_LINES)
        self.assertEqual(frame.getNumberOfSamples(), _N_SAMPLES)
        self.assertAlmostEqual(frame.getStartingRange(), 800000.0, places=0)
        self.assertEqual(frame.getPolarization(), "HH")

    def test_sensing_times(self):
        self.sensor.parse()
        frame = self.sensor.frame
        t0 = frame.getSensingStart()
        t1 = frame.getSensingStop()
        self.assertIsInstance(t0, datetime.datetime)
        self.assertIsInstance(t1, datetime.datetime)
        self.assertGreater(t1, t0)

    def test_orbit_populated(self):
        self.sensor.parse()
        orbit = self.sensor.frame.getOrbit()
        sv_list = orbit.getStateVectors()
        self.assertEqual(len(sv_list), _N_ORBIT_PTS)

    def test_invalid_frequency_raises(self):
        self.sensor.frequency = "C"
        with self.assertRaises(ValueError):
            self.sensor.parse()

    def test_invalid_polarization_raises(self):
        self.sensor.polarization = "XX"
        with self.assertRaises(ValueError):
            self.sensor.parse()


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_ExtractImage(unittest.TestCase):
    """Integration tests: extractImage()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "nisar.h5")
        self.outpath = os.path.join(self.tmpdir, "output.slc")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run_extract(self, use_complex32=False):
        _create_mock_nisar_h5(self.h5path, use_complex32=use_complex32)
        sensor = NISAR_RSLC()
        sensor.configure()
        sensor.hdf5file = self.h5path
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = self.outpath
        sensor.extractImage()
        return sensor

    def test_output_file_created(self):
        self._run_extract()
        self.assertTrue(os.path.isfile(self.outpath))

    def test_output_size_correct(self):
        self._run_extract()
        expected_bytes = _N_LINES * _N_SAMPLES * 8
        actual_bytes = os.path.getsize(self.outpath)
        self.assertEqual(actual_bytes, expected_bytes)

    def test_output_readable_as_complex64(self):
        self._run_extract()
        data = np.fromfile(self.outpath, dtype=np.complex64).reshape(
            _N_LINES, _N_SAMPLES
        )
        self.assertEqual(data.shape, (_N_LINES, _N_SAMPLES))
        self.assertGreater(np.abs(data).max(), 0.0)

    def test_complex32_conversion(self):
        """complex32 (float16 pair) input should produce valid complex64 output."""
        self._run_extract(use_complex32=True)
        expected_bytes = _N_LINES * _N_SAMPLES * 8
        actual_bytes = os.path.getsize(self.outpath)
        self.assertEqual(actual_bytes, expected_bytes)

        data = np.fromfile(self.outpath, dtype=np.complex64).reshape(
            _N_LINES, _N_SAMPLES
        )
        self.assertGreater(np.abs(data).max(), 0.0)

    def test_frame_image_set(self):
        sensor = self._run_extract()
        img = sensor.frame.getImage()
        self.assertIsNotNone(img)
        self.assertEqual(img.getWidth(), _N_SAMPLES)


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_ExtractDoppler(unittest.TestCase):
    """Integration tests: extractDoppler()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "nisar.h5")
        _create_mock_nisar_h5(self.h5path)

        self.sensor = NISAR_RSLC()
        self.sensor.configure()
        self.sensor.hdf5file = self.h5path
        self.sensor.frequency = "A"
        self.sensor.polarization = "HH"
        self.sensor.output = os.path.join(self.tmpdir, "output.slc")
        self.sensor.extractImage()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_returns_quadratic_dict(self):
        quad = self.sensor.extractDoppler()
        self.assertIn("a", quad)
        self.assertIn("b", quad)
        self.assertIn("c", quad)

    def test_doppler_vs_pixel_set(self):
        self.sensor.extractDoppler()
        dvp = self.sensor.frame._dopplerVsPixel
        self.assertIsInstance(dvp, list)
        self.assertGreater(len(dvp), 0)

    def test_doppler_values_reasonable(self):
        """Doppler should be in the same ballpark as input LUT (20-50 Hz)."""
        quad = self.sensor.extractDoppler()
        dvp = self.sensor.frame._dopplerVsPixel
        pix_0 = np.polyval(dvp[::-1], 0)
        pix_end = np.polyval(dvp[::-1], _N_SAMPLES - 1)
        self.assertGreater(pix_0, -10.0)
        self.assertLess(pix_0, 100.0)
        self.assertGreater(pix_end, -10.0)
        self.assertLess(pix_end, 100.0)


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_SLCFallback(unittest.TestCase):
    """Test that the sensor handles legacy 'SLC' product group name."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "slc.h5")
        _create_mock_nisar_h5(self.h5path, product="SLC")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_parse_with_slc_product(self):
        sensor = NISAR_RSLC()
        sensor.configure()
        sensor.hdf5file = self.h5path
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.parse()
        self.assertEqual(sensor._product_name, "SLC")


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_SSARBand(unittest.TestCase):
    """Test that the sensor handles S-band data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "ssar.h5")
        _create_mock_nisar_h5(self.h5path, band="SSAR")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_parse_ssar(self):
        sensor = NISAR_RSLC()
        sensor.configure()
        sensor.hdf5file = self.h5path
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.parse()
        self.assertEqual(sensor._root_path, "/science/SSAR")


@unittest.skipUnless(_ISCE2_AVAILABLE, "ISCE2 is not installed")
class TestNISAR_RSLC_Pickling(unittest.TestCase):
    """Test that the sensor survives __getstate__/__setstate__ (shelve)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5path = os.path.join(self.tmpdir, "nisar.h5")
        _create_mock_nisar_h5(self.h5path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_getstate_setstate(self):
        sensor = NISAR_RSLC()
        sensor.configure()
        sensor.hdf5file = self.h5path
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.parse()

        state = sensor.__getstate__()
        self.assertNotIn("logger", state)

        new_sensor = NISAR_RSLC.__new__(NISAR_RSLC)
        new_sensor.__setstate__(state)
        self.assertTrue(hasattr(new_sensor, "logger"))


if __name__ == "__main__":
    unittest.main()
