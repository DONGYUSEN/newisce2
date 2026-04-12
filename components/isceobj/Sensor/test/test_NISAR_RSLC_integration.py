#!/usr/bin/env python3
"""
Integration tests for NISAR_RSLC sensor against real NISAR-format HDF5 files.

Uses ISCE3 test data (SanAnd_129.h5, SanAnd_138.h5, REE_RSLC_out17.h5) which
follow the NISAR RSLC/SLC HDF5 format convention.

These tests mock the ISCE2 framework so they can run standalone without building
ISCE2. They exercise all the core I/O and computation paths in NISAR_RSLC.py.
"""

import datetime
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

# ── Bootstrap: allow importing NISAR_RSLC without a full ISCE2 install ──────

_ISCE3_TEST_DATA = os.environ.get(
    "ISCE3_TEST_DATA",
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "..",
            "isce3",
            "tests",
            "data",
        )
    ),
)

SANAND_129 = os.path.join(_ISCE3_TEST_DATA, "SanAnd_129.h5")
SANAND_138 = os.path.join(_ISCE3_TEST_DATA, "SanAnd_138.h5")
REE_RSLC = os.path.join(_ISCE3_TEST_DATA, "REE_RSLC_out17.h5")


# ── Stub ISCE2 modules if not available ──────────────────────────────────────


class _SensorStubBase:
    """Minimal stand-in for ISCE2 Sensor base class."""

    family = "sensor"
    parameter_list = ()

    def __init__(self, family="", name=""):
        self.family = family or self.__class__.family

    def getPort(self, *a, **kw):
        return None

    def _facilities(self):
        pass


_need_stub = False
try:
    import isce
except ImportError:
    _need_stub = True

if _need_stub:
    # Create stub modules so NISAR_RSLC.py can be imported
    from types import ModuleType

    def _make_module(name, attrs=None):
        m = ModuleType(name)
        if attrs:
            m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class FakeComponent:
        class Parameter:
            def __init__(self, *a, **kw):
                pass

    class FakeFrame:
        def __init__(self):
            self._instrument = FakeInstrument()
            self._orbit = FakeOrbit()
            self._dopplerVsPixel = None

        @property
        def instrument(self):
            return self._instrument

        def configure(self):
            pass

        def getInstrument(self):
            return self._instrument

        def getOrbit(self):
            return self._orbit

        def setImage(self, img):
            self._image = img

        def setSensingStart(self, v):
            self.sensingStart = v

        def setSensingMid(self, v):
            self.sensingMid = v

        def setSensingStop(self, v):
            self.sensingStop = v

        def setStartingRange(self, v):
            self.startingRange = v

        def setFarRange(self, v):
            self.farRange = v

        def setNumberOfLines(self, v):
            self.numberOfLines = v

        def setNumberOfSamples(self, v):
            self.numberOfSamples = v

        def setPassDirection(self, v):
            self.passDirection = v

        def setOrbitNumber(self, v):
            self.orbitNumber = v

        def setProcessingFacility(self, v):
            self.processingFacility = v

        def setPolarization(self, v):
            self.polarization = v

    class FakePlatform:
        def setMission(self, v):
            self.mission = v

        def setPointingDirection(self, v):
            self.pointingDirection = v

        def setPlanet(self, v):
            self.planet = v

        def setAntennaLength(self, v):
            self.antennaLength = v

    class FakeInstrument:
        def __init__(self):
            self._platform = FakePlatform()
            self.rangePixelSize = None

        def getPlatform(self):
            return self._platform

        def setRadarWavelength(self, v):
            self.radarWavelength = v

        def setPulseRepetitionFrequency(self, v):
            self.pulseRepetitionFrequency = v

        def setRangePixelSize(self, v):
            self.rangePixelSize = v

        def setRangeSamplingRate(self, v):
            self.rangeSamplingRate = v

        def setPulseLength(self, v):
            self.pulseLength = v

        def setChirpSlope(self, v):
            self.chirpSlope = v

        def setIncidenceAngle(self, v):
            self.incidenceAngle = v

        def getPulseRepetitionFrequency(self):
            return self.pulseRepetitionFrequency

    class FakeOrbit:
        def __init__(self):
            self._vectors = []

        def setReferenceFrame(self, v):
            self.referenceFrame = v

        def setOrbitSource(self, v):
            self.orbitSource = v

        def addStateVector(self, v):
            self._vectors.append(v)

    class FakeStateVector:
        def setTime(self, v):
            self.time = v

        def setPosition(self, v):
            self.position = v

        def setVelocity(self, v):
            self.velocity = v

    class FakePlanet:
        def __init__(self, pname="Earth"):
            self.name = pname

    class FakeSlcImage:
        def setFilename(self, v):
            self.filename = v

        def setXmin(self, v):
            self.xmin = v

        def setXmax(self, v):
            self.xmax = v

        def setWidth(self, v):
            self.width = v

        def setAccessMode(self, v):
            self.accessMode = v

        def renderHdr(self):
            pass

    # Build module tree
    _make_module("isce")
    _make_module("iscesys")
    _make_module("iscesys.Component")
    _make_module("iscesys.Component.Component", {"Component": FakeComponent})
    _make_module("isceobj", {"createSlcImage": lambda: FakeSlcImage()})
    _make_module("isceobj.Scene")
    _make_module("isceobj.Scene.Frame", {"Frame": FakeFrame})
    _make_module("isceobj.Orbit")
    _make_module("isceobj.Orbit.Orbit", {"StateVector": FakeStateVector})
    _make_module("isceobj.Planet")
    _make_module("isceobj.Planet.Planet", {"Planet": FakePlanet})
    _make_module(
        "isceobj.Planet.AstronomicalHandbook",
        {"Const": type("Const", (), {"c": 299792458.0})},
    )
    _make_module("isceobj.Constants", {"SPEED_OF_LIGHT": 299792458.0})
    _make_module("isceobj.Sensor")

    sensor_mod = _make_module("isceobj.Sensor.Sensor", {"Sensor": _SensorStubBase})

    sensor_pkg = sys.modules["isceobj.Sensor"]
    sensor_pkg.__path__ = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ]
    sensor_pkg.__package__ = "isceobj.Sensor"

# Import the module under test via the package path so relative imports work
import importlib

_nisar_mod = importlib.import_module("isceobj.Sensor.NISAR_RSLC")
NISAR_RSLC = _nisar_mod.NISAR_RSLC
_get_nisar_root_path = _nisar_mod._get_nisar_root_path
_get_product_group = _nisar_mod._get_product_group


# ── Test helpers ─────────────────────────────────────────────────────────────


def _skip_unless_file(path):
    """Decorator: skip test if HDF5 file doesn't exist."""
    return unittest.skipUnless(os.path.isfile(path), f"Test data not found: {path}")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRootPathDiscovery(unittest.TestCase):
    """Test _get_nisar_root_path against real files."""

    @_skip_unless_file(SANAND_129)
    def test_sanand_root_is_LSAR(self):
        with h5py.File(SANAND_129, "r") as f:
            self.assertEqual(_get_nisar_root_path(f), "/science/LSAR")

    @_skip_unless_file(REE_RSLC)
    def test_ree_rslc_root_is_LSAR(self):
        with h5py.File(REE_RSLC, "r") as f:
            self.assertEqual(_get_nisar_root_path(f), "/science/LSAR")


class TestProductGroupDiscovery(unittest.TestCase):
    """Test _get_product_group against real files."""

    @_skip_unless_file(SANAND_129)
    def test_sanand_product_group(self):
        with h5py.File(SANAND_129, "r") as f:
            grp = _get_product_group(f, "/science/LSAR")
            self.assertIn(grp, ("RSLC", "SLC"))

    @_skip_unless_file(REE_RSLC)
    def test_ree_product_group(self):
        with h5py.File(REE_RSLC, "r") as f:
            grp = _get_product_group(f, "/science/LSAR")
            self.assertIn(grp, ("RSLC", "SLC"))


class TestParseEpochFromUnits(unittest.TestCase):
    """Test _parseEpochFromUnits against real HDF5 datasets."""

    @_skip_unless_file(SANAND_129)
    def test_epoch_from_zeroDopplerTime(self):
        with h5py.File(SANAND_129, "r") as f:
            # We need to find the zeroDopplerTime dataset
            grp = _get_product_group(f, "/science/LSAR")
            ds = f[f"/science/LSAR/{grp}/swaths/zeroDopplerTime"]
            epoch = NISAR_RSLC._parseEpochFromUnits(ds)
            self.assertIsInstance(epoch, datetime.datetime)
            # units: 'seconds since 2018-10-09 22:42:03'
            self.assertEqual(epoch.year, 2018)
            self.assertEqual(epoch.month, 10)
            self.assertEqual(epoch.day, 9)

    @_skip_unless_file(SANAND_129)
    def test_epoch_from_orbit_time(self):
        with h5py.File(SANAND_129, "r") as f:
            grp = _get_product_group(f, "/science/LSAR")
            ds = f[f"/science/LSAR/{grp}/metadata/orbit/time"]
            epoch = NISAR_RSLC._parseEpochFromUnits(ds)
            self.assertIsInstance(epoch, datetime.datetime)
            self.assertEqual(epoch.year, 2018)


class TestParseSanAnd129(unittest.TestCase):
    """Full parse() test against SanAnd_129.h5."""

    @_skip_unless_file(SANAND_129)
    def test_parse_freqA_HH(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        frame = sensor.getFrame()

        # ── Platform ──
        platform = frame.getInstrument().getPlatform()
        self.assertEqual(platform.mission, "UAVSAR")
        self.assertEqual(platform.pointingDirection, 1)  # left → 1

        # ── Instrument ──
        inst = frame.getInstrument()
        # L-band center freq ~1.243 GHz → wavelength ~0.241m
        self.assertAlmostEqual(
            inst.radarWavelength, 299792458.0 / 1243000000.0, places=6
        )
        self.assertGreater(inst.pulseRepetitionFrequency, 0)
        self.assertGreater(inst.rangePixelSize, 0)
        self.assertGreater(inst.rangeSamplingRate, 0)

        # ── Frame ──
        self.assertEqual(frame.numberOfLines, 150)
        self.assertEqual(frame.numberOfSamples, 200)
        self.assertIsInstance(frame.sensingStart, datetime.datetime)
        self.assertIsInstance(frame.sensingStop, datetime.datetime)
        self.assertGreater(frame.sensingStop, frame.sensingStart)
        self.assertGreater(frame.startingRange, 0)
        self.assertGreater(frame.farRange, frame.startingRange)
        self.assertEqual(frame.orbitNumber, 18076)

        # ── Orbit ──
        orbit = frame.getOrbit()
        self.assertGreater(len(orbit._vectors), 0)
        self.assertEqual(len(orbit._vectors), 100)  # 100 state vectors per file

    @_skip_unless_file(SANAND_129)
    def test_parse_freqB(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "B"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        frame = sensor.getFrame()

        self.assertEqual(frame.numberOfLines, 150)
        self.assertEqual(frame.numberOfSamples, 50)  # freq B is smaller

    @_skip_unless_file(SANAND_129)
    def test_invalid_frequency_raises(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "C"  # doesn't exist
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        with self.assertRaises(ValueError):
            sensor.parse()


class TestParseSanAnd138(unittest.TestCase):
    """Parse SanAnd_138.h5 — verify it's a valid pair with SanAnd_129."""

    @_skip_unless_file(SANAND_138)
    @_skip_unless_file(SANAND_129)
    def test_pair_consistency(self):
        """Both files should parse and share compatible orbit/geometry."""
        s1 = NISAR_RSLC()
        s1.hdf5file = SANAND_129
        s1.frequency = "A"
        s1.polarization = "HH"
        s1.output = "/dev/null"
        s1.parse()

        s2 = NISAR_RSLC()
        s2.hdf5file = SANAND_138
        s2.frequency = "A"
        s2.polarization = "HH"
        s2.output = "/dev/null"
        s2.parse()

        f1 = s1.getFrame()
        f2 = s2.getFrame()

        # Same track
        self.assertEqual(f1.orbitNumber, f2.orbitNumber)

        # Same look direction
        i1 = f1.getInstrument().getPlatform()
        i2 = f2.getInstrument().getPlatform()
        self.assertEqual(i1.pointingDirection, i2.pointingDirection)

        # Same line count
        self.assertEqual(f1.numberOfLines, f2.numberOfLines)


class TestParseREE_RSLC(unittest.TestCase):
    """Parse REE_RSLC_out17.h5 — SLC path, complex32 dtype."""

    @_skip_unless_file(REE_RSLC)
    def test_parse_ree_rslc(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = REE_RSLC
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        frame = sensor.getFrame()

        self.assertEqual(frame.numberOfLines, 129)
        self.assertEqual(frame.numberOfSamples, 129)
        self.assertIsInstance(frame.sensingStart, datetime.datetime)

        # Orbit
        orbit = frame.getOrbit()
        self.assertEqual(len(orbit._vectors), 28)  # REE has 28 state vectors


class TestExtractImage(unittest.TestCase):
    """Test extractImage() writes correct binary data."""

    @_skip_unless_file(SANAND_129)
    def test_extract_complex64(self):
        """SanAnd_129 has complex64 data — verify round-trip."""
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"

        with tempfile.NamedTemporaryFile(suffix=".slc", delete=False) as tmp:
            sensor.output = tmp.name

        try:
            sensor.extractImage()

            # Verify binary output
            data = np.fromfile(sensor.output, dtype=np.complex64)
            self.assertEqual(data.shape[0], 150 * 200)  # n_lines * n_samples
            data_2d = data.reshape(150, 200)

            # Compare with original HDF5 data
            with h5py.File(SANAND_129, "r") as f:
                grp = _get_product_group(f, "/science/LSAR")
                orig = f[f"/science/LSAR/{grp}/swaths/frequencyA/HH"][:]
                np.testing.assert_array_almost_equal(data_2d, orig, decimal=5)
        finally:
            os.unlink(sensor.output)
            xml = sensor.output + ".xml"
            if os.path.exists(xml):
                os.unlink(xml)
            vrt = sensor.output + ".vrt"
            if os.path.exists(vrt):
                os.unlink(vrt)

    @_skip_unless_file(REE_RSLC)
    def test_extract_complex32(self):
        """REE_RSLC has complex32 (float16 pairs) — verify conversion to complex64."""
        sensor = NISAR_RSLC()
        sensor.hdf5file = REE_RSLC
        sensor.frequency = "A"
        sensor.polarization = "HH"

        with tempfile.NamedTemporaryFile(suffix=".slc", delete=False) as tmp:
            sensor.output = tmp.name

        try:
            sensor.extractImage()

            data = np.fromfile(sensor.output, dtype=np.complex64)
            self.assertEqual(data.shape[0], 129 * 129)
            data_2d = data.reshape(129, 129)

            # Verify against original float16 data
            with h5py.File(REE_RSLC, "r") as f:
                raw = f["/science/LSAR/SLC/swaths/frequencyA/HH"][:]
                expected = raw["r"].astype(np.float32) + 1j * raw["i"].astype(
                    np.float32
                )
                np.testing.assert_array_almost_equal(data_2d, expected, decimal=3)
        finally:
            os.unlink(sensor.output)
            xml = sensor.output + ".xml"
            if os.path.exists(xml):
                os.unlink(xml)
            vrt = sensor.output + ".vrt"
            if os.path.exists(vrt):
                os.unlink(vrt)

    @_skip_unless_file(SANAND_129)
    def test_extract_freqB(self):
        """Test frequency B extraction."""
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "B"
        sensor.polarization = "HH"

        with tempfile.NamedTemporaryFile(suffix=".slc", delete=False) as tmp:
            sensor.output = tmp.name

        try:
            sensor.extractImage()

            data = np.fromfile(sensor.output, dtype=np.complex64)
            self.assertEqual(data.shape[0], 150 * 50)  # freq B: 150x50
        finally:
            os.unlink(sensor.output)
            for ext in (".xml", ".vrt"):
                p = sensor.output + ext
                if os.path.exists(p):
                    os.unlink(p)


class TestExtractDoppler(unittest.TestCase):
    """Test extractDoppler() against real 2D LUT data."""

    @_skip_unless_file(SANAND_129)
    def test_doppler_extraction(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        result = sensor.extractDoppler()

        # Check quadratic dict structure
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertEqual(result["b"], 0.0)
        self.assertEqual(result["c"], 0.0)
        self.assertIsInstance(result["a"], float)

        # Check dopplerVsPixel was set
        dvp = sensor.frame._dopplerVsPixel
        self.assertIsNotNone(dvp)
        self.assertIsInstance(dvp, list)
        self.assertGreater(len(dvp), 1)

    @_skip_unless_file(SANAND_129)
    def test_doppler_freqB(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "B"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        result = sensor.extractDoppler()

        self.assertIn("a", result)
        self.assertIsNotNone(sensor.frame._dopplerVsPixel)

    @_skip_unless_file(REE_RSLC)
    def test_doppler_ree_rslc(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = REE_RSLC
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"

        sensor.parse()
        result = sensor.extractDoppler()

        self.assertIn("a", result)
        self.assertIsNotNone(sensor.frame._dopplerVsPixel)


class TestSensingTimeConsistency(unittest.TestCase):
    """Verify that parsed sensing times are physically reasonable."""

    @_skip_unless_file(SANAND_129)
    def test_sensing_time_range(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"
        sensor.parse()

        frame = sensor.getFrame()
        duration = (frame.sensingStop - frame.sensingStart).total_seconds()
        self.assertGreater(duration, 0)
        # 150 lines at ~0.021s spacing ≈ ~3.15s total
        self.assertAlmostEqual(duration, 3.15, delta=0.2)

    @_skip_unless_file(SANAND_129)
    def test_orbit_covers_sensing(self):
        """Orbit time span should encompass the sensing window."""
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"
        sensor.parse()

        frame = sensor.getFrame()
        orbit = frame.getOrbit()

        orbit_start = orbit._vectors[0].time
        orbit_end = orbit._vectors[-1].time
        self.assertLessEqual(orbit_start, frame.sensingStart)
        self.assertGreaterEqual(orbit_end, frame.sensingStop)


class TestSlantRangeConsistency(unittest.TestCase):
    """Verify range geometry is self-consistent."""

    @_skip_unless_file(SANAND_129)
    def test_range_geometry(self):
        sensor = NISAR_RSLC()
        sensor.hdf5file = SANAND_129
        sensor.frequency = "A"
        sensor.polarization = "HH"
        sensor.output = "/dev/null"
        sensor.parse()

        frame = sensor.getFrame()
        inst = frame.getInstrument()

        # far_range = start_range + (n_samples - 1) * range_pixel_size
        expected_far = (
            frame.startingRange + (frame.numberOfSamples - 1) * inst.rangePixelSize
        )
        self.assertAlmostEqual(frame.farRange, expected_far, places=1)

        # Cross-check with HDF5 slantRange vector
        with h5py.File(SANAND_129, "r") as f:
            grp = _get_product_group(f, "/science/LSAR")
            sr = f[f"/science/LSAR/{grp}/swaths/frequencyA/slantRange"][:]
            self.assertAlmostEqual(frame.startingRange, sr[0], places=1)
            # Far range should be close to last slant range value
            self.assertAlmostEqual(frame.farRange, sr[-1], delta=sr[-1] * 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
