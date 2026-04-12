# tests/test_isce2_adapter.py
import datetime, math, sys, os
import numpy as np

THISDIR = os.path.dirname(os.path.abspath(__file__))
MODDIR = os.path.dirname(THISDIR)
CONTRIBDIR = os.path.dirname(MODDIR)
sys.path.insert(0, CONTRIBDIR)

from isce3_backproject.adapters.isce2_adapter import (
    orbit_from_isce2,
    radargrid_from_isce2,
    lookside_from_isce2,
)
from isce3_backproject.backproject import (
    DateTime as Isce3DateTime,
    Orbit as Isce3Orbit,
    LookSide,
    RadarGridParameters,
)


class FakeStateVector:
    """Mimics isceobj.Orbit.Orbit.StateVector."""

    def __init__(self, time, position, velocity):
        self.time = time
        self.position = position
        self.velocity = velocity

    def getTime(self):
        return self.time

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return self.velocity


class FakeOrbit:
    """Mimics isceobj.Orbit.Orbit.Orbit (iterable over state vectors)."""

    def __init__(self, svecs):
        self._stateVectors = svecs

    def __iter__(self):
        return iter(self._stateVectors)

    @property
    def stateVectors(self):
        return self._stateVectors


class FakePlatform:
    def __init__(self):
        self.pointingDirection = -1  # right-looking


class FakeInstrument:
    def __init__(self):
        self.PRF = 1000.0
        self.radarWavelength = 0.0555
        self.rangeSamplingRate = 18.962468e6
        self.chirpSlope = -7.2135e11
        self.pulseLength = 27.0e-6
        self.inPhaseValue = 15.5
        self.quadratureValue = 15.5
        self.platform = FakePlatform()

    def getRangePixelSize(self):
        SPEED_OF_LIGHT = 299792458.0
        return SPEED_OF_LIGHT / (2.0 * self.rangeSamplingRate)


class FakeFrame:
    def __init__(self, orbit, instrument):
        self.orbit = orbit
        self.instrument = instrument
        self.sensingStart = datetime.datetime(2023, 6, 15, 10, 30, 0, 0)
        self.PRF = instrument.PRF
        self.radarWavelength = instrument.radarWavelength
        self.rangeSamplingRate = instrument.rangeSamplingRate
        self.startingRange = 850000.0
        self.numberOfLines = 1024
        self.numberOfSamples = 2048
        self._dopplerVsPixel = [0.0, 0.0, 0.0, 0.0]

    def getInstrument(self):
        return self.instrument


def test_orbit_conversion():
    svecs = []
    for i in range(20):
        t = datetime.datetime(2023, 6, 15, 10, 30, i)
        angle = 0.001 * i
        r = 7.071e6
        pos = [r * math.cos(angle), r * math.sin(angle), 0.0]
        vel = [-r * 0.001 * math.sin(angle), r * 0.001 * math.cos(angle), 0.0]
        svecs.append(FakeStateVector(t, pos, vel))
    isce2_orbit = FakeOrbit(svecs)

    isce3_orbit = orbit_from_isce2(isce2_orbit)

    assert isinstance(isce3_orbit, Isce3Orbit)
    # Orbit should interpolate without error at mid-time
    print("  [PASS] test_orbit_conversion")


def test_lookside_conversion():
    assert lookside_from_isce2(-1) == LookSide.Right
    assert lookside_from_isce2(1) == LookSide.Left
    print("  [PASS] test_lookside_conversion")


def test_radargrid_conversion():
    svecs = []
    for i in range(20):
        t = datetime.datetime(2023, 6, 15, 10, 30, i)
        pos = [7.071e6, 0.0, 0.0]
        vel = [0.0, 7071.0, 0.0]
        svecs.append(FakeStateVector(t, pos, vel))
    orbit = FakeOrbit(svecs)
    instr = FakeInstrument()
    frame = FakeFrame(orbit, instr)

    grid = radargrid_from_isce2(frame)

    assert isinstance(grid, RadarGridParameters)
    length = grid.length() if callable(grid.length) else grid.length
    width = grid.width() if callable(grid.width) else grid.width
    assert length == 1024
    assert width == 2048
    print("  [PASS] test_radargrid_conversion")


if __name__ == "__main__":
    print("Running ISCE2 adapter tests...\n")
    passed = 0
    failed = 0
    for fn in [
        test_orbit_conversion,
        test_lookside_conversion,
        test_radargrid_conversion,
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
