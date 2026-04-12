#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2026 California Institute of Technology. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: ISCE2 NISAR Integration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
NISAR Level-1 RSLC (HDF5) reader for ISCE2.

Reads NISAR RSLC products and populates ISCE2 Frame/Instrument/Orbit objects
for use with stripmapApp InSAR processing.

References:
  - ISCE3: nisar/products/readers/SLC/RSLC.py
  - ISCE3: nisar/products/readers/Base/Base.py
  - ISCE2: components/isceobj/Sensor/UAVSAR_HDF5_SLC.py
"""

import datetime
import logging
import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("Python module h5py is required to process NISAR RSLC data")

import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.Planet import Planet
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from isceobj.Constants import SPEED_OF_LIGHT

from .Sensor import Sensor

# ── Component Parameters ────────────────────────────────────────────────────

HDF5FILE = Component.Parameter(
    "hdf5file",
    public_name="HDF5FILE",
    default=None,
    type=str,
    mandatory=True,
    intent="input",
    doc="Path to NISAR RSLC HDF5 file",
)

FREQUENCY = Component.Parameter(
    "frequency",
    public_name="FREQUENCY",
    default="A",
    type=str,
    mandatory=False,
    intent="input",
    doc="Frequency band: A or B (default: A)",
)

POLARIZATION = Component.Parameter(
    "polarization",
    public_name="POLARIZATION",
    default="HH",
    type=str,
    mandatory=False,
    intent="input",
    doc="Polarization channel: HH, HV, VH, VV (default: HH)",
)

# ── Constants ────────────────────────────────────────────────────────────────

SCIENCE_PATH = "/science/"
NISAR_SENSOR_LIST = ["LSAR", "SSAR"]


# ── Helper: discover root path ──────────────────────────────────────────────


def _get_nisar_root_path(h5file):
    """Discover the NISAR root path (/science/LSAR or /science/SSAR).

    Parameters
    ----------
    h5file : h5py.File
        An open HDF5 file handle.

    Returns
    -------
    str
        Root path, e.g. '/science/LSAR'.

    Raises
    ------
    RuntimeError
        If neither LSAR nor SSAR group is found.
    """
    for band in NISAR_SENSOR_LIST:
        path = SCIENCE_PATH + band
        if path in h5file:
            return path
    raise RuntimeError(
        "HDF5 file does not contain NISAR frequency band group "
        "(expected /science/LSAR or /science/SSAR)"
    )


def _get_product_group(h5file, root_path):
    """Return product group name: 'RSLC' (preferred) or 'SLC' (legacy).

    Parameters
    ----------
    h5file : h5py.File
        An open HDF5 file handle.
    root_path : str
        Root path, e.g. '/science/LSAR'.

    Returns
    -------
    str
        'RSLC' or 'SLC'.

    Raises
    ------
    RuntimeError
        If neither group is found.
    """
    grp = h5file[root_path]
    if "RSLC" in grp:
        return "RSLC"
    elif "SLC" in grp:
        return "SLC"
    raise RuntimeError(
        f"HDF5 file missing 'RSLC' or 'SLC' product group under {root_path}"
    )


# ── NISAR_RSLC Sensor Class ─────────────────────────────────────────────────


class NISAR_RSLC(Sensor):
    """NISAR Level-1 RSLC (HDF5) reader for ISCE2.

    Reads NISAR RSLC products (L-band or S-band) and populates the standard
    ISCE2 Frame/Instrument/Platform/Orbit objects so the data can be processed
    with stripmapApp.

    Usage
    -----
    In a stripmapApp XML or command line::

        sensor.name = NISAR_RSLC
        sensor.hdf5file = /path/to/NISAR_L1_RSLC.h5
        sensor.frequency = A          # A or B
        sensor.polarization = HH      # HH, HV, VH, VV
    """

    family = "nisar_rslc"
    logging_name = "isce.Sensor.NISAR_RSLC"

    parameter_list = (HDF5FILE, FREQUENCY, POLARIZATION) + Sensor.parameter_list

    def __init__(self, family="", name=""):
        super(NISAR_RSLC, self).__init__(
            family if family else self.__class__.family, name=name
        )
        self.frame = Frame()
        self.frame.configure()

        # Internal path cache (populated by parse())
        self._root_path = None  # e.g. '/science/LSAR'
        self._product_name = None  # 'RSLC' or 'SLC'
        self._product_path = None  # e.g. '/science/LSAR/RSLC'
        self._swath_path = None  # e.g. '/science/LSAR/RSLC/swaths'
        self._metadata_path = None  # e.g. '/science/LSAR/RSLC/metadata'
        self._proc_path = (
            None  # e.g. '/science/LSAR/RSLC/metadata/processingInformation'
        )
        self._ident_path = None  # e.g. '/science/LSAR/identification'

        self.lookMap = {"right": -1, "left": 1}

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger("isce.Sensor.NISAR_RSLC")

    def getFrame(self):
        return self.frame

    # ── parse ────────────────────────────────────────────────────────────

    def parse(self):
        """Open HDF5 file, discover paths, validate product, populate metadata."""
        try:
            fp = h5py.File(self.hdf5file, "r")
        except Exception as e:
            self.logger.error("Cannot open HDF5 file: %s" % e)
            raise

        # Discover root path and product group
        self._root_path = _get_nisar_root_path(fp)
        self._product_name = _get_product_group(fp, self._root_path)
        self._product_path = self._root_path + "/" + self._product_name
        self._swath_path = self._product_path + "/swaths"
        self._metadata_path = self._product_path + "/metadata"
        self._proc_path = self._metadata_path + "/processingInformation"
        self._ident_path = self._root_path + "/identification"

        # Validate product type
        self._validateProduct(fp)

        # Validate requested frequency and polarization
        self._validateFreqPol(fp)

        # Populate metadata
        self.populateMetadata(fp)
        fp.close()

    def _validateProduct(self, fp):
        """Check that the HDF5 file is a valid NISAR RSLC product."""
        ident = fp[self._ident_path]
        product_type = (
            ident["productType"][()].decode("utf-8")
            if isinstance(ident["productType"][()], bytes)
            else str(ident["productType"][()])
        )

        if product_type not in ("RSLC", "SLC"):
            self.logger.warning(
                "Product type is '%s', expected 'RSLC' or 'SLC'. "
                "Proceeding anyway." % product_type
            )

    def _validateFreqPol(self, fp):
        """Validate that requested frequency band and polarization exist."""
        freq_key = "frequency" + self.frequency
        freq_path = self._swath_path + "/" + freq_key

        if freq_path not in fp:
            available = [
                k for k in fp[self._swath_path].keys() if k.startswith("frequency")
            ]
            raise ValueError(
                "Frequency '%s' not found. Available: %s" % (self.frequency, available)
            )

        # Check polarization
        pol_path = freq_path + "/" + self.polarization
        if pol_path not in fp:
            # Try to discover available polarizations
            freq_grp = fp[freq_path]
            if "listOfPolarizations" in freq_grp:
                raw = freq_grp["listOfPolarizations"][:]
                avail = [
                    p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in raw
                ]
            else:
                avail = [
                    k
                    for k in freq_grp.keys()
                    if k in ("HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV")
                ]
            raise ValueError(
                "Polarization '%s' not found for frequency%s. Available: %s"
                % (self.polarization, self.frequency, avail)
            )

    # ── populate methods ─────────────────────────────────────────────────

    def populateMetadata(self, file):
        """Populate Frame, Platform, Instrument, Orbit from HDF5."""
        self._populatePlatform(file)
        self._populateInstrument(file)
        self._populateFrame(file)
        self._populateOrbit(file)

    def _populatePlatform(self, file):
        """Set platform: mission name, look direction, planet."""
        platform = self.frame.getInstrument().getPlatform()
        ident = file[self._ident_path]

        mission = (
            ident["missionId"][()].decode("utf-8")
            if isinstance(ident["missionId"][()], bytes)
            else str(ident["missionId"][()])
        )
        platform.setMission(mission)

        look_dir = (
            ident["lookDirection"][()].decode("utf-8")
            if isinstance(ident["lookDirection"][()], bytes)
            else str(ident["lookDirection"][()])
        )
        platform.setPointingDirection(self.lookMap[look_dir.lower()])
        platform.setPlanet(Planet(pname="Earth"))
        platform.setAntennaLength(12.0)

    def _populateInstrument(self, file):
        """Set instrument: wavelength, PRF, bandwidth, range sampling rate."""
        instrument = self.frame.getInstrument()
        freq_key = "frequency" + self.frequency
        swath_freq_path = self._swath_path + "/" + freq_key

        # Center frequency → wavelength
        center_freq = file[swath_freq_path + "/processedCenterFrequency"][()]
        wavelength = SPEED_OF_LIGHT / center_freq
        instrument.setRadarWavelength(wavelength)

        # PRF from zeroDopplerTimeSpacing
        zd_time_spacing = file[self._swath_path + "/zeroDopplerTimeSpacing"][()]
        prf = 1.0 / zd_time_spacing
        instrument.setPulseRepetitionFrequency(prf)

        # Range pixel size from slantRangeSpacing
        range_pixel_size = file[swath_freq_path + "/slantRangeSpacing"][()]
        instrument.setRangePixelSize(range_pixel_size)

        # Range sampling rate
        range_sampling_rate = SPEED_OF_LIGHT / (2.0 * range_pixel_size)
        instrument.setRangeSamplingRate(range_sampling_rate)

        # Processed range bandwidth → chirp parameters
        range_bandwidth = file[swath_freq_path + "/processedRangeBandwidth"][()]
        # Chirp slope set to 1.0 (same convention as UAVSAR_HDF5_SLC)
        # Only used in split-spectrum to compute bandwidth
        chirp_slope = 1.0
        chirp_length = range_bandwidth / chirp_slope
        instrument.setPulseLength(chirp_length)
        instrument.setChirpSlope(chirp_slope)

        instrument.setIncidenceAngle(0.0)

    def _populateFrame(self, file):
        """Set frame: sensing times, range geometry, image dimensions."""
        freq_key = "frequency" + self.frequency
        swath_freq_path = self._swath_path + "/" + freq_key

        # Starting slant range
        slant_range = file[swath_freq_path + "/slantRange"][0]
        self.frame.setStartingRange(slant_range)

        # Zero-Doppler time reference epoch + relative times
        zd_time_ds = file[self._swath_path + "/zeroDopplerTime"]
        reference_utc = self._parseEpochFromUnits(zd_time_ds)

        rel_start = float(zd_time_ds[0])
        rel_end = float(zd_time_ds[-1])
        rel_mid = 0.5 * (rel_start + rel_end)

        sensing_start = reference_utc + datetime.timedelta(seconds=rel_start)
        sensing_stop = reference_utc + datetime.timedelta(seconds=rel_end)
        sensing_mid = reference_utc + datetime.timedelta(seconds=rel_mid)

        self.frame.setSensingStart(sensing_start)
        self.frame.setSensingMid(sensing_mid)
        self.frame.setSensingStop(sensing_stop)

        # Pass direction
        ident = file[self._ident_path]
        pass_dir = (
            ident["orbitPassDirection"][()].decode("utf-8")
            if isinstance(ident["orbitPassDirection"][()], bytes)
            else str(ident["orbitPassDirection"][()])
        )
        self.frame.setPassDirection(pass_dir)

        # Orbit number (use absoluteOrbitNumber if available)
        if "absoluteOrbitNumber" in ident:
            orbit_num = int(ident["absoluteOrbitNumber"][()])
            self.frame.setOrbitNumber(orbit_num)

        self.frame.setProcessingFacility("JPL")
        self.frame.setPolarization(self.polarization)

        # Image dimensions from SLC dataset shape
        slc_ds = file[swath_freq_path + "/" + self.polarization]
        n_lines, n_samples = slc_ds.shape
        self.frame.setNumberOfLines(n_lines)
        self.frame.setNumberOfSamples(n_samples)

        # Far range
        range_pixel_size = self.frame.instrument.rangePixelSize
        far_range = slant_range + (n_samples - 1) * range_pixel_size
        self.frame.setFarRange(far_range)

    def _populateOrbit(self, file):
        """Extract orbit state vectors from metadata/orbit group."""
        orbit = self.frame.getOrbit()
        orbit.setReferenceFrame("ECR")
        orbit.setOrbitSource("Header")

        orbit_path = self._metadata_path + "/orbit"

        # Time reference epoch from zeroDopplerTime (orbit times share the epoch)
        orbit_time_ds = file[orbit_path + "/time"]
        reference_utc = self._parseEpochFromUnits(orbit_time_ds)

        t = orbit_time_ds[:]
        position = file[orbit_path + "/position"][:]
        velocity = file[orbit_path + "/velocity"][:]

        for i in range(len(t)):
            vec = StateVector()
            dt = reference_utc + datetime.timedelta(seconds=float(t[i]))
            vec.setTime(dt)
            vec.setPosition(
                [float(position[i, 0]), float(position[i, 1]), float(position[i, 2])]
            )
            vec.setVelocity(
                [float(velocity[i, 0]), float(velocity[i, 1]), float(velocity[i, 2])]
            )
            orbit.addStateVector(vec)

    # ── extractImage ─────────────────────────────────────────────────────

    def extractImage(self):
        """Read SLC data from HDF5 and write to binary complex64 file.

        Handles NISAR complex32 (float16 pairs) by converting to complex64.
        Uses chunked reading to manage memory for large scenes.
        """
        self.parse()

        fid = h5py.File(self.hdf5file, "r")
        freq_key = "frequency" + self.frequency
        ds_path = self._swath_path + "/" + freq_key + "/" + self.polarization
        ds = fid[ds_path]

        n_lines, n_samples = ds.shape
        chunk_lines = 512

        with open(self.output, "wb") as fout:
            for i0 in range(0, n_lines, chunk_lines):
                i1 = min(i0 + chunk_lines, n_lines)
                chunk = ds[i0:i1, :]

                # Handle NISAR complex32: structured dtype with 'r' and 'i'
                # float16 fields, OR standard complex types
                if chunk.dtype.names and "r" in chunk.dtype.names:
                    real = chunk["r"].astype(np.float32)
                    imag = chunk["i"].astype(np.float32)
                    slc_chunk = (real + 1j * imag).astype(np.complex64)
                elif chunk.dtype == np.complex64:
                    slc_chunk = chunk
                else:
                    slc_chunk = chunk.astype(np.complex64)

                slc_chunk.tofile(fout)

        fid.close()

        # Create ISCE2 SLC image object
        slc_image = isceobj.createSlcImage()
        slc_image.setFilename(self.output)
        slc_image.setXmin(0)
        slc_image.setXmax(n_samples)
        slc_image.setWidth(n_samples)
        slc_image.setAccessMode("r")
        slc_image.renderHdr()
        self.frame.setImage(slc_image)

    # ── extractDoppler ───────────────────────────────────────────────────

    def extractDoppler(self):
        """Extract Doppler centroid from NISAR 2D LUT and convert to 1D polynomial.

        NISAR provides a 2D Doppler centroid LUT (azimuth time x slant range).
        We reduce it to a 1D range polynomial by taking the median along azimuth,
        then fitting a high-order polynomial in pixel coordinates.

        Returns
        -------
        dict
            Quadratic Doppler coefficients {'a', 'b', 'c'} for insarApp
            compatibility. The per-pixel polynomial is stored in
            ``self.frame._dopplerVsPixel``.
        """
        fid = h5py.File(self.hdf5file, "r")
        freq_key = "frequency" + self.frequency

        # Build paths — NISAR has frequency-specific and shared coordinate vectors
        proc_freq_path = self._proc_path + "/parameters/" + freq_key
        proc_params_path = self._proc_path + "/parameters"

        # Read 2D Doppler LUT
        doppler_2d = fid[proc_freq_path + "/dopplerCentroid"][:]

        # Slant range coordinates for the Doppler LUT
        # Try frequency-level first, fall back to shared parameters level
        if (proc_freq_path + "/slantRange") in fid:
            doppler_rng = fid[proc_freq_path + "/slantRange"][:]
        else:
            doppler_rng = fid[proc_params_path + "/slantRange"][:]

        # Image slant range grid
        img_rng = fid[self._swath_path + "/" + freq_key + "/slantRange"][:]

        fid.close()

        # Reduce 2D → 1D: take median along azimuth dimension
        if doppler_2d.ndim == 2:
            doppler_1d = np.median(doppler_2d, axis=0)
        else:
            doppler_1d = doppler_2d

        # Interpolate Doppler values to image range grid
        from scipy.interpolate import UnivariateSpline

        # Clip to overlapping range
        ind0 = max(0, np.argmin(np.abs(doppler_rng - img_rng[0])) - 1)
        ind1 = min(len(doppler_rng), np.argmin(np.abs(doppler_rng - img_rng[-1])) + 2)
        dop_clipped = doppler_1d[ind0:ind1]
        rng_clipped = doppler_rng[ind0:ind1]

        n_pts = len(rng_clipped)
        if n_pts >= 4:
            f = UnivariateSpline(rng_clipped, dop_clipped, s=0, k=3)
            img_dop = f(img_rng)
        elif n_pts >= 2:
            f = UnivariateSpline(rng_clipped, dop_clipped, s=0, k=1)
            img_dop = f(img_rng)
        else:
            img_dop = np.full(len(img_rng), dop_clipped[0] if n_pts == 1 else 0.0)

        # Fit polynomial in pixel coordinates — cap order to avoid ill-conditioning
        dr = img_rng[1] - img_rng[0]
        pix = (img_rng - img_rng[0]) / dr
        fit_order = min(min(41, len(pix) - 1), max(1, n_pts - 1))
        fit = np.polyfit(pix, img_dop, fit_order)

        self.frame._dopplerVsPixel = list(fit[::-1])

        # insarApp-style quadratic (fixed Doppler at mid-scene)
        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        mid_dop = img_dop[len(img_dop) // 2]
        quadratic = {
            "a": mid_dop / prf,
            "b": 0.0,
            "c": 0.0,
        }

        return quadratic

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parseEpochFromUnits(dataset):
        """Parse reference epoch from HDF5 dataset 'units' attribute.

        NISAR zeroDopplerTime and orbit/time datasets store the epoch in their
        'units' attribute as 'seconds since YYYY-MM-DD HH:MM:SS' (with optional
        fractional seconds).

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset with a 'units' attribute containing the epoch.

        Returns
        -------
        datetime.datetime
            The reference epoch.
        """
        units = dataset.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")

        # Extract datetime string after 'seconds since '
        prefix = "seconds since "
        if units.startswith(prefix):
            dt_str = units[len(prefix) :]
        else:
            dt_str = units

        # Parse with optional fractional seconds
        fmt = "%Y-%m-%d %H:%M:%S"
        if "." in dt_str:
            fmt += ".%f"
        # Also handle 'T' separator (ISO 8601)
        if "T" in dt_str:
            fmt = fmt.replace(" ", "T")

        return datetime.datetime.strptime(dt_str.strip(), fmt)
