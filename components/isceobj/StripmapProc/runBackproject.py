import logging
import os
import copy
import datetime
import numpy as np
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger("isce.insar.runBackproject")


def _read_raw_data(frame):
    img = frame.getImage()
    filename = img.filename
    width = img.getWidth()
    n_lines = frame.numberOfLines
    xmin = img.getXmin() if hasattr(img, "getXmin") and img.getXmin() else 0
    xmax = img.getXmax() if hasattr(img, "getXmax") and img.getXmax() else width

    instr = frame.getInstrument()
    i_bias = instr.inPhaseValue
    q_bias = instr.quadratureValue

    raw_data = np.fromfile(filename, dtype=np.uint8).reshape(n_lines, width)

    first_sample = xmin // 2
    n_good = (xmax - xmin) // 2

    i_ch = raw_data[:, xmin::2].astype(np.float32)[:, :n_good]
    q_ch = raw_data[:, xmin + 1 :: 2].astype(np.float32)[:, :n_good]

    complex_data = (i_ch - i_bias) + 1j * (q_ch - q_bias)
    return complex_data.astype(np.complex64)


def focus_backproject(frame, outname, dem_path=None, num_threads=None):
    from isce3_backproject.adapters.isce2_adapter import (
        orbit_from_isce2,
        radargrid_from_isce2,
        lookside_from_isce2,
        frame_metadata_from_isce2,
    )
    from isce3_backproject.adapters.range_compress import (
        generate_chirp,
        range_compress_block,
    )
    from isce3_backproject.backproject import (
        LUT2d,
        RadarGeometry,
        DEMInterpolator,
        KnabKernel,
        TabulatedKernel,
        ErrorCode,
        backproject,
        set_num_threads,
        get_num_procs,
    )

    if num_threads is not None:
        set_num_threads(num_threads)

    meta = frame_metadata_from_isce2(frame)

    logger.info("Reading raw data...")
    raw_data = _read_raw_data(frame)
    n_az, n_rg = raw_data.shape
    logger.info(f"Raw data shape: {n_az} x {n_rg}")

    logger.info("Performing range compression...")
    chirp = generate_chirp(
        meta["chirp_slope"], meta["pulse_length"], meta["range_sampling_rate"]
    )
    rc_data = range_compress_block(raw_data, chirp)
    del raw_data

    orbit = meta["orbit"]
    lut = LUT2d(0.0)
    in_grid = meta["grid"]
    in_geom = RadarGeometry(in_grid, orbit, lut)
    out_geom = RadarGeometry(in_grid, orbit, lut)

    if dem_path is not None:
        from isce3_backproject.adapters.dem_adapter import dem_from_file

        dem = dem_from_file(dem_path)
    else:
        dem = DEMInterpolator(0.0)

    kernel = TabulatedKernel(KnabKernel(8.0, 0.9), 10000)
    range_pixel_spacing = SPEED_OF_LIGHT / (2.0 * meta["range_sampling_rate"])

    logger.info(f"Running backprojection ({get_num_procs()} cores available)...")
    slc_data, height_data, error_code = backproject(
        rc_data,
        in_geom,
        out_geom,
        dem,
        fc=meta["fc"],
        ds=range_pixel_spacing,
        kernel=kernel,
    )

    if error_code != ErrorCode.Success:
        raise RuntimeError(f"backproject() returned {error_code}")

    logger.info(f"Backprojection complete. Output shape: {slc_data.shape}")

    slc_data.astype(np.complex64).tofile(outname)

    width = slc_data.shape[1]
    length = slc_data.shape[0]

    slcImg = isceobj.createSlcImage()
    slcImg.setFilename(outname)
    slcImg.setWidth(width)
    slcImg.setLength(length)
    slcImg.setAccessMode("READ")
    slcImg.renderHdr()

    prf = frame.PRF
    delr = frame.instrument.getRangePixelSize()

    slcFrame = copy.deepcopy(frame)

    slcFrame.setStartingRange(frame.startingRange)
    slcFrame.setFarRange(frame.startingRange + (width - 1) * delr)

    tstart = frame.sensingStart
    tmid = tstart + datetime.timedelta(seconds=0.5 * length / prf)
    tend = tstart + datetime.timedelta(seconds=(length - 1) / prf)

    slcFrame.sensingStart = tstart
    slcFrame.sensingMid = tmid
    slcFrame.sensingStop = tend

    slcImg.setXmin(0)
    slcImg.setXmax(width)
    slcFrame.setImage(slcImg)

    slcFrame.setNumberOfSamples(width)
    slcFrame.setNumberOfLines(length)

    slcFrame._dopplerVsPixel = [0.0, 0.0, 0.0, 0.0]

    return slcFrame


def runFormSLCBackproject(self):
    dem_path = getattr(self._insar, "demFilename", None)
    num_threads = getattr(self, "backproject_threads", None)

    if self._insar.referenceRawProduct is None:
        print("Reference product was unpacked as an SLC. Skipping focusing ....")
        if self._insar.referenceSlcProduct is None:
            raise Exception("However, No reference SLC product found")

    else:
        frame = self._insar.loadProduct(self._insar.referenceRawProduct)
        outdir = os.path.join(self.reference.output + "_slc")
        outname = os.path.join(outdir, os.path.basename(self.reference.output) + ".slc")
        xmlname = outdir + ".xml"
        os.makedirs(outdir, exist_ok=True)

        slcFrame = focus_backproject(
            frame, outname, dem_path=dem_path, num_threads=num_threads
        )

        self._insar.referenceGeometrySystem = "Zero Doppler"
        self._insar.saveProduct(slcFrame, xmlname)
        self._insar.referenceSlcProduct = xmlname

        slcFrame = None
        frame = None

    if self._insar.secondaryRawProduct is None:
        print("Secondary product was unpacked as an SLC. Skipping focusing ....")
        if self._insar.secondarySlcProduct is None:
            raise Exception("However, No secondary SLC product found")

    else:
        frame = self._insar.loadProduct(self._insar.secondaryRawProduct)
        outdir = os.path.join(self.secondary.output + "_slc")
        outname = os.path.join(outdir, os.path.basename(self.secondary.output) + ".slc")
        xmlname = outdir + ".xml"
        os.makedirs(outdir, exist_ok=True)

        slcFrame = focus_backproject(
            frame, outname, dem_path=dem_path, num_threads=num_threads
        )

        self._insar.secondaryGeometrySystem = "Zero Doppler"
        self._insar.saveProduct(slcFrame, xmlname)
        self._insar.secondarySlcProduct = xmlname

        slcFrame = None
        frame = None

    return None
