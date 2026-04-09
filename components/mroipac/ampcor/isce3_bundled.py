#!/usr/bin/env python3

import os

import numpy as np


def _path_to_xml(path):
    if path is None:
        return None
    p = str(path)
    if p.endswith(".xml"):
        return p
    if p.endswith(".vrt"):
        return p[:-4] + ".xml"
    return p + ".xml"


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_outputs_from_arrays(
    offset_path,
    gross_path,
    snr_path,
    cov_path,
    corr_path,
    number_window_down,
    number_window_across,
    down_offsets,
    across_offsets,
    snr,
    cov1,
    cov2,
    cov3,
    gross_down,
    gross_across,
):
    nd = int(number_window_down)
    na = int(number_window_across)
    n = nd * na

    for out_path in (offset_path, gross_path, snr_path, cov_path, corr_path):
        _ensure_parent(out_path)

    offset_bip = np.empty((n, 2), dtype=np.float32)
    offset_bip[:, 0] = np.asarray(down_offsets, dtype=np.float32)
    offset_bip[:, 1] = np.asarray(across_offsets, dtype=np.float32)
    offset_bip.reshape(nd, na, 2).tofile(offset_path)

    gross_bip = np.empty((n, 2), dtype=np.float32)
    gross_bip[:, 0] = float(gross_down)
    gross_bip[:, 1] = float(gross_across)
    gross_bip.reshape(nd, na, 2).tofile(gross_path)

    np.asarray(snr, dtype=np.float32).reshape(nd, na).tofile(snr_path)

    cov_bip = np.empty((n, 3), dtype=np.float32)
    # Keep isce3 covariance band convention: [azaz, rgrg, azrg]
    cov_bip[:, 0] = np.asarray(cov2, dtype=np.float32)
    cov_bip[:, 1] = np.asarray(cov1, dtype=np.float32)
    cov_bip[:, 2] = np.asarray(cov3, dtype=np.float32)
    cov_bip.reshape(nd, na, 3).tofile(cov_path)

    # Legacy isce2 Ampcor does not expose correlation-peak rasters; write a
    # neutral placeholder to keep output contract stable.
    np.zeros((nd, na), dtype=np.float32).tofile(corr_path)


def _densify_legacy_ampcor_outputs(
    number_window_down,
    number_window_across,
    first_down,
    first_across,
    skip_down,
    skip_across,
    location_down,
    location_across,
    down_offset,
    across_offset,
    snr,
    cov1,
    cov2,
    cov3,
):
    nd = int(number_window_down)
    na = int(number_window_across)
    n = nd * na

    dense_down_off = np.zeros(n, dtype=np.float32)
    dense_across_off = np.zeros(n, dtype=np.float32)
    dense_snr = np.zeros(n, dtype=np.float32)
    dense_cov1 = np.full(n, 99.0, dtype=np.float32)
    dense_cov2 = np.full(n, 99.0, dtype=np.float32)
    dense_cov3 = np.zeros(n, dtype=np.float32)

    # Legacy ampcor returns window-center coordinates, while our target grid is
    # keyed by reference-start coordinates. Use nearest-grid indexing by skip
    # instead of exact coordinate equality so the mapping remains stable.
    skip_down_f = float(skip_down)
    skip_across_f = float(skip_across)
    for i, (dn, ac) in enumerate(zip(location_down, location_across)):
        idn = int(round((float(dn) - float(first_down)) / skip_down_f))
        iac = int(round((float(ac) - float(first_across)) / skip_across_f))
        if idn < 0 or idn >= nd or iac < 0 or iac >= na:
            continue
        j = idn * na + iac
        dense_down_off[j] = np.float32(down_offset[i])
        dense_across_off[j] = np.float32(across_offset[i])
        dense_snr[j] = np.float32(snr[i])
        dense_cov1[j] = np.float32(cov1[i])
        dense_cov2[j] = np.float32(cov2[i])
        dense_cov3[j] = np.float32(cov3[i])

    return (
        dense_down_off,
        dense_across_off,
        dense_snr,
        dense_cov1,
        dense_cov2,
        dense_cov3,
    )


class _BundledBase(object):
    """
    In-tree compatibility implementation for isce3 PyCPUAmpcor/PyCuAmpcor API.
    """

    def __init__(self):
        self.algorithm = 0
        self.deviceID = 0
        self.nStreams = 1
        self.derampMethod = 1
        self.derampAxis = 0

        self.referenceImageName = ""
        self.referenceImageHeight = 0
        self.referenceImageWidth = 0
        self.secondaryImageName = ""
        self.secondaryImageHeight = 0
        self.secondaryImageWidth = 0

        self.numberWindowDown = 0
        self.numberWindowAcross = 0

        self.windowSizeHeight = 64
        self.windowSizeWidth = 64

        self.offsetImageName = ""
        self.grossOffsetImageName = ""
        self.mergeGrossOffset = 1
        self.snrImageName = ""
        self.covImageName = ""
        self.corrImageName = ""

        self.rawDataOversamplingFactor = 1
        self.corrStatWindowSize = 21

        self.numberWindowDownInChunk = 1
        self.numberWindowAcrossInChunk = 64

        self.useMmap = 1

        self.halfSearchRangeAcross = 20
        self.halfSearchRangeDown = 20

        self.referenceStartPixelAcrossStatic = 0
        self.referenceStartPixelDownStatic = 0

        self.corrSurfaceOverSamplingMethod = 0
        self.corrSurfaceOverSamplingFactor = 16

        self.mmapSize = 8

        self.skipSampleDown = 32
        self.skipSampleAcross = 32
        self.corrSurfaceZoomInWindow = 8

        self._grossDown = 0
        self._grossAcross = 0

    def setupParams(self):
        return

    def setConstantGrossOffset(self, goDown, goAcross):
        self._grossDown = int(goDown)
        self._grossAcross = int(goAcross)
        return

    def setVaryingGrossOffset(self, vD, vA):
        if len(vD) == 0 or len(vA) == 0:
            self._grossDown = 0
            self._grossAcross = 0
        else:
            self._grossDown = int(vD[0])
            self._grossAcross = int(vA[0])
        return

    def checkPixelInImageRange(self):
        if self.numberWindowAcross <= 0 or self.numberWindowDown <= 0:
            raise ValueError("numberWindowAcross/numberWindowDown must be > 0.")
        if self.skipSampleAcross <= 0 or self.skipSampleDown <= 0:
            raise ValueError("skipSampleAcross/skipSampleDown must be > 0.")
        if self.referenceImageWidth <= 0 or self.referenceImageHeight <= 0:
            raise ValueError("reference image dimensions must be > 0.")
        if self.secondaryImageWidth <= 0 or self.secondaryImageHeight <= 0:
            raise ValueError("secondary image dimensions must be > 0.")

    def runAmpcor(self):
        raise NotImplementedError


class BundledPyCPUAmpcor(_BundledBase):
    """
    CPU implementation backed by in-tree mroipac.ampcor legacy backend.
    """

    def runAmpcor(self):
        import isceobj
        from mroipac.ampcor.Ampcor import Ampcor as LegacyAmpcor

        ref_xml = _path_to_xml(self.referenceImageName)
        sec_xml = _path_to_xml(self.secondaryImageName)
        if not ref_xml or (not os.path.exists(ref_xml)):
            raise ValueError("Reference XML not found: {0}".format(ref_xml))
        if not sec_xml or (not os.path.exists(sec_xml)):
            raise ValueError("Secondary XML not found: {0}".format(sec_xml))

        ref_img = isceobj.createSlcImage()
        ref_img.load(ref_xml)
        ref_img.setAccessMode("READ")
        ref_img.createImage()

        sec_img = isceobj.createSlcImage()
        sec_img.load(sec_xml)
        sec_img.setAccessMode("READ")
        sec_img.createImage()

        try:
            first_down = int(self.referenceStartPixelDownStatic)
            first_across = int(self.referenceStartPixelAcrossStatic)
            skip_down = int(self.skipSampleDown)
            skip_across = int(self.skipSampleAcross)
            n_down = int(self.numberWindowDown)
            n_across = int(self.numberWindowAcross)
            last_down = first_down + skip_down * (n_down - 1)
            last_across = first_across + skip_across * (n_across - 1)

            obj = LegacyAmpcor(name="bundled_cpu_ampcor")
            obj.configure()
            obj.setEngine("legacy")
            obj.setImageDataType1("complex")
            obj.setImageDataType2("complex")
            obj.setAcrossGrossOffset(int(self._grossAcross))
            obj.setDownGrossOffset(int(self._grossDown))
            obj.setWindowSizeWidth(int(self.windowSizeWidth))
            obj.setWindowSizeHeight(int(self.windowSizeHeight))
            obj.setSearchWindowSizeWidth(int(self.halfSearchRangeAcross))
            obj.setSearchWindowSizeHeight(int(self.halfSearchRangeDown))
            obj.setZoomWindowSize(int(self.corrSurfaceZoomInWindow))
            obj.setOversamplingFactor(int(self.corrSurfaceOverSamplingFactor))
            obj.setFirstSampleAcross(first_across)
            obj.setLastSampleAcross(last_across)
            obj.setNumberLocationAcross(n_across)
            obj.setFirstSampleDown(first_down)
            obj.setLastSampleDown(last_down)
            obj.setNumberLocationDown(n_down)
            obj.setFirstPRF(1.0)
            obj.setSecondPRF(1.0)
            obj.setFirstRangeSpacing(1.0)
            obj.setSecondRangeSpacing(1.0)

            obj.ampcor(ref_img, sec_img)

            location_across = np.asarray(
                getattr(obj, "locationAcross", []), dtype=np.int32
            )
            location_down = np.asarray(
                getattr(obj, "locationDown", []), dtype=np.int32
            )
            across_off = np.asarray(
                getattr(obj, "locationAcrossOffset", []), dtype=np.float32
            )
            down_off = np.asarray(
                getattr(obj, "locationDownOffset", []), dtype=np.float32
            )
            snr = np.asarray(getattr(obj, "snrRet", []), dtype=np.float32)
            cov1 = np.asarray(getattr(obj, "cov1Ret", []), dtype=np.float32)
            cov2 = np.asarray(getattr(obj, "cov2Ret", []), dtype=np.float32)
            cov3 = np.asarray(getattr(obj, "cov3Ret", []), dtype=np.float32)

            n_valid = min(
                len(location_across),
                len(location_down),
                len(across_off),
                len(down_off),
                len(snr),
                len(cov1),
                len(cov2),
                len(cov3),
            )
            location_across = location_across[:n_valid]
            location_down = location_down[:n_valid]
            across_off = across_off[:n_valid]
            down_off = down_off[:n_valid]
            snr = snr[:n_valid]
            cov1 = cov1[:n_valid]
            cov2 = cov2[:n_valid]
            cov3 = cov3[:n_valid]

            dense_down_off, dense_across_off, dense_snr, dense_cov1, dense_cov2, dense_cov3 = \
                _densify_legacy_ampcor_outputs(
                    n_down,
                    n_across,
                    first_down,
                    first_across,
                    skip_down,
                    skip_across,
                    location_down,
                    location_across,
                    down_off,
                    across_off,
                    snr,
                    cov1,
                    cov2,
                    cov3,
                )

            _write_outputs_from_arrays(
                self.offsetImageName,
                self.grossOffsetImageName,
                self.snrImageName,
                self.covImageName,
                self.corrImageName,
                self.numberWindowDown,
                self.numberWindowAcross,
                dense_down_off,
                dense_across_off,
                dense_snr,
                dense_cov1,
                dense_cov2,
                dense_cov3,
                self._grossDown,
                self._grossAcross,
            )
        finally:
            ref_img.finalizeImage()
            sec_img.finalizeImage()


class BundledPyCuAmpcor(_BundledBase):
    """
    GPU implementation backed by in-tree contrib.PyCuAmpcor.
    """

    def runAmpcor(self):
        from contrib.PyCuAmpcor import PyCuAmpcor

        obj = PyCuAmpcor.PyCuAmpcor()
        obj.algorithm = int(self.algorithm)
        obj.deviceID = int(self.deviceID)
        obj.nStreams = int(self.nStreams)
        if hasattr(obj, "derampMethod"):
            obj.derampMethod = int(self.derampMethod)
        if hasattr(obj, "derampAxis"):
            obj.derampAxis = int(self.derampAxis)

        obj.referenceImageName = str(self.referenceImageName)
        obj.referenceImageHeight = int(self.referenceImageHeight)
        obj.referenceImageWidth = int(self.referenceImageWidth)
        obj.secondaryImageName = str(self.secondaryImageName)
        obj.secondaryImageHeight = int(self.secondaryImageHeight)
        obj.secondaryImageWidth = int(self.secondaryImageWidth)

        obj.numberWindowDown = int(self.numberWindowDown)
        obj.numberWindowAcross = int(self.numberWindowAcross)
        obj.windowSizeHeight = int(self.windowSizeHeight)
        obj.windowSizeWidth = int(self.windowSizeWidth)

        obj.offsetImageName = str(self.offsetImageName)
        obj.grossOffsetImageName = str(self.grossOffsetImageName)
        obj.mergeGrossOffset = int(self.mergeGrossOffset)
        obj.snrImageName = str(self.snrImageName)
        obj.covImageName = str(self.covImageName)
        if hasattr(obj, "corrImageName"):
            obj.corrImageName = str(self.corrImageName)

        obj.rawDataOversamplingFactor = int(self.rawDataOversamplingFactor)
        obj.corrStatWindowSize = int(self.corrStatWindowSize)
        obj.numberWindowDownInChunk = int(self.numberWindowDownInChunk)
        obj.numberWindowAcrossInChunk = int(self.numberWindowAcrossInChunk)
        obj.useMmap = int(self.useMmap)
        obj.halfSearchRangeAcross = int(self.halfSearchRangeAcross)
        obj.halfSearchRangeDown = int(self.halfSearchRangeDown)
        obj.referenceStartPixelAcrossStatic = int(self.referenceStartPixelAcrossStatic)
        obj.referenceStartPixelDownStatic = int(self.referenceStartPixelDownStatic)
        obj.corrSurfaceOverSamplingMethod = int(self.corrSurfaceOverSamplingMethod)
        obj.corrSurfaceOverSamplingFactor = int(self.corrSurfaceOverSamplingFactor)
        obj.mmapSize = int(self.mmapSize)
        obj.skipSampleDown = int(self.skipSampleDown)
        obj.skipSampleAcross = int(self.skipSampleAcross)
        obj.corrSurfaceZoomInWindow = int(self.corrSurfaceZoomInWindow)

        obj.setupParams()
        # PyCuAmpcor uses (goAcross, goDown) order.
        obj.setConstantGrossOffset(int(self._grossAcross), int(self._grossDown))
        obj.checkPixelInImageRange()
        obj.runAmpcor()
