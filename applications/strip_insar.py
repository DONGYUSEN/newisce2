#!/usr/bin/env python3

import os
import sys
import time
import logging
import xml.etree.ElementTree as ET
import contextlib
import importlib
import threading
import subprocess
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import isce
import isceobj
from iscesys.Component.Application import Application
from iscesys.Component.Configurable import SELF
import isceobj.StripmapProc as StripmapProc
from isceobj.Scene.Frame import FrameMixin
logger = logging.getLogger("strip_insar")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep strip_insar progress visible, suppress noisy INFO from dependencies
    logger.setLevel(level)


SENSOR_NAME = Application.Parameter(
    "sensorName",
    public_name="sensor name",
    default=None,
    type=str,
    mandatory=False,
    doc="Sensor name for both reference and secondary",
)

REFERENCE_SENSOR_NAME = Application.Parameter(
    "referenceSensorName",
    public_name="reference sensor name",
    default=None,
    type=str,
    mandatory=True,
    doc="Reference sensor name if mixing sensors",
)

SECONDARY_SENSOR_NAME = Application.Parameter(
    "secondarySensorName",
    public_name="secondary sensor name",
    default=None,
    type=str,
    mandatory=True,
    doc="Secondary sensor name if mixing sensors",
)

CORRELATION_METHOD = Application.Parameter(
    "correlation_method",
    public_name="correlation_method",
    default="cchz_wave",
    type=str,
    mandatory=False,
    doc="Select coherence estimation method: cchz_wave, phase_gradient",
)

REFERENCE_DOPPLER_METHOD = Application.Parameter(
    "referenceDopplerMethod",
    public_name="reference doppler method",
    default=None,
    type=str,
    mandatory=False,
    doc="Doppler calculation method. Choices: 'useDOPIQ', 'useDefault'.",
)

SECONDARY_DOPPLER_METHOD = Application.Parameter(
    "secondaryDopplerMethod",
    public_name="secondary doppler method",
    default=None,
    type=str,
    mandatory=False,
    doc="Doppler calculation method. Choices: 'useDOPIQ','useDefault'.",
)

ORBIT_INTERPOLATION_METHOD = Application.Parameter(
    "orbitInterpolationMethod",
    public_name="orbit interpolation method",
    default="HERMITE",
    type=str,
    mandatory=False,
    doc="Orbit interpolation method. Choices: HERMITE, SCH, LEGENDRE.",
)

UNWRAPPER_NAME = Application.Parameter(
    "unwrapper_name",
    public_name="unwrapper name",
    default="grass",
    type=str,
    mandatory=False,
    doc="Unwrapping method to use.",
)

DO_UNWRAP = Application.Parameter(
    "do_unwrap",
    public_name="do unwrap",
    default=True,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired.",
)

SNAPHU_GMTSAR_PREPROCESS = Application.Parameter(
    "snaphuGmtsarPreprocess",
    public_name="snaphu gmtsar preprocess",
    default=True,
    type=bool,
    mandatory=False,
    doc="Enable GMTSAR-style coherence-mask preprocessing before snaphu unwrapping.",
)

SNAPHU_CORR_THRESHOLD = Application.Parameter(
    "snaphuCorrThreshold",
    public_name="snaphu coherence threshold",
    default=0.20,
    type=float,
    mandatory=False,
    doc="Coherence threshold used for snaphu preprocessing.",
)

SNAPHU_INTERP_MASKED_PHASE = Application.Parameter(
    "snaphuInterpMaskedPhase",
    public_name="snaphu interpolate masked phase",
    default=False,
    type=bool,
    mandatory=False,
    doc="Interpolate masked wrapped phase before snaphu (GMTSAR interp-style).",
)

SNAPHU_INTERP_RADIUS = Application.Parameter(
    "snaphuInterpRadius",
    public_name="snaphu interpolation radius",
    default=300,
    type=int,
    mandatory=False,
    doc="Interpolation search radius in pixels for masked wrapped phase.",
)

SNAPHU_TILE_NROW = Application.Parameter(
    "snaphuTileNRow",
    public_name="snaphu tile nrow",
    default=2,
    type=int,
    mandatory=False,
    doc="Number of snaphu tiles along azimuth (rows).",
)

SNAPHU_TILE_NCOL = Application.Parameter(
    "snaphuTileNCol",
    public_name="snaphu tile ncol",
    default=2,
    type=int,
    mandatory=False,
    doc="Number of snaphu tiles along range (columns).",
)

SNAPHU_ROW_OVERLAP = Application.Parameter(
    "snaphuRowOverlap",
    public_name="snaphu row overlap",
    default=400,
    type=int,
    mandatory=False,
    doc="snaphu tile overlap in azimuth direction (pixels), enforced to >= 400 in tile mode.",
)

SNAPHU_COL_OVERLAP = Application.Parameter(
    "snaphuColOverlap",
    public_name="snaphu col overlap",
    default=400,
    type=int,
    mandatory=False,
    doc="snaphu tile overlap in range direction (pixels), enforced to >= 400 in tile mode.",
)

USE_HIGH_RESOLUTION_DEM_ONLY = Application.Parameter(
    "useHighResolutionDemOnly",
    public_name="useHighResolutionDemOnly",
    default=False,
    type=int,
    mandatory=False,
    doc="If True, only download highest resolution SRTM DEM.",
)

DEM_FILENAME = Application.Parameter(
    "demFilename",
    public_name="demFilename",
    default="",
    type=str,
    mandatory=False,
    doc="Filename of the DEM init file.",
)

REGION_OF_INTEREST = Application.Parameter(
    "regionOfInterest",
    public_name="regionOfInterest",
    default=None,
    container=list,
    type=float,
    doc="Region of interest - South, North, West, East in degrees.",
)

GEOCODE_BOX = Application.Parameter(
    "geocode_bbox",
    public_name="geocode bounding box",
    default=None,
    container=list,
    type=float,
    doc="Bounding box for geocoding - South, North, West, East in degrees.",
)

GEO_POSTING = Application.Parameter(
    "geoPosting",
    public_name="geoPosting",
    default=None,
    type=float,
    mandatory=False,
    doc="Output posting for geocoded images in degrees.",
)

POSTING = Application.Parameter(
    "posting",
    public_name="posting",
    default=30,
    type=int,
    mandatory=False,
    doc="posting for interferogram.",
)

NUMBER_RANGE_LOOKS = Application.Parameter(
    "numberRangeLooks",
    public_name="range looks",
    default=None,
    type=int,
    mandatory=False,
    doc="Number of range looks.",
)

NUMBER_AZIMUTH_LOOKS = Application.Parameter(
    "numberAzimuthLooks",
    public_name="azimuth looks",
    default=None,
    type=int,
    mandatory=False,
    doc="Number of azimuth looks.",
)

FILTER_STRENGTH = Application.Parameter(
    "filterStrength",
    public_name="filter strength",
    default=0.5,
    type=float,
    mandatory=False,
    doc="Goldstein filter strength.",
)

USE_GPU = Application.Parameter(
    "useGPU",
    public_name="use GPU",
    default=True,
    type=bool,
    mandatory=False,
    doc="Prefer GPU-enabled processing where supported.",
)

USE_EXTERNAL_COREGISTRATION = Application.Parameter(
    "useExternalCoregistration",
    public_name="use external coregistration",
    default=False,
    type=bool,
    mandatory=False,
    doc="Use external coregistration.",
)

DO_DENSEOFFSETS = Application.Parameter(
    "doDenseOffsets",
    public_name="do denseoffsets",
    default=False,
    type=bool,
    mandatory=False,
    doc="Run dense offsets.",
)

DO_RUBBERSHEETINGAZIMUTH = Application.Parameter(
    "doRubbersheetingAzimuth",
    public_name="do rubbersheetingAzimuth",
    default=False,
    type=bool,
    mandatory=False,
    doc="Run azimuth rubbersheeting.",
)

DO_RUBBERSHEETINGRANGE = Application.Parameter(
    "doRubbersheetingRange",
    public_name="do rubbersheetingRange",
    default=False,
    type=bool,
    mandatory=False,
    doc="Run range rubbersheeting.",
)

RUBBERSHEET_SNR_THRESHOLD = Application.Parameter(
    "rubberSheetSNRThreshold",
    public_name="rubber sheet SNR Threshold",
    default=5.0,
    type=float,
    mandatory=False,
    doc="SNR threshold for rubbersheeting.",
)

RUBBERSHEET_FILTER_SIZE = Application.Parameter(
    "rubberSheetFilterSize",
    public_name="rubber sheet filter size",
    default=9,
    type=int,
    mandatory=False,
    doc="Filter size for rubbersheeting.",
)

DENSE_WINDOW_WIDTH = Application.Parameter(
    "denseWindowWidth",
    public_name="dense window width",
    default=96,
    type=int,
    mandatory=False,
    doc="Dense offset correlation window width.",
)

DENSE_WINDOW_HEIGHT = Application.Parameter(
    "denseWindowHeight",
    public_name="dense window height",
    default=96,
    type=int,
    mandatory=False,
    doc="Dense offset correlation window height.",
)

DENSE_SEARCH_WIDTH = Application.Parameter(
    "denseSearchWidth",
    public_name="dense search width",
    default=48,
    type=int,
    mandatory=False,
    doc="Dense offset search half-width.",
)

DENSE_SEARCH_HEIGHT = Application.Parameter(
    "denseSearchHeight",
    public_name="dense search height",
    default=48,
    type=int,
    mandatory=False,
    doc="Dense offset search half-height.",
)

DENSE_SKIP_WIDTH = Application.Parameter(
    "denseSkipWidth",
    public_name="dense skip width",
    default=32,
    type=int,
    mandatory=False,
    doc="Dense offset skip width.",
)

DENSE_SKIP_HEIGHT = Application.Parameter(
    "denseSkipHeight",
    public_name="dense skip height",
    default=32,
    type=int,
    mandatory=False,
    doc="Dense offset skip height.",
)

DO_SPLIT_SPECTRUM = Application.Parameter(
    "doSplitSpectrum",
    public_name="do split spectrum",
    default=False,
    type=bool,
    mandatory=False,
    doc="Enable split-spectrum processing.",
)

DO_DISPERSIVE = Application.Parameter(
    "doDispersive",
    public_name="do dispersive",
    default=False,
    type=bool,
    mandatory=False,
    doc="Enable dispersive phase estimation.",
)

DISPERSIVE_FILTER_FILLING_METHOD = Application.Parameter(
    "dispersive_filling_method",
    public_name="dispersive filter filling method",
    default="nearest_neighbour",
    type=str,
    mandatory=False,
    doc="Method to fill masked holes in dispersive filtering.",
)

DISPERSIVE_FILTER_KERNEL_XSIZE = Application.Parameter(
    "kernel_x_size",
    public_name="dispersive filter kernel x-size",
    default=800,
    type=float,
    mandatory=False,
    doc="Kernel x-size for dispersive filtering.",
)

DISPERSIVE_FILTER_KERNEL_YSIZE = Application.Parameter(
    "kernel_y_size",
    public_name="dispersive filter kernel y-size",
    default=800,
    type=float,
    mandatory=False,
    doc="Kernel y-size for dispersive filtering.",
)

DISPERSIVE_FILTER_KERNEL_SIGMA_X = Application.Parameter(
    "kernel_sigma_x",
    public_name="dispersive filter kernel sigma_x",
    default=100,
    type=float,
    mandatory=False,
    doc="Kernel sigma_x for dispersive filtering.",
)

DISPERSIVE_FILTER_KERNEL_SIGMA_Y = Application.Parameter(
    "kernel_sigma_y",
    public_name="dispersive filter kernel sigma_y",
    default=100,
    type=float,
    mandatory=False,
    doc="Kernel sigma_y for dispersive filtering.",
)

DISPERSIVE_FILTER_KERNEL_ROTATION = Application.Parameter(
    "kernel_rotation",
    public_name="dispersive filter kernel rotation",
    default=0.0,
    type=float,
    mandatory=False,
    doc="Kernel rotation for dispersive filtering.",
)

DISPERSIVE_FILTER_ITERATION_NUMBER = Application.Parameter(
    "dispersive_filter_iterations",
    public_name="dispersive filter number of iterations",
    default=5,
    type=int,
    mandatory=False,
    doc="Iteration count for dispersive filtering.",
)

DISPERSIVE_FILTER_MASK_TYPE = Application.Parameter(
    "dispersive_filter_mask_type",
    public_name="dispersive filter mask type",
    default="connected_components",
    type=str,
    mandatory=False,
    doc="Mask type for dispersive filtering.",
)

DISPERSIVE_FILTER_COHERENCE_THRESHOLD = Application.Parameter(
    "dispersive_filter_coherence_threshold",
    public_name="dispersive filter coherence threshold",
    default=0.5,
    type=float,
    mandatory=False,
    doc="Coherence threshold for dispersive filtering mask.",
)

HEIGHT_RANGE = Application.Parameter(
    "heightRange",
    public_name="height range",
    default=None,
    container=list,
    type=float,
    doc="Altitude range used for DEM bbox estimation.",
)

DO_MULTILOOK = Application.Parameter(
    "do_multilook",
    public_name="do multilook",
    default=False,
    type=bool,
    mandatory=False,
    doc="Apply multilooking to SLC before coregistration.",
)

MULTILOOK_AZ = Application.Parameter(
    "multilookAz",
    public_name="multilook azimuth looks",
    default=None,
    type=int,
    mandatory=False,
    doc="Azimuth looks for multilook (e.g., 2).",
)

MULTILOOK_RG = Application.Parameter(
    "multilookRg",
    public_name="multilook range looks",
    default=None,
    type=int,
    mandatory=False,
    doc="Range looks for multilook (e.g., 4).",
)

GEOCODE_LIST = Application.Parameter(
    "geocode_list",
    public_name="geocode list",
    default=None,
    container=list,
    type=str,
    doc="List of products to geocode.",
)

OFFSET_GEOCODE_LIST = Application.Parameter(
    "off_geocode_list",
    public_name="offset geocode list",
    default=None,
    container=list,
    mandatory=False,
    doc="List of offset-specific files to geocode.",
)

PICKLE_DUMPER_DIR = Application.Parameter(
    "pickleDumpDir",
    public_name="pickle dump directory",
    default="PICKLE",
    type=str,
    mandatory=False,
    doc="Directory for pickle objects.",
)

PICKLE_LOAD_DIR = Application.Parameter(
    "pickleLoadDir",
    public_name="pickle load directory",
    default="PICKLE",
    type=str,
    mandatory=False,
    doc="Directory to load pickle objects from.",
)

RENDERER = Application.Parameter(
    "renderer",
    public_name="renderer",
    default="xml",
    type=str,
    mandatory=True,
    doc="Format for steps serialization: xml or pickle.",
)

REFERENCE = Application.Facility(
    "reference",
    public_name="Reference",
    module="isceobj.StripmapProc.Sensor",
    factory="createSensor",
    args=(SENSOR_NAME, REFERENCE_SENSOR_NAME, "reference"),
    mandatory=False,
    doc="Reference raw data component",
)

SECONDARY = Application.Facility(
    "secondary",
    public_name="Secondary",
    module="isceobj.StripmapProc.Sensor",
    factory="createSensor",
    args=(SENSOR_NAME, SECONDARY_SENSOR_NAME, "secondary"),
    mandatory=False,
    doc="Secondary raw data component",
)

DEM_STITCHER = Application.Facility(
    "demStitcher",
    public_name="demStitcher",
    module="iscesys.DataManager",
    factory="createManager",
    args=("dem1", "iscestitcher"),
    mandatory=False,
    doc="Object that creates a DEM from frame bounding boxes.",
)

RUN_UNWRAPPER = Application.Facility(
    "runUnwrapper",
    public_name="Run unwrapper",
    module="isceobj.StripmapProc",
    factory="createUnwrapper",
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module",
)

RUN_UNWRAP_2STAGE = Application.Facility(
    "runUnwrap2Stage",
    public_name="Run unwrapper 2 Stage",
    module="isceobj.TopsProc",
    factory="createUnwrap2Stage",
    args=(SELF(), False, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module 2Stage",
)

_INSAR = Application.Facility(
    "_insar",
    public_name="insar",
    module="isceobj.StripmapProc",
    factory="createStripmapProc",
    args=("strip_insarContext", isceobj.createCatalog("strip_insar")),
    mandatory=False,
    doc="InsarProc object",
)


class StripInsarApp(Application, FrameMixin):
    family = "insar"

    parameter_list = (
        SENSOR_NAME,
        REFERENCE_SENSOR_NAME,
        SECONDARY_SENSOR_NAME,
        FILTER_STRENGTH,
        USE_GPU,
        USE_EXTERNAL_COREGISTRATION,
        CORRELATION_METHOD,
        REFERENCE_DOPPLER_METHOD,
        SECONDARY_DOPPLER_METHOD,
        ORBIT_INTERPOLATION_METHOD,
        UNWRAPPER_NAME,
        DO_UNWRAP,
        SNAPHU_GMTSAR_PREPROCESS,
        SNAPHU_CORR_THRESHOLD,
        SNAPHU_INTERP_MASKED_PHASE,
        SNAPHU_INTERP_RADIUS,
        SNAPHU_TILE_NROW,
        SNAPHU_TILE_NCOL,
        SNAPHU_ROW_OVERLAP,
        SNAPHU_COL_OVERLAP,
        USE_HIGH_RESOLUTION_DEM_ONLY,
        DEM_FILENAME,
        GEO_POSTING,
        POSTING,
        NUMBER_RANGE_LOOKS,
        NUMBER_AZIMUTH_LOOKS,
        DO_MULTILOOK,
        MULTILOOK_AZ,
        MULTILOOK_RG,
        GEOCODE_LIST,
        OFFSET_GEOCODE_LIST,
        GEOCODE_BOX,
        REGION_OF_INTEREST,
        DO_DENSEOFFSETS,
        DO_RUBBERSHEETINGRANGE,
        DO_RUBBERSHEETINGAZIMUTH,
        RUBBERSHEET_SNR_THRESHOLD,
        RUBBERSHEET_FILTER_SIZE,
        DENSE_WINDOW_WIDTH,
        DENSE_WINDOW_HEIGHT,
        DENSE_SEARCH_WIDTH,
        DENSE_SEARCH_HEIGHT,
        DENSE_SKIP_WIDTH,
        DENSE_SKIP_HEIGHT,
        DO_SPLIT_SPECTRUM,
        DO_DISPERSIVE,
        DISPERSIVE_FILTER_FILLING_METHOD,
        DISPERSIVE_FILTER_KERNEL_XSIZE,
        DISPERSIVE_FILTER_KERNEL_YSIZE,
        DISPERSIVE_FILTER_KERNEL_SIGMA_X,
        DISPERSIVE_FILTER_KERNEL_SIGMA_Y,
        DISPERSIVE_FILTER_KERNEL_ROTATION,
        DISPERSIVE_FILTER_ITERATION_NUMBER,
        DISPERSIVE_FILTER_MASK_TYPE,
        DISPERSIVE_FILTER_COHERENCE_THRESHOLD,
        HEIGHT_RANGE,
        PICKLE_DUMPER_DIR,
        PICKLE_LOAD_DIR,
        RENDERER,
    )

    facility_list = (
        REFERENCE,
        SECONDARY,
        DEM_STITCHER,
        RUN_UNWRAPPER,
        RUN_UNWRAP_2STAGE,
        _INSAR,
    )

    _pickleObj = "_insar"

    def __init__(self, family="", name="", cmdline=None):
        import isceobj

        super().__init__(family=family, name=name, cmdline=cmdline)

        from isceobj.StripmapProc import StripmapProc
        from iscesys.StdOEL.StdOELPy import create_writer

        self._stdWriter = create_writer("log", "", True, filename="strip_insar.log")
        self._add_methods()
        self._insarProcFact = StripmapProc
        self.timeStart = None
        self.work_dir = os.path.abspath(os.getcwd())
        # Compatibility with StripmapProc.runVerifyDEM, which checks this
        # attribute on the application object.
        self.heightRange = None
        self.demImage = None
        self.extremes = None
        self.violations = None
        self.useZeroTiles = True
        self.quiet_console = True
        self.detail_log = os.path.abspath("strip_insar.detail.log")

        return None

    def Usage(self):
        print("用法:")
        print("  strip_insar.py <input-file.xml>   # 使用配置文件运行")
        print("  strip_insar.py --steps             # 分步骤运行")
        print("  strip_insar.py --help              # 查看帮助")

    @property
    def frame(self):
        return self.insar.frame

    def _configure(self):
        self.insar.procDoc._addItem(
            "ISCE_VERSION",
            "Release: %s, svn-%s, %s" % (
                isce.release_version,
                isce.release_svn_revision,
                isce.release_date,
            ),
            ["strip_insar"],
        )

        if self.geocode_list is None:
            self.geocode_list = self.insar.geocode_list

        if self.off_geocode_list is None:
            self.off_geocode_list = self.insar.off_geocode_list

        if self.multilookAz is None:
            self.multilookAz = self.numberAzimuthLooks
        if self.multilookRg is None:
            self.multilookRg = self.numberRangeLooks

        if self.multilookAz is not None and self.multilookAz < 1:
            raise ValueError(f"multilookAz must be >= 1, got {self.multilookAz}")
        if self.multilookRg is not None and self.multilookRg < 1:
            raise ValueError(f"multilookRg must be >= 1, got {self.multilookRg}")
        if self.multilookAz is not None and not isinstance(self.multilookAz, int):
            raise TypeError(f"multilookAz must be int, got {type(self.multilookAz).__name__}")
        if self.multilookRg is not None and not isinstance(self.multilookRg, int):
            raise TypeError(f"multilookRg must be int, got {type(self.multilookRg).__name__}")

        return None

    @property
    def insar(self):
        return self._insar

    @insar.setter
    def insar(self, value):
        self._insar = value

    @property
    def procDoc(self):
        return self.insar.procDoc

    def _add_methods(self):
        self.runPreprocessor = StripmapProc.createPreprocessor(self)
        self.runFormSLC = StripmapProc.createFormSLC(self)
        self.runCrop = StripmapProc.createCrop(self)
        self.runMultilook = self._create_multilook_wrapper()
        self.runSplitSpectrum = StripmapProc.createSplitSpectrum(self)
        self.runTopo = StripmapProc.createTopo(self)
        self.runNormalizeSecondarySampling = StripmapProc.createNormalizeSecondarySampling(self)
        self.runGeo2rdr = StripmapProc.createGeo2rdr(self)
        self.runRdrDemOffset = StripmapProc.createRdrDemOffset(self)
        self.runRectRangeOffset = StripmapProc.createRectRangeOffset(self)
        self.runResampleSlc = StripmapProc.createResampleSlc(self)
        self.runRefineSecondaryTiming = StripmapProc.createRefineSecondaryTiming(self)
        self.runDenseOffsets = StripmapProc.createDenseOffsets(self)
        self.runRubbersheetRange = StripmapProc.createRubbersheetRange(self)
        self.runRubbersheetAzimuth = StripmapProc.createRubbersheetAzimuth(self)
        self.runResampleSubbandSlc = StripmapProc.createResampleSubbandSlc(self)
        self.runInterferogram = StripmapProc.createInterferogram(self)
        self.runFilter = StripmapProc.createFilter(self)
        self.verifyDEM = StripmapProc.createVerifyDEM(self)
        self.runGeocode = StripmapProc.createGeocode(self)

    def _create_multilook_wrapper(self):
        def run_multilook(self_app):
            nalks = self_app.multilookAz or 1
            nrlks = self_app.multilookRg or 1

            if not self_app.do_multilook:
                logger.info("[多视] 未启用多视，使用原始SLC（通过符号链接节省空间）")
                self_app._create_slc_symlink()
                return

            if nalks <= 1 and nrlks <= 1:
                logger.info("[多视] 视数为1，无需处理")
                self_app._create_slc_symlink()
                return

            logger.info(f"[多视] 开始多视处理: az={nalks}, rg={nrlks}")

            try:
                from multilook import multilook_isce_image
            except ImportError:
                from applications.multilook import multilook_isce_image

            ref_slc = getattr(self_app.insar, 'referenceSlcCropProduct', None)
            sec_slc = getattr(self_app.insar, 'secondarySlcCropProduct', None)

            if ref_slc:
                ml_path = self_app._get_ml_path(ref_slc, 'reference')
                multilook_isce_image(ref_slc, ml_path, nalks, nrlks, update_xml=True)
                self_app.insar.referenceSlcCropProduct = ml_path

            if sec_slc:
                ml_path = self_app._get_ml_path(sec_slc, 'secondary')
                multilook_isce_image(sec_slc, ml_path, nalks, nrlks, update_xml=True)
                self_app.insar.secondarySlcCropProduct = ml_path

            logger.info("[多视] 多视处理完成")

        return run_multilook

    def _get_ml_path(self, original_path, role):
        base_dir = os.path.join(self.work_dir, "02_ml_slc")
        os.makedirs(base_dir, exist_ok=True)
        basename = os.path.basename(original_path)
        name, ext = os.path.splitext(basename)
        if ext == '.slc':
            ml_name = f"{name}.ml.slc"
        else:
            ml_name = f"{name}.ml{ext}"
        return os.path.join(base_dir, ml_name)

    def _get_ml_xml_path(self, original_path):
        basename = os.path.basename(original_path)
        name, ext = os.path.splitext(basename)
        if ext == '.slc':
            xml_name = f"{name}.slc.xml"
        elif ext == '.xml':
            xml_name = basename
            name = name.replace('.slc', '')
            basename = name + '.slc' + ext
        else:
            xml_name = f"{name}{ext}.xml"
        return os.path.join(self.work_dir, "02_ml_slc", xml_name)

    def _create_slc_symlink(self):
        import shutil

        for role in ['reference', 'secondary']:
            slc_attr = f'{role}SlcCropProduct'
            orig_path = getattr(self.insar, slc_attr, None)
            if not orig_path:
                continue

            ml_dir = os.path.join(self.work_dir, "02_ml_slc")
            os.makedirs(ml_dir, exist_ok=True)

            basename = os.path.basename(orig_path)
            name, ext = os.path.splitext(basename)
            if ext == '.slc':
                ml_path = os.path.join(ml_dir, f"{name}.ml.slc")
            else:
                ml_path = os.path.join(ml_dir, f"{name}.ml{ext}")

            orig_abs = os.path.abspath(orig_path)
            ml_abs = os.path.abspath(ml_path)

            if not os.path.exists(ml_abs):
                if os.path.exists(orig_abs):
                    os.symlink(orig_abs, ml_abs)
                    logger.info(f"[多视] 创建符号链接: {ml_abs} -> {orig_abs}")
                else:
                    logger.warning(f"[多视] 原始文件不存在: {orig_abs}")
                    continue

            orig_xml = orig_abs + '.xml' if not orig_abs.endswith('.xml') else orig_abs
            ml_xml = ml_abs + '.xml' if not ml_abs.endswith('.xml') else ml_abs

            if not os.path.exists(ml_xml) and os.path.exists(orig_xml):
                shutil.copy(orig_xml, ml_xml)
                logger.info(f"[多视] 复制XML: {ml_xml}")

    def _steps(self):
        self.step("startup", func=self.startup, doc="初始化处理")
        self.step("preprocess", func=self.runPreprocessor, doc="预处理原始数据")
        self.step("cropraw", func=self.runCrop, args=(True,))
        self.step("formslc", func=self.runFormSLC)
        self.step("cropslc", func=self.runCrop, args=(False,))
        self.step("multilook", func=self.runMultilook, doc="SLC多视处理")
        self.step("verifyDEM", func=self.verifyDEM)
        self.step("topo", func=self.runTopo)
        self.step("normalize_secondary_sampling", func=self.runNormalizeSecondarySampling)
        self.step("geo2rdr", func=self.runGeo2rdr)
        self.step("rdrdem_offset", func=self.runRdrDemOffset)
        self.step("rect_rgoffset", func=self.runRectRangeOffset)
        self.step("coarse_resample", func=self.runResampleSlc, args=("coarse",))
        self.step("misregistration", func=self.runRefineSecondaryTiming)
        self.step("refined_resample", func=self.runResampleSlc, args=("refined",))
        self.step("dense_offsets", func=self.runDenseOffsets)
        self.step("rubber_sheet_range", func=self.runRubbersheetRange)
        self.step("rubber_sheet_azimuth", func=self.runRubbersheetAzimuth)
        self.step("fine_resample", func=self.runResampleSlc, args=("fine",))
        self.step("split_range_spectrum", func=self.runSplitSpectrum)
        self.step("sub_band_resample", func=self.runResampleSubbandSlc, args=(True,))
        self.step("interferogram", func=self.runInterferogram)
        self.step("filter", func=self.runFilter, args=(self.filterStrength,))
        self.step("unwrap", func=self.runUnwrapper)
        self.step("geocode", func=self.runGeocode, args=(self.geocode_list, self.geocode_bbox))
        self.step("geocodeoffsets", func=self.runGeocode, args=(self.off_geocode_list, self.geocode_bbox, True))
        self.step("export_products", func=self.export_geocode_products, doc="导出GeoTIFF/PNG/KML及LOS位移产品")
        self.step("endup", func=self.endup)
        return None

    def startup(self):
        logger.info("=" * 60)
        logger.info("StripInSAR 处理启动")
        logger.info("=" * 60)
        self._insar.timeStart = time.time()

    def endup(self):
        self.procDoc.renderXml()
        self._insar.timeEnd = time.time()
        elapsed = self._insar.timeEnd - self._insar.timeStart
        logger.info("=" * 60)
        logger.info(f"处理完成 | 总耗时: {elapsed:.1f} 秒")
        self._write_result_summary(elapsed)
        logger.info("=" * 60)

    def renderProcDoc(self):
        self.procDoc.renderXml()

    def _init(self):
        message = (
            "ISCE VERSION = %s, RELEASE_SVN_REVISION = %s, "
            + "RELEASE_DATE = %s, CURRENT_SVN_REVISION = %s"
        ) % (
            isce.__version__,
            isce.release_svn_revision,
            isce.release_date,
            isce.svn_revision,
        )
        logger.info(message)
        return None

    def help(self):
        pass

    def main(self):
        self.timeStart = time.time()
        self._insar.timeStart = self.timeStart
        self.help()
        stages = [
            ("preprocess", lambda: self.runPreprocessor()),
            ("cropraw", lambda: self.runCrop(True)),
            ("formslc", lambda: self.runFormSLC()),
            ("cropslc", lambda: self.runCrop(False)),
            (
                "multilook",
                lambda: self.runMultilook() if self.do_multilook else self._create_slc_symlink(),
            ),
            ("verifyDEM", lambda: self.verifyDEM()),
            ("topo", lambda: self.runTopo()),
            ("normalize_secondary_sampling", lambda: self.runNormalizeSecondarySampling()),
            ("geo2rdr", lambda: self.runGeo2rdr()),
            ("rdrdem_offset", lambda: self.runRdrDemOffset()),
            ("rect_rgoffset", lambda: self.runRectRangeOffset()),
            ("coarse_resample", lambda: self.runResampleSlc("coarse")),
            ("misregistration", lambda: self.runRefineSecondaryTiming()),
            ("refined_resample", lambda: self.runResampleSlc("refined")),
            ("dense_offsets", lambda: self.runDenseOffsets()),
            ("rubber_sheet_range", lambda: self.runRubbersheetRange()),
            ("rubber_sheet_azimuth", lambda: self.runRubbersheetAzimuth()),
            ("fine_resample", lambda: self.runResampleSlc("fine")),
            ("split_range_spectrum", lambda: self.runSplitSpectrum()),
            ("sub_band_resample", lambda: self.runResampleSubbandSlc(True)),
            ("interferogram", lambda: self.runInterferogram()),
            ("filter", lambda: self.runFilter(self.filterStrength)),
            ("unwrap", lambda: self.runUnwrapper()),
            ("geocode", lambda: self.runGeocode(self.geocode_list, self.geocode_bbox)),
            ("geocodeoffsets", lambda: self.runGeocode(self.off_geocode_list, self.geocode_bbox, True)),
            ("export_products", lambda: self.export_geocode_products()),
        ]
        total = len(stages)
        for idx, (name, fn) in enumerate(stages, 1):
            self._run_stage(name, fn, idx, total)

        self.timeEnd = time.time()
        logger.info("Total Time: %i seconds" % (self.timeEnd - self.timeStart))
        self._archive_products_by_type()
        self._write_result_summary(self.timeEnd - self.timeStart)
        self.renderProcDoc()
        return None

    def _run_stage(self, name, fn, idx, total):
        self._write_current_stage(name, idx, total)
        if self.quiet_console:
            self._emit_stage_console_line("[阶段 %d/%d] 开始: %s" % (idx, total, name))
        else:
            logger.info("[阶段 %d/%d] 开始: %s", idx, total, name)
        t0 = time.time()
        stop_evt = threading.Event()
        hb = None
        try:
            if self.quiet_console:
                with self._redirect_stage_io(self.detail_log):
                    if name == "preprocess":
                        self._precheck_preprocess_sensor_config()
                    if name == "unwrap":
                        self._precheck_unwrap_stage()
                    fn()
            else:
                hb = threading.Thread(
                    target=self._stage_heartbeat,
                    args=(name, idx, total, t0, stop_evt, None),
                    daemon=True,
                )
                hb.start()
                if name == "preprocess":
                    self._precheck_preprocess_sensor_config()
                if name == "unwrap":
                    self._precheck_unwrap_stage()
                fn()
        except Exception:
            logger.exception("[阶段 %d/%d] 失败: %s", idx, total, name)
            self._log_detail_tail()
            raise
        finally:
            stop_evt.set()
            if hb is not None:
                hb.join(timeout=1.0)
        if name == "preprocess":
            self._normalize_product_xml_paths()
        if name == "unwrap":
            self._check_unwrap_outputs()
        dt = time.time() - t0
        if self.quiet_console:
            self._emit_stage_console_line("[阶段 %d/%d] 完成: %s (%.1fs)" % (idx, total, name, dt))
        else:
            logger.info("[阶段 %d/%d] 完成: %s (%.1fs)", idx, total, name, dt)
        self._write_current_stage(name, idx, total, done=True)

    def _emit_stage_console_line(self, line):
        try:
            os.write(1, (str(line).rstrip() + "\n").encode("utf-8", errors="replace"))
        except Exception:
            pass

    def _stage_heartbeat(self, name, idx, total, t0, stop_evt, console_fd, interval=5):
        """
        Emit periodic progress heartbeat while a long stage is running.
        """
        while not stop_evt.wait(interval):
            dt = time.time() - t0
            msg = "[阶段 %d/%d] 进行中: %s (%.0fs)\n" % (idx, total, name, dt)
            if console_fd is not None:
                try:
                    os.write(console_fd, msg.encode("utf-8", errors="replace"))
                except Exception:
                    pass
            else:
                logger.info("[阶段 %d/%d] 进行中: %s (%.0fs)", idx, total, name, dt)

    @contextlib.contextmanager
    def _redirect_stage_io(self, logfile):
        """
        Redirect low-level stdout/stderr (FD 1/2) to logfile, while returning
        a duplicate of original stdout FD for heartbeat printing.
        """
        with open(logfile, "a", encoding="utf-8") as detail_fp:
            detail_fp.write("\n==== stage output redirect start ====\n")
            detail_fp.flush()
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            console_fd = os.dup(saved_stdout_fd)
            try:
                os.dup2(detail_fp.fileno(), 1)
                os.dup2(detail_fp.fileno(), 2)
                yield console_fd
            finally:
                try:
                    os.dup2(saved_stdout_fd, 1)
                    os.dup2(saved_stderr_fd, 2)
                finally:
                    os.close(saved_stdout_fd)
                    os.close(saved_stderr_fd)
                    try:
                        os.close(console_fd)
                    except Exception:
                        pass

    def _normalize_product_xml_paths(self):
        """
        Keep product XML paths inside managed output directories.
        """
        mapping = {
            "referenceRawProduct": "raw",
            "secondaryRawProduct": "raw",
            "referenceRawCropProduct": "raw",
            "secondaryRawCropProduct": "raw",
            "referenceSlcProduct": "slc",
            "secondarySlcProduct": "slc",
            "referenceSlcCropProduct": "slc",
            "secondarySlcCropProduct": "slc",
        }
        for attr, key in mapping.items():
            val = getattr(self.insar, attr, None)
            if not val:
                continue
            if os.path.isabs(val):
                continue
            base_dir = os.path.join(self.work_dir, OutputPlanner.OUTPUT_DIRS[key])
            setattr(self.insar, attr, os.path.join(base_dir, os.path.basename(val)))

    def _write_result_summary(self, elapsed_seconds):
        """
        Emit concise final result summary to console and to a result file.
        """
        out = []
        out.append("StripInSAR Result Summary")
        out.append(f"elapsed_seconds: {elapsed_seconds:.1f}")
        out.append(f"work_dir: {self.work_dir}")

        key_products = [
            ("reference_slc_crop_xml", getattr(self.insar, "referenceSlcCropProduct", None)),
            ("secondary_slc_crop_xml", getattr(self.insar, "secondarySlcCropProduct", None)),
            ("ifg", os.path.join(self.insar.ifgDirname, self.insar.ifgFilename)),
            ("filt_ifg", os.path.join(self.insar.ifgDirname, "filt_" + self.insar.ifgFilename)),
            ("coherence", os.path.join(self.insar.ifgDirname, self.insar.coherenceFilename)),
            ("unwrapped", os.path.join(self.insar.ifgDirname, self.insar.unwrappedIfgFilename)),
            ("geometry_lat", os.path.join(self.insar.geometryDirname, self.insar.latFilename)),
            ("geometry_lon", os.path.join(self.insar.geometryDirname, self.insar.lonFilename)),
            ("geometry_los", os.path.join(self.insar.geometryDirname, self.insar.losFilename)),
            ("dense_offsets", os.path.join(self.insar.denseOffsetsDirname, self.insar.denseOffsetFilename + ".bil")),
        ]

        out.append("products:")
        for key, path in key_products:
            if not path:
                continue
            p = path if os.path.isabs(path) else os.path.join(self.work_dir, path)
            exists = os.path.exists(p) or os.path.exists(p + ".xml") or os.path.exists(p + ".vrt")
            status = "OK" if exists else "MISSING"
            out.append(f"  - {key}: {p} [{status}]")

        geo_list = list(getattr(self.insar, "geocode_list", []) or [])
        if geo_list:
            out.append("geocoded_products:")
            for prod in geo_list:
                p = prod if os.path.isabs(prod) else os.path.join(self.work_dir, prod)
                out.append(f"  - {p}.geo")

        # Optional outputs (06/08/09) are conditional by workflow switches.
        out.append("notes:")
        out.append("  - 06_unwrapped populated only when do_unwrap=True and downstream writer stores products there.")
        out.append("  - 08_dense_offsets populated only when doDenseOffsets=True.")
        out.append("  - 09_ionosphere populated only when doDispersive=True.")

        result_file = os.path.join(self.work_dir, "log", "strip_insar.result.txt")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")

        logger.info("结果摘要: %s", result_file)
        for line in out:
            logger.info(line)

    def _archive_products_by_type(self):
        """
        Organize outputs into dedicated folders:
        - 06_unwrapped: unwrapped products
        - 07_geocoded: geocoded products
        Keep symlinks at original locations for backward compatibility.
        """
        ifg_dir = getattr(self.insar, "ifgDirname", None)
        if not ifg_dir or (not os.path.isdir(ifg_dir)):
            return

        unwrapped_dir = os.path.join(self.work_dir, OutputPlanner.OUTPUT_DIRS["unwrapped"])
        geocoded_dir = os.path.join(self.work_dir, OutputPlanner.OUTPUT_DIRS["geocode"])
        os.makedirs(unwrapped_dir, exist_ok=True)
        os.makedirs(geocoded_dir, exist_ok=True)

        def _safe_move_with_symlink(src, dst_dir):
            if (not os.path.exists(src)) and (not os.path.islink(src)):
                return None
            dst = os.path.join(dst_dir, os.path.basename(src))
            # skip if destination already exists
            if os.path.exists(dst) or os.path.islink(dst):
                return dst
            # if source is already a symlink, keep as is
            if os.path.islink(src):
                return dst
            os.replace(src, dst)
            try:
                os.symlink(dst, src)
            except Exception:
                # if symlink fails, restore by moving back
                os.replace(dst, src)
                return None
            return dst

        moved_unw = 0
        moved_geo = 0

        for name in os.listdir(ifg_dir):
            src = os.path.join(ifg_dir, name)
            # Unwrapped family
            if (".unw" in name) or (".conncomp" in name):
                if _safe_move_with_symlink(src, unwrapped_dir):
                    moved_unw += 1
                continue
            # Geocoded family
            if ".geo" in name:
                if _safe_move_with_symlink(src, geocoded_dir):
                    moved_geo += 1
                continue

        if moved_unw or moved_geo:
            logger.info(
                "结果归档完成: unwrapped->%s (%d), geocoded->%s (%d)",
                unwrapped_dir, moved_unw, geocoded_dir, moved_geo
            )

    def export_geocode_products(self):
        """
        After geocode, generate:
        - LOS displacement (meter)
        - GeoTIFF/PNG/KML for LOS, filtered phase, coherence, average intensity
        """
        geo_dir = os.path.join(self.work_dir, OutputPlanner.OUTPUT_DIRS["geocode"])
        out_dir = os.path.join(self.work_dir, "10_products")
        os.makedirs(out_dir, exist_ok=True)

        def _exists_any(p):
            return os.path.exists(p) or os.path.exists(p + ".vrt") or os.path.exists(p + ".xml")

        def _pick(cands):
            for c in cands:
                if _exists_any(c):
                    return c
            return None

        def _run(cmd):
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError("command failed: {0}\n{1}".format(" ".join(cmd), res.stderr.strip()))
            return res

        def _to_tif(src, dst, band=None, scale=False):
            cmd = ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=DEFLATE"]
            if band is not None:
                cmd += ["-b", str(int(band))]
            cmd += [src, dst]
            _run(cmd)

        def _read_arr(src, band=1):
            from osgeo import gdal
            ds = gdal.Open(src, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError("无法打开数据集: {0}".format(src))
            rb = ds.GetRasterBand(int(band))
            arr = rb.ReadAsArray()
            nodata = rb.GetNoDataValue()
            gt = ds.GetGeoTransform(can_return_null=True)
            prj = ds.GetProjectionRef()
            return ds, rb, arr, nodata, gt, prj

        def _write_float_tif(dst, arr, gt, prj, nodata=np.nan):
            from osgeo import gdal
            ysize, xsize = arr.shape
            drv = gdal.GetDriverByName("GTiff")
            ds = drv.Create(dst, int(xsize), int(ysize), 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
            if ds is None:
                raise RuntimeError("创建GeoTIFF失败: {0}".format(dst))
            if gt:
                ds.SetGeoTransform(gt)
            if prj:
                ds.SetProjection(prj)
            b = ds.GetRasterBand(1)
            b.WriteArray(arr.astype(np.float32))
            if nodata is not None:
                try:
                    b.SetNoDataValue(float(nodata))
                except Exception:
                    pass
            b.FlushCache()
            ds.FlushCache()
            ds = None

        def _to_png_stretched(src_tif, dst_png, log10_mode=False):
            """
            PNG generation policy:
            - mask invalid pixels (nodata/non-finite; plus <=0 for log10 mode)
            - apply 2%-98% stretch on valid pixels
            - write RGBA PNG with alpha=0 for invalid pixels
            """
            try:
                import numpy as np
                from osgeo import gdal
            except Exception:
                # fallback: no masking policy guaranteed
                cmd = ["gdal_translate", "-of", "PNG", "-scale", src_tif, dst_png]
                _run(cmd)
                return

            ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError("无法打开用于PNG生成的数据集: {0}".format(src_tif))
            b = ds.GetRasterBand(1)
            arr = b.ReadAsArray()
            if arr is None:
                raise RuntimeError("无法读取数据波段: {0}".format(src_tif))
            arr = np.asarray(arr, dtype=np.float64)
            nodata = b.GetNoDataValue()

            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= (arr != float(nodata))

            work = arr.copy()
            if log10_mode:
                valid &= (work > 0)
                work[valid] = np.log10(work[valid])

            if np.count_nonzero(valid) < 8:
                # fallback if too few valid values
                cmd = ["gdal_translate", "-of", "PNG", "-scale", src_tif, dst_png]
                _run(cmd)
                return

            vals = work[valid]
            p2, p98 = np.percentile(vals, [2.0, 98.0])
            if (not np.isfinite(p2)) or (not np.isfinite(p98)) or (p98 <= p2):
                p2 = float(np.min(vals))
                p98 = float(np.max(vals))

            rgb = np.zeros(arr.shape, dtype=np.uint8)
            if p98 > p2:
                s = (work - p2) / (p98 - p2)
                s = np.clip(s, 0.0, 1.0)
                # reserve 0 for invalid pixels
                rgb[valid] = (s[valid] * 254.0 + 1.0).astype(np.uint8)
            else:
                rgb[valid] = 128

            alpha = np.zeros(arr.shape, dtype=np.uint8)
            alpha[valid] = 255

            ysize, xsize = arr.shape
            mem = gdal.GetDriverByName("MEM").Create("", int(xsize), int(ysize), 4, gdal.GDT_Byte)
            try:
                gt = ds.GetGeoTransform(can_return_null=True)
                if gt:
                    mem.SetGeoTransform(gt)
                proj = ds.GetProjectionRef()
                if proj:
                    mem.SetProjection(proj)
                for ib in (1, 2, 3):
                    mem.GetRasterBand(ib).WriteArray(rgb)
                mem.GetRasterBand(4).WriteArray(alpha)
                mem.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
                gdal.GetDriverByName("PNG").CreateCopy(dst_png, mem, strict=0)
            finally:
                mem = None
                ds = None

        def _to_png_phase_cyclic(src_tif, dst_png):
            """
            Build cyclic color PNG for wrapped phase:
            - input assumed phase in radians
            - invalid pixels masked out (alpha=0)
            - hue mapped from [-pi, pi] cyclically
            """
            from osgeo import gdal
            ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError("无法打开用于相位PNG生成的数据集: {0}".format(src_tif))
            b = ds.GetRasterBand(1)
            arr = b.ReadAsArray()
            if arr is None:
                raise RuntimeError("无法读取相位数据: {0}".format(src_tif))
            arr = np.asarray(arr, dtype=np.float64)
            nodata = b.GetNoDataValue()
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= (arr != float(nodata))

            # normalize phase to [0, 2*pi)
            twopi = 2.0 * np.pi
            p = np.mod(arr + np.pi, twopi)
            h = p / twopi  # [0, 1)

            # HSV(h, 1, 1) -> RGB vectorized
            i = np.floor(h * 6.0).astype(np.int32)
            f = h * 6.0 - i
            q = 1.0 - f
            t = f
            i = np.mod(i, 6)

            r = np.zeros(arr.shape, dtype=np.float64)
            g = np.zeros(arr.shape, dtype=np.float64)
            bb = np.zeros(arr.shape, dtype=np.float64)

            m0 = (i == 0); r[m0] = 1.0; g[m0] = t[m0]; bb[m0] = 0.0
            m1 = (i == 1); r[m1] = q[m1]; g[m1] = 1.0; bb[m1] = 0.0
            m2 = (i == 2); r[m2] = 0.0; g[m2] = 1.0; bb[m2] = t[m2]
            m3 = (i == 3); r[m3] = 0.0; g[m3] = q[m3]; bb[m3] = 1.0
            m4 = (i == 4); r[m4] = t[m4]; g[m4] = 0.0; bb[m4] = 1.0
            m5 = (i == 5); r[m5] = 1.0; g[m5] = 0.0; bb[m5] = q[m5]

            rgb_r = np.zeros(arr.shape, dtype=np.uint8)
            rgb_g = np.zeros(arr.shape, dtype=np.uint8)
            rgb_b = np.zeros(arr.shape, dtype=np.uint8)
            rgb_r[valid] = np.clip(np.rint(r[valid] * 255.0), 0, 255).astype(np.uint8)
            rgb_g[valid] = np.clip(np.rint(g[valid] * 255.0), 0, 255).astype(np.uint8)
            rgb_b[valid] = np.clip(np.rint(bb[valid] * 255.0), 0, 255).astype(np.uint8)

            alpha = np.zeros(arr.shape, dtype=np.uint8)
            alpha[valid] = 255

            ysize, xsize = arr.shape
            mem = gdal.GetDriverByName("MEM").Create("", int(xsize), int(ysize), 4, gdal.GDT_Byte)
            try:
                gt = ds.GetGeoTransform(can_return_null=True)
                if gt:
                    mem.SetGeoTransform(gt)
                proj = ds.GetProjectionRef()
                if proj:
                    mem.SetProjection(proj)
                mem.GetRasterBand(1).WriteArray(rgb_r)
                mem.GetRasterBand(2).WriteArray(rgb_g)
                mem.GetRasterBand(3).WriteArray(rgb_b)
                mem.GetRasterBand(4).WriteArray(alpha)
                mem.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
                gdal.GetDriverByName("PNG").CreateCopy(dst_png, mem, strict=0)
            finally:
                mem = None
                ds = None

        def _bbox_from_tif(tif):
            info = _run(["gdalinfo", "-json", tif]).stdout
            j = json.loads(info)
            cc = j.get("cornerCoordinates", {})
            ul = cc.get("upperLeft")
            lr = cc.get("lowerRight")
            if (not ul) or (not lr):
                return None
            west, north = float(ul[0]), float(ul[1])
            east, south = float(lr[0]), float(lr[1])
            return (south, north, west, east)

        def _write_kml(png_path, tif_path, kml_path, layer_name):
            bbox = _bbox_from_tif(tif_path)
            if bbox is None:
                return
            south, north, west, east = bbox
            href = os.path.basename(png_path)
            kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{layer_name}</name>
    <GroundOverlay>
      <name>{layer_name}</name>
      <Icon><href>{href}</href></Icon>
      <LatLonBox>
        <north>{north}</north><south>{south}</south>
        <east>{east}</east><west>{west}</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>
"""
            with open(kml_path, "w", encoding="utf-8") as f:
                f.write(kml)

        ifg_dir = getattr(self.insar, "ifgDirname", "")
        # Prefer archived geocoded directory, fallback to ifg dir.
        los_src = _pick([
            os.path.join(geo_dir, "los.rdr.geo"),
            os.path.join(ifg_dir, "los.rdr.geo"),
        ])
        filt_src = _pick([
            os.path.join(geo_dir, "filt_topophase.flat.geo"),
            os.path.join(ifg_dir, "filt_topophase.flat.geo"),
        ])
        coh_src = _pick([
            os.path.join(geo_dir, "phsig.cor.geo"),
            os.path.join(ifg_dir, "phsig.cor.geo"),
        ])
        unw_src = _pick([
            os.path.join(geo_dir, "filt_topophase.unw.geo"),
            os.path.join(ifg_dir, "filt_topophase.unw.geo"),
        ])
        if unw_src is None:
            unw_src = _pick([
                os.path.join(geo_dir, "filt_topophase.unw"),
                os.path.join(ifg_dir, "filt_topophase.unw"),
            ])

        # Build true average intensity from master/slave SLC amplitudes and geocode it.
        avg_true_geo = self._build_true_avg_intensity_geo(ifg_dir)

        layers = []

        # 0) Build physically meaningful float layers first
        # LOS: use incidence angle band (band-1) for display/export
        if los_src:
            los_tif = os.path.join(out_dir, "los.tif")
            _to_tif(los_src, los_tif, band=1)
            layers.append(("los", los_tif, False))

        # Filtered wrapped phase: derive phase angle from complex interferogram
        if filt_src:
            try:
                ds, rb, arr, nodata, gt, prj = _read_arr(filt_src, band=1)
                carr = np.asarray(arr)
                if np.iscomplexobj(carr):
                    phase = np.angle(carr).astype(np.float64)
                else:
                    # Some layouts expose real/imag as two float bands.
                    if getattr(ds, "RasterCount", 0) >= 2:
                        imag = ds.GetRasterBand(2).ReadAsArray()
                        imag = np.asarray(imag, dtype=np.float64)
                        real = np.asarray(carr, dtype=np.float64)
                        phase = np.arctan2(imag, real).astype(np.float64)
                    else:
                        # Fallback for non-complex source
                        phase = carr.astype(np.float64)
                valid = np.isfinite(phase)
                if nodata is not None:
                    valid &= (np.real(carr) != float(nodata))
                out = np.full(phase.shape, np.nan, dtype=np.float64)
                out[valid] = phase[valid]
                phase_tif = os.path.join(out_dir, "filt_phase.tif")
                _write_float_tif(phase_tif, out, gt, prj, nodata=np.nan)
                ds = None
                layers.append(("filt_phase", phase_tif, "phase_cyclic"))
            except Exception as err:
                logger.warning("滤波相位导出失败，降级为直接导出band1: %s", str(err))
                phase_tif = os.path.join(out_dir, "filt_phase.tif")
                _to_tif(filt_src, phase_tif, band=1)
                layers.append(("filt_phase", phase_tif, "phase_cyclic"))

        # Coherence
        if coh_src:
            coh_tif = os.path.join(out_dir, "coherence.tif")
            _to_tif(coh_src, coh_tif, band=1)
            layers.append(("coherence", coh_tif, False))

        # True avg intensity. Do not fall back to unw band1 because that is not intensity.
        if avg_true_geo:
            avg_tif = os.path.join(out_dir, "avg_intensity.tif")
            _to_tif(avg_true_geo, avg_tif, band=1)
            layers.append(("avg_intensity", avg_tif, True))
        else:
            logger.warning("真实平均强度未生成，已跳过 avg_intensity 导出，避免误用 unwrapped 文件。")

        # 1) Export GeoTIFF/PNG/KML for requested layers
        for name, tif, mode in layers:
            png = os.path.join(out_dir, f"{name}.png")
            kml = os.path.join(out_dir, f"{name}.kml")
            if mode == "phase_cyclic":
                _to_png_phase_cyclic(tif, png)
            else:
                _to_png_stretched(tif, png, log10_mode=bool(mode))
            _write_kml(png, tif, kml, name)

        # 2) LOS displacement in meter from unwrapped phase
        if unw_src:
            # Try band-2 first (common ISCE unw layout: [amp, unw_phase]), fallback to band-1.
            disp_tif = os.path.join(out_dir, "los_displacement_m.tif")
            lam = self._autodetect_wavelength_from_products() or 0.237930522222
            calc_expr = f"-(A*{lam})/(4*3.141592653589793)"
            ok = False
            for band in (2, 1):
                cmd = [
                    "gdal_calc.py",
                    "-A", unw_src,
                    "--A_band", str(band),
                    "--calc", calc_expr,
                    "--type", "Float32",
                    "--NoDataValue", "nan",
                    "--format", "GTiff",
                    "--outfile", disp_tif,
                ]
                try:
                    _run(cmd)
                    ok = True
                    break
                except Exception:
                    continue
            if ok:
                disp_png = os.path.join(out_dir, "los_displacement_m.png")
                disp_kml = os.path.join(out_dir, "los_displacement_m.kml")
                _to_png_stretched(disp_tif, disp_png, log10_mode=False)
                _write_kml(disp_png, disp_tif, disp_kml, "los_displacement_m")
                logger.info("位移产品已生成: %s", disp_tif)
            else:
                logger.warning("未能从解缠产品生成位移GeoTIFF（gdal_calc失败）。")
        logger.info("最终可视化产品目录: %s", out_dir)

    def _build_true_avg_intensity_geo(self, ifg_dir):
        """
        Compute true average intensity from master/slave SLC amplitudes:
            (abs(master) + abs(slave)) / 2
        then geocode it.
        Returns geocoded file path or None.
        """
        try:
            xml_base = os.path.dirname(os.path.abspath(getattr(self, "input_xml", "") or "")) or os.getcwd()
            ref_xml = getattr(self.insar, "referenceSlcCropProduct", None)
            sec_xml = getattr(self.insar, "secondarySlcCropProduct", None)
            if not ref_xml or not sec_xml:
                return None

            def _img_path(prod_xml):
                prod = self.insar.loadProduct(prod_xml)
                if prod is None:
                    return None
                img = prod.getImage() if hasattr(prod, "getImage") else getattr(prod, "image", None)
                if img is None:
                    return None
                getf = getattr(img, "getFilename", None)
                path = getf() if callable(getf) else getattr(img, "filename", None)
                if not path:
                    return None
                path = str(path)
                if os.path.isabs(path):
                    return path
                # Resolve relative paths against the input XML directory first,
                # then current working directory as fallback.
                cand = os.path.join(xml_base, path)
                if os.path.exists(cand):
                    return cand
                return os.path.abspath(path)

            ref_slc = _img_path(ref_xml)
            sec_slc = _img_path(sec_xml)
            if (not ref_slc) or (not sec_slc):
                return None
            if (not os.path.exists(ref_slc)) or (not os.path.exists(sec_slc)):
                return None

            avg_radar = os.path.join(ifg_dir, "avg_intensity_true.bil")
            cmd = [
                "imageMath.py",
                '-e=(abs(a)+abs(b))/2.0',
                "--a=" + ref_slc,
                "--b=" + sec_slc,
                "-o", avg_radar,
                "-t", "float32",
                "-s", "BIL",
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                logger.warning("真实平均强度计算失败，回退近似方案: %s", res.stderr.strip())
                return None

            # Geocode the newly generated radar product.
            try:
                self.runGeocode([avg_radar], self.geocode_bbox)
            except Exception as err:
                logger.warning("真实平均强度地理编码失败，回退近似方案: %s", str(err))
                return None

            avg_geo = avg_radar + ".geo"
            if os.path.exists(avg_geo) or os.path.exists(avg_geo + ".xml") or os.path.exists(avg_geo + ".vrt"):
                logger.info("真实平均强度已生成并地理编码: %s", avg_geo)
                return avg_geo
            return None
        except Exception as err:
            logger.warning("真实平均强度流程异常，回退近似方案: %s", str(err))
            return None

    def _autodetect_wavelength_from_products(self):
        product_attrs = (
            "referenceSlcCropProduct",
            "referenceSlcProduct",
            "referenceRawProduct",
            "secondarySlcCropProduct",
            "secondarySlcProduct",
            "secondaryRawProduct",
        )
        seen = set()
        for attr in product_attrs:
            xmlname = getattr(self.insar, attr, None)
            if not xmlname or xmlname in seen:
                continue
            seen.add(xmlname)
            try:
                prod = self.insar.loadProduct(xmlname)
            except Exception:
                continue
            if prod is None:
                continue
            try:
                inst = prod.getInstrument() if hasattr(prod, "getInstrument") else getattr(prod, "instrument", None)
                getter = getattr(inst, "getRadarWavelength", None) if inst is not None else None
                wvl = getter() if callable(getter) else getattr(inst, "radarWavelength", None)
                wvl = float(wvl) if wvl is not None else None
            except Exception:
                wvl = None
            if (wvl is not None) and (0.001 < wvl < 1.0):
                return wvl
        return None

    def _write_current_stage(self, name, idx, total, done=False):
        stage_file = os.path.join(self.work_dir, "log", "current_stage.txt")
        os.makedirs(os.path.dirname(stage_file), exist_ok=True)
        status = "DONE" if done else "RUNNING"
        with open(stage_file, "w", encoding="utf-8") as f:
            f.write(f"stage={name}\n")
            f.write(f"index={idx}\n")
            f.write(f"total={total}\n")
            f.write(f"status={status}\n")

    def _log_detail_tail(self, n_lines=60):
        if not self.detail_log or (not os.path.isfile(self.detail_log)):
            return
        try:
            with open(self.detail_log, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-n_lines:]
            logger.error("失败阶段的详细日志末尾（%d行）:", len(lines))
            for ln in lines:
                logger.error(ln.rstrip("\n"))
        except Exception:
            pass

    def _check_unwrap_outputs(self):
        if not bool(getattr(self, "do_unwrap", False)):
            return
        ifg_dir = getattr(self.insar, "ifgDirname", None)
        ifg_name = getattr(self.insar, "ifgFilename", None)
        if (not ifg_dir) or (not ifg_name):
            return
        wrap = os.path.join(ifg_dir, "filt_" + ifg_name)
        if ".flat" in wrap:
            unw = wrap.replace(".flat", ".unw")
        elif ".int" in wrap:
            unw = wrap.replace(".int", ".unw")
        else:
            unw = wrap + ".unw"
        if not (os.path.exists(unw) or os.path.exists(unw + ".xml")):
            raise RuntimeError(
                "unwrap阶段未生成解缠结果: {0}. "
                "请检查grass/snaphu/icu依赖与日志文件 {1}".format(
                    unw, self.detail_log
                )
            )

    def _precheck_unwrap_stage(self):
        """
        Preflight checks before unwrapping to catch common failures early.
        """
        if not bool(getattr(self, "do_unwrap", False)):
            logger.info("unwrap预检: do_unwrap=False，跳过解缠。")
            self._append_stage_note("unwrap_precheck=SKIP(do_unwrap=False)")
            return

        unwrap_name = str(getattr(self, "unwrapper_name", "grass") or "grass").lower()
        ifg_dir = getattr(self.insar, "ifgDirname", None)
        ifg_name = getattr(self.insar, "ifgFilename", None)
        coh_name = getattr(self.insar, "coherenceFilename", None)

        if (not ifg_dir) or (not ifg_name) or (not coh_name):
            raise RuntimeError(
                "unwrap预检失败: ifgDirname/ifgFilename/coherenceFilename 未正确配置"
            )

        wrap = os.path.join(ifg_dir, "filt_" + ifg_name)
        coh = os.path.join(ifg_dir, coh_name)
        missing = []
        for p in (wrap, wrap + ".xml", coh, coh + ".xml"):
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            raise RuntimeError(
                "unwrap预检失败: 缺少解缠输入文件: {0}".format(", ".join(missing))
            )

        # Backend-specific import checks.
        backend_modules = {
            "grass": "mroipac.grass.grass",
            "snaphu": "contrib.Snaphu.Snaphu",
            "snaphu_mcf": "contrib.Snaphu.Snaphu",
            "icu": "mroipac.icu.Icu",
        }
        mod = backend_modules.get(unwrap_name)
        if mod is None:
            supported = ", ".join(sorted(backend_modules.keys()))
            raise RuntimeError(
                "unwrap预检失败: 未识别解缠器 '%s'，仅支持: %s"
                % (unwrap_name, supported)
            )
        try:
            importlib.import_module(mod)
        except Exception as err:
            raise RuntimeError(
                "unwrap预检失败: 解缠器 '%s' 依赖模块不可用 (%s)" % (unwrap_name, str(err))
            )
        self._append_stage_note("unwrap_precheck=OK backend=%s" % unwrap_name)

    def _precheck_preprocess_sensor_config(self):
        """
        Ensure sensor names/doppler methods are set before runPreprocessor.
        Prevent fallback to useDOPIQ when sensor name is missing.
        """
        def _norm_sensor_name(name):
            if not name:
                return None
            n = str(name).strip().lower()
            if n in ("lutan1", "lutan_1", "lt1"):
                return "lutan1"
            return n

        ref_cls = self.reference.__class__.__name__ if self.reference is not None else ""
        sec_cls = self.secondary.__class__.__name__ if self.secondary is not None else ""
        guessed = None
        for c in (ref_cls, sec_cls):
            cl = str(c).lower()
            if "lutan" in cl:
                guessed = "lutan1"
                break

        base = _norm_sensor_name(getattr(self, "sensorName", None)) or guessed
        refn = _norm_sensor_name(getattr(self, "referenceSensorName", None)) or base
        secn = _norm_sensor_name(getattr(self, "secondarySensorName", None)) or base

        if base:
            self.sensorName = base
        if refn:
            self.referenceSensorName = refn
        if secn:
            self.secondarySensorName = secn

        # Force known-safe doppler backend for LUTAN1 to avoid useDOPIQ path.
        if str(refn or "").lower() == "lutan1" and not getattr(self, "referenceDopplerMethod", None):
            self.referenceDopplerMethod = "useDEFAULT"
        if str(secn or "").lower() == "lutan1" and not getattr(self, "secondaryDopplerMethod", None):
            self.secondaryDopplerMethod = "useDEFAULT"

        logger.info(
            "preprocess预检: sensorName=%r, referenceSensorName=%r, secondarySensorName=%r, "
            "referenceDopplerMethod=%r, secondaryDopplerMethod=%r",
            getattr(self, "sensorName", None),
            getattr(self, "referenceSensorName", None),
            getattr(self, "secondarySensorName", None),
            getattr(self, "referenceDopplerMethod", None),
            getattr(self, "secondaryDopplerMethod", None),
        )
        self._append_stage_note(
            "preprocess_precheck=sensor={0},ref={1},sec={2},rdop={3},sdop={4}".format(
                getattr(self, "sensorName", None),
                getattr(self, "referenceSensorName", None),
                getattr(self, "secondarySensorName", None),
                getattr(self, "referenceDopplerMethod", None),
                getattr(self, "secondaryDopplerMethod", None),
            )
        )

    def _append_stage_note(self, note):
        stage_file = os.path.join(self.work_dir, "log", "current_stage.txt")
        try:
            with open(stage_file, "a", encoding="utf-8") as f:
                f.write(str(note).strip() + "\n")
        except Exception:
            pass



class OutputPlanner:
    OUTPUT_DIRS = {
        "raw": "01_raw_data",
        "slc": "02_slc",
        "ml_slc": "02_ml_slc",
        "coreg": "03_coregistered_slc",
        "geometry": "04_geometry",
        "interferogram": "05_interferogram",
        "coherence": "05_interferogram",
        "unwrapped": "06_unwrapped",
        "geocode": "07_geocoded",
        "dense_offsets": "08_dense_offsets",
        "ionosphere": "09_ionosphere",
        "cache": "cache",
        "log": "log",
        "pickle": "PICKLE",
    }
    PROC_DIR_ATTR_MAP = {
        "geometryDirname": "geometry",
        "offsetsDirname": "coreg",
        "denseOffsetsDirname": "dense_offsets",
        "coregDirname": "coreg",
        "ifgDirname": "interferogram",
        "misregDirname": "coreg",
        "splitSpectrumDirname": "ionosphere",
        "ionosphereDirname": "ionosphere",
    }

    @staticmethod
    def setup_output_structure(base_dir=None):
        if base_dir is None:
            base_dir = os.getcwd()

        created_dirs = {}
        for key, dir_name in OutputPlanner.OUTPUT_DIRS.items():
            dir_path = os.path.join(base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            created_dirs[key] = dir_path

        return created_dirs

    @staticmethod
    def get_output_path(product_type, filename):
        dir_map = OutputPlanner.OUTPUT_DIRS
        if product_type in dir_map:
            return os.path.join(dir_map[product_type], filename)
        return filename

    @staticmethod
    def apply_layout(insar_app, created_dirs):
        """
        Force StripmapProc output directories into the managed folder tree.
        """
        proc = getattr(insar_app, "insar", None)
        if proc is None:
            return

        for attr, key in OutputPlanner.PROC_DIR_ATTR_MAP.items():
            target = created_dirs.get(key)
            if not target:
                continue
            setattr(proc, attr, target)

        # Keep runtime logs and pickles inside managed folders.
        insar_app.pickleDumpDir = created_dirs.get("pickle", insar_app.pickleDumpDir)
        insar_app.pickleLoadDir = created_dirs.get("pickle", insar_app.pickleLoadDir)

    @staticmethod
    def normalize_sensor_output(sensor, default_name, raw_dir):
        """
        Normalize sensor output prefix so runPreprocessor creates products under
        managed raw/slc trees instead of arbitrary working-directory paths.
        """
        if sensor is None:
            return
        out = getattr(sensor, "output", None)
        if out in [None, ""]:
            out = default_name
        stem = os.path.splitext(os.path.basename(str(out)))[0] or default_name
        sensor.output = os.path.join(raw_dir, stem)


def main():
    import argparse

    def _norm_key(name):
        if not name:
            return ""
        return "".join(str(name).strip().replace("_", " ").split()).lower()

    def _norm_orbit_method(val):
        token = str(val or "").strip().upper()
        alias = {
            "HERMITE": "HERMITE",
            "SCH": "SCH",
            "LEGENDRE": "LEGENDRE",
            "LAGRANGE": "LEGENDRE",
            "0": "HERMITE",
            "1": "SCH",
            "2": "LEGENDRE",
        }
        return alias.get(token)

    def _parse_bool_text(val):
        if isinstance(val, bool):
            return bool(val)
        sval = str(val or "").strip().lower()
        if sval in ("1", "true", "t", "yes", "y", "on"):
            return True
        if sval in ("0", "false", "f", "no", "n", "off"):
            return False
        return None

    def _xml_prop_in_node(node, pname):
        target = _norm_key(pname)
        for p in node.findall("property"):
            n = p.get("name")
            if _norm_key(n) != target:
                continue
            v = p.find("value")
            if v is not None and v.text is not None:
                return v.text.strip()
            if p.text is not None:
                return p.text.strip()
        return None

    def _xml_prop(root, pname):
        # 1) root-level property
        val = _xml_prop_in_node(root, pname)
        if val:
            return val
        # 2) nested under <component name="insar"> for stripmapApp-style XML
        for comp in root.findall("component"):
            if _norm_key(comp.get("name")) == "insar":
                val = _xml_prop_in_node(comp, pname)
                if val:
                    return val
        return None

    def _resolve_parameter_xml(input_path):
        """
        Prefer the original parameter XML over strip_insar.xml output.
        If the caller points at strip_insar.xml, fall back to sibling
        parameter files such as input.xml or stripmapApp.xml.
        """
        xml_path = os.path.abspath(input_path)
        if os.path.basename(xml_path).lower() != "strip_insar.xml":
            return xml_path

        base_dir = os.path.dirname(xml_path)
        candidates = [
            os.path.join(base_dir, "input.xml"),
            os.path.join(base_dir, "stripmapApp.xml"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                logger.info(
                    "启动参数是 strip_insar.xml，改用参数文件: %s",
                    cand,
                )
                return os.path.abspath(cand)

        raise RuntimeError(
            "启动参数不能是 strip_insar.xml；请显式提供原始参数文件（input.xml 或 stripmapApp.xml）。"
        )

    parser = argparse.ArgumentParser(
        description="StripInSAR - 简化的条带InSAR处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  strip_insar.py input.xml
  strip_insar.py --steps input.xml
  strip_insar.py --output-dir /path/to/output input.xml

多视参数示例 (在input.xml中设置):
  <property name='do_multilook'><value>True</value></property>
  <property name='multilookAz'><value>2</value></property>
  <property name='multilookRg'><value>4</value></property>
        """
    )
    parser.add_argument("input_xml", nargs="?", help="输入配置文件路径")
    parser.add_argument("--steps", action="store_true", help="分步骤运行")
    parser.add_argument("--output-dir", "-o", default=None, help="输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--help-steps", action="store_true", help="显示步骤帮助")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.help_steps:
        print("可用步骤: startup, preprocess, cropraw, formslc, cropslc, "
              "multilook, verifyDEM, topo, normalize_secondary_sampling, geo2rdr, "
              "rdrdem_offset, rect_rgoffset, coarse_resample, misregistration, "
              "refined_resample, dense_offsets, rubber_sheet_range, "
              "rubber_sheet_azimuth, fine_resample, split_range_spectrum, "
              "sub_band_resample, interferogram, filter, unwrap, geocode, "
              "geocodeoffsets, endup")
        return 0

    if not args.input_xml:
        print("错误: 请提供输入配置文件")
        parser.print_help()
        return 1

    if not os.path.isfile(args.input_xml):
        logger.error(f"输入文件不存在: {args.input_xml}")
        return 1

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        logger.info(f"输出目录: {os.path.abspath(output_dir)}")

    created_dirs = OutputPlanner.setup_output_structure()

    # Pass XML path to Application parser so facilities (reference/secondary)
    # are materialized from input configuration.
    xml_path = _resolve_parameter_xml(args.input_xml)
    insar = StripInsarApp(name="strip_insar", cmdline=[xml_path])
    insar.configure()
    # Ensure orbit interpolation method is honored from XML even when
    # configurable key binding differs across XML styles.
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        root = None
    if root is not None:
        orbit_val = (
            _xml_prop(root, "orbitInterpolationMethod")
            or _xml_prop(root, "orbit interpolation method")
        )
        orbit_norm = _norm_orbit_method(orbit_val)
        if orbit_norm is not None:
            insar.orbitInterpolationMethod = orbit_norm
            logger.info(
                "轨道插值方法已从XML回填: orbitInterpolationMethod=%s (raw=%r)",
                orbit_norm,
                orbit_val,
            )
        elif orbit_val not in [None, ""]:
            logger.warning(
                "XML中轨道插值方法未识别: %r，保留当前值=%r",
                orbit_val,
                getattr(insar, "orbitInterpolationMethod", None),
            )

        # Unwrap/snaphu config backfill from parameter XML.
        # This keeps strip_insar behavior consistent with stripmapApp-style XML.
        unwrap_val = (
            _xml_prop(root, "unwrapper_name")
            or _xml_prop(root, "unwrapper name")
        )
        if unwrap_val not in [None, ""]:
            insar.unwrapper_name = str(unwrap_val).strip()

        do_unwrap_val = (
            _xml_prop(root, "do_unwrap")
            or _xml_prop(root, "do unwrap")
        )
        do_unwrap_bool = _parse_bool_text(do_unwrap_val)
        if do_unwrap_bool is not None:
            insar.do_unwrap = do_unwrap_bool

        snaphu_corr_val = (
            _xml_prop(root, "snaphuCorrThreshold")
            or _xml_prop(root, "snaphu coherence threshold")
        )
        if snaphu_corr_val not in [None, ""]:
            try:
                insar.snaphuCorrThreshold = float(snaphu_corr_val)
            except Exception:
                logger.warning(
                    "snaphuCorrThreshold 无法解析为浮点数: %r，保留当前值=%r",
                    snaphu_corr_val,
                    getattr(insar, "snaphuCorrThreshold", None),
                )

        snaphu_bool_keys = (
            ("snaphuGmtsarPreprocess", "snaphu gmtsar preprocess"),
            ("snaphuInterpMaskedPhase", "snaphu interpolate masked phase"),
        )
        for key_a, key_b in snaphu_bool_keys:
            raw = _xml_prop(root, key_a) or _xml_prop(root, key_b)
            val = _parse_bool_text(raw)
            if val is not None:
                setattr(insar, key_a, val)

        snaphu_int_keys = (
            ("snaphuInterpRadius", "snaphu interpolation radius"),
            ("snaphuTileNRow", "snaphu tile nrow"),
            ("snaphuTileNCol", "snaphu tile ncol"),
            ("snaphuRowOverlap", "snaphu row overlap"),
            ("snaphuColOverlap", "snaphu col overlap"),
        )
        for key_a, key_b in snaphu_int_keys:
            raw = _xml_prop(root, key_a) or _xml_prop(root, key_b)
            if raw in [None, ""]:
                continue
            try:
                setattr(insar, key_a, int(raw))
            except Exception:
                logger.warning("%s 无法解析为整数: %r", key_a, raw)

    # Always print resolved unwrapping selection at startup for traceability.
    logger.info(
        "最终解缠器选择: do_unwrap=%s, unwrapper_name=%r, "
        "snaphuGmtsarPreprocess=%r, snaphuCorrThreshold=%r, "
        "snaphuInterpMaskedPhase=%r, snaphuInterpRadius=%r, "
        "snaphuTileNRow=%r, snaphuTileNCol=%r, "
        "snaphuRowOverlap=%r, snaphuColOverlap=%r",
        str(getattr(insar, "do_unwrap", None)),
        getattr(insar, "unwrapper_name", None),
        getattr(insar, "snaphuGmtsarPreprocess", None),
        getattr(insar, "snaphuCorrThreshold", None),
        getattr(insar, "snaphuInterpMaskedPhase", None),
        getattr(insar, "snaphuInterpRadius", None),
        getattr(insar, "snaphuTileNRow", None),
        getattr(insar, "snaphuTileNCol", None),
        getattr(insar, "snaphuRowOverlap", None),
        getattr(insar, "snaphuColOverlap", None),
    )
    OutputPlanner.apply_layout(insar, created_dirs)
    # Refresh geocode lists after directory remapping to avoid stale paths
    # (e.g. old "interferogram/" instead of "05_interferogram/").
    insar.geocode_list = list(insar.insar.geocode_list)
    insar.off_geocode_list = list(insar.insar.off_geocode_list)
    insar.quiet_console = (not args.verbose)
    insar.detail_log = os.path.join(created_dirs["log"], "strip_insar.detail.log")
    if insar.quiet_console:
        os.environ.setdefault("CPL_LOG", insar.detail_log)
        os.environ.setdefault("CPL_DEBUG", "OFF")
        os.environ.setdefault("CPL_QUIET", "YES")
        logger.info("详细过程日志: %s", insar.detail_log)
    logger.info(
        "目录映射: ifgDir=%s, geoDir=%s, denseDir=%s, ionoDir=%s",
        getattr(insar.insar, "ifgDirname", None),
        getattr(insar.insar, "geometryDirname", None),
        getattr(insar.insar, "denseOffsetsDirname", None),
        getattr(insar.insar, "ionosphereDirname", None),
    )

    # Fallback: if facilities were not materialized by configure(), recover
    # sensor names from XML and build sensor objects explicitly.
    if insar.reference is None or insar.secondary is None:
        from isceobj.StripmapProc.Sensor import createSensor as _create_sensor

        def _norm_name(s):
            return "".join(str(s or "").split()).lower()

        def _xml_prop_in_node(node, pname):
            target = _norm_name(pname)
            for p in node.findall("property"):
                n = p.get("name")
                if _norm_name(n) != target:
                    continue
                v = p.find("value")
                if v is not None and v.text is not None:
                    return v.text.strip()
                if p.text is not None:
                    return p.text.strip()
            return None

        def _xml_prop(root, pname):
            # 1) root-level property
            val = _xml_prop_in_node(root, pname)
            if val:
                return val

            # 2) nested under <component name="insar"> for insarApp-style XML
            for comp in root.findall("component"):
                if _norm_name(comp.get("name")) == "insar":
                    val = _xml_prop_in_node(comp, pname)
                    if val:
                        return val
            return None

        def _find_component(root, comp_name):
            target = _norm_name(comp_name)
            for comp in root.findall("component"):
                if _norm_name(comp.get("name")) == target:
                    return comp
                # support insarApp style: <component name="insar"><component name="reference">...</component>
                if _norm_name(comp.get("name")) == "insar":
                    for sub in comp.findall("component"):
                        if _norm_name(sub.get("name")) == target:
                            return sub
            return None

        def _read_catalog_props(main_xml, comp_name):
            comp = _find_component(main_xml, comp_name)
            if comp is None:
                logger.info("组件[%s]未在主XML显式声明，使用catalog兜底解析。", comp_name)
                return _read_fallback_catalog_props(comp_name)
            cat_path = None
            cat = comp.find("catalog")
            if cat is not None and cat.text is not None and cat.text.strip():
                cat_path = cat.text.strip()
            else:
                # Compatibility: some XMLs use <property name="catalog">...</property>
                for p in comp.findall("property"):
                    pname = p.get("name")
                    if _norm_name(pname) != "catalog":
                        continue
                    v = p.find("value")
                    if v is not None and v.text is not None and v.text.strip():
                        cat_path = v.text.strip()
                    elif p.text is not None and p.text.strip():
                        cat_path = p.text.strip()
                    break

            if not cat_path:
                logger.info("组件[%s]未找到catalog字段，使用默认catalog兜底。", comp_name)
                return _read_fallback_catalog_props(comp_name)
            if not os.path.isabs(cat_path):
                cat_path = os.path.join(os.path.dirname(xml_path), cat_path)
            if not os.path.isfile(cat_path):
                logger.warning("组件[%s] catalog文件不存在: %s", comp_name, cat_path)
                return {}
            logger.info("组件[%s] 使用catalog: %s", comp_name, cat_path)
            try:
                croot = ET.parse(cat_path).getroot()
            except Exception:
                logger.warning("组件[%s] catalog解析失败: %s", comp_name, cat_path)
                return {}
            props = {}
            for p in croot.findall("property"):
                name = p.get("name")
                if not name:
                    continue
                v = p.find("value")
                val = None
                if v is not None and v.text is not None:
                    val = v.text.strip()
                elif p.text is not None:
                    val = p.text.strip()
                if val:
                    key = _norm_name(name)
                    if key in ("tiff", "xml", "orbitfile", "safe") and not os.path.isabs(val):
                        val = os.path.abspath(os.path.join(os.path.dirname(cat_path), val))
                    props[key] = val
            return props

        def _read_fallback_catalog_props(comp_name):
            """
            Fallback for lightweight strip_insar.xml that omits explicit catalog links.
            """
            names = []
            if _norm_name(comp_name) == "reference":
                names = ["master.xml", "reference.xml"]
            elif _norm_name(comp_name) == "secondary":
                names = ["slave.xml", "secondary.xml"]
            else:
                return {}

            for fname in names:
                cpath = os.path.join(os.path.dirname(xml_path), fname)
                if not os.path.isfile(cpath):
                    continue
                logger.info("组件[%s] 使用默认catalog兜底: %s", comp_name, cpath)
                try:
                    croot = ET.parse(cpath).getroot()
                except Exception:
                    continue
                props = {}
                for p in croot.findall("property"):
                    name = p.get("name")
                    if not name:
                        continue
                    v = p.find("value")
                    val = None
                    if v is not None and v.text is not None:
                        val = v.text.strip()
                    elif p.text is not None:
                        val = p.text.strip()
                    if val:
                        key = _norm_name(name)
                        if key in ("tiff", "xml", "orbitfile", "safe") and not os.path.isabs(val):
                            val = os.path.abspath(os.path.join(os.path.dirname(cpath), val))
                        props[key] = val
                return props
            logger.warning("组件[%s] 默认catalog兜底失败（未找到 master/slave 或 reference/secondary xml）。", comp_name)
            return {}

        def _infer_sensor_from_catalog(ref_props, sec_props):
            """
            Heuristic sensor inference when sensorName is absent in main XML.
            """
            env_sensor = os.environ.get("ISCE_SENSOR_NAME", "").strip()
            if env_sensor:
                return env_sensor

            candidates = []
            tiff_paths = []
            for props in (ref_props or {}, sec_props or {}):
                for key in ("tiff", "xml", "safe"):
                    val = props.get(key)
                    if val:
                        candidates.append(str(val).lower())
                tiff_val = props.get("tiff")
                if tiff_val:
                    tiff_paths.append(str(tiff_val))

            text = " ".join(candidates)
            if ("lutan" in text) or ("lt1" in text):
                return "LUTAN1"
            if ("capella" in text):
                return "capella"
            if (".safe" in text) or ("sentinel" in text):
                return "sentinel1"
            for tiff in tiff_paths:
                base, ext = os.path.splitext(tiff)
                ext_l = ext.lower()
                if ext_l in (".tif", ".tiff"):
                    meta_xml = base + ".meta.xml"
                    if os.path.isfile(meta_xml):
                        return "LUTAN1"
            # Practical fallback for tiff-driven stripmap catalogs used in this workflow.
            if tiff_paths:
                logger.warning(
                    "未显式提供sensorName，且无法精确识别传感器；检测到TIFF输入，默认使用 LUTAN1。"
                    "如需覆盖请设置环境变量 ISCE_SENSOR_NAME。"
                )
                return "LUTAN1"
            return None

        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            root = None

        common = insar.sensorName
        ref_name = insar.referenceSensorName
        sec_name = insar.secondarySensorName
        if root is not None:
            common = common or _xml_prop(root, "sensorName") or _xml_prop(root, "sensor name")
            ref_name = ref_name or _xml_prop(root, "referenceSensorName")
            sec_name = sec_name or _xml_prop(root, "secondarySensorName")

        # Keep app-level sensor names consistent with recovered XML values.
        if common not in [None, ""]:
            insar.sensorName = common
        if ref_name in [None, ""] and common not in [None, ""]:
            ref_name = common
        if sec_name in [None, ""] and common not in [None, ""]:
            sec_name = common
        if ref_name not in [None, ""]:
            insar.referenceSensorName = ref_name
        if sec_name not in [None, ""]:
            insar.secondarySensorName = sec_name

        ref_props = _read_catalog_props(root, "reference") if root is not None else {}
        sec_props = _read_catalog_props(root, "secondary") if root is not None else {}

        if common in [None, ""] and ref_name in [None, ""] and sec_name in [None, ""]:
            guessed = _infer_sensor_from_catalog(ref_props, sec_props)
            if guessed:
                common = guessed
                ref_name = guessed
                sec_name = guessed
                logger.info("未显式提供sensorName，已从catalog推断传感器: %s", guessed)

        if insar.reference is None:
            insar.reference = _create_sensor(common, ref_name, "reference")
        if insar.secondary is None:
            insar.secondary = _create_sensor(common, sec_name, "secondary")

        # Inject catalog values for manual sensor creation path.
        # This keeps compatibility with master/secondary XML used by insarmapApp.
        if insar.reference is not None:
            if ref_props.get("tiff"):
                insar.reference.tiff = ref_props.get("tiff")
            if ref_props.get("xml"):
                insar.reference.xml = ref_props.get("xml")
            if ref_props.get("safe"):
                insar.reference.safe = ref_props.get("safe")
            if ref_props.get("orbitfile"):
                insar.reference.orbitFile = ref_props.get("orbitfile")
            if ref_props.get("output"):
                insar.reference.output = ref_props.get("output")
            elif not getattr(insar.reference, "output", None):
                insar.reference.output = "reference"

        if insar.secondary is not None:
            if sec_props.get("tiff"):
                insar.secondary.tiff = sec_props.get("tiff")
            if sec_props.get("xml"):
                insar.secondary.xml = sec_props.get("xml")
            if sec_props.get("safe"):
                insar.secondary.safe = sec_props.get("safe")
            if sec_props.get("orbitfile"):
                insar.secondary.orbitFile = sec_props.get("orbitfile")
            if sec_props.get("output"):
                insar.secondary.output = sec_props.get("output")
            elif not getattr(insar.secondary, "output", None):
                insar.secondary.output = "secondary"

    # Normalize sensor output prefixes into managed raw directory.
    OutputPlanner.normalize_sensor_output(insar.reference, "reference", created_dirs["raw"])
    OutputPlanner.normalize_sensor_output(insar.secondary, "secondary", created_dirs["raw"])
    # Preprocess input sanity for Lutan1-like catalogs.
    for role, sensor in (("reference", insar.reference), ("secondary", insar.secondary)):
        if sensor is None:
            continue
        tiff = getattr(sensor, "tiff", None)
        safe = getattr(sensor, "safe", None)
        xmlf = getattr(sensor, "xml", None)
        logger.info(
            "传感器输入检查[%s]: tiff=%r, xml=%r, safe=%r",
            role, tiff, xmlf, safe
        )
        if (not tiff) and (not safe):
            logger.error(
                "传感器输入缺失[%s]: 需要在catalog中提供 tiff（推荐）或 safe。", role
            )
            return 2

    if insar.reference is None or insar.secondary is None:
        logger.error(
            "未能从XML实例化 reference/secondary 组件。"
            "请检查输入XML是否包含 <component name=\"reference\"> 与 "
            "<component name=\"secondary\">，以及 sensorName/referenceSensorName/"
            "secondarySensorName 配置。"
        )
        logger.error(
            "诊断: sensorName=%r, referenceSensorName=%r, secondarySensorName=%r",
            getattr(insar, "sensorName", None),
            getattr(insar, "referenceSensorName", None),
            getattr(insar, "secondarySensorName", None),
        )
        logger.error(
            "可尝试: ISCE_SENSOR_NAME=LUTAN1 strip_insar.py %s",
            os.path.basename(args.input_xml),
        )
        return 2

    # Final guard: keep sensor names consistent before entering processing.
    if getattr(insar, "sensorName", None):
        sname = str(insar.sensorName).strip().lower()
        if sname in ("lutan1", "lutan_1", "lt1"):
            insar.sensorName = "lutan1"
        if not getattr(insar, "referenceSensorName", None):
            insar.referenceSensorName = insar.sensorName
        if not getattr(insar, "secondarySensorName", None):
            insar.secondarySensorName = insar.sensorName
    if str(getattr(insar, "referenceSensorName", "")).lower() == "lutan1" and not getattr(insar, "referenceDopplerMethod", None):
        insar.referenceDopplerMethod = "useDEFAULT"
    if str(getattr(insar, "secondarySensorName", "")).lower() == "lutan1" and not getattr(insar, "secondaryDopplerMethod", None):
        insar.secondaryDopplerMethod = "useDEFAULT"

    if args.steps:
        insar.run()
    else:
        insar._init()
        insar.main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
