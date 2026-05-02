#!/usr/bin/env python3

import os
import sys
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import isce
import isceobj
from iscesys.Component.Application import Application
from isceobj.StripmapProc import StripmapProc
from isceobj.Scene.Frame import FrameMixin

logger = logging.getLogger("strip_insar")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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

SNAPHU_CORR_THRESHOLD = Application.Parameter(
    "snaphuCorrThreshold",
    public_name="snaphu coherence threshold",
    default=0.20,
    type=float,
    mandatory=False,
    doc="Coherence threshold used for snaphu preprocessing.",
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

DO_DENSEOFFSETS = Application.Parameter(
    "doDenseOffsets",
    public_name="do denseoffsets",
    default=False,
    type=bool,
    mandatory=False,
    doc="Run dense offsets.",
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
    args=(Application.SELF(), DO_UNWRAP, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module",
)

RUN_UNWRAP_2STAGE = Application.Facility(
    "runUnwrap2Stage",
    public_name="Run unwrapper 2 Stage",
    module="isceobj.TopsProc",
    factory="createUnwrap2Stage",
    args=(Application.SELF(), False, UNWRAPPER_NAME),
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
        CORRELATION_METHOD,
        REFERENCE_DOPPLER_METHOD,
        SECONDARY_DOPPLER_METHOD,
        ORBIT_INTERPOLATION_METHOD,
        UNWRAPPER_NAME,
        DO_UNWRAP,
        SNAPHU_CORR_THRESHOLD,
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
        super().__init__(family=family, name=name, cmdline=cmdline)

        from isceobj.StripmapProc import StripmapProc
        from iscesys.StdOEL.StdOELPy import create_writer

        self._stdWriter = create_writer("log", "", True, filename="strip_insar.log")
        self._add_methods()
        self._insarProcFact = StripmapProc
        self.timeStart = None
        self.work_dir = os.path.abspath(os.getcwd())

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
        logger.info("=" * 60)

    def renderProcDoc(self):
        self.procDoc.renderXml()


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


def main():
    import argparse

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

    OutputPlanner.setup_output_structure()

    insar = StripInsarApp(name="strip_insar")
    insar.configure()

    if args.steps:
        insar.run()
    else:
        insar._init()
        insar.main()

    return 0


if __name__ == "__main__":
    sys.exit(main())