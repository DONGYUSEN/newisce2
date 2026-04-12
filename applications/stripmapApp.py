#!/usr/bin/env python3
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright by California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Heresh Fattahi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






from __future__ import print_function
import time
import sys
import os
import xml.etree.ElementTree as ET
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
import isceobj.StripmapProc as StripmapProc
from isceobj.Scene.Frame import FrameMixin
from isceobj.Util.decorators import use_api

try:
    from isce2.applications.postprocess_hook import run_auto_postprocess
except ImportError:
    try:
        from applications.postprocess_hook import run_auto_postprocess
    except ImportError:
        from postprocess_hook import run_auto_postprocess

logger = logging.getLogger('isce.insar')


def _autodetect_wavelength_for_postprocess(insar_obj):
    """Try to read radar wavelength from stripmap product XMLs."""
    if insar_obj is None:
        return None

    product_attrs = (
        "referenceSlcCropProduct",
        "referenceSlcProduct",
        "referenceRawProduct",
        "secondarySlcCropProduct",
        "secondarySlcProduct",
        "secondaryRawProduct",
    )
    xml_candidates = []
    for attr in product_attrs:
        try:
            xmlname = getattr(insar_obj, attr, None)
        except Exception:
            xmlname = None
        if xmlname:
            xml_candidates.append(xmlname)

    seen = set()
    for xmlname in xml_candidates:
        if xmlname in seen:
            continue
        seen.add(xmlname)

        try:
            prod = insar_obj.loadProduct(xmlname)
        except Exception:
            continue

        if prod is None:
            continue

        inst = None
        try:
            inst = prod.getInstrument() if hasattr(prod, "getInstrument") else getattr(prod, "instrument", None)
        except Exception:
            inst = None
        if inst is None:
            continue

        wvl = None
        try:
            getter = getattr(inst, "getRadarWavelength", None)
            if callable(getter):
                wvl = getter()
            else:
                wvl = getattr(inst, "radarWavelength", None)
            if wvl is not None:
                wvl = float(wvl)
        except Exception:
            wvl = None

        if (wvl is not None) and (0.001 < wvl < 1.0):
            return wvl

    return None


def _ensure_postprocess_utm_args(post_args):
    extras = []
    current = str(post_args or "")

    if "--to-utm" not in current:
        extras.append("--to-utm")
    if "--utm-res-mode" not in current:
        extras.extend(["--utm-res-mode", "multilook"])
    if "--square-pixel" not in current:
        extras.append("--square-pixel")

    if not extras:
        return current
    if current:
        return (current + " " + " ".join(extras)).strip()
    return " ".join(extras)


def _safe_bool_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 'yes', 'on', 'y', 't'):
        return True
    if sval in ('0', 'false', 'no', 'off', 'n', 'f', 'none', 'null', ''):
        return False
    return default


def _find_input_xml_from_cmdline(cmdline):
    if not cmdline:
        return None

    for token in cmdline:
        if not isinstance(token, str):
            continue
        if token.startswith('-'):
            continue
        if token.lower().endswith('.xml') and os.path.isfile(token):
            return token
    return None


def _read_root_xml_property(xml_path, property_name):
    """
    Read a top-level XML property directly under <stripmapApp>.
    """
    if (not xml_path) or (not os.path.isfile(xml_path)):
        return None

    norm = ''.join(str(property_name).split()).lower()
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    for node in root.findall('property'):
        pname = node.get('name')
        if pname is None:
            name_node = node.find('name')
            pname = (name_node.text if name_node is not None else None)
        if pname is None:
            continue
        if ''.join(str(pname).split()).lower() != norm:
            continue

        value_node = node.find('value')
        if (value_node is not None) and (value_node.text is not None):
            return value_node.text.strip()
        if node.text is not None:
            return node.text.strip()
        return None

    return None


def _read_component_xml_property(xml_path, component_name, property_name):
    """
    Read a property under a named top-level component in stripmapApp XML.
    """
    if (not xml_path) or (not os.path.isfile(xml_path)):
        return None

    comp_norm = ''.join(str(component_name).split()).lower()
    prop_norm = ''.join(str(property_name).split()).lower()
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    for comp in root.findall('component'):
        cname = comp.get('name')
        if cname is None:
            name_node = comp.find('name')
            cname = (name_node.text if name_node is not None else None)
        if cname is None:
            continue
        if ''.join(str(cname).split()).lower() != comp_norm:
            continue

        for node in comp.findall('property'):
            pname = node.get('name')
            if pname is None:
                name_node = node.find('name')
                pname = (name_node.text if name_node is not None else None)
            if pname is None:
                continue
            if ''.join(str(pname).split()).lower() != prop_norm:
                continue

            value_node = node.find('value')
            if (value_node is not None) and (value_node.text is not None):
                return value_node.text.strip()
            if node.text is not None:
                return node.text.strip()
            return None

    return None


SENSOR_NAME = Application.Parameter(
        'sensorName',
        public_name='sensor name',
        default = None,
        type = str,
        mandatory = False,
        doc = 'Sensor name for both reference and secondary')


REFERENCE_SENSOR_NAME = Application.Parameter(
        'referenceSensorName',
        public_name='reference sensor name',
        default = None,
        type=str,
        mandatory = True,
        doc = "Reference sensor name if mixing sensors")

SECONDARY_SENSOR_NAME = Application.Parameter(
        'secondarySensorName',
        public_name='secondary sensor name',
        default = None,
        type=str,
        mandatory = True,
        doc = "Secondary sensor name if mixing sensors")


CORRELATION_METHOD = Application.Parameter(
   'correlation_method',
   public_name='correlation_method',
   default='cchz_wave',
   type=str,
   mandatory=False,
   doc=(
   """Select coherence estimation method:
      cchz=cchz_wave
      phase_gradient=phase gradient"""
        )
                                           )
REFERENCE_DOPPLER_METHOD = Application.Parameter(
    'referenceDopplerMethod',
    public_name='reference doppler method',
    default=None,
    type=str, mandatory=False,
    doc= "Doppler calculation method.Choices: 'useDOPIQ', 'useDefault'."
)

SECONDARY_DOPPLER_METHOD = Application.Parameter(
    'secondaryDopplerMethod',
    public_name='secondary doppler method',
    default=None,
    type=str, mandatory=False,
    doc="Doppler calculation method. Choices: 'useDOPIQ','useDefault'.")

ORBIT_INTERPOLATION_METHOD = Application.Parameter(
    'orbitInterpolationMethod',
    public_name='orbit interpolation method',
    default='HERMITE',
    type=str,
    mandatory=False,
    doc="Orbit interpolation method for topo/geo2rdr. Choices: HERMITE, SCH, LEGENDRE (alias: LAGRANGE->LEGENDRE).",
)


UNWRAPPER_NAME = Application.Parameter(
    'unwrapper_name',
    public_name='unwrapper name',
    default='grass',
    type=str,
    mandatory=False,
    doc="Unwrapping method to use. To be used in  combination with UNWRAP."
)

DO_UNWRAP = Application.Parameter(
    'do_unwrap',
    public_name='do unwrap',
    default=True,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be unsed in combination with UNWRAPPER_NAME."
)

SNAPHU_GMTSAR_PREPROCESS = Application.Parameter(
    'snaphuGmtsarPreprocess',
    public_name='snaphu gmtsar preprocess',
    default=True,
    type=bool,
    mandatory=False,
    doc='Enable GMTSAR-style coherence-mask preprocessing before snaphu unwrapping.'
)

SNAPHU_CORR_THRESHOLD = Application.Parameter(
    'snaphuCorrThreshold',
    public_name='snaphu coherence threshold',
    default=0.10,
    type=float,
    mandatory=False,
    doc='Coherence threshold used for snaphu GMTSAR-style preprocessing.'
)

SNAPHU_INTERP_MASKED_PHASE = Application.Parameter(
    'snaphuInterpMaskedPhase',
    public_name='snaphu interpolate masked phase',
    default=False,
    type=bool,
    mandatory=False,
    doc='Interpolate masked wrapped phase before snaphu (GMTSAR interp-style).'
)

SNAPHU_INTERP_RADIUS = Application.Parameter(
    'snaphuInterpRadius',
    public_name='snaphu interpolation radius',
    default=300,
    type=int,
    mandatory=False,
    doc='Interpolation search radius in pixels for masked wrapped phase.'
)

DO_UNWRAP_2STAGE = Application.Parameter(
    'do_unwrap_2stage',
    public_name='do unwrap 2 stage',
    default=False,
    type=bool,
    mandatory=False,
    doc="True if unwrapping is desired. To be unsed in combination with UNWRAPPER_NAME."
)

UNWRAPPER_2STAGE_NAME = Application.Parameter(
    'unwrapper_2stage_name',
    public_name='unwrapper 2stage name',
    default='REDARC0',
    type=str,
    mandatory=False,
    doc="2 Stage Unwrapping method to use. Available: MCF, REDARC0, REDARC1, REDARC2"
)

SOLVER_2STAGE = Application.Parameter(
    'solver_2stage',
    public_name='SOLVER_2STAGE',
    default='pulp',
    type=str,
    mandatory=False,
    doc='Linear Programming Solver for 2Stage; Options: pulp, gurobi, glpk; Used only for Redundant Arcs'
)

USE_HIGH_RESOLUTION_DEM_ONLY = Application.Parameter(
    'useHighResolutionDemOnly',
    public_name='useHighResolutionDemOnly',
    default=False,
    type=int,
    mandatory=False,
    doc=(
    """If True and a dem is not specified in input, it will only
    download the SRTM highest resolution dem if it is available
    and fill the missing portion with null values (typically -32767).
    若为 True 且未在输入中指定 DEM，仅下载可用的最高分辨率 SRTM，
    缺失区域用空值（通常 -32767）填充。"""
    )
)

DEM_FILENAME = Application.Parameter(
     'demFilename',
     public_name='demFilename',
     default='',
     type=str,
     mandatory=False,
     doc="Filename of the DEM init file / DEM 初始化文件路径"
)

REGION_OF_INTEREST = Application.Parameter(
        'regionOfInterest',
        public_name = 'regionOfInterest',
        default = None,
        container = list,
        type = float,
        doc = 'Region of interest - South, North, West, East in degrees / 处理区域：南北西东（度）')


GEOCODE_BOX = Application.Parameter(
    'geocode_bbox',
    public_name='geocode bounding box',
    default = None,
    container=list,
    type=float,
    doc='Bounding box for geocoding - South, North, West, East in degrees / 地理编码范围：南北西东（度）'
                                    )

GEO_POSTING = Application.Parameter(
    'geoPosting',
    public_name='geoPosting',
    default=None,
    type=float,
    mandatory=False,
    doc=(
    "Output posting for geocoded images in degrees (latitude = longitude) / 地理编码输出分辨率（度，纬经相同）"
    )
                                    )

POSTING = Application.Parameter(
    'posting',
    public_name='posting',
    default=30,
    type=int,
    mandatory=False,
    doc="posting for interferogram / 干涉图目标像元间距")


NUMBER_RANGE_LOOKS = Application.Parameter(
    'numberRangeLooks',
    public_name='range looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of range looks / 距离向多视数'
                                    )

NUMBER_AZIMUTH_LOOKS = Application.Parameter(
    'numberAzimuthLooks',
    public_name='azimuth looks',
    default=None,
    type=int,
    mandatory=False,
    doc='Number of azimuth looks / 方位向多视数'
                                 )

RANGE_CROP_FAR_PIXELS = Application.Parameter(
    'rangeCropFarPixels',
    public_name='rangeCropFarPixels',
    default=None,
    type=int,
    mandatory=False,
    doc='Crop N pixels at far-range edge for sensors that support it (e.g., Lutan1) / 对支持该参数的传感器（如 Lutan1）在远距端裁剪像素数'
)

FILTER_STRENGTH = Application.Parameter('filterStrength',
                                      public_name='filter strength',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='')

USE_GPU = Application.Parameter(
    'useGPU',
    public_name='use GPU',
    default=True,
    type=bool,
    mandatory=False,
    doc='Prefer GPU-enabled processing where stripmap pipeline supports it. / 在 stripmap 支持的步骤优先使用 GPU。'
)

USE_EXTERNAL_COREGISTRATION = Application.Parameter(
    'useExternalCoregistration',
    public_name='use external coregistration',
    default=False,
    type=bool,
    mandatory=False,
    doc='Enable integrated external coregistration-assisted template/initial-offset selection in refine-secondary-timing.'
)

############################################## Modified by V.Brancato 10.07.2019
DO_RUBBERSHEETINGAZIMUTH = Application.Parameter('doRubbersheetingAzimuth', 
                                      public_name='do rubbersheetingAzimuth',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')
DO_RUBBERSHEETINGRANGE = Application.Parameter('doRubbersheetingRange', 
                                      public_name='do rubbersheetingRange',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')
#################################################################################
RUBBERSHEET_SNR_THRESHOLD = Application.Parameter('rubberSheetSNRThreshold',
                                      public_name='rubber sheet SNR Threshold',
                                      default = 5.0,
                                      type = float,
                                      mandatory = False,
                                      doc='')

RUBBERSHEET_FILTER_SIZE = Application.Parameter('rubberSheetFilterSize',
                                      public_name='rubber sheet filter size',
                                      default = 9,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DO_DENSEOFFSETS  = Application.Parameter('doDenseOffsets',
                                      public_name='do denseoffsets',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')

ENABLE_RDRDEM_OFFSET_LOOP = Application.Parameter(
    'enableRdrdemOffsetLoop',
    public_name='enable rdrdem offset loop',
    default=True,
    type=bool,
    mandatory=False,
    doc='Enable rdrdem_offset + rect_rgoffset loop for range.off geometric closure.'
)

REFINE_TIMING_AZIMUTH_AZIMUTH_ORDER = Application.Parameter(
    'refineTimingAzimuthAzimuthOrder',
    public_name='refine timing azimuth-azimuth order',
    default=0,
    type=int,
    mandatory=False,
    doc='Fallback refine-secondary-timing polynomial azimuth order for azimuth offsets.'
)

REFINE_TIMING_AZIMUTH_RANGE_ORDER = Application.Parameter(
    'refineTimingAzimuthRangeOrder',
    public_name='refine timing azimuth-range order',
    default=0,
    type=int,
    mandatory=False,
    doc='Fallback refine-secondary-timing polynomial range order for azimuth offsets.'
)

REFINE_TIMING_RANGE_AZIMUTH_ORDER = Application.Parameter(
    'refineTimingRangeAzimuthOrder',
    public_name='refine timing range-azimuth order',
    default=0,
    type=int,
    mandatory=False,
    doc='Fallback refine-secondary-timing polynomial azimuth order for range offsets.'
)

REFINE_TIMING_RANGE_RANGE_ORDER = Application.Parameter(
    'refineTimingRangeRangeOrder',
    public_name='refine timing range-range order',
    default=0,
    type=int,
    mandatory=False,
    doc='Fallback refine-secondary-timing polynomial range order for range offsets.'
)

REFINE_TIMING_SNR_THRESHOLD = Application.Parameter(
    'refineTimingSnrThreshold',
    public_name='refine timing SNR threshold',
    default=1.2,
    type=float,
    mandatory=False,
    doc='Fallback refine-secondary-timing SNR threshold used by offoutliers.'
)

DENSE_WINDOW_WIDTH = Application.Parameter('denseWindowWidth',
                                      public_name='dense window width',
                                      default=96,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_WINDOW_HEIGHT = Application.Parameter('denseWindowHeight',
                                      public_name='dense window height',
                                      default=96,
                                      type = int,
                                      mandatory = False,
                                      doc = '')


DENSE_SEARCH_WIDTH = Application.Parameter('denseSearchWidth',
                                      public_name='dense search width',
                                      default=48,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SEARCH_HEIGHT = Application.Parameter('denseSearchHeight',
                                      public_name='dense search height',
                                      default=48,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SKIP_WIDTH = Application.Parameter('denseSkipWidth',
                                      public_name='dense skip width',
                                      default=32,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DENSE_SKIP_HEIGHT = Application.Parameter('denseSkipHeight',
                                      public_name='dense skip height',
                                      default=32,
                                      type = int,
                                      mandatory = False,
                                      doc = '')

DO_SPLIT_SPECTRUM = Application.Parameter('doSplitSpectrum',
                                      public_name='do split spectrum',
                                      default = False,
                                      type = bool,
                                      mandatory = False,
                                      doc = '')

DO_DISPERSIVE = Application.Parameter('doDispersive',
                                      public_name='do dispersive',
                                      default=False,
                                      type=bool,
                                      mandatory=False,
                                      doc='')

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
     public_name='geocode list',
     default = None,
     container=list,
     type=str,
     doc = "List of products to geocode."
                                      )

OFFSET_GEOCODE_LIST = Application.Parameter(
        'off_geocode_list',
        public_name='offset geocode list',
        default=None,
        container=list,
        mandatory=False,
        doc='List of offset-specific files to geocode')

HEIGHT_RANGE = Application.Parameter(
        'heightRange',
        public_name = 'height range',
        default = None,
        container = list,
        type = float,
        doc = 'Altitude range in scene for cropping')

PICKLE_DUMPER_DIR = Application.Parameter(
    'pickleDumpDir',
    public_name='pickle dump directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc=(
    "If steps is used, the directory in which to store pickle objects."
    )
                                          )
PICKLE_LOAD_DIR = Application.Parameter(
    'pickleLoadDir',
    public_name='pickle load directory',
    default='PICKLE',
    type=str,
    mandatory=False,
    doc=(
    "If steps is used, the directory from which to retrieve pickle objects."
    )
                                        )

RENDERER = Application.Parameter(
    'renderer',
    public_name='renderer',
    default='xml',
    type=str,
    mandatory=True,
    doc=(
    "Format in which the data is serialized when using steps. Options are xml (default) or pickle."
    )
                                        )

DISPERSIVE_FILTER_FILLING_METHOD = Application.Parameter('dispersive_filling_method',
                                            public_name = 'dispersive filter filling method',
                                            default='nearest_neighbour',
                                            type=str,
                                            mandatory=False,
                                            doc='method to fill the holes left by masking the ionospheric phase estimate')
					    
DISPERSIVE_FILTER_KERNEL_XSIZE = Application.Parameter('kernel_x_size',
                                      public_name='dispersive filter kernel x-size',
                                      default=800,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel x-size for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_YSIZE = Application.Parameter('kernel_y_size',
                                      public_name='dispersive filter kernel y-size',
                                      default=800,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel y-size for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_SIGMA_X = Application.Parameter('kernel_sigma_x',
                                      public_name='dispersive filter kernel sigma_x',
                                      default=100,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel sigma_x for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_SIGMA_Y = Application.Parameter('kernel_sigma_y',
                                      public_name='dispersive filter kernel sigma_y',
                                      default=100,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel sigma_y for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_KERNEL_ROTATION = Application.Parameter('kernel_rotation',
                                      public_name='dispersive filter kernel rotation',
                                      default=0.0,
                                      type=float,
                                      mandatory=False,
                                      doc='kernel rotation angle for the Gaussian low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_ITERATION_NUMBER = Application.Parameter('dispersive_filter_iterations',
                                      public_name='dispersive filter number of iterations',
                                      default=5,
                                      type=int,
                                      mandatory=False,
                                      doc='number of iterations for the iterative low-pass filtering of the dispersive and non-disperive phase')

DISPERSIVE_FILTER_MASK_TYPE = Application.Parameter('dispersive_filter_mask_type',
                                      public_name='dispersive filter mask type',
                                      default="connected_components",
                                      type=str,
                                      mandatory=False,
                                      doc='The type of mask for the iterative low-pass filtering of the estimated dispersive phase. If method is coherence, then a mask based on coherence files of low-band and sub-band interferograms is generated using the mask coherence thresold which can be also setup. If method is connected_components, then mask is formed based on connected component files with non zero values. If method is phase, then pixels with zero phase values in unwrapped sub-band interferograms are masked out.')

DISPERSIVE_FILTER_COHERENCE_THRESHOLD = Application.Parameter('dispersive_filter_coherence_threshold',
                                      public_name='dispersive filter coherence threshold',
                                      default=0.5,
                                      type=float,
                                      mandatory=False,
                                      doc='Coherence threshold to generate a mask file which gets used in the iterative filtering of the dispersive and non-disperive phase')
#Facility declarations

REFERENCE = Application.Facility(
    'reference',
    public_name='Reference',
    module='isceobj.StripmapProc.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME, REFERENCE_SENSOR_NAME, 'reference'),
    mandatory=False,
    doc="Reference raw data component"
                              )

SECONDARY = Application.Facility(
    'secondary',
    public_name='Secondary',
    module='isceobj.StripmapProc.Sensor',
    factory='createSensor',
    args=(SENSOR_NAME, SECONDARY_SENSOR_NAME,'secondary'),
    mandatory=False,
    doc="Secondary raw data component"
                             )

DEM_STITCHER = Application.Facility(
    'demStitcher',
    public_name='demStitcher',
    module='iscesys.DataManager',
    factory='createManager',
    args=('dem1','iscestitcher',),
    mandatory=False,
    doc="Object that based on the frame bounding boxes creates a DEM"
)

RUN_UNWRAPPER = Application.Facility(
    'runUnwrapper',
    public_name='Run unwrapper',
    module='isceobj.StripmapProc',
    factory='createUnwrapper',
    args=(SELF(), DO_UNWRAP, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

RUN_UNWRAP_2STAGE = Application.Facility(
    'runUnwrap2Stage',
    public_name='Run unwrapper 2 Stage',
    module='isceobj.TopsProc',
    factory='createUnwrap2Stage',
    args=(SELF(), DO_UNWRAP_2STAGE, UNWRAPPER_NAME),
    mandatory=False,
    doc="Unwrapping module"
)

_INSAR = Application.Facility(
    '_insar',
    public_name='insar',
    module='isceobj.StripmapProc',
    factory='createStripmapProc',
    args = ('stripmapAppContext',isceobj.createCatalog('stripmapProc')),
    mandatory=False,
    doc="InsarProc object"
)



## Common interface for stripmap insar applications.
class _RoiBase(Application, FrameMixin):

    family = 'insar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
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
                      DO_UNWRAP_2STAGE,
                      UNWRAPPER_2STAGE_NAME,
                      SOLVER_2STAGE,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      DEM_FILENAME,
                      GEO_POSTING,
                      POSTING,
                      NUMBER_RANGE_LOOKS,
                      NUMBER_AZIMUTH_LOOKS,
                      RANGE_CROP_FAR_PIXELS,
                      GEOCODE_LIST,
                      OFFSET_GEOCODE_LIST,
                      GEOCODE_BOX,
                      REGION_OF_INTEREST,
                      HEIGHT_RANGE,
                      DO_RUBBERSHEETINGRANGE, #Modified by V. Brancato 10.07.2019
                      DO_RUBBERSHEETINGAZIMUTH,  #Modified by V. Brancato 10.07.2019
                      RUBBERSHEET_SNR_THRESHOLD,
                      RUBBERSHEET_FILTER_SIZE,
                      DO_DENSEOFFSETS,
                      ENABLE_RDRDEM_OFFSET_LOOP,
                      REFINE_TIMING_AZIMUTH_AZIMUTH_ORDER,
                      REFINE_TIMING_AZIMUTH_RANGE_ORDER,
                      REFINE_TIMING_RANGE_AZIMUTH_ORDER,
                      REFINE_TIMING_RANGE_RANGE_ORDER,
                      REFINE_TIMING_SNR_THRESHOLD,
                      DENSE_WINDOW_WIDTH,
                      DENSE_WINDOW_HEIGHT,
                      DENSE_SEARCH_WIDTH,
                      DENSE_SEARCH_HEIGHT,
                      DENSE_SKIP_WIDTH,
                      DENSE_SKIP_HEIGHT,
                      DO_SPLIT_SPECTRUM,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      RENDERER,
                      DO_DISPERSIVE,
                      DISPERSIVE_FILTER_FILLING_METHOD,
                      DISPERSIVE_FILTER_KERNEL_XSIZE,
                      DISPERSIVE_FILTER_KERNEL_YSIZE,
                      DISPERSIVE_FILTER_KERNEL_SIGMA_X,
                      DISPERSIVE_FILTER_KERNEL_SIGMA_Y,
                      DISPERSIVE_FILTER_KERNEL_ROTATION,
                      DISPERSIVE_FILTER_ITERATION_NUMBER,
                      DISPERSIVE_FILTER_MASK_TYPE,
                      DISPERSIVE_FILTER_COHERENCE_THRESHOLD)

    facility_list = (REFERENCE,
                     SECONDARY,
                     DEM_STITCHER,
                     RUN_UNWRAPPER,
                     RUN_UNWRAP_2STAGE,
                     _INSAR)


    _pickleObj = "_insar"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        super().__init__(family=family, name=name,
            cmdline=cmdline)

        from isceobj.StripmapProc import StripmapProc
        from iscesys.StdOEL.StdOELPy import create_writer
        self._stdWriter = create_writer("log", "", True, filename="roi.log")
        self._add_methods()
        self._insarProcFact = StripmapProc
        self.timeStart = None
        return None

    def Usage(self):
        print("Usages / 用法: ")
        print("stripmapApp.py <input-file.xml>   # 使用配置文件运行")
        print("stripmapApp.py --steps            # 分步骤运行")
        print("stripmapApp.py --help             # 查看帮助")
        print("stripmapApp.py --help --steps     # 查看步骤帮助")

    def _init(self):

        message =  (
            ("ISCE VERSION = %s, RELEASE_SVN_REVISION = %s,"+
             "RELEASE_DATE = %s, CURRENT_SVN_REVISION = %s") %
            (isce.__version__,
             isce.release_svn_revision,
             isce.release_date,
             isce.svn_revision)
            )
        logger.info(message)

        print(message)
        return None

    ## You need this to use the FrameMixin
    @property
    def frame(self):
        return self.insar.frame


    def _configure(self):

        self.insar.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["stripmapProc"]
            )

        #Ensure consistency in geocode_list maintained by insarApp and
        #InsarProc. If it is configured in both places, the one in insarApp
        #will be used. It is complicated to try to merge the two lists
        #because InsarProc permits the user to change the name of the files
        #and the linkage between filename and filetype is lost by the time
        #geocode_list is fully configured.  In order to safely change file
        #names and also specify the geocode_list, then insarApp should not
        #be given a geocode_list from the user.
        if(self.geocode_list is not None):
            #if geocode_list defined here, then give it to InsarProc
            #for consistency between insarApp and InsarProc and warn the user

            #check if the two geocode_lists differ in content
            g_count = 0
            for g in self.geocode_list:
                if g not in self.insar.geocode_list:
                    g_count += 1

            #warn if there are any differences in content
            if g_count > 0:
                print()
                logger.warning((
                    "Some filenames in stripmapApp.geocode_list configuration "+
                    "are different from those in StripmapProc. Using names given"+
                    " to stripmapApp. / stripmapApp.geocode_list 与 StripmapProc 不一致，采用 stripmapApp 配置。"))
                print("stripmapApp.geocode_list = {}".format(self.geocode_list))
        else:
            self.geocode_list = self.insar.geocode_list


        if (self.off_geocode_list is None):
            self.off_geocode_list = self.insar.off_geocode_list
        else:
            g_count = 0
            for g in self.off_geocode_list:
                if g not in self.insar.off_geocode_list:
                    g_count += 1

            if g_count > 0:
                self.off_geocode_list = self.insar.off_geocode_list

        # Compatibility: honor legacy top-level <property name="useGPU">.
        # Some XMLs place this switch at root instead of inside component "insar".
        xml_path = _find_input_xml_from_cmdline(getattr(self, 'cmdline', None))
        root_use_gpu = _read_root_xml_property(xml_path, 'useGPU')
        if root_use_gpu is None:
            root_use_gpu = _read_root_xml_property(xml_path, 'use GPU')
        parsed_root_use_gpu = _safe_bool_value(root_use_gpu, default=None)
        if parsed_root_use_gpu is not None:
            self.useGPU = bool(parsed_root_use_gpu)
            logger.info(
                'Applying top-level XML useGPU=%s from %s',
                str(self.useGPU), str(xml_path)
            )

        # Compatibility: allow app-level rangeCropFarPixels in stripmapApp XML.
        # Guard against stale/default values: only apply to sensors when explicitly
        # configured in the current input XML.
        xml_crop = _read_component_xml_property(xml_path, 'insar', 'rangeCropFarPixels')
        if xml_crop is None:
            xml_crop = _read_root_xml_property(xml_path, 'rangeCropFarPixels')
        crop_from_xml = False
        if xml_crop is not None:
            try:
                self.rangeCropFarPixels = int(str(xml_crop).strip())
                crop_from_xml = True
            except Exception:
                logger.warning(
                    'Invalid rangeCropFarPixels value in XML (%s); ignoring.',
                    str(xml_crop)
                )
                self.rangeCropFarPixels = None
        elif (xml_path is not None) and (self.rangeCropFarPixels is not None):
            logger.info(
                'stripmapApp.rangeCropFarPixels=%s is not explicitly set in %s; '
                'skip applying far-range crop to sensors.',
                str(self.rangeCropFarPixels), str(xml_path)
            )
            self.rangeCropFarPixels = None

        if self.rangeCropFarPixels is not None:
            for role, sensor in (('reference', self.reference), ('secondary', self.secondary)):
                if hasattr(sensor, 'rangeCropFarPixels'):
                    sensor.rangeCropFarPixels = int(self.rangeCropFarPixels)
                    logger.info(
                        'Applying stripmapApp.rangeCropFarPixels=%d to %s sensor%s.',
                        int(self.rangeCropFarPixels),
                        role,
                        (' (from XML)' if crop_from_xml else '')
                    )
                else:
                    logger.warning(
                        'stripmapApp.rangeCropFarPixels is set, but %s sensor does not support it; ignored.',
                        role
                    )

        # Use one switch to control the full rdrdem_offset + rect_rgoffset
        # closure path and its downstream usage.
        self.enableRdrdemOffsetLoop = bool(getattr(self, 'enableRdrdemOffsetLoop', True))
        self.useRdrdemRectRangeOffset = bool(self.enableRdrdemOffsetLoop)
        logger.info(
            'Range-off closure switches: enableRdrdemOffsetLoop=%s, useRdrdemRectRangeOffset=%s',
            str(self.enableRdrdemOffsetLoop),
            str(self.useRdrdemRectRangeOffset),
        )

        return None

    @property
    def insar(self):
        return self._insar
    @insar.setter
    def insar(self, value):
        self._insar = value
        return None

    @property
    def procDoc(self):
        return self.insar.procDoc
    @procDoc.setter
    def procDoc(self):
        raise AttributeError(
            "Can not assign to .insar.procDoc-- but you hit all its other stuff"
            )

    def _finalize(self):
        pass

    def help(self):
        from isceobj.Sensor import SENSORS
        print(self.__doc__)
        lsensors = list(SENSORS.keys())
        lsensors.sort()
        print("The currently supported sensors are / 当前支持的传感器: ", lsensors)
        return None

    def help_steps(self):
        print(self.__doc__)
        print("A description of the individual steps can be found in the README file")
        print("and also in the ISCE.pdf document")
        print("各步骤说明可在 README 与 ISCE.pdf 中查看。")
        return

    def renderProcDoc(self):
        self.procDoc.renderXml()


    def startup(self):
        self.help()
        self._insar.timeStart = time.time()

    def endup(self):
        saved_post_args = os.environ.get("ISCE_AUTO_POSTPROCESS_ARGS")
        injected_args = False
        try:
            auto_wvl = _autodetect_wavelength_for_postprocess(self._insar)
            post_args = os.environ.get("ISCE_AUTO_POSTPROCESS_ARGS", "")
            post_args = _ensure_postprocess_utm_args(post_args)
            if (auto_wvl is not None) and ("--wavelength" not in post_args):
                extra = "--wavelength {0:.12g}".format(auto_wvl)
                merged = (post_args + " " + extra).strip() if post_args else extra
                post_args = merged
                logger.info(
                    "Auto postprocess stripmap wavelength override enabled: %.12g m / 已启用 stripmap 自动波长覆盖",
                    auto_wvl,
                )

            os.environ["ISCE_AUTO_POSTPROCESS_ARGS"] = post_args
            injected_args = True
            run_auto_postprocess(logger, 'stripmapApp')
        finally:
            if injected_args:
                if saved_post_args is None:
                    os.environ.pop("ISCE_AUTO_POSTPROCESS_ARGS", None)
                else:
                    os.environ["ISCE_AUTO_POSTPROCESS_ARGS"] = saved_post_args
        self.renderProcDoc()
        self._insar.timeEnd = time.time()
        if hasattr(self._insar, 'timeStart'):
            logger.info("Total Time: %i seconds" %
                        (self._insar.timeEnd-self._insar.timeStart))
        return None


    ## Add instance attribute RunWrapper functions, which emulate methods.
    def _add_methods(self):
        self.runPreprocessor = StripmapProc.createPreprocessor(self)
        self.runFormSLC = StripmapProc.createFormSLC(self)
        self.runCrop = StripmapProc.createCrop(self)
        self.runSplitSpectrum = StripmapProc.createSplitSpectrum(self)
        self.runTopo = StripmapProc.createTopo(self)
        self.runNormalizeSecondarySampling = StripmapProc.createNormalizeSecondarySampling(self)
        self.runGeo2rdr = StripmapProc.createGeo2rdr(self)
        self.runRdrDemOffset = StripmapProc.createRdrDemOffset(self)
        self.runRectRangeOffset = StripmapProc.createRectRangeOffset(self)
        self.runResampleSlc = StripmapProc.createResampleSlc(self)
        self.runRefineSecondaryTiming = StripmapProc.createRefineSecondaryTiming(self)
        self.runDenseOffsets = StripmapProc.createDenseOffsets(self)
        self.runRubbersheetRange = StripmapProc.createRubbersheetRange(self) #Modified by V. Brancato 10.07.2019
        self.runRubbersheetAzimuth =StripmapProc.createRubbersheetAzimuth(self) #Modified by V. Brancato 10.07.2019
        self.runResampleSubbandSlc = StripmapProc.createResampleSubbandSlc(self)
        self.runInterferogram = StripmapProc.createInterferogram(self)
        self.runFilter = StripmapProc.createFilter(self)
        self.runDispersive = StripmapProc.createDispersive(self)
        self.verifyDEM = StripmapProc.createVerifyDEM(self)
        self.runGeocode = StripmapProc.createGeocode(self)
        return None

    def _steps(self):

        self.step('startup', func=self.startup,
                     doc=("Print a helpful message and set the startTime of processing / 输出帮助信息并记录开始时间")
                  )

        # Run a preprocessor for the two sets of frames
        self.step('preprocess',
                  func=self.runPreprocessor,
                  doc=(
                """Preprocess the reference and secondary sensor data to raw images / 将主辅影像预处理为原始图像"""
                )
                  )

        self.step('cropraw',
                func = self.runCrop,
                args=(True,))

        self.step('formslc', func=self.runFormSLC)

        self.step('cropslc', func=self.runCrop,
                args=(False,))

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        self.step('verifyDEM', func=self.verifyDEM)

        self.step('topo', func=self.runTopo)

        self.step('normalize_secondary_sampling', func=self.runNormalizeSecondarySampling)

        self.step('geo2rdr', func=self.runGeo2rdr)
        self.step('rdrdem_offset', func=self.runRdrDemOffset)
        self.step('rect_rgoffset', func=self.runRectRangeOffset)

        self.step('coarse_resample', func=self.runResampleSlc,
                    args=('coarse',))

        self.step('misregistration', func=self.runRefineSecondaryTiming)

        self.step('refined_resample', func=self.runResampleSlc,
                    args=('refined',))

        self.step('dense_offsets', func=self.runDenseOffsets)
######################################################################## Modified by V. Brancato 10.07.2019
        self.step('rubber_sheet_range', func=self.runRubbersheetRange)
	
        self.step('rubber_sheet_azimuth',func=self.runRubbersheetAzimuth)
#########################################################################

        self.step('fine_resample', func=self.runResampleSlc,
                    args=('fine',))

        self.step('split_range_spectrum', func=self.runSplitSpectrum)

        self.step('sub_band_resample', func=self.runResampleSubbandSlc,
                    args=(True,))

        self.step('interferogram', func=self.runInterferogram)

        self.step('sub_band_interferogram', func=self.runInterferogram,
                args=("sub",))

        self.step('filter', func=self.runFilter,
                  args=(self.filterStrength,))

        self.step('filter_low_band', func=self.runFilter,
                  args=(self.filterStrength,"low",))

        self.step('filter_high_band', func=self.runFilter,
                  args=(self.filterStrength,"high",))

        self.step('unwrap', func=self.runUnwrapper)

        self.step('unwrap_low_band', func=self.runUnwrapper, args=("low",))

        self.step('unwrap_high_band', func=self.runUnwrapper, args=("high",))

        self.step('ionosphere', func=self.runDispersive)

        self.step('geocode', func=self.runGeocode,
                args=(self.geocode_list, self.geocode_bbox))

        self.step('geocodeoffsets', func=self.runGeocode,
                args=(self.off_geocode_list, self.geocode_bbox, True))

        return None

    ## Main has the common start to both insarApp and dpmApp.
    #@use_api
    def main(self):
        self.timeStart = time.time()
        self._insar.timeStart = self.timeStart
        self.help()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        #Crop raw data if desired
        self.runCrop(True)

        self.runFormSLC()

        self.runCrop(False)

        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        self.verifyDEM()

        # run topo (mapping from radar to geo coordinates)
        self.runTopo()

        # normalize secondary PRF/DC when mismatch is likely to impact correlation quality
        self.runNormalizeSecondarySampling()

        # run geo2rdr (mapping from geo to radar coordinates)
        self.runGeo2rdr()

        # radar-dem closure loop for range offset geometry correction
        self.runRdrDemOffset()
        self.runRectRangeOffset()

        # resampling using only geometry offsets
        self.runResampleSlc('coarse')

        # refine geometry offsets using offsets computed by cross correlation
        self.runRefineSecondaryTiming()

        # resampling using refined offsets
        self.runResampleSlc('refined')

        # run dense offsets
        self.runDenseOffsets()
	
############ Modified by V. Brancato 10.07.2019
        # adding the azimuth offsets computed from cross correlation to geometry offsets 
        self.runRubbersheetAzimuth()
       
        # adding the range offsets computed from cross correlation to geometry offsets 
        self.runRubbersheetRange()
####################################################################################
        # resampling using rubbersheeted offsets
        # which include geometry + constant range + constant azimuth
        # + dense azimuth offsets
        self.runResampleSlc('fine')

        #run split range spectrum
        self.runSplitSpectrum()

        self.runResampleSubbandSlc(misreg=True)
        # forming the interferogram
        self.runInterferogram()

        self.runInterferogram(igramSpectrum = "sub")

        # Filtering and estimating coherence
        self.runFilter(self.filterStrength)

        self.runFilter(self.filterStrength, igramSpectrum = "low")

        self.runFilter(self.filterStrength, igramSpectrum = "high")

        # unwrapping
        self.runUnwrapper()

        self.runUnwrapper(igramSpectrum = "low")

        self.runUnwrapper(igramSpectrum = "high")

        self.runDispersive()

        self.runGeocode(self.geocode_list, self.geocode_bbox)

        self.runGeocode(self.geocode_list, self.geocode_bbox, True)


        self.timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(self.timeEnd - self.timeStart))

        self.renderProcDoc()

        return None

class Insar(_RoiBase):
    """
    Insar Application:
    Implements InSAR processing flow for a pair of scenes from
    sensor raw data to geocoded, flattened interferograms.
    """

    family = "insar"

    def __init__(self, family='',name='',cmdline=None):
        #to allow inheritance with different family name use the locally
        #defined only if the subclass (if any) does not specify one

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

    def Usage(self):
        print("Usages: ")
        print("stripmapApp.py <input-file.xml>")
        print("stripmapApp.py --steps")
        print("stripmapApp.py --help")
        print("stripmapApp.py --help --steps")


    ## extends _InsarBase_steps, but not in the same was as main
    def _steps(self):
        super()._steps()

        # Geocode
        #self.step('geocode', func=self.runGeocode,
        #        args=(self.geocode_list, self.unwrap, self.geocode_bbox))

        self.step('endup', func=self.endup)

        return None

    ## main() extends _InsarBase.main()
    def main(self):

        super().main()
        print("self.timeStart = {}".format(self.timeStart))

        # self.runCorrect()

        #self.runRgoffset()

        # Cull offoutliers
        #self.iterate_runOffoutliers()

        self.runResampleSlc()
        #self.runResamp_only()

        self.runRefineSecondaryTiming()

        #self.insar.topoIntImage=self.insar.resampOnlyImage
        #self.runTopo()
#        self.runCorrect()

        # Coherence ?
        #self.runCoherence(method=self.correlation_method)


        # Filter ?
        self.runFilter(self.filterStrength)

        # Unwrap ?
        self.runUnwrapper()

        # Geocode
        #self.runGeocode(self.geocode_list, self.unwrap, self.geocode_bbox)

        self.endup()

        return None


if __name__ == "__main__":
    #make an instance of Insar class named 'stripmapApp'
    insar = Insar(name="stripmapApp")
    #configure the insar application
    insar.configure()
    #invoke the base class run method, which returns status
    status = insar.run()
    #inform Python of the status of the run to return to the shell
    raise SystemExit(status)
