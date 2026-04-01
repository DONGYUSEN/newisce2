#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Giangi Sacco, Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import time
import sys
from isce import logging

import isce
import isceobj
import iscesys
from iscesys.Component.Application import Application
from iscesys.Compatibility import Compatibility
from iscesys.Component.Configurable import SELF
from isceobj import RtcProc
from isceobj.Util.decorators import use_api

try:
    from isce2.applications.postprocess_hook import run_auto_postprocess
except ImportError:
    try:
        from applications.postprocess_hook import run_auto_postprocess
    except ImportError:
        from postprocess_hook import run_auto_postprocess

logger = logging.getLogger('isce.grdsar')


SENSOR_NAME = Application.Parameter(
    'sensorName',
    public_name='sensor name',
    default='SENTINEL1',
    type=str,
    mandatory=True,
    doc="Sensor name / 传感器名称"
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
     doc="Filename of the Digital Elevation Model (DEM) / DEM 文件路径"
                                     )

WATER_FILENAME = Application.Parameter(
        'waterFilename',
        public_name='waterFilename',
        default='',
        type=str,
        mandatory=False,
        doc='Filename with SWBD data / SWBD 水体掩膜文件路径')

APPLY_WATER_MASK = Application.Parameter(
        'applyWaterMask',
        public_name='apply water mask',
        default=False,
        type=bool,
        mandatory=False,
        doc = 'Flag to apply water mask to images / 是否对影像应用水体掩膜')

GEOCODE_BOX = Application.Parameter(
    'geocode_bbox',
    public_name='geocode bounding box',
    default = None,
    container=list,
    type=float,
    doc='Bounding box for geocoding - South, North, West, East in degrees / 地理编码范围：南北西东（度）'
                                    )

EPSG = Application.Parameter(
    'epsg',
    public_name='epsg id',
    default = '',
    type=str,
    doc='epsg code for roi / 目标投影 EPSG 代码'
                                    )

GSPACING = Application.Parameter('gspacing',
            public_name='geocode spacing',
            default = None,
            type = float,
            doc = 'Desired grid spacing of geocoded product in meters. If not set, auto use ceil(max(multilook x/y resolution)) with square pixels. / 地理编码输出分辨率（米）；未设置时自动使用多视分辨率最大值向上取整并强制方形像元。'
                                    ) 

INTMETHOD = Application.Parameter('intmethod',
            public_name='geocode interpolation method',
            default = 'bilinear',
            type = str,
            doc = 'Desired grid spacing of geocoded product in meters, in the specified UTM grid. / 地理编码插值方法。'
                                    ) 

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
    ))

NUMBER_AZIMUTH_LOOKS = Application.Parameter('numberAzimuthLooks',
                                           public_name='azimuth looks',
                                           default=None,
                                           type=int,
                                           mandatory=False,
                                           doc='')


NUMBER_RANGE_LOOKS = Application.Parameter('numberRangeLooks',
    public_name='range looks',
    default=None,
    type=int,
    mandatory=False,
    doc=''
)

POSTING = Application.Parameter('posting',
            public_name='posting',
            default = 10.0,
            type = float,
            mandatory = False,
            doc = 'Posting of data. This can be any integer multiple of the product resolution. Used to determine looks') 

POLARIZATIONS = Application.Parameter('polarizations',
            public_name='polarizations',
            default = [],
            type = str,
            container = list,
            doc = 'Polarizations to process')

GEOCODE_LIST = Application.Parameter(
    'geocode_list',
     public_name='geocode list',
     default = None,
     container=list,
     type=str,
     doc = "List of products to geocode."
                                      )

APPLY_THERMAL_NOISE_CORRECTION = Application.Parameter(
        'apply_thermal_noise_correction',
        public_name='apply thermal noise correction',
        default=False,
        type=bool,
        mandatory=False,
        doc = 'Flag to apply thermal noise correction. Currently only available for Sentinel-1.')


#Facility declarations
REFERENCE = Application.Facility(
    'reference',
    public_name='Reference',
    module='isceobj.Sensor.GRD',
    factory='createSensor',
    args=(SENSOR_NAME, 'reference'),
    mandatory=True,
    doc="GRD data component"
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


_GRD = Application.Facility(
    '_grd',
    public_name='rtcproc',
    module='isceobj.RtcProc',
    factory='createRtcProc',
    args = ('rtcAppContext',isceobj.createCatalog('rtcProc')),
    mandatory=False,
    doc="RtcProc object"
)


class GRDSAR(Application):

    family = 'grdsar'
    ## Define Class parameters in this list
    parameter_list = (SENSOR_NAME,
                      USE_HIGH_RESOLUTION_DEM_ONLY,
                      DEM_FILENAME,
                      NUMBER_AZIMUTH_LOOKS,
                      NUMBER_RANGE_LOOKS,
                      POSTING,
                      GEOCODE_BOX,
                      EPSG,
                      GSPACING,
                      INTMETHOD,
                      PICKLE_DUMPER_DIR,
                      PICKLE_LOAD_DIR,
                      RENDERER,
                      POLARIZATIONS,
                      GEOCODE_LIST,
                      APPLY_THERMAL_NOISE_CORRECTION)

    facility_list = (REFERENCE,
                     DEM_STITCHER,
                     _GRD)

    _pickleObj = "_grd"

    def __init__(self, family='', name='',cmdline=None):
        import isceobj
        from isceobj.RtcProc import RtcProc
        from iscesys.StdOEL.StdOELPy import create_writer

        super().__init__(
            family=family if family else  self.__class__.family, name=name,
            cmdline=cmdline)

        self._stdWriter = create_writer("log", "", True, filename="grdsar.log")
        self._add_methods()
        self._insarProcFact = RtcProc
        return None



    def Usage(self):
        print("Usages / 用法: ")
        print("rtcApp.py <input-file.xml>   # 使用配置文件运行")
        print("rtcApp.py --steps            # 分步骤运行")
        print("rtcApp.py --help             # 查看帮助")
        print("rtcApp.py --help --steps     # 查看步骤帮助")


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

    def _configure(self):

        self.grd.procDoc._addItem("ISCE_VERSION",
            "Release: %s, svn-%s, %s. Current svn-%s" %
            (isce.release_version, isce.release_svn_revision,
             isce.release_date, isce.svn_revision
            ),
            ["rtcProc"]
            )

        if(self.geocode_list is None):
            self.geocode_list = self.grd.geocode_list
        else:
            g_count = 0
            for g in self.geocode_list:
                if g not in self.grd.geocode_list:
                    g_count += 1
            #warn if there are any differences in content
            if g_count > 0:
                print()
                logger.warning((
                    "Some filenames in rtcApp.geocode_list configuration "+
                    "are different from those in rtcProc. Using names given"+
                    " to grdApp. / rtcApp.geocode_list 与 rtcProc 不一致，采用 rtcApp 配置。"))
                print("grdApp.geocode_list = {}".format(self.geocode_list))
                print(("grdProc.geocode_list = {}".format(
                        self.grd.geocode_list)))

            self.grd.geocode_list = self.geocode_list

        return None

    @property
    def grd(self):
        return self._grd
    
    @grd.setter
    def grd(self, value):
        self._grd = value
        return None

    @property
    def procDoc(self):
        return self.grd.procDoc

    @procDoc.setter
    def procDoc(self):
        raise AttributeError(
            "Can not assign to .grd.procDoc-- but you hit all its other stuff"
            )

    def _finalize(self):
        pass

    def help(self):
        from isceobj.Sensor.GRD import SENSORS
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
        self._grd.timeStart = time.time()

    def endup(self):
        run_auto_postprocess(logger, 'rtcApp')
        self.renderProcDoc()
        self._grd.timeEnd = time.time()
        logger.info("Total Time: %i seconds" %
                    (self._grd.timeEnd-self._grd.timeStart))
        return None


    ## Add instance attribute RunWrapper functions, which emulate methods.
    def _add_methods(self):
        self.runPreprocessor = RtcProc.createPreprocessor(self)
        self.verifyDEM = RtcProc.createVerifyDEM(self)
        self.multilook = RtcProc.createLooks(self)
        self.runTopo  = RtcProc.createTopo(self)
        self.runNormalize = RtcProc.createNormalize(self)
        self.runGeocode = RtcProc.createGeocode(self)

        return None

    def _steps(self):

        self.step('startup', func=self.startup,
                     doc=("Print a helpful message and set the startTime of processing / 输出帮助信息并记录开始时间")
                  )

        # Run a preprocessor for the two sets of frames
        self.step('preprocess',
                  func=self.runPreprocessor,
                  doc=(
                """Unpack the input data / 解包输入数据"""
                )
                  )

        # Verify whether the DEM was initialized properly.  If not, download
        # a DEM
        self.step('verifyDEM', func=self.verifyDEM)

        #Multilook product as needed
        self.step('multilook', func=self.multilook)

        ##Run topo for each bursts
        self.step('topo', func=self.runTopo)

	    ##Run normalize to get gamma0
        self.step('normalize', func=self.runNormalize)

        # Geocode
        self.step('geocode', func=self.runGeocode)

        self.step('endup', func=self.endup)

        return None

    @use_api
    def main(self):
        self.help()

        self._grd.timeStart = time.time()

        # Run a preprocessor for the two sets of frames
        self.runPreprocessor()

        #Verify whether user defined  a dem component.  If not, then download
        # SRTM DEM.
        self.verifyDEM()

        #Multilook as needed
        self.multilook()

        ##Run topo for each burst
        self.runTopo()
	
	##Run normalize to get gamma0
        self.runNormalize()

        # Geocode
        self.runGeocode()

        self.endup()

        return None




if __name__ == "__main__":
    import sys
    grdsar = GRDSAR(name="rtcApp")
    grdsar.configure()
    grdsar.run()
