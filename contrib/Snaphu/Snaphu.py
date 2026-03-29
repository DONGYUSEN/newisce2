#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




from iscesys.Component.Component import Component
from . import snaphu
import math
import os
import xml.etree.ElementTree as ET

ALTITUDE = Component.Parameter(
    'altitude',
    public_name='ALTITUDE',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Altitude'
)


AZIMUTH_LOOKS = Component.Parameter(
    'azimuthLooks',
    public_name='AZIMUTH_LOOKS',
    default=1,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of looks in the azimuth direction'
)


CORR_FILE = Component.Parameter(
    'corrfile',
    public_name='CORR_FILE',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Correlation file name'
)


CORR_LOOKS = Component.Parameter(
    'corrLooks',
    public_name='CORR_LOOKS',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Correlation looks'
)


COR_FILE_FORMAT = Component.Parameter(
    'corFileFormat',
    public_name='COR_FILE_FORMAT',
    default='ALT_LINE_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Correlation file format'
)


COSTMODE = Component.Parameter(
    'costMode',
    public_name='COSTMODE',
    default='DEFO',
    type=str,
    mandatory=True,
    intent='input',
    doc='Cost function mode. Options are "TOPO","DEFO","SMOOTH".'
)


DEFORMATION_MAX_CYCLES = Component.Parameter(
    'defoMaxCycles',
    public_name='DEFORMATION_MAX_CYCLES',
    default=1.2,
    type=float,
    mandatory=True,
    intent='input',
    doc='Deformation max cycles'
)


DUMP_CONNECTED_COMPONENTS = Component.Parameter(
    'dumpConnectedComponents',
    public_name='DUMP_CONNECTED_COMPONENTS',
    default=True,
    type=bool,
    mandatory=False,
    intent='input',
    doc='Dump the connected component to a file with extension .conncomp'
)


EARTHRADIUS = Component.Parameter(
    'earthRadius',
    public_name='EARTHRADIUS',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc='Earth radius'
)


INIT_METHOD = Component.Parameter(
    'initMethod',
    public_name='INIT_METHOD',
    default='MST',
    type=str,
    mandatory=False,
    intent='input',
    doc='Init method. Options are "MST" or "MCF"'
)


INIT_ONLY = Component.Parameter(
    'initOnly',
    public_name='INIT_ONLY',
    default=False,
    type=bool,
    mandatory=False,
    intent='input',
    doc='Is this is set along with the DUMP_CONNECTED_COMPONENTS flag, then only the' +\
        'connected components are computed and dumped into a file with extension .conncomp'
)


INPUT = Component.Parameter(
    'input',
    public_name='INPUT',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Input file name'
)


INT_FILE_FORMAT = Component.Parameter(
    'intFileFormat',
    public_name='INT_FILE_FORMAT',
    default='COMPLEX_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Interferogram file format'
)


MAX_COMPONENTS = Component.Parameter(
    'maxComponents',
    public_name='MAX_COMPONENTS',
    default=32,
    type=int,
    mandatory=False,
    intent='input',
    doc='Max number of components'
)


OUTPUT = Component.Parameter(
    'output',
    public_name='OUTPUT',
    default=None,
    type=str,
    mandatory=True,
    intent='input',
    doc='Output file name'
)


RANGE_LOOKS = Component.Parameter(
    'rangeLooks',
    public_name='RANGE_LOOKS',
    default=1,
    type=int,
    mandatory=True,
    intent='input',
    doc='Number of looks in the range direction'
)


UNW_FILE_FORMAT = Component.Parameter(
    'unwFileFormat',
    public_name='UNW_FILE_FORMAT',
    default='ALT_LINE_DATA',
    type=str,
    mandatory=False,
    intent='input',
    doc='Unwrap file format'
)


WAVELENGTH = Component.Parameter(
    'wavelength',
    public_name='WAVELENGTH',
    default=None,
    type=float,
    mandatory=True,
    intent='input',
    doc='Wave length'
)


WIDTH = Component.Parameter(
    'width',
    public_name='WIDTH',
    default=None,
    type=int,
    mandatory=True,
    intent='input',
    doc='Image width'
)

class Snaphu(Component):

    parameter_list = (
                      ALTITUDE,
                      INPUT,
                      DUMP_CONNECTED_COMPONENTS,
                      WIDTH,
                      EARTHRADIUS,
                      INIT_ONLY,
                      CORR_LOOKS,
                      COR_FILE_FORMAT,
                      CORR_FILE,
                      WAVELENGTH,
                      MAX_COMPONENTS,
                      RANGE_LOOKS,
                      DEFORMATION_MAX_CYCLES,
                      UNW_FILE_FORMAT,
                      OUTPUT,
                      AZIMUTH_LOOKS,
                      INIT_METHOD,
                      COSTMODE,
                      INT_FILE_FORMAT
                     )

    """The Snaphu cost unwrapper"""

    fileFormats = { 'COMPLEX_DATA'  : 1,
                    'FLOAT_DATA'    : 2,
                    'ALT_LINE_DATA' : 3,
                    'ALT_SAMPLE_DATA' : 4}
    
    logging_name = "contrib.Snaphu.Snaphu"


    family = 'snaphu'

    def __init__(self,family='',name=''):
        super(Snaphu, self).__init__(family if family else  self.__class__.family, name=name)
        self.minConnectedComponentFrac = 0.01
        self.connectedComponentCostThreshold = 300
        self.magnitude = None
        # Runtime tuning knobs for parallel/tiled unwrapping.
        # These can be configured via setters or environment variables.
        self.nproc = self._safe_env_int('ISCE_SNAPHU_NPROC', 1, minimum=1)
        self.tileNRow = self._safe_env_int('ISCE_SNAPHU_NTILEROW', 1, minimum=1)
        self.tileNCol = self._safe_env_int('ISCE_SNAPHU_NTILECOL', 1, minimum=1)
        self.rowOverlap = self._safe_env_int('ISCE_SNAPHU_ROWOVRLP', 0, minimum=0)
        self.colOverlap = self._safe_env_int('ISCE_SNAPHU_COLOVRLP', 0, minimum=0)
        # snaphu internally warns when tile overlap is less than 400 pixels.
        # Keep this configurable but default to a safe value to reduce tile failures.
        self.minTileOverlap = self._safe_env_int('ISCE_SNAPHU_MIN_OVERLAP', 400, minimum=0)
        self.tilingMode = os.environ.get('ISCE_SNAPHU_MODE', 'manual').strip().lower()
        self.balancedOverlap = self._safe_env_int('ISCE_SNAPHU_BALANCED_OVERLAP', 512, minimum=0)
        self.balancedMemFraction = self._safe_env_float(
            'ISCE_SNAPHU_BALANCED_MEM_FRAC', 0.60, minimum=0.05, maximum=0.95
        )
        self.balancedBytesPerPixel = self._safe_env_float(
            'ISCE_SNAPHU_BALANCED_BYTES_PER_PIXEL', 120.0, minimum=1.0
        )
        self.balancedMaxGrid = self._safe_env_int('ISCE_SNAPHU_BALANCED_MAX_GRID', 8, minimum=1)
        self.balancedMaxNProc = self._safe_env_int('ISCE_SNAPHU_BALANCED_MAX_NPROC', 4, minimum=1)
        self.balancedMinTileSide = self._safe_env_int(
            'ISCE_SNAPHU_BALANCED_MIN_TILE_SIDE', 2048, minimum=256
        )
        self.balancedMemGB = self._safe_env_float(
            'ISCE_SNAPHU_BALANCED_MEM_GB', 0.0, minimum=0.0
        )

    def _safe_env_int(self, key, default, minimum=None):
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            parsed = int(value)
        except Exception:
            self.logger.warning('Invalid %s=%r; using default %s', key, value, default)
            return default
        if (minimum is not None) and (parsed < minimum):
            self.logger.warning('%s=%s is below minimum %s; using %s', key, parsed, minimum, default)
            return default
        return parsed

    def _safe_env_float(self, key, default, minimum=None, maximum=None):
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            parsed = float(value)
        except Exception:
            self.logger.warning('Invalid %s=%r; using default %s', key, value, default)
            return default
        if (minimum is not None) and (parsed < minimum):
            self.logger.warning('%s=%s is below minimum %s; using %s', key, parsed, minimum, default)
            return default
        if (maximum is not None) and (parsed > maximum):
            self.logger.warning('%s=%s is above maximum %s; using %s', key, parsed, maximum, default)
            return default
        return parsed

    def _read_meminfo_total_bytes(self):
        try:
            with open('/proc/meminfo', 'r') as fp:
                for line in fp:
                    if not line.startswith('MemTotal:'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except Exception:
            return None
        return None

    def _read_cgroup_limit_bytes(self):
        for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            try:
                with open(path, 'r') as fp:
                    raw = fp.read().strip()
            except Exception:
                continue

            if (not raw) or (raw == 'max'):
                continue

            try:
                value = int(raw)
            except Exception:
                continue

            if value <= 0:
                continue
            if value >= (1 << 60):
                continue
            return value
        return None

    def _get_balanced_memory_budget_bytes(self):
        if self.balancedMemGB > 0.0:
            return int(self.balancedMemGB * (1024.0 ** 3) * self.balancedMemFraction)

        limit = self._read_cgroup_limit_bytes()
        if limit is not None:
            return int(limit * self.balancedMemFraction)

        total = self._read_meminfo_total_bytes()
        if total is not None:
            return int(total * self.balancedMemFraction)

        return None


    def setCorrfile(self, corrfile):
        """Set the correlation filename for unwrapping"""
        self.corrfile = corrfile

    def setDefoMaxCycles(self, ncycles):
        """Set the maximum phase discontinuity expected."""
        self.defoMaxCycles = ncycles

    def setCorrLooks(self, looks):
        """Set the number of looks used for computing correlation"""
        self.corrLooks = looks

    def setInput(self,input):
        """Set the input filename for unwrapping"""
        self.input = input
        
    def setOutput(self,output):
        """Set the output filename for unwrapping"""
        self.output = output
        
    def setWidth(self,width):
        """Set the image width"""
        self.width = width
        
    def setWavelength(self,wavelength):
        """Set the radar wavelength"""
        self.wavelength = wavelength

    def setRangeLooks(self, looks):
        self.rangeLooks = looks

    def setAzimuthLooks(self, looks):
        self.azimuthLooks = looks

    def setNProc(self, nproc):
        self.nproc = int(nproc)

    def setTileNRow(self, nrows):
        self.tileNRow = int(nrows)

    def setTileNCol(self, ncols):
        self.tileNCol = int(ncols)

    def setRowOverlap(self, overlap):
        self.rowOverlap = int(overlap)

    def setColOverlap(self, overlap):
        self.colOverlap = int(overlap)

    def _infer_input_width(self):
        """Infer raster width from <input>.xml when available."""
        if not self.input:
            return None
        xml_path = self.input + '.xml'
        if not os.path.isfile(xml_path):
            return None
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return None

        for prop in root.findall('property'):
            if str(prop.get('name', '')).lower() != 'width':
                continue
            value = prop.findtext('value')
            if value is None:
                continue
            try:
                return int(float(value))
            except Exception:
                return None
        return None

    def _infer_input_length(self):
        """Infer raster length from <input>.xml when available."""
        if not self.input:
            return None
        xml_path = self.input + '.xml'
        if not os.path.isfile(xml_path):
            return None
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return None

        for prop in root.findall('property'):
            if str(prop.get('name', '')).lower() != 'length':
                continue
            value = prop.findtext('value')
            if value is None:
                continue
            try:
                return int(float(value))
            except Exception:
                return None
        return None

    def _tile_shape(self, nlines, linelen, ntilerow, ntilecol, rowovrlp, colovrlp):
        ni = int(math.ceil((nlines + (ntilerow - 1) * rowovrlp) / float(ntilerow)))
        nj = int(math.ceil((linelen + (ntilecol - 1) * colovrlp) / float(ntilecol)))
        return ni, nj

    def _choose_balanced_tiling(self):
        if self.tilingMode not in ('balanced', 'balance'):
            return

        nlines = self._infer_input_length()
        linelen = int(self.width) if self.width is not None else self._infer_input_width()
        if (nlines is None) or (linelen is None) or (nlines <= 0) or (linelen <= 0):
            self.logger.warning(
                'ISCE_SNAPHU_MODE=balanced requires valid raster size; using single tile fallback.'
            )
            self.tileNRow = 1
            self.tileNCol = 1
            self.rowOverlap = 0
            self.colOverlap = 0
            self.nproc = 1
            return

        overlap = max(0, int(self.balancedOverlap))
        mem_budget = self._get_balanced_memory_budget_bytes()
        max_grid = max(1, int(self.balancedMaxGrid))
        max_nproc = max(1, int(self.balancedMaxNProc))
        cpu_count = max(1, os.cpu_count() or 1)
        max_nproc = min(max_nproc, cpu_count)
        min_tile_side = max(256, int(self.balancedMinTileSide))
        bytes_per_pixel = max(1.0, float(self.balancedBytesPerPixel))

        best = None
        for ntilerow in range(1, max_grid + 1):
            for ntilecol in range(1, max_grid + 1):
                # Respect snaphu constraints in src/snaphu_io.c.
                if (ntilerow + overlap > nlines) or (ntilecol + overlap > linelen):
                    continue
                if (ntilerow * ntilerow > nlines) or (ntilecol * ntilecol > linelen):
                    continue

                ni, nj = self._tile_shape(nlines, linelen, ntilerow, ntilecol, overlap, overlap)
                if (ntilerow > 1 or ntilecol > 1) and (min(ni, nj) < min_tile_side):
                    continue

                ntiles = ntilerow * ntilecol
                tile_pixels = float(ni) * float(nj)
                nproc_limit = min(ntiles, max_nproc)
                nproc = 0
                for candidate_nproc in range(nproc_limit, 0, -1):
                    if (mem_budget is None) or (candidate_nproc * tile_pixels * bytes_per_pixel <= mem_budget):
                        nproc = candidate_nproc
                        break
                if nproc == 0:
                    continue

                seam_ratio = 0.0
                if ntilerow > 1:
                    seam_ratio += float(overlap * (ntilerow - 1)) / float(max(1, nlines))
                if ntilecol > 1:
                    seam_ratio += float(overlap * (ntilecol - 1)) / float(max(1, linelen))

                time_cost = (float(nlines) * float(linelen) / float(max(1, nproc))) * (1.0 + 0.25 * seam_ratio)
                split_penalty = 1.0 + 0.03 * float(ntiles - 1)
                score = time_cost * split_penalty

                candidate = (
                    score,
                    -nproc,
                    ntiles,
                    -min(ni, nj),
                    ntilerow,
                    ntilecol,
                    ni,
                    nj,
                    nproc,
                )
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            self.logger.warning(
                'Balanced mode could not find a feasible tile layout; using single tile fallback.'
            )
            self.tileNRow = 1
            self.tileNCol = 1
            self.rowOverlap = 0
            self.colOverlap = 0
            self.nproc = 1
            return

        _, _, _, _, ntilerow, ntilecol, ni, nj, nproc = best
        self.tileNRow = int(ntilerow)
        self.tileNCol = int(ntilecol)
        self.rowOverlap = overlap if self.tileNRow > 1 else 0
        self.colOverlap = overlap if self.tileNCol > 1 else 0
        self.nproc = max(1, min(int(nproc), self.tileNRow * self.tileNCol))
        self.logger.info(
            'Balanced tiling selected: %dx%d tiles, overlap(row/col)=%d/%d, '
            'tile_size~%dx%d, nproc=%d, memory_budget=%s bytes',
            self.tileNRow,
            self.tileNCol,
            self.rowOverlap,
            self.colOverlap,
            ni,
            nj,
            self.nproc,
            str(mem_budget) if mem_budget is not None else 'unknown',
        )

    def _normalize_tiling(self):
        """Normalize tile-related runtime knobs to avoid unstable settings."""
        if self.tilingMode not in ('manual', 'balanced', 'balance'):
            self.logger.warning(
                'Unknown ISCE_SNAPHU_MODE=%r; falling back to manual mode.',
                self.tilingMode
            )
            self.tilingMode = 'manual'

        self._choose_balanced_tiling()

        self.tileNRow = max(1, int(self.tileNRow))
        self.tileNCol = max(1, int(self.tileNCol))
        self.rowOverlap = max(0, int(self.rowOverlap))
        self.colOverlap = max(0, int(self.colOverlap))
        self.nproc = max(1, int(self.nproc))

        tile_mode = (self.tileNRow > 1) or (self.tileNCol > 1)
        if tile_mode:
            min_overlap = max(0, int(self.minTileOverlap))
            if self.tileNRow > 1 and self.rowOverlap < min_overlap:
                self.logger.warning(
                    'Increasing ISCE_SNAPHU_ROWOVRLP from %d to %d for safer tiled unwrapping',
                    self.rowOverlap, min_overlap
                )
                self.rowOverlap = min_overlap
            if self.tileNCol > 1 and self.colOverlap < min_overlap:
                self.logger.warning(
                    'Increasing ISCE_SNAPHU_COLOVRLP from %d to %d for safer tiled unwrapping',
                    self.colOverlap, min_overlap
                )
                self.colOverlap = min_overlap

            ntiles = self.tileNRow * self.tileNCol
            if self.nproc > ntiles:
                self.logger.warning(
                    'Reducing ISCE_SNAPHU_NPROC from %d to %d (tile count)',
                    self.nproc, ntiles
                )
                self.nproc = ntiles
        else:
            # Single-tile mode ignores overlap and multiprocess options.
            self.rowOverlap = 0
            self.colOverlap = 0
            self.nproc = 1

        # Clamp overlap to avoid impossible settings for small rasters.
        length = self._infer_input_length()
        width = int(self.width) if self.width is not None else None
        if (length is not None) and self.tileNRow > 1:
            max_row_ov = max(0, length - self.tileNRow)
            if self.rowOverlap > max_row_ov:
                self.logger.warning(
                    'Clamping ISCE_SNAPHU_ROWOVRLP from %d to %d to fit raster length',
                    self.rowOverlap, max_row_ov
                )
                self.rowOverlap = max_row_ov
        if (width is not None) and self.tileNCol > 1:
            max_col_ov = max(0, width - self.tileNCol)
            if self.colOverlap > max_col_ov:
                self.logger.warning(
                    'Clamping ISCE_SNAPHU_COLOVRLP from %d to %d to fit raster width',
                    self.colOverlap, max_col_ov
                )
                self.colOverlap = max_col_ov
   
    def setIntFileFormat(self, instr):
        self.intFileFormat = str(instr)

    def setCorFileFormat(self, instr):
        self.corFileFormat = str(instr)

    def setUnwFileFormat(self, instr):
        self.unwFileFormat = str(instr)

    def setCostMode(self,costMode):
        #moved the selection into prepare otherwise using configurable to
        #init  would not work
        self.costMode = costMode    

    def setInitOnly(self, logic):
        self.initOnly = logic

    def dumpConnectedComponents(self, logic):
        self.dumpConnectedComponents = logic
        
    def setAltitude(self,altitude):
        """Set the satellite altitude"""
        self.altitude = altitude
        
    def setEarthRadius(self,earthRadius):
        """Set the local Earth radius"""
        self.earthRadius = earthRadius

    def setInitMethod(self, method):
        """Set the initialization method."""
        #moved the selection into prepare otherwise using configurable to
        #init  would not work
        self.initMethod = method
       

    def setMaxComponents(self, num):
        """Set the maximum number of connected components."""
        self.maxComponents = num
    
    def prepare(self):
        """Perform some initialization of defaults"""
        self._normalize_tiling()

        snaphu.setDefaults_Py()
        snaphu.setInitOnly_Py(int(self.initOnly))
        snaphu.setInput_Py(self.input)
        snaphu.setOutput_Py(self.output)
        if self.magnitude is not None:
            snaphu.setMagnitude_Py(self.magnitude)
        snaphu.setWavelength_Py(self.wavelength)
        
        if not self.costMode in ['TOPO','DEFO','SMOOTH']:
            self.logger.error('Invalid cost mode %s' % (self.costMode))
        #must be one of the 3 above
        snaphu.setCostMode_Py(1 if self.costMode == 'TOPO' else
                             (2 if self.costMode == 'DEFO' else 3))
        snaphu.setAltitude_Py(self.altitude)
        snaphu.setEarthRadius_Py(self.earthRadius)       
        if self.corrfile is not None:
            snaphu.setCorrfile_Py(self.corrfile)

        if self.corrLooks is not None:
            snaphu.setCorrLooks_Py(self.corrLooks)

        if self.defoMaxCycles is not None:
            snaphu.setDefoMaxCycles_Py(self.defoMaxCycles)

        if not self.initMethod in ['MST','MCF']:
            self.logger.error('Invalid init method %s' % (self.initMethod))
        snaphu.setInitMethod_Py(1 if self.initMethod == 'MST' else 2)
                               
        snaphu.setMaxComponents_Py(self.maxComponents)
        snaphu.setRangeLooks_Py(int(self.rangeLooks))
        snaphu.setAzimuthLooks_Py(int(self.azimuthLooks))
        if hasattr(snaphu, 'setNProc_Py'):
            snaphu.setNProc_Py(int(self.nproc))
        if hasattr(snaphu, 'setTileNRow_Py'):
            snaphu.setTileNRow_Py(int(self.tileNRow))
        if hasattr(snaphu, 'setTileNCol_Py'):
            snaphu.setTileNCol_Py(int(self.tileNCol))
        if hasattr(snaphu, 'setRowOverlap_Py'):
            snaphu.setRowOverlap_Py(int(self.rowOverlap))
        if hasattr(snaphu, 'setColOverlap_Py'):
            snaphu.setColOverlap_Py(int(self.colOverlap))
        snaphu.setMinConnectedComponentFraction_Py(int(self.minConnectedComponentFrac))
        snaphu.setConnectedComponentThreshold_Py(int(self.connectedComponentCostThreshold))
        snaphu.setIntFileFormat_Py( int(self.fileFormats[self.intFileFormat]))
        snaphu.setCorFileFormat_Py( int(self.fileFormats[self.corFileFormat]))
        snaphu.setUnwFileFormat_Py( int(self.fileFormats[self.unwFileFormat]))
    

    def unwrap(self):
        """Unwrap the interferogram"""       

        ###Connected components can be dumped out in non-initonly mode
        if not self.initOnly and self.dumpConnectedComponents:
            snaphu.setConnectedComponents_Py(self.output+'.conncomp')
#            snaphu.setRegrowComponents_Py(int(True))

        snaphu.snaphu_Py(self.width)
        self._unwrappingCompleted = True

        ##Second pass if initOnly mode was used.
        if self.initOnly and self.dumpConnectedComponents:
            self.growConnectedComponentsOnly()

    def growConnectedComponentsOnly(self,infile=None,outfile=None):
        '''
        Grows the connected components using an unwrapped file.
        '''
        print('Growing connected components on second pass')
        if infile is None:
            inputFile = self.output
        else:
            inputFile = infile

        if outfile is None:
            outputFile = inputFile + '.conncomp'
        else:
            outputFile = outfile

        self.prepare()
        snaphu.setInitOnly_Py(int(False))
        snaphu.setInput_Py(inputFile)
        snaphu.setConnectedComponents_Py(outputFile)
        snaphu.setRegrowComponents_Py(int(True))
        snaphu.setUnwrappedInput_Py(int(True))
        snaphu.snaphu_Py(self.width)
          
