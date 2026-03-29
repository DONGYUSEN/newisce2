#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2014 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import xml.etree.ElementTree as ET
import datetime
import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Orbit.OrbitExtender import OrbitExtender
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from isceobj.Util import Poly1D
import os
import glob
import numpy as np
import math
import re

sep = "\n"
tab = "    "
lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

XML = Component.Parameter('xml',
        public_name = 'xml',
        default = None,
        type = str,
        doc = 'Input XML file')

TIFF = Component.Parameter('tiff',
        public_name = 'tiff',
        default = None,
        type = str,
        doc = 'Input Tiff file')

MANIFEST = Component.Parameter('manifest',
        public_name = 'manifest',
        default = None,
        type = str,
        doc = 'Manifest file with IPF version')

SAFE = Component.Parameter('safe',
        public_name = 'safe',
        default = None,
        type = str,
        doc = 'SAFE folder / zip file')

ORBIT_FILE = Component.Parameter('orbitFile',
        public_name = 'orbit file',
        default = None,
        type = str,
        doc = 'External orbit file with state vectors')

ORBIT_DIR = Component.Parameter('orbitDir',
        public_name = 'orbit directory',
        default = None,
        type = str,
        doc = 'Directory to search for orbit files')

POLARIZATION = Component.Parameter('polarization',
        public_name = 'polarization',
        default = 'vv',
        type = str,
        mandatory = True,
        doc = 'Polarization')

from .Sensor import Sensor
class Tianyi(Sensor):
    """
        A Class representing Tianyi satellite StripMap data
    """

    family = 'tysm'
    logging = 'isce.sensor.TY_SM'

    parameter_list = (  XML,
                        TIFF,
                        MANIFEST,
                        SAFE,
                        ORBIT_FILE,
                        ORBIT_DIR,
                        POLARIZATION,) + Sensor.parameter_list

    def __init__(self,family='',name=''):

        super(Tianyi,self).__init__(family if family else  self.__class__.family, name=name)
        
        self.frame = Frame()
        self.frame.configure()
    
        self._xml_root=None
    

    def validateUserInputs(self):
        '''
        Validate inputs from user.
        Populate tiff and xml from SAFE folder name.
        '''

        import fnmatch
        import zipfile

        if not self.xml:
            if not self.safe:
                raise Exception('SAFE directory is not provided')


        ####First find annotation file
        ####Dont need swath number when driving with xml and tiff file
        swathid = None
        if not self.xml:
            # Tianyi products follow Sentinel-1 stripmap-like naming/layout.
            swathid = 's1?-s?-slc-{}'.format(self.polarization)

        dirname = self.safe
        if not self.xml:
            match = None
                
            if dirname.endswith('.zip'):
                pattern = os.path.join('*SAFE','annotation', swathid) + '*.xml'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()

                if (len(match) == 0):
                    raise Exception('No annotation xml file found in zip file: {0}'.format(dirname))

                ####Add /vsizip at the start to make it a zip file
                self.xml = '/vsizip/'+os.path.join(dirname, match[0]) 

            else:
                pattern = os.path.join('annotation',swathid)+'*.xml'
                match = glob.glob( os.path.join(dirname, pattern))

                if (len(match) == 0):
                    raise Exception('No annotation xml file found in {0}'.format(dirname))
            
                self.xml = match[0]

        if not self.xml:
            raise Exception('No annotation files found')

        print('Input XML file: ', self.xml)

        ####Find TIFF file
        if (not self.tiff) and (self.safe):
            match = None
            if swathid is None:
                swathid = 's1?-s?-slc-{}'.format(self.polarization)

            if dirname.endswith('.zip'):
                pattern = os.path.join('*SAFE','measurement', swathid) + '*.tiff'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()

                if (len(match) == 0):
                    raise Exception('No tiff file found in zip file: {0}'.format(dirname))

                ####Add /vsizip at the start to make it a zip file
                self.tiff = '/vsizip/' + os.path.join(dirname, match[0]) 


            else:
                pattern = os.path.join('measurement', swathid) + '*.tiff'
                match = glob.glob(os.path.join(dirname, pattern))

                if len(match) == 0 :
                    raise Exception('No tiff file found in directory: {0}'.format(dirname))

                self.tiff = match[0]

            print('Input TIFF files: ', self.tiff)


        ####Find manifest files
        if self.safe:
            if dirname.endswith('.zip'):
                pattern='*SAFE/manifest.safe'
                zf = zipfile.ZipFile(dirname, 'r')
                match = fnmatch.filter(zf.namelist(), pattern)
                zf.close()
                self.manifest = '/vsizip/' + os.path.join(dirname, match[0])
            else:
                self.manifest = os.path.join(dirname, 'manifest.safe')
    
            print('Manifest files: ', self.manifest)


        return
                                                
    def getFrame(self):
        return self.frame
    
    def parse(self):
        '''
        Actual parsing of the metadata for the product.
        '''
        ###Check user inputs
        self.validateUserInputs()

        if self.xml.startswith('/vsizip'):
            import zipfile
            parts = self.xml.split(os.path.sep)

            if parts[2] == '':
                parts[2] = os.path.sep

            zipname = os.path.join(*(parts[2:-3]))
            fname = os.path.join(*(parts[-3:]))

            zf = zipfile.ZipFile(zipname, 'r')
            xmlstr = zf.read(fname)
            zf.close()
        else:
            with open(self.xml,'r') as fid:
                xmlstr = fid.read()

        self._xml_root = ET.fromstring(xmlstr)                    
        self.populateMetadata()
    
        if self.manifest:
            self.populateIPFVersion()
        else:
            self.frame.setProcessingFacility('Tianyi')
            self.frame.setProcessingSoftwareVersion('IPFx.xx')

        if not self.orbitFile:
            if self.orbitDir:
                self.orbitFile = self._find_orbit_file_from_dir(
                    self.orbitDir, self.frame.sensingStart, self.frame.sensingStop
                )
                if self.orbitFile:
                    print('Using external orbit file: ', self.orbitFile)

        if self.orbitFile:
            orb = self.extractPreciseOrbit()
            if (orb is not None) and (len(orb) > 0):
                self.frame.orbit.setOrbitSource(os.path.basename(self.orbitFile))
            else:
                print('Warning: external orbit is empty/invalid, fallback to annotation orbit')
                orb = self.extractOrbitFromAnnotation()
                self.frame.orbit.setOrbitSource('Annotation')
        else:
            orb = self.extractOrbitFromAnnotation()
            self.frame.orbit.setOrbitSource('Annotation')

        for sv in orb:
            self.frame.orbit.addStateVector(sv)


    def grab_from_xml(self, path):
        node = self._xml_root.find(path)
        if (node is None) or (node.text is None):
            raise Exception('Tag = %s not found' % (path))

        return node.text.strip()

    def _grab_any_text(self, paths, default=None):
        for path in paths:
            node = self._xml_root.find(path)
            if (node is not None) and (node.text is not None):
                val = node.text.strip()
                if val != '':
                    return val
        return default

    def _grab_any_float(self, paths, default=None):
        text = self._grab_any_text(paths, default=None)
        if text is None:
            return default
        try:
            return float(text)
        except Exception:
            return default

    def _grab_any_int(self, paths, default=None):
        text = self._grab_any_text(paths, default=None)
        if text is None:
            return default
        try:
            return int(text)
        except Exception:
            return default

    def convertToDateTime(self, string):
        formats = (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        )
        for fmt in formats:
            try:
                return datetime.datetime.strptime(string, fmt)
            except ValueError:
                continue
        raise ValueError("Unsupported datetime string: {}".format(string))

    def _convert_orbit_time_text(self, string):
        text = string.strip()
        if text.startswith('UTC='):
            text = text[4:]
        elif text.startswith('UTC'):
            text = text.split('=', 1)[-1].strip()
        return self.convertToDateTime(text)

    def _parse_orbit_coverage_from_name(self, orbit_path):
        bname = os.path.basename(orbit_path)
        mtch = re.search(r'_(\d{8}T\d{6})_(\d{8}T\d{6})_', bname)
        if mtch is None:
            return None
        try:
            tstart = datetime.datetime.strptime(mtch.group(1), "%Y%m%dT%H%M%S")
            tstop = datetime.datetime.strptime(mtch.group(2), "%Y%m%dT%H%M%S")
        except ValueError:
            return None
        return (tstart, tstop)

    def _orbit_file_time_bounds(self, orbit_path):
        try:
            root = ET.parse(orbit_path).getroot()
        except Exception:
            return None

        node = root.find('Data_Block/List_of_OSVs')
        if node is None:
            node = root.find('.//List_of_OSVs')
        if node is None:
            return None

        tmin = None
        tmax = None
        for child in list(node):
            utc = child.findtext('UTC')
            if utc is None:
                utc = child.findtext('timeUTC')
            if utc is None:
                utc = child.findtext('time')
            if utc is None:
                continue
            try:
                timestamp = self._convert_orbit_time_text(utc)
            except Exception:
                continue
            if (tmin is None) or (timestamp < tmin):
                tmin = timestamp
            if (tmax is None) or (timestamp > tmax):
                tmax = timestamp

        if (tmin is None) or (tmax is None):
            return None

        return (tmin, tmax)

    def _find_orbit_file_from_dir(self, orbit_dir, sensing_start, sensing_stop):
        if (orbit_dir is None) or (not os.path.isdir(orbit_dir)):
            return None

        patterns = ['*.EOF', '*.eof', '*.XML', '*.xml']
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(orbit_dir, pat)))
            candidates.extend(glob.glob(os.path.join(orbit_dir, '**', pat), recursive=True))

        if len(candidates) == 0:
            return None

        margin = datetime.timedelta(seconds=120.0)
        tstart = sensing_start - margin
        tstop = sensing_stop + margin
        best = None
        best_span = None

        for orbit_path in sorted(set(candidates)):
            coverage = self._parse_orbit_coverage_from_name(orbit_path)
            if coverage is None:
                coverage = self._orbit_file_time_bounds(orbit_path)
            if coverage is None:
                continue

            cstart, cstop = coverage
            if (cstart <= tstart) and (cstop >= tstop):
                span = (cstop - cstart).total_seconds()
                if (best is None) or (span < best_span):
                    best = orbit_path
                    best_span = span

        return best

    def _get_tiff_shape(self):
        if not self.tiff:
            return None, None

        try:
            from osgeo import gdal
            src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
            if src is None:
                return None, None
            width = src.RasterXSize
            length = src.RasterYSize
            src = None
            return length, width
        except Exception:
            return None, None

    def _geodetic_to_ecef(self, lat_deg, lon_deg, height_m=0.0):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        a = 6378137.0
        e2 = 6.69437999014e-3
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (n + height_m) * cos_lat * cos_lon
        y = (n + height_m) * cos_lat * sin_lon
        z = (n * (1.0 - e2) + height_m) * sin_lat
        return np.array([x, y, z], dtype=np.float64)

    def _extract_orbit_records_from_annotation(self):
        records = []
        node = self._xml_root.find('generalAnnotation/orbitList')
        orbit_nodes = []
        if node is not None:
            orbit_nodes.extend(node.findall('orbit'))
        if len(orbit_nodes) == 0:
            orbit_nodes.extend(self._xml_root.findall('.//orbitList/orbit'))

        for child in orbit_nodes:
            try:
                timestamp = self.convertToDateTime(child.find('time').text.strip())
                posnode = child.find('position')
                velnode = child.find('velocity')
                pos = np.array(
                    [float(posnode.find(tag).text) for tag in ['x', 'y', 'z']],
                    dtype=np.float64,
                )
                vel = np.array(
                    [float(velnode.find(tag).text) for tag in ['x', 'y', 'z']],
                    dtype=np.float64,
                )
                records.append((timestamp, pos, vel))
            except Exception:
                continue

        return records

    def _infer_pass_direction_from_orbit(self, orbit_records):
        if not orbit_records:
            return None

        _, pos, vel = orbit_records[len(orbit_records) // 2]
        pos_norm = np.linalg.norm(pos)
        vel_norm = np.linalg.norm(vel)
        if (pos_norm <= 0.0) or (vel_norm <= 0.0):
            return None

        lon = math.atan2(pos[1], pos[0])
        lat = math.asin(pos[2] / pos_norm)
        north = np.array(
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            dtype=np.float64,
        )
        north_rate = float(np.dot(vel, north))
        return 'ASCENDING' if north_rate >= 0.0 else 'DESCENDING'

    
    def populateMetadata(self):
        """
            Create metadata objects from the metadata files
        """
        mission = self._grab_any_text(['adsHeader/missionId'], default='TIANYI')
        swath = self._grab_any_text(['adsHeader/swath'], default='S1')
        polarization = self._grab_any_text(['adsHeader/polarisation'], default='VV')

        frequency = self._grab_any_float(
            ['generalAnnotation/productInformation/radarFrequency'], default=5.4e9
        )

        rangeSamplingRate = self._grab_any_float(
            ['generalAnnotation/productInformation/rangeSamplingRate'], default=None
        )
        rangePixelSize = self._grab_any_float(
            ['imageAnnotation/imageInformation/rangePixelSpacing'], default=None
        )
        if (rangePixelSize is None) and (rangeSamplingRate is not None):
            rangePixelSize = Const.c / (2.0 * rangeSamplingRate)
        if (rangeSamplingRate is None) and (rangePixelSize is not None) and (rangePixelSize > 0.0):
            rangeSamplingRate = Const.c / (2.0 * rangePixelSize)
        if (rangeSamplingRate is None) or (rangePixelSize is None):
            rangeSamplingRate = 120e6
            rangePixelSize = Const.c / (2.0 * rangeSamplingRate)

        prf = self._grab_any_float(
            [
                'generalAnnotation/productInformation/prf',
                'generalAnnotation/downlinkInformationList/downlinkInformation/prf',
                'imageAnnotation/imageInformation/azimuthFrequency',
            ],
            default=None,
        )
        if prf is None:
            azimuth_dt = self._grab_any_float(
                ['imageAnnotation/imageInformation/azimuthTimeInterval'], default=None
            )
            prf = (1.0 / azimuth_dt) if (azimuth_dt and azimuth_dt > 0.0) else 4105.0903

        lines = self._grab_any_int(['imageAnnotation/imageInformation/numberOfLines'], default=None)
        samples = self._grab_any_int(['imageAnnotation/imageInformation/numberOfSamples'], default=None)
        tiff_lines, tiff_samples = self._get_tiff_shape()
        if lines is None:
            lines = tiff_lines if tiff_lines is not None else 1000
        if samples is None:
            samples = tiff_samples if tiff_samples is not None else 1000

        slant_range_time = self._grab_any_float(
            ['imageAnnotation/imageInformation/slantRangeTime'], default=None
        )
        reference_range = self._grab_any_float(
            ['imageAnnotation/processingInformation/referenceRange'], default=None
        )
        if slant_range_time is not None:
            startingRange = slant_range_time * Const.c / 2.0
        elif reference_range is not None:
            startingRange = reference_range
        else:
            startingRange = 800000.0

        incidenceAngle = self._grab_any_float(
            ['imageAnnotation/imageInformation/incidenceAngleMidSwath'], default=None
        )
        if incidenceAngle is None:
            geo_inc = []
            for node in self._xml_root.findall('.//geolocationGridPoint/incidenceAngle'):
                try:
                    geo_inc.append(float(node.text))
                except Exception:
                    continue
            incidenceAngle = float(np.mean(geo_inc)) if geo_inc else 30.0

        start_text = self._grab_any_text(
            ['adsHeader/startTime', 'imageAnnotation/imageInformation/productFirstLineUtcTime']
        )
        stop_text = self._grab_any_text(
            ['adsHeader/stopTime', 'imageAnnotation/imageInformation/productLastLineUtcTime']
        )
        if (start_text is None) or (stop_text is None):
            raise Exception('Could not find sensing start/stop time in Tianyi metadata')
        dataStartTime = self.convertToDateTime(start_text)
        dataStopTime = self.convertToDateTime(stop_text)

        pulseLength = self._grab_any_float(
            [
                'generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseLength',
                'imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/chirp_duration',
            ],
            default=2.68e-5,
        )
        chirpSlope = self._grab_any_float(
            [
                'generalAnnotation/downlinkInformationList/downlinkInformation/downlinkValues/txPulseRampRate',
                'imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/chirp_rate',
            ],
            default=None,
        )
        if (chirpSlope is None) or (abs(chirpSlope) < 1.0e3):
            range_bw = self._grab_any_float(
                [
                    'imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/processingBandwidth',
                    'imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/rangeProcessing/lookBandwidth',
                ],
                default=100e6,
            )
            chirpSlope = range_bw / pulseLength if pulseLength > 0.0 else 1.0e12
        pulseBandwidth = abs(pulseLength * chirpSlope)

        orbit_records = self._extract_orbit_records_from_annotation()
        pass_from_xml = self._grab_any_text(['generalAnnotation/productInformation/pass'], default=None)
        pass_from_xml = pass_from_xml.upper() if pass_from_xml else None
        pass_from_orbit = self._infer_pass_direction_from_orbit(orbit_records)
        passDirection = pass_from_xml
        if pass_from_orbit:
            if pass_from_xml and (pass_from_xml != pass_from_orbit):
                print(
                    'Warning: pass direction mismatch (xml={}, orbit={}), using orbit-derived value'.format(
                        pass_from_xml, pass_from_orbit
                    )
                )
            passDirection = pass_from_orbit
        if passDirection not in ['ASCENDING', 'DESCENDING']:
            passDirection = 'DESCENDING'

        lookSide = self.calculateLookDirection(passDirection=passDirection, orbit_records=orbit_records)

        # Populate platform and instrument information
        self.populatePlatformAndInstrument(mission, lookSide, frequency, prf, pulseLength, pulseBandwidth, incidenceAngle, rangePixelSize, rangeSamplingRate, swath)

        # Populate frame information
        self.populateFrame(dataStartTime, dataStopTime, passDirection, polarization, startingRange, samples, rangePixelSize, lines)

    def calculateLookDirection(self, passDirection=None, orbit_records=None):
        """
        Calculate look direction using XML fields first, then orbit/geometry inference.
        """
        explicit_look = self._grab_any_text(
            [
                'imageAnnotation/imageInformation/look_side',
                'imageAnnotation/imageInformation/lookSide',
                'generalAnnotation/productInformation/lookDirection',
            ],
            default=None,
        )
        if explicit_look:
            norm = explicit_look.strip().upper()
            if norm.startswith('L'):
                return lookMap['LEFT']
            if norm.startswith('R'):
                return lookMap['RIGHT']

        if passDirection is None:
            passDirection = self._grab_any_text(['generalAnnotation/productInformation/pass'], default='DESCENDING')
            passDirection = passDirection.upper()
        if orbit_records is None:
            orbit_records = self._extract_orbit_records_from_annotation()

        corners = self.extractCornerCoordinates()
        if corners and orbit_records:
            lat = 0.25 * (
                corners['top_left']['lat']
                + corners['top_right']['lat']
                + corners['bottom_left']['lat']
                + corners['bottom_right']['lat']
            )
            lon = 0.25 * (
                corners['top_left']['lon']
                + corners['top_right']['lon']
                + corners['bottom_left']['lon']
                + corners['bottom_right']['lon']
            )
            sat_pos = orbit_records[len(orbit_records) // 2][1]
            sat_vel = orbit_records[len(orbit_records) // 2][2]
            los = self._geodetic_to_ecef(lat, lon, 0.0) - sat_pos
            right_vec = np.cross(sat_vel, sat_pos)
            los_norm = np.linalg.norm(los)
            right_norm = np.linalg.norm(right_vec)
            if (los_norm > 0.0) and (right_norm > 0.0):
                score = float(np.dot(los / los_norm, right_vec / right_norm))
                if abs(score) > 1.0e-7:
                    return lookMap['RIGHT'] if score > 0.0 else lookMap['LEFT']

        if corners:
            left_lon = 0.5 * (corners['top_left']['lon'] + corners['bottom_left']['lon'])
            right_lon = 0.5 * (corners['top_right']['lon'] + corners['bottom_right']['lon'])
            delta_lon = right_lon - left_lon
            if abs(delta_lon) > 1.0e-7:
                if passDirection == 'ASCENDING':
                    return lookMap['RIGHT'] if delta_lon > 0.0 else lookMap['LEFT']
                return lookMap['RIGHT'] if delta_lon < 0.0 else lookMap['LEFT']

        print('Warning: could not infer look side from XML/orbit; fallback to LEFT for Tianyi')
        return lookMap['LEFT']

    def extractCornerCoordinates(self):
        """
        Extract corner coordinates from geolocation grid (or TIFF GCPs fallback).
        """
        points = []
        for point in self._xml_root.findall('.//geolocationGrid/geolocationGridPointList/geolocationGridPoint'):
            try:
                points.append(
                    {
                        'y': int(float(point.find('line').text)),
                        'x': int(float(point.find('pixel').text)),
                        'lat': float(point.find('latitude').text),
                        'lon': float(point.find('longitude').text),
                    }
                )
            except Exception:
                continue

        if len(points) < 4 and self.tiff:
            try:
                from osgeo import gdal

                src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
                if src is not None:
                    for gcp in src.GetGCPs():
                        points.append(
                            {
                                'x': int(round(gcp.GCPPixel)),
                                'y': int(round(gcp.GCPLine)),
                                'lat': float(gcp.GCPY),
                                'lon': float(gcp.GCPX),
                            }
                        )
                    src = None
            except Exception:
                pass

        if len(points) < 4:
            return None

        top_left = min(points, key=lambda p: (p['y'], p['x']))
        top_right = min(points, key=lambda p: (p['y'], -p['x']))
        bottom_left = min(points, key=lambda p: (-p['y'], p['x']))
        bottom_right = max(points, key=lambda p: (p['y'], p['x']))

        return {
            'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right,
        }

    def populatePlatformAndInstrument(self, mission, lookSide, frequency, prf, pulseLength, pulseBandwidth, incidenceAngle, rangePixelSize, rangeSamplingRate, swath):
        """
        Populate platform and instrument information
        """
        ####Populate platform
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname="Earth"))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(2.0)  # Default value

        ####Populate instrument
        instrument = self.frame.getInstrument()
        chirpSlope = pulseBandwidth / pulseLength if pulseLength > 0.0 else 1.0e12
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(chirpSlope)
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setBeamNumber(swath)
        instrument.setPulseLength(pulseLength)
        # Stripmap formSLC requires explicit IQ bias values.
        instrument.setInPhaseValue(0.0)
        instrument.setQuadratureValue(0.0)

    def populateFrame(self, dataStartTime, dataStopTime, passDirection, polarization, startingRange, samples, rangePixelSize, lines):
        """
        Populate frame information
        """
        #Populate Frame
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime)/2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization) 
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange + (samples-1)*rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        

    def extractOrbitFromAnnotation(self):
        '''
        Extract orbit information from xml node.
        '''
        frameOrbit = Orbit()
        frameOrbit.setOrbitSource('Header')
        orbit_records = self._extract_orbit_records_from_annotation()
        if not orbit_records:
            raise Exception('No valid orbit vectors found in Tianyi XML annotation')

        for timestamp, pos, vel in orbit_records:
            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(pos.tolist())
            vec.setVelocity(vel.tolist())
            frameOrbit.addStateVector(vec)

        try:
            planet = self.frame.instrument.platform.planet
            orbExt = OrbitExtender(planet=planet)
            orbExt.configure()
            newOrb = orbExt.extendOrbit(frameOrbit)
            return newOrb
        except ModuleNotFoundError as err:
            # Some container builds miss Orbit2sch. Annotation orbit is still usable.
            if 'Orbit2sch' in str(err):
                print('Info: Orbit2sch module is unavailable; using annotation orbit directly.')
                return frameOrbit
            print('Warning: Orbit extension failed, fallback to annotation orbit:', err)
            return frameOrbit
        except Exception as err:
            # Some builds do not include Stanford-licensed orbit extenders.
            # Fall back to annotation orbit to keep metadata parsing usable.
            print('Warning: Orbit extension failed, fallback to annotation orbit:', err)
            return frameOrbit

    def extractPreciseOrbit(self):
        '''
        Extract precise orbit from given Orbit file.
        '''
        try:
            fp = open(self.orbitFile,'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
            return None

        _xml_root = ET.ElementTree(file=fp).getroot()
       
        node = _xml_root.find('Data_Block/List_of_OSVs')
        if node is None:
            node = _xml_root.find('.//List_of_OSVs')
        if node is None:
            fp.close()
            return None

        orb = Orbit()
        orb.configure()

        margin = datetime.timedelta(seconds=40.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        for child in list(node):
            utc = child.findtext('UTC')
            if utc is None:
                utc = child.findtext('timeUTC')
            if utc is None:
                utc = child.findtext('time')
            if utc is None:
                continue
            try:
                timestamp = self._convert_orbit_time_text(utc)
            except Exception:
                continue

            if (timestamp >= tstart) and (timestamp < tend):

                pos = [] 
                vel = []

                for tag in ['VX', 'VY', 'VZ']:
                    val = child.findtext(tag)
                    if val is None:
                        val = child.findtext(tag.lower())
                    if val is None:
                        break
                    vel.append(float(val))

                for tag in ['X', 'Y', 'Z']:
                    val = child.findtext(tag)
                    if val is None:
                        val = child.findtext(tag.lower())
                    if val is None:
                        break
                    pos.append(float(val))

                if (len(pos) != 3) or (len(vel) != 3):
                    continue

                vec = StateVector()
                vec.setTime(timestamp)
                vec.setPosition(pos)
                vec.setVelocity(vel)
                orb.addStateVector(vec)

        fp.close()

        return orb

    def extractImage(self):
        """
           Use gdal python bindings to extract image
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for Tianyi.')

        self.parse()
        width = self.frame.getNumberOfSamples()
        lgth = self.frame.getNumberOfLines()

        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
        band = src.GetRasterBand(1)
        fid = open(self.output, 'wb')
        for ii in range(lgth):
            data = band.ReadAsArray(0,ii,width,1)
            data.tofile(fid)

        fid.close()
        src = None
        band = None

        ####
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setLength(self.frame.getNumberOfLines())
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        self.frame.setImage(slcImage)

    def extractDoppler(self):
        '''
        self.parse()
        Extract doppler information as needed by mocomp
        '''
        node = self._xml_root.find('dopplerCentroid/dcEstimateList')
        if node is None:
            node = self._xml_root.find('.//dopplerCentroid/dcEstimateList')
        if node is None:
            raise Exception('Could not find dopplerCentroid/dcEstimateList in Tianyi annotation XML')

        tdiff = 1.0e9
        dpoly = None

        for burst in node.findall('dcEstimate'):
            aztime = burst.findtext('azimuthTime')
            if aztime is None:
                continue
            try:
                refTime = self.convertToDateTime(aztime.strip())
            except Exception:
                continue

            delta = abs((refTime - self.frame.sensingMid).total_seconds())
            if delta >= tdiff:
                continue

            coeff_text = burst.findtext('dataDcPolynomial')
            if coeff_text is None:
                coeff_text = burst.findtext('geometryDcPolynomial')
            if coeff_text is None:
                continue

            try:
                coeffs = [float(val) for val in coeff_text.split()]
            except Exception:
                continue
            if len(coeffs) == 0:
                continue

            t0_text = burst.findtext('t0')
            if t0_text is not None:
                r0 = 0.5 * Const.c * float(t0_text)
                rnorm = 0.5 * Const.c
            else:
                r0 = self.frame.startingRange
                rnorm = max(1.0, self.frame.getInstrument().getRangePixelSize())

            poly = Poly1D.Poly1D()
            poly.initPoly(order=len(coeffs) - 1)
            poly.setMean(r0)
            poly.setNorm(rnorm)
            poly.setCoeffs(coeffs)

            dpoly = poly
            tdiff = delta

        if dpoly is None:
            raise Exception('Could not extract valid Doppler polynomial from Tianyi annotation XML')

        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        rmid = (
            self.frame.startingRange
            + 0.5 * self.frame.getNumberOfSamples() * self.frame.getInstrument().getRangePixelSize()
        )

        quadratic = {
            'a': dpoly(rmid) / prf if prf else 0.0,
            'b': 0.0,
            'c': 0.0,
        }

        npix = max(dpoly._order + 2, 4)
        pix = np.linspace(0, self.frame.getNumberOfSamples() - 1, num=npix)
        rngs = self.frame.startingRange + pix * self.frame.getInstrument().getRangePixelSize()
        evals = dpoly(rngs)
        fit = np.polyfit(pix, evals, dpoly._order)
        self.frame._dopplerVsPixel = list(fit[::-1])
        print('Doppler Fit : ', self.frame._dopplerVsPixel)

        return quadratic



    def populateIPFVersion(self):
        '''
        Get IPF version from the manifest file.
        '''

        try:
            if self.manifest.startswith('/vsizip'):
                import zipfile
                parts = self.manifest.split(os.path.sep)
                if parts[2] == '':
                    parts[2] = os.path.sep
                zipname = os.path.join(*(parts[2:-2]))
                fname = os.path.join(*(parts[-2:]))
                print('MANS: ', zipname, fname)

                zf = zipfile.ZipFile(zipname, 'r')
                xmlstr = zf.read(fname)

            else:
                with open(self.manifest, 'r') as fid:
                    xmlstr = fid.read()

            ####Setup namespace
            nsp = "{http://www.esa.int/safe/sentinel-1.0}"

            root = ET.fromstring(xmlstr)

            elem = root.find('.//metadataObject[@ID="processing"]')

            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility').attrib
            self.frame.setProcessingFacility(rdict['site'] +', '+ rdict['country'])

            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility/' + nsp + 'software').attrib

            self.frame.setProcessingSoftwareVersion(rdict['name'] + ' ' + rdict['version'])

        except:   ###Not a critical error ... continuing
            print('Could not read version number successfully from manifest file: ', self.manifest)
            pass

        return
