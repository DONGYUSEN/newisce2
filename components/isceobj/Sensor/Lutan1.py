#!/usr/bin/python3

# Reader for Lutan-1 SLC data
# Used Sentinel1.py and ALOS.py as templates
# Author: Bryan Marfito, EOS-RS


import os
import glob
import zipfile
import numpy as np
import xml.etree.ElementTree as ET
import datetime
import isce
import isceobj
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Sensor.Sensor import Sensor
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from isceobj.Orbit.OrbitExtender import OrbitExtender
from osgeo import gdal
import warnings

try:
    from scipy.interpolate import UnivariateSpline
except Exception:
    UnivariateSpline = None

lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

# Antenna dimensions 9.8 x 3.4 m
antennaLength = 9.8

XML = Component.Parameter('xml',
        public_name = 'xml',
        default = None,
        type = str,
        doc = 'Input XML file')


TIFF = Component.Parameter('tiff',
                            public_name ='tiff',
                            default = None,
                            type=str,
                            doc = 'Input image file')

ORBIT_FILE = Component.Parameter('orbitFile',
                            public_name ='orbitFile',
                            default = None,
                            type=str,
                            doc = 'Orbit file')

SAFE = Component.Parameter(
    'safe',
    public_name='safe',
    default=None,
    type=str,
    doc='Lutan product directory or zip file'
)

ORBIT_FILTER = Component.Parameter(
    'orbitFilter',
    public_name='orbit filter',
    default=True,
    type=bool,
    mandatory=False,
    doc='Enable robust smoothing filter for Lutan orbit state vectors.'
)

ORBIT_FILTER_DEGREE = Component.Parameter(
    'orbitFilterDegree',
    public_name='orbit filter degree',
    default=5,
    type=int,
    mandatory=False,
    doc='Spline/polynomial degree for orbit filtering.'
)

ORBIT_FILTER_SIGMA = Component.Parameter(
    'orbitFilterSigma',
    public_name='orbit filter sigma',
    default=4.0,
    type=float,
    mandatory=False,
    doc='Outlier rejection threshold in robust sigma units.'
)

ORBIT_FILTER_MAXITER = Component.Parameter(
    'orbitFilterMaxIter',
    public_name='orbit filter max iter',
    default=3,
    type=int,
    mandatory=False,
    doc='Maximum robust iterations for orbit filtering.'
)

ORBIT_FILTER_IGNORE_START = Component.Parameter(
    'orbitFilterIgnoreStart',
    public_name='orbit filter ignore start',
    default=-1,
    type=int,
    mandatory=False,
    doc='Number of state vectors at the beginning excluded from fit (-1: auto).'
)

ORBIT_FILTER_IGNORE_END = Component.Parameter(
    'orbitFilterIgnoreEnd',
    public_name='orbit filter ignore end',
    default=-1,
    type=int,
    mandatory=False,
    doc='Number of state vectors at the end excluded from fit (-1: auto).'
)


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.Lutan1'

    parameter_list = (
        SAFE,
        TIFF,
        ORBIT_FILE,
        ORBIT_FILTER,
        ORBIT_FILTER_DEGREE,
        ORBIT_FILTER_SIGMA,
        ORBIT_FILTER_MAXITER,
        ORBIT_FILTER_IGNORE_START,
        ORBIT_FILTER_IGNORE_END,
    ) + Sensor.parameter_list

    def __init__(self, name = ''):
        super(Lutan1,self).__init__(self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._xml_root = None
        self.doppler_coeff = None

    @staticmethod
    def _is_tiff_name(name):
        low = name.lower()
        return low.endswith('.tif') or low.endswith('.tiff')

    @staticmethod
    def _meta_from_tiff_name(name):
        base, _ = os.path.splitext(name)
        return base + '.meta.xml'

    def _resolve_inputs_from_safe(self):
        if not self.safe:
            return

        safe = self.safe.rstrip('/')
        self.safe = safe

        if safe.lower().endswith('.zip'):
            zabs = os.path.abspath(safe)
            with zipfile.ZipFile(zabs, 'r') as zf:
                members = zf.namelist()
            by_lower = {m.lower(): m for m in members}

            tiffs = sorted([m for m in members if (not m.endswith('/')) and self._is_tiff_name(m)])
            if len(tiffs) == 0:
                raise Exception('No TIFF file found in zip: {}'.format(safe))

            chosen_tiff = None
            chosen_xml = None
            for tif in tiffs:
                meta = self._meta_from_tiff_name(tif)
                if meta.lower() in by_lower:
                    chosen_tiff = tif
                    chosen_xml = by_lower[meta.lower()]
                    break

            if chosen_tiff is None:
                chosen_tiff = tiffs[0]
                meta = self._meta_from_tiff_name(chosen_tiff)
                if meta.lower() in by_lower:
                    chosen_xml = by_lower[meta.lower()]

            if self.tiff is None:
                self.tiff = '/vsizip/{}/{}'.format(zabs, chosen_tiff.lstrip('/'))

            if self.xml is None:
                if chosen_xml is None:
                    raise Exception(
                        'No matching .meta.xml found for TIFF {} in zip {}'.format(chosen_tiff, safe)
                    )
                self.xml = '/vsizip/{}/{}'.format(zabs, chosen_xml.lstrip('/'))
            return

        if os.path.isdir(safe):
            tiffs = sorted(glob.glob(os.path.join(safe, '**', '*.tif'), recursive=True))
            tiffs += sorted(glob.glob(os.path.join(safe, '**', '*.tiff'), recursive=True))
            if len(tiffs) == 0:
                raise Exception('No TIFF file found in directory: {}'.format(safe))

            chosen_tiff = None
            chosen_xml = None
            for tif in tiffs:
                meta = self._meta_from_tiff_name(tif)
                if os.path.isfile(meta):
                    chosen_tiff = tif
                    chosen_xml = meta
                    break

            if chosen_tiff is None:
                chosen_tiff = tiffs[0]
                meta = self._meta_from_tiff_name(chosen_tiff)
                if os.path.isfile(meta):
                    chosen_xml = meta

            if self.tiff is None:
                self.tiff = chosen_tiff

            if self.xml is None:
                if chosen_xml is None:
                    raise Exception(
                        'No matching .meta.xml found for TIFF {} in directory {}'.format(chosen_tiff, safe)
                    )
                self.xml = chosen_xml
            return

        raise Exception('safe path does not exist or is invalid: {}'.format(safe))

    def parse(self):
        if (self.tiff is None or self.xml is None) and self.safe:
            self._resolve_inputs_from_safe()

        if self.xml is None:
            if self.tiff is None:
                raise Exception('Provide safe, or provide tiff/xml for Lutan1 parser.')
            tifBase, tifExt = os.path.splitext(self.tiff)
            if tifExt.lower() not in ['.tif', '.tiff']:
                raise Exception('Unexpected TIFF extension for Lutan1 input: {}'.format(self.tiff))
            self.xml = tifBase + '.meta.xml'

        if self.xml.startswith('/vsizip/'):
            vsipath = self.xml[len('/vsizip/'):]
            if '.zip/' not in vsipath:
                raise Exception('Invalid /vsizip xml path: {}'.format(self.xml))

            zipname, member = vsipath.split('.zip/', 1)
            zipname = zipname + '.zip'
            if not zipname.startswith('/'):
                zipname = '/' + zipname

            with zipfile.ZipFile(zipname, 'r') as zf:
                xmlbytes = zf.read(member)
            xmlstr = xmlbytes.decode('utf-8', errors='replace')
        else:
            with open(self.xml, 'r') as fid:
                xmlstr = fid.read()

        self._xml_root = ET.fromstring(xmlstr)
        self.populateMetadata()

        if self.orbitFile:
            # Check if orbit file exists or not
            if os.path.isfile(self.orbitFile) == True:
                orb = self.extractOrbit()
                self.frame.orbit.setOrbitSource(os.path.basename(self.orbitFile))
            else:
                warnings.warn(
                    "WARNING! orbitFile is set but not found. Falling back to annotation orbit."
                )
                orb = self.extractOrbitFromAnnotation()
                self.frame.orbit.setOrbitSource('Annotation')
        else:
            warnings.warn("WARNING! No orbit file found. Orbit information from the annotation file is used for processing.")
            orb = self.extractOrbitFromAnnotation()
            self.frame.orbit.setOrbitSource(os.path.basename(self.xml))
            self.frame.orbit.setOrbitSource('Annotation')

        if self.orbitFilter:
            orb = self._filterOrbit(orb)

        for sv in orb:
            self.frame.orbit.addStateVector(sv)

    def convertToDateTime(self,string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt

    @staticmethod
    def _robustScale(values):
        vals = np.asarray(values, dtype=np.float64)
        if vals.size == 0:
            return 0.0
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad > 0:
            return 1.4826 * mad
        std = np.std(vals)
        return float(std) if np.isfinite(std) else 0.0

    def _evaluateFit(self, t, y, mask, degree):
        idx = np.where(mask)[0]
        if idx.size < 3:
            return y.copy(), 'none'

        deg = max(1, min(int(degree), idx.size - 1, 5))
        tx = t[idx]
        yy = y[idx]

        if UnivariateSpline is not None and tx.size >= deg + 2:
            try:
                txu, iu = np.unique(tx, return_index=True)
                yyu = yy[iu]
                if txu.size >= deg + 2:
                    rough = np.diff(yyu)
                    noise = self._robustScale(rough) * np.sqrt(2.0) if rough.size > 1 else self._robustScale(yyu)
                    if not np.isfinite(noise) or noise <= 0:
                        noise = self._robustScale(yyu - np.median(yyu))
                    s_val = float(txu.size) * (noise ** 2) if (np.isfinite(noise) and noise > 0) else 0.0
                    sp = UnivariateSpline(txu, yyu, k=deg, s=s_val)
                    return sp(t), 'spline'
            except Exception:
                pass

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coeff = np.polyfit(tx, yy, deg)
        return np.polyval(coeff, t), 'poly'

    def _robustFitComponent(self, t, y, baseMask):
        mask = baseMask.copy()
        fit = y.copy()
        method_used = 'none'
        sigma = max(1.0, float(self.orbitFilterSigma))
        max_iter = max(1, int(self.orbitFilterMaxIter))
        degree = max(1, int(self.orbitFilterDegree))

        for _ in range(max_iter):
            fit, method = self._evaluateFit(t, y, mask, degree)
            method_used = method
            resid = y - fit
            scale = self._robustScale(resid[mask])
            if (not np.isfinite(scale)) or (scale <= 0):
                break
            new_mask = baseMask & (np.abs(resid) <= sigma * scale)
            if new_mask.sum() < max(3, degree + 1):
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        fit, method = self._evaluateFit(t, y, mask, degree)
        method_used = method
        return fit, mask, method_used

    def _resolveIgnoreCounts(self, nvec):
        ig_start = int(self.orbitFilterIgnoreStart)
        ig_end = int(self.orbitFilterIgnoreEnd)
        degree = max(1, int(self.orbitFilterDegree))

        if ig_start < 0:
            ig_start = max(1, int(round(0.02 * nvec))) if nvec >= 20 else 1
        if ig_end < 0:
            ig_end = max(1, int(round(0.05 * nvec))) if nvec >= 20 else 1

        ig_start = max(0, ig_start)
        ig_end = max(0, ig_end)

        min_keep = max(6, degree + 1)
        max_drop = max(0, nvec - min_keep)
        if (ig_start + ig_end) > max_drop:
            if max_drop == 0:
                ig_start, ig_end = 0, 0
            else:
                total = ig_start + ig_end
                if total <= 0:
                    ig_start, ig_end = 0, 0
                else:
                    ig_start = int(round(max_drop * (ig_start / float(total))))
                    ig_end = max_drop - ig_start

        return ig_start, ig_end

    def _filterOrbit(self, orb):
        vecs = list(orb)
        nvec = len(vecs)
        min_needed = max(8, int(self.orbitFilterDegree) + 2)
        if nvec < min_needed:
            warnings.warn(
                'Lutan1 orbit filter skipped: only {} state vectors (need >= {}).'.format(nvec, min_needed)
            )
            return vecs

        t0 = vecs[0].getTime()
        t = np.array([(sv.getTime() - t0).total_seconds() for sv in vecs], dtype=np.float64)
        pos = np.array([sv.getPosition() for sv in vecs], dtype=np.float64)
        vel = np.array([sv.getVelocity() for sv in vecs], dtype=np.float64)

        base_mask = np.ones(nvec, dtype=bool)
        ig_start, ig_end = self._resolveIgnoreCounts(nvec)
        if ig_start > 0:
            base_mask[:min(ig_start, nvec)] = False
        if ig_end > 0:
            base_mask[max(0, nvec - ig_end):] = False

        if base_mask.sum() < max(6, int(self.orbitFilterDegree) + 1):
            warnings.warn(
                'Lutan1 orbit filter skipped: usable vectors after ignore_start/end are insufficient.'
            )
            return vecs

        pos_f = np.zeros_like(pos)
        vel_f = np.zeros_like(vel)
        pos_masks = []
        vel_masks = []
        methods = []

        for comp in range(3):
            fit, mask, method = self._robustFitComponent(t, pos[:, comp], base_mask)
            pos_f[:, comp] = fit
            pos_masks.append(mask)
            methods.append('pos_{}={}'.format(comp, method))

        for comp in range(3):
            fit, mask, method = self._robustFitComponent(t, vel[:, comp], base_mask)
            vel_f[:, comp] = fit
            vel_masks.append(mask)
            methods.append('vel_{}={}'.format(comp, method))

        pos_ok = pos_masks[0] & pos_masks[1] & pos_masks[2]
        vel_ok = vel_masks[0] & vel_masks[1] & vel_masks[2]
        outliers = base_mask & (~pos_ok | ~vel_ok)
        n_out = int(np.count_nonzero(outliers))
        if n_out > 0:
            warnings.warn(
                'Lutan1 orbit filter detected {} outliers (ignore_start={}, ignore_end={}, degree={}).'.format(
                    n_out, ig_start, ig_end, int(self.orbitFilterDegree)
                )
            )

        if UnivariateSpline is None:
            warnings.warn(
                'SciPy unavailable: Lutan1 orbit filter used polynomial fallback instead of smoothing spline.'
            )

        print(
            'Lutan1 orbit filter settings: degree={}, ignore_start={}, ignore_end={}'.format(
                int(self.orbitFilterDegree), ig_start, ig_end
            )
        )
        print('Lutan1 orbit filter methods:', ', '.join(methods))

        out_vecs = []
        for i, sv in enumerate(vecs):
            new_sv = StateVector()
            new_sv.setTime(sv.getTime())
            new_sv.setPosition(pos_f[i, :].tolist())
            new_sv.setVelocity(vel_f[i, :].tolist())
            out_vecs.append(new_sv)
        return out_vecs


    def grab_from_xml(self, path):
        try:
            res = self._xml_root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))
        
        return res
    

    def populateMetadata(self):
        mission = self.grab_from_xml('generalHeader/mission')
        polarization = self.grab_from_xml('productInfo/acquisitionInfo/polarisationMode')
        frequency = float(self.grab_from_xml('instrument/radarParameters/centerFrequency'))
        passDirection = self.grab_from_xml('productInfo/missionInfo/orbitDirection')
        rangePixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/columnSpacing'))
        azimuthPixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/rowSpacing'))
        rangeSamplingRate = Const.c/(2.0*rangePixelSize)

        prf = float(self.grab_from_xml('instrument/settings/settingRecord/PRF'))
        lines = int(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/numberOfRows'))
        samples = int(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/numberOfColumns'))

        startingRange = float(self.grab_from_xml('productInfo/sceneInfo/rangeTime/firstPixel'))*Const.c/2.0
        #slantRange = float(self.grab_from_xml('productSpecific/complexImageInfo/'))
        incidenceAngle = float(self.grab_from_xml('productInfo/sceneInfo/sceneCenterCoord/incidenceAngle'))
        dataStartTime = self.convertToDateTime(self.grab_from_xml('productInfo/sceneInfo/start/timeUTC'))
        dataStopTime = self.convertToDateTime(self.grab_from_xml('productInfo/sceneInfo/stop/timeUTC'))
        pulseLength = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseLength'))
        pulseBandwidth = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseBandwidth'))
        chirpSlope = pulseBandwidth/pulseLength

        if self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/chirpSlope') == "DOWN":
            chirpSlope = -1.0 * chirpSlope
        else:
            pass

        # Check for satellite's look direction
        if self.grab_from_xml('productInfo/acquisitionInfo/lookDirection') == "LEFT":
            lookSide = lookMap['LEFT']
            print("Look direction: LEFT")
        else:
            lookSide = lookMap['RIGHT']
            print("Look direction: RIGHT")

        processingFacility = self.grab_from_xml('productInfo/generationInfo/level1ProcessingFacility')

        # Platform parameters
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname='Earth'))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(antennaLength)

        # Instrument parameters
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(chirpSlope)
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setPulseLength(pulseLength)

        # Frame parameters
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        self.frame.setProcessingFacility(processingFacility)

        # Two-way travel time 
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime) / 2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange +  (samples - 1) * rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)

        return


    def extractOrbit(self):

        '''
        Extract orbit information from the orbit file
        '''

        try:
            fp = open(self.orbitFile, 'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
        
        _xml_root = ET.ElementTree(file=fp).getroot()
        node = _xml_root.find('Data_Block/List_of_OSVs')

        orb = Orbit()
        orb.configure()

        # I based the margin on the data that I have.
        # Lutan-1 position and velocity sampling frequency is 1 Hz
        margin = datetime.timedelta(seconds=1.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin
        
        for child in node:
            timestamp = self.convertToDateTime(child.find('UTC').text)
            if (timestamp >= tstart) and (timestamp <= tend):
                pos = []
                vel = []
                for tag in ['VX', 'VY', 'VZ']:
                    vel.append(float(child.find(tag).text))

                for tag in ['X', 'Y', 'Z']:
                    pos.append(float(child.find(tag).text))

                vec = StateVector()
                vec.setTime(timestamp)
                vec.setPosition(pos)
                vec.setVelocity(vel)
                orb.addStateVector(vec)

        fp.close()

        return orb
    
    def extractOrbitFromAnnotation(self):

        '''
        Extract orbit information from xml annotation
        WARNING! Only use this method if orbit file is not available
        '''

        try:
            fp = open(self.xml, 'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)

        _xml_root = ET.ElementTree(file=fp).getroot()
        node = _xml_root.find('platform/orbit')
        countNode = len(list(_xml_root.find('platform/orbit')))

        frameOrbit = Orbit()
        frameOrbit.setOrbitSource('Header')
        margin = datetime.timedelta(seconds=1.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        for k in range(1,countNode):
            timestamp = self.convertToDateTime(node.find('stateVec[{}]/timeUTC'.format(k)).text)
            if (timestamp >= tstart) and (timestamp <= tend):
                pos = [float(node.find('stateVec[{}]/posX'.format(k)).text), float(node.find('stateVec[{}]/posY'.format(k)).text), float(node.find('stateVec[{}]/posZ'.format(k)).text)]
                vel = [float(node.find('stateVec[{}]/velX'.format(k)).text), float(node.find('stateVec[{}]/velY'.format(k)).text), float(node.find('stateVec[{}]/velZ'.format(k)).text)]

                vec = StateVector()
                vec.setTime(timestamp)
                vec.setPosition(pos)
                vec.setVelocity(vel)
                frameOrbit.addStateVector(vec)
        
        fp.close()
        return frameOrbit
    
    def extractImage(self):
        self.parse()
        width = self.frame.getNumberOfSamples()
        lgth = self.frame.getNumberOfLines()
        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)

        # Band 1 as real and band 2 as imaginary numbers
        # Confirmed by Zhang Yunjun
        band1 = src.GetRasterBand(1)
        band2 = src.GetRasterBand(2)
        cJ = np.complex64(1.0j)

        fid = open(self.output, 'wb')
        for ii in range(lgth):
            # Combine the real and imaginary to make
            # them in to complex numbers
            real = band1.ReadAsArray(0,ii,width,1)
            imag = band2.ReadAsArray(0,ii,width,1)
            # Data becomes np.complex128 after combining them
            data = real + (cJ * imag)
            data.tofile(fid)

        fid.close()
        real = None
        imag = None
        src = None
        band1 = None
        band2 = None

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
        Extract Doppler polynomial from annotation metadata and convert it to
        Doppler-vs-pixel coefficients expected by ISCE processors.
        '''
        if self._xml_root is None:
            self.parse()

        from isceobj.Util import Poly1D

        dc_root = self._xml_root.find('processing/doppler/dopplerCentroid')
        if dc_root is None:
            dc_root = self._xml_root.find('.//processing/doppler/dopplerCentroid')
        if dc_root is None:
            raise Exception('Could not find processing/doppler/dopplerCentroid in Lutan1 metadata.')

        estimates = dc_root.findall('dopplerEstimate')
        if len(estimates) == 0:
            single = dc_root.find('dopplerEstimate')
            if single is not None:
                estimates = [single]

        if len(estimates) == 0:
            raise Exception('Could not find dopplerEstimate records in Lutan1 metadata.')

        tdiff = 1.0e99
        dpoly = None
        selected_coeffs = None

        for est in estimates:
            # Prefer combinedDoppler, fallback to baseband/geometric when needed.
            dop_node = est.find('combinedDoppler')
            if dop_node is None:
                dop_node = est.find('basebandDoppler')
            if dop_node is None:
                dop_node = est.find('geometricDoppler')
            if dop_node is None:
                continue

            coeff_nodes = dop_node.findall('coefficient')
            if len(coeff_nodes) > 0:
                max_exp = -1
                parsed = {}
                for cnode in coeff_nodes:
                    try:
                        exp = int(cnode.attrib.get('exponent', '0'))
                        val = float(cnode.text)
                    except Exception:
                        continue
                    parsed[exp] = val
                    if exp > max_exp:
                        max_exp = exp
                if max_exp < 0:
                    continue
                coeffs = [0.0] * (max_exp + 1)
                for exp, val in parsed.items():
                    coeffs[exp] = val
            else:
                coeff_text = dop_node.findtext('dataDcPolynomial')
                if coeff_text is None:
                    coeff_text = dop_node.findtext('geometryDcPolynomial')
                if coeff_text is None:
                    continue
                try:
                    coeffs = [float(val) for val in coeff_text.split()]
                except Exception:
                    continue

            if len(coeffs) == 0:
                continue

            ref_text = dop_node.findtext('referencePoint')
            if ref_text is None:
                continue

            try:
                tref = float(ref_text)  # two-way range time in seconds
            except Exception:
                continue

            etime = est.findtext('timeUTC')
            if etime is None:
                delta = 0.0
            else:
                try:
                    delta = abs((self.convertToDateTime(etime.strip()) - self.frame.sensingMid).total_seconds())
                except Exception:
                    delta = 0.0

            if delta >= tdiff:
                continue

            rref = 0.5 * Const.c * tref
            poly = Poly1D.Poly1D()
            poly.initPoly(order=len(coeffs) - 1)
            poly.setMean(rref)
            poly.setNorm(0.5 * Const.c)
            poly.setCoeffs(coeffs)

            dpoly = poly
            selected_coeffs = list(coeffs)
            tdiff = delta

        if dpoly is None:
            raise Exception('Could not extract valid Doppler polynomial from Lutan1 metadata.')

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
        self.doppler_coeff = list(selected_coeffs) if selected_coeffs is not None else None
        print("Average doppler quadratic(a, b, c): ", quadratic['a'], quadratic['b'], quadratic['c'])
        print("Doppler Fit : ", self.frame._dopplerVsPixel)

        return quadratic
