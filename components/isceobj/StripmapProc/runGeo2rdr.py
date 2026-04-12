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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import logging
import os

import isceobj
import numpy as np

logger = logging.getLogger('isce.insar.runGeo2rdr')

_ORBIT_METHOD_CODES = {
    'HERMITE': 0,
    'SCH': 1,
    'LEGENDRE': 2,
}

_ORBIT_METHOD_ALIASES = {
    'HERMITE': 'HERMITE',
    'SCH': 'SCH',
    'LEGENDRE': 'LEGENDRE',
    'LAGRANGE': 'LEGENDRE',
    '0': 'HERMITE',
    '1': 'SCH',
    '2': 'LEGENDRE',
}


def _gpu_geo2rdr_available():
    try:
        from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr  # noqa: F401
        return True
    except Exception:
        return False


def _iter_orbit_state_vectors(orbit):
    svs = getattr(orbit, '_stateVectors', None)
    if svs is not None:
        return list(svs)

    svobj = getattr(orbit, 'stateVectors', None)
    if svobj is not None:
        if hasattr(svobj, 'list'):
            return list(svobj.list)
        return list(svobj)

    return []


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if sval in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    return bool(default)


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_orbit_method(value):
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    return _ORBIT_METHOD_ALIASES.get(token)


def _resolve_orbit_interpolation_method(self):
    env_geo2rdr = os.environ.get('ISCE_GEO2RDR_ORBIT_INTERPOLATION_METHOD')
    env_global = os.environ.get('ISCE_ORBIT_INTERPOLATION_METHOD')
    xml_value = getattr(self, 'orbitInterpolationMethod', None)
    chosen = (
        _normalize_orbit_method(env_geo2rdr)
        or _normalize_orbit_method(env_global)
        or _normalize_orbit_method(xml_value)
        or 'HERMITE'
    )
    logger.info(
        'geo2rdr orbit interpolation method: %s (env_geo2rdr=%s env_global=%s xml=%s)',
        chosen,
        str(env_geo2rdr),
        str(env_global),
        str(xml_value),
    )
    return chosen


def _orbit_method_code(method_name):
    return int(_ORBIT_METHOD_CODES.get(str(method_name).strip().upper(), 0))


def _normalization_applied(self):
    sec_prod = ''
    try:
        sec_prod = str(getattr(self._insar, 'secondarySlcCropProduct', '') or '')
    except Exception:
        sec_prod = ''
    if not sec_prod:
        return False
    token = os.path.basename(sec_prod).strip().lower()
    return ('_norm.xml' in token) or token.endswith('_norm')


def _infer_single_band_dtype(path, width, length):
    npx = int(width) * int(length)
    if npx <= 0:
        return np.float32
    try:
        size = os.path.getsize(path)
    except OSError:
        return np.float32

    f32 = npx * np.dtype(np.float32).itemsize
    f64 = npx * np.dtype(np.float64).itemsize
    if size == f32:
        return np.float32
    if size == f64:
        return np.float64
    logger.warning(
        'Cannot infer offset dtype from filesize: %s (size=%d, width=%d, length=%d). '
        'Fallback to float32.',
        path,
        int(size),
        int(width),
        int(length),
    )
    return np.float32


def _force_offset_to_valid_mean_if_normalized(self, offsetFilename, offset_name):
    # Hard-disabled to keep geo2rdr outputs untouched.
    # This blocks any in-place rewrite on both azimuth.off and range.off.
    logger.info(
        'Skipping normalization-aware %s post-processing: in-place offset edits are disabled.',
        offset_name,
    )
    return


def _force_azimuth_offset_to_valid_mean_if_normalized(self, azimuthOffsetFilename):
    _force_offset_to_valid_mean_if_normalized(self, azimuthOffsetFilename, 'azimuth.off')


def _force_range_offset_to_valid_mean_if_normalized(self, rangeOffsetFilename):
    _force_offset_to_valid_mean_if_normalized(self, rangeOffsetFilename, 'range.off')


def _integrated_external_enabled(self=None):
    env_value = os.environ.get('ISCE_EXTERNAL_REGISTRATION_ENABLED')
    if env_value is not None:
        return _parse_bool(env_value, default=True)
    if self is not None and hasattr(self, 'useExternalCoregistration'):
        try:
            return bool(getattr(self, 'useExternalCoregistration'))
        except Exception:
            pass
    return True


def _geo2rdr_input_frame(self):
    external_enabled = _integrated_external_enabled(self)
    use_reference_when_external = _parse_bool(
        os.environ.get('ISCE_GEO2RDR_USE_REFERENCE_WHEN_EXTERNAL'),
        default=False,
    )
    if external_enabled and use_reference_when_external:
        logger.info(
            'External registration enabled and ISCE_GEO2RDR_USE_REFERENCE_WHEN_EXTERNAL=1: '
            'geo2rdr input switched to referenceSlcCropProduct.'
        )
        return self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    if external_enabled:
        logger.info(
            'External registration enabled: geo2rdr keeps secondarySlcCropProduct '
            '(normal-clip when present) for geometry/flatten consistency.'
        )
    return self._insar.loadProduct(self._insar.secondarySlcCropProduct)


def runGeo2rdr(self):
    use_gpu = bool(getattr(self, 'useGPU', True))
    if use_gpu and _gpu_geo2rdr_available():
        try:
            return runGeo2rdrGPU(self)
        except Exception as err:
            logger.warning('GPU geo2rdr failed (%s). Falling back to CPU geo2rdr.', str(err), exc_info=True)
    elif use_gpu:
        logger.info('GPU geo2rdr requested but GPU geo2rdr module is unavailable. Using CPU geo2rdr.')

    return runGeo2rdrCPU(self)


def runGeo2rdrCPU(self):
    from zerodop.geo2rdr import createGeo2rdr

    logger.info('Running geo2rdr on CPU')

    info = _geo2rdr_input_frame(self)

    offsetsDir = self.insar.offsetsDirname
    os.makedirs(offsetsDir, exist_ok=True)

    grdr = createGeo2rdr()
    grdr.configure()
    orbit_method = _resolve_orbit_interpolation_method(self)

    planet = info.getInstrument().getPlatform().getPlanet()
    grdr.slantRangePixelSpacing = info.getInstrument().getRangePixelSize()
    grdr.prf = info.PRF
    grdr.radarWavelength = info.getInstrument().getRadarWavelength()
    grdr.orbit = info.getOrbit()
    grdr.width = info.getImage().getWidth()
    grdr.length = info.getImage().getLength()

    grdr.wireInputPort(name='planet', object=planet)
    grdr.lookSide = info.instrument.platform.pointingDirection

    grdr.setSensingStart(info.getSensingStart())
    grdr.rangeFirstSample = info.startingRange
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1

    if self.insar.secondaryGeometrySystem.lower().startswith('native'):
        p = [x / info.PRF for x in info._dopplerVsPixel]
    else:
        p = [0.]

    grdr.dopplerCentroidCoeffs = p
    grdr.fmrateCoeffs = [0.]
    grdr.orbitInterpolationMethod = orbit_method

    rangeOffsetFilename = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    azimuthOffsetFilename = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)
    grdr.rangeOffsetImageName = rangeOffsetFilename
    grdr.azimuthOffsetImageName = azimuthOffsetFilename

    latFilename = os.path.join(self.insar.geometryDirname, self.insar.latFilename + '.full')
    lonFilename = os.path.join(self.insar.geometryDirname, self.insar.lonFilename + '.full')
    heightFilename = os.path.join(self.insar.geometryDirname, self.insar.heightFilename + '.full')

    demImg = isceobj.createImage()
    demImg.load(heightFilename + '.xml')
    demImg.setAccessMode('READ')
    grdr.demImage = demImg

    latImg = isceobj.createImage()
    latImg.load(latFilename + '.xml')
    latImg.setAccessMode('READ')
    grdr.latImage = latImg

    lonImg = isceobj.createImage()
    lonImg.load(lonFilename + '.xml')
    lonImg.setAccessMode('READ')
    grdr.lonImage = lonImg
    grdr.outputPrecision = 'DOUBLE'

    grdr.geo2rdr()
    _force_azimuth_offset_to_valid_mean_if_normalized(self, azimuthOffsetFilename)
    return


def runGeo2rdrGPU(self):
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU
    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr

    logger.info('Running geo2rdr on GPU')

    info = _geo2rdr_input_frame(self)
    orbit_method = _resolve_orbit_interpolation_method(self)
    offsetsDir = self.insar.offsetsDirname
    os.makedirs(offsetsDir, exist_ok=True)

    latFilename = os.path.join(self.insar.geometryDirname, self.insar.latFilename + '.full')
    lonFilename = os.path.join(self.insar.geometryDirname, self.insar.lonFilename + '.full')
    heightFilename = os.path.join(self.insar.geometryDirname, self.insar.heightFilename + '.full')
    rangeOffsetFilename = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    azimuthOffsetFilename = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)

    latImage = isceobj.createImage()
    latImage.load(latFilename + '.xml')
    latImage.setAccessMode('READ')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.load(lonFilename + '.xml')
    lonImage.setAccessMode('READ')
    lonImage.createImage()

    demImage = isceobj.createImage()
    demImage.load(heightFilename + '.xml')
    demImage.setAccessMode('READ')
    demImage.createImage()

    grdr = PyGeo2rdr()
    grdr.setRangePixelSpacing(info.getInstrument().getRangePixelSize())
    grdr.setPRF(info.PRF)
    grdr.setRadarWavelength(info.getInstrument().getRadarWavelength())

    orbit = info.getOrbit()
    state_vectors = _iter_orbit_state_vectors(orbit)
    if len(state_vectors) == 0:
        raise RuntimeError('No orbit state vectors found for GPU geo2rdr.')

    grdr.createOrbit(0, len(state_vectors))
    for idx, sv in enumerate(state_vectors):
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
        grdr.setOrbitVector(idx, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

    grdr.setOrbitMethod(_orbit_method_code(orbit_method))
    grdr.setWidth(info.getImage().getWidth())
    grdr.setLength(info.getImage().getLength())
    grdr.setSensingStart(DTU.seconds_since_midnight(info.getSensingStart()))
    grdr.setRangeFirstSample(info.startingRange)
    grdr.setNumberRangeLooks(1)
    grdr.setNumberAzimuthLooks(1)

    planet = Planet(pname='Earth')
    grdr.setEllipsoidMajorSemiAxis(planet.ellipsoid.a)
    grdr.setEllipsoidEccentricitySquared(planet.ellipsoid.e2)

    if self.insar.secondaryGeometrySystem.lower().startswith('native'):
        p = [x / info.PRF for x in info._dopplerVsPixel]
    else:
        p = [0.]
    grdr.createPoly(len(p) - 1, 0.0, 1.0)
    for idx, coeff in enumerate(p):
        grdr.setPolyCoeff(idx, float(coeff))

    grdr.setDemLength(demImage.getLength())
    grdr.setDemWidth(demImage.getWidth())
    grdr.setBistaticFlag(0)

    rangeOffsetImage = isceobj.createImage()
    rangeOffsetImage.setFilename(rangeOffsetFilename)
    rangeOffsetImage.setAccessMode('write')
    rangeOffsetImage.setDataType('FLOAT')
    rangeOffsetImage.setCaster('write', 'DOUBLE')
    rangeOffsetImage.setWidth(demImage.getWidth())
    rangeOffsetImage.setLength(demImage.getLength())
    rangeOffsetImage.createImage()

    azimuthOffsetImage = isceobj.createImage()
    azimuthOffsetImage.setFilename(azimuthOffsetFilename)
    azimuthOffsetImage.setAccessMode('write')
    azimuthOffsetImage.setDataType('FLOAT')
    azimuthOffsetImage.setCaster('write', 'DOUBLE')
    azimuthOffsetImage.setWidth(demImage.getWidth())
    azimuthOffsetImage.setLength(demImage.getLength())
    azimuthOffsetImage.createImage()

    grdr.setLatAccessor(latImage.getImagePointer())
    grdr.setLonAccessor(lonImage.getImagePointer())
    grdr.setHgtAccessor(demImage.getImagePointer())
    grdr.setAzAccessor(0)
    grdr.setRgAccessor(0)
    grdr.setAzOffAccessor(azimuthOffsetImage.getImagePointer())
    grdr.setRgOffAccessor(rangeOffsetImage.getImagePointer())

    grdr.geo2rdr()

    rangeOffsetImage.finalizeImage()
    rangeOffsetImage.renderHdr()
    azimuthOffsetImage.finalizeImage()
    azimuthOffsetImage.renderHdr()
    latImage.finalizeImage()
    lonImage.finalizeImage()
    demImage.finalizeImage()
    _force_azimuth_offset_to_valid_mean_if_normalized(self, azimuthOffsetFilename)

    logger.info('GPU geo2rdr completed.')
    return
