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
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Util.Poly2D import Poly2D

logger = logging.getLogger('isce.insar.runTopo')

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


def _gpu_topo_available():
    try:
        from zerodop.GPUtopozero.GPUtopozero import PyTopozero  # noqa: F401
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


def _estimate_bbox_from_outputs(lat_filename, lon_filename, width, length):
    lat = np.memmap(lat_filename, dtype=np.float64, mode='r', shape=(length, width))
    lon = np.memmap(lon_filename, dtype=np.float64, mode='r', shape=(length, width))
    bbox = [
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
    ]
    del lat
    del lon
    return bbox


def _normalize_orbit_method(value):
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    return _ORBIT_METHOD_ALIASES.get(token)


def _resolve_orbit_interpolation_method(self):
    env_topo = os.environ.get('ISCE_TOPO_ORBIT_INTERPOLATION_METHOD')
    env_global = os.environ.get('ISCE_ORBIT_INTERPOLATION_METHOD')
    xml_value = getattr(self, 'orbitInterpolationMethod', None)
    chosen = (
        _normalize_orbit_method(env_topo)
        or _normalize_orbit_method(env_global)
        or _normalize_orbit_method(xml_value)
        or 'HERMITE'
    )
    logger.info(
        'topo orbit interpolation method: %s (env_topo=%s env_global=%s xml=%s)',
        chosen,
        str(env_topo),
        str(env_global),
        str(xml_value),
    )
    return chosen


def _orbit_method_code(method_name):
    return int(_ORBIT_METHOD_CODES.get(str(method_name).strip().upper(), 0))


def runTopo(self):
    use_gpu = bool(getattr(self, 'useGPU', True))
    if use_gpu and _gpu_topo_available():
        try:
            return runTopoGPU(self)
        except Exception as err:
            logger.warning('GPU topo failed (%s). Falling back to CPU topo.', str(err), exc_info=True)
    elif use_gpu:
        logger.info('GPU topo requested but GPU topozero module is unavailable. Using CPU topo.')

    return runTopoCPU(self)


def runTopoCPU(self):
    from zerodop.topozero import createTopozero

    logger.info('Running topo on CPU')
    orbit_method = _resolve_orbit_interpolation_method(self)

    geometryDir = self.insar.geometryDirname
    os.makedirs(geometryDir, exist_ok=True)

    demFilename = self.verifyDEM()
    objDem = isceobj.createDemImage()
    objDem.load(demFilename + '.xml')

    info = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    intImage = info.getImage()

    planet = info.getInstrument().getPlatform().getPlanet()
    topo = createTopozero()

    topo.slantRangePixelSpacing = 0.5 * SPEED_OF_LIGHT / info.rangeSamplingRate
    topo.prf = info.PRF
    topo.radarWavelength = info.radarWavelegth
    topo.orbit = info.orbit
    topo.width = intImage.getWidth()
    topo.length = intImage.getLength()
    topo.wireInputPort(name='dem', object=objDem)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.lookSide = info.getInstrument().getPlatform().pointingDirection
    topo.sensingStart = info.getSensingStart()
    topo.rangeFirstSample = info.startingRange

    topo.demInterpolationMethod = 'BIQUINTIC'
    topo.orbitInterpolationMethod = orbit_method
    topo.latFilename = os.path.join(geometryDir, self.insar.latFilename + '.full')
    topo.lonFilename = os.path.join(geometryDir, self.insar.lonFilename + '.full')
    topo.losFilename = os.path.join(geometryDir, self.insar.losFilename + '.full')
    topo.heightFilename = os.path.join(geometryDir, self.insar.heightFilename + '.full')

    dop = [x / 1.0 for x in info._dopplerVsPixel]
    doppler = Poly2D()
    doppler.setWidth(topo.width // topo.numberRangeLooks)
    doppler.setLength(topo.length // topo.numberAzimuthLooks)

    if self._insar.referenceGeometrySystem.lower().startswith('native'):
        doppler.initPoly(rangeOrder=len(dop) - 1, azimuthOrder=0, coeffs=[dop])
    else:
        doppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])

    topo.polyDoppler = doppler
    topo.topo()

    from isceobj.Catalog import recordInputsAndOutputs
    recordInputsAndOutputs(self._insar.procDoc, topo, 'runTopo', logger, 'runTopo')

    self._insar.estimatedBbox = [
        topo.minimumLatitude,
        topo.maximumLatitude,
        topo.minimumLongitude,
        topo.maximumLongitude,
    ]
    return topo


def runTopoGPU(self):
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU
    from zerodop.GPUtopozero.GPUtopozero import PyTopozero

    logger.info('Running topo on GPU')
    orbit_method = _resolve_orbit_interpolation_method(self)

    geometryDir = self.insar.geometryDirname
    os.makedirs(geometryDir, exist_ok=True)

    demFilename = self.verifyDEM()
    info = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    intImage = info.getImage()

    width = intImage.getWidth()
    length = intImage.getLength()
    r0 = info.startingRange
    dr = 0.5 * SPEED_OF_LIGHT / info.rangeSamplingRate
    t0 = info.getSensingStart()
    wvl = info.getInstrument().getRadarWavelength()
    look_side = info.getInstrument().getPlatform().pointingDirection

    latFilename = os.path.join(geometryDir, self.insar.latFilename + '.full')
    lonFilename = os.path.join(geometryDir, self.insar.lonFilename + '.full')
    losFilename = os.path.join(geometryDir, self.insar.losFilename + '.full')
    hgtFilename = os.path.join(geometryDir, self.insar.heightFilename + '.full')

    latImage = isceobj.createImage()
    latImage.initImage(latFilename, 'write', width, 'DOUBLE')
    latImage.createImage()

    lonImage = isceobj.createImage()
    lonImage.initImage(lonFilename, 'write', width, 'DOUBLE')
    lonImage.createImage()

    losImage = isceobj.createImage()
    losImage.initImage(losFilename, 'write', width, 'FLOAT', bands=2, scheme='BIL')
    losImage.setCaster('write', 'DOUBLE')
    losImage.createImage()

    heightImage = isceobj.createImage()
    heightImage.initImage(hgtFilename, 'write', width, 'DOUBLE')
    heightImage.createImage()

    demImage = isceobj.createDemImage()
    demImage.load(demFilename + '.xml')
    demImage.setCaster('read', 'FLOAT')
    demImage.createImage()

    doppler = Poly2D(name='stripmap_dopplerPoly')
    doppler.setWidth(width)
    doppler.setLength(length)
    doppler.setNormRange(1.0)
    doppler.setNormAzimuth(1.0)
    doppler.setMeanRange(0.0)
    doppler.setMeanAzimuth(0.0)
    if self._insar.referenceGeometrySystem.lower().startswith('native'):
        coeffs = [x / 1.0 for x in info._dopplerVsPixel]
        doppler.initPoly(rangeOrder=len(coeffs) - 1, azimuthOrder=0, coeffs=[coeffs])
    else:
        doppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
    doppler.createPoly2D()

    slantRangeImage = Poly2D(name='stripmap_slantRangePoly')
    slantRangeImage.setWidth(width)
    slantRangeImage.setLength(length)
    slantRangeImage.setNormRange(1.0)
    slantRangeImage.setNormAzimuth(1.0)
    slantRangeImage.setMeanRange(0.0)
    slantRangeImage.setMeanAzimuth(0.0)
    slantRangeImage.initPoly(rangeOrder=1, azimuthOrder=0, coeffs=[[r0, dr]])
    slantRangeImage.createPoly2D()

    orbit = info.getOrbit()
    state_vectors = _iter_orbit_state_vectors(orbit)
    if len(state_vectors) == 0:
        raise RuntimeError('No orbit state vectors found for GPU topo.')

    pegHdg = np.radians(orbit.getENUHeading(t0))
    elp = Planet(pname='Earth').ellipsoid

    topo = PyTopozero()
    topo.set_firstlat(demImage.getFirstLatitude())
    topo.set_firstlon(demImage.getFirstLongitude())
    topo.set_deltalat(demImage.getDeltaLatitude())
    topo.set_deltalon(demImage.getDeltaLongitude())
    topo.set_major(elp.a)
    topo.set_eccentricitySquared(elp.e2)
    topo.set_rSpace(dr)
    topo.set_r0(r0)
    topo.set_pegHdg(pegHdg)
    topo.set_prf(info.PRF)
    topo.set_t0(DTU.seconds_since_midnight(t0))
    topo.set_wvl(wvl)
    topo.set_thresh(.05)
    topo.set_demAccessor(demImage.getImagePointer())
    topo.set_dopAccessor(doppler.getPointer())
    topo.set_slrngAccessor(slantRangeImage.getPointer())
    topo.set_latAccessor(latImage.getImagePointer())
    topo.set_lonAccessor(lonImage.getImagePointer())
    topo.set_losAccessor(losImage.getImagePointer())
    topo.set_heightAccessor(heightImage.getImagePointer())
    topo.set_incAccessor(0)
    topo.set_maskAccessor(0)
    topo.set_numIter(25)
    topo.set_idemWidth(demImage.getWidth())
    topo.set_idemLength(demImage.getLength())
    topo.set_ilrl(look_side)
    topo.set_extraIter(10)
    topo.set_length(length)
    topo.set_width(width)
    topo.set_nRngLooks(1)
    topo.set_nAzLooks(1)
    topo.set_demMethod(5)
    topo.set_orbitMethod(_orbit_method_code(orbit_method))

    topo.set_orbitNvecs(len(state_vectors))
    topo.set_orbitBasis(1)
    topo.createOrbit()
    for idx, sv in enumerate(state_vectors):
        td = DTU.seconds_since_midnight(sv.getTime())
        pos = sv.getPosition()
        vel = sv.getVelocity()
        topo.set_orbitVector(idx, td, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

    topo.runTopo()

    latImage.finalizeImage()
    latImage.renderHdr()
    lonImage.finalizeImage()
    lonImage.renderHdr()
    heightImage.finalizeImage()
    heightImage.renderHdr()
    losImage.setImageType('bil')
    losImage.finalizeImage()
    losImage.renderHdr()
    demImage.finalizeImage()

    try:
        slantRangeImage.finalizeImage()
    except Exception:
        pass

    bbox = _estimate_bbox_from_outputs(latFilename, lonFilename, width, length)
    self._insar.estimatedBbox = bbox
    logger.info('GPU topo completed. Estimated bbox: %s', bbox)
    return topo
