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

logger = logging.getLogger('isce.insar.runGeo2rdr')


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

    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)

    offsetsDir = self.insar.offsetsDirname
    os.makedirs(offsetsDir, exist_ok=True)

    grdr = createGeo2rdr()
    grdr.configure()

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

    grdr.rangeOffsetImageName = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    grdr.azimuthOffsetImageName = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)

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
    return


def runGeo2rdrGPU(self):
    from isceobj.Planet.Planet import Planet
    from iscesys import DateTimeUtil as DTU
    from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr

    logger.info('Running geo2rdr on GPU')

    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
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

    grdr.setOrbitMethod(0)
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

    logger.info('GPU geo2rdr completed.')
    return
