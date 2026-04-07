#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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






# giangi: taken Piyush code for snaphu and adapted

import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
import copy
import os
import logging
import numpy as np

logger = logging.getLogger('isce.insar.runUnwrapSnaphu')


def _infer_real_dtype(filename, width, length):
    nelem = int(width) * int(length)
    fsize = os.path.getsize(filename)
    if fsize == nelem * np.dtype(np.float32).itemsize:
        return np.float32
    if fsize == nelem * np.dtype(np.float64).itemsize:
        return np.float64
    return np.float32


def _infer_complex_dtype(filename, width, length):
    nelem = int(width) * int(length)
    fsize = os.path.getsize(filename)
    if fsize == nelem * np.dtype(np.complex64).itemsize:
        return np.complex64
    if fsize == nelem * np.dtype(np.complex128).itemsize:
        return np.complex128
    return np.complex64


def _image_shape_from_xml(xml_path):
    img = isceobj.createImage()
    img.load(xml_path)
    return int(img.getLength()), int(img.getWidth())


def _write_float_image(path, array2d):
    arr = np.asarray(array2d, dtype=np.float32)
    arr.tofile(path)

    out = isceobj.createImage()
    out.setFilename(path)
    out.setWidth(int(arr.shape[1]))
    out.setLength(int(arr.shape[0]))
    out.setDataType('FLOAT')
    out.bands = 1
    out.scheme = 'BIP'
    out.setAccessMode('read')
    out.renderHdr()
    out.renderVRT()


def _interpolate_masked_phase_nearest(phase, valid, radius):
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        logger.warning(
            'snaphuInterpMaskedPhase requested but scipy is unavailable; skip interpolation.'
        )
        return phase

    invalid = np.logical_not(valid)
    if not np.any(invalid):
        return phase

    dist, inds = distance_transform_edt(invalid, return_indices=True)
    filled = np.array(phase, copy=True)
    pick = invalid & (dist <= float(radius))
    if np.any(pick):
        rr = inds[0][pick]
        cc = inds[1][pick]
        filled[pick] = phase[rr, cc]
    return filled


def _prepare_snaphu_inputs(wrapName, corName, corr_threshold, do_interp, interp_radius):
    int_xml = wrapName + '.xml'
    cor_xml = corName + '.xml'

    length_i, width_i = _image_shape_from_xml(int_xml)
    length_c, width_c = _image_shape_from_xml(cor_xml)
    if (length_i != length_c) or (width_i != width_c):
        raise RuntimeError(
            'snaphu preprocess shape mismatch: interferogram=({0},{1}) coherence=({2},{3})'.format(
                length_i, width_i, length_c, width_c
            )
        )

    int_dtype = _infer_complex_dtype(wrapName, width_i, length_i)
    cor_dtype = _infer_real_dtype(corName, width_c, length_c)

    ifg = np.memmap(wrapName, dtype=int_dtype, mode='r', shape=(length_i, width_i))
    coh = np.memmap(corName, dtype=cor_dtype, mode='r', shape=(length_c, width_c))

    phase = np.angle(ifg).astype(np.float32)
    corr = np.asarray(coh, dtype=np.float32)
    corr = np.clip(corr, 0.0, 1.0)

    valid = np.isfinite(corr) & (corr >= float(corr_threshold)) & np.isfinite(phase)
    corr_masked = np.where(valid, corr, 0.0).astype(np.float32)
    phase_masked = np.where(valid, phase, 0.0).astype(np.float32)

    if bool(do_interp):
        phase_masked = _interpolate_masked_phase_nearest(phase_masked, valid, interp_radius)

    phase_file = wrapName + '.snaphu_phase'
    corr_file = corName + '.snaphu_corr'
    _write_float_image(phase_file, phase_masked)
    _write_float_image(corr_file, corr_masked)
    return phase_file, corr_file

def runSnaphu(self, igramSpectrum = "full", costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
   
    print("igramSpectrum: ", igramSpectrum)

    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)


    wrapName = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'

    corName = os.path.join(ifgDirname , self.insar.coherenceFilename)
    snaphu_input = wrapName
    snaphu_corr = corName
    snaphu_int_format = 'COMPLEX_DATA'

    if bool(getattr(self, 'snaphuGmtsarPreprocess', True)):
        corr_thr = float(getattr(self, 'snaphuCorrThreshold', 0.10))
        do_interp = bool(getattr(self, 'snaphuInterpMaskedPhase', False))
        interp_radius = int(getattr(self, 'snaphuInterpRadius', 300))
        logger.info(
            'snaphu GMTSAR-style preprocess enabled: corr_threshold=%.3f, interp=%s, interp_radius=%d',
            corr_thr,
            str(do_interp),
            interp_radius,
        )
        snaphu_input, snaphu_corr = _prepare_snaphu_inputs(
            wrapName,
            corName,
            corr_thr,
            do_interp,
            interp_radius,
        )
        snaphu_int_format = 'FLOAT_DATA'

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    wavelength = referenceFrame.getInstrument().getRadarWavelength()
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    #width      = self.insar.resampIntImage.width

    orbit = referenceFrame.orbit
    prf = referenceFrame.PRF
    elp = copy.copy(referenceFrame.instrument.platform.planet.ellipsoid)
    sv = orbit.interpolate(referenceFrame.sensingMid, method='hermite')
    hdg = orbit.getHeading()
    llh = elp.xyz_to_llh(sv.getPosition())
    elp.setSCH(llh[0], llh[1], hdg)

    earthRadius = elp.pegRadCur
    sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
    azimuthSpacing = vsch[0] * earthRadius / ((earthRadius + sch[2]) *prf)


    earthRadius = elp.pegRadCur
    altitude   = sch[2]
    rangeLooks = 1  # self.numberRangeLooks #insar.topo.numberRangeLooks
    azimuthLooks = 1 # self.numberAzimuthLooks #insar.topo.numberAzimuthLooks

    if not self.numberAzimuthLooks:
        self.numberAzimuthLooks = 1

    if not self.numberRangeLooks:
        self.numberRangeLooks = 1

    azres = referenceFrame.platform.antennaLength/2.0
    azfact = self.numberAzimuthLooks * azres / azimuthSpacing

    rBW = referenceFrame.instrument.pulseLength * referenceFrame.instrument.chirpSlope
    rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
    rngfact = rgres/referenceFrame.getInstrument().getRangePixelSize()

    corrLooks = self.numberRangeLooks * self.numberAzimuthLooks/(azfact*rngfact) 
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(snaphu_input)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(snaphu_corr)
    snp.setInitMethod(initMethod)
    #snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.setIntFileFormat(snaphu_int_format)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()
    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()
    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self.insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderHdr()
        connImage.renderVRT()

    return

'''
def runUnwrapMcf(self):
    runSnaphu(self, igramSpectrum = "full", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "low", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "high", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
'''

def runUnwrap(self, igramSpectrum = "full"):

    runSnaphu(self, igramSpectrum = igramSpectrum, costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)

    return

