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
import logging
import os
import numpy as np
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT

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


def runUnwrap(self,costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
    
    wrapName = self.insar.topophaseFlatFilename
    unwrapName = self.insar.unwrappedIntFilename

    wavelength = self.insar.referenceFrame.getInstrument().getRadarWavelength()
    width      = self.insar.resampIntImage.width 
    earthRadius = self.insar.peg.radiusOfCurvature 
    altitude   = self.insar.averageHeight
    corrfile  = self.insar.getCoherenceFilename()
    snaphu_input = wrapName
    snaphu_corr = corrfile
    snaphu_int_format = None
    snaphu_cor_format = None
    if bool(getattr(self, 'snaphuGmtsarPreprocess', True)):
        corr_thr = float(getattr(self, 'snaphuCorrThreshold', 0.20))
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
            corrfile,
            corr_thr,
            do_interp,
            interp_radius,
        )
        snaphu_int_format = 'FLOAT_DATA'
        snaphu_cor_format = 'FLOAT_DATA'

    rangeLooks = self.insar.topo.numberRangeLooks
    azimuthLooks = self.insar.topo.numberAzimuthLooks

    azres = self.insar.referenceFrame.platform.antennaLength/2.0
    azfact = self.insar.topo.numberAzimuthLooks *azres / self.insar.topo.azimuthSpacing

    rBW = self.insar.referenceFrame.instrument.pulseLength * self.insar.referenceFrame.instrument.chirpSlope
    rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
    rngfact = rgres/self.insar.topo.slantRangePixelSpacing

    corrLooks = self.insar.topo.numberRangeLooks * self.insar.topo.numberAzimuthLooks/(azfact*rngfact) 
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
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    if snaphu_int_format is not None:
        snp.setIntFileFormat(snaphu_int_format)
    if snaphu_cor_format is not None:
        snp.setCorFileFormat(snaphu_cor_format)
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self.insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.finalizeImage()
        connImage.renderHdr()

    return
def runUnwrapMcf(self):
    runUnwrap(self,costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
