#
# Author: Piyush Agram
# Copyright 2016
#

import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
from isceobj.Planet.Planet import Planet
import os
import numpy as np
import logging
from isceobj.TopsProc.runIon import maskUnwrap

logger = logging.getLogger('isce.topsinsar.runUnwrapSnaphu')


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


def runUnwrap(self,costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
    

    wrapName = os.path.join( self._insar.mergedDirname, self._insar.filtFilename)
    unwrapName = os.path.join( self._insar.mergedDirname, self._insar.unwrappedIntFilename)

    img = isceobj.createImage()
    img.load(wrapName + '.xml')


    swathList = self._insar.getValidSwathList(self.swaths)

    for swath in swathList[0:1]:
        ifg = self._insar.loadProduct( os.path.join(self._insar.fineIfgDirname, 'IW{0}.xml'.format(swath)))


        wavelength = ifg.bursts[0].radarWavelength
        width      = img.getWidth()


        ####tmid 
        tstart = ifg.bursts[0].sensingStart
        tend   = ifg.bursts[-1].sensingStop
        tmid = tstart + 0.5*(tend - tstart) 

        #some times tmid may exceed the time span, so use mid burst instead
        #14-APR-2018, Cunren Liang
        #orbit = ifg.bursts[0].orbit
        burst_index = int(np.around(len(ifg.bursts)/2))
        orbit = ifg.bursts[burst_index].orbit
        peg = orbit.interpolateOrbit(tmid, method='hermite')


        refElp = Planet(pname='Earth').ellipsoid
        llh = refElp.xyz_to_llh(peg.getPosition())
        hdg = orbit.getENUHeading(tmid)
        refElp.setSCH(llh[0], llh[1], hdg)

        earthRadius = refElp.pegRadCur

        altitude   = llh[2]

    corrfile  = os.path.join(self._insar.mergedDirname, self._insar.coherenceFilename)
    snaphu_input = wrapName
    snaphu_corr = corrfile
    snaphu_int_format = None
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

    rangeLooks = self.numberRangeLooks
    azimuthLooks = self.numberAzimuthLooks

    azfact = 0.8
    rngfact = 0.8

    corrLooks = rangeLooks * azimuthLooks/(azfact*rngfact) 
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
    snp.setCorFileFormat('FLOAT_DATA')

    tile_nrow = max(1, int(getattr(self, 'snaphuTileNRow', 2)))
    tile_ncol = max(1, int(getattr(self, 'snaphuTileNCol', 2)))
    row_overlap = max(0, int(getattr(self, 'snaphuRowOverlap', 400)))
    col_overlap = max(0, int(getattr(self, 'snaphuColOverlap', 400)))
    min_overlap = 400

    if tile_nrow > 1 and row_overlap < min_overlap:
        logger.warning(
            'snaphu row overlap=%d is below required minimum %d; using %d',
            row_overlap,
            min_overlap,
            min_overlap,
        )
        row_overlap = min_overlap
    if tile_ncol > 1 and col_overlap < min_overlap:
        logger.warning(
            'snaphu col overlap=%d is below required minimum %d; using %d',
            col_overlap,
            min_overlap,
            min_overlap,
        )
        col_overlap = min_overlap

    snp.setTileNRow(tile_nrow)
    snp.setTileNCol(tile_ncol)
    snp.setRowOverlap(row_overlap)
    snp.setColOverlap(col_overlap)
    # Keep global minimum overlap floor at 400 in tile mode.
    snp.minTileOverlap = max(min_overlap, int(getattr(snp, 'minTileOverlap', min_overlap)))

    logger.info(
        'snaphu tiling configured for topsApp: tiles=%dx%d, overlap(row/col)=%d/%d',
        tile_nrow,
        tile_ncol,
        row_overlap,
        col_overlap,
    )
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderVRT()
    outImage.createImage()
    outImage.finalizeImage()
    outImage.renderHdr()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self._insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderVRT()
        connImage.createImage()
        connImage.finalizeImage()
        connImage.renderHdr()

        #mask the areas where values are zero.
        #15-APR-2018, Cunren Liang
        maskUnwrap(unwrapName, wrapName)

    return


def runUnwrapMcf(self):
    runUnwrap(self,costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
