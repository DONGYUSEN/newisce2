#
#

import isce
import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D
import logging
from isceobj.Util.decorators import use_api

import os
import numpy as np
import shelve

logger = logging.getLogger('isce.insar.runResampleSlc')


def _infer_offset_dtype(filename, width, length):
    """
    Infer raster numeric type from file size first, then XML metadata.
    """
    nelems = int(width) * int(length)
    fsize = os.path.getsize(filename)

    if fsize == nelems * np.dtype(np.float64).itemsize:
        return np.float64
    if fsize == nelems * np.dtype(np.float32).itemsize:
        return np.float32

    img = isceobj.createImage()
    img.load(filename + '.xml')
    data_type = str(getattr(img, 'dataType', '')).upper()
    if data_type in ('DOUBLE', 'FLOAT64'):
        return np.float64
    if data_type in ('FLOAT', 'FLOAT32'):
        return np.float32

    raise ValueError(
        'Cannot infer dtype for "{0}" (size={1}, width={2}, length={3}, xml={4}).'.format(
            filename, fsize, width, length, data_type
        )
    )


def _merge_range_poly_into_raster(rgname, width, length, rgpoly):
    """
    Merge range offset polynomial into per-pixel range offsets.
    This keeps interpolation and flatten phase model consistent.
    """
    if rgpoly is None:
        return None

    coeffs = getattr(rgpoly, '_coeffs', None)
    if coeffs is None:
        try:
            coeffs = rgpoly.getCoeffs()
        except Exception:
            return None

    if (coeffs is None) or (len(coeffs) == 0):
        return None

    mean_az = getattr(rgpoly, '_meanAzimuth', 0.0)
    norm_az = getattr(rgpoly, '_normAzimuth', 1.0) or 1.0
    mean_rg = getattr(rgpoly, '_meanRange', 0.0)
    norm_rg = getattr(rgpoly, '_normRange', 1.0) or 1.0

    outname = rgname + '.withpoly'
    in_dtype = _infer_offset_dtype(rgname, width, length)
    logger.info('Merging range poly into raster using dtype=%s for %s', np.dtype(in_dtype).name, rgname)
    rin = np.memmap(rgname, dtype=in_dtype, mode='r', shape=(length, width))
    rout = np.memmap(outname, dtype=in_dtype, mode='w+', shape=(length, width))

    max_rg_order = max(len(row) for row in coeffs) - 1
    x = (np.arange(width, dtype=np.float64) - mean_rg) / norm_rg
    xpow = [np.ones(width, dtype=np.float64)]
    for _ in range(max_rg_order):
        xpow.append(xpow[-1] * x)

    row_bases = []
    for row in coeffs:
        base = np.zeros(width, dtype=np.float64)
        for jj, val in enumerate(row):
            if val != 0.0:
                base += val * xpow[jj]
        row_bases.append(base)

    for ii in range(length):
        y = (float(ii) - mean_az) / norm_az
        ypow = 1.0
        poly_row = np.zeros(width, dtype=np.float64)
        for az_order, base in enumerate(row_bases):
            if az_order > 0:
                ypow *= y
            if ypow != 0.0:
                poly_row += ypow * base

        rout[ii, :] = (rin[ii, :].astype(np.float64, copy=False) + poly_row).astype(in_dtype, copy=False)

    rout.flush()
    del rout
    del rin

    outimg = isceobj.createImage()
    outimg.load(rgname + '.xml')
    outimg.filename = outname
    outimg.dataType = 'DOUBLE' if in_dtype == np.float64 else 'FLOAT'
    outimg.setAccessMode('READ')
    outimg.renderHdr()
    return outimg

def runResampleSlc(self, kind='coarse'):
    '''
    Kind can either be coarse, refined or fine.
    '''

    if kind not in ['coarse', 'refined', 'fine']:
        raise Exception('Unknown operation type {0} in runResampleSlc'.format(kind))

    if kind == 'fine':
        if not (self.doRubbersheetingRange | self.doRubbersheetingAzimuth): # Modified by V. Brancato 10.10.2019
            print('Rubber sheeting not requested, skipping resampling ....')
            return

    logger.info("Resampling secondary SLC")

    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)
    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)

    inimg = isceobj.createSlcImage()
    inimg.load(secondaryFrame.getImage().filename + '.xml')
    inimg.setAccessMode('READ')

    prf = secondaryFrame.PRF

    doppler = secondaryFrame._dopplerVsPixel
    coeffs = [2*np.pi*val/prf for val in doppler]
    
    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(coeffs)-1, azimuthOrder=0, coeffs=[coeffs])

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    rObj.radarWavelength = secondaryFrame.getInstrument().getRadarWavelength() 
    rObj.dopplerPoly = dpoly 

    # for now let's start with None polynomial. Later this should change to
    # the misregistration polynomial

    misregFile = os.path.join(self.insar.misregDirname, self.insar.misregFilename)
    if ((kind in ['refined','fine']) and os.path.exists(misregFile+'_az.xml')):
        azpoly = self._insar.loadProduct(misregFile + '_az.xml')
        rgpoly = self._insar.loadProduct(misregFile + '_rg.xml')
    else:
        print(misregFile , " does not exist.")
        azpoly = None
        rgpoly = None

    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    #Since the app is based on geometry module we expect pixel-by-pixel offset
    #field
    offsetsDir = self.insar.offsetsDirname 
    
    # Modified by V. Brancato 10.10.2019
    #rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    
    if kind in ['coarse', 'refined']:
        azname = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)
        rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
        flatten = True
    else:
        if self.doRubbersheetingAzimuth:
           print('Rubbersheeting in azimuth is turned on, taking azimuth cross-correlation offsets')
           azname = os.path.join(offsetsDir, self.insar.azimuthRubbersheetFilename)
        else:
           print('Rubbersheeting in azimuth is turned off, taking azimuth geometric offsets')
           azname = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)

        if self.doRubbersheetingRange:
           print('Rubbersheeting in range is turned on, taking the cross-correlation offsets') 
           print('Setting Flattening to False') 
           rgname = os.path.join(offsetsDir, self.insar.rangeRubbersheetFilename) 
           flatten=False
        else:
           print('Rubbersheeting in range is turned off, taking range geometric offsets')
           rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
           flatten=True
    
    rngImg = isceobj.createImage()
    rngImg.load(rgname + '.xml')
    rngImg.setAccessMode('READ')

    aziImg = isceobj.createImage()
    aziImg.load(azname + '.xml')
    aziImg.setAccessMode('READ')

    width = rngImg.getWidth()
    length = rngImg.getLength()

# Modified by V. Brancato 10.10.2019
    #flatten = True
    rObj.flatten = flatten
    rObj.outputWidth = width
    rObj.outputLines = length

    # Keep flattening phase model consistent with range misregistration model.
    # If rgpoly is applied for interpolation but not included in flattening,
    # residual fringes can remain in topophase.flat.
    if flatten and (rgpoly is not None):
        merged = _merge_range_poly_into_raster(rgname, width, length, rgpoly)
        if merged is not None:
            rngImg = merged
            rObj.rangeOffsetsPoly = None
            print('Flattening uses range offsets merged with misreg polynomial.')

    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg

    if referenceFrame is not None:
        rObj.startingRange = secondaryFrame.startingRange
        rObj.referenceStartingRange = referenceFrame.startingRange
        rObj.referenceSlantRangePixelSpacing = referenceFrame.getInstrument().getRangePixelSize()
        rObj.referenceWavelength = referenceFrame.getInstrument().getRadarWavelength()

    
    # preparing the output directory for coregistered secondary slc
    coregDir = self.insar.coregDirname

    os.makedirs(coregDir, exist_ok=True)

    # output file name of the coregistered secondary slc
    img = secondaryFrame.getImage()

    if kind  == 'coarse':
        coregFilename = os.path.join(coregDir , self._insar.coarseCoregFilename)
    elif kind == 'refined':
        coregFilename = os.path.join(coregDir, self._insar.refinedCoregFilename)
    elif kind == 'fine':
        coregFilename = os.path.join(coregDir, self._insar.fineCoregFilename)
    else:
        print('Exception: Should not have gotten to this stage')

    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = coregFilename
    imgOut.setAccessMode('write')

    rObj.resamp_slc(imageOut=imgOut)

    imgOut.renderHdr()

    return
