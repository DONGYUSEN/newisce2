
#
# Author: Heresh Fattahi, 2017
# Modified by V. Brancato (10.2019)
#         (Included flattening when rubbersheeting in range is turned on

import isceobj
import logging
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import os
import glob
import shelve
from osgeo import gdal
import numpy as np

logger = logging.getLogger('isce.insar.runInterferogram')

# Added by V. Brancato 10.09.2019
def write_xml(fileName,width,length,bands,dataType,scheme):

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()
    
    return None


def _infer_single_band_dtype(filename, width, length):
    """
    Infer numeric dtype of a single-band raster from size first, then XML metadata.
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


def _safe_float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _offset_valid_mask(arr, nodata, invalid_low):
    mask = np.isfinite(arr)
    if np.isfinite(nodata):
        mask &= (arr != nodata)
    if np.isfinite(invalid_low):
        mask &= (arr >= invalid_low)
    return mask


def _load_external_registration_meta(misreg_file):
    if not glob.glob(misreg_file + '*'):
        return None
    try:
        db = shelve.open(misreg_file, flag='r')
    except Exception:
        return None
    try:
        return db.get('external_registration', None)
    finally:
        db.close()


def _external_force_explicit_range_flatten(self):
    env_override = os.environ.get('ISCE_EXTERNAL_FORCE_RANGE_FLATTEN')
    if env_override is not None:
        sval = str(env_override).strip().lower()
        return sval in ('1', 'true', 'yes', 'on')

    misreg_file = os.path.join(self.insar.misregDirname, self.insar.misregFilename)
    ext_meta = _load_external_registration_meta(misreg_file)
    if not isinstance(ext_meta, dict):
        return False

    if 'flatten_in_interferogram' in ext_meta:
        return bool(ext_meta.get('flatten_in_interferogram'))
    return bool(ext_meta.get('enabled', False))


def _use_rectified_range_offset(self=None):
    env_override = os.environ.get('ISCE_USE_RECT_RANGE_OFFSET')
    if env_override is not None:
        sval = str(env_override).strip().lower()
        return sval in ('1', 'true', 'yes', 'on')
    if self is not None and hasattr(self, 'useRdrdemRectRangeOffset'):
        try:
            return bool(getattr(self, 'useRdrdemRectRangeOffset'))
        except Exception:
            pass
    return False


def _resolved_range_offset_file(self):
    default = os.path.join(self.insar.offsetsDirname, self.insar.rangeOffsetFilename)
    if not _use_rectified_range_offset(self):
        return default

    candidates = []
    rect_from_attr = getattr(getattr(self, '_insar', None), 'rectRangeOffsetFilename', None)
    if rect_from_attr:
        candidates.append(os.path.join(self.insar.offsetsDirname, os.path.basename(str(rect_from_attr))))
    candidates.append(os.path.join(self.insar.offsetsDirname, 'range_rect.off'))
    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(cand + '.xml'):
            logger.info('Using rectified range offset for flat-earth removal: %s', cand)
            return cand
    return default


def compute_FlatEarth(self,ifgFilename,width,length,radarWavelength):
    from imageMath import IML
    import logging
    
    # If rubbersheeting has been performed add back the range sheet offsets
    
    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
    #radarWavelength = info.getInstrument().getRadarWavelength() 
    rangePixelSize = info.getInstrument().getRangePixelSize()
    fact = 4 * np.pi* rangePixelSize / radarWavelength

    cJ = np.complex64(-1j)

    # Open the range sheet offset
    rngOff = _resolved_range_offset_file(self)
    
    print(rngOff)
    if os.path.exists(rngOff):
       rng_dtype = _infer_single_band_dtype(rngOff, width, length)
       logger.info('compute_FlatEarth uses range offsets dtype=%s for %s', np.dtype(rng_dtype).name, rngOff)
       rng2 = np.memmap(rngOff, dtype=rng_dtype, mode='r', shape=(length,width))
    else:
       print('No range offsets provided')
       rng2 = np.zeros((length,width), dtype=np.float64)

    nodata = _safe_float_env('ISCE_GEO2RDR_OFFSET_NODATA', -999999.0)
    invalid_low = _safe_float_env('ISCE_GEO2RDR_OFFSET_INVALID_LOW', -1.0e5)

    # Open the interferogram
    #ifgFilename= os.path.join(self.insar.ifgDirname, self.insar.ifgFilename)
    intf = np.memmap(ifgFilename,dtype=np.complex64,mode='r+',shape=(length,width))
    invalid_total = 0
    for ll in range(length):
        line_off = np.asarray(rng2[ll, :], dtype=np.float64)
        valid = _offset_valid_mask(line_off, nodata, invalid_low)
        invalid_total += int(line_off.size - np.count_nonzero(valid))
        line_off[~valid] = 0.0

        intf[ll,:] *= np.exp(cJ*fact*line_off)

    logger.info(
        'compute_FlatEarth sanitized invalid range offsets: invalid=%d/%d (%.2f%%), '
        'nodata=%.1f, invalid_low=%.1f',
        int(invalid_total),
        int(length * width),
        100.0 * float(invalid_total) / float(max(length * width, 1)),
        float(nodata),
        float(invalid_low),
    )
    
    del rng2
    del intf
       
    return 
    
    

def multilook(infile, outname=None, alks=5, rlks=15):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    print('Multilooking {0} ...'.format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + '.xml')

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname

def computeCoherence(slc1name, slc2name, corname, virtual=True):
    from mroipac.correlation.correlation import Correlation
                          
    slc1 = isceobj.createImage()
    slc1.load( slc1name + '.xml')
    slc1.createImage()


    slc2 = isceobj.createImage()
    slc2.load( slc2name + '.xml')
    slc2.createImage()

    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(corname)
    cohImage.setWidth(slc1.getWidth())
    cohImage.setAccessMode('write')
    cohImage.createImage()

    cor = Correlation()
    cor.configure()
    cor.wireInputPort(name='slc1', object=slc1)
    cor.wireInputPort(name='slc2', object=slc2)
    cor.wireOutputPort(name='correlation', object=cohImage)
    cor.coregisteredSlcFlag = True
    cor.calculateCorrelation()

    cohImage.finalizeImage()
    slc1.finalizeImage()
    slc2.finalizeImage()
    return

# Modified by V. Brancato on 10.09.2019 (added self)
# Modified by V. Brancato on 11.13.2019 (added radar wavelength for low and high band flattening
def generateIgram(
    self,
    imageSlc1,
    imageSlc2,
    resampName,
    azLooks,
    rgLooks,
    radarWavelength,
    force_rangeoff_flatten=False,
):
    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    
    
    apply_explicit_flatten = bool(self.doRubbersheetingRange) or bool(force_rangeoff_flatten)
    if not apply_explicit_flatten:
     intWidth = int(slcWidth/rgLooks)    # Modified by V. Brancato intWidth = int(slcWidth / rgLooks)
    else:
     intWidth = int(slcWidth)
    
    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    if '.flat' in resampName:
        resampAmp = resampName.replace('.flat', '.amp')
    elif '.int' in resampName:
        resampAmp = resampName.replace('.int', '.amp')
    else:
        resampAmp += '.amp'

    if not apply_explicit_flatten:
        resampInt = resampName
    else:
        resampInt = resampName + ".full"

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode('write')
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode('write')
    objAmp.createImage()
    
    if not apply_explicit_flatten:
       print('Rubbersheeting in range is off, interferogram is already flattened')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = azLooks
       objCrossmul.LooksAcross = rgLooks

       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
    else:
     # Modified by V. Brancato 10.09.2019 (added option to add Range Rubber sheet Flat-earth back)
       if self.doRubbersheetingRange:
           print('Rubbersheeting in range is on, removing flat-Earth phase')
       else:
           print('External registration mode: removing flat-Earth phase with explicit range.off')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = 1
       objCrossmul.LooksAcross = 1
       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
       
       # Remove Flat-Earth component
       compute_FlatEarth(self,resampInt,intWidth,lines,radarWavelength)
       
       # Perform Multilook
       multilook(resampInt, outname=resampName, alks=azLooks, rlks=rgLooks)  #takeLooks(objAmp,azLooks,rgLooks)
       multilook(resampAmp, outname=resampAmp.replace(".full",""), alks=azLooks, rlks=rgLooks)  #takeLooks(objInt,azLooks,rgLooks)
       
       #os.system('rm ' + resampInt+'.full* ' + resampAmp + '.full* ')
       # End of modification 
    for obj in [objInt, objAmp, objSlc1, objSlc2]:
        obj.finalizeImage()

    return imageInt, imageAmp


def subBandIgram(self, referenceSlc, secondarySlc, subBandDir,radarWavelength):

    img1 = isceobj.createImage()
    img1.load(referenceSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(secondarySlc + '.xml')

    azLooks = self.numberAzimuthLooks
    rgLooks = self.numberRangeLooks

    ifgDir = os.path.join(self.insar.ifgDirname, subBandDir)

    os.makedirs(ifgDir, exist_ok=True)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)

    generateIgram(self,img1, img2, interferogramName, azLooks, rgLooks,radarWavelength)
    
    return interferogramName

def runSubBandInterferograms(self):
    
    logger.info("Generating sub-band interferograms")

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

    azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting,
                                        self.numberAzimuthLooks, self.numberRangeLooks)

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)

    ###########
    referenceSlc =  referenceFrame.getImage().filename
    lowBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname)
    highBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname)
    referenceLowBandSlc = os.path.join(lowBandDir, os.path.basename(referenceSlc))
    referenceHighBandSlc = os.path.join(highBandDir, os.path.basename(referenceSlc))
    ##########
    secondarySlc = secondaryFrame.getImage().filename
    coregDir = os.path.join(self.insar.coregDirname, self.insar.lowBandSlcDirname) 
    secondaryLowBandSlc = os.path.join(coregDir , os.path.basename(secondarySlc))
    coregDir = os.path.join(self.insar.coregDirname, self.insar.highBandSlcDirname)
    secondaryHighBandSlc = os.path.join(coregDir , os.path.basename(secondarySlc))
    ##########

    interferogramName = subBandIgram(self, referenceLowBandSlc, secondaryLowBandSlc, self.insar.lowBandSlcDirname,self.insar.lowBandRadarWavelength)

    interferogramName = subBandIgram(self, referenceHighBandSlc, secondaryHighBandSlc, self.insar.highBandSlcDirname,self.insar.highBandRadarWavelength)
    
def runFullBandInterferogram(self):
    logger.info("Generating interferogram")

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    referenceSlc =  referenceFrame.getImage().filename
   
    if (self.doRubbersheetingRange | self.doRubbersheetingAzimuth):    
        secondarySlc = os.path.join(self._insar.coregDirname, self._insar.fineCoregFilename)
    else:
        secondarySlc = os.path.join(self._insar.coregDirname, self._insar.refinedCoregFilename)

    img1 = isceobj.createImage()
    img1.load(referenceSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(secondarySlc + '.xml')

    azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting, 
                            self.numberAzimuthLooks, self.numberRangeLooks) 

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)
    ifgDir = self.insar.ifgDirname

    if os.path.isdir(ifgDir):
        logger.info('Interferogram directory {0} already exists.'.format(ifgDir))
    else:
        os.makedirs(ifgDir)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)
    
    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
    radarWavelength = info.getInstrument().getRadarWavelength()
    
    force_rangeoff_flatten = _external_force_explicit_range_flatten(self)
    if force_rangeoff_flatten:
        logger.info(
            'External registration active: forcing explicit flat-earth removal with %s.',
            _resolved_range_offset_file(self),
        )
    generateIgram(
        self,
        img1,
        img2,
        interferogramName,
        azLooks,
        rgLooks,
        radarWavelength,
        force_rangeoff_flatten=force_rangeoff_flatten,
    )


    ###Compute coherence
    cohname = os.path.join(self.insar.ifgDirname, self.insar.correlationFilename)
    computeCoherence(referenceSlc, secondarySlc, cohname+'.full')
    multilook(cohname+'.full', outname=cohname, alks=azLooks, rlks=rgLooks)


    ##Multilook relevant geometry products
    for fname in [self.insar.latFilename, self.insar.lonFilename, self.insar.losFilename]:
        inname =  os.path.join(self.insar.geometryDirname, fname)
        multilook(inname + '.full', outname= inname, alks=azLooks, rlks=rgLooks)

def runInterferogram(self, igramSpectrum = "full"):

    logger.info("igramSpectrum = {0}".format(igramSpectrum))

    if igramSpectrum == "full":
        runFullBandInterferogram(self)


    elif igramSpectrum == "sub":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        runSubBandInterferograms(self) 
