#

#
import isce
import isceobj
from mroipac.ampcor.DenseAmpcor import DenseAmpcor
from isceobj.Util.decorators import use_api
import os
import logging

logger = logging.getLogger('isce.insar.runDenseOffsets')


def _safe_float_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _ensure_binary_slc(infile):
    '''
    Dense offset modules expect a binary ENVI payload in addition to VRT/XML metadata.
    '''
    if os.path.isfile(infile):
        return
    cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
    status = os.system(cmd)
    if status:
        raise RuntimeError('Failed to materialize SLC binary with command: {0}'.format(cmd))


@use_api
def estimateOffsetField(reference, secondary, denseOffsetFileName,
                        ww=64, wh=64,
                        sw=20, shh=20,
                        kw=32, kh=32,
                        covth=1.0e6):
    '''
    Estimate offset field between burst and simamp using CPU DenseAmpcor.
    '''

    # Loading the secondary image object
    sim = isceobj.createSlcImage()
    sim.load(secondary + '.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    # Loading the reference image object
    sar = isceobj.createSlcImage()
    sar.load(reference + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    objOffset = DenseAmpcor(name='dense')
    objOffset.configure()

    objOffset.setWindowSizeWidth(ww)
    objOffset.setWindowSizeHeight(wh)
    objOffset.setSearchWindowSizeWidth(sw)
    objOffset.setSearchWindowSizeHeight(shh)
    objOffset.skipSampleAcross = kw
    objOffset.skipSampleDown = kh
    objOffset.margin = max(50, int(max(ww, wh) + 2 * max(sw, shh)))
    objOffset.oversamplingFactor = 32
    objOffset.thresholdCov = float(covth)

    objOffset.setAcrossGrossOffset(0)
    objOffset.setDownGrossOffset(0)

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    if sar.dataType.startswith('C'):
        objOffset.setImageDataType1('mag')
    else:
        objOffset.setImageDataType1('real')

    if sim.dataType.startswith('C'):
        objOffset.setImageDataType2('mag')
    else:
        objOffset.setImageDataType2('real')

    objOffset.offsetImageName = denseOffsetFileName + '.bil'
    objOffset.snrImageName = denseOffsetFileName + '_snr.bil'
    objOffset.covImageName = denseOffsetFileName + '_cov.bil'

    logger.info(
        'DenseAmpcor config: window=(%d,%d), search=(%d,%d), skip=(%d,%d), margin=%d, cov_threshold=%.3f',
        ww, wh, sw, shh, kw, kh, objOffset.margin, float(covth)
    )
    objOffset.denseampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()
    return (objOffset.locationDown[0][0], objOffset.locationAcross[0][0])


@use_api
def estimateOffsetFieldGPU(reference, secondary, denseOffsetFileName,
                           ww=64, wh=64,
                           sw=20, shh=20,
                           kw=32, kh=32):
    '''
    Estimate offset field between burst and simamp using GPU PyCuAmpcor.
    '''
    from contrib.PyCuAmpcor import PyCuAmpcor

    _ensure_binary_slc(reference)
    _ensure_binary_slc(secondary)

    sar = isceobj.createSlcImage()
    sar.load(reference + '.xml')
    sar.setAccessMode('READ')
    sar.renderHdr()

    sim = isceobj.createSlcImage()
    sim.load(secondary + '.xml')
    sim.setAccessMode('READ')
    sim.renderHdr()

    width = sar.getWidth()
    length = sar.getLength()
    margin = max(50, int(max(ww, wh) + 2 * max(sw, shh)))

    number_window_down = (length - 2 * margin - 2 * shh - wh) // kh
    number_window_across = (width - 2 * margin - 2 * sw - ww) // kw
    if (number_window_down <= 0) or (number_window_across <= 0):
        raise ValueError(
            'Invalid GPU dense-offset layout: windows down/across = ({0}, {1}). '
            'Check dense window/search/skip parameters.'.format(number_window_down, number_window_across)
        )

    objOffset = PyCuAmpcor.PyCuAmpcor()
    objOffset.algorithm = 0
    objOffset.derampMethod = 1
    objOffset.referenceImageName = reference + '.vrt'
    objOffset.referenceImageHeight = length
    objOffset.referenceImageWidth = width
    objOffset.secondaryImageName = secondary + '.vrt'
    objOffset.secondaryImageHeight = length
    objOffset.secondaryImageWidth = width

    objOffset.windowSizeWidth = ww
    objOffset.windowSizeHeight = wh
    objOffset.halfSearchRangeAcross = sw
    objOffset.halfSearchRangeDown = shh
    objOffset.skipSampleAcross = kw
    objOffset.skipSampleDown = kh
    objOffset.corrSurfaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = 32

    gross_offset_across = 0
    gross_offset_down = 0
    objOffset.referenceStartPixelDownStatic = margin + shh
    objOffset.referenceStartPixelAcrossStatic = margin + sw
    objOffset.numberWindowDown = number_window_down
    objOffset.numberWindowAcross = number_window_across

    objOffset.deviceID = 0
    objOffset.nStreams = 2
    objOffset.numberWindowDownInChunk = 1
    objOffset.numberWindowAcrossInChunk = min(64, max(1, number_window_across))
    objOffset.mmapSize = 16

    objOffset.offsetImageName = denseOffsetFileName + '.bil'
    objOffset.grossOffsetImageName = denseOffsetFileName + '.gross'
    objOffset.snrImageName = denseOffsetFileName + '_snr.bil'
    objOffset.covImageName = denseOffsetFileName + '_cov.bil'
    objOffset.mergeGrossOffset = 1

    logger.info(
        'PyCuAmpcor config: window=(%d,%d), search=(%d,%d), skip=(%d,%d), margin=%d, windows=(%d,%d)',
        ww, wh, sw, shh, kw, kh, margin, number_window_down, number_window_across
    )
    objOffset.setupParams()
    objOffset.setConstantGrossOffset(gross_offset_across, gross_offset_down)
    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()

    outImg = isceobj.createImage()
    outImg.setDataType('FLOAT')
    outImg.setFilename(objOffset.offsetImageName)
    outImg.setBands(2)
    outImg.scheme = 'BIP'
    outImg.setWidth(objOffset.numberWindowAcross)
    outImg.setLength(objOffset.numberWindowDown)
    outImg.setAccessMode('read')
    outImg.renderHdr()

    snrImg = isceobj.createImage()
    snrImg.setFilename(objOffset.snrImageName)
    snrImg.setDataType('FLOAT')
    snrImg.setBands(1)
    snrImg.setWidth(objOffset.numberWindowAcross)
    snrImg.setLength(objOffset.numberWindowDown)
    snrImg.setAccessMode('read')
    snrImg.renderHdr()

    covImg = isceobj.createImage()
    covImg.setFilename(objOffset.covImageName)
    covImg.setDataType('FLOAT')
    covImg.setBands(3)
    covImg.scheme = 'BIP'
    covImg.setWidth(objOffset.numberWindowAcross)
    covImg.setLength(objOffset.numberWindowDown)
    covImg.setAccessMode('read')
    covImg.renderHdr()

    offset_top = objOffset.referenceStartPixelDownStatic + (objOffset.windowSizeHeight - 1) // 2
    offset_left = objOffset.referenceStartPixelAcrossStatic + (objOffset.windowSizeWidth - 1) // 2
    return (offset_top, offset_left)


def runDenseOffsets(self):

    if not self.doDenseOffsets:
        if self.doRubbersheetingAzimuth or self.doRubbersheetingRange:
            print('Rubbersheeting requested but doDenseOffsets is False; dense offsets will not run.')
        return
    print('Dense offsets explicitly requested')

    referenceFrame = self.insar.loadProduct(self._insar.referenceSlcCropProduct)
    referenceSlc = referenceFrame.getImage().filename

    secondarySlc = os.path.join(self.insar.coregDirname, self._insar.refinedCoregFilename)

    dirname = self.insar.denseOffsetsDirname
    os.makedirs(dirname, exist_ok=True)

    denseOffsetFilename = os.path.join(dirname, self.insar.denseOffsetFilename)
    denseCovThreshold = _safe_float_env('ISCE_DENSE_OFFSET_COV_THRESHOLD', 1.0e6)

    field = None
    use_gpu = bool(getattr(self, 'useGPU', False))
    gpu_available = False
    if use_gpu:
        try:
            gpu_available = bool(self._insar.hasGPU())
        except Exception as err:
            logger.warning('Unable to evaluate Stripmap GPU availability: %s', str(err))

    if use_gpu and gpu_available:
        try:
            field = estimateOffsetFieldGPU(
                referenceSlc,
                secondarySlc,
                denseOffsetFilename,
                ww=self.denseWindowWidth,
                wh=self.denseWindowHeight,
                sw=self.denseSearchWidth,
                shh=self.denseSearchHeight,
                kw=self.denseSkipWidth,
                kh=self.denseSkipHeight
            )
            logger.info('Dense offsets computed with GPU (PyCuAmpcor).')
        except Exception as err:
            logger.warning(
                'GPU dense offsets failed (%s). Falling back to CPU DenseAmpcor.',
                str(err),
                exc_info=True
            )

    if field is None:
        if use_gpu and (not gpu_available):
            logger.info('GPU requested but unavailable for stripmap dense offsets. Using CPU DenseAmpcor.')
        field = estimateOffsetField(
            referenceSlc,
            secondarySlc,
            denseOffsetFilename,
            ww=self.denseWindowWidth,
            wh=self.denseWindowHeight,
            sw=self.denseSearchWidth,
            shh=self.denseSearchHeight,
            kw=self.denseSkipWidth,
            kh=self.denseSkipHeight,
            covth=denseCovThreshold
        )

    self._insar.offset_top = field[0]
    self._insar.offset_left = field[1]

    return None
