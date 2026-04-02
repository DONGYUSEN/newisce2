#

#
import isce
import isceobj
from mroipac.ampcor.DenseAmpcor import DenseAmpcor
from isceobj.Util.decorators import use_api
from isceobj.StripmapProc.externalRegistration import (
    _read_slc_as_memmap,
    _AmplitudeAccessor,
    _phase_correlation_displacement,
)
import os
import logging
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('isce.insar.runDenseOffsets')


def _safe_float_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _safe_int_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return int(default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _safe_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 'yes', 'on', 'y', 't'):
        return True
    if sval in ('0', 'false', 'no', 'off', 'n', 'f', 'none', 'null', ''):
        return False
    return bool(default)


def _safe_bool_env(name, default):
    return _safe_bool(os.environ.get(name), default=default)


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


def _resolve_dense_threads():
    cpu_total = max(1, int(os.cpu_count() or 1))
    frac = _safe_float_env('ISCE_DENSE_THREAD_FRACTION', 0.8)
    frac = max(0.05, min(1.0, float(frac)))
    min_threads = max(1, _safe_int_env('ISCE_DENSE_MIN_THREADS', 1))
    threads = max(min_threads, int(math.floor(cpu_total * frac)))
    threads = min(threads, cpu_total)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    return threads, cpu_total, frac


def _compute_dense_layout(width, length, ww, wh, sw, shh, kw, kh, margin):
    grid = _dense_grid_definition(width, length, ww, wh, sw, shh, kw, kh, margin)
    return {
        'num_across': int(len(grid['grid_across'])),
        'num_down': int(len(grid['grid_down'])),
        'num_windows': int(len(grid['grid_across']) * len(grid['grid_down'])),
    }


def _dense_grid_definition(width, length, ww, wh, sw, shh, kw, kh, margin):
    if kw <= 0 or kh <= 0:
        raise ValueError('Dense skip must be positive: skip=(%d,%d)' % (kw, kh))

    # Keep formula consistent with DenseAmpcor.denseampcor().
    coarseAcross = 0
    coarseDown = 0
    xMargin = 2 * sw + ww
    yMargin = 2 * shh + wh
    pixLocOffAc = ww // 2 + sw - 1
    pixLocOffDn = wh // 2 + shh - 1

    offAc = max(margin, -coarseAcross) + xMargin
    if offAc % kw != 0:
        leftlim = offAc
        offAc = kw * (1 + int(offAc / kw)) - pixLocOffAc
        while offAc < leftlim:
            offAc += kw

    offDn = max(margin, -coarseDown) + yMargin
    if offDn % kh != 0:
        toplim = offDn
        offDn = kh * (1 + int(offDn / kh)) - pixLocOffDn
        while offDn < toplim:
            offDn += kh

    offAcmax = int(coarseAcross + ((1.0 / 1.0) - 1) * width)
    lastAc = int(min(width, width - offAcmax) - xMargin - 1 - margin)
    offDnmax = int(coarseDown + ((1.0 / 1.0) - 1) * length)
    lastDn = int(min(length, length - offDnmax) - yMargin - 1 - margin)

    gridLocAcross = range(offAc + pixLocOffAc, lastAc - pixLocOffAc, kw)
    gridLocDown = range(offDn + pixLocOffDn, lastDn - pixLocOffDn, kh)
    return {
        'grid_across': list(gridLocAcross),
        'grid_down': list(gridLocDown),
    }


def _render_dense_output_headers(base, num_across, num_down):
    out_img = isceobj.createImage()
    out_img.setDataType('FLOAT')
    out_img.setFilename(base + '.bil')
    out_img.setBands(2)
    out_img.scheme = 'BIL'
    out_img.setWidth(int(num_across))
    out_img.setLength(int(num_down))
    out_img.setAccessMode('read')
    out_img.renderHdr()

    snr_img = isceobj.createImage()
    snr_img.setFilename(base + '_snr.bil')
    snr_img.setDataType('FLOAT')
    snr_img.setBands(1)
    snr_img.setWidth(int(num_across))
    snr_img.setLength(int(num_down))
    snr_img.setAccessMode('read')
    snr_img.renderHdr()

    cov_img = isceobj.createImage()
    cov_img.setDataType('FLOAT')
    cov_img.setFilename(base + '_cov.bil')
    cov_img.setBands(3)
    cov_img.scheme = 'BIL'
    cov_img.setWidth(int(num_across))
    cov_img.setLength(int(num_down))
    cov_img.setAccessMode('read')
    cov_img.renderHdr()


@use_api
def estimateOffsetFieldExternal(reference, secondary, denseOffsetFileName,
                                ww=64, wh=64,
                                sw=20, shh=20,
                                kw=32, kh=32,
                                threads=None):
    '''
    Estimate dense offsets with integrated external phase-correlation logic on CPU.
    Falls back to DenseAmpcor at caller level when this path fails.
    '''
    precompute_amp = _safe_bool_env('ISCE_DENSE_EXTERNAL_PRECOMPUTE_AMP', False)
    quality_threshold = _safe_float_env('ISCE_DENSE_EXTERNAL_QUALITY_THRESHOLD', 0.20)
    min_valid = max(1, _safe_int_env('ISCE_DENSE_EXTERNAL_MIN_VALID', 100))

    workers_cfg = _safe_int_env('ISCE_DENSE_EXTERNAL_WORKERS', 0)
    if workers_cfg > 0:
        workers = max(1, workers_cfg)
    elif threads is not None:
        workers = max(1, min(int(threads), 8))
    else:
        workers = max(1, min(int(os.cpu_count() or 1), 8))

    margin = max(50, int(max(ww, wh) + 2 * max(sw, shh)))

    master = _read_slc_as_memmap(reference)
    slave = _read_slc_as_memmap(secondary)
    if master.shape != slave.shape:
        h = min(master.shape[0], slave.shape[0])
        w = min(master.shape[1], slave.shape[1])
        master = master[:h, :w]
        slave = slave[:h, :w]
        logger.warning(
            'External dense offsets cropped mismatched SLC sizes to %dx%d',
            h,
            w,
        )

    master_amp = _AmplitudeAccessor(
        master,
        precompute=precompute_amp,
        logger=logger,
        label='dense_reference',
    )
    slave_amp = _AmplitudeAccessor(
        slave,
        precompute=precompute_amp,
        logger=logger,
        label='dense_secondary',
    )
    h, w = master_amp.shape

    grid = _dense_grid_definition(w, h, ww, wh, sw, shh, kw, kh, margin)
    grid_across = grid['grid_across']
    grid_down = grid['grid_down']
    num_across = int(len(grid_across))
    num_down = int(len(grid_down))
    if num_across <= 0 or num_down <= 0:
        raise RuntimeError(
            'External dense offsets invalid grid: num_down={0}, num_across={1}'.format(
                num_down, num_across
            )
        )

    logger.info(
        'External dense offsets config: window=(%d,%d), search=(%d,%d), skip=(%d,%d), '
        'margin=%d, workers=%d, quality_threshold=%.3f, grid=(%d,%d)',
        int(ww), int(wh), int(sw), int(shh), int(kw), int(kh),
        int(margin), int(workers), float(quality_threshold), num_down, num_across
    )

    az_off = np.full((num_down, num_across), -10000.0, dtype=np.float32)
    rg_off = np.full((num_down, num_across), -10000.0, dtype=np.float32)
    snr = np.zeros((num_down, num_across), dtype=np.float32)
    cov1 = np.full((num_down, num_across), 999.0, dtype=np.float32)
    cov2 = np.full((num_down, num_across), 999.0, dtype=np.float32)
    cov3 = np.full((num_down, num_across), 999.0, dtype=np.float32)

    def _evaluate_point(row, col):
        pm = master_amp.extract_patch(row, col, ww)
        ps = slave_amp.extract_patch(row, col, ww)
        if (pm is None) or (ps is None):
            return None
        result = _phase_correlation_displacement(pm, ps)
        if result is None:
            return None
        daz, drg, q = result
        if q < quality_threshold:
            return None
        if (abs(float(daz)) > float(shh)) or (abs(float(drg)) > float(sw)):
            return None
        return float(daz), float(drg), float(q)

    valid_count = 0
    if workers <= 1:
        for i, row in enumerate(grid_down):
            for j, col in enumerate(grid_across):
                one = _evaluate_point(row, col)
                if one is None:
                    continue
                valid_count += 1
                daz, drg, q = one
                az_off[i, j] = daz
                rg_off[i, j] = drg
                snr[i, j] = q
                cval = max(0.0, 1.0 - q)
                cov1[i, j] = cval
                cov2[i, j] = cval
                cov3[i, j] = cval
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, row in enumerate(grid_down):
                results = executor.map(
                    _evaluate_point,
                    [row] * num_across,
                    grid_across,
                )
                for j, one in enumerate(results):
                    if one is None:
                        continue
                    valid_count += 1
                    daz, drg, q = one
                    az_off[i, j] = daz
                    rg_off[i, j] = drg
                    snr[i, j] = q
                    cval = max(0.0, 1.0 - q)
                    cov1[i, j] = cval
                    cov2[i, j] = cval
                    cov3[i, j] = cval

    if valid_count < min_valid:
        raise RuntimeError(
            'External dense offsets failed: valid points={0}, required={1}'.format(
                int(valid_count), int(min_valid)
            )
        )

    logger.info(
        'External dense offsets finished: valid_points=%d/%d (%.2f%%)',
        int(valid_count),
        int(num_across * num_down),
        100.0 * float(valid_count) / float(max(1, num_across * num_down)),
    )

    outdata = np.empty((2 * num_down, num_across), dtype=np.float32)
    outdata[::2, :] = az_off
    outdata[1::2, :] = rg_off
    outdata.tofile(denseOffsetFileName + '.bil')
    del outdata

    snr.tofile(denseOffsetFileName + '_snr.bil')

    covdata = np.empty((3 * num_down, num_across), dtype=np.float32)
    covdata[::3, :] = cov1
    covdata[1::3, :] = cov2
    covdata[2::3, :] = cov3
    covdata.tofile(denseOffsetFileName + '_cov.bil')
    del covdata

    _render_dense_output_headers(denseOffsetFileName, num_across, num_down)
    return (int(grid_down[0]), int(grid_across[0]))


def _estimate_dense_memory_mb(num_windows, threads, ww, wh, sw, shh):
    # Shared arrays in DenseAmpcor: 2x int + 6x float ~= 32 bytes/window.
    shared_bytes = float(num_windows) * 32.0
    # Temporary writer arrays: offset(2 bands) + covariance(3 bands) ~= 20 bytes/window.
    write_bytes = float(num_windows) * 20.0
    # Per-worker correlation scratch (rough proxy; padded patch size for two images).
    patch_w = int(ww + 2 * sw)
    patch_h = int(wh + 2 * shh)
    worker_bytes = float(max(1, threads)) * float(max(1, patch_w * patch_h)) * 16.0
    # Safety factor for Python/object overhead and library internals.
    safety = max(1.0, _safe_float_env('ISCE_DENSE_MEMORY_SAFETY_FACTOR', 3.0))
    total_bytes = (shared_bytes + write_bytes + worker_bytes) * safety
    return total_bytes / (1024.0 * 1024.0)


def _apply_dense_safeguards(num_windows, est_mem_mb, threads):
    min_threads = max(1, _safe_int_env('ISCE_DENSE_MIN_THREADS', 1))
    soft_max_windows = max(1, _safe_int_env('ISCE_DENSE_SOFT_MAX_WINDOWS', 400000))
    hard_max_windows = max(soft_max_windows, _safe_int_env('ISCE_DENSE_HARD_MAX_WINDOWS', 800000))
    soft_max_mem_mb = max(1.0, _safe_float_env('ISCE_DENSE_SOFT_MAX_EST_MEMORY_MB', 6144.0))
    hard_max_mem_mb = max(soft_max_mem_mb, _safe_float_env('ISCE_DENSE_HARD_MAX_EST_MEMORY_MB', 12288.0))
    auto_reduce = _safe_bool_env('ISCE_DENSE_AUTO_REDUCE_THREADS', True)
    reject_on_hard = _safe_bool_env('ISCE_DENSE_REJECT_ON_HARD_LIMIT', True)

    triggered_soft = (num_windows > soft_max_windows) or (est_mem_mb > soft_max_mem_mb)
    if triggered_soft and auto_reduce and (threads > min_threads):
        ratio = 1.0
        if num_windows > soft_max_windows:
            ratio = min(ratio, float(soft_max_windows) / float(num_windows))
        if est_mem_mb > soft_max_mem_mb and est_mem_mb > 0:
            ratio = min(ratio, float(soft_max_mem_mb) / float(est_mem_mb))
        new_threads = max(min_threads, int(math.floor(float(threads) * ratio)))
        if new_threads >= threads:
            new_threads = max(min_threads, threads - 1)
        if new_threads < threads:
            threads = new_threads
            reduced = True
        else:
            reduced = False
    else:
        reduced = False

    return threads, reduced, {
        'soft_max_windows': soft_max_windows,
        'hard_max_windows': hard_max_windows,
        'soft_max_mem_mb': soft_max_mem_mb,
        'hard_max_mem_mb': hard_max_mem_mb,
        'triggered_soft': triggered_soft,
        'reject_on_hard': reject_on_hard,
    }


def _enforce_dense_hard_limits(num_windows, est_mem_mb, guard):
    hard_max_windows = int(guard['hard_max_windows'])
    hard_max_mem_mb = float(guard['hard_max_mem_mb'])
    reject_on_hard = bool(guard.get('reject_on_hard', True))
    triggered_hard = (num_windows > hard_max_windows) or (est_mem_mb > hard_max_mem_mb)
    if triggered_hard and reject_on_hard:
        raise RuntimeError(
            'Dense offsets rejected by hard limits: windows={0} (hard_max={1}), '
            'est_memory_mb={2:.1f} (hard_max={3:.1f}).'.format(
                int(num_windows), int(hard_max_windows), float(est_mem_mb), float(hard_max_mem_mb)
            )
        )


@use_api
def estimateOffsetField(reference, secondary, denseOffsetFileName,
                        ww=64, wh=64,
                        sw=20, shh=20,
                        kw=32, kh=32,
                        covth=1.0e6,
                        threads=None):
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
    if threads is not None:
        objOffset.numberThreads = max(1, int(threads))
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
        'DenseAmpcor config: window=(%d,%d), search=(%d,%d), skip=(%d,%d), '
        'margin=%d, cov_threshold=%.3f, threads=%d',
        ww, wh, sw, shh, kw, kh, objOffset.margin, float(covth), int(objOffset.numberThreads)
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
    dense_threads, cpu_total, thread_frac = _resolve_dense_threads()
    dense_margin = max(
        50,
        int(max(self.denseWindowWidth, self.denseWindowHeight) + 2 * max(self.denseSearchWidth, self.denseSearchHeight))
    )

    ref_img = isceobj.createSlcImage()
    ref_img.load(referenceSlc + '.xml')
    width = int(ref_img.getWidth())
    length = int(ref_img.getLength())
    layout = _compute_dense_layout(
        width,
        length,
        int(self.denseWindowWidth),
        int(self.denseWindowHeight),
        int(self.denseSearchWidth),
        int(self.denseSearchHeight),
        int(self.denseSkipWidth),
        int(self.denseSkipHeight),
        dense_margin,
    )
    if layout['num_windows'] <= 0:
        raise RuntimeError(
            'Dense offsets preflight failed: estimated window count is zero '
            '(num_down={0}, num_across={1}). Check dense window/search/skip settings.'.format(
                int(layout['num_down']),
                int(layout['num_across']),
            )
        )

    est_mem_mb = _estimate_dense_memory_mb(
        layout['num_windows'],
        dense_threads,
        int(self.denseWindowWidth),
        int(self.denseWindowHeight),
        int(self.denseSearchWidth),
        int(self.denseSearchHeight),
    )
    dense_threads, reduced, guard = _apply_dense_safeguards(
        layout['num_windows'],
        est_mem_mb,
        dense_threads,
    )
    if reduced:
        est_mem_mb = _estimate_dense_memory_mb(
            layout['num_windows'],
            dense_threads,
            int(self.denseWindowWidth),
            int(self.denseWindowHeight),
            int(self.denseSearchWidth),
            int(self.denseSearchHeight),
        )
        os.environ['OMP_NUM_THREADS'] = str(dense_threads)
        logger.warning(
            'Dense safeguard reduced threads due to soft limits: threads=%d, windows=%d, est_memory_mb=%.1f, '
            'soft_max_windows=%d, soft_max_est_memory_mb=%.1f',
            int(dense_threads),
            int(layout['num_windows']),
            float(est_mem_mb),
            int(guard['soft_max_windows']),
            float(guard['soft_max_mem_mb']),
        )

    _enforce_dense_hard_limits(layout['num_windows'], est_mem_mb, guard)

    logger.info(
        'Dense offsets preflight: estimated_windows=%d (%d x %d), estimated_memory_mb=%.1f, '
        'threads=%d, cpu_total=%d, thread_fraction=%.2f, OMP_NUM_THREADS=%s',
        int(layout['num_windows']),
        int(layout['num_down']),
        int(layout['num_across']),
        float(est_mem_mb),
        int(dense_threads),
        int(cpu_total),
        float(thread_frac),
        str(os.environ.get('OMP_NUM_THREADS', 'unset')),
    )

    field = None
    external_dense_enabled = _safe_bool_env('ISCE_DENSE_EXTERNAL_ENABLED', True)
    use_gpu_raw = getattr(self, 'useGPU', False)
    use_gpu = _safe_bool(use_gpu_raw, default=False)
    logger.info(
        'Dense offsets useGPU raw=%r (type=%s) parsed=%s',
        use_gpu_raw, type(use_gpu_raw).__name__, use_gpu
    )
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
        if external_dense_enabled:
            try:
                field = estimateOffsetFieldExternal(
                    referenceSlc,
                    secondarySlc,
                    denseOffsetFilename,
                    ww=self.denseWindowWidth,
                    wh=self.denseWindowHeight,
                    sw=self.denseSearchWidth,
                    shh=self.denseSearchHeight,
                    kw=self.denseSkipWidth,
                    kh=self.denseSkipHeight,
                    threads=dense_threads,
                )
                logger.info('Dense offsets computed with external coregistration path on CPU.')
            except Exception as err:
                logger.warning(
                    'External dense-offset path failed (%s). Falling back to CPU DenseAmpcor.',
                    str(err),
                    exc_info=True
                )
        else:
            logger.info('External dense-offset path disabled by ISCE_DENSE_EXTERNAL_ENABLED=0.')

    if field is None:
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
            covth=denseCovThreshold,
            threads=dense_threads,
        )

    self._insar.offset_top = field[0]
    self._insar.offset_left = field[1]

    return None
