#!/usr/bin/env python3
#
# Author: Piyush Agram
# Copyright 2016
#
# Heresh Fattahi: Adopted for stack processing
# Updated: joint (template, search) probe, coarse-to-fine, GPU safety fallback

import argparse
import logging
import os
import subprocess
import time

import numpy as np

import isce
import isceobj
from isceobj.Location.Offset import Offset, OffsetField

import s1a_isce_utils as ut

logger = logging.getLogger('isce.topsinsar.rangecoreg')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _parse_bool_env(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _coerce_hw_pair(value, minimum=1):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError('Expected two values for (down, across), got {0}'.format(value))
        down = int(value[0])
        across = int(value[1])
    else:
        down = int(value)
        across = int(value)

    down = max(int(minimum), down)
    across = max(int(minimum), across)
    return int(down), int(across)


def _shape_label(pair):
    down, across = _coerce_hw_pair(pair, minimum=1)
    return '{0}x{1}'.format(int(down), int(across))


def _half_search_pair(search_half):
    sh, sw = _coerce_hw_pair(search_half, minimum=1)
    return max(8, int(np.rint(0.5 * sh))), max(8, int(np.rint(0.5 * sw)))


def _split_hw(text):
    token = str(text).strip().lower().replace(' ', '')
    if not token:
        raise ValueError('Empty size token')
    if 'x' in token:
        parts = token.split('x')
        if len(parts) != 2:
            raise ValueError('Invalid size token: {0}'.format(text))
        return int(parts[0]), int(parts[1])
    val = int(token)
    return val, val


def _default_probe_candidates():
    return [
        {'template_window': (128, 128), 'coarse_search_half': (64, 64)},
        {'template_window': (256, 256), 'coarse_search_half': (128, 128)},
        {'template_window': (512, 512), 'coarse_search_half': (256, 256)},
        {'template_window': (1024, 1024), 'coarse_search_half': (512, 512)},
        {'template_window': (1024, 1024), 'coarse_search_half': (1024, 1024)},
    ]


def _parse_probe_candidates(raw):
    if not raw:
        return _default_probe_candidates()

    out = []
    for item in str(raw).split(','):
        token = item.strip()
        if not token:
            continue

        if ':' in token:
            t_str, s_str = token.split(':', 1)
            template = _split_hw(t_str)
            search = _split_hw(s_str)
        else:
            # Backward shorthand: TxS means square template and square search.
            t, s = _split_hw(token)
            template = (int(t), int(t))
            search = (int(s), int(s))

        out.append(
            {
                'template_window': _coerce_hw_pair(template, minimum=16),
                'coarse_search_half': _coerce_hw_pair(search, minimum=4),
            }
        )

    if not out:
        raise ValueError('No valid probe candidates parsed from: {0}'.format(raw))

    return out


def createParser():
    parser = argparse.ArgumentParser(description='Estimate range misregistration using overlap bursts')

    parser.add_argument(
        '-o', '--out_range',
        type=str,
        dest='output',
        default='misreg.txt',
        help='Output textfile with the constant range offset',
    )
    parser.add_argument(
        '-t', '--snr_threshold',
        type=float,
        dest='offsetSNRThreshold',
        default=6.0,
        help='SNR threshold for overlap masking',
    )

    parser.add_argument('-m', '--reference', type=str, dest='reference', required=True, help='Reference image')
    parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True, help='Secondary image')

    parser.add_argument(
        '--probe_candidates',
        type=str,
        default=os.environ.get('ISCE_RANGE_MISREG_PROBE_CANDIDATES', ''),
        help=(
            'Joint probe candidates. Format: "T:S,T:S" where T/S are "downxacross" or scalar; '
            'example "128x128:64x64,256x256:128x128". Empty uses defaults.'
        ),
    )
    parser.add_argument(
        '--probe_grid',
        type=int,
        default=int(os.environ.get('ISCE_RANGE_MISREG_PROBE_GRID', 3)),
        help='Coarse probe grid size per dimension (default: 3 for 3x3).',
    )
    parser.add_argument(
        '--probe_refine_grid',
        type=int,
        default=int(os.environ.get('ISCE_RANGE_MISREG_PROBE_REFINE_GRID', 5)),
        help='Fine probe grid size per dimension (default: 5 for 5x5).',
    )
    parser.add_argument(
        '--final_locations_across',
        type=int,
        default=int(os.environ.get('ISCE_RANGE_MISREG_FINAL_LOC_ACROSS', 80)),
        help='Final dense match count across.',
    )
    parser.add_argument(
        '--final_locations_down',
        type=int,
        default=int(os.environ.get('ISCE_RANGE_MISREG_FINAL_LOC_DOWN', 20)),
        help='Final dense match count down.',
    )
    parser.add_argument(
        '--probe_min_valid_ratio',
        type=float,
        default=float(os.environ.get('ISCE_RANGE_MISREG_PROBE_MIN_VALID_RATIO', 0.55)),
        help='Minimum valid ratio to prefer in probe selection.',
    )
    parser.add_argument(
        '--probe_score_margin',
        type=float,
        default=float(os.environ.get('ISCE_RANGE_MISREG_PROBE_SCORE_MARGIN', 0.08)),
        help='Tie margin for probe score selection.',
    )
    parser.add_argument(
        '--probe_time_weight',
        type=float,
        default=float(os.environ.get('ISCE_RANGE_MISREG_PROBE_TIME_WEIGHT', 0.35)),
        help='Speed penalty factor in probe scoring.',
    )

    parser.set_defaults(useGPU=_parse_bool_env('ISCE_RANGE_MISREG_USE_GPU', True))
    parser.add_argument('--useGPU', dest='useGPU', action='store_true', help='Enable GPU Ampcor when available.')
    parser.add_argument('--noGPU', dest='useGPU', action='store_false', help='Disable GPU Ampcor and force CPU.')

    parser.set_defaults(cpuFallback=_parse_bool_env('ISCE_RANGE_MISREG_CPU_FALLBACK', True))
    parser.add_argument(
        '--cpu_fallback',
        dest='cpuFallback',
        action='store_true',
        help='Allow automatic CPU fallback when GPU run fails.',
    )
    parser.add_argument(
        '--no_cpu_fallback',
        dest='cpuFallback',
        action='store_false',
        help='Disable CPU fallback; fail if GPU fails.',
    )

    parser.add_argument(
        '--gpu_device',
        type=int,
        default=int(os.environ.get('ISCE_RANGE_MISREG_GPU_DEVICE', 0)),
        help='GPU device id for PyCuAmpcor.',
    )

    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def _cleanup_gpu_ampcor_outputs(prefix):
    for suffix in ('.bip', '.gross', '_snr.bip', '_cov.bip'):
        path = prefix + suffix
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _query_gpu_memory_mb(device_id=0):
    info = {'total_mb': None, 'free_mb': None}

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info['total_mb'] = int(mem.total // (1024 * 1024))
        info['free_mb'] = int(mem.free // (1024 * 1024))
        return info
    except Exception:
        pass

    cmd = [
        'nvidia-smi',
        '--query-gpu=memory.total,memory.free',
        '--format=csv,noheader,nounits',
        '-i',
        str(int(device_id)),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True).strip()
        row = out.splitlines()[0].split(',')
        info['total_mb'] = int(row[0].strip())
        info['free_mb'] = int(row[1].strip())
    except Exception:
        pass

    return info


def _recommend_gpu_streams(template_window, free_mb):
    template_h, template_w = _coerce_hw_pair(template_window, minimum=1)
    edge = max(template_h, template_w)

    if free_mb is None:
        return 1

    if edge >= 1024:
        return 1 if free_mb < 12000 else 2
    if edge >= 512:
        return 1 if free_mb < 8192 else 2
    if edge >= 256:
        return 2 if free_mb < 12288 else 4
    return 2 if free_mb < 8192 else 4


def _recommend_gpu_chunks(template_window, search_half, num_across, num_down, free_mb, n_streams):
    template_h, template_w = _coerce_hw_pair(template_window, minimum=16)
    search_h, search_w = _coerce_hw_pair(search_half, minimum=4)
    chip_h = int(template_h + 2 * search_h)
    chip_w = int(template_w + 2 * search_w)

    if free_mb is None:
        return max(1, min(int(num_down), 4)), max(1, min(int(num_across), 8))

    budget_bytes = int(float(free_mb) * 1024.0 * 1024.0 * 0.30 / max(1, int(n_streams)))
    bytes_per_window = max(1, chip_h * chip_w * 4 * 10)
    max_windows = max(1, int(budget_bytes // bytes_per_window))

    chunk_across = max(1, min(int(num_across), int(np.sqrt(max_windows))))
    chunk_down = max(1, min(int(num_down), max_windows // max(1, chunk_across)))

    return int(chunk_down), int(chunk_across)


def _ensure_binary_slc(infile):
    if os.path.isfile(infile):
        return

    cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
    status = os.system(cmd)
    if status:
        raise RuntimeError('Failed to materialize SLC binary for GPU Ampcor: {0}'.format(cmd))


def _ampcor_field_cpu(
    reference,
    secondary,
    template_window=(64, 64),
    search_half=(16, 16),
    num_locations_across=80,
    num_locations_down=20,
    gross_range=0,
    gross_azimuth=0,
):
    from mroipac.ampcor.Ampcor import Ampcor

    mImg = isceobj.createSlcImage()
    mImg.load(reference + '.xml')
    mImg.setAccessMode('READ')
    mImg.createImage()

    sImg = isceobj.createSlcImage()
    sImg.load(secondary + '.xml')
    sImg.setAccessMode('READ')
    sImg.createImage()

    objAmpcor = Ampcor('ampcor_burst')
    objAmpcor.configure()
    objAmpcor.setImageDataType1('mag')
    objAmpcor.setImageDataType2('mag')

    wh, ww = _coerce_hw_pair(template_window, minimum=16)
    shh, sw = _coerce_hw_pair(search_half, minimum=4)

    objAmpcor.windowSizeWidth = int(ww)
    objAmpcor.windowSizeHeight = int(wh)
    objAmpcor.searchWindowSizeWidth = int(sw)
    objAmpcor.searchWindowSizeHeight = int(shh)
    objAmpcor.oversamplingFactor = 32

    xMargin = 2 * objAmpcor.searchWindowSizeWidth + objAmpcor.windowSizeWidth
    yMargin = 2 * objAmpcor.searchWindowSizeHeight + objAmpcor.windowSizeHeight

    firstAc = 1000
    coarseAcross = int(np.rint(gross_range))
    coarseDown = int(np.rint(gross_azimuth))

    min_width = min(mImg.getWidth(), sImg.getWidth())
    min_length = min(mImg.getLength(), sImg.getLength())

    offAc = int(firstAc + xMargin + max(0, -coarseAcross))
    offDn = int(objAmpcor.windowSizeHeight // 2 + 1 + yMargin + max(0, -coarseDown))
    lastAc = int(min_width - firstAc - xMargin - max(0, coarseAcross))
    lastDn = int(min_length - objAmpcor.windowSizeHeight // 2 - 1 - yMargin - max(0, coarseDown))

    if (lastAc <= offAc) or (lastDn <= offDn):
        mImg.finalizeImage()
        sImg.finalizeImage()
        raise ValueError(
            'Invalid CPU Ampcor ROI: offAc={0}, lastAc={1}, offDn={2}, lastDn={3}'.format(
                offAc, lastAc, offDn, lastDn
            )
        )

    objAmpcor.setFirstSampleAcross(offAc)
    objAmpcor.setLastSampleAcross(lastAc)
    objAmpcor.setNumberLocationAcross(max(2, int(num_locations_across)))

    objAmpcor.setFirstSampleDown(offDn)
    objAmpcor.setLastSampleDown(lastDn)
    objAmpcor.setNumberLocationDown(max(2, int(num_locations_down)))

    objAmpcor.setAcrossGrossOffset(coarseAcross)
    objAmpcor.setDownGrossOffset(coarseDown)

    objAmpcor.setFirstPRF(1.0)
    objAmpcor.setSecondPRF(1.0)
    objAmpcor.setFirstRangeSpacing(1.0)
    objAmpcor.setSecondRangeSpacing(1.0)

    objAmpcor(mImg, sImg)

    mImg.finalizeImage()
    sImg.finalizeImage()

    return objAmpcor.getOffsetField()


def _ampcor_field_gpu(
    reference,
    secondary,
    out_prefix,
    template_window=(128, 128),
    search_half=(40, 40),
    num_locations_across=60,
    num_locations_down=60,
    gross_range=0,
    gross_azimuth=0,
    device_id=0,
    n_streams=None,
    chunk_down=None,
    chunk_across=None,
    mmap_gb=8,
):
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
    sim_width = sim.getWidth()
    sim_length = sim.getLength()

    wh, ww = _coerce_hw_pair(template_window, minimum=16)
    shh, sw = _coerce_hw_pair(search_half, minimum=4)

    margin_across = 2 * sw + ww
    margin_down = 2 * shh + wh

    nAcross = max(2, int(num_locations_across))
    nDown = max(2, int(num_locations_down))

    offAc = int(max(101, -int(np.rint(gross_range))) + margin_across)
    offDn = int(max(101, -int(np.rint(gross_azimuth))) + margin_down)
    lastAc = int(min(width, sim_width - offAc) - margin_across)
    lastDn = int(min(length, sim_length - offDn) - margin_down)

    if (lastAc <= offAc) or (lastDn <= offDn):
        raise ValueError(
            'Invalid GPU Ampcor ROI: offAc={0}, lastAc={1}, offDn={2}, lastDn={3}'.format(
                offAc, lastAc, offDn, lastDn
            )
        )

    skip_across = int((lastAc - offAc) / (nAcross - 1.0))
    skip_down = int((lastDn - offDn) / (nDown - 1.0))
    if (skip_across <= 0) or (skip_down <= 0):
        raise ValueError(
            'Invalid GPU Ampcor skip spacing: skipAcross={0}, skipDown={1}'.format(
                skip_across, skip_down
            )
        )

    num_across = int((lastAc - offAc) / skip_across) + 1
    num_down = int((lastDn - offDn) / skip_down) + 1

    objOffset = PyCuAmpcor.PyCuAmpcor()
    objOffset.algorithm = 0
    objOffset.derampMethod = 1
    objOffset.referenceImageName = reference + '.vrt'
    objOffset.referenceImageHeight = length
    objOffset.referenceImageWidth = width
    objOffset.secondaryImageName = secondary + '.vrt'
    objOffset.secondaryImageHeight = sim_length
    objOffset.secondaryImageWidth = sim_width

    objOffset.windowSizeWidth = int(ww)
    objOffset.windowSizeHeight = int(wh)
    objOffset.halfSearchRangeAcross = int(sw)
    objOffset.halfSearchRangeDown = int(shh)
    objOffset.skipSampleAcross = int(skip_across)
    objOffset.skipSampleDown = int(skip_down)
    objOffset.corrSurfaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = 16

    objOffset.referenceStartPixelDownStatic = int(offDn)
    objOffset.referenceStartPixelAcrossStatic = int(offAc)
    objOffset.numberWindowDown = int(num_down)
    objOffset.numberWindowAcross = int(num_across)

    gpu_mem = _query_gpu_memory_mb(device_id)
    free_mb = gpu_mem.get('free_mb')
    total_mb = gpu_mem.get('total_mb')

    if n_streams is None:
        n_streams = _recommend_gpu_streams((wh, ww), free_mb)

    if chunk_down is None or chunk_across is None:
        rec_down, rec_across = _recommend_gpu_chunks(
            (wh, ww),
            (shh, sw),
            num_across=num_across,
            num_down=num_down,
            free_mb=free_mb,
            n_streams=n_streams,
        )
        if chunk_down is None:
            chunk_down = rec_down
        if chunk_across is None:
            chunk_across = rec_across

    objOffset.deviceID = int(device_id)
    objOffset.nStreams = max(1, int(n_streams))
    objOffset.numberWindowDownInChunk = max(1, min(int(num_down), int(chunk_down)))
    objOffset.numberWindowAcrossInChunk = max(1, min(int(num_across), int(chunk_across)))
    objOffset.mmapSize = max(1, int(mmap_gb))

    objOffset.offsetImageName = out_prefix + '.bip'
    objOffset.grossOffsetImageName = out_prefix + '.gross'
    objOffset.snrImageName = out_prefix + '_snr.bip'
    objOffset.covImageName = out_prefix + '_cov.bip'
    objOffset.mergeGrossOffset = 1

    logger.info(
        'GPU Ampcor config template=(%d,%d) search_half=(%d,%d) stride=(%d,%d) grid=(%d,%d) '
        'chunk=(%d,%d) streams=%d device=%d gpu_mem_mb(total/free)=(%s/%s)',
        wh,
        ww,
        shh,
        sw,
        skip_down,
        skip_across,
        num_down,
        num_across,
        objOffset.numberWindowDownInChunk,
        objOffset.numberWindowAcrossInChunk,
        objOffset.nStreams,
        objOffset.deviceID,
        str(total_mb),
        str(free_mb),
    )

    _cleanup_gpu_ampcor_outputs(out_prefix)
    objOffset.setupParams()
    objOffset.setConstantGrossOffset(int(np.rint(gross_range)), int(np.rint(gross_azimuth)))
    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()

    offset_raw = np.fromfile(objOffset.offsetImageName, dtype=np.float32)
    snr_raw = np.fromfile(objOffset.snrImageName, dtype=np.float32)
    cov_raw = np.fromfile(objOffset.covImageName, dtype=np.float32)

    expected_offset = num_down * num_across * 2
    expected_snr = num_down * num_across
    expected_cov = num_down * num_across * 3

    if offset_raw.size != expected_offset:
        raise RuntimeError('GPU offset output size mismatch: got {0}, expected {1}'.format(offset_raw.size, expected_offset))
    if snr_raw.size != expected_snr:
        raise RuntimeError('GPU SNR output size mismatch: got {0}, expected {1}'.format(snr_raw.size, expected_snr))
    if cov_raw.size != expected_cov:
        raise RuntimeError('GPU covariance output size mismatch: got {0}, expected {1}'.format(cov_raw.size, expected_cov))

    offset_bip = offset_raw.reshape(num_down, num_across * 2)
    snr_img = snr_raw.reshape(num_down, num_across)
    cov_bip = cov_raw.reshape(num_down, num_across * 3)

    field = OffsetField()
    az_center = (wh - 1) // 2
    rg_center = (ww - 1) // 2

    for idn in range(num_down):
        down = int(offDn + idn * skip_down + az_center)
        for iac in range(num_across):
            across = int(offAc + iac * skip_across + rg_center)

            az_off = float(offset_bip[idn, 2 * iac])
            rg_off = float(offset_bip[idn, 2 * iac + 1])

            snr_val = float(snr_img[idn, iac])
            cov_azaz = float(cov_bip[idn, 3 * iac])
            cov_rgrg = float(cov_bip[idn, 3 * iac + 1])
            cov_azrg = float(cov_bip[idn, 3 * iac + 2])

            one = Offset()
            one.setCoordinate(across, down)
            one.setOffset(rg_off, az_off)
            one.setSignalToNoise(snr_val)
            one.setCovariance(cov_rgrg, cov_azaz, cov_azrg)
            field.addOffset(one)

    _cleanup_gpu_ampcor_outputs(out_prefix)
    return field


def _run_gpu_ampcor_with_retry(reference, secondary, out_prefix, **kwargs):
    device_id = int(kwargs.get('device_id', 0))
    template_window = _coerce_hw_pair(kwargs.get('template_window', (128, 128)), minimum=16)

    gpu_mem = _query_gpu_memory_mb(device_id)
    free_mb = gpu_mem.get('free_mb')

    base_streams = kwargs.get('n_streams')
    if base_streams is None:
        base_streams = _recommend_gpu_streams(template_window, free_mb)

    retry_plan = [
        {'n_streams': max(1, int(base_streams)), 'chunk_down': kwargs.get('chunk_down'), 'chunk_across': kwargs.get('chunk_across')},
        {'n_streams': max(1, int(base_streams)), 'chunk_down': 1, 'chunk_across': 1},
        {'n_streams': 1, 'chunk_down': 1, 'chunk_across': 1},
    ]

    errors = []
    seen = set()
    for plan in retry_plan:
        key = (int(plan['n_streams']), plan['chunk_down'], plan['chunk_across'])
        if key in seen:
            continue
        seen.add(key)

        attempt = dict(kwargs)
        attempt['n_streams'] = plan['n_streams']
        attempt['chunk_down'] = plan['chunk_down']
        attempt['chunk_across'] = plan['chunk_across']

        try:
            return _ampcor_field_gpu(reference, secondary, out_prefix, **attempt)
        except Exception as err:
            errors.append({'streams': plan['n_streams'], 'chunk_down': plan['chunk_down'], 'chunk_across': plan['chunk_across'], 'error': str(err)})
            message = str(err)
            if ('Invalid GPU Ampcor ROI' in message) or ('Invalid GPU Ampcor skip spacing' in message):
                raise
            logger.warning(
                'GPU Ampcor retry failed streams=%s chunk_down=%s chunk_across=%s: %s',
                str(plan['n_streams']),
                str(plan['chunk_down']),
                str(plan['chunk_across']),
                message,
            )

    raise RuntimeError('GPU Ampcor failed after retries: {0}'.format(errors))


def _run_ampcor_field(
    reference,
    secondary,
    out_prefix,
    template_window,
    search_half,
    num_locations_across,
    num_locations_down,
    gross_range,
    gross_azimuth,
    use_gpu,
    cpu_fallback,
    gpu_device,
):
    if use_gpu:
        try:
            field = _run_gpu_ampcor_with_retry(
                reference,
                secondary,
                out_prefix,
                template_window=template_window,
                search_half=search_half,
                num_locations_across=num_locations_across,
                num_locations_down=num_locations_down,
                gross_range=gross_range,
                gross_azimuth=gross_azimuth,
                device_id=gpu_device,
            )
            return field, 'gpu'
        except Exception as err:
            if not cpu_fallback:
                raise
            logger.warning('GPU Ampcor failed; fallback to CPU Ampcor. reason: %s', str(err))

    field = _ampcor_field_cpu(
        reference,
        secondary,
        template_window=template_window,
        search_half=search_half,
        num_locations_across=num_locations_across,
        num_locations_down=num_locations_down,
        gross_range=gross_range,
        gross_azimuth=gross_azimuth,
    )
    return field, 'cpu'


def _offset_arrays(field):
    rg = []
    az = []
    snr = []
    across = []
    down = []

    for one in field:
        rg.append(float(one.dx))
        az.append(float(one.dy))
        snr.append(float(one.snr))
        across.append(float(one.x))
        down.append(float(one.y))

    return {
        'rg': np.asarray(rg, dtype=np.float64),
        'az': np.asarray(az, dtype=np.float64),
        'snr': np.asarray(snr, dtype=np.float64),
        'across': np.asarray(across, dtype=np.float64),
        'down': np.asarray(down, dtype=np.float64),
    }


def _linear_fit_rms(across, down, values):
    if values.size < 3:
        return float(np.inf)
    A = np.stack([across, down, np.ones_like(across)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    pred = np.dot(A, coeffs)
    return float(np.sqrt(np.mean((values - pred) ** 2)))


def _evaluate_field(field, snr_threshold):
    arr = _offset_arrays(field)
    total = int(arr['rg'].size)
    if total == 0:
        return {
            'valid_count': 0,
            'valid_ratio': 0.0,
            'mean_snr': 0.0,
            'peak_snr': 0.0,
            'rg_std': float(np.inf),
            'az_std': float(np.inf),
            'rg_fit_rms': float(np.inf),
            'az_fit_rms': float(np.inf),
            'fit_rms': float(np.inf),
            'rg_median': 0.0,
            'az_median': 0.0,
            'quality': 0.0,
        }

    valid = np.isfinite(arr['rg'])
    valid &= np.isfinite(arr['az'])
    valid &= np.isfinite(arr['snr'])
    valid &= arr['snr'] >= float(snr_threshold)

    count = int(np.count_nonzero(valid))
    if count == 0:
        return {
            'valid_count': 0,
            'valid_ratio': 0.0,
            'mean_snr': 0.0,
            'peak_snr': 0.0,
            'rg_std': float(np.inf),
            'az_std': float(np.inf),
            'rg_fit_rms': float(np.inf),
            'az_fit_rms': float(np.inf),
            'fit_rms': float(np.inf),
            'rg_median': 0.0,
            'az_median': 0.0,
            'quality': 0.0,
        }

    rg = arr['rg'][valid]
    az = arr['az'][valid]
    snr = arr['snr'][valid]
    across = arr['across'][valid]
    down = arr['down'][valid]

    rg_std = float(np.std(rg))
    az_std = float(np.std(az))
    rg_fit_rms = _linear_fit_rms(across, down, rg)
    az_fit_rms = _linear_fit_rms(across, down, az)
    fit_rms = float(np.sqrt(rg_fit_rms ** 2 + az_fit_rms ** 2))

    valid_ratio = float(count) / float(total)
    mean_snr = float(np.mean(snr))
    peak_snr = float(np.max(snr))

    quality = float(valid_ratio * peak_snr / (1.0 + fit_rms + rg_std))

    return {
        'valid_count': count,
        'valid_ratio': valid_ratio,
        'mean_snr': mean_snr,
        'peak_snr': peak_snr,
        'rg_std': rg_std,
        'az_std': az_std,
        'rg_fit_rms': float(rg_fit_rms),
        'az_fit_rms': float(az_fit_rms),
        'fit_rms': fit_rms,
        'rg_median': float(np.median(rg)),
        'az_median': float(np.median(az)),
        'quality': quality,
    }


def _joint_probe_for_candidate(reference, secondary, work_dir, candidate, inps):
    template = _coerce_hw_pair(candidate['template_window'], minimum=16)
    coarse_search = _coerce_hw_pair(candidate['coarse_search_half'], minimum=4)
    refine_search = _half_search_pair(coarse_search)

    tag = 't{0}_s{1}'.format(_shape_label(template), _shape_label(coarse_search))
    coarse_prefix = os.path.join(work_dir, 'probe_{0}_coarse'.format(tag))
    refine_prefix = os.path.join(work_dir, 'probe_{0}_refine'.format(tag))

    coarse_t0 = time.perf_counter()
    coarse_field, coarse_engine = _run_ampcor_field(
        reference,
        secondary,
        coarse_prefix,
        template_window=template,
        search_half=coarse_search,
        num_locations_across=max(3, int(inps.probe_grid)),
        num_locations_down=max(3, int(inps.probe_grid)),
        gross_range=0.0,
        gross_azimuth=0.0,
        use_gpu=bool(inps.useGPU),
        cpu_fallback=bool(inps.cpuFallback),
        gpu_device=int(inps.gpu_device),
    )
    coarse_elapsed = float(time.perf_counter() - coarse_t0)
    coarse_metrics = _evaluate_field(coarse_field, inps.offsetSNRThreshold)

    gross_rg = float(coarse_metrics['rg_median'])
    gross_az = float(coarse_metrics['az_median'])

    refine_t0 = time.perf_counter()
    refine_field, refine_engine = _run_ampcor_field(
        reference,
        secondary,
        refine_prefix,
        template_window=template,
        search_half=refine_search,
        num_locations_across=max(3, int(inps.probe_refine_grid)),
        num_locations_down=max(3, int(inps.probe_refine_grid)),
        gross_range=float(np.rint(gross_rg)),
        gross_azimuth=float(np.rint(gross_az)),
        use_gpu=bool(inps.useGPU),
        cpu_fallback=bool(inps.cpuFallback),
        gpu_device=int(inps.gpu_device),
    )
    refine_elapsed = float(time.perf_counter() - refine_t0)
    refine_metrics = _evaluate_field(refine_field, inps.offsetSNRThreshold)

    elapsed = float(coarse_elapsed + refine_elapsed)
    denom = max(elapsed, 1.0e-3) ** float(max(0.0, inps.probe_time_weight))
    score = float(refine_metrics['quality'] / denom)

    return {
        'template_window': template,
        'coarse_search_half': coarse_search,
        'refine_search_half': refine_search,
        'gross_range': float(refine_metrics['rg_median']),
        'gross_azimuth': float(refine_metrics['az_median']),
        'coarse_probe': coarse_metrics,
        'refine_probe': refine_metrics,
        'coarse_elapsed_sec': coarse_elapsed,
        'refine_elapsed_sec': refine_elapsed,
        'elapsed_sec': elapsed,
        'valid_count': int(refine_metrics['valid_count']),
        'valid_ratio': float(refine_metrics['valid_ratio']),
        'peak_snr': float(refine_metrics['peak_snr']),
        'fit_rms': float(refine_metrics['fit_rms']),
        'quality': float(refine_metrics['quality']),
        'score': score,
        'engine': 'gpu' if ('gpu' in (coarse_engine, refine_engine)) else 'cpu',
    }


def _select_joint_probe(reference, secondary, work_dir, inps):
    candidates = _parse_probe_candidates(inps.probe_candidates)

    metrics = []
    for candidate in candidates:
        row = {
            'template_window': tuple(candidate['template_window']),
            'coarse_search_half': tuple(candidate['coarse_search_half']),
            'success': False,
            'error': None,
        }
        try:
            one = _joint_probe_for_candidate(reference, secondary, work_dir, candidate, inps)
            row.update(one)
            row['success'] = True
        except Exception as err:
            row['error'] = str(err)
            logger.warning(
                'Probe candidate failed template=%s coarse_search_half=%s: %s',
                _shape_label(candidate['template_window']),
                _shape_label(candidate['coarse_search_half']),
                str(err),
            )

        metrics.append(row)

    good = [x for x in metrics if x.get('success')]
    if not good:
        raise RuntimeError('Joint probe failed for all candidates.')

    for row in good:
        logger.info(
            'Probe template=%s coarse_search_half=%s refine_search_half=%s: '
            'time=%.3fs valid_ratio=%.3f peak_snr=%.3f fit_rms=%.4f score=%.5f engine=%s',
            _shape_label(row['template_window']),
            _shape_label(row['coarse_search_half']),
            _shape_label(row['refine_search_half']),
            float(row['elapsed_sec']),
            float(row['valid_ratio']),
            float(row['peak_snr']),
            float(row['fit_rms']),
            float(row['score']),
            str(row.get('engine', 'cpu')),
        )

    min_valid = max(0.0, min(1.0, float(inps.probe_min_valid_ratio)))
    tie_margin = max(0.0, float(inps.probe_score_margin))

    eligible = [x for x in good if float(x['valid_ratio']) >= min_valid]
    pool = eligible if eligible else good

    best_score = max(float(x['score']) for x in pool)
    near_best = [x for x in pool if float(x['score']) >= (1.0 - tie_margin) * best_score]

    chosen = sorted(
        near_best,
        key=lambda x: (
            float(x['fit_rms']),
            -float(x['valid_ratio']),
            float(x['elapsed_sec']),
            -int(x['template_window'][0]),
            -int(x['coarse_search_half'][0]),
        ),
    )[0]

    logger.info(
        'Selected template=%s coarse_search_half=%s refine_search_half=%s '
        '(score=%.5f, fit_rms=%.4f, valid_ratio=%.3f, engine=%s)',
        _shape_label(chosen['template_window']),
        _shape_label(chosen['coarse_search_half']),
        _shape_label(chosen['refine_search_half']),
        float(chosen['score']),
        float(chosen['fit_rms']),
        float(chosen['valid_ratio']),
        str(chosen.get('engine', 'cpu')),
    )

    return chosen, metrics


def _run_coarse_to_fine(reference, secondary, work_dir, tag, selected, inps):
    template = _coerce_hw_pair(selected['template_window'], minimum=16)
    coarse_search = _coerce_hw_pair(selected['coarse_search_half'], minimum=4)
    refine_search = _coerce_hw_pair(selected['refine_search_half'], minimum=4)

    coarse_prefix = os.path.join(work_dir, '{0}_coarse'.format(tag))
    refine_prefix = os.path.join(work_dir, '{0}_refine'.format(tag))

    coarse_field, coarse_engine = _run_ampcor_field(
        reference,
        secondary,
        coarse_prefix,
        template_window=template,
        search_half=coarse_search,
        num_locations_across=max(3, int(inps.probe_grid)),
        num_locations_down=max(3, int(inps.probe_grid)),
        gross_range=0.0,
        gross_azimuth=0.0,
        use_gpu=bool(inps.useGPU),
        cpu_fallback=bool(inps.cpuFallback),
        gpu_device=int(inps.gpu_device),
    )
    coarse_metrics = _evaluate_field(coarse_field, inps.offsetSNRThreshold)

    gross_rg = float(np.rint(coarse_metrics['rg_median']))
    gross_az = float(np.rint(coarse_metrics['az_median']))

    refine_field, refine_engine = _run_ampcor_field(
        reference,
        secondary,
        refine_prefix,
        template_window=template,
        search_half=refine_search,
        num_locations_across=max(2, int(inps.final_locations_across)),
        num_locations_down=max(2, int(inps.final_locations_down)),
        gross_range=gross_rg,
        gross_azimuth=gross_az,
        use_gpu=bool(inps.useGPU),
        cpu_fallback=bool(inps.cpuFallback),
        gpu_device=int(inps.gpu_device),
    )

    return refine_field, coarse_metrics, ('gpu' if ('gpu' in (coarse_engine, refine_engine)) else 'cpu')


def _write_result(output, medianval, meanval, stdval, snr_threshold, npts, selected):
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output, 'w') as f:
        f.write('median : ' + str(medianval) + '\n')
        f.write('mean : ' + str(meanval) + '\n')
        f.write('std : ' + str(stdval) + '\n')
        f.write('snr threshold : ' + str(snr_threshold) + '\n')
        f.write('mumber of coherent points : ' + str(npts) + '\n')

        if selected is not None:
            f.write('template window : ' + _shape_label(selected['template_window']) + '\n')
            f.write('coarse search half : ' + _shape_label(selected['coarse_search_half']) + '\n')
            f.write('refine search half : ' + _shape_label(selected['refine_search_half']) + '\n')
            f.write('probe score : ' + str(selected.get('score')) + '\n')
            f.write('probe valid ratio : ' + str(selected.get('valid_ratio')) + '\n')
            f.write('probe fit rms : ' + str(selected.get('fit_rms')) + '\n')


def main(iargs=None):
    inps = cmdLineParse(iargs)

    reference_swaths = ut.getSwathList(os.path.join(inps.reference, 'overlap'))
    secondary_swaths = ut.getSwathList(os.path.join(inps.secondary, 'overlap'))
    swath_list = list(sorted(set(reference_swaths + secondary_swaths)))

    # Use first valid burst pair for global joint probe selection.
    probe_reference = None
    probe_secondary = None

    burst_pairs = []
    range_pixel_size = None

    for swath in swath_list:
        referenceTop = ut.loadProduct(os.path.join(inps.reference, 'overlap', 'IW{0}_top.xml'.format(swath)))
        referenceBottom = ut.loadProduct(os.path.join(inps.reference, 'overlap', 'IW{0}_bottom.xml'.format(swath)))
        secondaryTop = ut.loadProduct(os.path.join(inps.secondary, 'overlap', 'IW{0}_top.xml'.format(swath)))
        secondaryBottom = ut.loadProduct(os.path.join(inps.secondary, 'overlap', 'IW{0}_bottom.xml'.format(swath)))

        if range_pixel_size is None:
            range_pixel_size = referenceTop.bursts[0].rangePixelSize

        min_reference = referenceTop.bursts[0].burstNumber
        max_reference = referenceTop.bursts[-1].burstNumber
        min_secondary = secondaryTop.bursts[0].burstNumber
        max_secondary = secondaryTop.bursts[-1].burstNumber

        min_burst = max(min_secondary, min_reference)
        max_burst = min(max_secondary, max_reference) + 1

        for name, pair in (('top', (referenceTop, secondaryTop)), ('bottom', (referenceBottom, secondaryBottom))):
            for ii in range(min_burst, max_burst):
                mFile = pair[0].bursts[ii - min_reference].image.filename
                sFile = pair[1].bursts[ii - min_secondary].image.filename
                burst_pairs.append((swath, name, ii, mFile, sFile))
                if probe_reference is None:
                    probe_reference = mFile
                    probe_secondary = sFile

    if not burst_pairs:
        raise RuntimeError('No common overlap bursts found for range misregistration estimate.')

    output_dir = os.path.dirname(inps.output) if os.path.dirname(inps.output) else '.'
    probe_work_dir = os.path.join(output_dir, '.range_misreg_probe_tmp')
    os.makedirs(probe_work_dir, exist_ok=True)

    selected, _ = _select_joint_probe(probe_reference, probe_secondary, probe_work_dir, inps)

    range_offsets = []
    snr = []

    for swath, burst_name, ii, mFile, sFile in burst_pairs:
        tag = 'IW{0}_{1}_{2}'.format(swath, burst_name, ii)

        field, coarse_metrics, engine = _run_coarse_to_fine(
            mFile,
            sFile,
            probe_work_dir,
            tag,
            selected,
            inps,
        )

        logger.info(
            'Processed %s with %s coarse-to-fine, gross(range/az)=%.3f/%.3f',
            tag,
            engine,
            float(coarse_metrics['rg_median']),
            float(coarse_metrics['az_median']),
        )

        for offset in field:
            range_offsets.append(float(offset.dx))
            snr.append(float(offset.snr))

    range_offsets = np.asarray(range_offsets, dtype=np.float64)
    snr = np.asarray(snr, dtype=np.float64)

    mask = np.logical_and(snr > float(inps.offsetSNRThreshold), np.abs(range_offsets) < 1.2)
    val = range_offsets[mask]

    if val.size == 0:
        raise RuntimeError(
            'No valid range offsets left after SNR/outlier culling. '
            'Try lower --snr_threshold or larger search range candidates.'
        )

    medianval = float(np.median(val))
    meanval = float(np.mean(val))
    stdval = float(np.std(val))

    medianval = medianval * float(range_pixel_size)
    meanval = meanval * float(range_pixel_size)
    stdval = stdval * float(range_pixel_size)

    _write_result(
        inps.output,
        medianval=medianval,
        meanval=meanval,
        stdval=stdval,
        snr_threshold=inps.offsetSNRThreshold,
        npts=int(val.size),
        selected=selected,
    )


if __name__ == '__main__':
    main()
