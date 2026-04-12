#

import copy
import datetime
import logging
import os

import numpy as np
import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D
from isceobj.Constants import SPEED_OF_LIGHT

logger = logging.getLogger('isce.insar.runNormalizeSecondarySampling')


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if sval in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    return bool(default)


def _parse_bool_env(name, default=False):
    return _parse_bool(os.environ.get(name), default=default)


def _safe_float_env(name, default):
    return _safe_float(os.environ.get(name), default)


def _text_tag(value):
    if value is None:
        return ''
    try:
        return str(value).strip().lower()
    except Exception:
        return ''


def _is_lutan1_tag(text):
    token = ''.join(ch for ch in _text_tag(text) if ch.isalnum())
    return ('lutan1' in token) or ('lt1' in token)


def _is_lutan1_dataset(self, referenceFrame, secondaryFrame):
    candidates = []
    for attr in ('referenceSensorName', 'secondarySensorName', 'sensorName'):
        if hasattr(self, attr):
            candidates.append(getattr(self, attr))

    for frame in (referenceFrame, secondaryFrame):
        try:
            candidates.append(frame.getInstrument().getPlatform().getMission())
        except Exception:
            pass

    return any(_is_lutan1_tag(val) for val in candidates)


def _safe_range_sampling_rate(frame, default=1.0):
    try:
        rate = float(frame.getInstrument().getRangeSamplingRate())
    except Exception:
        rate = 0.0

    if rate > 0.0:
        return rate

    try:
        dr = float(frame.getInstrument().getRangePixelSize())
    except Exception:
        dr = 0.0

    if dr > 0.0:
        return float(SPEED_OF_LIGHT / (2.0 * dr))

    return max(float(default), 1.0e-6)


def _safe_range_pixel_size(frame, default=1.0):
    try:
        dr = float(frame.getInstrument().getRangePixelSize())
    except Exception:
        dr = 0.0

    if dr > 0.0:
        return dr

    rate = _safe_range_sampling_rate(frame, default=1.0)
    return float(SPEED_OF_LIGHT / (2.0 * rate))


def _doppler_coeffs(frame):
    vals = getattr(frame, '_dopplerVsPixel', None)
    if vals is None:
        return []

    out = []
    for val in list(vals):
        try:
            out.append(float(val))
        except Exception:
            out.append(0.0)
    return out


def _poly_eval(coeffs, xvals):
    if len(coeffs) == 0:
        return np.zeros_like(xvals, dtype=np.float64)
    return np.polyval(list(coeffs)[::-1], xvals)


def _doppler_mismatch(referenceFrame, secondaryFrame):
    ref = _doppler_coeffs(referenceFrame)
    sec = _doppler_coeffs(secondaryFrame)

    metrics = {
        'available': False,
        'max_norm_diff': 0.0,
        'max_hz_diff': 0.0,
    }

    if (len(ref) == 0) or (len(sec) == 0):
        return metrics

    width = min(
        _safe_int(referenceFrame.numberOfSamples, 0),
        _safe_int(secondaryFrame.numberOfSamples, 0),
    )
    width = max(width, 2)
    xvals = np.linspace(0.0, float(width - 1), num=5, dtype=np.float64)

    ref_hz = _poly_eval(ref, xvals)
    sec_hz = _poly_eval(sec, xvals)

    ref_prf = max(_safe_float(referenceFrame.PRF, 1.0), 1.0e-6)
    sec_prf = max(_safe_float(secondaryFrame.PRF, 1.0), 1.0e-6)

    ref_norm = ref_hz / ref_prf
    sec_norm = sec_hz / sec_prf

    metrics['available'] = True
    metrics['max_norm_diff'] = float(np.max(np.abs(ref_norm - sec_norm)))
    metrics['max_hz_diff'] = float(np.max(np.abs(ref_hz - sec_hz)))
    return metrics


def _stable_norm_stem(xmlname):
    stem = os.path.splitext(xmlname)[0]
    while stem.endswith('_norm'):
        stem = stem[:-5]
    return stem


def _build_azimuth_stretch_poly(alpha):
    # resamp_slc evaluates source_row ~= out_row + az_offset.
    # For PRF normalization to reference: source_row = out_row * alpha.
    # Therefore az_offset = (alpha - 1) * out_row.
    slope = float(alpha - 1.0)
    poly = Poly2D()
    poly.setMeanAzimuth(0.0)
    poly.setNormAzimuth(1.0)
    poly.setMeanRange(0.0)
    poly.setNormRange(1.0)
    poly.initPoly(rangeOrder=0, azimuthOrder=1, coeffs=[[0.0], [slope]])
    return poly


def _build_range_stretch_poly(beta):
    # resamp_slc evaluates source_col ~= out_col + rg_offset.
    # For range sampling normalization to reference: source_col = out_col * beta.
    # Therefore rg_offset = (beta - 1) * out_col.
    slope = float(beta - 1.0)
    poly = Poly2D()
    poly.setMeanAzimuth(0.0)
    poly.setNormAzimuth(1.0)
    poly.setMeanRange(0.0)
    poly.setNormRange(1.0)
    poly.initPoly(rangeOrder=1, azimuthOrder=0, coeffs=[[0.0, slope]])
    return poly


def _resample_secondary_to_reference_sampling(self, secondaryFrame, referenceFrame, outslc):
    inimg = isceobj.createSlcImage()
    inimg.load(secondaryFrame.getImage().filename + '.xml')
    inimg.setAccessMode('READ')

    sec_prf = max(_safe_float(secondaryFrame.PRF, 0.0), 1.0e-6)
    ref_prf = max(_safe_float(referenceFrame.PRF, 0.0), 1.0e-6)
    sec_rsr = max(_safe_range_sampling_rate(secondaryFrame, default=1.0), 1.0e-6)
    ref_rsr = max(_safe_range_sampling_rate(referenceFrame, default=1.0), 1.0e-6)
    alpha = sec_prf / ref_prf
    beta = sec_rsr / ref_rsr

    in_lines = _safe_int(secondaryFrame.numberOfLines, inimg.getLength())
    in_width = _safe_int(secondaryFrame.numberOfSamples, inimg.getWidth())
    out_lines = int(np.floor(float(in_lines - 1) / alpha) + 1)
    out_width = int(np.floor(float(in_width - 1) / beta) + 1)
    out_lines = max(1, out_lines)
    out_width = max(1, out_width)

    doppler = _doppler_coeffs(secondaryFrame)
    if len(doppler) == 0:
        doppler = [0.0]
    dcoeffs = [2.0 * np.pi * val / sec_prf for val in doppler]
    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(dcoeffs) - 1, azimuthOrder=0, coeffs=[dcoeffs])

    outimg = isceobj.createSlcImage()
    outimg.setFilename(outslc)
    outimg.setAccessMode('write')
    outimg.setWidth(out_width)

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    rObj.radarWavelength = secondaryFrame.getInstrument().getRadarWavelength()
    rObj.dopplerPoly = dpoly
    rObj.azimuthOffsetsPoly = _build_azimuth_stretch_poly(alpha)
    rObj.rangeOffsetsPoly = _build_range_stretch_poly(beta)
    rObj.flatten = False
    rObj.outputWidth = out_width
    rObj.outputLines = out_lines
    rObj.imageIn = inimg
    rObj.startingRange = secondaryFrame.startingRange
    rObj.referenceStartingRange = secondaryFrame.startingRange
    rObj.referenceSlantRangePixelSpacing = referenceFrame.getInstrument().getRangePixelSize()
    rObj.referenceWavelength = referenceFrame.getInstrument().getRadarWavelength()

    logger.info(
        'Secondary sampling normalization: alpha(PRF)=%.9f beta(RSR)=%.9f, '
        'input=(lines=%d,width=%d), output=(lines=%d,width=%d)',
        alpha,
        beta,
        in_lines,
        in_width,
        out_lines,
        out_width,
    )
    rObj.resamp_slc(imageOut=outimg)

    outimg.setLength(out_lines)
    outimg.renderHdr()

    return out_lines, out_width, beta


def _remap_doppler_for_range_scale(coeffs, beta, out_width):
    if len(coeffs) == 0:
        return []
    if out_width <= 1:
        return [float(coeffs[0])]

    order = max(0, len(coeffs) - 1)
    nfit = min(max(out_width, order + 1), 4096)
    x_out = np.linspace(0.0, float(out_width - 1), num=nfit, dtype=np.float64)
    x_src = float(beta) * x_out
    y_src = _poly_eval(coeffs, x_src)
    fit_order = min(order, nfit - 1)
    fit = np.polyfit(x_out, y_src, fit_order)
    mapped = list(fit[::-1])
    while len(mapped) < (order + 1):
        mapped.append(0.0)
    return mapped


def _update_secondary_frame_for_sampling(frame, referenceFrame, outslc, out_lines, out_width, beta):
    outframe = copy.deepcopy(frame)

    outimg = isceobj.createSlcImage()
    outimg.load(outslc + '.xml')
    outframe.image = outimg
    outframe.image.filename = outslc

    outframe.image.width = int(out_width)
    outframe.image.length = int(out_lines)
    outframe.image.xmax = outframe.image.width
    outframe.image.coord1.coordSize = outframe.image.width
    outframe.image.coord1.coordEnd = outframe.image.width
    outframe.image.coord2.coordSize = int(out_lines)
    outframe.image.coord2.coordEnd = int(out_lines)

    outframe.numberOfSamples = outframe.image.width
    outframe.numberOfLines = int(out_lines)

    outframe.getInstrument().setPulseRepetitionFrequency(float(referenceFrame.PRF))
    outframe.getInstrument().setRangeSamplingRate(_safe_range_sampling_rate(referenceFrame, default=1.0))
    outframe.getInstrument().setRangePixelSize(_safe_range_pixel_size(referenceFrame, default=1.0))

    try:
        outframe.setStartingRange(frame.startingRange)
    except Exception:
        pass
    try:
        dr = _safe_range_pixel_size(referenceFrame, default=1.0)
        far = float(frame.startingRange) + max(0, int(out_width) - 1) * dr
        outframe.setFarRange(far)
    except Exception:
        pass

    mapped_dop = _remap_doppler_for_range_scale(_doppler_coeffs(frame), beta, int(out_width))
    if len(mapped_dop) > 0:
        outframe._dopplerVsPixel = mapped_dop

    sensing_start = frame.getSensingStart()
    if out_lines > 1:
        sensing_stop = sensing_start + datetime.timedelta(seconds=float(out_lines - 1) / float(referenceFrame.PRF))
    else:
        sensing_stop = sensing_start
    sensing_mid = sensing_start + 0.5 * (sensing_stop - sensing_start)

    outframe.setSensingStart(sensing_start)
    outframe.setSensingStop(sensing_stop)
    outframe.setSensingMid(sensing_mid)
    return outframe


def _clone_frame_with_new_slc(frame, outslc):
    outframe = copy.deepcopy(frame)
    outimg = isceobj.createSlcImage()
    outimg.load(outslc + '.xml')
    outframe.image = outimg
    outframe.image.filename = outslc

    outframe.image.width = int(outimg.getWidth())
    outframe.image.length = int(outimg.getLength())
    outframe.image.xmax = outframe.image.width
    outframe.image.coord1.coordSize = outframe.image.width
    outframe.image.coord1.coordEnd = outframe.image.width
    outframe.image.coord2.coordSize = outframe.image.length
    outframe.image.coord2.coordEnd = outframe.image.length

    outframe.numberOfSamples = outframe.image.width
    outframe.numberOfLines = outframe.image.length
    return outframe


def _infer_slc_complex_dtype(filename, width, length):
    nelems = int(width) * int(length)
    fsize = os.path.getsize(filename)
    if fsize == nelems * np.dtype(np.complex64).itemsize:
        return np.complex64
    if fsize == nelems * np.dtype(np.complex128).itemsize:
        return np.complex128

    img = isceobj.createImage()
    img.load(filename + '.xml')
    dtype = str(getattr(img, 'dataType', '')).upper()
    if dtype in ('CFLOAT', 'COMPLEX', 'COMPLEX64'):
        return np.complex64
    if dtype in ('CDOUBLE', 'COMPLEX128'):
        return np.complex128

    raise ValueError(
        'Cannot infer complex dtype for SLC "{0}" (size={1}, width={2}, length={3}, xml={4}).'.format(
            filename, fsize, width, length, dtype
        )
    )


def _render_slc_like(srcname, outname, width, length):
    outimg = isceobj.createSlcImage()
    outimg.load(srcname + '.xml')
    outimg.filename = outname
    outimg.setWidth(int(width))
    outimg.setLength(int(length))
    outimg.setAccessMode('READ')
    outimg.renderHdr()
    return outimg


def _sensor_meta_xml_path(self, role):
    obj = getattr(self, role, None)
    if obj is not None:
        for attr in ('xml', 'xmlFile', '_xmlFile'):
            path = getattr(obj, attr, None)
            if path and os.path.exists(path):
                return path

    attr_name = '{0}MetaXml'.format(role)
    if hasattr(self, attr_name):
        path = getattr(self, attr_name)
        if path and os.path.exists(path):
            return path

    return None


def _extract_xml_scalar(xmlpath, tag_names):
    if (xmlpath is None) or (not os.path.exists(xmlpath)):
        return None

    try:
        import xml.etree.ElementTree as ET
        xtree = ET.parse(xmlpath).getroot()
    except Exception:
        return None

    wanted = set([name.lower() for name in tag_names])
    for elem in xtree.iter():
        tag = str(elem.tag).lower()
        if tag in wanted:
            text = (elem.text or '').strip()
            if text != '':
                try:
                    return float(text)
                except Exception:
                    continue
    return None


def _infer_total_processed_azimuth_bandwidth(self, frame, role, prf):
    override = _safe_float_env('ISCE_DC_BAZ_HZ_OVERRIDE', -1.0)
    if override > 0.0:
        return override, 'env:ISCE_DC_BAZ_HZ_OVERRIDE'

    frame_candidates = []
    for name in (
        'totalProcessedAzimuthBandwidth',
        'azimuthLookBandwidth',
        '_totalProcessedAzimuthBandwidth',
        '_azimuthLookBandwidth',
    ):
        if hasattr(frame, name):
            try:
                frame_candidates.append(float(getattr(frame, name)))
            except Exception:
                pass

    for val in frame_candidates:
        if val > 0.0:
            return val, 'frame-attribute'

    xmlpath = _sensor_meta_xml_path(self, role)
    xml_val = _extract_xml_scalar(
        xmlpath,
        [
            'totalProcessedAzimuthBandwidth',
            'azimuthLookBandwidth',
            'processedAzimuthBandwidth',
        ],
    )
    if (xml_val is not None) and (xml_val > 0.0):
        return float(xml_val), 'sensor-xml'

    ratio = max(_safe_float_env('ISCE_DC_BAZ_FALLBACK_RATIO', 0.55), 1.0e-3)
    fallback = max(float(prf) * ratio, 1.0)
    return fallback, 'fallback-ratio'


def _doppler_delta_hz_vector(referenceFrame, secondaryFrame, width):
    xvals = np.arange(max(int(width), 1), dtype=np.float64)
    ref = _poly_eval(_doppler_coeffs(referenceFrame), xvals)
    sec = _poly_eval(_doppler_coeffs(secondaryFrame), xvals)
    return sec - ref


def _doppler_delta_metrics(referenceFrame, secondaryFrame, width):
    ref = _doppler_coeffs(referenceFrame)
    sec = _doppler_coeffs(secondaryFrame)

    metrics = {
        'available': False,
        'max_abs_hz': 0.0,
        'max_abs_over_ref_prf': 0.0,
        'max_abs_over_sec_prf': 0.0,
    }

    if (len(ref) == 0) or (len(sec) == 0):
        return metrics

    width = max(int(width), 2)
    nsamp = min(max(width, 2), 8192)
    xvals = np.linspace(0.0, float(width - 1), num=nsamp, dtype=np.float64)
    ref_hz = _poly_eval(ref, xvals)
    sec_hz = _poly_eval(sec, xvals)
    delta_hz = sec_hz - ref_hz

    ref_prf = max(_safe_float(referenceFrame.PRF, 1.0), 1.0e-6)
    sec_prf = max(_safe_float(secondaryFrame.PRF, 1.0), 1.0e-6)

    max_abs_hz = float(np.max(np.abs(delta_hz)))
    metrics['available'] = True
    metrics['max_abs_hz'] = max_abs_hz
    metrics['max_abs_over_ref_prf'] = float(max_abs_hz / ref_prf)
    metrics['max_abs_over_sec_prf'] = float(max_abs_hz / sec_prf)
    return metrics


def _apply_secondary_dc_frequency_shift(frame, delta_hz, outslc):
    inname = frame.getImage().filename
    inimg = isceobj.createSlcImage()
    inimg.load(inname + '.xml')
    width = int(inimg.getWidth())
    length = int(inimg.getLength())
    if width <= 0 or length <= 0:
        raise RuntimeError('Invalid SLC dimensions for DC frequency shift: width={0}, length={1}'.format(width, length))

    prf = max(_safe_float(frame.PRF, 0.0), 1.0e-6)
    in_dtype = _infer_slc_complex_dtype(inname, width, length)
    out_dtype = np.complex64 if in_dtype == np.complex64 else np.complex128

    din = np.memmap(inname, dtype=in_dtype, mode='r', shape=(length, width))
    dout = np.memmap(outslc, dtype=out_dtype, mode='w+', shape=(length, width))

    two_pi_over_prf = np.float64(2.0 * np.pi / prf)
    delta = np.asarray(delta_hz, dtype=np.float64)
    if delta.size != width:
        raise RuntimeError('DC shift vector width mismatch: expected {0}, got {1}'.format(width, delta.size))

    chunk_lines = max(_safe_int(os.environ.get('ISCE_DC_SHIFT_CHUNK_LINES', 256), 256), 1)
    for start in range(0, length, chunk_lines):
        stop = min(length, start + chunk_lines)
        lines = np.arange(start, stop, dtype=np.float64)[:, None]
        phase = np.exp(-1j * two_pi_over_prf * (lines * delta[None, :])).astype(out_dtype, copy=False)
        block = din[start:stop, :].astype(out_dtype, copy=False)
        dout[start:stop, :] = block * phase

    dout.flush()
    del dout
    del din
    _render_slc_like(inname, outslc, width, length)
    return _clone_frame_with_new_slc(frame, outslc)


def _build_commonband_window(length, prf, bandwidth_hz):
    if length <= 1:
        return np.ones((max(length, 1),), dtype=np.float32)

    half_bw = 0.5 * max(float(bandwidth_hz), 0.0)
    taper_hz = max(_safe_float_env('ISCE_DC_COMMONBAND_TAPER_HZ', min(0.1 * half_bw, 25.0)), 0.0)
    taper_hz = min(taper_hz, half_bw)

    freqs = np.fft.fftfreq(length, d=1.0 / float(prf))
    af = np.abs(freqs)

    if half_bw <= 0.0:
        return np.zeros((length,), dtype=np.float32)

    if taper_hz <= 0.0:
        return (af <= half_bw).astype(np.float32)

    pass_edge = max(half_bw - taper_hz, 0.0)
    win = np.zeros((length,), dtype=np.float32)
    inner = af <= pass_edge
    trans = (af > pass_edge) & (af <= half_bw)
    win[inner] = 1.0
    if np.any(trans):
        xi = (af[trans] - pass_edge) / max(taper_hz, 1.0e-6)
        win[trans] = 0.5 * (1.0 + np.cos(np.pi * xi))
    return win


def _apply_azimuth_commonband_filter(frame, source_doppler_hz, target_doppler_hz, bandwidth_hz, outslc):
    inname = frame.getImage().filename
    inimg = isceobj.createSlcImage()
    inimg.load(inname + '.xml')
    width = int(inimg.getWidth())
    length = int(inimg.getLength())
    if width <= 0 or length <= 0:
        raise RuntimeError('Invalid SLC dimensions for common-band filtering: width={0}, length={1}'.format(width, length))

    prf = max(_safe_float(frame.PRF, 0.0), 1.0e-6)
    in_dtype = _infer_slc_complex_dtype(inname, width, length)
    out_dtype = np.complex64 if in_dtype == np.complex64 else np.complex128

    src = np.asarray(source_doppler_hz, dtype=np.float64)
    tgt = np.asarray(target_doppler_hz, dtype=np.float64)
    if (src.size != width) or (tgt.size != width):
        raise RuntimeError(
            'Common-band doppler vector width mismatch: width={0}, src={1}, tgt={2}'.format(width, src.size, tgt.size)
        )

    din = np.memmap(inname, dtype=in_dtype, mode='r', shape=(length, width))
    dout = np.memmap(outslc, dtype=out_dtype, mode='w+', shape=(length, width))

    win = _build_commonband_window(length, prf, bandwidth_hz).astype(out_dtype, copy=False)
    two_pi_over_prf = np.float64(2.0 * np.pi / prf)
    lines = np.arange(length, dtype=np.float64)[:, None]

    chunk_cols = max(_safe_int(os.environ.get('ISCE_DC_COMMONBAND_CHUNK_COLS', 256), 256), 1)
    for c0 in range(0, width, chunk_cols):
        c1 = min(width, c0 + chunk_cols)
        block = din[:, c0:c1].astype(out_dtype, copy=False)
        src_hz = src[c0:c1][None, :]
        tgt_hz = tgt[c0:c1][None, :]

        deramp = np.exp(-1j * two_pi_over_prf * (lines * src_hz)).astype(out_dtype, copy=False)
        reramp = np.exp(1j * two_pi_over_prf * (lines * tgt_hz)).astype(out_dtype, copy=False)

        base = block * deramp
        spec = np.fft.fft(base, axis=0)
        spec *= win[:, None]
        filt = np.fft.ifft(spec, axis=0).astype(out_dtype, copy=False)
        dout[:, c0:c1] = filt * reramp

    dout.flush()
    del dout
    del din
    _render_slc_like(inname, outslc, width, length)
    return _clone_frame_with_new_slc(frame, outslc)


def runNormalizeSecondarySampling(self):
    if hasattr(self, 'normalizeSecondaryForCorrelation') or hasattr(self, 'normalizeSecondaryForce'):
        logger.info(
            'XML normalize trigger switches are deprecated and ignored; '
            'secondary normalization now auto-triggers from measured PRF/RSR/DC mismatch.'
        )

    if (self._insar.referenceSlcCropProduct is None) or (self._insar.secondarySlcCropProduct is None):
        logger.warning('Missing cropped SLC products, skipping secondary pre-normalization.')
        return None

    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct(self._insar.secondarySlcCropProduct)

    ref_prf = max(_safe_float(referenceFrame.PRF, 0.0), 1.0e-6)
    sec_prf = max(_safe_float(secondaryFrame.PRF, 0.0), 1.0e-6)
    ref_rsr = max(_safe_range_sampling_rate(referenceFrame, default=1.0), 1.0e-6)
    sec_rsr = max(_safe_range_sampling_rate(secondaryFrame, default=1.0), 1.0e-6)
    prf_rel_diff = abs(sec_prf - ref_prf) / ref_prf
    rsr_rel_diff = abs(sec_rsr - ref_rsr) / ref_rsr
    alpha = sec_prf / ref_prf
    beta = sec_rsr / ref_rsr

    lines = max(_safe_int(secondaryFrame.numberOfLines, 0), 1)
    width = max(_safe_int(secondaryFrame.numberOfSamples, 0), 1)
    az_drift_lines = abs(alpha - 1.0) * float(lines - 1)
    rg_drift_pixels = abs(beta - 1.0) * float(width - 1)

    prf_threshold = max(_safe_float_env('ISCE_NORMALIZE_PRF_REL_THRESHOLD', 5.0e-4), 0.0)
    rsr_threshold = max(
        _safe_float_env('ISCE_NORMALIZE_RANGE_REL_THRESHOLD', 5.0e-4),
        0.0,
    )
    drift_threshold = max(
        _safe_float_env('ISCE_NORMALIZE_AZ_DRIFT_LINES_THRESHOLD', 1.0),
        0.0,
    )
    dc_threshold = max(
        _safe_float_env('ISCE_NORMALIZE_DOPPLER_NORM_THRESHOLD', 0.02),
        0.0,
    )

    dc_metrics = _doppler_mismatch(referenceFrame, secondaryFrame)
    dc_norm_diff = float(dc_metrics.get('max_norm_diff', 0.0))
    dc_affects = bool(dc_metrics['available']) and (dc_norm_diff >= dc_threshold)
    prf_affects = (prf_rel_diff >= prf_threshold) or (az_drift_lines >= drift_threshold)
    range_affects = (rsr_rel_diff >= rsr_threshold) or (rg_drift_pixels >= 1.0)
    force_dc_with_prf = _parse_bool_env(
        'ISCE_FORCE_DC_WITH_PRF',
        _parse_bool_env('ISCE_LT1_FORCE_DC_WITH_PRF', True),
    )
    force_dc_harmonize = bool(force_dc_with_prf and prf_affects)

    if os.environ.get('ISCE_DC_POLICY_ENABLE') is None:
        dc_policy_enabled = _parse_bool_env('ISCE_LT1_DC_POLICY_ENABLE', True)
    else:
        dc_policy_enabled = _parse_bool_env('ISCE_DC_POLICY_ENABLE', True)
    dc_small_thresh = max(_safe_float_env('ISCE_DC_SMALL_NORM_THRESHOLD', 0.02), 0.0)
    dc_medium_thresh = max(_safe_float_env('ISCE_DC_MEDIUM_NORM_THRESHOLD', 0.10), dc_small_thresh)
    overlap_force_commonband = min(max(_safe_float_env('ISCE_DC_OVERLAP_FORCE_COMMONBAND', 0.70), 0.0), 1.0)
    overlap_abort_threshold = min(max(_safe_float_env('ISCE_DC_OVERLAP_ABORT_THRESHOLD', 0.40), 0.0), 1.0)
    abort_on_very_low_overlap = _parse_bool_env('ISCE_DC_ABORT_ON_VERY_LOW_OVERLAP', False)
    dc_deramp_enabled = _parse_bool_env('ISCE_DC_DERAMP_ENABLE', True)
    dc_commonband_enabled = _parse_bool_env('ISCE_DC_COMMONBAND_ENABLE', True)
    dc_force_deramp_medium = _parse_bool_env('ISCE_DC_FORCE_DERAMP_FOR_MEDIUM', True)
    dc_force_deramp_large = _parse_bool_env('ISCE_DC_FORCE_DERAMP_FOR_LARGE', True)

    ref_baz_hz, ref_baz_src = _infer_total_processed_azimuth_bandwidth(self, referenceFrame, 'reference', ref_prf)
    sec_baz_hz, sec_baz_src = _infer_total_processed_azimuth_bandwidth(self, secondaryFrame, 'secondary', sec_prf)
    common_baz_hz = max(min(ref_baz_hz, sec_baz_hz), 1.0e-6)

    dc_delta_metrics = _doppler_delta_metrics(
        referenceFrame,
        secondaryFrame,
        min(
            max(_safe_int(referenceFrame.numberOfSamples, 0), 1),
            max(_safe_int(secondaryFrame.numberOfSamples, 0), 1),
        ),
    )
    if bool(dc_delta_metrics['available']):
        delta_fd_abs_hz = float(dc_delta_metrics['max_abs_hz'])
        max_df_over_prf = float(dc_delta_metrics['max_abs_over_ref_prf'])
    else:
        delta_fd_abs_hz = abs(dc_norm_diff) * 0.5 * (ref_prf + sec_prf)
        max_df_over_prf = dc_norm_diff
    overlap = max(0.0, min(1.0, 1.0 - (delta_fd_abs_hz / common_baz_hz)))

    if dc_policy_enabled and (not bool(dc_metrics['available'])):
        dc_regime = 'unavailable'
    elif not dc_policy_enabled:
        dc_regime = 'legacy'
    else:
        if (max_df_over_prf < dc_small_thresh) and (overlap >= overlap_force_commonband):
            dc_regime = 'small'
        elif (max_df_over_prf < dc_medium_thresh) and (overlap >= overlap_force_commonband):
            dc_regime = 'medium'
        else:
            dc_regime = 'large'

    dc_policy_small_harmonize = bool(
        dc_policy_enabled and (dc_regime == 'small') and bool(dc_delta_metrics['available']) and (max_df_over_prf > 0.0)
    )
    dc_policy_requires_processing = bool(dc_policy_enabled and (dc_regime in ('medium', 'large')))
    need_norm = (
        prf_affects
        or range_affects
        or dc_affects
        or force_dc_harmonize
        or dc_policy_requires_processing
        or dc_policy_small_harmonize
    )

    logger.info(
        'Secondary pre-normalization check: prf_rel_diff=%.8f, az_drift_lines=%.4f, '
        'rsr_rel_diff=%.8f, rg_drift_pixels=%.4f, dc_norm_diff=%.8f, '
        'prf_affects=%s, range_affects=%s, dc_affects=%s',
        prf_rel_diff,
        az_drift_lines,
        rsr_rel_diff,
        rg_drift_pixels,
        dc_norm_diff,
        str(prf_affects),
        str(range_affects),
        str(dc_affects),
    )
    logger.info(
        'DC policy metrics: enabled=%s regime=%s max_abs_df_over_prf=%.8f '
        'delta_fd_hz=%.6f baz_ref_hz=%.6f(%s) baz_sec_hz=%.6f(%s) '
        'common_baz_hz=%.6f overlap=%.6f',
        str(dc_policy_enabled),
        dc_regime,
        max_df_over_prf,
        delta_fd_abs_hz,
        ref_baz_hz,
        ref_baz_src,
        sec_baz_hz,
        sec_baz_src,
        common_baz_hz,
        overlap,
    )
    if dc_policy_enabled:
        logger.info(
            'DC policy thresholds: small=%.6f medium=%.6f force_commonband_overlap=%.3f '
            'abort_overlap=%.3f',
            dc_small_thresh,
            dc_medium_thresh,
            overlap_force_commonband,
            overlap_abort_threshold,
        )
        logger.info(
            'DC deramp policy: base_enable=%s force_medium=%s force_large=%s',
            str(dc_deramp_enabled),
            str(dc_force_deramp_medium),
            str(dc_force_deramp_large),
        )
    logger.info(
        'Auto-normalization thresholds: prf_rel>=%.8f range_rel>=%.8f '
        'az_drift_lines>=%.4f doppler_norm>=%.8f',
        prf_threshold,
        rsr_threshold,
        drift_threshold,
        dc_threshold,
    )
    logger.info(
        'PRF-triggered forced DC harmonization policy: enabled=%s effective=%s '
        '(env: ISCE_FORCE_DC_WITH_PRF; legacy alias: ISCE_LT1_FORCE_DC_WITH_PRF).',
        str(force_dc_with_prf),
        str(force_dc_harmonize),
    )

    if not need_norm:
        logger.info('Secondary PRF/DC mismatch judged low impact; skipping pre-normalization.')
        return None

    stem = _stable_norm_stem(self._insar.secondarySlcCropProduct)
    outxml = stem + '_norm.xml'
    outdir = stem + '_norm'
    outslc = os.path.join(outdir, os.path.basename(secondaryFrame.getImage().filename))

    workFrame = copy.deepcopy(secondaryFrame)
    if prf_affects or range_affects:
        os.makedirs(outdir, exist_ok=True)
        out_lines, out_width, beta = _resample_secondary_to_reference_sampling(
            self,
            secondaryFrame,
            referenceFrame,
            outslc,
        )
        workFrame = _update_secondary_frame_for_sampling(
            secondaryFrame,
            referenceFrame,
            outslc,
            out_lines,
            out_width,
            beta,
        )

    if dc_policy_enabled and (dc_regime in ('medium', 'large')):
        force_deramp = (
            (dc_regime == 'medium' and bool(dc_force_deramp_medium))
            or (dc_regime == 'large' and bool(dc_force_deramp_large))
        )
        deramp_effective = bool(dc_deramp_enabled) or bool(force_deramp)
        if not deramp_effective:
            logger.warning(
                'DC regime=%s requires deramp/reramp but ISCE_DC_DERAMP_ENABLE=0. '
                'Proceeding without signal-level frequency shift.',
                dc_regime,
            )
        else:
            if (not bool(dc_deramp_enabled)) and bool(force_deramp):
                logger.info(
                    'DC regime=%s forcing deramp/reramp despite ISCE_DC_DERAMP_ENABLE=0 '
                    '(force flag active).',
                    dc_regime,
                )
            os.makedirs(outdir, exist_ok=True)
            sec_delta_hz = _doppler_delta_hz_vector(
                referenceFrame,
                workFrame,
                max(_safe_int(workFrame.numberOfSamples, 0), 1),
            )
            shifted_slc = workFrame.getImage().filename + '.dcshift'
            workFrame = _apply_secondary_dc_frequency_shift(workFrame, sec_delta_hz, shifted_slc)
            logger.info(
                'Secondary deramp/reramp applied for DC regime=%s with |delta_fd|_max=%.6f Hz.',
                dc_regime,
                float(np.max(np.abs(sec_delta_hz))) if sec_delta_hz.size > 0 else 0.0,
            )

    if dc_policy_enabled and (dc_regime == 'large'):
        if overlap < overlap_abort_threshold:
            msg = (
                'Very low overlap detected (overlap={0:.6f} < {1:.6f}). '
                'Dataset may be unsuitable for high-resolution interferometry.'
            ).format(overlap, overlap_abort_threshold)
            if abort_on_very_low_overlap:
                raise RuntimeError(msg + ' Aborting as configured (ISCE_DC_ABORT_ON_VERY_LOW_OVERLAP=1).')
            logger.warning(msg + ' Continuing because ISCE_DC_ABORT_ON_VERY_LOW_OVERLAP=0.')

        if not dc_commonband_enabled:
            logger.warning(
                'DC regime=large but ISCE_DC_COMMONBAND_ENABLE=0. Skipping common-band filtering.'
            )
        else:
            common_bw_hz = max(min(ref_baz_hz, sec_baz_hz) - delta_fd_abs_hz, 0.0)
            min_common_bw_hz = max(_safe_float_env('ISCE_DC_COMMONBAND_MIN_BW_HZ', 50.0), 0.0)
            if common_bw_hz <= min_common_bw_hz:
                raise RuntimeError(
                    'Computed common azimuth bandwidth too small: {0:.6f} Hz (min required {1:.6f} Hz). '
                    'Consider lower resolution or reject this pair.'.format(common_bw_hz, min_common_bw_hz)
                )

            ref_dop = _doppler_coeffs(referenceFrame)
            sec_dop = _doppler_coeffs(workFrame)
            if (len(ref_dop) == 0) or (len(sec_dop) == 0):
                logger.warning(
                    'Doppler coefficients unavailable for common-band filtering. Skipping common-band step.'
                )
            else:
                os.makedirs(outdir, exist_ok=True)

                sec_width = max(_safe_int(workFrame.numberOfSamples, 0), 1)
                sec_x = np.arange(sec_width, dtype=np.float64)
                sec_src_hz = _poly_eval(sec_dop, sec_x)
                sec_tgt_hz = _poly_eval(ref_dop, sec_x)
                sec_cb_slc = workFrame.getImage().filename + '.cband'
                workFrame = _apply_azimuth_commonband_filter(
                    workFrame,
                    sec_src_hz,
                    sec_tgt_hz,
                    common_bw_hz,
                    sec_cb_slc,
                )

                ref_stem = _stable_norm_stem(self._insar.referenceSlcCropProduct)
                ref_dir = ref_stem + '_cband'
                os.makedirs(ref_dir, exist_ok=True)
                ref_cb_slc = os.path.join(ref_dir, os.path.basename(referenceFrame.getImage().filename))
                ref_width = max(_safe_int(referenceFrame.numberOfSamples, 0), 1)
                ref_x = np.arange(ref_width, dtype=np.float64)
                ref_hz = _poly_eval(ref_dop, ref_x)
                ref_cb_frame = _apply_azimuth_commonband_filter(
                    referenceFrame,
                    ref_hz,
                    ref_hz,
                    common_bw_hz,
                    ref_cb_slc,
                )
                ref_cb_xml = ref_stem + '_cband.xml'
                self._insar.saveProduct(ref_cb_frame, ref_cb_xml)
                self._insar.referenceSlcCropProduct = ref_cb_xml
                referenceFrame = ref_cb_frame

                logger.info(
                    'Common-band filtering applied for large DC mismatch: common_bw_hz=%.6f '
                    '(ref_baz_hz=%.6f, sec_baz_hz=%.6f, delta_fd_abs_hz=%.6f).',
                    common_bw_hz,
                    ref_baz_hz,
                    sec_baz_hz,
                    delta_fd_abs_hz,
                )

    dc_harmonize_required = bool(
        dc_affects
        or force_dc_harmonize
        or dc_policy_small_harmonize
        or (dc_policy_enabled and (dc_regime in ('medium', 'large')))
    )
    if dc_harmonize_required:
        harmonize = (
            _parse_bool_env('ISCE_NORMALIZE_HARMONIZE_DOPPLER', True)
            or force_dc_harmonize
            or dc_policy_small_harmonize
            or (dc_policy_enabled and (dc_regime in ('medium', 'large')))
        )
        if harmonize:
            ref_dop = _doppler_coeffs(referenceFrame)
            if len(ref_dop) > 0:
                workFrame._dopplerVsPixel = list(ref_dop)
                if force_dc_harmonize and (not dc_affects) and (not dc_policy_enabled):
                    logger.info(
                        'Secondary Doppler polynomial forcibly harmonized to reference Doppler for PRF normalization.'
                    )
                else:
                    logger.info('Secondary Doppler polynomial harmonized to reference Doppler.')
            else:
                logger.warning('Reference Doppler coefficients unavailable, skipping DC harmonization.')

    self._insar.saveProduct(workFrame, outxml)
    self._insar.secondarySlcCropProduct = outxml
    logger.info('Secondary pre-normalization product saved to %s', outxml)
    return None
