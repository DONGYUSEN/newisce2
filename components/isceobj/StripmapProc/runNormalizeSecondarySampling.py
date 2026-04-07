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


def runNormalizeSecondarySampling(self):
    if hasattr(self, 'normalizeSecondaryForCorrelation') and (not bool(self.normalizeSecondaryForCorrelation)):
        logger.info(
            'normalizeSecondaryForCorrelation=False is ignored; secondary normalization now auto-triggers on PRF/DC/range mismatch.'
        )
    if hasattr(self, 'normalizeSecondaryForce') and bool(self.normalizeSecondaryForce):
        logger.info(
            'normalizeSecondaryForce=True is ignored; secondary normalization now uses automatic mismatch triggering.'
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

    prf_threshold = max(_safe_float(getattr(self, 'normalizeSecondaryPrfThreshold', 5.0e-4), 5.0e-4), 0.0)
    rsr_threshold = max(
        _safe_float(getattr(self, 'normalizeSecondaryRangeThreshold', 5.0e-4), 5.0e-4),
        0.0,
    )
    drift_threshold = max(
        _safe_float(getattr(self, 'normalizeSecondaryAzimuthDriftThreshold', 1.0), 1.0),
        0.0,
    )
    dc_threshold = max(
        _safe_float(getattr(self, 'normalizeSecondaryDopplerThreshold', 0.02), 0.02),
        0.0,
    )

    dc_metrics = _doppler_mismatch(referenceFrame, secondaryFrame)
    dc_affects = bool(dc_metrics['available']) and (dc_metrics['max_norm_diff'] >= dc_threshold)
    prf_affects = (prf_rel_diff >= prf_threshold) or (az_drift_lines >= drift_threshold)
    range_affects = (rsr_rel_diff >= rsr_threshold) or (rg_drift_pixels >= 1.0)

    need_norm = prf_affects or range_affects or dc_affects

    logger.info(
        'Secondary pre-normalization check: prf_rel_diff=%.8f, az_drift_lines=%.4f, '
        'rsr_rel_diff=%.8f, rg_drift_pixels=%.4f, dc_norm_diff=%.8f, '
        'prf_affects=%s, range_affects=%s, dc_affects=%s',
        prf_rel_diff,
        az_drift_lines,
        rsr_rel_diff,
        rg_drift_pixels,
        float(dc_metrics.get('max_norm_diff', 0.0)),
        str(prf_affects),
        str(range_affects),
        str(dc_affects),
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

    if dc_affects:
        harmonize = bool(getattr(self, 'normalizeSecondaryDopplerToReference', True))
        if harmonize:
            ref_dop = _doppler_coeffs(referenceFrame)
            if len(ref_dop) > 0:
                workFrame._dopplerVsPixel = list(ref_dop)
                logger.info('Secondary Doppler polynomial harmonized to reference Doppler.')
            else:
                logger.warning('Reference Doppler coefficients unavailable, skipping DC harmonization.')

    self._insar.saveProduct(workFrame, outxml)
    self._insar.secondarySlcCropProduct = outxml
    logger.info('Secondary pre-normalization product saved to %s', outxml)
    return None
