#

import copy
import datetime
import logging
import os

import numpy as np
import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D

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


def _resample_secondary_to_reference_prf(self, secondaryFrame, referenceFrame, outslc):
    inimg = isceobj.createSlcImage()
    inimg.load(secondaryFrame.getImage().filename + '.xml')
    inimg.setAccessMode('READ')

    sec_prf = max(_safe_float(secondaryFrame.PRF, 0.0), 1.0e-6)
    ref_prf = max(_safe_float(referenceFrame.PRF, 0.0), 1.0e-6)
    alpha = sec_prf / ref_prf

    in_lines = _safe_int(secondaryFrame.numberOfLines, inimg.getLength())
    width = _safe_int(secondaryFrame.numberOfSamples, inimg.getWidth())
    out_lines = int(np.floor(float(in_lines - 1) / alpha) + 1)
    out_lines = max(1, out_lines)

    doppler = _doppler_coeffs(secondaryFrame)
    if len(doppler) == 0:
        doppler = [0.0]
    dcoeffs = [2.0 * np.pi * val / sec_prf for val in doppler]
    dpoly = Poly2D()
    dpoly.initPoly(rangeOrder=len(dcoeffs) - 1, azimuthOrder=0, coeffs=[dcoeffs])

    outimg = isceobj.createSlcImage()
    outimg.setFilename(outslc)
    outimg.setAccessMode('write')
    outimg.setWidth(width)

    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    rObj.radarWavelength = secondaryFrame.getInstrument().getRadarWavelength()
    rObj.dopplerPoly = dpoly
    rObj.azimuthOffsetsPoly = _build_azimuth_stretch_poly(alpha)
    rObj.flatten = False
    rObj.outputWidth = width
    rObj.outputLines = out_lines
    rObj.imageIn = inimg
    rObj.startingRange = secondaryFrame.startingRange
    rObj.referenceStartingRange = secondaryFrame.startingRange
    rObj.referenceSlantRangePixelSpacing = secondaryFrame.getInstrument().getRangePixelSize()
    rObj.referenceWavelength = secondaryFrame.getInstrument().getRadarWavelength()

    logger.info(
        'Secondary PRF normalization with alpha=secondaryPRF/referencePRF=%.9f, input_lines=%d, output_lines=%d',
        alpha,
        in_lines,
        out_lines,
    )
    rObj.resamp_slc(imageOut=outimg)

    outimg.setLength(out_lines)
    outimg.renderHdr()

    return out_lines


def _update_secondary_frame_for_prf(frame, referenceFrame, outslc, out_lines):
    outframe = copy.deepcopy(frame)

    outimg = isceobj.createSlcImage()
    outimg.load(outslc + '.xml')
    outframe.image = outimg
    outframe.image.filename = outslc

    outframe.image.width = _safe_int(frame.numberOfSamples, outimg.getWidth())
    outframe.image.length = int(out_lines)
    outframe.image.xmax = outframe.image.width
    outframe.image.coord1.coordSize = outframe.image.width
    outframe.image.coord1.coordEnd = outframe.image.width
    outframe.image.coord2.coordSize = int(out_lines)
    outframe.image.coord2.coordEnd = int(out_lines)

    outframe.numberOfSamples = outframe.image.width
    outframe.numberOfLines = int(out_lines)

    outframe.getInstrument().setPulseRepetitionFrequency(float(referenceFrame.PRF))

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
    enabled = bool(getattr(self, 'normalizeSecondaryForCorrelation', True))
    if not enabled:
        logger.info('Secondary PRF/DC pre-normalization disabled by configuration.')
        return None

    if (self._insar.referenceSlcCropProduct is None) or (self._insar.secondarySlcCropProduct is None):
        logger.warning('Missing cropped SLC products, skipping secondary pre-normalization.')
        return None

    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct(self._insar.secondarySlcCropProduct)

    ref_prf = max(_safe_float(referenceFrame.PRF, 0.0), 1.0e-6)
    sec_prf = max(_safe_float(secondaryFrame.PRF, 0.0), 1.0e-6)
    prf_rel_diff = abs(sec_prf - ref_prf) / ref_prf
    alpha = sec_prf / ref_prf

    lines = max(_safe_int(secondaryFrame.numberOfLines, 0), 1)
    az_drift_lines = abs(alpha - 1.0) * float(lines - 1)

    prf_threshold = max(_safe_float(getattr(self, 'normalizeSecondaryPrfThreshold', 5.0e-4), 5.0e-4), 0.0)
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

    force_norm = bool(getattr(self, 'normalizeSecondaryForce', False))
    need_norm = force_norm or prf_affects or dc_affects

    logger.info(
        'Secondary pre-normalization check: prf_rel_diff=%.8f, az_drift_lines=%.4f, '
        'dc_norm_diff=%.8f, prf_affects=%s, dc_affects=%s, force=%s',
        prf_rel_diff,
        az_drift_lines,
        float(dc_metrics.get('max_norm_diff', 0.0)),
        str(prf_affects),
        str(dc_affects),
        str(force_norm),
    )

    if not need_norm:
        logger.info('Secondary PRF/DC mismatch judged low impact; skipping pre-normalization.')
        return None

    stem = _stable_norm_stem(self._insar.secondarySlcCropProduct)
    outxml = stem + '_norm.xml'
    outdir = stem + '_norm'
    outslc = os.path.join(outdir, os.path.basename(secondaryFrame.getImage().filename))

    workFrame = copy.deepcopy(secondaryFrame)
    if prf_affects or force_norm:
        os.makedirs(outdir, exist_ok=True)
        out_lines = _resample_secondary_to_reference_prf(self, secondaryFrame, referenceFrame, outslc)
        workFrame = _update_secondary_frame_for_prf(secondaryFrame, referenceFrame, outslc, out_lines)

    if dc_affects or force_norm:
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
