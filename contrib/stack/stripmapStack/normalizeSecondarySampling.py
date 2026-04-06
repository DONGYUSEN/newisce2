#!/usr/bin/env python3

import argparse
import copy
import datetime
import glob
import os
import shelve
import shutil

import numpy as np
import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D


def createParser():
    parser = argparse.ArgumentParser(
        description='Auto-normalize secondary PRF/DC before stripmapStack correlation.'
    )
    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
                        help='Reference metadata directory containing data shelve')
    parser.add_argument('-s', '--secondary', dest='secondary', type=str, required=True,
                        help='Secondary metadata directory containing data shelve')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, required=True,
                        help='Output normalized secondary metadata directory')
    parser.add_argument('--normalizeSecondaryPrfThreshold', dest='normalizeSecondaryPrfThreshold',
                        type=float, default=5.0e-4,
                        help='Relative PRF mismatch threshold (default: 5e-4)')
    parser.add_argument('--normalizeSecondaryAzimuthDriftThreshold', dest='normalizeSecondaryAzimuthDriftThreshold',
                        type=float, default=1.0,
                        help='Estimated azimuth drift threshold in lines (default: 1.0)')
    parser.add_argument('--normalizeSecondaryDopplerThreshold', dest='normalizeSecondaryDopplerThreshold',
                        type=float, default=0.02,
                        help='Normalized Doppler mismatch threshold (default: 0.02)')
    parser.add_argument('--normalizeSecondaryDopplerToReference', dest='normalizeSecondaryDopplerToReference',
                        type=int, default=1,
                        help='Set to 1 to harmonize secondary Doppler to reference, 0 to disable')
    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


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


def _build_azimuth_stretch_poly(alpha):
    slope = float(alpha - 1.0)
    poly = Poly2D()
    poly.setMeanAzimuth(0.0)
    poly.setNormAzimuth(1.0)
    poly.setMeanRange(0.0)
    poly.setNormRange(1.0)
    poly.initPoly(rangeOrder=0, azimuthOrder=1, coeffs=[[0.0], [slope]])
    return poly


def _resample_secondary_to_reference_prf(secondaryFrame, referenceFrame, outslc):
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


def _copy_secondary_metadata(srcdir, dstdir):
    os.makedirs(dstdir, exist_ok=True)
    copied = 0
    for fname in glob.glob(os.path.join(srcdir, 'data*')):
        if os.path.isfile(fname):
            shutil.copy2(fname, os.path.join(dstdir, os.path.basename(fname)))
            copied += 1
    if copied == 0:
        raise RuntimeError('No data* shelve files found in secondary directory: {0}'.format(srcdir))


def main(iargs=None):
    inps = cmdLineParse(iargs)

    with shelve.open(os.path.join(inps.reference, 'data'), flag='r') as rdb:
        referenceFrame = rdb['frame']

    with shelve.open(os.path.join(inps.secondary, 'data'), flag='r') as sdb:
        secondaryFrame = sdb['frame']

    _copy_secondary_metadata(inps.secondary, inps.outdir)

    ref_prf = max(_safe_float(referenceFrame.PRF, 0.0), 1.0e-6)
    sec_prf = max(_safe_float(secondaryFrame.PRF, 0.0), 1.0e-6)
    prf_rel_diff = abs(sec_prf - ref_prf) / ref_prf
    alpha = sec_prf / ref_prf

    lines = max(_safe_int(secondaryFrame.numberOfLines, 0), 1)
    az_drift_lines = abs(alpha - 1.0) * float(lines - 1)

    prf_threshold = max(_safe_float(inps.normalizeSecondaryPrfThreshold, 5.0e-4), 0.0)
    drift_threshold = max(_safe_float(inps.normalizeSecondaryAzimuthDriftThreshold, 1.0), 0.0)
    dc_threshold = max(_safe_float(inps.normalizeSecondaryDopplerThreshold, 0.02), 0.0)

    dc_metrics = _doppler_mismatch(referenceFrame, secondaryFrame)
    dc_affects = bool(dc_metrics['available']) and (dc_metrics['max_norm_diff'] >= dc_threshold)
    prf_affects = (prf_rel_diff >= prf_threshold) or (az_drift_lines >= drift_threshold)
    need_norm = prf_affects or dc_affects

    print('normalizeSecondarySampling:')
    print('  prf_rel_diff={0:.8f}, az_drift_lines={1:.4f}, dc_norm_diff={2:.8f}'.format(
        prf_rel_diff, az_drift_lines, float(dc_metrics.get('max_norm_diff', 0.0))))
    print('  triggers -> prf:{0} dc:{1}'.format(prf_affects, dc_affects))

    workFrame = copy.deepcopy(secondaryFrame)
    if prf_affects:
        outslc = os.path.join(inps.outdir, os.path.basename(secondaryFrame.getImage().filename))
        out_lines = _resample_secondary_to_reference_prf(secondaryFrame, referenceFrame, outslc)
        workFrame = _update_secondary_frame_for_prf(secondaryFrame, referenceFrame, outslc, out_lines)
        print('  PRF normalized: output SLC = {0}'.format(outslc))

    doppler_to_ref = (_safe_int(inps.normalizeSecondaryDopplerToReference, 1) != 0)
    if dc_affects and doppler_to_ref:
        ref_dop = _doppler_coeffs(referenceFrame)
        if len(ref_dop) > 0:
            workFrame._dopplerVsPixel = list(ref_dop)
            print('  DC harmonized to reference Doppler.')
        else:
            print('  DC harmonization skipped: reference Doppler unavailable.')

    with shelve.open(os.path.join(inps.outdir, 'data'), flag='w') as odb:
        odb['frame'] = workFrame
        odop = _doppler_coeffs(workFrame)
        if len(odop) == 0:
            odop = [0.0]
        odb['doppler'] = odop
        odb['secondary_normalization'] = {
            'needed': bool(need_norm),
            'prf_rel_diff': float(prf_rel_diff),
            'az_drift_lines': float(az_drift_lines),
            'dc_norm_diff': float(dc_metrics.get('max_norm_diff', 0.0)),
            'prf_trigger': bool(prf_affects),
            'dc_trigger': bool(dc_affects),
        }

    print('  normalized metadata written to: {0}'.format(inps.outdir))


if __name__ == '__main__':
    main()
