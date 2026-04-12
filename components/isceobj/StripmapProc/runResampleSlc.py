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
import glob

logger = logging.getLogger('isce.insar.runResampleSlc')


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


def _integrated_external_enabled(self=None):
    env_value = os.environ.get('ISCE_EXTERNAL_REGISTRATION_ENABLED')
    if env_value is not None:
        return _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_ENABLED', True)
    if self is not None and hasattr(self, 'useExternalCoregistration'):
        try:
            return bool(getattr(self, 'useExternalCoregistration'))
        except Exception:
            pass
    return True


def _use_rectified_range_offset(self=None):
    env_value = os.environ.get('ISCE_USE_RECT_RANGE_OFFSET')
    if env_value is not None:
        return _parse_bool(env_value, default=True)
    if self is not None and hasattr(self, 'useRdrdemRectRangeOffset'):
        try:
            return bool(getattr(self, 'useRdrdemRectRangeOffset'))
        except Exception:
            pass
    return False


def _resolved_range_offset_name(self, rgname):
    if not _use_rectified_range_offset(self):
        return rgname

    offsets_dir = os.path.dirname(rgname)
    candidates = []

    rect_from_attr = getattr(getattr(self, '_insar', None), 'rectRangeOffsetFilename', None)
    if rect_from_attr:
        candidates.append(os.path.join(offsets_dir, os.path.basename(str(rect_from_attr))))
    candidates.append(os.path.join(offsets_dir, 'range_rect.off'))

    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(cand + '.xml'):
            logger.info('Using rectified range offset for resampling/flatten: %s', cand)
            return cand

    logger.warning('Rectified range offset requested but not found; fallback to %s', rgname)
    return rgname


def _safe_float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _invalid_low_threshold():
    return _safe_float_env('ISCE_GEO2RDR_OFFSET_INVALID_LOW', -1.0e5)


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


def _valid_mask(arr, nodata, invalid_low):
    mask = np.isfinite(arr)
    if np.isfinite(nodata):
        mask &= (arr != nodata)
    if np.isfinite(invalid_low):
        mask &= (arr >= invalid_low)
    return mask


def _valid_bbox(mask):
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (r0, r1, c0, c1)


def _render_like(srcname, outname, dtype):
    outimg = isceobj.createImage()
    outimg.load(srcname + '.xml')
    outimg.filename = outname
    if dtype == np.float64:
        outimg.dataType = 'DOUBLE'
    elif dtype == np.float32:
        outimg.dataType = 'FLOAT'
    elif dtype == np.uint8:
        outimg.dataType = 'BYTE'
    outimg.setAccessMode('READ')
    outimg.renderHdr()
    return outimg


def _prepare_geo2rdr_offset_images(azname, rgname, force_mask=False):
    nodata = _safe_float_env('ISCE_GEO2RDR_OFFSET_NODATA', -999999.0)
    invalid_low = _invalid_low_threshold()
    enable_mask = _parse_bool_env('ISCE_GEO2RDR_OFFSET_MASK_ENABLE', True)
    if force_mask:
        enable_mask = True
    emit_mask = _parse_bool_env('ISCE_GEO2RDR_OFFSET_EMIT_VALIDMASK', True)
    fill_value = _safe_float_env('ISCE_GEO2RDR_OFFSET_FILL_VALUE', nodata)

    rngImg = isceobj.createImage()
    rngImg.load(rgname + '.xml')
    rngImg.setAccessMode('READ')
    aziImg = isceobj.createImage()
    aziImg.load(azname + '.xml')
    aziImg.setAccessMode('READ')

    width = rngImg.getWidth()
    length = rngImg.getLength()

    if (aziImg.getWidth() != width) or (aziImg.getLength() != length):
        logger.warning(
            'geo2rdr offset size mismatch, skip diagnostic/mask: az=(%d,%d) rg=(%d,%d)',
            aziImg.getWidth(),
            aziImg.getLength(),
            width,
            length,
        )
        return rngImg, aziImg, rgname

    az_dtype = _infer_offset_dtype(azname, width, length)
    rg_dtype = _infer_offset_dtype(rgname, width, length)

    az = np.memmap(azname, dtype=az_dtype, mode='r', shape=(length, width))
    rg = np.memmap(rgname, dtype=rg_dtype, mode='r', shape=(length, width))

    valid = _valid_mask(az, nodata, invalid_low) & _valid_mask(rg, nodata, invalid_low)
    total = int(width) * int(length)
    valid_count = int(np.count_nonzero(valid))
    invalid_count = total - valid_count
    valid_ratio = float(valid_count) / float(max(total, 1))
    bbox = _valid_bbox(valid)

    if valid_count > 0:
        az_valid = az[valid].astype(np.float64, copy=False)
        rg_valid = rg[valid].astype(np.float64, copy=False)
        logger.info(
            'geo2rdr offsets diagnostic: valid=%d/%d (%.2f%%), invalid=%d, '
            'valid_bbox=[row:%s, col:%s], az[min,max,mean]=[%.4f, %.4f, %.4f], '
            'rg[min,max,mean]=[%.4f, %.4f, %.4f], nodata=%.1f, invalid_low=%.1f',
            valid_count,
            total,
            100.0 * valid_ratio,
            invalid_count,
            'NA' if bbox is None else '%d..%d' % (bbox[0], bbox[1]),
            'NA' if bbox is None else '%d..%d' % (bbox[2], bbox[3]),
            float(np.min(az_valid)),
            float(np.max(az_valid)),
            float(np.mean(az_valid)),
            float(np.min(rg_valid)),
            float(np.max(rg_valid)),
            float(np.mean(rg_valid)),
            nodata,
            invalid_low,
        )
    else:
        logger.warning(
            'geo2rdr offsets diagnostic: no valid pixels found in az/range offsets '
            '(nodata=%.1f, invalid_low=%.1f).',
            nodata,
            invalid_low,
        )

    if not enable_mask:
        del rg
        del az
        return rngImg, aziImg, rgname

    az_out = azname + '.masked'
    rg_out = rgname + '.masked'
    az_w = np.memmap(az_out, dtype=az_dtype, mode='w+', shape=(length, width))
    rg_w = np.memmap(rg_out, dtype=rg_dtype, mode='w+', shape=(length, width))

    az_w[:, :] = az[:, :]
    rg_w[:, :] = rg[:, :]
    az_w[~valid] = fill_value
    rg_w[~valid] = fill_value
    az_w.flush()
    rg_w.flush()
    del az_w
    del rg_w

    if emit_mask:
        mask_name = azname + '.validmask'
        mask_w = np.memmap(mask_name, dtype=np.uint8, mode='w+', shape=(length, width))
        mask_w[:, :] = valid.astype(np.uint8, copy=False)
        mask_w.flush()
        del mask_w
        _render_like(azname, mask_name, np.uint8)

    del rg
    del az

    logger.info(
        'geo2rdr offsets sanitized with shared-valid mask: invalid pixels set to %.4f '
        '(ISCE_GEO2RDR_OFFSET_MASK_ENABLE=1).',
        fill_value,
    )

    rngImg = _render_like(rgname, rg_out, rg_dtype)
    aziImg = _render_like(azname, az_out, az_dtype)
    return rngImg, aziImg, rg_out


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


def _extract_initial_integer_offset(ext_meta):
    if not isinstance(ext_meta, dict):
        return 0.0, 0.0
    init = ext_meta.get('initial_integer_offset', None)
    if not isinstance(init, dict):
        return 0.0, 0.0
    try:
        az = float(init.get('azimuth', 0.0))
    except Exception:
        az = 0.0
    try:
        rg = float(init.get('range', 0.0))
    except Exception:
        rg = 0.0
    return az, rg


def _write_constant_offset_image(filename, width, length, value, dtype=np.float64):
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(int(length), int(width)))
    arr[:, :] = float(value)
    arr.flush()
    del arr

    img = isceobj.createImage()
    img.setFilename(filename)
    img.setWidth(int(width))
    img.setLength(int(length))
    img.setAccessMode('READ')
    img.bands = 1
    if dtype == np.float64:
        img.dataType = 'DOUBLE'
    else:
        img.dataType = 'FLOAT'
    img.scheme = 'BIP'
    img.renderHdr()
    return img


def _prepare_external_initial_offset_images(offsets_dir, width, length, init_az, init_rg):
    azname = os.path.join(offsets_dir, 'external_initial_azimuth.off')
    rgname = os.path.join(offsets_dir, 'external_initial_range.off')
    azimg = _write_constant_offset_image(azname, width, length, init_az, dtype=np.float64)
    rgimg = _write_constant_offset_image(rgname, width, length, init_rg, dtype=np.float64)
    return rgimg, azimg, rgname


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

    external_enabled = _integrated_external_enabled(self)
    if (kind == 'coarse') and external_enabled:
        logger.info(
            'External registration enabled: skip coarse resample output. '
            'geo2rdr offsets remain available for geometry/flattening chain.'
        )
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
    ext_meta = _load_external_registration_meta(misregFile) if (kind == 'refined') else None
    use_external_only = bool(ext_meta) and (
        str(ext_meta.get('resample_mode', '')).strip().lower() in ('external_only', 'external-only')
    )

    #Since the app is based on geometry module we expect pixel-by-pixel offset
    #field
    offsetsDir = self.insar.offsetsDirname 
    os.makedirs(offsetsDir, exist_ok=True)
    
    # Modified by V. Brancato 10.10.2019
    #rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    
    if kind in ['coarse', 'refined']:
        if (kind == 'refined') and use_external_only:
            init_az, init_rg = _extract_initial_integer_offset(ext_meta)
            width = min(
                int(referenceFrame.getImage().getWidth()),
                int(secondaryFrame.getImage().getWidth()),
            )
            length = min(
                int(referenceFrame.getImage().getLength()),
                int(secondaryFrame.getImage().getLength()),
            )
            logger.info(
                'Refined resample uses external-only offsets (no geo2rdr/coarse): '
                'initial_integer_offset=(az=%.3f, rg=%.3f), output_shape=%dx%d.',
                float(init_az),
                float(init_rg),
                int(length),
                int(width),
            )
            rngImg, aziImg, rgname_for_flatten = _prepare_external_initial_offset_images(
                offsetsDir,
                width,
                length,
                init_az,
                init_rg,
            )
            # External registration path keeps flattening explicit in interferogram
            # (range.off based), so refined resample does not pre-flatten here.
            flatten = False
        else:
            azname = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)
            rgname = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
            rgname = _resolved_range_offset_name(self, rgname)
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
           rgname = _resolved_range_offset_name(self, rgname)
           flatten=True
    
    if kind in ['coarse', 'refined']:
        if (kind == 'refined') and use_external_only:
            pass
        else:
            rgname_for_flatten = rgname
            rngImg, aziImg, rgname_for_flatten = _prepare_geo2rdr_offset_images(
                azname,
                rgname,
                force_mask=(kind == 'coarse'),
            )
    else:
        rgname_for_flatten = rgname
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

    # Default policy: do not merge rgpoly into range raster.
    # Keep official-style separation: geometry raster + polynomial correction.
    merge_rgpoly = _parse_bool_env('ISCE_REFINED_MERGE_RGPOLY_INTO_RASTER', False)
    if flatten and (rgpoly is not None) and merge_rgpoly:
        merged = _merge_range_poly_into_raster(rgname_for_flatten, width, length, rgpoly)
        if merged is not None:
            rngImg = merged
            rObj.rangeOffsetsPoly = None
            print('Flattening uses range offsets merged with misreg polynomial.')
    elif flatten and (rgpoly is not None):
        logger.info(
            'Range misregistration polynomial is kept separate from geo2rdr range raster '
            '(ISCE_REFINED_MERGE_RGPOLY_INTO_RASTER=0).'
        )

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
