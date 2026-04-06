#

import isce
import isceobj
from iscesys.StdOEL.StdOELPy import create_writer
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.StripmapProc.externalRegistration import estimate_misregistration_polys
from isceobj.Util.Poly2D import Poly2D
from isceobj.Location.Offset import OffsetField, Offset

import numpy as np
import os
import shelve
import logging

logger = logging.getLogger('isce.insar.runRefineSecondaryTiming')


class ExternalRegistrationQualityError(RuntimeError):
    """
    Raised when integrated external registration violates quality gates.
    """

    def __init__(self, message, violations=None):
        RuntimeError.__init__(self, message)
        self.violations = list(violations or [])


def _parse_bool_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    sval = str(val).strip().lower()
    return sval in ('1', 'true', 'yes', 'on')


def _safe_int_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return int(default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _safe_float_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _safe_float_list_env(name, default_values):
    val = os.environ.get(name)
    if val is None:
        return list(default_values)

    text = str(val).replace(',', ' ')
    out = []
    for token in text.split():
        try:
            fval = float(token)
        except Exception:
            continue
        if fval > 0.0:
            out.append(fval)

    if not out:
        return list(default_values)

    uniq = []
    seen = set()
    for fval in out:
        key = round(fval, 8)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(fval)

    return uniq


def _safe_nonnegative_int(value, default):
    try:
        return max(0, int(value))
    except Exception:
        return int(default)


def _safe_int_list_env(name, default_values):
    val = os.environ.get(name)
    if val is None:
        return [int(v) for v in default_values]

    text = str(val).replace(',', ' ')
    out = []
    for token in text.split():
        try:
            ival = int(token)
        except Exception:
            continue
        if ival > 0:
            out.append(ival)

    if not out:
        return [int(v) for v in default_values]

    uniq = []
    seen = set()
    for ival in out:
        if ival in seen:
            continue
        seen.add(ival)
        uniq.append(int(ival))
    return uniq


def _integrated_external_enabled():
    return _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_ENABLED', True)


def _prefer_gpu_ampcor():
    return _parse_bool_env('ISCE_PREFER_GPU_AMPCOR', True)


def _allow_cpu_ampcor_fallback():
    return _parse_bool_env('ISCE_ALLOW_CPU_AMPCOR_FALLBACK', False)


def _gpu_ampcor_available(self):
    if not bool(getattr(self, 'useGPU', False)):
        return False

    try:
        return bool(self._insar.hasGPU())
    except Exception as err:
        logger.warning('Failed to query GPU Ampcor availability: %s', str(err))
        return False


def _ensure_binary_slc(infile):
    '''
    PyCuAmpcor expects a binary ENVI payload in addition to VRT/XML metadata.
    '''
    if os.path.isfile(infile):
        return

    cmd = 'gdal_translate -of ENVI {0}.vrt {0}'.format(infile)
    status = os.system(cmd)
    if status:
        raise RuntimeError(
            'Failed to materialize SLC binary for GPU Ampcor with command: {0}'.format(cmd)
        )


def _integrated_external_config():
    cfg = {
        'coarse_window': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_WINDOW_SIZE', 256),
        'coarse_multiscale': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_COARSE_MULTISCALE', True),
        'coarse_window_factors': _safe_float_list_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_WINDOW_FACTORS',
            [0.5, 1.0, 2.0],
        ),
        'coarse_search_ranges': _safe_int_list_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_SEARCH_RANGES',
            [32, 64, 128, 256, 512, 1024],
        ),
        'coarse_window_scale': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_COARSE_WINDOW_SCALE', 4.0),
        'coarse_grid_size': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_COARSE_GRID_SIZE', 3),
        'coarse_consistency_priority': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_CONSISTENCY_PRIORITY',
            True,
        ),
        'coarse_correlation_threshold': _safe_float_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_CORRELATION_THRESHOLD',
            0.06,
        ),
        'coarse_min_window': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_COARSE_MIN_WINDOW', 96),
        'coarse_max_window': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_COARSE_MAX_WINDOW', 4096),
        'coarse_min_valid': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_COARSE_MIN_VALID', 9),
        'coarse_prefer_larger_window': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_PREFER_LARGER_WINDOW',
            True,
        ),
        'coarse_log_candidates': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_COARSE_LOG', True),
        'coarse_quality_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_COARSE_QUALITY', 0.06),
        'coarse_auto_efficiency_balance': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_AUTO_EFFICIENCY_BALANCE',
            True,
        ),
        'coarse_quality_margin_for_smaller_window': _safe_float_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_QUALITY_MARGIN',
            0.03,
        ),
        'coarse_spread_margin_for_smaller_window': _safe_float_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_SPREAD_MARGIN',
            2.0,
        ),
        'fine_window': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_WINDOW', 128),
        'fine_spacing': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_SPACING', 128),
        'fine_quality_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_FINE_QUALITY', 0.05),
        'fine_workers': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_WORKERS', 0),
        'fine_chunk_size': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_CHUNK_SIZE', 128),
        'precompute_amplitude': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_PRECOMPUTE_AMP', True),
        'max_points': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MAX_POINTS', 0),
        'max_iterations': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MAX_ITERATIONS', 8),
        'sigma_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_SIGMA_THRESHOLD', 2.5),
        'min_points': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MIN_POINTS', 36),
    }
    return cfg


def _integrated_external_quality_gates():
    gates = {
        'max_azimuth_rms': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_MAX_AZ_RMS', 1.0),
        'max_coarse_spread': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_MAX_COARSE_SPREAD', 2.0),
        'min_coarse_valid': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MIN_COARSE_VALID', 3),
    }
    return gates


def _integrated_external_keep_on_quality_failure():
    return _parse_bool_env(
        'ISCE_EXTERNAL_REGISTRATION_NO_AMPCOR_FALLBACK_ON_QUALITY_FAIL',
        True,
    )


def _integrated_external_force_dense_rubbersheet_on_quality_failure():
    return _parse_bool_env(
        'ISCE_EXTERNAL_REGISTRATION_FORCE_DENSE_RUBBERSHEET_ON_QUALITY_FAIL',
        True,
    )


def _integrated_external_no_ampcor_fallback_on_error():
    return _parse_bool_env(
        'ISCE_EXTERNAL_REGISTRATION_NO_AMPCOR_FALLBACK_ON_ERROR',
        True,
    )


def _validate_external_registration_quality(ext_meta, gates):
    coarse = (ext_meta or {}).get('coarse') or {}
    fit = (ext_meta or {}).get('fit') or {}

    try:
        az_rms = float(fit.get('azimuth_rms', np.inf))
    except Exception:
        az_rms = np.inf

    try:
        coarse_spread = float(coarse.get('spread', np.inf))
    except Exception:
        coarse_spread = np.inf

    try:
        coarse_valid = int(coarse.get('num_valid', 0))
    except Exception:
        coarse_valid = 0

    violations = []

    if coarse_valid < int(gates['min_coarse_valid']):
        violations.append(
            'coarse valid corners {0} < {1}'.format(
                coarse_valid,
                int(gates['min_coarse_valid']),
            )
        )

    if coarse_spread > float(gates['max_coarse_spread']):
        violations.append(
            'coarse spread {0:.4f} > {1:.4f}'.format(
                coarse_spread,
                float(gates['max_coarse_spread']),
            )
        )

    if az_rms > float(gates['max_azimuth_rms']):
        violations.append(
            'azimuth rms {0:.4f} > {1:.4f}'.format(
                az_rms,
                float(gates['max_azimuth_rms']),
            )
        )

    if violations:
        raise ExternalRegistrationQualityError(
            'Integrated external registration rejected by quality gate: ' + '; '.join(violations),
            violations=violations,
        )

    return None


def _save_external_solution(self, outShelveFile, azpoly, rgpoly, ext_meta):
    odb = shelve.open(outShelveFile)
    odb['raw_field'] = None
    odb['cull_field'] = None
    odb['azpoly'] = azpoly
    odb['rgpoly'] = rgpoly
    odb['external_registration'] = ext_meta
    odb.close()

    self._insar.saveProduct(azpoly, outShelveFile + '_az.xml')
    self._insar.saveProduct(rgpoly, outShelveFile + '_rg.xml')
    return None


def _zero_poly():
    poly = Poly2D()
    poly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])
    return poly


def _save_zero_misreg_polys(self, outShelveFile, ext_meta):
    azpoly = _zero_poly()
    rgpoly = _zero_poly()
    _save_external_solution(self, outShelveFile, azpoly, rgpoly, ext_meta)
    return None


def _enable_dense_rubbersheet(self, reason):
    before = {
        'doDenseOffsets': bool(getattr(self, 'doDenseOffsets', False)),
        'doRubbersheetingAzimuth': bool(getattr(self, 'doRubbersheetingAzimuth', False)),
        'doRubbersheetingRange': bool(getattr(self, 'doRubbersheetingRange', False)),
    }

    logger.warning(
        'Dense/rubbersheet auto-enable requested due to %s, '
        'but auto-enable is disabled. Keeping XML-configured flags unchanged: %s',
        reason,
        before,
    )
    return None


def _fallback_fit_config(self):
    """
    Configure polynomial orders for the Ampcor fallback path.
    """
    defaults = {
        'azaz_order': _safe_int_env('ISCE_FALLBACK_MISREG_AZAZ_ORDER', 2),
        'azrg_order': _safe_int_env('ISCE_FALLBACK_MISREG_AZRG_ORDER', 2),
        'rgaz_order': _safe_int_env('ISCE_FALLBACK_MISREG_RGAZ_ORDER', 2),
        'rgrg_order': _safe_int_env('ISCE_FALLBACK_MISREG_RGRG_ORDER', 2),
        'snr': _safe_float_env('ISCE_FALLBACK_MISREG_SNR_THRESHOLD', 5.0),
    }

    cfg = {
        'azaz_order': _safe_nonnegative_int(
            getattr(self, 'refineTimingAzimuthAzimuthOrder', defaults['azaz_order']),
            defaults['azaz_order'],
        ),
        'azrg_order': _safe_nonnegative_int(
            getattr(self, 'refineTimingAzimuthRangeOrder', defaults['azrg_order']),
            defaults['azrg_order'],
        ),
        'rgaz_order': _safe_nonnegative_int(
            getattr(self, 'refineTimingRangeAzimuthOrder', defaults['rgaz_order']),
            defaults['rgaz_order'],
        ),
        'rgrg_order': _safe_nonnegative_int(
            getattr(self, 'refineTimingRangeRangeOrder', defaults['rgrg_order']),
            defaults['rgrg_order'],
        ),
        'snr': float(getattr(self, 'refineTimingSnrThreshold', defaults['snr'])),
    }

    return cfg


def estimateOffsetField(reference, secondary, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between burst and simamp.
    '''


    sim = isceobj.createSlcImage()
    sim.load(secondary+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    sar = isceobj.createSlcImage()
    sar.load(reference + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = Ampcor(name='reference_offset1')
    objOffset.configure()
    objOffset.setAcrossGrossOffset(rgoffset)
    objOffset.setDownGrossOffset(azoffset)
    objOffset.setWindowSizeWidth(128)
    objOffset.setWindowSizeHeight(128)
    objOffset.setSearchWindowSizeWidth(40)
    objOffset.setSearchWindowSizeHeight(40)
    margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

    nAcross = 60
    nDown = 60


    offAc = max(101,-rgoffset)+margin
    offDn = max(101,-azoffset)+margin


    lastAc = int( min(width, sim.getWidth() - offAc) - margin)
    lastDn = int( min(length, sim.getLength() - offDn) - margin)

    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(offAc)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(lastAc)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(offDn)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(lastDn)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(nAcross)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(nDown)

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    objOffset.setImageDataType1('complex')
    objOffset.setImageDataType2('complex')

    objOffset.ampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()

    result = objOffset.getOffsetField()
    return result


def estimateOffsetFieldGPU(reference, secondary, outPrefix, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between reference and secondary SLC using GPU PyCuAmpcor.
    Returned object matches Ampcor OffsetField interface used by fitOffsets().
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
    sim_width = sim.getWidth()
    sim_length = sim.getLength()

    # Keep window/search geometry consistent with CPU Ampcor path.
    ww = 128
    wh = 128
    sw = 40
    shh = 40
    margin = 2 * sw + ww
    nAcross = 60
    nDown = 60

    offAc = max(101, -int(rgoffset)) + margin
    offDn = max(101, -int(azoffset)) + margin
    lastAc = int(min(width, sim_width - offAc) - margin)
    lastDn = int(min(length, sim_length - offDn) - margin)

    if (lastAc <= offAc) or (lastDn <= offDn):
        raise ValueError(
            'Invalid GPU Ampcor ROI for refine timing: '
            'offAc={0}, lastAc={1}, offDn={2}, lastDn={3}'.format(
                offAc, lastAc, offDn, lastDn
            )
        )

    skip_across = int((lastAc - offAc) / (nAcross - 1.0))
    skip_down = int((lastDn - offDn) / (nDown - 1.0))
    if (skip_across <= 0) or (skip_down <= 0):
        raise ValueError(
            'Invalid GPU Ampcor skip spacing for refine timing: '
            'skipAcross={0}, skipDown={1}'.format(skip_across, skip_down)
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

    objOffset.windowSizeWidth = ww
    objOffset.windowSizeHeight = wh
    objOffset.halfSearchRangeAcross = sw
    objOffset.halfSearchRangeDown = shh
    objOffset.skipSampleAcross = skip_across
    objOffset.skipSampleDown = skip_down
    objOffset.corrSurfaceOverSamplingMethod = 0
    objOffset.corrSurfaceOverSamplingFactor = 16

    objOffset.referenceStartPixelDownStatic = offDn
    objOffset.referenceStartPixelAcrossStatic = offAc
    objOffset.numberWindowDown = num_down
    objOffset.numberWindowAcross = num_across

    objOffset.deviceID = 0
    objOffset.nStreams = 2
    objOffset.numberWindowDownInChunk = 1
    objOffset.numberWindowAcrossInChunk = min(64, max(1, num_across))
    objOffset.mmapSize = 16

    objOffset.offsetImageName = outPrefix + '.bip'
    objOffset.grossOffsetImageName = outPrefix + '.gross'
    objOffset.snrImageName = outPrefix + '_snr.bip'
    objOffset.covImageName = outPrefix + '_cov.bip'
    objOffset.mergeGrossOffset = 1

    logger.info(
        'GPU Ampcor config: window=(%d,%d), search=(%d,%d), '
        'skip=(%d,%d), windows=(%d,%d), start=(%d,%d)',
        ww, wh, sw, shh, skip_across, skip_down, num_across, num_down, offAc, offDn
    )

    objOffset.setupParams()
    objOffset.setConstantGrossOffset(int(rgoffset), int(azoffset))
    objOffset.checkPixelInImageRange()
    objOffset.runAmpcor()

    offset_raw = np.fromfile(objOffset.offsetImageName, dtype=np.float32)
    snr_raw = np.fromfile(objOffset.snrImageName, dtype=np.float32)
    cov_raw = np.fromfile(objOffset.covImageName, dtype=np.float32)

    expected_offset = num_down * num_across * 2
    expected_snr = num_down * num_across
    expected_cov = num_down * num_across * 3

    if offset_raw.size != expected_offset:
        raise RuntimeError(
            'GPU Ampcor offset output size mismatch: got {0}, expected {1}'.format(
                offset_raw.size, expected_offset
            )
        )

    if snr_raw.size != expected_snr:
        raise RuntimeError(
            'GPU Ampcor SNR output size mismatch: got {0}, expected {1}'.format(
                snr_raw.size, expected_snr
            )
        )

    if cov_raw.size != expected_cov:
        raise RuntimeError(
            'GPU Ampcor covariance output size mismatch: got {0}, expected {1}'.format(
                cov_raw.size, expected_cov
            )
        )

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

            # PyCuAmpcor offset BIP order: [azimuth, range]
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

    return field


def fitOffsets(field,azrgOrder=0,azazOrder=0,
        rgrgOrder=0,rgazOrder=0,snr=5.0):
    '''
    Estimate constant range and azimith shifs.
    '''


    stdWriter = create_writer("log","",True,filename='off.log')

    for distance in [10,5,3,1]:
        inpts = len(field._offsets)
        print("DEBUG %%%%%%%%")
        print(inpts)
        print("DEBUG %%%%%%%%")
        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=field)
        objOff.setSNRThreshold(snr)
        objOff.setDistance(distance)
        objOff.setStdWriter(stdWriter)

        objOff.offoutliers()

        field = objOff.getRefinedOffsetField()
        outputs = len(field._offsets)

        print('%d points left'%(len(field._offsets)))


    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)

    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


def _save_ampcor_solution(self, outShelveFile, field, fallback_cfg, azratio, rgratio, source):
    odb = shelve.open(outShelveFile)
    odb['raw_field'] = field

    shifts, cull = fitOffsets(
        field,
        azazOrder=fallback_cfg['azaz_order'],
        azrgOrder=fallback_cfg['azrg_order'],
        rgazOrder=fallback_cfg['rgaz_order'],
        rgrgOrder=fallback_cfg['rgrg_order'],
        snr=fallback_cfg['snr']
    )
    odb['cull_field'] = cull

    # Scale by PRF/range sampling ratios.
    for row in shifts[0]._coeffs:
        for ind, val in enumerate(row):
            row[ind] = val * azratio

    for row in shifts[1]._coeffs:
        for ind, val in enumerate(row):
            row[ind] = val * rgratio

    odb['azpoly'] = shifts[0]
    odb['rgpoly'] = shifts[1]
    odb['registration_source'] = source
    odb.close()

    self._insar.saveProduct(shifts[0], outShelveFile + '_az.xml')
    self._insar.saveProduct(shifts[1], outShelveFile + '_rg.xml')
    return None


def runRefineSecondaryTiming(self):

    logger.info("Running refine secondary timing")
    secondaryFrame = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    referenceSlc = referenceFrame.getImage().filename

    secondarySlc = os.path.join(self.insar.coregDirname, self._insar.coarseCoregFilename)

    rgratio = referenceFrame.instrument.getRangePixelSize() / secondaryFrame.instrument.getRangePixelSize()
    azratio = secondaryFrame.PRF / referenceFrame.PRF

    print('*************************************')
    print('rgratio, azratio: ', rgratio, azratio)
    print('*************************************')

    misregDir = self.insar.misregDirname
    os.makedirs(misregDir, exist_ok=True)

    outShelveFile = os.path.join(misregDir, self.insar.misregFilename)
    fallback_cfg = _fallback_fit_config(self)

    # Policy:
    # 1) External coregistration is OFF by default and only enabled by
    #    stripmapApp parameter useExternalCoregistration=True.
    # 2) When external coregistration is disabled:
    #      useGPU=True  -> GPU Ampcor (fallback to CPU Ampcor on failure/unavailable)
    #      useGPU=False -> CPU Ampcor
    use_gpu_flag = bool(getattr(self, 'useGPU', False))
    external_enabled = bool(getattr(self, 'useExternalCoregistration', False))

    if external_enabled:
        try:
            ext_cfg = _integrated_external_config()
            ext_gates = _integrated_external_quality_gates()
            logger.info('Running integrated external registration with config: %s', ext_cfg)
            logger.info('Integrated external registration quality gates: %s', ext_gates)
            azpoly, rgpoly, ext_meta = estimate_misregistration_polys(
                referenceSlc,
                secondarySlc,
                az_ratio=azratio,
                rg_ratio=rgratio,
                config=ext_cfg,
                logger=logger,
            )
            try:
                _validate_external_registration_quality(ext_meta, ext_gates)
                ext_meta['quality_gate'] = {
                    'passed': True,
                    'gates': ext_gates,
                    'mode': 'strict',
                }
            except ExternalRegistrationQualityError as qerr:
                ext_meta['quality_gate'] = {
                    'passed': False,
                    'gates': ext_gates,
                    'mode': 'accepted_without_ampcor_fallback',
                    'reason': str(qerr),
                    'violations': list(getattr(qerr, 'violations', [])),
                }

                if _integrated_external_keep_on_quality_failure():
                    logger.warning(
                        'Integrated external registration quality gate failed, '
                        'but keeping external polynomials (no Ampcor fallback): %s',
                        qerr,
                    )
                    if _integrated_external_force_dense_rubbersheet_on_quality_failure():
                        _enable_dense_rubbersheet(
                            self,
                            reason='external registration quality gate failure',
                        )
                    _save_external_solution(self, outShelveFile, azpoly, rgpoly, ext_meta)
                    logger.info(
                        'Integrated external registration accepted after quality-gate failure; '
                        'continuing with external misregistration polynomials.'
                    )
                    return None

                raise

            _save_external_solution(self, outShelveFile, azpoly, rgpoly, ext_meta)
            logger.info('Integrated external registration succeeded; using generated misregistration polynomials.')
            return None
        except Exception as err:
            logger.warning(
                'Integrated external registration failed (%s). Falling back to Ampcor path.',
                err,
                exc_info=True,
            )
    else:
        logger.info('Integrated external registration disabled by useExternalCoregistration=False.')

    if use_gpu_flag:
        gpu_available = _gpu_ampcor_available(self)
        if gpu_available:
            try:
                logger.info('useGPU=True: running GPU Ampcor for misregistration.')
                gpu_prefix = os.path.join(misregDir, 'gpu_ampcor_offsets')
                field = estimateOffsetFieldGPU(referenceSlc, secondarySlc, gpu_prefix)
                logger.info('GPU Ampcor fit configuration: %s', fallback_cfg)
                _save_ampcor_solution(
                    self,
                    outShelveFile,
                    field,
                    fallback_cfg,
                    azratio,
                    rgratio,
                    source='gpu_ampcor',
                )
                logger.info('GPU Ampcor succeeded; using GPU misregistration polynomials.')
                return None
            except Exception as err:
                logger.warning(
                    'GPU Ampcor failed (%s). Falling back to CPU Ampcor.',
                    err,
                    exc_info=True,
                )
        else:
            logger.warning(
                'useGPU=True, but GPU Ampcor is unavailable. Falling back to CPU Ampcor.'
            )

    logger.info('Running CPU Ampcor for misregistration.')
    field = estimateOffsetField(referenceSlc, secondarySlc)
    logger.info('CPU Ampcor fit configuration: %s', fallback_cfg)
    _save_ampcor_solution(
        self,
        outShelveFile,
        field,
        fallback_cfg,
        azratio,
        rgratio,
        source='cpu_ampcor',
    )
    return None
