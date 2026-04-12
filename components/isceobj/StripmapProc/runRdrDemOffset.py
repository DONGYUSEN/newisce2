import os
import logging
import numpy as np

import isceobj
from mroipac.ampcor.Ampcor import Ampcor
from isceobj.Location.Offset import Offset
from isceobj.Location.Offset import OffsetField

logger = logging.getLogger('isce.insar.runRdrDemOffset')


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if sval in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    return bool(default)


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_bool_env(name, default=False):
    val = os.environ.get(name, None)
    if val is None:
        return bool(default)
    return _parse_bool(val, default)


def _prefer_gpu_ampcor():
    return _parse_bool_env('ISCE_PREFER_GPU_AMPCOR', True)


def _allow_cpu_ampcor_fallback():
    return _parse_bool_env('ISCE_ALLOW_CPU_AMPCOR_FALLBACK', True)


def _gpu_ampcor_available(self):
    if not bool(getattr(self, 'useGPU', True)):
        return False
    try:
        return bool(self._insar.hasGPU())
    except Exception as err:
        logger.warning('Failed to query GPU availability for rdrdem_offset: %s', str(err))
        return False


def _infer_dtype_from_xml(xml_path):
    img = isceobj.createImage()
    img.load(xml_path)
    dname = str(getattr(img, 'dataType', '')).upper()
    if dname in ('CFLOAT', 'CFLOAT32'):
        return np.complex64
    if dname in ('CDOUBLE', 'CFLOAT64'):
        return np.complex128
    if dname in ('DOUBLE', 'FLOAT64'):
        return np.float64
    return np.float32


def _render_real_like(src_xml, outname, width, length):
    out = isceobj.createImage()
    out.load(src_xml)
    out.filename = outname
    out.setWidth(int(width))
    out.setLength(int(length))
    out.bands = 1
    out.scheme = 'BIP'
    out.dataType = 'FLOAT'
    out.setAccessMode('READ')
    out.renderHdr()


def _slc_to_magnitude(slc_path, out_path, block_lines=2048):
    if slc_path.endswith('.xml'):
        slc_path = slc_path[:-4]
    xml = slc_path + '.xml'
    if not os.path.exists(slc_path):
        raise RuntimeError('SLC does not exist: {}'.format(slc_path))
    if not os.path.exists(xml):
        raise RuntimeError('SLC XML does not exist: {}'.format(xml))

    img = isceobj.createImage()
    img.load(xml)
    width = int(img.getWidth())
    length = int(img.getLength())
    dtype = _infer_dtype_from_xml(xml)

    src = np.memmap(slc_path, dtype=dtype, mode='r', shape=(length, width))
    dst = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(length, width))

    blk = max(64, int(block_lines))
    for i0 in range(0, length, blk):
        i1 = min(length, i0 + blk)
        chunk = src[i0:i1, :]
        if np.iscomplexobj(chunk):
            dst[i0:i1, :] = np.abs(chunk).astype(np.float32, copy=False)
        else:
            dst[i0:i1, :] = np.abs(chunk.astype(np.float64, copy=False)).astype(np.float32, copy=False)

    dst.flush()
    del dst
    del src
    _render_real_like(xml, out_path, width, length)
    return width, length


def _dem_to_simulated_radar(dem_path, out_path, scale=3.0, bias=100.0, block_lines=2048):
    if dem_path.endswith('.xml'):
        dem_path = dem_path[:-4]
    xml = dem_path + '.xml'
    if not os.path.exists(dem_path):
        raise RuntimeError('DEM raster does not exist: {}'.format(dem_path))
    if not os.path.exists(xml):
        raise RuntimeError('DEM XML does not exist: {}'.format(xml))

    img = isceobj.createImage()
    img.load(xml)
    width = int(img.getWidth())
    length = int(img.getLength())
    dtype = _infer_dtype_from_xml(xml)

    src = np.memmap(dem_path, dtype=dtype, mode='r', shape=(length, width))
    dst = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(length, width))

    blk = max(64, int(block_lines))
    for i0 in range(0, length, blk):
        i1 = min(length, i0 + blk)
        chunk = src[i0:i1, :].astype(np.float64, copy=False)
        grad = np.zeros((i1 - i0, width), dtype=np.float64)
        grad[:, :-1] = np.diff(chunk, axis=1)
        grad[:, -1] = grad[:, -2] if width > 1 else 0.0
        dst[i0:i1, :] = (float(scale) * grad + float(bias)).astype(np.float32, copy=False)

    dst.flush()
    del dst
    del src
    _render_real_like(xml, out_path, width, length)
    return width, length


def _decimate_real_image(in_path, out_path, rlks=1, alks=1):
    if in_path.endswith('.xml'):
        in_path = in_path[:-4]
    xml = in_path + '.xml'
    img = isceobj.createImage()
    img.load(xml)
    width = int(img.getWidth())
    length = int(img.getLength())

    dtype = _infer_dtype_from_xml(xml)
    src = np.memmap(in_path, dtype=dtype, mode='r', shape=(length, width))

    rlks = max(1, int(rlks))
    alks = max(1, int(alks))
    out_arr = np.asarray(src[::alks, ::rlks], dtype=np.float32)
    out_len, out_wid = out_arr.shape

    dst = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(out_len, out_wid))
    dst[:, :] = out_arr
    dst.flush()
    del dst
    del src

    _render_real_like(xml, out_path, out_wid, out_len)
    return out_wid, out_len


def _configure_ampcor_real(reference_real, secondary_real, gross_rg, gross_az, n_across, n_down):
    m = isceobj.createImage()
    m.load(reference_real + '.xml')
    m.setFilename(reference_real)
    m.setAccessMode('read')
    m.createImage()

    s = isceobj.createImage()
    s.load(secondary_real + '.xml')
    s.setFilename(secondary_real)
    s.setAccessMode('read')
    s.createImage()

    ampcor = Ampcor(name='csar2_rdrdem_ampcor')
    ampcor.configure()
    ampcor.setImageDataType1('real')
    ampcor.setImageDataType2('real')
    ampcor.setReferenceSlcImage(m)
    ampcor.setSecondarySlcImage(s)

    ampcor.setAcrossGrossOffset(int(gross_rg))
    ampcor.setDownGrossOffset(int(gross_az))

    ww = _safe_int(os.environ.get('ISCE_RDRDEM_AMPCOR_WINDOW', 64), 64)
    sh = _safe_int(os.environ.get('ISCE_RDRDEM_AMPCOR_SEARCH', 16), 16)
    ampcor.setWindowSizeWidth(ww)
    ampcor.setWindowSizeHeight(ww)
    ampcor.setSearchWindowSizeWidth(sh)
    ampcor.setSearchWindowSizeHeight(sh)

    first_sample = max(35, 35 - int(gross_rg))
    first_line = max(35, 35 - int(gross_az))
    ampcor.setFirstSampleAcross(first_sample)
    ampcor.setLastSampleAcross(int(m.width))
    ampcor.setNumberLocationAcross(max(6, int(n_across)))
    ampcor.setFirstSampleDown(first_line)
    ampcor.setLastSampleDown(int(m.length))
    ampcor.setNumberLocationDown(max(6, int(n_down)))

    ampcor.setAcrossLooks(1)
    ampcor.setDownLooks(1)
    ampcor.setOversamplingFactor(64)
    ampcor.setZoomWindowSize(16)
    ampcor.setDebugFlag(False)
    ampcor.setDisplayFlag(False)

    return ampcor, m, s


def _run_cpu_ampcor_real(reference_real, secondary_real, gross_rg, gross_az, n_across, n_down):
    ampcor, m, s = _configure_ampcor_real(
        reference_real=reference_real,
        secondary_real=secondary_real,
        gross_rg=gross_rg,
        gross_az=gross_az,
        n_across=n_across,
        n_down=n_down,
    )
    try:
        ampcor.ampcor()
        field = ampcor.getOffsetField()
    finally:
        m.finalizeImage()
        s.finalizeImage()
    return field


def _run_gpu_ampcor_real(reference_real, secondary_real, gross_rg, gross_az, n_across, n_down, out_prefix):
    from contrib.PyCuAmpcor import PyCuAmpcor

    ref = isceobj.createImage()
    ref.load(reference_real + '.xml')
    sec = isceobj.createImage()
    sec.load(secondary_real + '.xml')

    width = int(ref.getWidth())
    length = int(ref.getLength())
    sec_width = int(sec.getWidth())
    sec_length = int(sec.getLength())

    ww = _safe_int(os.environ.get('ISCE_RDRDEM_AMPCOR_WINDOW', 64), 64)
    sh = _safe_int(os.environ.get('ISCE_RDRDEM_AMPCOR_SEARCH', 16), 16)
    ww = max(16, int(ww))
    sh = max(4, int(sh))

    margin_across = 2 * sh + ww
    margin_down = 2 * sh + ww

    off_ac = max(35, 35 - int(gross_rg)) + margin_across
    off_dn = max(35, 35 - int(gross_az)) + margin_down
    last_ac = int(min(width, sec_width - off_ac) - margin_across)
    last_dn = int(min(length, sec_length - off_dn) - margin_down)
    if (last_ac <= off_ac) or (last_dn <= off_dn):
        raise ValueError(
            'Invalid GPU Ampcor ROI for rdrdem_offset: offAc={0}, lastAc={1}, offDn={2}, lastDn={3}'.format(
                off_ac, last_ac, off_dn, last_dn
            )
        )

    n_across = max(6, int(n_across))
    n_down = max(6, int(n_down))
    skip_across = int((last_ac - off_ac) / (n_across - 1.0))
    skip_down = int((last_dn - off_dn) / (n_down - 1.0))
    if (skip_across <= 0) or (skip_down <= 0):
        raise ValueError(
            'Invalid GPU Ampcor skip spacing for rdrdem_offset: skipAcross={0}, skipDown={1}'.format(
                skip_across, skip_down
            )
        )

    num_across = int((last_ac - off_ac) / skip_across) + 1
    num_down = int((last_dn - off_dn) / skip_down) + 1

    obj = PyCuAmpcor.PyCuAmpcor()
    obj.algorithm = 0
    obj.derampMethod = 1
    obj.referenceImageName = reference_real + '.vrt'
    obj.referenceImageHeight = int(length)
    obj.referenceImageWidth = int(width)
    obj.secondaryImageName = secondary_real + '.vrt'
    obj.secondaryImageHeight = int(sec_length)
    obj.secondaryImageWidth = int(sec_width)

    obj.windowSizeWidth = int(ww)
    obj.windowSizeHeight = int(ww)
    obj.halfSearchRangeAcross = int(sh)
    obj.halfSearchRangeDown = int(sh)
    obj.skipSampleAcross = int(skip_across)
    obj.skipSampleDown = int(skip_down)
    obj.corrSurfaceOverSamplingMethod = 0
    obj.corrSurfaceOverSamplingFactor = 16

    obj.referenceStartPixelDownStatic = int(off_dn)
    obj.referenceStartPixelAcrossStatic = int(off_ac)
    obj.numberWindowDown = int(num_down)
    obj.numberWindowAcross = int(num_across)

    obj.deviceID = max(0, _safe_int(os.environ.get('ISCE_RDRDEM_GPU_DEVICE', 0), 0))
    obj.nStreams = max(1, _safe_int(os.environ.get('ISCE_RDRDEM_GPU_STREAMS', 1), 1))
    obj.numberWindowDownInChunk = max(1, _safe_int(os.environ.get('ISCE_RDRDEM_GPU_CHUNK_DOWN', 1), 1))
    obj.numberWindowAcrossInChunk = max(
        1,
        min(
            int(num_across),
            _safe_int(os.environ.get('ISCE_RDRDEM_GPU_CHUNK_ACROSS', min(64, int(num_across))), min(64, int(num_across))),
        ),
    )
    obj.mmapSize = max(1, _safe_int(os.environ.get('ISCE_RDRDEM_GPU_MMAP_GB', 16), 16))

    obj.offsetImageName = out_prefix + '.bip'
    obj.grossOffsetImageName = out_prefix + '.gross'
    obj.snrImageName = out_prefix + '_snr.bip'
    obj.covImageName = out_prefix + '_cov.bip'
    obj.mergeGrossOffset = 1

    logger.info(
        'rdrdem_offset GPU Ampcor config: template_window=(%d,%d), search_half=(%d,%d), '
        'sampling_stride=(%d,%d), grid=(%d,%d), start=(%d,%d), chunk=(%d,%d), streams=%d, device=%d',
        ww, ww, sh, sh,
        skip_down, skip_across,
        num_down, num_across,
        off_dn, off_ac,
        obj.numberWindowDownInChunk, obj.numberWindowAcrossInChunk,
        obj.nStreams, obj.deviceID,
    )

    obj.setupParams()
    obj.setConstantGrossOffset(int(gross_rg), int(gross_az))
    obj.checkPixelInImageRange()
    obj.runAmpcor()

    off_raw = np.fromfile(obj.offsetImageName, dtype=np.float32)
    snr_raw = np.fromfile(obj.snrImageName, dtype=np.float32)
    expected_off = int(num_down) * int(num_across) * 2
    expected_snr = int(num_down) * int(num_across)
    if off_raw.size != expected_off:
        raise RuntimeError(
            'GPU Ampcor offset output size mismatch for rdrdem_offset: got {0}, expected {1}'.format(
                int(off_raw.size), int(expected_off)
            )
        )
    if snr_raw.size != expected_snr:
        raise RuntimeError(
            'GPU Ampcor snr output size mismatch for rdrdem_offset: got {0}, expected {1}'.format(
                int(snr_raw.size), int(expected_snr)
            )
        )

    off_bip = off_raw.reshape(int(num_down), int(num_across) * 2)
    snr_img = snr_raw.reshape(int(num_down), int(num_across))
    field = OffsetField()
    az_center = (int(ww) - 1) // 2
    rg_center = (int(ww) - 1) // 2

    for idn in range(int(num_down)):
        down = int(off_dn + idn * int(skip_down) + az_center)
        for iac in range(int(num_across)):
            across = int(off_ac + iac * int(skip_across) + rg_center)
            az_off = float(off_bip[idn, 2 * iac])
            rg_off = float(off_bip[idn, 2 * iac + 1])
            one = Offset()
            one.setCoordinate(across, down)
            one.setOffset(rg_off, az_off)
            one.setSignalToNoise(float(snr_img[idn, iac]))
            field.addOffset(one)
    return field


def _fit_affine(offsets, snr_threshold=2.0, max_iter=3, sigma=3.0):
    pts = []
    for one in offsets:
        x, y = one.getCoordinate()
        dx, dy = one.getOffset()
        snr = one.getSignalToNoise()
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            continue
        if (not np.isfinite(dx)) or (not np.isfinite(dy)):
            continue
        if (not np.isfinite(snr)) or (float(snr) < float(snr_threshold)):
            continue
        pts.append((float(x), float(y), float(dx), float(dy)))

    if len(pts) < 12:
        raise RuntimeError('Too few valid offsets for affine fit: {}'.format(len(pts)))

    arr = np.asarray(pts, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    tx = x + arr[:, 2]
    ty = y + arr[:, 3]

    keep = np.ones(arr.shape[0], dtype=bool)
    for _ in range(max(1, int(max_iter))):
        xx = x[keep]
        yy = y[keep]
        txx = tx[keep]
        tyy = ty[keep]
        A = np.column_stack((xx, yy, np.ones_like(xx)))

        cx, _, _, _ = np.linalg.lstsq(A, txx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, tyy, rcond=None)

        pred_x = A @ cx
        pred_y = A @ cy
        res = np.sqrt((txx - pred_x) ** 2 + (tyy - pred_y) ** 2)
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        if mad <= 1.0e-6:
            break

        th = med + float(sigma) * 1.4826 * mad
        local_keep = (res <= th)
        if int(np.count_nonzero(local_keep)) < 12:
            break

        keep_idx = np.flatnonzero(keep)
        new_keep = np.zeros_like(keep)
        new_keep[keep_idx[local_keep]] = True
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    xx = x[keep]
    yy = y[keep]
    txx = tx[keep]
    tyy = ty[keep]
    A = np.column_stack((xx, yy, np.ones_like(xx)))
    cx, _, _, _ = np.linalg.lstsq(A, txx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, tyy, rcond=None)

    m11, m12, t1 = [float(v) for v in cx]
    m21, m22, t2 = [float(v) for v in cy]
    return [m11, m12, m21, m22, t1, t2], int(np.count_nonzero(keep)), int(arr.shape[0])


def _save_affine(path, affine, n_inlier, n_total):
    with open(path, 'w') as f:
        f.write('# m11 m12 m21 m22 t1 t2 n_inlier n_total\n')
        f.write('{:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {} {}\n'.format(
            float(affine[0]), float(affine[1]), float(affine[2]), float(affine[3]),
            float(affine[4]), float(affine[5]), int(n_inlier), int(n_total)
        ))


def runRdrDemOffset(self):
    enabled = _parse_bool(
        os.environ.get('ISCE_ENABLE_RDRDEM_OFFSET_LOOP', None),
        default=bool(getattr(self, 'enableRdrdemOffsetLoop', False)),
    )
    if not enabled:
        logger.info('rdrdem_offset disabled. Skipping.')
        return None

    if (self._insar.referenceSlcCropProduct is None):
        logger.warning('referenceSlcCropProduct is missing. Skip rdrdem_offset.')
        return None

    ref_frame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    ref_slc = ref_frame.getImage().filename

    offsets_dir = self.insar.offsetsDirname
    geom_dir = self.insar.geometryDirname
    os.makedirs(offsets_dir, exist_ok=True)

    hgt_file = os.path.join(geom_dir, self.insar.heightFilename + '.full')
    if not os.path.exists(hgt_file):
        logger.warning('DEM raster not found for rdrdem_offset: %s', hgt_file)
        return None

    work_dir = os.path.join(offsets_dir, 'rdrdem_offset')
    os.makedirs(work_dir, exist_ok=True)

    amp_real = os.path.join(work_dir, 'reference_amp.float')
    sim_real = os.path.join(work_dir, 'dem_sim.float')

    _slc_to_magnitude(ref_slc, amp_real)
    _dem_to_simulated_radar(hgt_file, sim_real,
        scale=_safe_float(os.environ.get('ISCE_RDRDEM_SIM_SCALE', 3.0), 3.0),
        bias=_safe_float(os.environ.get('ISCE_RDRDEM_SIM_BIAS', 100.0), 100.0),
    )

    rlks = max(1, _safe_int(os.environ.get('ISCE_RDRDEM_MATCH_RLKS', 8), 8))
    alks = max(1, _safe_int(os.environ.get('ISCE_RDRDEM_MATCH_ALKS', 8), 8))
    amp_lk = os.path.join(work_dir, 'reference_amp_lk.float')
    sim_lk = os.path.join(work_dir, 'dem_sim_lk.float')
    lk_width, lk_length = _decimate_real_image(amp_real, amp_lk, rlks=rlks, alks=alks)
    _decimate_real_image(sim_real, sim_lk, rlks=rlks, alks=alks)

    n_across = min(40, max(8, lk_width // 512))
    n_down = min(40, max(8, lk_length // 512))

    field = None
    ampcor_engine = 'none'
    prefer_gpu = _prefer_gpu_ampcor()
    gpu_available = _gpu_ampcor_available(self)
    if prefer_gpu and gpu_available:
        try:
            field = _run_gpu_ampcor_real(
                amp_lk,
                sim_lk,
                gross_rg=1,
                gross_az=1,
                n_across=n_across,
                n_down=n_down,
                out_prefix=os.path.join(work_dir, 'rdrdem_gpu_ampcor'),
            )
            ampcor_engine = 'gpu'
            logger.info('rdrdem_offset Ampcor engine: GPU (PyCuAmpcor).')
        except Exception as err:
            if _allow_cpu_ampcor_fallback():
                logger.warning(
                    'rdrdem_offset GPU Ampcor failed (%s); falling back to CPU Ampcor.',
                    str(err),
                )
            else:
                raise
    elif prefer_gpu and (not gpu_available):
        logger.info('rdrdem_offset GPU preferred but unavailable; using CPU Ampcor.')

    if field is None:
        try:
            field = _run_cpu_ampcor_real(
                amp_lk,
                sim_lk,
                gross_rg=1,
                gross_az=1,
                n_across=n_across,
                n_down=n_down,
            )
            ampcor_engine = 'cpu'
            logger.info('rdrdem_offset Ampcor engine: CPU.')
        except Exception as err:
            logger.warning('rdrdem_offset CPU Ampcor failed (%s); fallback to identity affine.', str(err))
            field = None

    affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    n_inlier = 0
    n_total = 0
    try:
        offsets = []
        for one in list(getattr(field, '_offsets', []) or []):
            x, y = one.getCoordinate()
            dx, dy = one.getOffset()
            snr = one.getSignalToNoise()

            scaled = Offset()
            scaled.setCoordinate(float(x) * float(rlks), float(y) * float(alks))
            scaled.setOffset(float(dx) * float(rlks), float(dy) * float(alks))
            scaled.setSignalToNoise(float(snr))
            offsets.append(scaled)

        affine, n_inlier, n_total = _fit_affine(
            offsets,
            snr_threshold=_safe_float(os.environ.get('ISCE_RDRDEM_SNR_THRESHOLD', 2.0), 2.0),
            max_iter=_safe_int(os.environ.get('ISCE_RDRDEM_AFFINE_MAXITER', 3), 3),
            sigma=_safe_float(os.environ.get('ISCE_RDRDEM_AFFINE_SIGMA', 3.0), 3.0),
        )
    except Exception as err:
        logger.warning(
            'rdrdem_offset affine fit failed (%s); use identity affine and continue.',
            str(err),
        )

    self._insar.radarDemAffineTransform = affine
    self._insar.rdrdemOffsetQuality = {
        'n_inlier': int(n_inlier),
        'n_total': int(n_total),
        'looks': {'range': int(rlks), 'azimuth': int(alks)},
    }

    aff_file = os.path.join(offsets_dir, 'rdrdem_affine.txt')
    _save_affine(aff_file, affine, n_inlier, n_total)
    logger.info(
        'rdrdem_offset solved affine: [%s], inlier/total=%d/%d, engine=%s, saved=%s',
        ', '.join('{:.6f}'.format(float(v)) for v in affine),
        int(n_inlier),
        int(n_total),
        ampcor_engine,
        aff_file,
    )
    return None
