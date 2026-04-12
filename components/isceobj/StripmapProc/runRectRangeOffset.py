import os
import logging
import numpy as np

import isceobj

logger = logging.getLogger('isce.insar.runRectRangeOffset')


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    sval = str(value).strip().lower()
    if sval in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if sval in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    return bool(default)


def _safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return float(default)


def _offset_valid_mask(values, nodata, invalid_low, invalid_high):
    mask = np.isfinite(values)
    if np.isfinite(nodata):
        mask &= (values != nodata)
    if np.isfinite(invalid_low):
        mask &= (values >= invalid_low)
    if np.isfinite(invalid_high):
        mask &= (values <= invalid_high)
    return mask


def _sanitize_offset_raster(path, dtype, length, width, nodata, invalid_low, invalid_high):
    arr = np.memmap(path, dtype=dtype, mode='r+', shape=(length, width))
    valid = _offset_valid_mask(arr, nodata, invalid_low, invalid_high)
    total = int(length) * int(width)
    invalid = total - int(np.count_nonzero(valid))
    if invalid > 0:
        arr[~valid] = float(nodata)
        arr.flush()
    del arr
    return invalid, total


def _infer_offset_dtype(filename, width, length):
    nelems = int(width) * int(length)
    fsize = os.path.getsize(filename)
    if fsize == nelems * np.dtype(np.float64).itemsize:
        return np.float64
    if fsize == nelems * np.dtype(np.float32).itemsize:
        return np.float32

    img = isceobj.createImage()
    img.load(filename + '.xml')
    dname = str(getattr(img, 'dataType', '')).upper()
    if dname in ('DOUBLE', 'FLOAT64'):
        return np.float64
    return np.float32


def _render_like(srcname, outname, dtype):
    out = isceobj.createImage()
    out.load(srcname + '.xml')
    out.filename = outname
    out.dataType = 'DOUBLE' if dtype == np.float64 else 'FLOAT'
    out.setAccessMode('READ')
    out.renderHdr()


def _load_affine(self, offsets_dir):
    vals = getattr(self._insar, 'radarDemAffineTransform', None)
    if isinstance(vals, (list, tuple)) and len(vals) == 6:
        try:
            return [float(v) for v in vals]
        except Exception:
            pass

    txt = os.path.join(offsets_dir, 'rdrdem_affine.txt')
    if not os.path.exists(txt):
        return None

    with open(txt, 'r') as f:
        for line in f:
            s = str(line).strip()
            if (not s) or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                return [float(parts[i]) for i in range(6)]
            except Exception:
                continue
    return None


def _is_identity_affine(affine, tol=1.0e-3):
    if affine is None:
        return True
    m11, m12, m21, m22, t1, t2 = [float(v) for v in affine]
    return (
        abs(m11 - 1.0) <= tol
        and abs(m22 - 1.0) <= tol
        and abs(m12) <= tol
        and abs(m21) <= tol
        and abs(t1) <= 1.0
        and abs(t2) <= 1.0
    )


def _warp_affine_bilinear(src, affine, nodata, invalid_low, invalid_high, chunk_rows=256):
    m11, m12, m21, m22, t1, t2 = [float(v) for v in affine]
    length, width = src.shape

    out = np.memmap(
        _warp_affine_bilinear.out_path,
        dtype=src.dtype,
        mode='w+',
        shape=(length, width),
    )

    x = np.arange(width, dtype=np.float64)[None, :]

    for r0 in range(0, length, int(max(32, chunk_rows))):
        r1 = min(length, r0 + int(max(32, chunk_rows)))
        y = np.arange(r0, r1, dtype=np.float64)[:, None]

        sx = m11 * x + m12 * y + t1
        sy = m21 * x + m22 * y + t2

        x0 = np.floor(sx).astype(np.int64)
        y0 = np.floor(sy).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        valid = (
            (x0 >= 0)
            & (x1 < width)
            & (y0 >= 0)
            & (y1 < length)
        )

        out_block = np.full((r1 - r0, width), float(nodata), dtype=np.float64)
        if np.any(valid):
            vx0 = x0[valid]
            vy0 = y0[valid]
            vx1 = x1[valid]
            vy1 = y1[valid]
            vwx = sx[valid] - vx0
            vwy = sy[valid] - vy0

            f00 = src[vy0, vx0].astype(np.float64, copy=False)
            f01 = src[vy0, vx1].astype(np.float64, copy=False)
            f10 = src[vy1, vx0].astype(np.float64, copy=False)
            f11 = src[vy1, vx1].astype(np.float64, copy=False)

            nbr_valid = (
                _offset_valid_mask(f00, nodata, invalid_low, invalid_high)
                & _offset_valid_mask(f01, nodata, invalid_low, invalid_high)
                & _offset_valid_mask(f10, nodata, invalid_low, invalid_high)
                & _offset_valid_mask(f11, nodata, invalid_low, invalid_high)
            )

            bil = (
                (1.0 - vwx) * (1.0 - vwy) * f00
                + vwx * (1.0 - vwy) * f01
                + (1.0 - vwx) * vwy * f10
                + vwx * vwy * f11
            )
            bil_out = np.full_like(bil, float(nodata), dtype=np.float64)
            bil_out[nbr_valid] = bil[nbr_valid]
            out_block[valid] = bil_out

        out[r0:r1, :] = out_block.astype(src.dtype, copy=False)

    out.flush()
    del out


def runRectRangeOffset(self):
    enabled = _parse_bool(
        os.environ.get('ISCE_ENABLE_RDRDEM_OFFSET_LOOP', None),
        default=bool(getattr(self, 'enableRdrdemOffsetLoop', False)),
    )
    if not enabled:
        logger.info('rect_rgoffset disabled. Skipping.')
        return None

    offsets_dir = self.insar.offsetsDirname
    os.makedirs(offsets_dir, exist_ok=True)

    in_rg = os.path.join(offsets_dir, self.insar.rangeOffsetFilename)
    if not os.path.exists(in_rg):
        logger.warning('range offset raster missing, skip rect_rgoffset: %s', in_rg)
        return None

    affine = _load_affine(self, offsets_dir)
    if affine is None:
        logger.warning('No radar-dem affine transform found; skip rect_rgoffset.')
        return None

    out_rg = os.path.join(offsets_dir, 'range_rect.off')

    img = isceobj.createImage()
    img.load(in_rg + '.xml')
    width = int(img.getWidth())
    length = int(img.getLength())
    dtype = _infer_offset_dtype(in_rg, width, length)

    if _is_identity_affine(affine):
        src = np.memmap(in_rg, dtype=dtype, mode='r', shape=(length, width))
        dst = np.memmap(out_rg, dtype=dtype, mode='w+', shape=(length, width))
        dst[:, :] = src[:, :]
        dst.flush()
        del dst
        del src
        _render_like(in_rg, out_rg, dtype)
        self._insar.rectRangeOffsetFilename = os.path.basename(out_rg)
        logger.info('rect_rgoffset affine ~ identity, copied range offset to %s', out_rg)
        return None

    nodata = _safe_float(os.environ.get('ISCE_GEO2RDR_OFFSET_NODATA', -999999.0), -999999.0)
    invalid_low = _safe_float(os.environ.get('ISCE_GEO2RDR_OFFSET_INVALID_LOW', -1.0e5), -1.0e5)
    invalid_high = _safe_float(os.environ.get('ISCE_GEO2RDR_OFFSET_INVALID_HIGH', 1.0e5), 1.0e5)
    src = np.memmap(in_rg, dtype=dtype, mode='r', shape=(length, width))
    _warp_affine_bilinear.out_path = out_rg
    _warp_affine_bilinear(
        src,
        affine,
        nodata=nodata,
        invalid_low=invalid_low,
        invalid_high=invalid_high,
        chunk_rows=int(max(32, _safe_float(os.environ.get('ISCE_RECT_RGOFF_CHUNK_ROWS', 256), 256))),
    )
    del src

    invalid_cnt, total_cnt = _sanitize_offset_raster(
        out_rg,
        dtype,
        length,
        width,
        nodata,
        invalid_low,
        invalid_high,
    )

    _render_like(in_rg, out_rg, dtype)
    self._insar.rectRangeOffsetFilename = os.path.basename(out_rg)
    logger.info(
        'rect_rgoffset generated %s using affine [%s]; sanitized invalid=%d/%d '
        '(nodata=%.1f, invalid_low=%.1f, invalid_high=%.1f)',
        out_rg,
        ', '.join('{:.6f}'.format(float(v)) for v in affine),
        int(invalid_cnt),
        int(total_cnt),
        float(nodata),
        float(invalid_low),
        float(invalid_high),
    )
    return None
