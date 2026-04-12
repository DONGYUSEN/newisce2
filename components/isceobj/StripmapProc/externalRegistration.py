#
# Integrated lightweight external registration for StripmapProc.
#

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import isceobj
from isceobj.Util.Poly2D import Poly2D


_DEFAULT_CONFIG = {
    'staged_enable': True,
    'stage1_disable': False,
    'adaptive_window_policy': True,
    'resolution_m': None,
    'secondary_prf_hz': None,
    'stage1_source': 'geo2rdr_mean',
    'stage1_require_geo2rdr': True,
    'stage1_geo2rdr_azimuth_offset': None,
    'stage1_geo2rdr_range_offset': None,
    'stage1_geo2rdr_nodata': -999999.0,
    'stage1_geo2rdr_invalid_low': -1.0e5,
    'use_geo2rdr_valid_mask': True,
    'resample_mode': 'geo2rdr_plus_misreg',
    'stage1_window': 2048,
    'stage1_search_half': 1024,
    'stage1_grid_size': 4,
    'stage1_quality_threshold': 18.0,
    'stage1_min_valid': 4,
    'stage1_outlier_sigma': 2.5,
    'stage1_outlier_max_iterations': 8,
    'stage2_windows': [1024, 512, 256, 128],
    'stage2_search_half_scale': 0.5,
    'stage2_search_half_scales': [],
    'stage2_search_half_min': 1,
    'stage2_search_half_max': 16384,
    'stage2_grid_size': 4,
    'stage2_quality_threshold': 18.0,
    'stage2_min_valid': 4,
    'stage2_prefer_larger_window': True,
    'stage2_prefer_smaller_when_close': True,
    'stage2_valid_margin_for_smaller_window': 1,
    'stage2_spread_margin_for_smaller_window': 2.0,
    'stage2_prefer_smaller_search_when_close': True,
    'stage2_valid_margin_for_smaller_search': 1,
    'stage2_spread_margin_for_smaller_search': 1.0,
    'stage2_quality_margin_for_smaller_search': 1.0,
    'coarse_window': 256,
    'coarse_multiscale': True,
    'coarse_window_factors': [0.5, 1.0, 2.0],
    'coarse_search_ranges': [32, 64, 128, 256, 512, 1024],
    'coarse_window_scale': 4.0,
    'coarse_consistency_priority': True,
    'coarse_correlation_threshold': 18.0,
    'coarse_min_window': 96,
    'coarse_max_window': 4096,
    'coarse_grid_size': 3,
    'coarse_min_valid': 9,
    'coarse_prefer_larger_window': True,
    'coarse_log_candidates': True,
    'coarse_quality_threshold': 18.0,
    'coarse_auto_efficiency_balance': True,
    'coarse_quality_margin_for_smaller_window': 3.0,
    'coarse_spread_margin_for_smaller_window': 2.0,
    'coarse_pass_zero_offset_to_fine': False,
    'fine_large_coarse_threshold': 1024,
    'fine_window_cap_for_large_coarse': 1024,
    'fine_large_coarse_grid_size': 60,
    'fine_window': 128,
    'fine_force_search_half_from_window': True,
    'fine_force_search_half_to_half_window': True,
    'fine_force_search_half_scale': 0.25,
    'fine_search_scale': 0.25,
    'fine_search_min': 16,
    'fine_search_max': 128,
    'fine_spacing': 128,
    'fine_quality_threshold': 18.0,
    'offset_filter_method': 'modified_zscore',
    'offset_filter_iqr_coeff': 1.5,
    'offset_filter_z_threshold': 3.0,
    'offset_filter_modified_z_threshold': 3.29,
    'fine_workers': 0,
    'fine_chunk_size': 128,
    'fine_max_az_points': 60,
    'fine_max_rg_points': 30,
    'fine_max_total_points': 60 * 30,
    'precompute_amplitude': True,
    'max_points': 60 * 30,
    'max_points_cap': 60 * 30,
    'max_iterations': 8,
    'sigma_threshold': 2.5,
    'min_points': 36,
}


class _AmplitudeAccessor(object):
    """
    Access amplitude patches from a complex SLC source.
    Precompute full-image amplitude when requested/possible.
    """

    def __init__(self, complex_data, precompute=True, logger=None, label=''):
        self._complex = complex_data
        self._amp = None
        self._label = label
        self._logger = logger

        if bool(precompute):
            try:
                self._amp = np.abs(complex_data).astype(np.float32, copy=False)
                if self._logger is not None:
                    self._logger.info(
                        'External registration amplitude precompute enabled for %s, shape=%s',
                        self._label,
                        str(self._amp.shape),
                    )
            except MemoryError:
                self._amp = None
                if self._logger is not None:
                    self._logger.warning(
                        'External registration amplitude precompute OOM for %s; fallback to on-demand patches.',
                        self._label,
                    )

    @property
    def shape(self):
        if self._amp is not None:
            return self._amp.shape
        return self._complex.shape

    def extract_patch(self, center_row, center_col, window):
        if self._amp is not None:
            return _extract_patch(self._amp, center_row, center_col, window)

        patch = _extract_patch(self._complex, center_row, center_col, window)
        if patch is None:
            return None
        return np.abs(patch).astype(np.float32, copy=False)

    def extract_search_chip(
        self,
        center_row,
        center_col,
        template_window,
        search_half,
        row_shift=0.0,
        col_shift=0.0,
    ):
        search_shape = _full_search_shape(template_window, search_half)
        return self.extract_patch(
            center_row + float(row_shift),
            center_col + float(col_shift),
            search_shape,
        )


def _normalize_slc_path(path):
    if path.endswith('.xml'):
        return path[:-4]
    return path


def _normalize_image_path(path):
    if path is None:
        return None
    p = str(path).strip()
    if not p:
        return None
    if p.endswith('.xml'):
        return p[:-4]
    return p


def _slc_dtype(dtype_name):
    if not dtype_name:
        return np.complex64
    key = str(dtype_name).strip().upper()
    mapping = {
        'CFLOAT': np.complex64,
        'CFLOAT32': np.complex64,
        'CDOUBLE': np.complex128,
        'CFLOAT64': np.complex128,
    }
    return mapping.get(key, np.complex64)


def _offset_dtype(dtype_name):
    if not dtype_name:
        return np.float32
    key = str(dtype_name).strip().upper()
    mapping = {
        'DOUBLE': np.float64,
        'FLOAT': np.float32,
        'REAL': np.float32,
    }
    return mapping.get(key, np.float32)


def _read_slc_as_memmap(path):
    slc_path = _normalize_slc_path(path)
    xml_path = slc_path + '.xml'

    if not os.path.exists(slc_path):
        raise RuntimeError('SLC file does not exist: {0}'.format(slc_path))
    if not os.path.exists(xml_path):
        raise RuntimeError('SLC XML does not exist: {0}'.format(xml_path))

    img = isceobj.createSlcImage()
    img.load(xml_path)
    width = int(img.getWidth())
    length = int(img.getLength())
    dtype = _slc_dtype(getattr(img, 'dataType', None))
    data = np.memmap(slc_path, dtype=dtype, mode='r', shape=(length, width))

    if np.iscomplexobj(data):
        return data
    return np.asarray(data, dtype=np.float32).astype(np.complex64)


def _read_real_image_as_memmap(path):
    img_path = _normalize_image_path(path)
    if img_path is None:
        raise RuntimeError('Image path is not configured.')
    xml_path = img_path + '.xml'

    if not os.path.exists(img_path):
        raise RuntimeError('Image file does not exist: {0}'.format(img_path))
    if not os.path.exists(xml_path):
        raise RuntimeError('Image XML does not exist: {0}'.format(xml_path))

    img = isceobj.createImage()
    img.load(xml_path)
    width = int(img.getWidth())
    length = int(img.getLength())
    dtype = _offset_dtype(getattr(img, 'dataType', None))
    data = np.memmap(img_path, dtype=dtype, mode='r', shape=(length, width))
    return data, width, length, dtype


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _geo2rdr_invalid_low(cfg):
    fallback = _safe_float(os.environ.get('ISCE_GEO2RDR_OFFSET_INVALID_LOW', -1.0e5), -1.0e5)
    return _safe_float(cfg.get('stage1_geo2rdr_invalid_low', fallback), fallback)


def _geo2rdr_valid_mask_arrays(az, rg, nodata, invalid_low):
    valid = np.isfinite(az) & np.isfinite(rg)
    if np.isfinite(nodata):
        valid &= (az != nodata) & (rg != nodata)
    if np.isfinite(invalid_low):
        valid &= (az >= invalid_low) & (rg >= invalid_low)
    return valid


def _geo2rdr_valid_scalar(azv, rgv, nodata, invalid_low):
    if (not np.isfinite(azv)) or (not np.isfinite(rgv)):
        return False
    if np.isfinite(nodata) and ((azv == nodata) or (rgv == nodata)):
        return False
    if np.isfinite(invalid_low) and ((azv < invalid_low) or (rgv < invalid_low)):
        return False
    return True


def _build_geo2rdr_sampling_guard(cfg, target_shape, logger=None):
    if not bool(cfg.get('use_geo2rdr_valid_mask', True)):
        return None

    az_path = _normalize_image_path(cfg.get('stage1_geo2rdr_azimuth_offset'))
    rg_path = _normalize_image_path(cfg.get('stage1_geo2rdr_range_offset'))
    if (az_path is None) or (rg_path is None):
        if logger is not None:
            logger.warning(
                'External registration geo2rdr valid-mask guard skipped: offset paths are missing.'
            )
        return None

    try:
        az, az_w, az_l, _ = _read_real_image_as_memmap(az_path)
        rg, rg_w, rg_l, _ = _read_real_image_as_memmap(rg_path)
    except Exception as err:
        if logger is not None:
            logger.warning(
                'External registration geo2rdr valid-mask guard skipped: cannot load offsets (%s).',
                str(err),
            )
        return None

    if (az_w != rg_w) or (az_l != rg_l):
        if logger is not None:
            logger.warning(
                'External registration geo2rdr valid-mask guard skipped: shape mismatch az=%dx%d rg=%dx%d.',
                int(az_l),
                int(az_w),
                int(rg_l),
                int(rg_w),
            )
        return None

    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    guard_h = min(target_h, int(az_l))
    guard_w = min(target_w, int(az_w))
    if (guard_h <= 0) or (guard_w <= 0):
        return None

    invalid_low = _geo2rdr_invalid_low(cfg)
    if logger is not None:
        logger.info(
            'External registration geo2rdr valid-mask guard enabled: shape=%dx%d, '
            'nodata=%.1f, invalid_low=%.1f.',
            int(guard_h),
            int(guard_w),
            float(cfg.get('stage1_geo2rdr_nodata', -999999.0)),
            float(invalid_low),
        )

    return {
        'az': az,
        'rg': rg,
        'height': int(guard_h),
        'width': int(guard_w),
        'nodata': float(cfg.get('stage1_geo2rdr_nodata', -999999.0)),
        'invalid_low': float(invalid_low),
        'az_path': str(az_path),
        'rg_path': str(rg_path),
    }


def _geo2rdr_guard_valid(guard, row, col):
    if guard is None:
        return True

    rr = int(np.rint(float(row)))
    cc = int(np.rint(float(col)))
    if (rr < 0) or (cc < 0) or (rr >= int(guard['height'])) or (cc >= int(guard['width'])):
        return False

    azv = float(guard['az'][rr, cc])
    rgv = float(guard['rg'][rr, cc])
    nodata = float(guard['nodata'])
    invalid_low = _safe_float(guard.get('invalid_low', -1.0e5), -1.0e5)
    return _geo2rdr_valid_scalar(azv, rgv, nodata, invalid_low)


def _coerce_hw_pair(value, default=None):
    if value is None:
        if default is None:
            raise ValueError('Missing shape value')
        value = default

    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    if len(items) == 0:
        raise ValueError('Empty shape value')
    if len(items) == 1:
        one = int(np.rint(float(items[0])))
        return one, one

    first = int(np.rint(float(items[0])))
    second = int(np.rint(float(items[1])))
    return first, second


def _positive_hw_pair(value, default=None, minimum=1):
    hh, ww = _coerce_hw_pair(value, default=default)
    hh = max(int(minimum), int(hh))
    ww = max(int(minimum), int(ww))
    return hh, ww


def _full_search_shape(template_window, search_half):
    win_h, win_w = _positive_hw_pair(template_window, minimum=1)
    srch_h, srch_w = _positive_hw_pair(search_half, minimum=0)
    return (int(win_h + 2 * srch_h), int(win_w + 2 * srch_w))


def _result_hw_pair(result, pair_key, scalar_key):
    pair = result.get(pair_key) or {}
    if isinstance(pair, dict):
        return (
            int(pair.get('down', result.get(scalar_key, 0))),
            int(pair.get('across', result.get(scalar_key, 0))),
        )
    return _positive_hw_pair(result.get(scalar_key, 0), minimum=0)


def _extract_patch(arr, center_row, center_col, window):
    win_h, win_w = _positive_hw_pair(window, minimum=1)
    half_h = int(win_h // 2)
    half_w = int(win_w // 2)
    crow = int(np.rint(center_row))
    ccol = int(np.rint(center_col))
    r0 = crow - half_h
    c0 = ccol - half_w
    r1 = r0 + int(win_h)
    c1 = c0 + int(win_w)

    if r0 < 0 or c0 < 0 or r1 > arr.shape[0] or c1 > arr.shape[1]:
        return None

    return arr[r0:r1, c0:c1]


def _parabolic_subpixel(values, idx):
    n = len(values)
    left = float(values[(idx - 1) % n])
    center = float(values[idx])
    right = float(values[(idx + 1) % n])
    denom = left - 2.0 * center + right
    if abs(denom) < 1.0e-12:
        return 0.0
    return 0.5 * (left - right) / denom


def _parabolic_subpixel_clamped(values, idx):
    if idx <= 0 or idx >= (len(values) - 1):
        return 0.0
    left = float(values[idx - 1])
    center = float(values[idx])
    right = float(values[idx + 1])
    denom = left - 2.0 * center + right
    if abs(denom) < 1.0e-12:
        return 0.0
    return 0.5 * (left - right) / denom


def _normalized_corr_centered(a, b):
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 1.0e-12:
        return 0.0
    # GMTSAR-style correlation/SNR score in [0, 100].
    return float(100.0 * np.abs(np.sum(a * b)) / denom)


def _phase_correlation_displacement(master_amp_patch, slave_amp_patch):
    ma = master_amp_patch.astype(np.float64, copy=False)
    sa = slave_amp_patch.astype(np.float64, copy=False)
    ma -= np.mean(ma)
    sa -= np.mean(sa)

    if (np.std(ma) < 1.0e-10) or (np.std(sa) < 1.0e-10):
        return None

    fma = np.fft.fft2(ma)
    fsa = np.fft.fft2(sa)
    cps = fma * np.conj(fsa)
    mag = np.abs(cps)
    cps /= np.where(mag > 1.0e-12, mag, 1.0)

    corr = np.abs(np.fft.ifft2(cps))
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape

    dy = float(iy)
    dx = float(ix)
    if dy > (h // 2):
        dy -= h
    if dx > (w // 2):
        dx -= w

    dy += _parabolic_subpixel(corr[:, ix], iy)
    dx += _parabolic_subpixel(corr[iy, :], ix)

    # dy/dx above are shifts to apply on slave patch to align master.
    # We store displacement of slave relative to master, so use negative sign.
    disp_az = -dy
    disp_rg = -dx

    iy_int = int(np.rint(dy))
    ix_int = int(np.rint(dx))
    shifted_slave = np.roll(np.roll(sa, iy_int, axis=0), ix_int, axis=1)
    quality = _normalized_corr_centered(ma, shifted_slave)

    return disp_az, disp_rg, quality


def _valid_window_sums(arr, win_h, win_w):
    integral = np.pad(arr, ((1, 0), (1, 0)), mode='constant')
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    return (
        integral[win_h:, win_w:]
        - integral[:-win_h, win_w:]
        - integral[win_h:, :-win_w]
        + integral[:-win_h, :-win_w]
    )


def _valid_cross_correlation(search_chip, template_centered):
    hh, ww = search_chip.shape
    th, tw = template_centered.shape
    fft_shape = (hh + th - 1, ww + tw - 1)
    corr_full = np.fft.irfft2(
        np.fft.rfft2(search_chip, s=fft_shape)
        * np.fft.rfft2(template_centered[::-1, ::-1], s=fft_shape),
        s=fft_shape,
    )
    return corr_full[th - 1:hh, tw - 1:ww]


def _search_patch_displacement(master_amp_patch, slave_search_chip, search_half):
    search_h, search_w = _positive_hw_pair(search_half, minimum=0)
    tmpl = np.asarray(master_amp_patch, dtype=np.float64)
    chip = np.asarray(slave_search_chip, dtype=np.float64)

    tmpl_h, tmpl_w = tmpl.shape
    expected_shape = _full_search_shape((tmpl_h, tmpl_w), (search_h, search_w))
    if chip.shape != expected_shape:
        return None

    tmpl_centered = tmpl - np.mean(tmpl)
    tmpl_energy = float(np.sum(tmpl_centered * tmpl_centered))
    if tmpl_energy <= 1.0e-12:
        return None

    numerator = _valid_cross_correlation(chip, tmpl_centered)
    chip_sum = _valid_window_sums(chip, tmpl_h, tmpl_w)
    chip_sum_sq = _valid_window_sums(chip * chip, tmpl_h, tmpl_w)
    npix = float(tmpl_h * tmpl_w)
    chip_var = chip_sum_sq - (chip_sum * chip_sum) / npix
    valid = np.isfinite(chip_var) & (chip_var > 1.0e-12)
    if not np.any(valid):
        return None

    corr_surface = np.full(numerator.shape, -np.inf, dtype=np.float64)
    corr_surface[valid] = numerator[valid] / np.sqrt(tmpl_energy * chip_var[valid])
    if not np.isfinite(np.max(corr_surface)):
        return None

    peak_row, peak_col = np.unravel_index(np.argmax(corr_surface), corr_surface.shape)
    peak_quality = float(100.0 * corr_surface[peak_row, peak_col])
    if not np.isfinite(peak_quality):
        return None

    sub_row = _parabolic_subpixel_clamped(corr_surface[:, peak_col], int(peak_row))
    sub_col = _parabolic_subpixel_clamped(corr_surface[peak_row, :], int(peak_col))

    disp_az = (float(peak_row) + float(sub_row)) - float(search_h)
    disp_rg = (float(peak_col) + float(sub_col)) - float(search_w)
    return float(disp_az), float(disp_rg), peak_quality


def _template_search_displacement(
    master_amp,
    slave_amp,
    row,
    col,
    template_window,
    search_half,
    gross_az=0.0,
    gross_rg=0.0,
):
    pm = master_amp.extract_patch(row, col, template_window)
    ps = slave_amp.extract_search_chip(
        row,
        col,
        template_window,
        search_half,
        row_shift=gross_az,
        col_shift=gross_rg,
    )
    if (pm is None) or (ps is None):
        return None

    residual = _search_patch_displacement(pm, ps, search_half)
    if residual is None:
        return None

    res_az, res_rg, quality = residual
    return (
        float(gross_az) + float(res_az),
        float(gross_rg) + float(res_rg),
        float(quality),
    )


def _sanitize_window_size(window, h, w, cfg):
    win = int(np.rint(window))
    if win % 2 != 0:
        win += 1

    min_win = max(32, int(cfg.get('coarse_min_window', 96)))
    max_win_cfg = max(min_win, int(cfg.get('coarse_max_window', 4096)))
    # Keep corners valid at 1/4 and 3/4 image positions.
    max_by_shape = max(32, (min(h, w) // 2) - 2)
    max_win = min(max_win_cfg, max_by_shape)
    if max_win < min_win:
        min_win = max_win
    win = max(min_win, min(win, max_win))
    if win % 2 != 0:
        win = max(min_win, win - 1)
    return int(win)


def _coarse_correlation_threshold(cfg):
    val = cfg.get('coarse_correlation_threshold', None)
    if val is None:
        return float(cfg.get('coarse_quality_threshold', 18.0))
    try:
        return float(val)
    except Exception:
        return float(cfg.get('coarse_quality_threshold', 18.0))


def _coarse_candidate_search_ranges(master_amp, cfg):
    h, w = master_amp.shape
    scale = float(cfg.get('coarse_window_scale', 4.0))
    if scale <= 0.0:
        scale = 4.0

    ranges = []
    raw_ranges = cfg.get('coarse_search_ranges', [])
    if isinstance(raw_ranges, (list, tuple, np.ndarray)):
        items = list(raw_ranges)
    else:
        items = [raw_ranges]

    for one in items:
        try:
            ival = int(np.rint(float(one)))
        except Exception:
            continue
        if ival > 0:
            ranges.append(ival)

    if not ranges:
        # Backward compatibility: derive search ranges from legacy window-factor settings.
        base = int(cfg.get('coarse_window', 256))
        use_multiscale = bool(cfg.get('coarse_multiscale', True))
        factors = cfg.get('coarse_window_factors', [1.0])
        if not use_multiscale:
            factors = [1.0]
        for factor in factors:
            try:
                fval = float(factor)
            except Exception:
                continue
            if fval <= 0.0:
                continue
            win = _sanitize_window_size(base * fval, h, w, cfg)
            sr = max(8, int(np.rint(float(win) / scale)))
            ranges.append(sr)

    if not ranges:
        ranges = [64, 128, 256, 512, 1024]

    uniq = []
    seen = set()
    for one in sorted(ranges):
        if one in seen:
            continue
        seen.add(one)
        uniq.append(int(one))
    return uniq


def _coarse_candidate_windows(master_amp, cfg):
    h, w = master_amp.shape
    scale = float(cfg.get('coarse_window_scale', 4.0))
    if scale <= 0.0:
        scale = 4.0

    search_ranges = _coarse_candidate_search_ranges(master_amp, cfg)
    candidates = []
    seen = set()
    for requested_sr in search_ranges:
        sr = max(8, int(requested_sr))
        win = _sanitize_window_size(scale * sr, h, w, cfg)
        effective_sr = max(8, int(np.rint(float(win) / scale)))
        key = (int(win), int(effective_sr))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                'window': int(win),
                'search_range': int(effective_sr),
                'requested_search_range': int(sr),
                'window_scale': float(scale),
            }
        )

    if not candidates:
        win = _sanitize_window_size(256, h, w, cfg)
        candidates = [
            {
                'window': int(win),
                'search_range': max(8, int(np.rint(float(win) / scale))),
                'requested_search_range': max(8, int(np.rint(float(win) / scale))),
                'window_scale': float(scale),
            }
        ]

    return candidates


def _coarse_registration_for_window(master_amp, slave_amp, window, search_range, cfg):
    h, w = master_amp.shape
    template_hw = _positive_hw_pair(window, minimum=32)
    search_hw = _positive_hw_pair(search_range, minimum=0)

    grid_size = max(3, int(cfg.get('coarse_grid_size', 3)))
    fracs = np.linspace(0.25, 0.75, grid_size, dtype=np.float64)
    positions = [(h * fr, w * fc) for fr in fracs for fc in fracs]

    az = []
    rg = []
    qualities = []
    quality_threshold = float(cfg['coarse_quality_threshold'])
    for row, col in positions:
        result = _template_search_displacement(
            master_amp,
            slave_amp,
            row,
            col,
            template_hw,
            search_hw,
        )
        if result is None:
            continue
        daz, drg, q = result
        if q < quality_threshold:
            continue
        if (abs(float(daz)) > float(search_hw[0])) or (abs(float(drg)) > float(search_hw[1])):
            continue
        az.append(daz)
        rg.append(drg)
        qualities.append(q)

    # Enforce at least 3x3 valid coarse points by default.
    required_valid = min(len(positions), max(9, int(cfg.get('coarse_min_valid', 9))))
    if len(az) < required_valid:
        return {
            'status': 'failed',
            'window': int(window),
            'search_range': int(search_range),
            'template_window': {'down': int(template_hw[0]), 'across': int(template_hw[1])},
            'search_half': {'down': int(search_hw[0]), 'across': int(search_hw[1])},
            'num_valid': int(len(az)),
            'error': 'valid windows < {0}'.format(required_valid),
        }

    az_arr = np.array(az, dtype=np.float64)
    rg_arr = np.array(rg, dtype=np.float64)
    q_arr = np.array(qualities, dtype=np.float64)
    spread = float(np.std(az_arr) + np.std(rg_arr))
    quality_median = float(np.median(q_arr))

    return {
        'status': 'ok',
        'window': int(window),
        'search_range': int(search_range),
        'template_window': {'down': int(template_hw[0]), 'across': int(template_hw[1])},
        'search_half': {'down': int(search_hw[0]), 'across': int(search_hw[1])},
        'num_valid': int(len(az)),
        'azimuth_offset': float(np.median(az_arr)),
        'range_offset': float(np.median(rg_arr)),
        'qualities': [float(v) for v in qualities],
        'spread': spread,
        'quality_median': quality_median,
    }


def _coarse_registration(master_amp, slave_amp, cfg, logger=None):
    candidates = _coarse_candidate_windows(master_amp, cfg)
    results = []
    for cand in candidates:
        res = _coarse_registration_for_window(
            master_amp,
            slave_amp,
            cand['window'],
            cand['search_range'],
            cfg,
        )
        res['window_scale'] = float(cand.get('window_scale', cfg.get('coarse_window_scale', 4.0)))
        res['requested_search_range'] = int(cand.get('requested_search_range', cand['search_range']))
        results.append(res)

    if (logger is not None) and bool(cfg.get('coarse_log_candidates', True)):
        for r in results:
            template_h, template_w = _result_hw_pair(r, 'template_window', 'window')
            search_h, search_w = _result_hw_pair(r, 'search_half', 'search_range')
            if r.get('status') == 'ok':
                logger.info(
                    'External coarse candidate: template=(%d,%d) search_half=(%d,%d) valid=%d quality_median=%.5f spread=%.6f',
                    template_h,
                    template_w,
                    search_h,
                    search_w,
                    int(r.get('num_valid', 0)),
                    float(r.get('quality_median', 0.0)),
                    float(r.get('spread', 0.0)),
                )
            else:
                logger.info(
                    'External coarse candidate: template=(%d,%d) search_half=(%d,%d) failed (%s)',
                    template_h,
                    template_w,
                    search_h,
                    search_w,
                    str(r.get('error', 'unknown')),
                )

    valid = [r for r in results if r.get('status') == 'ok']
    if not valid:
        raise RuntimeError(
            'Integrated external coarse registration failed for all windows: {0}'.format(
                [r.get('window') for r in results]
            )
        )

    correlation_threshold = _coarse_correlation_threshold(cfg)
    qualified = [
        r for r in valid
        if float(r.get('quality_median', 0.0)) >= correlation_threshold
    ]
    consistency_priority = bool(cfg.get('coarse_consistency_priority', True))
    prefer_larger = bool(cfg.get('coarse_prefer_larger_window', True))

    if qualified:
        if consistency_priority:
            # Priority requested: valid count first, then quality, then spread.
            best = max(
                qualified,
                key=lambda r: (
                    int(r.get('num_valid', 0)),
                    float(r.get('quality_median', 0.0)),
                    -float(r.get('spread', np.inf)),
                    int(r.get('window', 0)) if prefer_larger else 0,
                ),
            )
            selection_mode = 'consistency_first'

            if bool(cfg.get('coarse_auto_efficiency_balance', True)):
                quality_margin = max(
                    0.0,
                    float(cfg.get('coarse_quality_margin_for_smaller_window', 0.03)),
                )
                spread_margin = max(
                    0.0,
                    float(cfg.get('coarse_spread_margin_for_smaller_window', 2.0)),
                )

                best_window = int(best.get('window', 0))
                best_quality = float(best.get('quality_median', 0.0))
                best_spread = float(best.get('spread', np.inf))

                # When quality gain is limited, prefer a smaller coarse window
                # to reduce runtime while keeping fit reliability.
                smaller_candidates = [
                    r for r in qualified
                    if int(r.get('window', 0)) < best_window
                    and (best_quality - float(r.get('quality_median', 0.0))) <= quality_margin
                    and (float(r.get('spread', np.inf)) - best_spread) <= spread_margin
                ]

                if smaller_candidates:
                    balanced = max(
                        smaller_candidates,
                        key=lambda r: (
                            int(r.get('num_valid', 0)),
                            -int(r.get('window', 0)),
                            float(r.get('quality_median', 0.0)),
                            -float(r.get('spread', np.inf)),
                        ),
                    )
                    if logger is not None:
                        best_template_h, best_template_w = _result_hw_pair(best, 'template_window', 'window')
                        best_search_h, best_search_w = _result_hw_pair(best, 'search_half', 'search_range')
                        balanced_template_h, balanced_template_w = _result_hw_pair(
                            balanced,
                            'template_window',
                            'window',
                        )
                        balanced_search_h, balanced_search_w = _result_hw_pair(
                            balanced,
                            'search_half',
                            'search_range',
                        )
                        logger.info(
                            'External coarse auto-balance switched to smaller template/search pair: '
                            'from template=(%d,%d) search_half=(%d,%d) quality=%.5f spread=%.6f '
                            'to template=(%d,%d) search_half=(%d,%d) quality=%.5f spread=%.6f '
                            '(quality_margin=%.5f, spread_margin=%.5f)',
                            best_template_h,
                            best_template_w,
                            best_search_h,
                            best_search_w,
                            float(best.get('quality_median', 0.0)),
                            float(best.get('spread', 0.0)),
                            balanced_template_h,
                            balanced_template_w,
                            balanced_search_h,
                            balanced_search_w,
                            float(balanced.get('quality_median', 0.0)),
                            float(balanced.get('spread', 0.0)),
                            quality_margin,
                            spread_margin,
                        )
                    best = balanced
                    selection_mode = 'consistency_first_auto_balance'
        else:
            best = max(
                qualified,
                key=lambda r: (
                    int(r.get('num_valid', 0)),
                    float(r.get('quality_median', 0.0)),
                    -float(r.get('spread', np.inf)),
                    int(r.get('window', 0)) if prefer_larger else 0,
                ),
            )
            selection_mode = 'legacy_rank'
    else:
        # Fallback: if no candidate passes correlation threshold, keep best-correlation candidate.
        best = max(
            valid,
            key=lambda r: (
                float(r.get('quality_median', 0.0)),
                int(r.get('num_valid', 0)),
                -float(r.get('spread', np.inf)),
                int(r.get('window', 0)) if prefer_larger else 0,
            ),
        )
        selection_mode = 'quality_fallback'

    if (logger is not None) and bool(cfg.get('coarse_log_candidates', True)):
        best_template_h, best_template_w = _result_hw_pair(best, 'template_window', 'window')
        best_search_h, best_search_w = _result_hw_pair(best, 'search_half', 'search_range')
        logger.info(
            'External coarse selected: mode=%s template=(%d,%d) search_half=(%d,%d) valid=%d quality_median=%.5f spread=%.6f az=%.4f rg=%.4f',
            selection_mode,
            best_template_h,
            best_template_w,
            best_search_h,
            best_search_w,
            int(best.get('num_valid', 0)),
            float(best.get('quality_median', 0.0)),
            float(best.get('spread', 0.0)),
            float(best.get('azimuth_offset', 0.0)),
            float(best.get('range_offset', 0.0)),
        )

    summary = []
    for r in results:
        summary.append(
            {
                'search_range': int(r.get('search_range', 0)),
                'requested_search_range': int(r.get('requested_search_range', r.get('search_range', 0))),
                'window': int(r.get('window', 0)),
                'template_window': dict(r.get('template_window') or {}),
                'search_half': dict(r.get('search_half') or {}),
                'window_scale': float(r.get('window_scale', cfg.get('coarse_window_scale', 4.0))),
                'status': str(r.get('status', 'failed')),
                'num_valid': int(r.get('num_valid', 0)),
                'score': [
                    float(r.get('spread', np.inf)) if r.get('status') == 'ok' else None,
                    int(r.get('num_valid', 0)),
                    float(r.get('quality_median', 0.0)) if r.get('status') == 'ok' else None,
                ],
                'quality_median': float(r.get('quality_median', 0.0))
                if r.get('status') == 'ok'
                else None,
                'spread': float(r.get('spread', 0.0))
                if r.get('status') == 'ok'
                else None,
            }
        )

    best_az = float(best['azimuth_offset'])
    best_rg = float(best['range_offset'])
    pass_zero_offset = bool(cfg.get('coarse_pass_zero_offset_to_fine', False))
    coarse_for_fine_az = 0.0 if pass_zero_offset else best_az
    coarse_for_fine_rg = 0.0 if pass_zero_offset else best_rg
    if (logger is not None) and pass_zero_offset:
        logger.info(
            'External coarse-to-fine bridge: forcing coarse offsets to zero for fine stage '
            '(estimated coarse az=%.4f rg=%.4f kept in diagnostics only).',
            best_az,
            best_rg,
        )

    return {
        'azimuth_offset': float(coarse_for_fine_az),
        'range_offset': float(coarse_for_fine_rg),
        'estimated_azimuth_offset': float(best_az),
        'estimated_range_offset': float(best_rg),
        'initial_offset': (
            float(coarse_for_fine_az),
            float(coarse_for_fine_rg),
        ),
        'search_range': int(best.get('search_range', 0)),
        'requested_search_range': int(best.get('requested_search_range', best.get('search_range', 0))),
        'template_window': dict(best.get('template_window') or {}),
        'search_half': dict(best.get('search_half') or {}),
        'qualities': [float(v) for v in best['qualities']],
        'window_size': int(best['window']),
        'window_scale': float(best.get('window_scale', cfg.get('coarse_window_scale', 4.0))),
        'num_valid': int(best['num_valid']),
        'quality_median': float(best.get('quality_median', 0.0)),
        'spread': float(best.get('spread', 0.0)),
        'selection_mode': selection_mode,
        'correlation_threshold': float(correlation_threshold),
        'candidate_summary': summary,
    }


def _iter_grid_points(height, width, spacing, margin):
    r = int(margin)
    while r < (height - int(margin)):
        c = int(margin)
        while c < (width - int(margin)):
            yield r, c
            c += int(spacing)
        r += int(spacing)


def _iter_uniform_grid_points(height, width, margin, rows_count, cols_count):
    r0 = float(margin)
    c0 = float(margin)
    r1 = float(height - int(margin) - 1)
    c1 = float(width - int(margin) - 1)
    if (r1 <= r0) or (c1 <= c0):
        return []

    nr = max(1, int(rows_count))
    nc = max(1, int(cols_count))
    rows = np.linspace(r0, r1, nr, dtype=np.float64)
    cols = np.linspace(c0, c1, nc, dtype=np.float64)
    return [(float(r), float(c)) for r in rows for c in cols]


def _resolve_window_and_search(window, search_half, height, width, min_window=32, min_search=4):
    win = int(np.rint(float(window)))
    if win % 2 != 0:
        win += 1
    win = max(int(min_window), win)

    sh, sw = _positive_hw_pair(search_half, minimum=int(min_search))
    shape_limit = max(2 * int(min_window) + 8, min(int(height), int(width)) - 8)
    if win > shape_limit:
        win = shape_limit
    if win % 2 != 0:
        win = max(int(min_window), win - 1)

    requested_margin = max((win // 2) + int(sh), (win // 2) + int(sw))
    max_margin = max(16.0, (0.5 * float(min(int(height), int(width)))) - 2.0)
    if requested_margin > max_margin:
        scale = float(max_margin) / float(max(requested_margin, 1.0))
        win = int(np.floor(float(win) * scale))
        if win % 2 != 0:
            win -= 1
        win = max(int(min_window), win)
        sh = max(int(min_search), int(np.floor(float(sh) * scale)))
        sw = max(int(min_search), int(np.floor(float(sw) * scale)))

    return int(win), (int(sh), int(sw))


def _adaptive_stage2_minima(cfg):
    if not bool(cfg.get('adaptive_window_policy', True)):
        return None, None, 'adaptive_window_policy_disabled'

    res = cfg.get('resolution_m', None)
    try:
        res = float(res)
    except Exception:
        res = None

    if (res is not None) and np.isfinite(res) and (res > 0.0):
        if res < 5.0:
            return 512, 256, 'resolution_lt_5m'
        if res < 15.0:
            return 256, 128, 'resolution_5_to_15m'
        return None, None, 'resolution_ge_15m'

    prf = cfg.get('secondary_prf_hz', None)
    try:
        prf = float(prf)
    except Exception:
        prf = None

    if (prf is not None) and np.isfinite(prf) and (prf > 0.0):
        if prf >= 2500.0:
            return 512, 256, 'prf_ge_2500hz_fallback'
        if prf >= 1200.0:
            return 256, 128, 'prf_1200_to_2500hz_fallback'
        return None, None, 'prf_lt_1200hz_fallback'

    return None, None, 'no_resolution_or_prf'


def _collect_grid_offsets(
    master_amp,
    slave_amp,
    template_window,
    search_half,
    gross_az,
    gross_rg,
    grid_size,
    quality_threshold,
    geo2rdr_guard=None,
):
    h, w = master_amp.shape
    template_h, template_w = _positive_hw_pair(template_window, minimum=16)
    search_h, search_w = _positive_hw_pair(search_half, minimum=0)

    margin_r = max(template_h // 2 + search_h + abs(int(np.rint(gross_az))) + 2, 8)
    margin_c = max(template_w // 2 + search_w + abs(int(np.rint(gross_rg))) + 2, 8)
    if (h <= (2 * margin_r + 1)) or (w <= (2 * margin_c + 1)):
        return []

    rows = np.linspace(float(margin_r), float(h - margin_r - 1), max(2, int(grid_size)), dtype=np.float64)
    cols = np.linspace(float(margin_c), float(w - margin_c - 1), max(2, int(grid_size)), dtype=np.float64)

    points = []
    for row in rows:
        for col in cols:
            if not _geo2rdr_guard_valid(geo2rdr_guard, row, col):
                continue
            one = _template_search_displacement(
                master_amp,
                slave_amp,
                row,
                col,
                (template_h, template_w),
                (search_h, search_w),
                gross_az=gross_az,
                gross_rg=gross_rg,
            )
            if one is None:
                continue

            total_az, total_rg, quality = one
            if float(quality) < float(quality_threshold):
                continue

            residual_az = float(total_az) - float(gross_az)
            residual_rg = float(total_rg) - float(gross_rg)
            if (abs(residual_az) > float(search_h)) or (abs(residual_rg) > float(search_w)):
                continue

            points.append(
                {
                    'row': float(row),
                    'col': float(col),
                    'azimuth': float(total_az),
                    'range': float(total_rg),
                    'quality': float(quality),
                    'residual_azimuth': float(residual_az),
                    'residual_range': float(residual_rg),
                }
            )

    return points


def _linear_inlier_mask(points, sigma_threshold, max_iterations, min_points):
    if len(points) < int(min_points):
        return np.zeros(len(points), dtype=bool)

    rows = np.array([p['row'] for p in points], dtype=np.float64)
    cols = np.array([p['col'] for p in points], dtype=np.float64)
    az = np.array([p['azimuth'] for p in points], dtype=np.float64)
    rg = np.array([p['range'] for p in points], dtype=np.float64)

    A = np.column_stack([np.ones_like(rows), rows, cols])
    mask = np.ones(rows.shape[0], dtype=bool)
    for _ in range(max(1, int(max_iterations))):
        if int(np.count_nonzero(mask)) < int(min_points):
            break

        coeff_az, _, _, _ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
        coeff_rg, _, _, _ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
        pred_az = A @ coeff_az
        pred_rg = A @ coeff_rg
        residual = np.sqrt((az - pred_az) ** 2 + (rg - pred_rg) ** 2)

        core = residual[mask]
        med = float(np.median(core))
        mad = float(np.median(np.abs(core - med)))
        sigma = max(1.4826 * mad, 1.0e-6)
        threshold = med + float(sigma_threshold) * sigma
        new_mask = residual <= threshold

        if int(np.count_nonzero(new_mask)) < int(min_points):
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    if int(np.count_nonzero(mask)) < int(min_points):
        return np.zeros(len(points), dtype=bool)
    return mask


def _summarize_points(points):
    if not points:
        return {
            'num_valid': 0,
            'quality_median': None,
            'spread': None,
            'mean_azimuth': None,
            'mean_range': None,
        }

    az = np.array([p['azimuth'] for p in points], dtype=np.float64)
    rg = np.array([p['range'] for p in points], dtype=np.float64)
    q = np.array([p['quality'] for p in points], dtype=np.float64)
    return {
        'num_valid': int(len(points)),
        'quality_median': float(np.median(q)),
        'spread': float(np.std(az) + np.std(rg)),
        'mean_azimuth': float(np.mean(az)),
        'mean_range': float(np.mean(rg)),
    }


def _safe_percentile(values, q):
    if values.size <= 0:
        return np.nan
    return float(np.percentile(values, float(q)))


def _offset_bounds(values, cfg, method):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return np.nan, np.nan

    m = str(method or 'modified_zscore').strip().lower()
    if m == 'iqr':
        q1 = _safe_percentile(arr, 25.0)
        q3 = _safe_percentile(arr, 75.0)
        iqr = float(q3 - q1)
        if (not np.isfinite(iqr)) or (abs(iqr) <= 1.0e-12):
            return float(q1), float(q3)
        coeff = float(cfg.get('offset_filter_iqr_coeff', 1.5))
        return float(q1 - coeff * iqr), float(q3 + coeff * iqr)

    if m == 'zscore':
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if (not np.isfinite(std)) or (std <= 1.0e-12):
            return float(mean), float(mean)
        thr = float(cfg.get('offset_filter_z_threshold', 3.0))
        return float(mean - thr * std), float(mean + thr * std)

    # Default: modified_zscore (GMTSAR filter_offset.csh default).
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if (not np.isfinite(mad)) or (mad <= 1.0e-12):
        return float(med), float(med)
    thr = float(cfg.get('offset_filter_modified_z_threshold', 3.29))
    delta = float(thr) * float(mad) / 0.6745
    return float(med - delta), float(med + delta)


def _gmtsar_style_filter_points(points, cfg, logger=None, tag='offset'):
    if not points:
        return [], {
            'tag': str(tag),
            'method': str(cfg.get('offset_filter_method', 'modified_zscore')),
            'input_points': 0,
            'kept_points': 0,
            'dx_bounds': [None, None],
            'dy_bounds': [None, None],
        }

    method = str(cfg.get('offset_filter_method', 'modified_zscore')).strip().lower()
    dx = np.array([float(p.get('range', np.nan)) for p in points], dtype=np.float64)
    dy = np.array([float(p.get('azimuth', np.nan)) for p in points], dtype=np.float64)

    finite_mask = np.isfinite(dx) & np.isfinite(dy)
    finite_points = [pt for keep, pt in zip(finite_mask.tolist(), points) if keep]
    if not finite_points:
        return [], {
            'tag': str(tag),
            'method': str(method),
            'input_points': int(len(points)),
            'kept_points': 0,
            'dx_bounds': [None, None],
            'dy_bounds': [None, None],
        }

    dx_finite = dx[finite_mask]
    dy_finite = dy[finite_mask]
    dx_l, dx_u = _offset_bounds(dx_finite, cfg, method)
    dy_l, dy_u = _offset_bounds(dy_finite, cfg, method)

    keep_mask = (
        (dx_finite >= float(dx_l))
        & (dx_finite <= float(dx_u))
        & (dy_finite >= float(dy_l))
        & (dy_finite <= float(dy_u))
    )
    filtered = [pt for keep, pt in zip(keep_mask.tolist(), finite_points) if keep]
    summary = {
        'tag': str(tag),
        'method': str(method),
        'input_points': int(len(points)),
        'finite_points': int(len(finite_points)),
        'kept_points': int(len(filtered)),
        'dx_bounds': [float(dx_l), float(dx_u)],
        'dy_bounds': [float(dy_l), float(dy_u)],
    }

    if logger is not None:
        logger.info(
            'External %s GMTSAR-style offset filter (%s): points %d -> %d, '
            'dx_bounds=[%.6f, %.6f], dy_bounds=[%.6f, %.6f].',
            str(tag),
            str(method),
            int(len(points)),
            int(len(filtered)),
            float(dx_l),
            float(dx_u),
            float(dy_l),
            float(dy_u),
        )
    return filtered, summary


def _stage1_initial_integer_offset(master_amp, slave_amp, cfg, logger=None):
    stage1_source = str(cfg.get('stage1_source', 'geo2rdr_mean')).strip().lower()
    if stage1_source == 'geo2rdr_mean':
        az_path = _normalize_image_path(cfg.get('stage1_geo2rdr_azimuth_offset'))
        rg_path = _normalize_image_path(cfg.get('stage1_geo2rdr_range_offset'))
        nodata = float(cfg.get('stage1_geo2rdr_nodata', -999999.0))
        invalid_low = _geo2rdr_invalid_low(cfg)
        require_geo2rdr = bool(cfg.get('stage1_require_geo2rdr', True))

        try:
            az, az_w, az_l, _ = _read_real_image_as_memmap(az_path)
            rg, rg_w, rg_l, _ = _read_real_image_as_memmap(rg_path)

            if (az_w != rg_w) or (az_l != rg_l):
                raise RuntimeError(
                    'geo2rdr offset shape mismatch: az={0}x{1}, rg={2}x{3}'.format(
                        az_l, az_w, rg_l, rg_w
                    )
                )

            valid = _geo2rdr_valid_mask_arrays(az, rg, nodata, invalid_low)
            valid_count = int(np.count_nonzero(valid))
            total_count = int(valid.size)
            if valid_count <= 0:
                raise RuntimeError(
                    'geo2rdr offsets have no valid pixels (nodata={0}, invalid_low={1})'.format(
                        nodata,
                        invalid_low,
                    )
                )

            mean_az = float(np.mean(az[valid], dtype=np.float64))
            mean_rg = float(np.mean(rg[valid], dtype=np.float64))
            init_az = int(np.trunc(mean_az))
            init_rg = int(np.trunc(mean_rg))

            if logger is not None:
                logger.info(
                    'External staged step-1: source=geo2rdr_mean valid=%d/%d (%.2f%%) '
                    'mean=(az=%.6f, rg=%.6f) init_integer=(az=%d, rg=%d), '
                    'nodata=%.1f, invalid_low=%.1f.',
                    valid_count,
                    total_count,
                    100.0 * float(valid_count) / float(max(total_count, 1)),
                    mean_az,
                    mean_rg,
                    init_az,
                    init_rg,
                    nodata,
                    invalid_low,
                )

            return {
                'source': 'geo2rdr_mean',
                'offset_paths': {
                    'azimuth': str(az_path),
                    'range': str(rg_path),
                },
                'nodata': float(nodata),
                'invalid_low': float(invalid_low),
                'valid_pixels': int(valid_count),
                'total_pixels': int(total_count),
                'mean_offset': {'azimuth': float(mean_az), 'range': float(mean_rg)},
                'initial_integer_offset': {'azimuth': int(init_az), 'range': int(init_rg)},
                'disabled_large_window_probe': True,
            }
        except Exception as err:
            if require_geo2rdr:
                raise RuntimeError(
                    'External staged step-1 failed to derive initial offset from geo2rdr: {0}'.format(str(err))
                )
            if logger is not None:
                logger.warning(
                    'External staged step-1 geo2rdr_mean unavailable (%s); fallback to large-window probe.',
                    str(err),
                )

    h, w = master_amp.shape
    window, search_half = _resolve_window_and_search(
        cfg.get('stage1_window', 2048),
        cfg.get('stage1_search_half', 1024),
        h,
        w,
        min_window=64,
        min_search=8,
    )

    raw_points = _collect_grid_offsets(
        master_amp,
        slave_amp,
        template_window=(window, window),
        search_half=search_half,
        gross_az=0.0,
        gross_rg=0.0,
        grid_size=max(2, int(cfg.get('stage1_grid_size', 4))),
        quality_threshold=float(cfg.get('stage1_quality_threshold', 18.0)),
    )
    min_valid = max(4, int(cfg.get('stage1_min_valid', 4)))
    if len(raw_points) < min_valid:
        raise RuntimeError(
            'External staged step-1 failed: valid points={0} (<{1})'.format(len(raw_points), min_valid)
        )

    mask = _linear_inlier_mask(
        raw_points,
        sigma_threshold=float(cfg.get('stage1_outlier_sigma', 2.5)),
        max_iterations=int(cfg.get('stage1_outlier_max_iterations', 8)),
        min_points=min_valid,
    )
    filtered_points = [pt for keep, pt in zip(mask.tolist(), raw_points) if keep]
    if len(filtered_points) < min_valid:
        raise RuntimeError(
            'External staged step-1 failed after linear filtering: kept={0} (<{1})'.format(
                len(filtered_points),
                min_valid,
            )
        )

    mean_az = float(np.mean([p['azimuth'] for p in filtered_points]))
    mean_rg = float(np.mean([p['range'] for p in filtered_points]))
    init_az = int(np.trunc(mean_az))
    init_rg = int(np.trunc(mean_rg))

    if logger is not None:
        logger.info(
            'External staged step-1: template=(%d,%d) search_half=(%d,%d) raw=%d kept=%d '
            'mean=(az=%.6f, rg=%.6f) init_integer=(az=%d, rg=%d).',
            int(window),
            int(window),
            int(search_half[0]),
            int(search_half[1]),
            int(len(raw_points)),
            int(len(filtered_points)),
            float(mean_az),
            float(mean_rg),
            int(init_az),
            int(init_rg),
        )

    return {
        'template_window': {'down': int(window), 'across': int(window)},
        'search_half': {'down': int(search_half[0]), 'across': int(search_half[1])},
        'grid_size': int(max(2, int(cfg.get('stage1_grid_size', 4)))),
        'raw_points': int(len(raw_points)),
        'kept_points': int(len(filtered_points)),
        'raw_summary': _summarize_points(raw_points),
        'filtered_summary': _summarize_points(filtered_points),
        'mean_offset': {'azimuth': float(mean_az), 'range': float(mean_rg)},
        'initial_integer_offset': {'azimuth': int(init_az), 'range': int(init_rg)},
    }


def _stage2_select_window(master_amp, slave_amp, init_az, init_rg, cfg, logger=None, geo2rdr_guard=None):
    h, w = master_amp.shape
    raw_windows = cfg.get('stage2_windows', [1024, 512, 256, 128])
    if not raw_windows:
        raw_windows = [int(cfg.get('fine_window', 128))]

    min_search = max(1, int(cfg.get('stage2_search_half_min', 1)))
    max_search = max(min_search, int(cfg.get('stage2_search_half_max', 16384)))
    min_window_eff = 32

    coerced_windows = []
    seen_windows = set()
    for requested in raw_windows:
        try:
            req = int(np.rint(float(requested)))
        except Exception:
            continue
        if req <= 0:
            continue
        if req in seen_windows:
            continue
        seen_windows.add(req)
        coerced_windows.append(req)
    if not coerced_windows:
        coerced_windows = [int(max(min_window_eff, int(cfg.get('fine_window', 128))))]
    raw_windows = coerced_windows
    base_scale = float(cfg.get('stage2_search_half_scale', 0.5))
    raw_scales = cfg.get('stage2_search_half_scales', [])
    search_scales = []
    if isinstance(raw_scales, (list, tuple, np.ndarray)):
        for one in raw_scales:
            try:
                sval = float(one)
            except Exception:
                continue
            if sval > 0.0:
                search_scales.append(float(sval))
    else:
        try:
            sval = float(raw_scales)
            if sval > 0.0:
                search_scales.append(float(sval))
        except Exception:
            pass

    if not search_scales:
        search_scales = [float(base_scale), float(base_scale) * 0.75, float(base_scale) * 1.25]

    uniq_scales = []
    seen_scales = set()
    for sval in sorted(search_scales):
        key = round(float(sval), 6)
        if key in seen_scales:
            continue
        seen_scales.add(key)
        uniq_scales.append(float(sval))
    search_scales = uniq_scales

    quality_threshold = float(cfg.get('stage2_quality_threshold', 18.0))
    grid_size = max(2, int(cfg.get('stage2_grid_size', 4)))
    min_valid = max(4, int(cfg.get('stage2_min_valid', 4)))
    prefer_larger = bool(cfg.get('stage2_prefer_larger_window', True))

    if logger is not None:
        logger.info(
            'External staged step-2 candidates=%s, search_half_scales=%s.',
            str(raw_windows),
            str([round(float(v), 4) for v in search_scales]),
        )

    candidates = []
    for requested in raw_windows:
        try:
            req = int(np.rint(float(requested)))
        except Exception:
            continue
        if req <= 0:
            continue
        for scale in search_scales:
            req_search = int(np.rint(max(1.0, float(req) * float(scale))))
            req_search = max(int(min_search), min(int(max_search), int(req_search)))
            window, search_half = _resolve_window_and_search(
                req,
                (req_search, req_search),
                h,
                w,
                min_window=min_window_eff,
                min_search=min_search,
            )

            points = _collect_grid_offsets(
                master_amp,
                slave_amp,
                template_window=(window, window),
                search_half=search_half,
                gross_az=float(init_az),
                gross_rg=float(init_rg),
                grid_size=grid_size,
                quality_threshold=quality_threshold,
                geo2rdr_guard=geo2rdr_guard,
            )
            filtered_points, filter_summary = _gmtsar_style_filter_points(
                points,
                cfg,
                logger=logger,
                tag='staged_step2_template_{0}_scale_{1:.3f}'.format(int(window), float(scale)),
            )
            summary = _summarize_points(filtered_points)
            status = 'ok' if len(filtered_points) >= min_valid else 'failed'
            candidates.append(
                {
                    'status': status,
                    'requested_window': int(req),
                    'requested_search_scale': float(scale),
                    'requested_search_half': int(req_search),
                    'window': int(window),
                    'template_window': {'down': int(window), 'across': int(window)},
                    'search_half': {'down': int(search_half[0]), 'across': int(search_half[1])},
                    'num_valid': int(len(filtered_points)),
                    'raw_num_valid': int(len(points)),
                    'quality_median': summary['quality_median'],
                    'spread': summary['spread'],
                    'mean_azimuth': summary['mean_azimuth'],
                    'mean_range': summary['mean_range'],
                    'filter': filter_summary,
                }
            )

    valid = [c for c in candidates if c.get('status') == 'ok']
    if not valid:
        raise RuntimeError(
            'External staged step-2 failed: no valid window candidates ({0})'.format(
                [int(c.get('requested_window', 0)) for c in candidates]
            )
        )

    if prefer_larger:
        best = max(
            valid,
            key=lambda c: (
                int(c.get('num_valid', 0)),
                -float(c.get('spread') or np.inf),
                float(c.get('quality_median') or 0.0),
                int(c.get('window', 0)),
                -max(int((c.get('search_half') or {}).get('down', 0)), int((c.get('search_half') or {}).get('across', 0))),
            ),
        )
    else:
        best = max(
            valid,
            key=lambda c: (
                int(c.get('num_valid', 0)),
                -float(c.get('spread') or np.inf),
                float(c.get('quality_median') or 0.0),
                -int(c.get('window', 0)),
                -max(int((c.get('search_half') or {}).get('down', 0)), int((c.get('search_half') or {}).get('across', 0))),
            ),
        )

    # When stage-2 candidates are close on valid-count/spread, prefer a smaller
    # template to improve efficiency and reduce local-minimum lock-in risk.
    if bool(cfg.get('stage2_prefer_smaller_when_close', True)):
        valid_margin = max(0, int(cfg.get('stage2_valid_margin_for_smaller_window', 1)))
        spread_margin = max(0.0, float(cfg.get('stage2_spread_margin_for_smaller_window', 2.0)))
        best_valid = int(best.get('num_valid', 0))
        best_spread = float(best.get('spread') or np.inf)
        best_window = int(best.get('window', 0))
        near = [
            c for c in valid
            if int(c.get('window', 0)) < best_window
            and int(c.get('num_valid', 0)) >= (best_valid - valid_margin)
            and float(c.get('spread') or np.inf) <= (best_spread + spread_margin)
        ]
        if near:
            smaller = max(
                near,
                key=lambda c: (
                    -int(c.get('window', 0)),
                    int(c.get('num_valid', 0)),
                    -float(c.get('spread') or np.inf),
                    float(c.get('quality_median') or 0.0),
                ),
            )
            if logger is not None:
                logger.info(
                    'External staged step-2 close-score preference switched to smaller template: '
                    'from window=%d(valid=%d, spread=%.6f) to window=%d(valid=%d, spread=%.6f), '
                    'valid_margin=%d spread_margin=%.6f.',
                    int(best_window),
                    int(best_valid),
                    float(best_spread),
                    int(smaller.get('window', 0)),
                    int(smaller.get('num_valid', 0)),
                    float(smaller.get('spread') or np.inf),
                    int(valid_margin),
                    float(spread_margin),
                )
            best = smaller

    # Joint optimization of search half with close-score preference.
    if bool(cfg.get('stage2_prefer_smaller_search_when_close', True)):
        valid_margin = max(0, int(cfg.get('stage2_valid_margin_for_smaller_search', 1)))
        spread_margin = max(0.0, float(cfg.get('stage2_spread_margin_for_smaller_search', 1.0)))
        quality_margin = max(0.0, float(cfg.get('stage2_quality_margin_for_smaller_search', 1.0)))
        best_valid = int(best.get('num_valid', 0))
        best_spread = float(best.get('spread') or np.inf)
        best_quality = float(best.get('quality_median') or 0.0)
        best_search = max(
            int((best.get('search_half') or {}).get('down', 0)),
            int((best.get('search_half') or {}).get('across', 0)),
        )
        near = [
            c for c in valid
            if max(
                int((c.get('search_half') or {}).get('down', 0)),
                int((c.get('search_half') or {}).get('across', 0)),
            ) < best_search
            and int(c.get('num_valid', 0)) >= (best_valid - valid_margin)
            and float(c.get('spread') or np.inf) <= (best_spread + spread_margin)
            and float(c.get('quality_median') or 0.0) >= (best_quality - quality_margin)
        ]
        if near:
            smaller = max(
                near,
                key=lambda c: (
                    -max(
                        int((c.get('search_half') or {}).get('down', 0)),
                        int((c.get('search_half') or {}).get('across', 0)),
                    ),
                    int(c.get('num_valid', 0)),
                    -float(c.get('spread') or np.inf),
                    float(c.get('quality_median') or 0.0),
                    int(c.get('window', 0)),
                ),
            )
            if logger is not None:
                logger.info(
                    'External staged step-2 close-score preference switched to smaller search-half: '
                    'from search_half=(%d,%d), valid=%d, spread=%.6f, quality=%.5f '
                    'to search_half=(%d,%d), valid=%d, spread=%.6f, quality=%.5f.',
                    int((best.get('search_half') or {}).get('down', 0)),
                    int((best.get('search_half') or {}).get('across', 0)),
                    int(best_valid),
                    float(best_spread),
                    float(best_quality),
                    int((smaller.get('search_half') or {}).get('down', 0)),
                    int((smaller.get('search_half') or {}).get('across', 0)),
                    int(smaller.get('num_valid', 0)),
                    float(smaller.get('spread') or np.inf),
                    float(smaller.get('quality_median') or 0.0),
                )
            best = smaller

    if logger is not None:
        for cand in candidates:
            if cand.get('status') == 'ok':
                logger.info(
                    'External staged step-2 candidate: template=(%d,%d) search_half=(%d,%d) '
                    'scale=%.3f valid=%d raw=%d quality_median=%.5f spread=%.6f',
                    int(cand['template_window']['down']),
                    int(cand['template_window']['across']),
                    int(cand['search_half']['down']),
                    int(cand['search_half']['across']),
                    float(cand.get('requested_search_scale', np.nan)),
                    int(cand['num_valid']),
                    int(cand.get('raw_num_valid', 0)),
                    float(cand['quality_median'] or 0.0),
                    float(cand['spread'] or 0.0),
                )
            else:
                logger.info(
                    'External staged step-2 candidate failed: template=(%d,%d) search_half=(%d,%d) '
                    'scale=%.3f valid=%d raw=%d',
                    int(cand['template_window']['down']),
                    int(cand['template_window']['across']),
                    int(cand['search_half']['down']),
                    int(cand['search_half']['across']),
                    float(cand.get('requested_search_scale', np.nan)),
                    int(cand['num_valid']),
                    int(cand.get('raw_num_valid', 0)),
                )
        logger.info(
            'External staged step-2 selected: template=(%d,%d) search_half=(%d,%d) valid=%d '
            'quality_median=%.5f spread=%.6f',
            int(best['template_window']['down']),
            int(best['template_window']['across']),
            int(best['search_half']['down']),
            int(best['search_half']['across']),
            int(best['num_valid']),
            float(best['quality_median'] or 0.0),
            float(best['spread'] or 0.0),
        )

    return {
        'best': best,
        'candidates': candidates,
        'grid_size': int(grid_size),
        'quality_threshold': float(quality_threshold),
        'initial_integer_offset': {'azimuth': int(init_az), 'range': int(init_rg)},
    }


def _points_minus_initial(points, init_az, init_rg):
    out = []
    for p in points:
        one = dict(p)
        one['azimuth'] = float(p['azimuth']) - float(init_az)
        one['range'] = float(p['range']) - float(init_rg)
        out.append(one)
    return out


def _evaluate_fine_point(
    master_amp,
    slave_amp,
    row,
    col,
    coarse_az,
    coarse_rg,
    template_window,
    search_half,
    quality_threshold,
):
    result = _template_search_displacement(
        master_amp,
        slave_amp,
        row,
        col,
        template_window,
        search_half,
        gross_az=coarse_az,
        gross_rg=coarse_rg,
    )
    if result is None:
        return None

    total_az, total_rg, quality = result
    if quality < quality_threshold:
        return None

    return {
        'row': float(row),
        'col': float(col),
        'azimuth': float(total_az),
        'range': float(total_rg),
        'quality': float(quality),
    }


def _fine_registration(
    master_amp,
    slave_amp,
    coarse_az,
    coarse_rg,
    cfg,
    coarse_window=None,
    coarse_search=None,
    logger=None,
    geo2rdr_guard=None,
):
    h, w = master_amp.shape

    requested_window = int(cfg['fine_window'])
    if coarse_window is not None:
        try:
            requested_window = int(np.rint(float(coarse_window)))
        except Exception:
            requested_window = int(cfg['fine_window'])

    if requested_window <= 0:
        requested_window = int(cfg['fine_window'])

    window = int(requested_window)
    if window % 2 != 0:
        window += 1

    large_threshold = max(32, int(cfg.get('fine_large_coarse_threshold', 1024)))
    window_cap = max(32, int(cfg.get('fine_window_cap_for_large_coarse', 1024)))
    fixed_grid_size = max(3, int(cfg.get('fine_large_coarse_grid_size', 60)))
    use_fixed_grid = (requested_window > large_threshold)
    if use_fixed_grid:
        window = min(window, window_cap)
        if window % 2 != 0:
            window -= 1

    # Do not re-limit search radius in fine stage when stage-2 already selected it.
    # Preferred policy: use stage-2 selected search_half directly.
    if coarse_search is not None:
        search_h, search_w = _positive_hw_pair(coarse_search, minimum=1)
        search_half = (search_h, search_w)
        if logger is not None:
            logger.info(
                'External fine registration uses stage-2 selected search_half directly: '
                'template=(%d,%d), search_half=(%d,%d).',
                int(window),
                int(window),
                int(search_h),
                int(search_w),
            )
    else:
        force_from_window = bool(cfg.get('fine_force_search_half_from_window', True))
        force_half_window = bool(cfg.get('fine_force_search_half_to_half_window', True))
        fine_search_min = max(1, int(cfg.get('fine_search_min', 16)))
        fine_search_max = max(fine_search_min, int(cfg.get('fine_search_max', 128)))
        if force_from_window:
            if force_half_window:
                search_h = max(1, int(window // 2))
                search_w = max(1, int(window // 2))
                search_half = (search_h, search_w)
                if logger is not None:
                    logger.info(
                        'External fine registration search_half fallback to half template: '
                        'template=(%d,%d) -> search_half=(%d,%d).',
                        int(window),
                        int(window),
                        int(search_h),
                        int(search_w),
                    )
            else:
                force_scale = float(cfg.get('fine_force_search_half_scale', cfg.get('fine_search_scale', 0.25)))
                if force_scale <= 0.0:
                    force_scale = 0.25
                search_h = int(np.rint(float(window) * force_scale))
                search_w = int(np.rint(float(window) * force_scale))
                search_h = max(fine_search_min, min(fine_search_max, search_h))
                search_w = max(fine_search_min, min(fine_search_max, search_w))
                search_half = (search_h, search_w)
                if logger is not None:
                    logger.info(
                        'External fine registration search_half fallback from template: '
                        'template=(%d,%d) scale=%.3f -> search_half=(%d,%d).',
                        int(window),
                        int(window),
                        float(force_scale),
                        int(search_h),
                        int(search_w),
                    )
        else:
            search_h = max(1, int(cfg.get('fine_search_max', 128)))
            search_w = max(1, int(cfg.get('fine_search_max', 128)))
            search_half = (search_h, search_w)

    spacing = max(16, int(cfg['fine_spacing']))
    margin = (
        max(window // 2, search_h, search_w)
        + max(abs(int(np.rint(coarse_az))), abs(int(np.rint(coarse_rg))))
        + 2
    )
    fine_max_az_points = max(1, int(cfg.get('fine_max_az_points', 60)))
    fine_max_rg_points = max(1, int(cfg.get('fine_max_rg_points', 30)))
    fine_max_total_points = max(1, int(cfg.get('fine_max_total_points', fine_max_az_points * fine_max_rg_points)))
    fine_dir_cap = int(fine_max_az_points * fine_max_rg_points)
    fine_max_total_points = min(int(fine_max_total_points), int(fine_dir_cap))

    max_points = int(cfg.get('max_points', 0))
    max_points_cap = int(cfg.get('max_points_cap', fine_max_total_points))
    if max_points_cap > 0:
        max_points_cap = min(int(max_points_cap), int(fine_max_total_points))
        if max_points <= 0:
            max_points = int(max_points_cap)
        else:
            max_points = min(int(max_points), int(max_points_cap))
    unlimited_points = (max_points <= 0)
    quality_threshold = float(cfg['fine_quality_threshold'])
    chunk_size = max(8, int(cfg.get('fine_chunk_size', 128)))
    workers_cfg = int(cfg.get('fine_workers', 0))
    if workers_cfg <= 0:
        workers = max(1, min(os.cpu_count() or 1, 8))
    else:
        workers = max(1, workers_cfg)

    if use_fixed_grid:
        grid_points = _iter_uniform_grid_points(
            h,
            w,
            margin,
            min(int(fixed_grid_size), int(fine_max_az_points)),
            min(int(fixed_grid_size), int(fine_max_rg_points)),
        )
        grid_mode = 'fixed_{0}x{1}'.format(
            min(int(fixed_grid_size), int(fine_max_az_points)),
            min(int(fixed_grid_size), int(fine_max_rg_points)),
        )
    else:
        grid_points = list(_iter_grid_points(h, w, spacing, margin))
        grid_mode = 'spacing_{0}'.format(spacing)

    # Hard-limit fine registration sampling density for stability and runtime:
    # azimuth points <= fine_max_az_points, range points <= fine_max_rg_points,
    # total points <= fine_max_total_points.
    if len(grid_points) > int(fine_max_total_points):
        grid_points = _iter_uniform_grid_points(
            h,
            w,
            margin,
            int(fine_max_az_points),
            int(fine_max_rg_points),
        )
        grid_mode = 'limited_uniform_{0}x{1}'.format(
            int(fine_max_az_points),
            int(fine_max_rg_points),
        )

    if geo2rdr_guard is not None:
        before = len(grid_points)
        grid_points = [
            (row, col)
            for row, col in grid_points
            if _geo2rdr_guard_valid(geo2rdr_guard, row, col)
        ]
        if logger is not None:
            logger.info(
                'External fine registration applied geo2rdr valid-mask guard: kept=%d/%d candidate points.',
                int(len(grid_points)),
                int(before),
            )

    if not grid_points:
        raise RuntimeError(
            'Integrated external fine registration failed: no candidate points '
            '(window={0}, margin={1}, image={2}x{3})'.format(window, margin, h, w)
        )

    points = []

    if logger is not None:
        logger.info(
            'External fine registration config bridged from coarse: coarse_template=%d fine_template=(%d,%d) search_half=(%d,%d) grid_mode=%s',
            int(requested_window),
            int(window),
            int(window),
            int(search_h),
            int(search_w),
            grid_mode,
        )
    if (logger is not None) and (workers > 1):
        logger.info(
            'External fine registration parallel mode: workers=%d, chunk_size=%d, candidate_points=%d',
            workers,
            chunk_size,
            len(grid_points),
        )
    if logger is not None:
        logger.info(
            'External fine registration point limits: az<=%d range<=%d total<=%d, effective_max_points=%s',
            int(fine_max_az_points),
            int(fine_max_rg_points),
            int(fine_max_total_points),
            str(max_points if not unlimited_points else 'unlimited'),
        )
    if (logger is not None) and unlimited_points:
        logger.info('External fine registration uses all valid points (max_points<=0).')

    if workers <= 1:
        for row, col in grid_points:
            one = _evaluate_fine_point(
                master_amp,
                slave_amp,
                row,
                col,
                coarse_az,
                coarse_rg,
                (window, window),
                search_half,
                quality_threshold,
            )
            if one is not None:
                points.append(one)
            if (not unlimited_points) and (len(points) >= max_points):
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(grid_points), chunk_size):
                chunk = grid_points[start:start + chunk_size]
                rows = [rc[0] for rc in chunk]
                cols = [rc[1] for rc in chunk]
                results = executor.map(
                    _evaluate_fine_point,
                    repeat(master_amp),
                    repeat(slave_amp),
                    rows,
                    cols,
                    repeat(coarse_az),
                    repeat(coarse_rg),
                    repeat((window, window)),
                    repeat(search_half),
                    repeat(quality_threshold),
                )
                for one in results:
                    if one is not None:
                        points.append(one)
                if (not unlimited_points) and (len(points) >= max_points):
                    break

    if len(points) < max(6, int(cfg['min_points'])):
        raise RuntimeError(
            'Integrated external fine registration failed: valid points={0}'.format(len(points))
        )

    return points


def _design_matrix(rows_norm, cols_norm):
    return np.column_stack(
        [
            np.ones_like(rows_norm),
            rows_norm,
            cols_norm,
            rows_norm * cols_norm,
            rows_norm * rows_norm,
            cols_norm * cols_norm,
        ]
    )


def _quadratic_inlier_mask(points, sigma_threshold, max_iterations, min_points):
    if len(points) < int(min_points):
        return np.zeros(len(points), dtype=bool)

    rows = np.array([p['row'] for p in points], dtype=np.float64)
    cols = np.array([p['col'] for p in points], dtype=np.float64)
    az = np.array([p['azimuth'] for p in points], dtype=np.float64)
    rg = np.array([p['range'] for p in points], dtype=np.float64)

    rows_mean = float(np.mean(rows))
    cols_mean = float(np.mean(cols))
    rows_std = float(np.std(rows))
    cols_std = float(np.std(cols))
    rows_std = rows_std if rows_std > 1.0e-10 else 1.0
    cols_std = cols_std if cols_std > 1.0e-10 else 1.0
    rows_norm = (rows - rows_mean) / rows_std
    cols_norm = (cols - cols_mean) / cols_std
    A = _design_matrix(rows_norm, cols_norm)

    mask = np.ones(rows.shape[0], dtype=bool)
    for _ in range(max(1, int(max_iterations))):
        if int(np.count_nonzero(mask)) < int(min_points):
            break

        coeff_az, _, _, _ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
        coeff_rg, _, _, _ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
        pred_az = A @ coeff_az
        pred_rg = A @ coeff_rg
        residual = np.sqrt((az - pred_az) ** 2 + (rg - pred_rg) ** 2)

        core = residual[mask]
        med = float(np.median(core))
        mad = float(np.median(np.abs(core - med)))
        sigma = max(1.4826 * mad, 1.0e-6)
        threshold = med + float(sigma_threshold) * sigma
        new_mask = residual <= threshold

        if int(np.count_nonzero(new_mask)) < int(min_points):
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    if int(np.count_nonzero(mask)) < int(min_points):
        return np.zeros(len(points), dtype=bool)
    return mask


def _quadratic_screen_points(points, cfg, logger=None, tag='fine'):
    min_points = max(6, int(cfg.get('min_points', 36)))
    sigma_threshold = float(cfg.get('sigma_threshold', 2.5))
    max_iterations = max(1, int(cfg.get('max_iterations', 8)))
    mask = _quadratic_inlier_mask(
        points,
        sigma_threshold=sigma_threshold,
        max_iterations=max_iterations,
        min_points=min_points,
    )
    filtered_points = [pt for keep, pt in zip(mask.tolist(), points) if keep]
    if len(filtered_points) < min_points:
        raise RuntimeError(
            'Integrated external quadratic screening failed ({0}): kept={1} (<{2})'.format(
                str(tag),
                int(len(filtered_points)),
                int(min_points),
            )
        )
    if logger is not None:
        logger.info(
            'External %s quadratic screening: points %d -> %d (sigma=%.2f, max_iter=%d).',
            str(tag),
            int(len(points)),
            int(len(filtered_points)),
            float(sigma_threshold),
            int(max_iterations),
        )
    return filtered_points, {
        'tag': str(tag),
        'input_points': int(len(points)),
        'kept_points': int(len(filtered_points)),
        'sigma_threshold': float(sigma_threshold),
        'max_iterations': int(max_iterations),
        'min_points': int(min_points),
    }


def _fit_model(points, cfg):
    rows = np.array([p['row'] for p in points], dtype=np.float64)
    cols = np.array([p['col'] for p in points], dtype=np.float64)
    az = np.array([p['azimuth'] for p in points], dtype=np.float64)
    rg = np.array([p['range'] for p in points], dtype=np.float64)

    rows_mean = float(np.mean(rows))
    cols_mean = float(np.mean(cols))
    rows_std = float(np.std(rows))
    cols_std = float(np.std(cols))
    rows_std = rows_std if rows_std > 1.0e-10 else 1.0
    cols_std = cols_std if cols_std > 1.0e-10 else 1.0

    rows_norm = (rows - rows_mean) / rows_std
    cols_norm = (cols - cols_mean) / cols_std
    A = _design_matrix(rows_norm, cols_norm)

    mask = np.ones(rows.shape[0], dtype=bool)
    max_iterations = max(1, int(cfg['max_iterations']))
    sigma_threshold = float(cfg['sigma_threshold'])

    for _ in range(max_iterations):
        if np.count_nonzero(mask) < 6:
            break
        coeff_az, _, _, _ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
        coeff_rg, _, _, _ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
        pred_az = A @ coeff_az
        pred_rg = A @ coeff_rg
        residual = np.sqrt((az - pred_az) ** 2 + (rg - pred_rg) ** 2)
        core = residual[mask]
        med = float(np.median(core))
        mad = float(np.median(np.abs(core - med)))
        sigma = max(1.4826 * mad, 1.0e-6)
        new_mask = residual <= (med + sigma_threshold * sigma)
        if np.count_nonzero(new_mask) < 6:
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    if np.count_nonzero(mask) < 6:
        raise RuntimeError(
            'Integrated external polynomial fit failed: inliers={0}'.format(np.count_nonzero(mask))
        )

    coeff_az, _, _, _ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
    coeff_rg, _, _, _ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
    pred_az = A @ coeff_az
    pred_rg = A @ coeff_rg
    az_rms = float(np.sqrt(np.mean((az[mask] - pred_az[mask]) ** 2)))
    rg_rms = float(np.sqrt(np.mean((rg[mask] - pred_rg[mask]) ** 2)))

    params = {
        'a0': float(coeff_az[0]),
        'a1': float(coeff_az[1]),
        'a2': float(coeff_az[2]),
        'a3': float(coeff_az[3]),
        'a4': float(coeff_az[4]),
        'a5': float(coeff_az[5]),
        'b0': float(coeff_rg[0]),
        'b1': float(coeff_rg[1]),
        'b2': float(coeff_rg[2]),
        'b3': float(coeff_rg[3]),
        'b4': float(coeff_rg[4]),
        'b5': float(coeff_rg[5]),
    }

    return {
        'parameters': params,
        'normalization': {
            'rows_mean': rows_mean,
            'rows_std': rows_std,
            'cols_mean': cols_mean,
            'cols_std': cols_std,
        },
        'inliers': int(np.count_nonzero(mask)),
        'total_points': int(rows.shape[0]),
        'azimuth_rms': az_rms,
        'range_rms': rg_rms,
    }


def _build_poly2d(params, norm, key_prefix):
    c0 = float(params[key_prefix + '0'])
    c1 = float(params[key_prefix + '1'])
    c2 = float(params[key_prefix + '2'])
    c3 = float(params[key_prefix + '3'])
    c4 = float(params[key_prefix + '4'])
    c5 = float(params[key_prefix + '5'])

    coeffs = [
        [c0, c2, c5],
        [c1, c3, 0.0],
        [c4, 0.0, 0.0],
    ]

    poly = Poly2D()
    poly.setMeanAzimuth(float(norm['rows_mean']))
    poly.setNormAzimuth(float(norm['rows_std']))
    poly.setMeanRange(float(norm['cols_mean']))
    poly.setNormRange(float(norm['cols_std']))
    poly.initPoly(rangeOrder=2, azimuthOrder=2, coeffs=coeffs)
    return poly


def _apply_ratio(poly, ratio):
    scale = float(ratio)
    for row in poly._coeffs:
        for i, val in enumerate(row):
            row[i] = val * scale
    return poly


def estimate_optimal_window(
    reference_slc,
    secondary_slc,
    config=None,
    logger=None,
):
    """
    Run staged external coarse steps (stage-1 + stage-2) and return the
    selected template/search settings for downstream Ampcor.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    # External optimal-window selection always uses staged mode.
    cfg['staged_enable'] = True

    master = _read_slc_as_memmap(reference_slc)
    slave = _read_slc_as_memmap(secondary_slc)

    if master.shape != slave.shape:
        h = min(master.shape[0], slave.shape[0])
        w = min(master.shape[1], slave.shape[1])
        master = master[:h, :w]
        slave = slave[:h, :w]
        if logger is not None:
            logger.warning(
                'External window-selection cropped mismatched SLC sizes to %dx%d',
                h,
                w,
            )

    precompute_amp = bool(cfg.get('precompute_amplitude', True))
    master_amp = _AmplitudeAccessor(
        master,
        precompute=precompute_amp,
        logger=logger,
        label='reference',
    )
    slave_amp = _AmplitudeAccessor(
        slave,
        precompute=precompute_amp,
        logger=logger,
        label='secondary',
    )
    geo2rdr_guard = _build_geo2rdr_sampling_guard(cfg, master_amp.shape, logger=logger)

    stage1_disable = bool(cfg.get('stage1_disable', False))
    if stage1_disable:
        init_az = 0
        init_rg = 0
        stage1 = {
            'disabled': True,
            'reason': 'configured_stage1_disable',
            'initial_integer_offset': {'azimuth': 0, 'range': 0},
        }
        if logger is not None:
            logger.info(
                'External window-selection step-1 disabled by configuration; '
                'using init_integer=(az=0, rg=0).'
            )
    else:
        stage1 = _stage1_initial_integer_offset(master_amp, slave_amp, cfg, logger=logger)
        init_az = int(stage1['initial_integer_offset']['azimuth'])
        init_rg = int(stage1['initial_integer_offset']['range'])

    stage2 = _stage2_select_window(
        master_amp,
        slave_amp,
        init_az=init_az,
        init_rg=init_rg,
        cfg=cfg,
        logger=logger,
        geo2rdr_guard=geo2rdr_guard,
    )
    best = dict(stage2.get('best') or {})
    if not best:
        raise RuntimeError('External window-selection failed: stage-2 best candidate is empty.')

    best_window = int(best.get('window', 0))
    search_cfg = dict(best.get('search_half') or {})
    if best_window <= 0:
        raise RuntimeError(
            'External window-selection failed: invalid best window={0}'.format(best_window)
        )
    if not search_cfg:
        raise RuntimeError('External window-selection failed: missing best search-half.')

    out = {
        'enabled': True,
        'mode': 'integrated_external_window_selection',
        'reference_slc': _normalize_slc_path(reference_slc),
        'secondary_slc': _normalize_slc_path(secondary_slc),
        'initial_integer_offset': {'azimuth': int(init_az), 'range': int(init_rg)},
        'template_window': {'down': int(best_window), 'across': int(best_window)},
        'search_half': {
            'down': int(search_cfg.get('down', 0)),
            'across': int(search_cfg.get('across', 0)),
        },
        'stage1': stage1,
        'stage2': stage2,
    }
    return out


def estimate_misregistration_polys(
    reference_slc,
    secondary_slc,
    az_ratio=1.0,
    rg_ratio=1.0,
    config=None,
    logger=None,
):
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    master = _read_slc_as_memmap(reference_slc)
    slave = _read_slc_as_memmap(secondary_slc)

    if master.shape != slave.shape:
        h = min(master.shape[0], slave.shape[0])
        w = min(master.shape[1], slave.shape[1])
        master = master[:h, :w]
        slave = slave[:h, :w]
        if logger is not None:
            logger.warning(
                'Integrated external registration cropped mismatched SLC sizes to %dx%d',
                h,
                w,
            )

    precompute_amp = bool(cfg.get('precompute_amplitude', True))
    master_amp = _AmplitudeAccessor(
        master,
        precompute=precompute_amp,
        logger=logger,
        label='reference',
    )
    slave_amp = _AmplitudeAccessor(
        slave,
        precompute=precompute_amp,
        logger=logger,
        label='secondary',
    )
    geo2rdr_guard = _build_geo2rdr_sampling_guard(cfg, master_amp.shape, logger=logger)

    staged_enable = bool(cfg.get('staged_enable', True))
    quadratic_screen = None
    gmtsar_filter = None
    if staged_enable:
        stage1_disable = bool(cfg.get('stage1_disable', False))
        if stage1_disable:
            init_az = 0
            init_rg = 0
            stage1 = {
                'disabled': True,
                'reason': 'configured_stage1_disable',
                'initial_integer_offset': {'azimuth': 0, 'range': 0},
            }
            if logger is not None:
                logger.info(
                    'External staged step-1 disabled by configuration; using init_integer=(az=0, rg=0).'
                )
        else:
            stage1 = _stage1_initial_integer_offset(master_amp, slave_amp, cfg, logger=logger)
            init_az = int(stage1['initial_integer_offset']['azimuth'])
            init_rg = int(stage1['initial_integer_offset']['range'])

        stage2 = _stage2_select_window(
            master_amp,
            slave_amp,
            init_az=init_az,
            init_rg=init_rg,
            cfg=cfg,
            logger=logger,
            geo2rdr_guard=geo2rdr_guard,
        )
        best = dict(stage2['best'])
        best_search = (
            int(best['search_half']['down']),
            int(best['search_half']['across']),
        )
        points = _fine_registration(
            master_amp,
            slave_amp,
            coarse_az=float(init_az),
            coarse_rg=float(init_rg),
            cfg=cfg,
            coarse_window=int(best['window']),
            coarse_search=best_search,
            logger=logger,
            geo2rdr_guard=geo2rdr_guard,
        )
        points, gmtsar_filter = _gmtsar_style_filter_points(
            points,
            cfg,
            logger=logger,
            tag='staged_fine',
        )
        min_points = max(6, int(cfg.get('min_points', 36)))
        if len(points) < min_points:
            raise RuntimeError(
                'Integrated external GMTSAR-style filtering failed (staged_fine): kept={0} (<{1})'.format(
                    int(len(points)),
                    int(min_points),
                )
            )

        residual_points = _points_minus_initial(points, init_az, init_rg)
        residual_points, quadratic_screen = _quadratic_screen_points(
            residual_points,
            cfg,
            logger=logger,
            tag='staged_fine_residual',
        )
        model = _fit_model(residual_points, cfg)
        coarse = {
            'selection_mode': 'staged_external_registration',
            'azimuth_offset': float(init_az),
            'range_offset': float(init_rg),
            'window_size': int(best['window']),
            'search_half': dict(best['search_half']),
            'num_valid': int(best.get('num_valid', 0)),
            'spread': float(best.get('spread', np.inf))
            if best.get('spread', None) is not None
            else np.inf,
            'quality_median': float(best.get('quality_median', 0.0))
            if best.get('quality_median', None) is not None
            else 0.0,
            'candidate_summary': list(stage2['candidates']),
        }
    else:
        coarse = _coarse_registration(master_amp, slave_amp, cfg, logger=logger)
        points = _fine_registration(
            master_amp,
            slave_amp,
            coarse_az=coarse['azimuth_offset'],
            coarse_rg=coarse['range_offset'],
            cfg=cfg,
            coarse_window=coarse.get('window_size'),
            coarse_search=coarse.get('search_range'),
            logger=logger,
            geo2rdr_guard=geo2rdr_guard,
        )
        points, gmtsar_filter = _gmtsar_style_filter_points(
            points,
            cfg,
            logger=logger,
            tag='legacy_fine',
        )
        min_points = max(6, int(cfg.get('min_points', 36)))
        if len(points) < min_points:
            raise RuntimeError(
                'Integrated external GMTSAR-style filtering failed (legacy_fine): kept={0} (<{1})'.format(
                    int(len(points)),
                    int(min_points),
                )
            )
        points, quadratic_screen = _quadratic_screen_points(
            points,
            cfg,
            logger=logger,
            tag='legacy_fine',
        )
        model = _fit_model(points, cfg)
        stage1 = None
        stage2 = None
        init_az = int(np.trunc(float(coarse.get('azimuth_offset', 0.0))))
        init_rg = int(np.trunc(float(coarse.get('range_offset', 0.0))))

    azpoly = _build_poly2d(model['parameters'], model['normalization'], 'a')
    rgpoly = _build_poly2d(model['parameters'], model['normalization'], 'b')
    azpoly = _apply_ratio(azpoly, az_ratio)
    rgpoly = _apply_ratio(rgpoly, rg_ratio)

    meta = {
        'enabled': True,
        'mode': 'integrated_external_registration',
        'reference_slc': _normalize_slc_path(reference_slc),
        'secondary_slc': _normalize_slc_path(secondary_slc),
        'coarse': coarse,
        'staged': bool(staged_enable),
        'initial_integer_offset': {'azimuth': int(init_az), 'range': int(init_rg)},
        'stage1': stage1,
        'stage2': stage2,
        'resample_mode': str(cfg.get('resample_mode', 'geo2rdr_plus_misreg')).strip().lower(),
        'offset_model': 'residual_poly_after_geo2rdr',
        'fit': {
            'inliers': model['inliers'],
            'total_points': model['total_points'],
            'azimuth_rms': model['azimuth_rms'],
            'range_rms': model['range_rms'],
            'gmtsar_filter': gmtsar_filter,
            'quadratic_screen': quadratic_screen,
        },
        'config': {
            key: cfg[key]
            for key in sorted(cfg.keys())
        },
        'geo2rdr_valid_guard': {
            'enabled': bool(geo2rdr_guard is not None),
            'height': int(geo2rdr_guard['height']) if geo2rdr_guard is not None else 0,
            'width': int(geo2rdr_guard['width']) if geo2rdr_guard is not None else 0,
            'nodata': float(geo2rdr_guard['nodata']) if geo2rdr_guard is not None else None,
            'azimuth_offset_path': str(geo2rdr_guard['az_path']) if geo2rdr_guard is not None else None,
            'range_offset_path': str(geo2rdr_guard['rg_path']) if geo2rdr_guard is not None else None,
        },
    }

    return azpoly, rgpoly, meta
