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
    'coarse_window': 256,
    'coarse_multiscale': True,
    'coarse_window_factors': [0.5, 1.0, 2.0],
    'coarse_search_ranges': [64, 128, 256, 512, 1024],
    'coarse_window_scale': 4.0,
    'coarse_consistency_priority': True,
    'coarse_correlation_threshold': 0.06,
    'coarse_min_window': 96,
    'coarse_max_window': 4096,
    'coarse_grid_size': 3,
    'coarse_min_valid': 9,
    'coarse_prefer_larger_window': True,
    'coarse_log_candidates': True,
    'coarse_quality_threshold': 0.06,
    'fine_window': 128,
    'fine_spacing': 128,
    'fine_quality_threshold': 0.05,
    'fine_workers': 0,
    'fine_chunk_size': 128,
    'precompute_amplitude': True,
    'max_points': 0,
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


def _normalize_slc_path(path):
    if path.endswith('.xml'):
        return path[:-4]
    return path


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


def _extract_patch(arr, center_row, center_col, window):
    half = int(window // 2)
    crow = int(np.rint(center_row))
    ccol = int(np.rint(center_col))
    r0 = crow - half
    c0 = ccol - half
    r1 = r0 + int(window)
    c1 = c0 + int(window)

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


def _normalized_corr_centered(a, b):
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.abs(np.sum(a * b)) / denom)


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
        return float(cfg.get('coarse_quality_threshold', 0.06))
    try:
        return float(val)
    except Exception:
        return float(cfg.get('coarse_quality_threshold', 0.06))


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

    grid_size = max(3, int(cfg.get('coarse_grid_size', 3)))
    fracs = np.linspace(0.25, 0.75, grid_size, dtype=np.float64)
    positions = [(h * fr, w * fc) for fr in fracs for fc in fracs]

    az = []
    rg = []
    qualities = []
    quality_threshold = float(cfg['coarse_quality_threshold'])
    for row, col in positions:
        pm = master_amp.extract_patch(row, col, window)
        ps = slave_amp.extract_patch(row, col, window)
        if (pm is None) or (ps is None):
            continue
        result = _phase_correlation_displacement(pm, ps)
        if result is None:
            continue
        daz, drg, q = result
        if q < quality_threshold:
            continue
        if max(abs(float(daz)), abs(float(drg))) > float(search_range):
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
            if r.get('status') == 'ok':
                logger.info(
                    'External coarse candidate: search_range=%d window=%d valid=%d quality_median=%.5f spread=%.6f',
                    int(r.get('search_range', 0)),
                    int(r.get('window', 0)),
                    int(r.get('num_valid', 0)),
                    float(r.get('quality_median', 0.0)),
                    float(r.get('spread', 0.0)),
                )
            else:
                logger.info(
                    'External coarse candidate: search_range=%d window=%d failed (%s)',
                    int(r.get('search_range', 0)),
                    int(r.get('window', 0)),
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
            best = min(
                qualified,
                key=lambda r: (
                    float(r.get('spread', np.inf)),
                    -int(r.get('num_valid', 0)),
                    -float(r.get('quality_median', 0.0)),
                    -int(r.get('window', 0)) if prefer_larger else 0,
                ),
            )
            selection_mode = 'consistency_first'
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
        logger.info(
            'External coarse selected: mode=%s search_range=%d window=%d valid=%d quality_median=%.5f spread=%.6f az=%.4f rg=%.4f',
            selection_mode,
            int(best.get('search_range', 0)),
            int(best.get('window', 0)),
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

    return {
        'azimuth_offset': float(best['azimuth_offset']),
        'range_offset': float(best['range_offset']),
        'initial_offset': (
            float(best['azimuth_offset']),
            float(best['range_offset']),
        ),
        'search_range': int(best.get('search_range', 0)),
        'requested_search_range': int(best.get('requested_search_range', best.get('search_range', 0))),
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


def _evaluate_fine_point(master_amp, slave_amp, row, col, coarse_az, coarse_rg, window, quality_threshold):
    pm = master_amp.extract_patch(row, col, window)
    ps = slave_amp.extract_patch(row + coarse_az, col + coarse_rg, window)
    if (pm is None) or (ps is None):
        return None

    result = _phase_correlation_displacement(pm, ps)
    if result is None:
        return None

    residual_az, residual_rg, quality = result
    if quality < quality_threshold:
        return None

    return {
        'row': float(row),
        'col': float(col),
        'azimuth': float(coarse_az + residual_az),
        'range': float(coarse_rg + residual_rg),
        'quality': float(quality),
    }


def _fine_registration(master_amp, slave_amp, coarse_az, coarse_rg, cfg, logger=None):
    h, w = master_amp.shape
    window = int(cfg['fine_window'])
    if window % 2 != 0:
        window += 1

    spacing = max(16, int(cfg['fine_spacing']))
    margin = (window // 2) + max(abs(int(np.rint(coarse_az))), abs(int(np.rint(coarse_rg)))) + 2
    max_points = int(cfg.get('max_points', 0))
    unlimited_points = (max_points <= 0)
    quality_threshold = float(cfg['fine_quality_threshold'])
    chunk_size = max(8, int(cfg.get('fine_chunk_size', 128)))
    workers_cfg = int(cfg.get('fine_workers', 0))
    if workers_cfg <= 0:
        workers = max(1, min(os.cpu_count() or 1, 8))
    else:
        workers = max(1, workers_cfg)

    grid_points = list(_iter_grid_points(h, w, spacing, margin))
    points = []

    if (logger is not None) and (workers > 1):
        logger.info(
            'External fine registration parallel mode: workers=%d, chunk_size=%d, candidate_points=%d',
            workers,
            chunk_size,
            len(grid_points),
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
                window,
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
                    repeat(window),
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

    coarse = _coarse_registration(master_amp, slave_amp, cfg, logger=logger)
    points = _fine_registration(
        master_amp,
        slave_amp,
        coarse_az=coarse['azimuth_offset'],
        coarse_rg=coarse['range_offset'],
        cfg=cfg,
        logger=logger,
    )
    model = _fit_model(points, cfg)

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
        'fit': {
            'inliers': model['inliers'],
            'total_points': model['total_points'],
            'azimuth_rms': model['azimuth_rms'],
            'range_rms': model['range_rms'],
        },
        'config': {
            key: cfg[key]
            for key in sorted(cfg.keys())
        },
    }

    return azpoly, rgpoly, meta
