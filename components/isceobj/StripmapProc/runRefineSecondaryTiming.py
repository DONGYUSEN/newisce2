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
import subprocess
import time

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


def _safe_positive_int_env(name, default):
    val = _safe_int_env(name, default)
    if val <= 0:
        return int(default)
    return int(val)


def _safe_window_list_env(name, default_values):
    vals = _safe_int_list_env(name, default_values)
    out = []
    seen = set()
    for val in vals:
        ival = int(val)
        if ival <= 0:
            continue
        if ival in seen:
            continue
        seen.add(ival)
        out.append(ival)
    if not out:
        return [int(v) for v in default_values]
    out.sort()
    return out


def _query_gpu_memory_mb(device_id):
    """
    Query GPU memory from nvidia-smi.
    Returns dict(total_mb=..., free_mb=...) with None values when unavailable.
    """
    try:
        proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=3.0,
        )
    except Exception:
        return {'total_mb': None, 'free_mb': None}

    if proc.returncode != 0:
        return {'total_mb': None, 'free_mb': None}

    records = []
    for line in proc.stdout.splitlines():
        parts = [token.strip() for token in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            total_mb = int(parts[1])
            free_mb = int(parts[2])
        except Exception:
            continue
        records.append((idx, total_mb, free_mb))

    if not records:
        return {'total_mb': None, 'free_mb': None}

    target = None
    for rec in records:
        if rec[0] == int(device_id):
            target = rec
            break
    if target is None:
        target = records[0]

    return {'total_mb': int(target[1]), 'free_mb': int(target[2])}


def _resolve_gpu_mmap_gb(default_value=16):
    env_default = _safe_positive_int_env('ISCE_GPU_AMPCOR_MMAP_GB', default_value)
    try:
        phys_pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        total_gb = int((float(phys_pages) * float(page_size)) / (1024.0 ** 3))
    except Exception:
        total_gb = None

    if total_gb is None:
        return int(env_default)

    upper = max(1, total_gb // 4)
    return int(max(1, min(int(env_default), int(upper))))


def _recommend_gpu_streams(window_size, free_mb):
    n_streams = _safe_positive_int_env('ISCE_GPU_AMPCOR_NSTREAMS', 2)
    if free_mb is None:
        return n_streams

    if (int(window_size) >= 512 and int(free_mb) < 16384) or (int(free_mb) < 8192):
        return 1
    return max(1, int(n_streams))


def _recommend_gpu_chunks(window_size, num_across, num_down, free_mb, n_streams):
    override_down = _safe_int_env('ISCE_GPU_AMPCOR_CHUNK_DOWN', 0)
    override_across = _safe_int_env('ISCE_GPU_AMPCOR_CHUNK_ACROSS', 0)

    if int(window_size) <= 128:
        base_across = 64
    elif int(window_size) <= 256:
        base_across = 24
    elif int(window_size) <= 512:
        base_across = 8
    else:
        base_across = 2

    if free_mb is not None:
        scale = float(free_mb) / 8192.0
        scale = max(0.25, min(2.0, scale))
        base_across = int(max(1, round(base_across * scale)))

    if int(n_streams) > 1:
        base_across = max(1, int(base_across // int(n_streams)))

    chunk_down = 1
    if free_mb is not None and int(window_size) <= 256 and int(free_mb) >= 24576:
        chunk_down = min(2, int(num_down))

    chunk_across = max(1, min(int(num_across), int(base_across)))

    if override_down > 0:
        chunk_down = max(1, min(int(num_down), int(override_down)))
    if override_across > 0:
        chunk_across = max(1, min(int(num_across), int(override_across)))

    return chunk_down, chunk_across


def _coerce_hw_pair(value, default=None, minimum=1):
    if value is None:
        if default is None:
            raise ValueError('Missing template/search value')
        value = default

    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    if len(items) == 0:
        raise ValueError('Empty template/search value')
    if len(items) == 1:
        one = max(int(minimum), int(np.rint(float(items[0]))))
        return one, one

    first = max(int(minimum), int(np.rint(float(items[0]))))
    second = max(int(minimum), int(np.rint(float(items[1]))))
    return first, second


def _safe_hw_pair_env(name, default, minimum=1):
    raw = os.environ.get(name)
    if raw is None:
        return _coerce_hw_pair(default, minimum=minimum)

    text = str(raw).strip().lower()
    if not text:
        return _coerce_hw_pair(default, minimum=minimum)

    for sep in ('x', ',', ';', '/', ':'):
        text = text.replace(sep, ' ')
    parts = [token for token in text.split() if token]

    try:
        if len(parts) >= 2:
            return _coerce_hw_pair((parts[0], parts[1]), minimum=minimum)
        if len(parts) == 1:
            return _coerce_hw_pair(parts[0], minimum=minimum)
    except Exception:
        pass

    return _coerce_hw_pair(default, minimum=minimum)


def _shape_label(pair):
    hh, ww = _coerce_hw_pair(pair, minimum=0)
    return '{0}x{1}'.format(int(hh), int(ww))


def _parse_template_search_token(token):
    text = str(token).strip().lower()
    if not text:
        return None

    if ':' in text:
        template_txt, search_txt = text.split(':', 1)
    elif '/' in text:
        template_txt, search_txt = text.split('/', 1)
    else:
        return None

    def _parse_shape(spec):
        spec = str(spec).strip().lower()
        if not spec:
            return None
        if 'x' in spec:
            parts = spec.split('x', 1)
            return _coerce_hw_pair((parts[0], parts[1]), minimum=1)
        one = max(1, int(np.rint(float(spec))))
        return (one, one)

    template_window = _parse_shape(template_txt)
    search_half = _parse_shape(search_txt)
    if (template_window is None) or (search_half is None):
        return None
    return {
        'template_window': template_window,
        'search_half': search_half,
    }


def _legacy_template_search_candidates(default_pairs):
    raw = os.environ.get('ISCE_GPU_AMPCOR_TEMPLATE_SEARCH_CANDIDATES')
    if raw:
        candidates = []
        seen = set()
        text = str(raw).replace(',', ' ').replace(';', ' ')
        for token in text.split():
            one = _parse_template_search_token(token)
            if one is None:
                continue
            key = tuple(one['template_window']) + tuple(one['search_half'])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(one)
        if candidates:
            return candidates

    default_candidates = []
    default_seen = set()
    for template_window, search_half in list(default_pairs):
        one = {
            'template_window': _coerce_hw_pair(template_window, minimum=1),
            'search_half': _coerce_hw_pair(search_half, minimum=1),
        }
        key = tuple(one['template_window']) + tuple(one['search_half'])
        if key in default_seen:
            continue
        default_seen.add(key)
        default_candidates.append(one)

    has_window_override = os.environ.get('ISCE_GPU_AMPCOR_WINDOW_CANDIDATES') is not None
    windows = _safe_window_list_env(
        'ISCE_GPU_AMPCOR_WINDOW_CANDIDATES',
        [pair[0][0] for pair in default_pairs],
    )
    has_fixed_search = os.environ.get('ISCE_GPU_AMPCOR_SEARCH_HALF') is not None
    has_search_ratio_override = os.environ.get('ISCE_GPU_AMPCOR_SEARCH_TO_WINDOW_RATIO') is not None
    if not (has_window_override or has_fixed_search or has_search_ratio_override):
        return default_candidates

    search_ratio = max(0.05, _safe_float_env('ISCE_GPU_AMPCOR_SEARCH_TO_WINDOW_RATIO', 0.5))
    candidates = []
    seen = set()
    for win in windows:
        default_search = max(4, int(np.rint(float(win) * search_ratio)))
        if has_fixed_search:
            search_half = _safe_hw_pair_env(
                'ISCE_GPU_AMPCOR_SEARCH_HALF',
                (default_search, default_search),
                minimum=4,
            )
        else:
            search_half = (default_search, default_search)
        one = {
            'template_window': (int(win), int(win)),
            'search_half': tuple(search_half),
        }
        key = tuple(one['template_window']) + tuple(one['search_half'])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(one)
    return candidates


def _cleanup_gpu_ampcor_outputs(out_prefix):
    for suffix in ('.bip', '.gross', '_snr.bip', '_cov.bip'):
        path = out_prefix + suffix
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _offset_field_arrays(field):
    offsets = list(getattr(field, '_offsets', []) or [])
    if len(offsets) == 0:
        return None

    across = []
    down = []
    rg = []
    az = []
    snr = []
    for one in offsets:
        x, y = one.getCoordinate()
        dx, dy = one.getOffset()
        across.append(float(x))
        down.append(float(y))
        rg.append(float(dx))
        az.append(float(dy))
        snr.append(float(one.getSignalToNoise()))

    return {
        'across': np.array(across, dtype=np.float64),
        'down': np.array(down, dtype=np.float64),
        'rg': np.array(rg, dtype=np.float64),
        'az': np.array(az, dtype=np.float64),
        'snr': np.array(snr, dtype=np.float64),
    }


def _linear_fit_rms(across, down, values):
    if values.size <= 1:
        return 0.0

    a0 = np.mean(across)
    d0 = np.mean(down)
    ascale = np.std(across)
    dscale = np.std(down)
    ascale = ascale if ascale > 1.0e-6 else 1.0
    dscale = dscale if dscale > 1.0e-6 else 1.0
    aa = (across - a0) / ascale
    dd = (down - d0) / dscale
    design = np.column_stack([np.ones_like(aa), aa, dd])
    coeff, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    pred = design @ coeff
    return float(np.sqrt(np.mean((values - pred) ** 2)))


def _evaluate_probe_field(field, snr_threshold):
    arrays = _offset_field_arrays(field)
    if arrays is None:
        return {
            'valid_count': 0,
            'valid_ratio': 0.0,
            'mean_snr': 0.0,
            'peak_snr': 0.0,
            'az_std': np.inf,
            'rg_std': np.inf,
            'spread': np.inf,
            'az_fit_rms': np.inf,
            'rg_fit_rms': np.inf,
            'fit_rms': np.inf,
            'az_median': 0.0,
            'rg_median': 0.0,
            'quality': 0.0,
        }

    valid = (
        np.isfinite(arrays['snr'])
        & np.isfinite(arrays['az'])
        & np.isfinite(arrays['rg'])
        & (arrays['snr'] >= float(snr_threshold))
    )
    valid_count = int(np.sum(valid))
    total = int(arrays['snr'].size)
    valid_ratio = float(valid_count) / float(max(1, total))
    if valid_count <= 0:
        return {
            'valid_count': valid_count,
            'valid_ratio': valid_ratio,
            'mean_snr': 0.0,
            'peak_snr': 0.0,
            'az_std': np.inf,
            'rg_std': np.inf,
            'spread': np.inf,
            'az_fit_rms': np.inf,
            'rg_fit_rms': np.inf,
            'fit_rms': np.inf,
            'az_median': 0.0,
            'rg_median': 0.0,
            'quality': 0.0,
        }

    az_valid = arrays['az'][valid]
    rg_valid = arrays['rg'][valid]
    snr_valid = arrays['snr'][valid]
    across_valid = arrays['across'][valid]
    down_valid = arrays['down'][valid]

    az_fit_rms = _linear_fit_rms(across_valid, down_valid, az_valid)
    rg_fit_rms = _linear_fit_rms(across_valid, down_valid, rg_valid)
    fit_rms = float(np.sqrt((az_fit_rms ** 2) + (rg_fit_rms ** 2)))
    az_std = float(np.std(az_valid))
    rg_std = float(np.std(rg_valid))
    spread = float(np.sqrt((az_std ** 2) + (rg_std ** 2)))
    mean_snr = float(np.mean(snr_valid))
    peak_snr = float(np.max(snr_valid))
    quality = float(valid_ratio * peak_snr / (1.0 + fit_rms))

    return {
        'valid_count': valid_count,
        'valid_ratio': valid_ratio,
        'mean_snr': mean_snr,
        'peak_snr': peak_snr,
        'az_std': az_std,
        'rg_std': rg_std,
        'spread': spread,
        'az_fit_rms': float(az_fit_rms),
        'rg_fit_rms': float(rg_fit_rms),
        'fit_rms': fit_rms,
        'az_median': float(np.median(az_valid)),
        'rg_median': float(np.median(rg_valid)),
        'quality': quality,
    }


def _refined_probe_search_half(search_half):
    shh, sw = _coerce_hw_pair(search_half, minimum=1)
    return (
        max(8, int(np.rint(float(shh) * 0.5))),
        max(8, int(np.rint(float(sw) * 0.5))),
    )


def _run_gpu_ampcor_probe_candidate(
    reference,
    secondary,
    misreg_dir,
    candidate,
    snr_threshold,
):
    template_window = _coerce_hw_pair(candidate['template_window'], minimum=1)
    coarse_search_half = _coerce_hw_pair(candidate['search_half'], minimum=1)
    refine_search_half = _refined_probe_search_half(coarse_search_half)
    coarse_grid = max(3, _safe_positive_int_env('ISCE_GPU_AMPCOR_PROBE_GRID', 3))
    refine_grid = max(coarse_grid, _safe_positive_int_env('ISCE_GPU_AMPCOR_PROBE_REFINE_GRID', 5))
    time_weight = max(0.0, _safe_float_env('ISCE_GPU_AMPCOR_PROBE_TIME_WEIGHT', 0.35))

    coarse_prefix = os.path.join(
        misreg_dir,
        'gpu_ampcor_probe_t{0}_s{1}_coarse'.format(
            _shape_label(template_window),
            _shape_label(coarse_search_half),
        ),
    )
    refine_prefix = os.path.join(
        misreg_dir,
        'gpu_ampcor_probe_t{0}_s{1}_refine'.format(
            _shape_label(template_window),
            _shape_label(refine_search_half),
        ),
    )
    _cleanup_gpu_ampcor_outputs(coarse_prefix)
    _cleanup_gpu_ampcor_outputs(refine_prefix)

    coarse_tic = time.perf_counter()
    coarse_field = estimateOffsetFieldGPU(
        reference,
        secondary,
        coarse_prefix,
        template_window=template_window,
        search_half=coarse_search_half,
        num_locations_across=coarse_grid,
        num_locations_down=coarse_grid,
        n_streams=1,
        chunk_down=1,
        chunk_across=1,
    )
    coarse_elapsed = float(time.perf_counter() - coarse_tic)
    coarse_metrics = _evaluate_probe_field(coarse_field, snr_threshold=float(snr_threshold))
    gross_az = float(coarse_metrics['az_median'])
    gross_rg = float(coarse_metrics['rg_median'])

    refine_tic = time.perf_counter()
    refine_field = estimateOffsetFieldGPU(
        reference,
        secondary,
        refine_prefix,
        azoffset=int(np.rint(gross_az)),
        rgoffset=int(np.rint(gross_rg)),
        template_window=template_window,
        search_half=refine_search_half,
        num_locations_across=refine_grid,
        num_locations_down=refine_grid,
        n_streams=1,
        chunk_down=1,
        chunk_across=1,
    )
    refine_elapsed = float(time.perf_counter() - refine_tic)
    refine_metrics = _evaluate_probe_field(refine_field, snr_threshold=float(snr_threshold))

    total_elapsed = float(coarse_elapsed + refine_elapsed)
    denom = max(total_elapsed, 1.0e-3) ** float(time_weight)
    score = float(refine_metrics['quality'] / denom)

    return {
        'template_window': template_window,
        'coarse_search_half': coarse_search_half,
        'refine_search_half': refine_search_half,
        'gross_azimuth_offset': float(refine_metrics['az_median']),
        'gross_range_offset': float(refine_metrics['rg_median']),
        'coarse_elapsed_sec': coarse_elapsed,
        'refine_elapsed_sec': refine_elapsed,
        'elapsed_sec': total_elapsed,
        'coarse_probe': coarse_metrics,
        'refine_probe': refine_metrics,
        'valid_count': int(refine_metrics['valid_count']),
        'valid_ratio': float(refine_metrics['valid_ratio']),
        'mean_snr': float(refine_metrics['mean_snr']),
        'peak_snr': float(refine_metrics['peak_snr']),
        'fit_rms': float(refine_metrics['fit_rms']),
        'quality': float(refine_metrics['quality']),
        'score': score,
    }


def _select_gpu_ampcor_template_search(
    reference,
    secondary,
    misreg_dir,
    snr_threshold,
):
    auto_enabled = _parse_bool_env('ISCE_GPU_AMPCOR_AUTO_TEMPLATE_SEARCH', True)
    fixed_template = _safe_hw_pair_env(
        'ISCE_GPU_AMPCOR_TEMPLATE_WINDOW',
        (128, 128),
        minimum=16,
    )
    fixed_search = _safe_hw_pair_env(
        'ISCE_GPU_AMPCOR_SEARCH_HALF_RANGE',
        (32, 32),
        minimum=4,
    )
    if not auto_enabled:
        fixed = _run_gpu_ampcor_probe_candidate(
            reference,
            secondary,
            misreg_dir,
            {
                'template_window': tuple(fixed_template),
                'search_half': tuple(fixed_search),
            },
            snr_threshold=snr_threshold,
        )
        logger.info(
            'GPU Ampcor joint auto-selection disabled; using fixed template=%s '
            'coarse_search_half=%s refine_search_half=%s gross=(az=%.3f, rg=%.3f).',
            _shape_label(fixed['template_window']),
            _shape_label(fixed['coarse_search_half']),
            _shape_label(fixed['refine_search_half']),
            float(fixed.get('gross_azimuth_offset', 0.0)),
            float(fixed.get('gross_range_offset', 0.0)),
        )
        return fixed, [dict(fixed, success=True, error=None)]

    default_pairs = [
        ((128, 128), (64, 64)),
        ((256, 256), (128, 128)),
        ((512, 512), (256, 256)),
        ((1024, 1024), (512, 512)),
        ((1024, 1024), (1024, 1024)),
    ]
    candidates = _legacy_template_search_candidates(default_pairs)
    min_valid_ratio = max(
        0.0,
        min(1.0, _safe_float_env('ISCE_GPU_AMPCOR_PROBE_MIN_VALID_RATIO', 0.55)),
    )
    tie_margin = max(0.0, _safe_float_env('ISCE_GPU_AMPCOR_PROBE_SCORE_MARGIN', 0.08))

    metrics = []
    for candidate in candidates:
        one = {
            'template_window': tuple(candidate['template_window']),
            'coarse_search_half': tuple(candidate['search_half']),
            'success': False,
            'error': None,
        }
        try:
            probe = _run_gpu_ampcor_probe_candidate(
                reference,
                secondary,
                misreg_dir,
                candidate,
                snr_threshold=snr_threshold,
            )
            one.update(probe)
            one['success'] = True
        except Exception as err:
            one['error'] = str(err)
            logger.warning(
                'GPU Ampcor joint probe failed for template=%s coarse_search_half=%s: %s',
                _shape_label(candidate['template_window']),
                _shape_label(candidate['search_half']),
                str(err),
            )
        metrics.append(one)

    successful = [row for row in metrics if bool(row.get('success'))]
    if not successful:
        raise RuntimeError('GPU Ampcor joint auto-selection failed for all candidates.')

    for row in successful:
        logger.info(
            'GPU Ampcor joint probe template=%s coarse_search_half=%s refine_search_half=%s: '
            'time=%.3fs valid_ratio=%.3f peak_snr=%.3f fit_rms=%.4f score=%.5f gross=(az=%.3f, rg=%.3f)',
            _shape_label(row['template_window']),
            _shape_label(row['coarse_search_half']),
            _shape_label(row['refine_search_half']),
            float(row['elapsed_sec']),
            float(row['valid_ratio']),
            float(row['peak_snr']),
            float(row['fit_rms']),
            float(row['score']),
            float(row['gross_azimuth_offset']),
            float(row['gross_range_offset']),
        )

    eligible = [row for row in successful if float(row['valid_ratio']) >= float(min_valid_ratio)]
    pool = eligible if eligible else successful
    best_score = max(float(row['score']) for row in pool)
    near_best = [
        row for row in pool
        if float(row['score']) >= (1.0 - float(tie_margin)) * float(best_score)
    ]
    chosen = sorted(
        near_best,
        key=lambda row: (
            float(row['fit_rms']),
            float(row['elapsed_sec']),
            -int(row['template_window'][0]),
            -int(row['coarse_search_half'][0]),
        ),
    )[0]
    logger.info(
        'GPU Ampcor joint auto-selection chose template=%s coarse_search_half=%s refine_search_half=%s '
        '(score=%.5f, fit_rms=%.4f, valid_ratio=%.3f).',
        _shape_label(chosen['template_window']),
        _shape_label(chosen['coarse_search_half']),
        _shape_label(chosen['refine_search_half']),
        float(chosen['score']),
        float(chosen['fit_rms']),
        float(chosen['valid_ratio']),
    )
    return chosen, metrics


def _run_cpu_ampcor_probe_candidate(
    reference,
    secondary,
    candidate,
    snr_threshold,
):
    template_window = _coerce_hw_pair(candidate['template_window'], minimum=1)
    coarse_search_half = _coerce_hw_pair(candidate['search_half'], minimum=1)
    refine_search_half = _refined_probe_search_half(coarse_search_half)
    coarse_grid = max(3, _safe_positive_int_env('ISCE_CPU_AMPCOR_PROBE_GRID', 3))
    refine_grid = max(coarse_grid, _safe_positive_int_env('ISCE_CPU_AMPCOR_PROBE_REFINE_GRID', 5))
    time_weight = max(0.0, _safe_float_env('ISCE_CPU_AMPCOR_PROBE_TIME_WEIGHT', 0.35))

    coarse_tic = time.perf_counter()
    coarse_field = estimateOffsetField(
        reference,
        secondary,
        template_window=template_window,
        search_half=coarse_search_half,
        num_locations_across=coarse_grid,
        num_locations_down=coarse_grid,
    )
    coarse_elapsed = float(time.perf_counter() - coarse_tic)
    coarse_metrics = _evaluate_probe_field(coarse_field, snr_threshold=float(snr_threshold))
    gross_az = float(coarse_metrics['az_median'])
    gross_rg = float(coarse_metrics['rg_median'])

    refine_tic = time.perf_counter()
    refine_field = estimateOffsetField(
        reference,
        secondary,
        azoffset=int(np.rint(gross_az)),
        rgoffset=int(np.rint(gross_rg)),
        template_window=template_window,
        search_half=refine_search_half,
        num_locations_across=refine_grid,
        num_locations_down=refine_grid,
    )
    refine_elapsed = float(time.perf_counter() - refine_tic)
    refine_metrics = _evaluate_probe_field(refine_field, snr_threshold=float(snr_threshold))

    total_elapsed = float(coarse_elapsed + refine_elapsed)
    denom = max(total_elapsed, 1.0e-3) ** float(time_weight)
    score = float(refine_metrics['quality'] / denom)

    return {
        'template_window': template_window,
        'coarse_search_half': coarse_search_half,
        'refine_search_half': refine_search_half,
        'gross_azimuth_offset': float(refine_metrics['az_median']),
        'gross_range_offset': float(refine_metrics['rg_median']),
        'coarse_elapsed_sec': coarse_elapsed,
        'refine_elapsed_sec': refine_elapsed,
        'elapsed_sec': total_elapsed,
        'coarse_probe': coarse_metrics,
        'refine_probe': refine_metrics,
        'valid_count': int(refine_metrics['valid_count']),
        'valid_ratio': float(refine_metrics['valid_ratio']),
        'mean_snr': float(refine_metrics['mean_snr']),
        'peak_snr': float(refine_metrics['peak_snr']),
        'fit_rms': float(refine_metrics['fit_rms']),
        'quality': float(refine_metrics['quality']),
        'score': score,
    }


def _select_cpu_ampcor_template_search(
    reference,
    secondary,
    snr_threshold,
):
    auto_enabled = _parse_bool_env('ISCE_CPU_AMPCOR_AUTO_TEMPLATE_SEARCH', True)
    fixed_template = _safe_hw_pair_env(
        'ISCE_CPU_AMPCOR_TEMPLATE_WINDOW',
        (128, 128),
        minimum=16,
    )
    fixed_search = _safe_hw_pair_env(
        'ISCE_CPU_AMPCOR_SEARCH_HALF_RANGE',
        (32, 32),
        minimum=4,
    )
    if not auto_enabled:
        fixed = _run_cpu_ampcor_probe_candidate(
            reference,
            secondary,
            {
                'template_window': tuple(fixed_template),
                'search_half': tuple(fixed_search),
            },
            snr_threshold=snr_threshold,
        )
        logger.info(
            'CPU Ampcor joint auto-selection disabled; using fixed template=%s '
            'coarse_search_half=%s refine_search_half=%s gross=(az=%.3f, rg=%.3f).',
            _shape_label(fixed['template_window']),
            _shape_label(fixed['coarse_search_half']),
            _shape_label(fixed['refine_search_half']),
            float(fixed.get('gross_azimuth_offset', 0.0)),
            float(fixed.get('gross_range_offset', 0.0)),
        )
        return fixed, [dict(fixed, success=True, error=None)]

    default_pairs = [
        ((128, 128), (64, 64)),
        ((256, 256), (128, 128)),
        ((512, 512), (256, 256)),
        ((1024, 1024), (512, 512)),
        ((1024, 1024), (1024, 1024)),
    ]
    candidates = _legacy_template_search_candidates(default_pairs)
    min_valid_ratio = max(
        0.0,
        min(1.0, _safe_float_env('ISCE_CPU_AMPCOR_PROBE_MIN_VALID_RATIO', 0.55)),
    )
    tie_margin = max(0.0, _safe_float_env('ISCE_CPU_AMPCOR_PROBE_SCORE_MARGIN', 0.08))

    metrics = []
    for candidate in candidates:
        one = {
            'template_window': tuple(candidate['template_window']),
            'coarse_search_half': tuple(candidate['search_half']),
            'success': False,
            'error': None,
        }
        try:
            probe = _run_cpu_ampcor_probe_candidate(
                reference,
                secondary,
                candidate,
                snr_threshold=snr_threshold,
            )
            one.update(probe)
            one['success'] = True
        except Exception as err:
            one['error'] = str(err)
            logger.warning(
                'CPU Ampcor joint probe failed for template=%s coarse_search_half=%s: %s',
                _shape_label(candidate['template_window']),
                _shape_label(candidate['search_half']),
                str(err),
            )
        metrics.append(one)

    successful = [row for row in metrics if bool(row.get('success'))]
    if not successful:
        raise RuntimeError('CPU Ampcor joint auto-selection failed for all candidates.')

    for row in successful:
        logger.info(
            'CPU Ampcor joint probe template=%s coarse_search_half=%s refine_search_half=%s: '
            'time=%.3fs valid_ratio=%.3f peak_snr=%.3f fit_rms=%.4f score=%.5f gross=(az=%.3f, rg=%.3f)',
            _shape_label(row['template_window']),
            _shape_label(row['coarse_search_half']),
            _shape_label(row['refine_search_half']),
            float(row['elapsed_sec']),
            float(row['valid_ratio']),
            float(row['peak_snr']),
            float(row['fit_rms']),
            float(row['score']),
            float(row['gross_azimuth_offset']),
            float(row['gross_range_offset']),
        )

    eligible = [row for row in successful if float(row['valid_ratio']) >= float(min_valid_ratio)]
    pool = eligible if eligible else successful
    best_score = max(float(row['score']) for row in pool)
    near_best = [
        row for row in pool
        if float(row['score']) >= (1.0 - float(tie_margin)) * float(best_score)
    ]
    chosen = sorted(
        near_best,
        key=lambda row: (
            float(row['fit_rms']),
            float(row['elapsed_sec']),
            -int(row['template_window'][0]),
            -int(row['coarse_search_half'][0]),
        ),
    )[0]
    logger.info(
        'CPU Ampcor joint auto-selection chose template=%s coarse_search_half=%s refine_search_half=%s '
        '(score=%.5f, fit_rms=%.4f, valid_ratio=%.3f).',
        _shape_label(chosen['template_window']),
        _shape_label(chosen['coarse_search_half']),
        _shape_label(chosen['refine_search_half']),
        float(chosen['score']),
        float(chosen['fit_rms']),
        float(chosen['valid_ratio']),
    )
    return chosen, metrics


def _run_gpu_ampcor_with_retry(reference, secondary, out_prefix, **kwargs):
    device_id = int(kwargs.get('device_id', 0))
    template_window = _coerce_hw_pair(kwargs.get('template_window', 128), minimum=1)
    gpu_mem = _query_gpu_memory_mb(device_id)
    free_mb = gpu_mem.get('free_mb')
    base_streams = kwargs.get('n_streams')
    if base_streams is None:
        base_streams = _recommend_gpu_streams(max(template_window), free_mb)
    attempts = []
    retry_plan = [
        {
            'n_streams': max(1, int(base_streams)),
            'chunk_down': kwargs.get('chunk_down'),
            'chunk_across': kwargs.get('chunk_across'),
        },
        {
            'n_streams': max(1, int(base_streams)),
            'chunk_down': 1,
            'chunk_across': 1,
        },
        {
            'n_streams': 1,
            'chunk_down': 1,
            'chunk_across': 1,
        },
    ]

    seen = set()
    for plan in retry_plan:
        key = (
            int(plan['n_streams']),
            None if plan['chunk_down'] is None else int(plan['chunk_down']),
            None if plan['chunk_across'] is None else int(plan['chunk_across']),
        )
        if key in seen:
            continue
        seen.add(key)
        attempt_kwargs = dict(kwargs)
        attempt_kwargs['n_streams'] = plan['n_streams']
        attempt_kwargs['chunk_down'] = plan['chunk_down']
        attempt_kwargs['chunk_across'] = plan['chunk_across']
        _cleanup_gpu_ampcor_outputs(out_prefix)
        try:
            return estimateOffsetFieldGPU(
                reference,
                secondary,
                out_prefix,
                **attempt_kwargs
            )
        except Exception as err:
            attempts.append(
                {
                    'streams': int(plan['n_streams']),
                    'chunk_down': None if plan['chunk_down'] is None else int(plan['chunk_down']),
                    'chunk_across': None if plan['chunk_across'] is None else int(plan['chunk_across']),
                    'error': str(err),
                }
            )
            message = str(err)
            if ('Invalid GPU Ampcor ROI' in message) or ('Invalid GPU Ampcor skip spacing' in message):
                raise
            logger.warning(
                'GPU Ampcor retry candidate failed with streams=%d chunk_down=%s chunk_across=%s: %s',
                int(plan['n_streams']),
                str(plan['chunk_down']),
                str(plan['chunk_across']),
                str(err),
            )

    raise RuntimeError('GPU Ampcor failed after retries: {0}'.format(attempts))


def _integrated_external_enabled():
    return _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_ENABLED', True)


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
    coarse_template_env = (
        'ISCE_EXTERNAL_REGISTRATION_COARSE_TEMPLATE_WINDOW'
        if os.environ.get('ISCE_EXTERNAL_REGISTRATION_COARSE_TEMPLATE_WINDOW') is not None
        else 'ISCE_EXTERNAL_REGISTRATION_WINDOW_SIZE'
    )
    coarse_search_env = (
        'ISCE_EXTERNAL_REGISTRATION_COARSE_SEARCH_HALF_RANGES'
        if os.environ.get('ISCE_EXTERNAL_REGISTRATION_COARSE_SEARCH_HALF_RANGES') is not None
        else 'ISCE_EXTERNAL_REGISTRATION_COARSE_SEARCH_RANGES'
    )
    fine_template_env = (
        'ISCE_EXTERNAL_REGISTRATION_FINE_TEMPLATE_WINDOW'
        if os.environ.get('ISCE_EXTERNAL_REGISTRATION_FINE_TEMPLATE_WINDOW') is not None
        else 'ISCE_EXTERNAL_REGISTRATION_FINE_WINDOW'
    )
    fine_stride_env = (
        'ISCE_EXTERNAL_REGISTRATION_FINE_SAMPLING_STRIDE'
        if os.environ.get('ISCE_EXTERNAL_REGISTRATION_FINE_SAMPLING_STRIDE') is not None
        else 'ISCE_EXTERNAL_REGISTRATION_FINE_SPACING'
    )
    max_points_cap = _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MAX_POINTS_CAP', 120 * 120)
    if max_points_cap <= 0:
        max_points_cap = 120 * 120
    max_points_cfg = _safe_int_env('ISCE_EXTERNAL_REGISTRATION_MAX_POINTS', max_points_cap)
    if max_points_cfg <= 0:
        max_points_cfg = max_points_cap
    max_points_cfg = min(int(max_points_cfg), int(max_points_cap))

    cfg = {
        'staged_enable': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_STAGED_ENABLE', True),
        'stage1_disable': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_DISABLE', True),
        'stage1_source': os.environ.get('ISCE_EXTERNAL_REGISTRATION_STAGE1_SOURCE', 'geo2rdr_mean'),
        'stage1_require_geo2rdr': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_REQUIRE_GEO2RDR', True),
        'stage1_geo2rdr_nodata': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_GEO2RDR_NODATA', -999999.0),
        'use_geo2rdr_valid_mask': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_GEO2RDR_VALID_MASK_ENABLE',
            True,
        ),
        'resample_mode': os.environ.get(
            'ISCE_EXTERNAL_REGISTRATION_RESAMPLE_MODE',
            'geo2rdr_plus_misreg',
        ),
        'stage1_window': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_WINDOW', 2048),
        'stage1_search_half': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_SEARCH_HALF', 1024),
        'stage1_grid_size': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_GRID_SIZE', 4),
        'stage1_quality_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_QUALITY', 0.05),
        'stage1_min_valid': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_MIN_VALID', 4),
        'stage1_outlier_sigma': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_STAGE1_OUTLIER_SIGMA', 2.5),
        'stage1_outlier_max_iterations': _safe_int_env(
            'ISCE_EXTERNAL_REGISTRATION_STAGE1_OUTLIER_MAX_ITERATIONS',
            8,
        ),
        'stage2_windows': _safe_int_list_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_WINDOWS', [256, 512, 1024]),
        'stage2_search_half_scale': _safe_float_env(
            'ISCE_EXTERNAL_REGISTRATION_STAGE2_SEARCH_HALF_SCALE',
            0.5,
        ),
        'stage2_search_half_min': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_SEARCH_HALF_MIN', 16),
        'stage2_search_half_max': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_SEARCH_HALF_MAX', 1024),
        'stage2_grid_size': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_GRID_SIZE', 4),
        'stage2_quality_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_QUALITY', 0.05),
        'stage2_min_valid': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_STAGE2_MIN_VALID', 4),
        'stage2_prefer_larger_window': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_STAGE2_PREFER_LARGER_WINDOW',
            True,
        ),
        'coarse_window': _safe_int_env(coarse_template_env, 256),
        'coarse_multiscale': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_COARSE_MULTISCALE', True),
        'coarse_window_factors': _safe_float_list_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_WINDOW_FACTORS',
            [0.5, 1.0, 2.0],
        ),
        'coarse_search_ranges': _safe_int_list_env(
            coarse_search_env,
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
        'coarse_pass_zero_offset_to_fine': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_COARSE_ZERO_OFFSET_FOR_FINE',
            False,
        ),
        'fine_window': _safe_int_env(fine_template_env, 128),
        'fine_force_search_half_from_window': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_FINE_FORCE_SEARCH_HALF_FROM_WINDOW',
            True,
        ),
        'fine_force_search_half_to_half_window': _parse_bool_env(
            'ISCE_EXTERNAL_REGISTRATION_FINE_FORCE_SEARCH_HALF_TO_HALF_WINDOW',
            True,
        ),
        'fine_force_search_half_scale': _safe_float_env(
            'ISCE_EXTERNAL_REGISTRATION_FINE_FORCE_SEARCH_HALF_SCALE',
            0.25,
        ),
        'fine_search_scale': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_FINE_SEARCH_SCALE', 0.25),
        'fine_search_min': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_SEARCH_MIN', 16),
        'fine_search_max': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_SEARCH_MAX', 128),
        'fine_spacing': _safe_int_env(fine_stride_env, 128),
        'fine_quality_threshold': _safe_float_env('ISCE_EXTERNAL_REGISTRATION_FINE_QUALITY', 0.05),
        'fine_workers': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_WORKERS', 0),
        'fine_chunk_size': _safe_int_env('ISCE_EXTERNAL_REGISTRATION_FINE_CHUNK_SIZE', 128),
        'precompute_amplitude': _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_PRECOMPUTE_AMP', True),
        'max_points': int(max_points_cfg),
        'max_points_cap': int(max_points_cap),
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


def estimateOffsetField(
    reference,
    secondary,
    azoffset=0,
    rgoffset=0,
    template_window=128,
    search_half=40,
    num_locations_across=60,
    num_locations_down=60,
):
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
    template_h, template_w = _coerce_hw_pair(template_window, minimum=16)
    search_h, search_w = _coerce_hw_pair(search_half, minimum=4)
    ww = int(template_w)
    wh = int(template_h)
    sw = int(search_w)
    shh = int(search_h)
    margin_across = 2 * sw + ww
    margin_down = 2 * shh + wh
    nAcross = max(2, int(num_locations_across))
    nDown = max(2, int(num_locations_down))

    objOffset.setAcrossGrossOffset(int(rgoffset))
    objOffset.setDownGrossOffset(int(azoffset))
    objOffset.setWindowSizeWidth(ww)
    objOffset.setWindowSizeHeight(wh)
    objOffset.setSearchWindowSizeWidth(sw)
    objOffset.setSearchWindowSizeHeight(shh)

    offAc = max(101, -int(rgoffset)) + margin_across
    offDn = max(101, -int(azoffset)) + margin_down

    lastAc = int(min(width, sim.getWidth() - offAc) - margin_across)
    lastDn = int(min(length, sim.getLength() - offDn) - margin_down)

    if (lastAc <= offAc) or (lastDn <= offDn):
        sar.finalizeImage()
        sim.finalizeImage()
        raise ValueError(
            'Invalid CPU Ampcor ROI for refine timing: '
            'offAc={0}, lastAc={1}, offDn={2}, lastDn={3}'.format(
                offAc, lastAc, offDn, lastDn
            )
        )

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


def estimateOffsetFieldGPU(
    reference,
    secondary,
    outPrefix,
    azoffset=0,
    rgoffset=0,
    template_window=128,
    search_half=40,
    num_locations_across=60,
    num_locations_down=60,
    device_id=0,
    n_streams=None,
    chunk_down=None,
    chunk_across=None,
    mmap_gb=None,
):
    '''
    Estimate offset field between reference and secondary SLC using GPU PyCuAmpcor.
    `template_window` and `search_half` use (down, across) semantics.
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

    template_h, template_w = _coerce_hw_pair(template_window, minimum=16)
    search_h, search_w = _coerce_hw_pair(search_half, minimum=4)
    ww = int(template_w)
    wh = int(template_h)
    sw = int(search_w)
    shh = int(search_h)
    margin_across = 2 * sw + ww
    margin_down = 2 * shh + wh
    nAcross = max(2, int(num_locations_across))
    nDown = max(2, int(num_locations_down))

    offAc = max(101, -int(rgoffset)) + margin_across
    offDn = max(101, -int(azoffset)) + margin_down
    lastAc = int(min(width, sim_width - offAc) - margin_across)
    lastDn = int(min(length, sim_length - offDn) - margin_down)

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

    gpu_mem = _query_gpu_memory_mb(device_id)
    free_mb = gpu_mem.get('free_mb')
    total_mb = gpu_mem.get('total_mb')
    if n_streams is None:
        n_streams = _recommend_gpu_streams(ww, free_mb)
    if chunk_down is None or chunk_across is None:
        rec_down, rec_across = _recommend_gpu_chunks(
            window_size=max(ww, wh),
            num_across=num_across,
            num_down=num_down,
            free_mb=free_mb,
            n_streams=n_streams,
        )
        if chunk_down is None:
            chunk_down = rec_down
        if chunk_across is None:
            chunk_across = rec_across
    if mmap_gb is None:
        mmap_gb = _resolve_gpu_mmap_gb(default_value=16)

    objOffset.deviceID = int(device_id)
    objOffset.nStreams = max(1, int(n_streams))
    objOffset.numberWindowDownInChunk = max(1, min(int(num_down), int(chunk_down)))
    objOffset.numberWindowAcrossInChunk = max(1, min(int(num_across), int(chunk_across)))
    objOffset.mmapSize = max(1, int(mmap_gb))

    objOffset.offsetImageName = outPrefix + '.bip'
    objOffset.grossOffsetImageName = outPrefix + '.gross'
    objOffset.snrImageName = outPrefix + '_snr.bip'
    objOffset.covImageName = outPrefix + '_cov.bip'
    objOffset.mergeGrossOffset = 1

    logger.info(
        'GPU Ampcor config: template_window=(%d,%d), search_half=(%d,%d), '
        'sampling_stride=(%d,%d), grid=(%d,%d), start=(%d,%d), '
        'chunk=(%d,%d), streams=%d, device=%d, gpu_mem_mb(total/free)=(%s/%s)',
        wh, ww, shh, sw, skip_down, skip_across, num_down, num_across, offDn, offAc,
        objOffset.numberWindowDownInChunk, objOffset.numberWindowAcrossInChunk,
        objOffset.nStreams, objOffset.deviceID,
        str(total_mb), str(free_mb)
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

    # External registration should directly use secondarySlcCropProduct.
    # If normalization generated *_norm.xml, secondarySlcCropProduct already points to it.
    secondarySlc = secondaryFrame.getImage().filename

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
    #      default useGPU=True -> prefer GPU Ampcor
    #      if GPU unavailable/fails, fallback to CPU Ampcor (joint probe + coarse-to-fine)
    use_gpu_flag = bool(getattr(self, 'useGPU', True))
    external_enabled = bool(getattr(self, 'useExternalCoregistration', False))

    if external_enabled:
        try:
            ext_cfg = _integrated_external_config()
            offsets_dir = self.insar.offsetsDirname
            ext_cfg['stage1_geo2rdr_azimuth_offset'] = os.path.join(
                offsets_dir,
                self.insar.azimuthOffsetFilename,
            )
            ext_cfg['stage1_geo2rdr_range_offset'] = os.path.join(
                offsets_dir,
                self.insar.rangeOffsetFilename,
            )
            external_secondary_slc = secondarySlc
            if _parse_bool_env('ISCE_EXTERNAL_REGISTRATION_USE_COARSE_COREG_FOR_STAGE23', True):
                coarse_coreg = os.path.join(
                    self.insar.coregDirname,
                    self._insar.coarseCoregFilename,
                )
                if os.path.exists(coarse_coreg) and os.path.exists(coarse_coreg + '.xml'):
                    external_secondary_slc = coarse_coreg
                    logger.info(
                        'Integrated external registration stage-2/3 input switched to coarse coreg SLC: %s',
                        external_secondary_slc,
                    )
                else:
                    logger.warning(
                        'Requested coarse_coreg input for external registration stage-2/3, '
                        'but file is missing: %s. Fallback to secondary crop SLC.',
                        coarse_coreg,
                    )
            ext_gates = _integrated_external_quality_gates()
            logger.info('Running integrated external registration with config: %s', ext_cfg)
            logger.info('Integrated external registration quality gates: %s', ext_gates)
            azpoly, rgpoly, ext_meta = estimate_misregistration_polys(
                referenceSlc,
                external_secondary_slc,
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
                selected_gpu_cfg, _probe_metrics = _select_gpu_ampcor_template_search(
                    referenceSlc,
                    secondarySlc,
                    misregDir,
                    snr_threshold=fallback_cfg['snr'],
                )
                field = _run_gpu_ampcor_with_retry(
                    referenceSlc,
                    secondarySlc,
                    gpu_prefix,
                    azoffset=int(np.rint(float(selected_gpu_cfg.get('gross_azimuth_offset', 0.0)))),
                    rgoffset=int(np.rint(float(selected_gpu_cfg.get('gross_range_offset', 0.0)))),
                    template_window=selected_gpu_cfg.get('template_window', (128, 128)),
                    search_half=selected_gpu_cfg.get('refine_search_half', (32, 32)),
                )
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
                logger.info(
                    'GPU Ampcor succeeded; using GPU misregistration polynomials '
                    '(template=%s, coarse_search_half=%s, refine_search_half=%s, gross=(az=%.3f, rg=%.3f)).',
                    _shape_label(selected_gpu_cfg.get('template_window', (128, 128))),
                    _shape_label(selected_gpu_cfg.get('coarse_search_half', (32, 32))),
                    _shape_label(selected_gpu_cfg.get('refine_search_half', (16, 16))),
                    float(selected_gpu_cfg.get('gross_azimuth_offset', 0.0)),
                    float(selected_gpu_cfg.get('gross_range_offset', 0.0)),
                )
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

    logger.info('Running CPU Ampcor for misregistration (joint probe + coarse-to-fine).')
    selected_cpu_cfg, _cpu_probe_metrics = _select_cpu_ampcor_template_search(
        referenceSlc,
        secondarySlc,
        snr_threshold=fallback_cfg['snr'],
    )
    field = estimateOffsetField(
        referenceSlc,
        secondarySlc,
        azoffset=int(np.rint(float(selected_cpu_cfg.get('gross_azimuth_offset', 0.0)))),
        rgoffset=int(np.rint(float(selected_cpu_cfg.get('gross_range_offset', 0.0)))),
        template_window=selected_cpu_cfg.get('template_window', (128, 128)),
        search_half=selected_cpu_cfg.get('refine_search_half', (32, 32)),
    )
    logger.info('CPU Ampcor fit configuration: %s', fallback_cfg)
    _save_ampcor_solution(
        self,
        outShelveFile,
        field,
        fallback_cfg,
        azratio,
        rgratio,
        source='cpu_ampcor_joint',
    )
    logger.info(
        'CPU Ampcor succeeded with joint probe '
        '(template=%s, coarse_search_half=%s, refine_search_half=%s, gross=(az=%.3f, rg=%.3f)).',
        _shape_label(selected_cpu_cfg.get('template_window', (128, 128))),
        _shape_label(selected_cpu_cfg.get('coarse_search_half', (32, 32))),
        _shape_label(selected_cpu_cfg.get('refine_search_half', (16, 16))),
        float(selected_cpu_cfg.get('gross_azimuth_offset', 0.0)),
        float(selected_cpu_cfg.get('gross_range_offset', 0.0)),
    )
    return None
