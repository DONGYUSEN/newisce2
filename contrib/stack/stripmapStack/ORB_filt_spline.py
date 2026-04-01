#!/usr/bin/env python3

"""
Filter orbit state vectors in SLC/aux parameter files using robust spline/polynomial fits.

Typical usage:
  ORB_filt_spline.py in.slc.par.orig out.slc.par --png png_dir --degree 5
  ORB_filt_spline.py in.slc.par.orig out.slc.par --ignore_start 3 --ignore_end 17 --degree 5
"""

import argparse
import os
import re
import warnings

import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
except Exception:
    UnivariateSpline = None

FLOAT_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?')
POS_RE = re.compile(r'^(\s*state_vector_position_(\d+)\s*:\s*)(.*)$', re.IGNORECASE)
VEL_RE = re.compile(r'^(\s*state_vector_velocity_(\d+)\s*:\s*)(.*)$', re.IGNORECASE)


def _to_float(text):
    return float(text.replace('D', 'E').replace('d', 'e'))


def _first_float(text, default=None):
    mm = FLOAT_RE.search(text)
    if mm is None:
        return default
    return _to_float(mm.group(0))


def _first_int(text, default=None):
    mm = FLOAT_RE.search(text)
    if mm is None:
        return default
    return int(round(_to_float(mm.group(0))))


def _triplet_and_tail(payload):
    mm = list(FLOAT_RE.finditer(payload))
    if len(mm) < 3:
        raise ValueError('Expected 3 numeric values in state vector line.')
    vals = np.array([_to_float(mm[i].group(0)) for i in range(3)], dtype=np.float64)
    tail = payload[mm[2].end():].rstrip('\n')
    return vals, tail


def _robust_scale(values):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad > 0:
        return 1.4826 * mad
    std = np.std(vals)
    return float(std) if np.isfinite(std) else 0.0


def _evaluate_fit(t, y, mask, degree):
    idx = np.where(mask)[0]
    if idx.size < 3:
        return y.copy(), 'none'

    deg = max(1, min(int(degree), idx.size - 1, 5))
    tx = t[idx]
    yy = y[idx]

    if UnivariateSpline is not None and tx.size >= deg + 2:
        try:
            txu, iu = np.unique(tx, return_index=True)
            yyu = yy[iu]
            if txu.size >= deg + 2:
                rough = np.diff(yyu)
                noise = _robust_scale(rough) * np.sqrt(2.0) if rough.size > 1 else _robust_scale(yyu)
                if not np.isfinite(noise) or noise <= 0:
                    noise = _robust_scale(yyu - np.median(yyu))
                s_val = float(txu.size) * (noise ** 2) if (np.isfinite(noise) and noise > 0) else 0.0
                sp = UnivariateSpline(txu, yyu, k=deg, s=s_val)
                return sp(t), 'spline'
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coeff = np.polyfit(tx, yy, deg)
    return np.polyval(coeff, t), 'poly'


def _robust_fit_component(t, y, base_mask, degree, sigma, max_iter):
    mask = base_mask.copy()
    fit = y.copy()
    method = 'none'

    for _ in range(max_iter):
        fit, method = _evaluate_fit(t, y, mask, degree)
        resid = y - fit
        scale = _robust_scale(resid[mask])
        if (not np.isfinite(scale)) or (scale <= 0):
            break
        new_mask = base_mask & (np.abs(resid) <= sigma * scale)
        if new_mask.sum() < max(3, int(degree) + 1):
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    fit, method = _evaluate_fit(t, y, mask, degree)
    return fit, mask, method


def _resolve_ignore_counts(nvec, degree, ignore_start, ignore_end):
    igs = int(ignore_start)
    ige = int(ignore_end)

    if igs < 0:
        igs = max(1, int(round(0.02 * nvec))) if nvec >= 20 else 1
    if ige < 0:
        ige = max(1, int(round(0.05 * nvec))) if nvec >= 20 else 1

    igs = max(0, igs)
    ige = max(0, ige)

    min_keep = max(6, int(degree) + 1)
    max_drop = max(0, nvec - min_keep)
    if (igs + ige) > max_drop:
        if max_drop == 0:
            igs, ige = 0, 0
        else:
            total = igs + ige
            if total <= 0:
                igs, ige = 0, 0
            else:
                igs = int(round(max_drop * (igs / float(total))))
                ige = max_drop - igs

    return igs, ige


def filter_state_vectors(t, pos, vel, degree=5, sigma=4.0, max_iter=3, ignore_start=-1, ignore_end=-1):
    nvec = pos.shape[0]
    igs, ige = _resolve_ignore_counts(nvec, degree, ignore_start, ignore_end)
    base_mask = np.ones(nvec, dtype=bool)
    if igs > 0:
        base_mask[:min(igs, nvec)] = False
    if ige > 0:
        base_mask[max(0, nvec - ige):] = False

    if base_mask.sum() < max(6, int(degree) + 1):
        raise RuntimeError('Usable state vectors after ignore_start/end are insufficient for fitting.')

    pos_f = np.zeros_like(pos)
    vel_f = np.zeros_like(vel)
    pos_masks = []
    vel_masks = []
    methods = []

    for comp in range(3):
        fit, mask, method = _robust_fit_component(
            t, pos[:, comp], base_mask, degree, sigma, max_iter
        )
        pos_f[:, comp] = fit
        pos_masks.append(mask)
        methods.append('pos_{}={}'.format(comp, method))

    for comp in range(3):
        fit, mask, method = _robust_fit_component(
            t, vel[:, comp], base_mask, degree, sigma, max_iter
        )
        vel_f[:, comp] = fit
        vel_masks.append(mask)
        methods.append('vel_{}={}'.format(comp, method))

    pos_ok = pos_masks[0] & pos_masks[1] & pos_masks[2]
    vel_ok = vel_masks[0] & vel_masks[1] & vel_masks[2]
    outlier_mask = base_mask & (~pos_ok | ~vel_ok)
    return pos_f, vel_f, outlier_mask, methods, igs, ige


def read_par(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    number_of_sv = None
    dt = None
    pos_map = {}
    vel_map = {}

    for il, line in enumerate(lines):
        lstr = line.strip().lower()
        if lstr.startswith('number_of_state_vectors'):
            number_of_sv = _first_int(line, default=number_of_sv)
        elif lstr.startswith('state_vector_interval'):
            dt = _first_float(line, default=dt)

        mpos = POS_RE.match(line)
        if mpos is not None:
            idx = int(mpos.group(2))
            vals, tail = _triplet_and_tail(mpos.group(3))
            pos_map[idx] = (il, mpos.group(1), tail, vals)
            continue

        mvel = VEL_RE.match(line)
        if mvel is not None:
            idx = int(mvel.group(2))
            vals, tail = _triplet_and_tail(mvel.group(3))
            vel_map[idx] = (il, mvel.group(1), tail, vals)

    common = sorted(set(pos_map.keys()) & set(vel_map.keys()))
    if len(common) == 0:
        raise RuntimeError('No paired state_vector_position/state_vector_velocity lines found in par file.')

    if number_of_sv is not None and number_of_sv != len(common):
        print(
            'WARNING: number_of_state_vectors={} but parsed pairs={}. Continue with parsed pairs.'.format(
                number_of_sv, len(common)
            )
        )

    if dt is None:
        dt = 1.0
        print('WARNING: state_vector_interval not found, using 1.0 s as default.')

    pos = np.vstack([pos_map[ii][3] for ii in common]).astype(np.float64)
    vel = np.vstack([vel_map[ii][3] for ii in common]).astype(np.float64)
    t = np.arange(len(common), dtype=np.float64) * float(dt)

    return {
        'lines': lines,
        'indices': common,
        'pos_map': pos_map,
        'vel_map': vel_map,
        't': t,
        'pos': pos,
        'vel': vel,
    }


def _format_triplet(prefix, values, tail):
    if tail:
        return '{}{:.9f} {:.9f} {:.9f}{}\n'.format(prefix, values[0], values[1], values[2], tail)
    return '{}{:.9f} {:.9f} {:.9f}\n'.format(prefix, values[0], values[1], values[2])


def write_filtered_par(path, parsed, pos_f, vel_f):
    out_lines = parsed['lines'][:]
    indices = parsed['indices']

    for i, idx in enumerate(indices):
        il, prefix, tail, _ = parsed['pos_map'][idx]
        out_lines[il] = _format_triplet(prefix, pos_f[i, :], tail)

        il, prefix, tail, _ = parsed['vel_map'][idx]
        out_lines[il] = _format_triplet(prefix, vel_f[i, :], tail)

    with open(path, 'w') as f:
        f.writelines(out_lines)


def save_plots(png_dir, t, pos, vel, pos_f, vel_f):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('WARNING: matplotlib not available, skip --png output.')
        return

    os.makedirs(png_dir, exist_ok=True)
    components = ['X', 'Y', 'Z']

    for comp in range(3):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, pos[:, comp], 'k-', linewidth=1.0, label='original')
        axes[0].plot(t, pos_f[:, comp], 'r-', linewidth=1.1, label='filtered')
        axes[0].set_ylabel('Position {}'.format(components[comp]))
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].plot(t, vel[:, comp], 'k-', linewidth=1.0, label='original')
        axes[1].plot(t, vel_f[:, comp], 'b-', linewidth=1.1, label='filtered')
        axes[1].set_ylabel('Velocity {}'.format(components[comp]))
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        fig.tight_layout()
        out_png = os.path.join(png_dir, 'orb_fit_{}.png'.format(components[comp].lower()))
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Filter orbit state vectors in SLC par files using robust spline/polynomial fit.'
    )
    parser.add_argument('input_par', type=str, help='Input *.slc.par file')
    parser.add_argument('output_par', type=str, help='Output *.slc.par file')
    parser.add_argument('--png', type=str, default=None, help='Optional output directory for diagnostics PNGs')
    parser.add_argument('--degree', type=int, default=5, help='Spline/polynomial degree (default: 5)')
    parser.add_argument('--sigma', type=float, default=4.0, help='Outlier rejection threshold in robust sigma (default: 4.0)')
    parser.add_argument('--max_iter', type=int, default=3, help='Maximum robust iterations (default: 3)')
    parser.add_argument('--ignore_start', type=int, default=-1, help='Ignore first N state vectors from fit (default: -1 auto)')
    parser.add_argument('--ignore_end', type=int, default=-1, help='Ignore last N state vectors from fit (default: -1 auto)')
    return parser


def main(iargs=None):
    inps = create_parser().parse_args(args=iargs)
    parsed = read_par(inps.input_par)

    nvec = parsed['pos'].shape[0]
    print('INFO: parsed {} state vectors from {}'.format(nvec, inps.input_par))
    if UnivariateSpline is None:
        print('INFO: SciPy not found; fallback to polynomial fitting.')

    pos_f, vel_f, outlier_mask, methods, igs, ige = filter_state_vectors(
        parsed['t'],
        parsed['pos'],
        parsed['vel'],
        degree=max(1, int(inps.degree)),
        sigma=max(1.0, float(inps.sigma)),
        max_iter=max(1, int(inps.max_iter)),
        ignore_start=int(inps.ignore_start),
        ignore_end=int(inps.ignore_end),
    )

    write_filtered_par(inps.output_par, parsed, pos_f, vel_f)
    print('INFO: wrote filtered state vectors to {}'.format(inps.output_par))
    print('INFO: effective ignore_start={}, ignore_end={}'.format(igs, ige))
    print('INFO: methods: {}'.format(', '.join(methods)))

    outlier_idx = np.where(outlier_mask)[0]
    if outlier_idx.size > 0:
        print('WARNING: detected {} outliers in fit domain.'.format(outlier_idx.size))
        print('WARNING: outlier vector indices (1-based, first 40): {}'.format(
            ','.join(str(int(parsed['indices'][ii])) for ii in outlier_idx[:40])
        ))
    else:
        print('INFO: no outliers detected by robust threshold.')

    if inps.png:
        save_plots(inps.png, parsed['t'], parsed['pos'], parsed['vel'], pos_f, vel_f)
        print('INFO: wrote diagnostics PNG(s) to {}'.format(inps.png))


if __name__ == '__main__':
    main()
