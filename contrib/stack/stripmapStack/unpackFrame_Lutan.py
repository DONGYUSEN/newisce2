#!/usr/bin/env python3

import argparse
import glob
import os
import shelve
import zipfile

import isce
from isceobj.Sensor import createSensor


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(
        description='Unpack Lutan SLC data and store metadata in shelve file.'
    )
    parser.add_argument(
        '-i', '--input', dest='input', type=str, required=True,
        help='Input Lutan zip file / directory / TIFF file'
    )
    parser.add_argument(
        '-o', '--output', dest='slcdir', type=str, required=True,
        help='Output SLC directory'
    )
    parser.add_argument(
        '--orbit-file', dest='orbitfile', type=str, default=None,
        help='Optional external orbit file'
    )
    parser.add_argument(
        '--crop-far', dest='crop_far', type=int, default=0,
        help='Crop N pixels from far-range edge (default: 0)'
    )

    return parser.parse_args()


def _is_tiff(name):
    low = name.lower()
    return low.endswith('.tif') or low.endswith('.tiff')


def _meta_from_tiff_name(name):
    base, _ = os.path.splitext(name)
    return base + '.meta.xml'


def _pick_lutan_tiff_from_zip(zippath):
    with zipfile.ZipFile(zippath, 'r') as zf:
        members = zf.namelist()
        memset = set(members)
        tiffs = sorted([m for m in members if _is_tiff(m) and not m.endswith('/')])
        if len(tiffs) == 0:
            raise RuntimeError('No TIFF file found in zip: {}'.format(zippath))

        candidates = [m for m in tiffs if _meta_from_tiff_name(m) in memset]
        chosen = candidates[0] if len(candidates) > 0 else tiffs[0]

    zabs = os.path.abspath(zippath)
    return '/vsizip/{}{}'.format(zabs, '/' + chosen.lstrip('/'))


def _pick_lutan_tiff_from_dir(indir):
    tiffs = sorted(glob.glob(os.path.join(indir, '**', '*.tif'), recursive=True))
    tiffs += sorted(glob.glob(os.path.join(indir, '**', '*.tiff'), recursive=True))
    if len(tiffs) == 0:
        raise RuntimeError('No TIFF file found in directory: {}'.format(indir))

    candidates = []
    for tif in tiffs:
        meta = _meta_from_tiff_name(tif)
        if os.path.isfile(meta):
            candidates.append(tif)

    return candidates[0] if len(candidates) > 0 else tiffs[0]


def resolve_tiff_input(inp):
    path = os.path.abspath(inp)
    if os.path.isfile(path):
        if path.lower().endswith('.zip'):
            return _pick_lutan_tiff_from_zip(path)
        if _is_tiff(path):
            return path
        raise RuntimeError('Input file is neither zip nor TIFF: {}'.format(path))

    if os.path.isdir(path):
        return _pick_lutan_tiff_from_dir(path)

    raise RuntimeError('Input path does not exist: {}'.format(inp))


def _build_sensor(tiff_path, orbitfile=None, crop_far=0):
    obj = createSensor('LUTAN1')
    obj.configure()
    obj.tiff = tiff_path
    obj.rangeCropFarPixels = max(0, int(crop_far))
    if orbitfile:
        obj.orbitFile = orbitfile
    return obj


def _acquisition_date_yyyymmdd(tiff_path, orbitfile=None, crop_far=0):
    probe = _build_sensor(tiff_path, orbitfile=orbitfile, crop_far=crop_far)
    probe.parse()
    tstart = probe.frame.getSensingStart()
    if tstart is None:
        raise RuntimeError('Could not get sensing start time from Lutan metadata.')
    return tstart.strftime('%Y%m%d')


def unpack(inp, slcdir, orbitfile=None, crop_far=0):
    '''
    Unpack Lutan data to binary SLC file.
    Output SLC filename uses acquisition date: YYYYMMDD.slc
    '''

    os.makedirs(slcdir, exist_ok=True)
    tiff_path = resolve_tiff_input(inp)
    acq_date = _acquisition_date_yyyymmdd(tiff_path, orbitfile=orbitfile, crop_far=crop_far)
    slc_path = os.path.join(slcdir, acq_date + '.slc')

    obj = _build_sensor(tiff_path, orbitfile=orbitfile, crop_far=crop_far)
    obj.output = slc_path
    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    pickName = os.path.join(slcdir, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame

    print('Input TIFF:', tiff_path)
    print('Output SLC:', slc_path)


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    inps.input = inps.input.rstrip('/')
    inps.slcdir = inps.slcdir.rstrip('/')
    unpack(inps.input, inps.slcdir, orbitfile=inps.orbitfile, crop_far=inps.crop_far)
