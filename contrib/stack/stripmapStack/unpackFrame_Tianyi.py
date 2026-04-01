#!/usr/bin/env python3

import argparse
import os
import shelve

import isce
from isceobj.Sensor import createSensor


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(
        description='Unpack Tianyi StripMap SLC data and store metadata in shelve file.'
    )
    parser.add_argument(
        '-i', '--input', dest='safe', type=str, required=True,
        help='Input Tianyi SAFE directory or zip file'
    )
    parser.add_argument(
        '-o', '--output', dest='slcdir', type=str, required=True,
        help='Output SLC directory'
    )
    parser.add_argument(
        '-p', '--pol', dest='polarization', type=str, default='vv',
        help='Polarization (default: vv)'
    )
    parser.add_argument(
        '-b', '--orbdir', dest='orbdir', type=str, default=None,
        help='Optional orbit directory'
    )
    parser.add_argument(
        '--orbit-file', dest='orbitfile', type=str, default=None,
        help='Optional external orbit file'
    )

    return parser.parse_args()


def _build_sensor(safe, pol, orbdir=None, orbitfile=None):
    obj = createSensor('TIANYI')
    obj.configure()
    obj.safe = safe
    obj.polarization = pol.lower()
    if orbdir:
        obj.orbitDir = orbdir
    if orbitfile:
        obj.orbitFile = orbitfile
    return obj


def _acquisition_date_yyyymmdd(safe, pol, orbdir=None, orbitfile=None):
    probe = _build_sensor(safe, pol, orbdir=orbdir, orbitfile=orbitfile)
    probe.parse()
    tstart = probe.frame.getSensingStart()
    if tstart is None:
        raise RuntimeError('Could not get sensing start time from Tianyi metadata.')
    return tstart.strftime('%Y%m%d')


def unpack(safe, slcdir, pol='vv', orbdir=None, orbitfile=None):
    '''
    Unpack SAFE/zip to binary SLC file.
    Output SLC filename uses acquisition date: YYYYMMDD.slc
    '''

    os.makedirs(slcdir, exist_ok=True)

    acq_date = _acquisition_date_yyyymmdd(
        safe, pol, orbdir=orbdir, orbitfile=orbitfile
    )
    slc_path = os.path.join(slcdir, acq_date + '.slc')

    obj = _build_sensor(safe, pol, orbdir=orbdir, orbitfile=orbitfile)
    obj.output = slc_path
    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    pickName = os.path.join(slcdir, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame

    print('Output SLC:', slc_path)


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    inps.safe = inps.safe.rstrip('/')
    inps.slcdir = inps.slcdir.rstrip('/')

    unpack(
        inps.safe,
        inps.slcdir,
        pol=inps.polarization,
        orbdir=inps.orbdir,
        orbitfile=inps.orbitfile
    )
