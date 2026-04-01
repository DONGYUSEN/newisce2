#!/usr/bin/env python3

import argparse
import glob
import os
import shelve

import isce
from isceobj.Sensor import createSensor


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack SAOCOM SLC data and store metadata in pickle file.')
    parser.add_argument('-i', '--input', dest='inputdir', type=str, required=True,
                        help='Input SAOCOM directory')
    parser.add_argument('-o', '--output', dest='outputdir', type=str, required=True,
                        help='Output SLC directory')
    parser.add_argument('-p', '--polarization', dest='polarization', type=str, default='auto',
                        help='Polarization to unpack: auto/HH/HV/VH/VV (default: auto)')

    return parser.parse_args()


def _pick_file(candidates):
    if not candidates:
        return None
    candidates = sorted(candidates)
    return candidates[0]


def _resolve_inputs(inputdir, polarization):
    pol = polarization.upper()
    if pol == 'AUTO':
        pol_order = ['HH', 'VV', 'HV', 'VH']
    else:
        pol_order = [pol]

    xemt = _pick_file(glob.glob(os.path.join(inputdir, 'S1*.xemt')))
    if xemt is None:
        xemt = _pick_file(glob.glob(os.path.join(inputdir, 'S1*.XEMT')))

    if xemt is None:
        raise RuntimeError(f'Cannot find SAOCOM xemt file under: {inputdir}')

    for p in pol_order:
        pl = p.lower()
        img = _pick_file(glob.glob(os.path.join(inputdir, f'S1*/Data/slc*-{pl}')))
        xml = _pick_file(glob.glob(os.path.join(inputdir, f'S1*/Data/slc*-{pl}.xml')))
        if (img is not None) and (xml is not None):
            return p, img, xml, xemt

    raise RuntimeError(
        f'Cannot find matching SAOCOM image/xml for polarization={polarization} under: {inputdir}'
    )


def unpack(inputdir, outputdir, polarization='auto'):
    '''
    Unpack SAOCOM data to binary SLC file.
    '''

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    date = os.path.basename(outputdir)

    selected_pol, imgname, xmlname, xemtname = _resolve_inputs(inputdir, polarization)

    obj = createSensor('SAOCOM_SLC')
    obj.configure()
    obj._imageFileName = imgname
    obj.xmlFile = xmlname
    obj.xemtFile = xemtname
    obj.output = os.path.join(outputdir, date + '.slc')

    print(f'Using SAOCOM polarization: {selected_pol}')
    print(obj._imageFileName)
    print(obj.xmlFile)
    print(obj.xemtFile)
    print(obj.output)

    obj.extractImage()
    obj.frame.getImage().renderHdr()
    obj.extractDoppler()

    pickName = os.path.join(outputdir, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = obj.frame


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    if inps.outputdir.endswith('/'):
        inps.outputdir = inps.outputdir[:-1]

    if inps.inputdir.endswith('/'):
        inps.inputdir = inps.inputdir[:-1]

    unpack(inps.inputdir, inps.outputdir, polarization=inps.polarization)
