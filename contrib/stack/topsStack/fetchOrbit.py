#!/usr/bin/env python3

import argparse
import datetime
import os
import re
import shutil
import zipfile

import requests

orbitMap = [('precise', 'AUX_POEORB'),
            ('restituted', 'AUX_RESORB')]

datefmt = "%Y%m%dT%H%M%S"
BASE_URLS = {
    'AUX_POEORB': 'https://s1qc.asf.alaska.edu/aux_poeorb/',
    'AUX_RESORB': 'https://s1qc.asf.alaska.edu/aux_resorb/',
}
STEP_BASE_URLS = {
    'AUX_POEORB': 'https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/',
    'AUX_RESORB': 'https://step.esa.int/auxdata/orbits/Sentinel-1/RESORB/',
}


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fetch orbits corresponding to given SAFE package')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='Path to SAFE package of interest')
    parser.add_argument('-o', '--output', dest='outdir', type=str, default='.',
                        help='Path to output directory')
    parser.add_argument('-t', '--token-file', dest='token_file', type=str, default=None,
                        help='Kept for backward compatibility; ignored')
    parser.add_argument('-u', '--username', dest='username', type=str, default=None,
                        help='Kept for backward compatibility; ignored')
    parser.add_argument('-p', '--password', dest='password', type=str, default=None,
                        help='Kept for backward compatibility; ignored')

    return parser.parse_args()


def FileToTimeStamp(safename):
    '''
    Return timestamp from SAFE name.
    '''
    safename = os.path.basename(safename)
    fields = safename.split('_')
    sstamp = []  # SAFE file start time

    try:
        tstamp = datetime.datetime.strptime(fields[-4], datefmt)
        sstamp = datetime.datetime.strptime(fields[-5], datefmt)
    except Exception:
        p = re.compile(r'(?<=_)\d{8}')
        dt2 = p.search(safename).group()
        tstamp = datetime.datetime.strptime(dt2, '%Y%m%d')
        sstamp = tstamp

    satName = fields[0]

    return tstamp, satName, sstamp


def parse_orbit_validity(fname):
    clean = os.path.basename(fname).replace('.zip', '')
    fields = clean.split('_')
    start = datetime.datetime.strptime(fields[-2][1:], datefmt)
    stop = datetime.datetime.strptime(fields[-1].replace('.EOF', ''), datefmt)
    return start, stop


def list_orbits(session, sat_name, orbit_code):
    base_url = BASE_URLS[orbit_code]
    response = session.get(base_url, timeout=60)
    response.raise_for_status()

    # Example name:
    # S1A_OPER_AUX_POEORB_OPOD_20231130T111111_V20231109T225942_20231111T005942.EOF
    suffix = orbit_code.split('_', 1)[1]
    pattern = r'href="(S1[AB]_OPER_AUX_{0}_OPOD_[^"]+\.EOF)"'.format(suffix)
    names = set(re.findall(pattern, response.text))
    names = [x for x in names if x.startswith(sat_name + '_')]
    return sorted(names), base_url


def list_step_orbits_for_month(session, sat_name, orbit_code, year, month):
    base = STEP_BASE_URLS[orbit_code]
    url = '{0}{1}/{2}/{3:02d}/'.format(base, sat_name, year, month)
    response = session.get(url, timeout=60)
    if response.status_code != 200:
        return [], url

    suffix = orbit_code.split('_', 1)[1]
    pattern = r'href="(S1[AB]_OPER_AUX_{0}_OPOD_[^"]+\.EOF(?:\.zip)?)"'.format(suffix)
    names = set(re.findall(pattern, response.text))
    names = [x for x in names if x.startswith(sat_name + '_')]
    return sorted(names), url


def find_step_orbit(session, sat_name, orbit_code, safe_start, safe_stop):
    months = []
    for dt in (safe_start - datetime.timedelta(days=1), safe_start, safe_stop, safe_stop + datetime.timedelta(days=1)):
        ym = (dt.year, dt.month)
        if ym not in months:
            months.append(ym)

    matches = []
    for year, month in months:
        names, month_url = list_step_orbits_for_month(session, sat_name, orbit_code, year, month)
        for name in names:
            tbef, taft = parse_orbit_validity(name)
            if (tbef <= safe_start) and (taft >= safe_stop):
                matches.append((name, month_url + name))

    if not matches:
        return None, None

    matches.sort(key=lambda x: x[0])
    return matches[-1]


def download_file(urls, outname, session=None):
    '''
    Download file from url to outname.
    '''

    if session is None:
        session = requests.session()

    last_error = None
    for url in urls:
        try:
            print('Downloading URL: ', url)
            request = session.get(url, stream=True, timeout=120, verify=True, allow_redirects=True)
            request.raise_for_status()

            with open(outname, 'wb') as f:
                for chunk in request.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return
        except Exception as err:
            last_error = err
            if os.path.exists(outname):
                os.remove(outname)

    raise last_error


def download_step_orbit(session, sat_name, orbit_code, safe_start, safe_stop, outdir):
    fname, url = find_step_orbit(session, sat_name, orbit_code, safe_start, safe_stop)
    if fname is None:
        raise RuntimeError('No STEP orbit found for {0} {1}'.format(sat_name, orbit_code))

    print('Downloading URL: ', url)
    req = session.get(url, timeout=120, verify=True)
    req.raise_for_status()

    if fname.endswith('.zip'):
        zip_path = os.path.join(outdir, fname)
        with open(zip_path, 'wb') as f:
            f.write(req.content)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            eof_members = [m for m in zf.namelist() if m.endswith('.EOF')]
            if len(eof_members) == 0:
                raise RuntimeError('No EOF file found inside {}'.format(zip_path))
            member = eof_members[0]
            out_eof = os.path.join(outdir, os.path.basename(member))
            with zf.open(member) as src, open(out_eof, 'wb') as dst:
                shutil.copyfileobj(src, dst)

        os.remove(zip_path)
        return out_eof

    outpath = os.path.join(outdir, fname)
    with open(outpath, 'wb') as f:
        f.write(req.content)
    return outpath


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()

    fileTS, satName, fileTSStart = FileToTimeStamp(inps.input)
    print('Reference time: ', fileTS)
    print('Satellite name: ', satName)

    os.makedirs(inps.outdir, exist_ok=True)

    match = None
    match_base = None
    match_code = None
    session = requests.Session()

    for _, orbit_code in orbitMap:
        orbit_names, base_url = list_orbits(session, satName, orbit_code)

        match = None
        for orbit_name in orbit_names:
            tbef, taft = parse_orbit_validity(orbit_name)
            if (tbef <= fileTSStart) and (taft >= fileTS):
                # Keep the latest matching file in lexical order
                match = orbit_name

        if match is not None:
            match_base = base_url
            match_code = orbit_code
            break

    if match is not None:
        output = os.path.join(inps.outdir, match)
        if os.path.exists(output):
            print('Orbit already exists: ', output)
        else:
            try:
                urls = [match_base + match]
                download_file(urls, output, session)
                print('Saved orbit to: ', output)
            except Exception:
                print('s1qc download failed, fallback to STEP mirror ...')
                out = download_step_orbit(session, satName, match_code, fileTSStart, fileTS, inps.outdir)
                print('Saved orbit to: ', out)
    else:
        print('Failed to find orbits for tref {0} ({1})'.format(fileTS, satName))
