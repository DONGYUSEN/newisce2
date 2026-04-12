#!/usr/bin/env python3

import os
import glob
import argparse
import re

import isce  # noqa
import isceobj
import subprocess
import shelve


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Prepare NISAR RSLC Stack files for stripmapStack processing."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        required=True,
        help="Directory containing NISAR RSLC HDF5 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output SLC directory",
    )
    parser.add_argument(
        "-p",
        "--polarization",
        dest="polarization",
        default="HH",
        help="SLC polarization (default=%(default)s)",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        default="A",
        choices=("A", "B"),
        help="Frequency band (choices=%(choices)s, default=%(default)s)",
    )
    return parser.parse_args()


def write_xml(shelveFile, slcFile):
    with shelve.open(shelveFile, flag="r") as db:
        frame = db["frame"]

    length = frame.numberOfLines
    width = frame.numberOfSamples
    print(width, length)

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.setLength(length)
    slc.filename = slcFile
    slc.setAccessMode("write")
    slc.renderHdr()
    slc.renderVRT()


def get_date(filename):
    """Extract YYYYMMDD date from NISAR filename.

    NISAR filenames follow the pattern:
        NISAR_L1_PR_RSLC_<YYYYMMDD>T<HHMMSS>_...
    Falls back to searching for any 8-digit date-like sequence.
    """
    basename = os.path.basename(filename)
    match = re.search(r"(\d{4})(\d{2})(\d{2})T\d{6}", basename)
    if match:
        return match.group(1) + match.group(2) + match.group(3)

    match = re.search(r"(20\d{6})", basename)
    if match:
        return match.group(1)

    raise ValueError(f"Cannot extract date from filename: {basename}")


def main():
    inps = get_cli_args()

    outputDir = os.path.abspath(inps.output)

    slc_files = sorted(
        glob.glob(os.path.join(inps.input_dir, "*.h5"))
        + glob.glob(os.path.join(inps.input_dir, "*.hdf5"))
    )

    if not slc_files:
        print(f"No HDF5 files found in {inps.input_dir}")
        return

    for h5_file in slc_files:
        imgDate = get_date(h5_file)
        print(imgDate)
        print(h5_file)
        imgDir = os.path.join(outputDir, imgDate)
        os.makedirs(imgDir, exist_ok=True)

        cmd = (
            "unpackFrame_NISAR_RSLC.py"
            + " -i "
            + h5_file
            + " -p "
            + inps.polarization
            + " -f "
            + inps.frequency
            + " -o "
            + imgDir
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        slcFile = os.path.join(imgDir, imgDate + ".slc")
        shelveFile = os.path.join(imgDir, "data")
        write_xml(shelveFile, slcFile)


if __name__ == "__main__":
    main()
