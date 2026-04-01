#!/usr/bin/env python3
"""Post-process ISCE2 TOPS products for mapping/export deliverables.

This utility targets the standard topsApp outputs in a processing directory:
1) Control UTM export resolution with integer posting (native or multilook-based).
2) Generate LOS displacement and look-angle products.
3) Export flat phase / unwrapped phase / log-intensity products to
   GeoTIFF + PNG + KMZ.
"""

import argparse
import glob
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal
from osgeo import osr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process ISCE2 TOPS products into GeoTIFF/PNG/KMZ with UTM control."
    )
    parser.add_argument(
        "--proc-dir",
        default=".",
        help="TOPS processing directory containing merged/, fine_interferogram/, topsApp.xml.",
    )
    parser.add_argument(
        "--outdir",
        default="products",
        help="Output directory (default: ./products).",
    )
    parser.add_argument(
        "--flat",
        default="merged/filt_topophase.flat.geo",
        help="Filtered flat interferogram (geocoded) path relative to --proc-dir.",
    )
    parser.add_argument(
        "--unw",
        default="merged/filt_topophase.unw.geo",
        help="Unwrapped interferogram (geocoded) path relative to --proc-dir.",
    )
    parser.add_argument(
        "--los",
        default="merged/los.rdr.geo",
        help="LOS geometry file (geocoded) path relative to --proc-dir.",
    )
    parser.add_argument(
        "--avg-amp",
        default="merged/topophase.cor.geo",
        help="Average-amplitude source path relative to --proc-dir (default: merged/topophase.cor.geo, band1).",
    )
    parser.add_argument(
        "--avg-amp-band",
        type=int,
        default=1,
        help="Band index (1-based) for --avg-amp source (default: 1).",
    )
    parser.add_argument(
        "--topsapp-xml",
        default="topsApp.xml",
        help="topsApp xml path (used for looks in multilook resolution mode).",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="Radar wavelength in meters. If omitted, auto-detect from fine_interferogram/IW*.xml.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=98.0,
        help="Stretch span percentage (default 98, i.e. 1%%-99%%).",
    )
    parser.add_argument(
        "--to-utm",
        action="store_true",
        help="Warp outputs to UTM GeoTIFF before PNG/KMZ export.",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=None,
        help="Target EPSG for UTM export. If omitted, auto-select from scene center.",
    )
    parser.add_argument(
        "--utm-res-mode",
        choices=["multilook", "native", "manual"],
        default="multilook",
        help="How to choose UTM resolution (default: multilook).",
    )
    parser.add_argument(
        "--utm-res",
        type=float,
        default=None,
        help="Manual square UTM resolution in meters (used with --utm-res-mode manual).",
    )
    parser.add_argument(
        "--utm-res-x",
        type=float,
        default=None,
        help="Manual UTM X resolution in meters (optional).",
    )
    parser.add_argument(
        "--utm-res-y",
        type=float,
        default=None,
        help="Manual UTM Y resolution in meters (optional).",
    )
    parser.add_argument(
        "--square-pixel",
        action="store_true",
        help="Force square UTM pixel size using max(xRes, yRes).",
    )
    parser.add_argument(
        "--range-looks",
        type=int,
        default=None,
        help="Override range looks for multilook resolution estimation.",
    )
    parser.add_argument(
        "--azimuth-looks",
        type=int,
        default=None,
        help="Override azimuth looks for multilook resolution estimation.",
    )
    return parser.parse_args()


def abs_path(proc_dir, rel_or_abs):
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(proc_dir, rel_or_abs)


def resolve_gdal_path(path):
    if os.path.exists(path):
        if path.endswith(".xml"):
            vrt = path[:-4] + ".vrt"
            if os.path.exists(vrt):
                return vrt
        return path

    if os.path.exists(path + ".vrt"):
        return path + ".vrt"
    if os.path.exists(path + ".xml"):
        vrt = path + ".vrt"
        if os.path.exists(vrt):
            return vrt
        return path + ".xml"
    return path


def open_ds(path):
    gpath = resolve_gdal_path(path)
    ds = gdal.Open(gpath, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("Failed to open GDAL dataset: {0}".format(path))
    return ds, gpath


def unique_items(seq):
    out = []
    seen = set()
    for x in seq:
        if x is None:
            continue
        k = str(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def input_candidates(workflow_name, kind):
    w = str(workflow_name or "").strip().lower()
    tops_first = w.startswith("tops")
    stripmap_first = w.startswith("stripmap")

    tops = {
        "flat": ["merged/filt_topophase.flat.geo", "merged/topophase.flat.geo"],
        "unw": ["merged/filt_topophase.unw.geo"],
        "los": ["merged/los.rdr.geo"],
        "avg_amp": ["merged/topophase.cor.geo", "merged/phsig.cor.geo"],
    }
    stripmap = {
        "flat": ["interferogram/filt_topophase.flat.geo", "interferogram/topophase.flat.geo"],
        "unw": ["interferogram/filt_topophase.unw.geo"],
        "los": ["geometry/los.rdr.geo", "interferogram/los.rdr.geo"],
        "avg_amp": ["interferogram/topophase.cor.geo", "interferogram/phsig.cor.geo"],
    }

    if tops_first:
        merged = tops.get(kind, []) + stripmap.get(kind, [])
    elif stripmap_first:
        merged = stripmap.get(kind, []) + tops.get(kind, [])
    else:
        merged = tops.get(kind, []) + stripmap.get(kind, [])
    return unique_items(merged)


def open_preferred_ds(proc_dir, preferred_rel, workflow_name, kind):
    tried = []
    candidates = unique_items([preferred_rel] + input_candidates(workflow_name, kind))
    for rel in candidates:
        p = abs_path(proc_dir, rel)
        try:
            ds, opened = open_ds(p)
            return ds, opened, rel
        except Exception:
            tried.append(p)
            continue
    raise RuntimeError(
        "Failed to open {0} dataset. Tried: {1}".format(kind, ", ".join(tried[:12]))
    )


def get_center_lonlat(ds):
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise RuntimeError("Dataset has no geotransform; cannot infer center lon/lat.")
    x = gt[0] + gt[1] * (0.5 * ds.RasterXSize) + gt[2] * (0.5 * ds.RasterYSize)
    y = gt[3] + gt[4] * (0.5 * ds.RasterXSize) + gt[5] * (0.5 * ds.RasterYSize)

    src = osr.SpatialReference()
    wkt = ds.GetProjection()
    if wkt:
        src.ImportFromWkt(wkt)
    else:
        src.ImportFromEPSG(4326)

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(src, dst)
    lon, lat, _ = tx.TransformPoint(float(x), float(y))
    return lon, lat


def meter_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    mlat = (
        111132.954
        - 559.822 * math.cos(2.0 * lat)
        + 1.175 * math.cos(4.0 * lat)
    )
    mlon = (
        111412.84 * math.cos(lat)
        - 93.5 * math.cos(3.0 * lat)
        + 0.118 * math.cos(5.0 * lat)
    )
    return mlat, max(1e-6, abs(mlon))


def estimate_native_res_m(ds):
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise RuntimeError("Dataset has no geotransform; cannot estimate native resolution.")

    proj = osr.SpatialReference()
    if ds.GetProjection():
        proj.ImportFromWkt(ds.GetProjection())
    else:
        proj.ImportFromEPSG(4326)

    x_res = abs(float(gt[1]))
    y_res = abs(float(gt[5]))
    if proj.IsGeographic():
        _, lat = get_center_lonlat(ds)
        mlat, mlon = meter_per_degree(lat)
        return x_res * mlon, y_res * mlat

    return x_res, y_res


def utm_epsg_from_lonlat(lon, lat):
    zone = int((lon + 180.0) / 6.0) + 1
    if lat >= 0.0:
        return 32600 + zone
    return 32700 + zone


def parse_tops_looks(tops_xml):
    if (tops_xml is None) or (not os.path.isfile(tops_xml)):
        return None, None

    try:
        root = ET.parse(tops_xml).getroot()
    except Exception:
        return None, None

    range_looks = None
    az_looks = None
    for prop in root.iter("property"):
        name = str(prop.get("name", "")).strip().lower()
        value = (prop.text or "").strip()
        if not value:
            child = prop.find("value")
            value = (child.text or "").strip() if child is not None and child.text else ""
        if not value:
            continue

        if ("range looks" in name) and (range_looks is None):
            try:
                range_looks = int(float(value))
            except Exception:
                pass
        if ("azimuth looks" in name) and (az_looks is None):
            try:
                az_looks = int(float(value))
            except Exception:
                pass

    return range_looks, az_looks


def _parse_float_token(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _normalize_token(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _detect_wavelength_from_xml(iw_xml):
    if (iw_xml is None) or (not os.path.isfile(iw_xml)):
        return None

    targets = ("radarwavelength", "radarwavelegth", "wavelength")
    try:
        root = ET.parse(iw_xml).getroot()
    except Exception:
        root = None

    if root is not None:
        for elem in root.iter():
            keys = []
            tag = elem.tag.split("}", 1)[-1] if isinstance(elem.tag, str) else ""
            if tag:
                keys.append(tag)
            for attr in ("name", "key", "public_name", "id", "value"):
                val = elem.get(attr)
                if val:
                    keys.append(val)

            if not any(any(tok in _normalize_token(k) for tok in targets) for k in keys):
                continue

            candidates = [elem.text]
            candidates.extend(child.text for child in list(elem))
            for c in candidates:
                val = _parse_float_token(c)
                if (val is not None) and (0.001 < val < 1.0):
                    return float(val)

    try:
        with open(iw_xml, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    m = re.search(
        r"radar[\W_]*wave(?:len|leg)th[^0-9+\-.eE]{0,128}([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        content,
        flags=re.IGNORECASE,
    )
    if m:
        val = _parse_float_token(m.group(1))
        if (val is not None) and (0.001 < val < 1.0):
            return float(val)
    return None


def detect_wavelength(proc_dir):
    cands = []
    for i in (1, 2, 3):
        cands.append(os.path.join(proc_dir, "fine_interferogram", "IW{0}.xml".format(i)))
        cands.append(os.path.join(proc_dir, "reference", "IW{0}.xml".format(i)))
    for p in (
        "stripmapApp.xml",
        "topsApp.xml",
        "stripmapProc.xml",
        "topsProc.xml",
        "insarProc.xml",
        "reference.xml",
        "secondary.xml",
    ):
        cands.append(os.path.join(proc_dir, p))
    cands.extend(sorted(glob.glob(os.path.join(proc_dir, "reference", "*.xml"))))
    cands.extend(sorted(glob.glob(os.path.join(proc_dir, "fine_interferogram", "*.xml"))))
    cands.extend(sorted(glob.glob(os.path.join(proc_dir, "interferogram", "*.xml"))))
    # stripmap products are often dumped in proc-dir root as *_raw.xml/*_slc.xml/*_crop.xml
    cands.extend(sorted(glob.glob(os.path.join(proc_dir, "*.xml"))))
    xml_cands = [p for p in unique_items(cands) if os.path.isfile(p)]
    pm_err = None
    if not xml_cands:
        pm_err = "No XML candidates found for wavelength detection."
    else:
        try:
            from iscesys.Component.ProductManager import ProductManager as PM

            pm = PM()
            pm.configure()
            for xmlp in xml_cands:
                try:
                    prod = pm.loadProduct(xmlp)
                except Exception:
                    continue

                if hasattr(prod, "bursts") and (len(prod.bursts) > 0):
                    w = getattr(prod.bursts[0], "radarWavelength", None)
                    if w is not None:
                        return float(w)

                w = getattr(prod, "radarWavelength", None)
                if w is not None:
                    return float(w)
                inst = getattr(prod, "instrument", None)
                if inst is not None:
                    w = getattr(inst, "radarWavelength", None)
                    if w is not None:
                        return float(w)
                    getter = getattr(inst, "getRadarWavelength", None)
                    if callable(getter):
                        try:
                            return float(getter())
                        except Exception:
                            pass
            pm_err = "No wavelength extracted from ProductManager candidates."
        except Exception as err:
            pm_err = str(err)

        for xmlp in xml_cands:
            xml_wvl = _detect_wavelength_from_xml(xmlp)
            if xml_wvl is not None:
                return float(xml_wvl)

    # Final fallback: parse wavelength from ISCE log text if available.
    log_cands = unique_items(
        [
            os.environ.get("ISCE_LOG_FILE"),
            os.path.join(proc_dir, "isce.log"),
            os.path.join(proc_dir, "log", "isce.log"),
            os.path.join(proc_dir, "logs", "isce.log"),
        ]
    )
    for logp in log_cands:
        if not logp or (not os.path.isfile(logp)):
            continue
        try:
            with open(logp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        for pat in (
            r"(?:^|[\s,;])reference\.wavelength\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
            r"(?:^|[\s,;])secondary\.wavelength\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
            r"(?:^|[\s,;])wavelength\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        ):
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if not m:
                continue
            val = _parse_float_token(m.group(1))
            if (val is not None) and (0.001 < val < 1.0):
                return float(val)

    raise RuntimeError(
        "Cannot detect wavelength from XML candidates ({0} files). ProductManager error: {1}".format(
            len(xml_cands), pm_err
        )
    )


def estimate_incidence_from_los(los_ds):
    band = los_ds.GetRasterBand(1)
    cx = los_ds.RasterXSize // 2
    cy = los_ds.RasterYSize // 2
    wx = max(1, min(64, los_ds.RasterXSize))
    wy = max(1, min(64, los_ds.RasterYSize))
    xoff = max(0, cx - wx // 2)
    yoff = max(0, cy - wy // 2)
    arr = band.ReadAsArray(xoff, yoff, wx, wy)
    if arr is None:
        raise RuntimeError("Failed to read incidence angles from LOS dataset.")
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0) & (arr < 90.0)
    if not np.any(valid):
        return 35.0
    return float(np.median(arr[valid]))


def estimate_multilook_res_m(proc_dir, range_looks, az_looks, los_ds):
    if range_looks is None or az_looks is None:
        raise RuntimeError("Range/azimuth looks are required for multilook resolution estimation.")

    try:
        from iscesys.Component.ProductManager import ProductManager as PM
    except Exception as err:
        raise RuntimeError("Cannot import ISCE ProductManager: {0}".format(err))

    ifg_dir = os.path.join(proc_dir, "fine_interferogram")
    iw_xml = None
    for i in (1, 2, 3):
        p = os.path.join(ifg_dir, "IW{0}.xml".format(i))
        if os.path.isfile(p):
            iw_xml = p
            break
    if iw_xml is None:
        raise RuntimeError("No IW*.xml found in fine_interferogram.")

    pm = PM()
    pm.configure()
    prod = pm.loadProduct(iw_xml)
    if (not hasattr(prod, "bursts")) or (len(prod.bursts) == 0):
        raise RuntimeError("Failed to read bursts from {0}".format(iw_xml))

    burst = prod.bursts[len(prod.bursts) // 2]
    dr = float(burst.rangePixelSize)
    dtaz = float(burst.azimuthTimeInterval)
    tm = burst.sensingMid
    sv = burst.orbit.interpolate(tm, method="hermite")
    vel = np.linalg.norm(np.asarray(sv.getVelocity(), dtype=np.float64))

    inc = estimate_incidence_from_los(los_ds)
    sin_inc = max(1e-3, math.sin(math.radians(inc)))
    range_ml = float(range_looks) * dr / sin_inc
    az_ml = float(az_looks) * dtaz * float(vel)
    return range_ml, az_ml, inc, vel


def stretch_to_u8(arr, valid_mask, percent=98.0):
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if not np.any(valid):
        return out, np.nan, np.nan

    tail = max(0.0, (100.0 - float(percent)) / 2.0)
    lo, hi = np.percentile(arr[valid], [tail, 100.0 - tail])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.nanmin(arr[valid]))
        hi = float(np.nanmax(arr[valid]))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            out[valid] = 1
            return out, lo, hi

    scaled = (arr - lo) * (254.0 / (hi - lo)) + 1.0
    out[valid] = np.clip(np.rint(scaled[valid]), 1, 255).astype(np.uint8)
    return out, float(lo), float(hi)


def wrapped_phase_to_u8(phase, valid_mask):
    out = np.zeros(phase.shape, dtype=np.uint8)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(phase)
    if not np.any(valid):
        return out
    wrapped = np.mod(phase[valid] + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    out[valid] = np.clip(np.rint(1.0 + wrapped * 254.0), 1, 255).astype(np.uint8)
    return out


def gamma_insar_phase_lut():
    """Build a Gamma InSAR-like cyclic color LUT (index 0 reserved for nodata)."""
    rgbs = np.zeros((256, 3), dtype=np.uint8)
    for kk in range(85):
        rgbs[kk, 0] = kk * 3
        rgbs[kk, 1] = 255 - kk * 3
        rgbs[kk, 2] = 255

    rgbs[85:170, 0] = rgbs[0:85, 2]
    rgbs[85:170, 1] = rgbs[0:85, 0]
    rgbs[85:170, 2] = rgbs[0:85, 1]

    rgbs[170:255, 0] = rgbs[0:85, 1]
    rgbs[170:255, 1] = rgbs[0:85, 2]
    rgbs[170:255, 2] = rgbs[0:85, 0]
    rgbs[255, :] = np.array([0, 255, 255], dtype=np.uint8)

    rgbs = np.roll(rgbs, int(256 / 2 - 214), axis=0)
    rgbs = np.flipud(rgbs)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[1:, :] = rgbs[:255, :]
    return lut


def colorize_u8_with_lut(view_u8, lut):
    idx = np.asarray(view_u8, dtype=np.uint8)
    return lut[idx]


def write_tiff(path, arr, template_ds, dtype, nodata=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        ny, nx = arr.shape
        nb = 1
    elif arr.ndim == 3 and arr.shape[2] >= 1:
        ny, nx, nb = arr.shape
    else:
        raise RuntimeError("Unsupported array shape for GeoTIFF write: {0}".format(arr.shape))
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, nx, ny, nb, dtype, options=["COMPRESS=LZW"])
    if ds is None:
        raise RuntimeError("Failed to create GeoTIFF: {0}".format(path))
    gt = template_ds.GetGeoTransform(can_return_null=True)
    if gt is not None:
        ds.SetGeoTransform(gt)
    proj = template_ds.GetProjection()
    if proj:
        ds.SetProjection(proj)
    if nb == 1:
        band = ds.GetRasterBand(1)
        band.WriteArray(arr)
        if nodata is not None:
            band.SetNoDataValue(nodata)
    else:
        for ib in range(nb):
            ds.GetRasterBand(ib + 1).WriteArray(arr[:, :, ib])
    ds.FlushCache()
    ds = None


def warp_to_utm(src_tif, dst_tif, epsg, xres, yres):
    opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:{0}".format(epsg),
        xRes=float(xres),
        yRes=float(yres),
        targetAlignedPixels=True,
        resampleAlg="near",
        creationOptions=["COMPRESS=LZW"],
    )
    out = gdal.Warp(dst_tif, src_tif, options=opts)
    if out is None:
        raise RuntimeError("Failed UTM warp: {0}".format(src_tif))
    out = None


def geotiff_to_png(src_tif, dst_png):
    out = gdal.Translate(dst_png, src_tif, format="PNG")
    if out is None:
        raise RuntimeError("Failed PNG export: {0}".format(src_tif))
    out = None


def geotiff_to_kmz(src_tif, dst_kmz):
    ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("Failed to open for KMZ: {0}".format(src_tif))

    src_proj = ds.GetProjection()
    if src_proj:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_proj)
        needs_4326 = not srs.IsGeographic()
    else:
        needs_4326 = False

    tmp_path = None
    src_for_kmz = src_tif
    if needs_4326:
        fd, tmp_path = tempfile.mkstemp(suffix=".tif", prefix="kmz4326_")
        os.close(fd)
        opts = gdal.WarpOptions(
            format="GTiff",
            dstSRS="EPSG:4326",
            resampleAlg="near",
            creationOptions=["COMPRESS=LZW"],
        )
        out = gdal.Warp(tmp_path, src_tif, options=opts)
        if out is None:
            raise RuntimeError("Failed temporary EPSG:4326 warp for KMZ: {0}".format(src_tif))
        out = None
        src_for_kmz = tmp_path

    out = gdal.Translate(dst_kmz, src_for_kmz, format="KMLSUPEROVERLAY")
    if out is None:
        raise RuntimeError("Failed KMZ export: {0}".format(src_for_kmz))
    out = None
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)


def integerize_resolution(xres, yres, square=False):
    if square:
        v = max(float(xres), float(yres))
        iv = max(1, int(math.ceil(v)))
        xr, yr = iv, iv
    else:
        xr = max(1, int(round(float(xres))))
        yr = max(1, int(round(float(yres))))
    return xr, yr


def main():
    args = parse_args()
    gdal.UseExceptions()
    workflow_name = os.environ.get("ISCE_AUTO_POSTPROCESS_WORKFLOW", "")

    proc_dir = os.path.abspath(args.proc_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    tops_xml = abs_path(proc_dir, args.topsapp_xml)

    flat_ds, flat_opened, flat_rel = open_preferred_ds(proc_dir, args.flat, workflow_name, "flat")
    unw_ds, unw_opened, unw_rel = open_preferred_ds(proc_dir, args.unw, workflow_name, "unw")
    los_ds, los_opened, los_rel = open_preferred_ds(proc_dir, args.los, workflow_name, "los")
    print("INFO: opened flat={0}".format(flat_opened))
    print("INFO: opened unw={0}".format(unw_opened))
    print("INFO: opened los={0}".format(los_opened))
    if workflow_name:
        print("INFO: workflow={0}, selected inputs: flat={1}, unw={2}, los={3}".format(workflow_name, flat_rel, unw_rel, los_rel))

    for name, ds in (("flat", flat_ds), ("unw", unw_ds), ("los", los_ds)):
        if ds.GetGeoTransform(can_return_null=True) is None:
            raise RuntimeError(
                "{0} input is not geocoded (missing geotransform). "
                "Use *.geo products from topsApp geocode step.".format(name)
            )

    if unw_ds.RasterCount < 2:
        raise RuntimeError("Unwrapped dataset must have at least 2 bands (amp + phase).")
    if los_ds.RasterCount < 1:
        raise RuntimeError("LOS dataset must have at least 1 band (incidence angle).")

    # Read core arrays.
    flat_arr = flat_ds.GetRasterBand(1).ReadAsArray()
    if flat_arr is None:
        raise RuntimeError("Failed reading flat raster band 1.")
    flat_phase = np.angle(flat_arr) if np.iscomplexobj(flat_arr) else np.asarray(flat_arr, dtype=np.float32)
    flat_valid = np.isfinite(flat_phase)
    if np.iscomplexobj(flat_arr):
        flat_valid &= (np.abs(flat_arr) > 0.0)

    unw_amp = np.asarray(unw_ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
    unw_phase = np.asarray(unw_ds.GetRasterBand(2).ReadAsArray(), dtype=np.float32)
    look_angle = np.asarray(los_ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
    if unw_amp is None or unw_phase is None or look_angle is None:
        raise RuntimeError("Failed reading one or more required product bands.")

    # Unwrapped/LOS validity still follows unw amplitude mask.
    unw_amp_valid = np.isfinite(unw_amp) & (unw_amp > 0.0)
    unw_valid = unw_amp_valid & np.isfinite(unw_phase)
    look_valid = np.isfinite(look_angle) & (look_angle > 0.0) & (look_angle < 90.0)

    # Avg intensity source: prefer merged master/slave average amplitude product.
    avg_amp = None
    avg_amp_nodata = None
    avg_amp_src = None
    avg_try = unique_items([args.avg_amp] + input_candidates(workflow_name, "avg_amp"))
    for rel in avg_try:
        try:
            avg_ds, avg_opened = open_ds(abs_path(proc_dir, rel))
        except Exception:
            continue

        if (avg_ds.RasterXSize != unw_ds.RasterXSize) or (avg_ds.RasterYSize != unw_ds.RasterYSize):
            print(
                "WARNING: avg-amp source size mismatch ({0}x{1} vs {2}x{3}); trying next.".format(
                    avg_ds.RasterXSize,
                    avg_ds.RasterYSize,
                    unw_ds.RasterXSize,
                    unw_ds.RasterYSize,
                )
            )
            continue
        if avg_ds.RasterCount < int(args.avg_amp_band):
            print(
                "WARNING: avg-amp source has only {0} band(s), requested band {1}; trying next.".format(
                    avg_ds.RasterCount, int(args.avg_amp_band)
                )
            )
            continue

        avg_band = avg_ds.GetRasterBand(int(args.avg_amp_band))
        avg_amp = np.asarray(avg_band.ReadAsArray(), dtype=np.float32)
        avg_amp_nodata = avg_band.GetNoDataValue()
        avg_amp_src = "{0} (band {1})".format(avg_opened, int(args.avg_amp_band))
        break

    if avg_amp is None:
        avg_amp = np.asarray(unw_amp, dtype=np.float32)
        avg_amp_nodata = None
        avg_amp_src = "unw band1 fallback"
    print("INFO: avg intensity source={0}".format(avg_amp_src))

    # Detect wavelength and derive LOS displacement.
    wavelength = float(args.wavelength) if args.wavelength is not None else detect_wavelength(proc_dir)
    los_disp = -1.0 * unw_phase.astype(np.float64) * wavelength / (4.0 * np.pi)
    los_disp = los_disp.astype(np.float32)
    disp_valid = unw_valid & np.isfinite(los_disp)
    print("INFO: wavelength={0:.9f} m".format(wavelength))

    # Intensity: full-scene output, but exclude invalid border pixels from stretch statistics.
    avg_amp_finite = np.isfinite(avg_amp)
    avg_amp_valid = avg_amp_finite & (avg_amp > 0.0)
    if (avg_amp_nodata is not None) and np.isfinite(avg_amp_nodata):
        avg_amp_valid &= (np.abs(avg_amp - float(avg_amp_nodata)) > 1e-12)
    if not np.any(avg_amp_valid):
        avg_amp_valid = avg_amp_finite
        print("WARNING: no positive valid pixels in avg-amp source; fallback to finite-only stretch statistics.")

    amp_safe = np.where(avg_amp_valid, np.maximum(np.abs(avg_amp), 1e-8), 1e-8).astype(np.float32)
    log_intensity = np.log10(amp_safe).astype(np.float32)
    print("INFO: avg intensity valid pixels for stretch: {0:.2f}%".format(100.0 * float(np.mean(avg_amp_valid))))
    phase_lut = gamma_insar_phase_lut()

    flat_view_u8 = wrapped_phase_to_u8(flat_phase, flat_valid)
    unw_view_u8 = stretch_to_u8(unw_phase, unw_valid, percent=args.percent)[0]

    # Prepare products.
    products = [
        {
            "name": "flat_phase",
            "raw": flat_phase.astype(np.float32),
            "raw_dtype": gdal.GDT_Float32,
            "raw_nodata": np.nan,
            "view": colorize_u8_with_lut(flat_view_u8, phase_lut),
            "view_dtype": gdal.GDT_Byte,
            "template": flat_ds,
        },
        {
            "name": "unwrapped_phase",
            "raw": unw_phase.astype(np.float32),
            "raw_dtype": gdal.GDT_Float32,
            "raw_nodata": np.nan,
            "view": colorize_u8_with_lut(unw_view_u8, phase_lut),
            "view_dtype": gdal.GDT_Byte,
            "template": unw_ds,
        },
        {
            "name": "avg_intensity_log",
            "raw": log_intensity.astype(np.float32),
            "raw_dtype": gdal.GDT_Float32,
            "raw_nodata": None,
            "view": stretch_to_u8(log_intensity, avg_amp_valid, percent=args.percent)[0],
            "view_dtype": gdal.GDT_Byte,
            "view_nodata": None,
            "template": unw_ds,
        },
        {
            "name": "los_displacement_m",
            "raw": los_disp,
            "raw_dtype": gdal.GDT_Float32,
            "raw_nodata": np.nan,
            "view": stretch_to_u8(los_disp, disp_valid, percent=args.percent)[0],
            "view_dtype": gdal.GDT_Byte,
            "template": unw_ds,
        },
        {
            "name": "look_angle_deg",
            "raw": look_angle.astype(np.float32),
            "raw_dtype": gdal.GDT_Float32,
            "raw_nodata": np.nan,
            "view": stretch_to_u8(look_angle, look_valid, percent=args.percent)[0],
            "view_dtype": gdal.GDT_Byte,
            "template": los_ds,
        },
    ]

    # Write geographic GeoTIFF first.
    raw_tifs = {}
    view_tifs = {}
    for p in products:
        raw_tif = os.path.join(outdir, "{0}.tif".format(p["name"]))
        view_tif = os.path.join(outdir, "{0}.view.tif".format(p["name"]))
        write_tiff(raw_tif, p["raw"], p["template"], p["raw_dtype"], nodata=p["raw_nodata"])
        write_tiff(
            view_tif,
            p["view"],
            p["template"],
            p.get("view_dtype", gdal.GDT_Byte),
            nodata=p.get("view_nodata", 0 if np.asarray(p["view"]).ndim == 2 else None),
        )
        raw_tifs[p["name"]] = raw_tif
        view_tifs[p["name"]] = view_tif

    # Optionally warp to UTM at integer posting.
    if args.to_utm:
        lon, lat = get_center_lonlat(unw_ds)
        utm_epsg = int(args.utm_epsg) if args.utm_epsg else utm_epsg_from_lonlat(lon, lat)

        if args.utm_res_mode == "manual":
            if args.utm_res is not None:
                xres = float(args.utm_res)
                yres = float(args.utm_res)
            else:
                if (args.utm_res_x is None) or (args.utm_res_y is None):
                    raise RuntimeError("Manual mode requires --utm-res or both --utm-res-x/--utm-res-y.")
                xres = float(args.utm_res_x)
                yres = float(args.utm_res_y)
        elif args.utm_res_mode == "native":
            xres, yres = estimate_native_res_m(unw_ds)
        else:
            rg_lk = args.range_looks
            az_lk = args.azimuth_looks
            if (rg_lk is None) or (az_lk is None):
                xml_rg, xml_az = parse_tops_looks(tops_xml)
                rg_lk = rg_lk if rg_lk is not None else (xml_rg if xml_rg is not None else 19)
                az_lk = az_lk if az_lk is not None else (xml_az if xml_az is not None else 7)
            try:
                xres, yres, inc, vel = estimate_multilook_res_m(proc_dir, rg_lk, az_lk, los_ds)
                print(
                    "INFO: multilook estimate: rangeLooks={0}, azLooks={1}, incidence={2:.3f} deg, "
                    "velocity={3:.3f} m/s, xRes={4:.3f} m, yRes={5:.3f} m".format(
                        rg_lk, az_lk, inc, vel, xres, yres
                    )
                )
            except Exception as err:
                print(
                    "WARNING: multilook resolution estimation failed ({0}); fallback to native resolution.".format(err)
                )
                xres, yres = estimate_native_res_m(unw_ds)

        xres_i, yres_i = integerize_resolution(xres, yres, square=args.square_pixel)
        print(
            "INFO: UTM export EPSG:{0}, integer posting xRes={1} m, yRes={2} m".format(
                utm_epsg, xres_i, yres_i
            )
        )

        warped_raw = {}
        warped_view = {}
        for name, src in raw_tifs.items():
            dst = os.path.join(outdir, "{0}.utm.tif".format(name))
            warp_to_utm(src, dst, utm_epsg, xres_i, yres_i)
            warped_raw[name] = dst
        for name, src in view_tifs.items():
            dst = os.path.join(outdir, "{0}.view.utm.tif".format(name))
            warp_to_utm(src, dst, utm_epsg, xres_i, yres_i)
            warped_view[name] = dst
        raw_tifs = warped_raw
        view_tifs = warped_view

    # Export PNG and KMZ from view GeoTIFF.
    outputs = []
    for name, view_tif in view_tifs.items():
        png = os.path.join(outdir, "{0}.png".format(name))
        kmz = os.path.join(outdir, "{0}.kmz".format(name))
        geotiff_to_png(view_tif, png)
        geotiff_to_kmz(view_tif, kmz)
        outputs.extend([raw_tifs[name], view_tif, png, kmz])

    print("DONE. Generated outputs:")
    for p in outputs:
        print("  {0}".format(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
