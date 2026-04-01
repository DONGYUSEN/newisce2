#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project a UTM optical image to SAR range/azimuth grid.
将 UTM 投影的光学影像重采样到 SAR 距离-方位坐标网格。

Supported SAR reference inputs:
支持的 SAR 参考输入：
1) Sentinel-1 SAFE zip (SLC or GRD): reads annotation XML from zip.
   Sentinel-1 SAFE 压缩包（SLC/GRD）：从 zip 内 annotation XML 读取参数。
2) Sentinel annotation XML directly.
   直接使用 Sentinel annotation XML。
3) ISCE product XML: reads grid parameters from XML; can use existing lat/lon
   lookups or auto-generate them from DEM + XML parameters.
   ISCE 产品 XML：从 XML 读取网格参数；可使用已有 lat/lon 查找表，
   或自动下载 DEM 并根据 XML 参数生成 lat/lon 查找表。

Output:
输出：
- GeoTIFF on SAR pixel grid (width/length exactly match SAR reference).
  基于 SAR 像元网格的 GeoTIFF（宽度/长度与 SAR 参考严格一致）。
- Pixel spacing metadata follows SAR XML parameters.
  像元间距元数据与 SAR XML 参数一致。
"""

import argparse
import math
import os
import zipfile
import xml.etree.ElementTree as ET
import datetime

import numpy as np
from osgeo import gdal
from osgeo import osr
from pyproj import CRS
from pyproj import Transformer
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata


def _find_text_anyns(root, path):
    xpath = ".//" + "/".join("{*}%s" % seg for seg in path.split("/"))
    node = root.find(xpath)
    if node is None:
        node = root.find(".//" + path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _as_float(value, name):
    if value is None:
        raise ValueError("Missing required field: %s" % name)
    return float(value)


def _as_int(value, name):
    if value is None:
        raise ValueError("Missing required field: %s" % name)
    return int(float(value))


def _read_sentinel_annotation_meta(root):
    lines = _as_int(_find_text_anyns(root, "imageAnnotation/imageInformation/numberOfLines"), "numberOfLines")
    samples = _as_int(_find_text_anyns(root, "imageAnnotation/imageInformation/numberOfSamples"), "numberOfSamples")

    rg_spacing = _find_text_anyns(root, "imageAnnotation/imageInformation/rangePixelSpacing")
    if rg_spacing is None:
        rg_spacing = _find_text_anyns(root, "imageAnnotation/imageInformation/groundRangePixelSpacing")
    az_spacing = _find_text_anyns(root, "imageAnnotation/imageInformation/azimuthPixelSpacing")

    rg_spacing = _as_float(rg_spacing, "rangePixelSpacing/groundRangePixelSpacing")
    az_spacing = _as_float(az_spacing, "azimuthPixelSpacing")

    points = []
    glist = root.findall(".//{*}geolocationGridPoint")
    if not glist:
        glist = root.findall(".//geolocationGridPoint")

    for p in glist:
        line = p.find("{*}line")
        if line is None:
            line = p.find("line")
        pixel = p.find("{*}pixel")
        if pixel is None:
            pixel = p.find("pixel")
        lat = p.find("{*}latitude")
        if lat is None:
            lat = p.find("latitude")
        lon = p.find("{*}longitude")
        if lon is None:
            lon = p.find("longitude")

        if line is None or pixel is None or lat is None or lon is None:
            continue
        points.append((float(line.text), float(pixel.text), float(lat.text), float(lon.text)))

    if len(points) < 4:
        raise RuntimeError("Annotation XML has insufficient geolocationGridPoint records.")

    return {
        "width": samples,
        "length": lines,
        "range_spacing": rg_spacing,
        "az_spacing": az_spacing,
        "geo_points": points,
    }


def _select_annotation_from_zip(zip_path, product_type=None, swath=None, pol=None):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if "/annotation/" in n and n.endswith(".xml") and "/annotation/calibration/" not in n]
        if not names:
            raise RuntimeError("No annotation XML found in zip: %s" % zip_path)

        cand = sorted(names)

        if product_type:
            token = "-%s-" % product_type.lower()
            cand2 = [n for n in cand if token in os.path.basename(n).lower()]
            if cand2:
                cand = cand2

        if swath:
            sw = swath.lower()
            cand2 = [n for n in cand if sw in os.path.basename(n).lower()]
            if cand2:
                cand = cand2

        if pol:
            pl = "-%s-" % pol.lower()
            cand2 = [n for n in cand if pl in os.path.basename(n).lower()]
            if cand2:
                cand = cand2

        if len(cand) > 1:
            print("WARNING: multiple annotation XML candidates, using first / 候选 annotation XML 多个，使用第一个: %s" % cand[0])

        chosen = cand[0]
        xml_bytes = zf.read(chosen)
    root = ET.fromstring(xml_bytes)
    return chosen, root


def _load_isce_product(xml_path, burst_index):
    try:
        from iscesys.Component.ProductManager import ProductManager as PM
    except Exception as err:
        raise RuntimeError("Cannot import ISCE ProductManager. Activate ISCE env first. %s" % err)

    pm = PM()
    pm.configure()
    obj = pm.loadProduct(xml_path)

    def pick_attr(target, names):
        for key in names:
            if hasattr(target, key):
                val = getattr(target, key)
                if val is not None:
                    return val
        return None

    target = obj
    if (not hasattr(obj, "numberOfSamples") or not hasattr(obj, "numberOfLines")) and hasattr(obj, "bursts"):
        if len(obj.bursts) == 0:
            raise RuntimeError("ISCE XML has bursts but empty burst list: %s" % xml_path)
        idx = burst_index
        if idx < 0:
            idx = len(obj.bursts) // 2
        idx = max(0, min(len(obj.bursts) - 1, idx))
        target = obj.bursts[idx]
        print("INFO: using burst index %d from bursts list / 从 bursts 列表中选择的 burst 索引: %d" % (idx, idx))

    width = pick_attr(target, ["numberOfSamples", "width"])
    length = pick_attr(target, ["numberOfLines", "length"])
    rg_spacing = pick_attr(target, ["groundRangePixelSize", "rangePixelSize"])
    az_spacing = pick_attr(target, ["azimuthPixelSize"])

    if width is None or length is None:
        raise RuntimeError("Cannot read numberOfSamples/numberOfLines from ISCE XML: %s" % xml_path)
    if rg_spacing is None or az_spacing is None:
        raise RuntimeError("Cannot read range/azimuth pixel spacing from ISCE XML: %s" % xml_path)

    meta = {
        "width": int(width),
        "length": int(length),
        "range_spacing": float(rg_spacing),
        "az_spacing": float(az_spacing),
        "geo_points": None,
        "target_kind": type(target).__name__,
        "target_index": None if target is obj else int(idx),
    }

    return meta, obj, target


def _auto_find_lookup(xml_path):
    base = os.path.dirname(os.path.abspath(xml_path))
    parent = os.path.dirname(base)
    candidates = [
        (os.path.join(base, "lat.rdr.vrt"), os.path.join(base, "lon.rdr.vrt")),
        (os.path.join(base, "lat.rdr"), os.path.join(base, "lon.rdr")),
        (os.path.join(base, "geometry", "lat.rdr.vrt"), os.path.join(base, "geometry", "lon.rdr.vrt")),
        (os.path.join(base, "geometry", "lat.rdr"), os.path.join(base, "geometry", "lon.rdr")),
        (os.path.join(parent, "geometry", "lat.rdr.vrt"), os.path.join(parent, "geometry", "lon.rdr.vrt")),
        (os.path.join(parent, "geometry", "lat.rdr"), os.path.join(parent, "geometry", "lon.rdr")),
    ]
    for latp, lonp in candidates:
        if os.path.isfile(latp) and os.path.isfile(lonp):
            return latp, lonp
    return None, None


class _BboxInfo(object):
    def __init__(self, snwe):
        self.extremes = snwe

    def getExtremes(self, _pad):
        return self.extremes


def _round_snwe(snwe):
    return [
        float(math.floor(float(snwe[0]))),
        float(math.ceil(float(snwe[1]))),
        float(math.floor(float(snwe[2]))),
        float(math.ceil(float(snwe[3]))),
    ]


def _estimate_bbox_from_isce_target(target, zrange=(-500.0, 9000.0), margin=0.2):
    if hasattr(target, "getBbox"):
        bbox = target.getBbox(list(zrange))
        return _round_snwe([
            float(bbox[0]) - margin,
            float(bbox[1]) + margin,
            float(bbox[2]) - margin,
            float(bbox[3]) + margin,
        ])

    orbit = getattr(target, "orbit", None)
    if orbit is None:
        raise RuntimeError("Cannot estimate bbox: target has no orbit.")

    if hasattr(target, "startingRange"):
        r0 = float(target.startingRange)
    else:
        raise RuntimeError("Cannot estimate bbox: missing startingRange.")

    if hasattr(target, "farRange"):
        r1 = float(target.farRange)
    elif hasattr(target, "getFarRange"):
        r1 = float(target.getFarRange())
    elif hasattr(target, "numberOfSamples") and hasattr(target, "rangePixelSize"):
        r1 = float(target.startingRange) + (float(target.numberOfSamples) - 1.0) * float(target.rangePixelSize)
    else:
        raise RuntimeError("Cannot estimate bbox: missing farRange/getFarRange.")

    t0 = getattr(target, "sensingStart", None)
    if t0 is None and hasattr(target, "getSensingStart"):
        t0 = target.getSensingStart()
    t1 = getattr(target, "sensingStop", None)
    if t1 is None and hasattr(target, "getSensingStop"):
        t1 = target.getSensingStop()
    if t0 is None or t1 is None:
        raise RuntimeError("Cannot estimate bbox: missing sensingStart/sensingStop.")

    tarr = []
    tdiff = (t1 - t0).total_seconds()
    for ind in range(11):
        tarr.append(t0 + datetime.timedelta(seconds=ind * tdiff / 10.0))

    look_side = None
    if hasattr(target, "side"):
        look_side = int(target.side)
    else:
        ins = getattr(target, "instrument", None)
        if ins is not None and hasattr(ins, "platform") and hasattr(ins.platform, "pointingDirection"):
            look_side = int(ins.platform.pointingDirection)

    wvl = None
    if hasattr(target, "radarWavelength"):
        wvl = float(target.radarWavelength)
    elif hasattr(target, "radarWavelegth"):
        wvl = float(target.radarWavelegth)
    else:
        ins = getattr(target, "instrument", None)
        if ins is not None and hasattr(ins, "getRadarWavelength"):
            wvl = float(ins.getRadarWavelength())

    doppler = None
    if hasattr(target, "_dopplerVsPixel") and hasattr(target, "startingRange") and hasattr(target, "instrument"):
        try:
            from isceobj.Util.Poly2D import Poly2D
            doppler = Poly2D()
            doppler._meanRange = float(target.startingRange)
            doppler._normRange = float(target.instrument.rangePixelSize)
            coeff = [float(x) for x in target._dopplerVsPixel]
            doppler.initPoly(azimuthOrder=0, rangeOrder=len(coeff) - 1, coeffs=[coeff])
        except Exception:
            doppler = None

    llh = []
    for z in [float(zrange[0]), float(zrange[1])]:
        for taz in tarr:
            for rng in [r0, r1]:
                pt = None
                if doppler is not None and wvl is not None and look_side is not None:
                    try:
                        pt = orbit.rdr2geo(taz, rng, doppler=doppler, height=z, wvl=wvl, side=look_side)
                    except Exception:
                        pt = None
                if pt is None and look_side is not None:
                    try:
                        pt = orbit.rdr2geo(taz, rng, height=z, side=look_side)
                    except Exception:
                        pt = None
                if pt is None:
                    pt = orbit.rdr2geo(taz, rng, height=z)

                pt = np.asarray(pt, dtype=np.float64)
                if np.sum(np.isnan(pt)) > 0:
                    sv = orbit.interpolateOrbit(taz, method="hermite")
                    if hasattr(target, "ellipsoid"):
                        pt = np.asarray(target.ellipsoid.xyz_to_llh(sv.getPosition()), dtype=np.float64)
                    elif hasattr(target, "planet") and hasattr(target.planet, "ellipsoid"):
                        pt = np.asarray(target.planet.ellipsoid.xyz_to_llh(sv.getPosition()), dtype=np.float64)
                llh.append(pt)

    llh = np.asarray(llh, dtype=np.float64)
    if llh.ndim != 2 or llh.shape[0] < 4:
        raise RuntimeError("Failed to estimate bbox from target orbit geometry.")

    bbox = [
        float(np.nanmin(llh[:, 0])) - margin,
        float(np.nanmax(llh[:, 0])) + margin,
        float(np.nanmin(llh[:, 1])) - margin,
        float(np.nanmax(llh[:, 1])) + margin,
    ]
    return _round_snwe(bbox)


def _download_dem_for_target(target, dem_path=None, dem_dir=None, zrange=(-500.0, 9000.0), use_high_res_only=False):
    import isceobj
    from iscesys.DataManager import createManager
    from isceobj.Util.ImageUtil import DemImageLib

    if dem_path:
        dem_base = os.path.abspath(dem_path)
        if dem_base.endswith(".xml"):
            dem_base = dem_base[:-4]
        if not os.path.isfile(dem_base + ".xml"):
            raise RuntimeError("DEM XML not found: %s.xml" % dem_base)
        return dem_base

    if dem_dir is None:
        dem_dir = os.path.join(os.getcwd(), "optical2sar_dem")
    dem_dir = os.path.abspath(dem_dir)
    os.makedirs(dem_dir, exist_ok=True)

    snwe = _estimate_bbox_from_isce_target(target, zrange=zrange, margin=0.2)
    print("INFO: DEM bbox estimate snwe=%s / 自动估计 DEM 范围 snwe=%s" % (snwe, snwe))

    class DemContainer(object):
        pass

    holder = DemContainer()
    holder.demStitcher = createManager("dem1", "iscestitcher")
    holder.useHighResolutionDemOnly = bool(use_high_res_only)
    holder.proceedIfZeroDem = True
    holder.demImage = None

    holder.demStitcher.noFilling = False
    holder.demStitcher.downloadDir = dem_dir

    old_cwd = os.getcwd()
    try:
        os.chdir(dem_dir)
        DemImageLib.createDem(
            snwe,
            _BboxInfo(snwe),
            holder,
            holder.demStitcher,
            holder.useHighResolutionDemOnly,
            True,
        )
    finally:
        os.chdir(old_cwd)

    if holder.demImage is None:
        raise RuntimeError("DEM creation failed: demImage is None.")

    dem_base = holder.demImage.filename
    if not os.path.isabs(dem_base):
        dem_base = os.path.abspath(os.path.join(dem_dir, dem_base))

    if not os.path.isfile(dem_base + ".xml"):
        alt = os.path.abspath(os.path.join(dem_dir, os.path.basename(dem_base)))
        if os.path.isfile(alt + ".xml"):
            dem_base = alt

    if not os.path.isfile(dem_base + ".xml"):
        raise RuntimeError("DEM metadata missing after download: %s.xml" % dem_base)

    print("INFO: DEM ready: %s / DEM 已就绪: %s" % (dem_base, dem_base))
    return dem_base


def _render_rdr_vrt(path_no_ext):
    import isceobj
    xml_path = path_no_ext + ".xml"
    if not os.path.isfile(xml_path):
        raise RuntimeError("RDR XML not found: %s" % xml_path)
    if not os.path.isfile(path_no_ext + ".vrt"):
        img = isceobj.createImage()
        img.load(xml_path)
        img.renderVRT()


def _ensure_topo_inputs(target, width, length):
    from isceobj.Constants import SPEED_OF_LIGHT

    if hasattr(target, "rangePixelSize") and hasattr(target, "azimuthTimeInterval") and hasattr(target, "startingRange"):
        dr = float(target.rangePixelSize)
        prf = 1.0 / float(target.azimuthTimeInterval)
        r0 = float(target.startingRange)
        return dr, prf, r0

    if hasattr(target, "rangeSamplingRate") and hasattr(target, "PRF") and hasattr(target, "startingRange"):
        dr = 0.5 * SPEED_OF_LIGHT / float(target.rangeSamplingRate)
        prf = float(target.PRF)
        r0 = float(target.startingRange)
        return dr, prf, r0

    if hasattr(target, "nearSlantRange") and hasattr(target, "farSlantRange") and hasattr(target, "azimuthTimeInterval"):
        if int(width) < 2:
            raise RuntimeError("Invalid width for near/far slant range conversion.")
        dr = (float(target.farSlantRange) - float(target.nearSlantRange)) / float(int(width) - 1)
        prf = 1.0 / float(target.azimuthTimeInterval)
        r0 = float(target.nearSlantRange)
        return dr, prf, r0

    raise RuntimeError("Cannot derive topo input spacing from XML target.")


def _extract_radar_wavelength(target):
    if hasattr(target, "radarWavelength"):
        return float(target.radarWavelength)
    if hasattr(target, "radarWavelegth"):
        return float(target.radarWavelegth)
    ins = getattr(target, "instrument", None)
    if ins is not None and hasattr(ins, "getRadarWavelength"):
        return float(ins.getRadarWavelength())
    raise RuntimeError("Cannot derive radar wavelength from XML target.")


def _extract_look_side(target):
    if hasattr(target, "side"):
        return int(target.side)
    ins = getattr(target, "instrument", None)
    if ins is not None and hasattr(ins, "platform") and hasattr(ins.platform, "pointingDirection"):
        return int(ins.platform.pointingDirection)
    return -1


def _extract_sensing_start(target):
    t0 = getattr(target, "sensingStart", None)
    if t0 is None and hasattr(target, "getSensingStart"):
        t0 = target.getSensingStart()
    if t0 is None:
        raise RuntimeError("Cannot derive sensingStart from XML target.")
    return t0


def _generate_rdr_from_xml_target(target, width, length, dem_base, rdr_dir, stem="auto"):
    import isceobj
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet

    os.makedirs(rdr_dir, exist_ok=True)

    dem_img = isceobj.createDemImage()
    dem_img.load(dem_base + ".xml")

    dr, prf, r0 = _ensure_topo_inputs(target, width, length)

    topo = createTopozero()
    topo.orbit = target.orbit
    topo.width = int(width)
    topo.length = int(length)
    topo.prf = prf
    topo.slantRangePixelSpacing = dr
    topo.rangeFirstSample = r0
    topo.radarWavelength = _extract_radar_wavelength(target)
    topo.sensingStart = _extract_sensing_start(target)
    topo.lookSide = _extract_look_side(target)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.demInterpolationMethod = "BIQUINTIC"
    topo.wireInputPort(name="dem", object=dem_img)
    topo.wireInputPort(name="planet", object=Planet(pname="Earth"))

    lat_path = os.path.join(rdr_dir, "lat_%s.rdr" % stem)
    lon_path = os.path.join(rdr_dir, "lon_%s.rdr" % stem)
    hgt_path = os.path.join(rdr_dir, "hgt_%s.rdr" % stem)
    los_path = os.path.join(rdr_dir, "los_%s.rdr" % stem)
    topo.latFilename = lat_path
    topo.lonFilename = lon_path
    topo.heightFilename = hgt_path
    topo.losFilename = los_path

    topo.topo()
    _render_rdr_vrt(lat_path)
    _render_rdr_vrt(lon_path)

    lat_vrt = lat_path + ".vrt" if os.path.isfile(lat_path + ".vrt") else lat_path
    lon_vrt = lon_path + ".vrt" if os.path.isfile(lon_path + ".vrt") else lon_path
    return lat_vrt, lon_vrt


def _fill_grid_nans(line_nodes, pix_nodes, grid):
    if not np.isnan(grid).any():
        return grid
    ln, px = np.meshgrid(line_nodes, pix_nodes, indexing="ij")
    valid = np.isfinite(grid)
    if np.count_nonzero(valid) < 4:
        raise RuntimeError("Too few valid geolocation grid points after parsing.")

    pts = np.column_stack((ln[valid], px[valid]))
    vals = grid[valid]
    out = griddata(pts, vals, (ln, px), method="linear")
    miss = ~np.isfinite(out)
    if np.any(miss):
        out2 = griddata(pts, vals, (ln, px), method="nearest")
        out[miss] = out2[miss]
    return out


class AnnotationGridProvider(object):
    def __init__(self, points, width, length):
        self.width = int(width)
        self.length = int(length)
        self.pix = np.arange(self.width, dtype=np.float64)

        arr = np.asarray(points, dtype=np.float64)
        lines = np.unique(arr[:, 0])
        pixels = np.unique(arr[:, 1])
        lines.sort()
        pixels.sort()
        if lines.size < 2 or pixels.size < 2:
            raise RuntimeError("Invalid geolocation grid: need >=2 nodes in both line/pixel dimensions.")

        line_idx = {v: i for i, v in enumerate(lines.tolist())}
        pix_idx = {v: i for i, v in enumerate(pixels.tolist())}

        lat_grid = np.full((lines.size, pixels.size), np.nan, dtype=np.float64)
        lon_grid = np.full((lines.size, pixels.size), np.nan, dtype=np.float64)

        for ln, px, lat, lon in points:
            lat_grid[line_idx[ln], pix_idx[px]] = lat
            lon_grid[line_idx[ln], pix_idx[px]] = lon

        lat_grid = _fill_grid_nans(lines, pixels, lat_grid)
        lon_grid = _fill_grid_nans(lines, pixels, lon_grid)

        # Extend to image borders to avoid spline extrapolation at edges.
        if lines[0] > 0:
            lines = np.insert(lines, 0, 0.0)
            lat_grid = np.vstack((lat_grid[0:1, :], lat_grid))
            lon_grid = np.vstack((lon_grid[0:1, :], lon_grid))
        if lines[-1] < (self.length - 1):
            lines = np.append(lines, float(self.length - 1))
            lat_grid = np.vstack((lat_grid, lat_grid[-1:, :]))
            lon_grid = np.vstack((lon_grid, lon_grid[-1:, :]))

        if pixels[0] > 0:
            pixels = np.insert(pixels, 0, 0.0)
            lat_grid = np.hstack((lat_grid[:, 0:1], lat_grid))
            lon_grid = np.hstack((lon_grid[:, 0:1], lon_grid))
        if pixels[-1] < (self.width - 1):
            pixels = np.append(pixels, float(self.width - 1))
            lat_grid = np.hstack((lat_grid, lat_grid[:, -1:]))
            lon_grid = np.hstack((lon_grid, lon_grid[:, -1:]))

        self.lat_spl = RectBivariateSpline(lines, pixels, lat_grid, kx=1, ky=1)
        self.lon_spl = RectBivariateSpline(lines, pixels, lon_grid, kx=1, ky=1)

    def read_block(self, y0, y1):
        line_vec = np.arange(y0, y1, dtype=np.float64)
        lat = self.lat_spl(line_vec, self.pix)
        lon = self.lon_spl(line_vec, self.pix)
        return lat, lon


class RasterLookupProvider(object):
    def __init__(self, lat_path, lon_path, width, length):
        self.width = int(width)
        self.length = int(length)
        self.lat_ds = gdal.Open(lat_path, gdal.GA_ReadOnly)
        self.lon_ds = gdal.Open(lon_path, gdal.GA_ReadOnly)
        if self.lat_ds is None or self.lon_ds is None:
            raise RuntimeError("Cannot open lat/lon lookup files: %s , %s" % (lat_path, lon_path))
        if self.lat_ds.RasterXSize != self.width or self.lat_ds.RasterYSize != self.length:
            raise RuntimeError("lat lookup size mismatch. expected %dx%d, got %dx%d" % (
                self.width, self.length, self.lat_ds.RasterXSize, self.lat_ds.RasterYSize
            ))
        if self.lon_ds.RasterXSize != self.width or self.lon_ds.RasterYSize != self.length:
            raise RuntimeError("lon lookup size mismatch. expected %dx%d, got %dx%d" % (
                self.width, self.length, self.lon_ds.RasterXSize, self.lon_ds.RasterYSize
            ))
        self.lat_band = self.lat_ds.GetRasterBand(1)
        self.lon_band = self.lon_ds.GetRasterBand(1)

    def read_block(self, y0, y1):
        nlines = y1 - y0
        lat = self.lat_band.ReadAsArray(0, y0, self.width, nlines).astype(np.float64, copy=False)
        lon = self.lon_band.ReadAsArray(0, y0, self.width, nlines).astype(np.float64, copy=False)
        return lat, lon


def _build_reference(reference, product_type, swath, pol, burst_index, lat_rdr, lon_rdr,
                     dem=None, dem_dir=None, rdr_dir=None, dem_highres_only=False,
                     zmin=-500.0, zmax=9000.0):
    ref = os.path.abspath(reference)
    if ref.lower().endswith(".zip"):
        chosen, root = _select_annotation_from_zip(ref, product_type=product_type, swath=swath, pol=pol)
        meta = _read_sentinel_annotation_meta(root)
        provider = AnnotationGridProvider(meta["geo_points"], meta["width"], meta["length"])
        print("INFO: reference from zip annotation: %s / 使用 zip 内 annotation 作为参考: %s" % (chosen, chosen))
        return meta, provider

    if ref.lower().endswith(".xml"):
        root = ET.parse(ref).getroot()
        has_geogrid = root.find(".//{*}geolocationGridPoint") is not None or root.find(".//geolocationGridPoint") is not None
        if has_geogrid:
            meta = _read_sentinel_annotation_meta(root)
            provider = AnnotationGridProvider(meta["geo_points"], meta["width"], meta["length"])
            print("INFO: reference from annotation xml: %s / 使用 annotation XML 作为参考: %s" % (ref, ref))
            return meta, provider

        meta, _obj, target = _load_isce_product(ref, burst_index=burst_index)
        latp = lat_rdr
        lonp = lon_rdr
        explicit_lookup = (lat_rdr is not None) or (lon_rdr is not None)
        auto_generated = False

        if latp is None or lonp is None:
            latp, lonp = _auto_find_lookup(ref)
            if latp and lonp:
                print("INFO: auto found lat/lon lookup: %s , %s / 自动发现 lat/lon 查找表: %s , %s" % (latp, lonp, latp, lonp))

        if latp is None or lonp is None:
            zrange = (float(zmin), float(zmax))
            if rdr_dir is None:
                rdr_dir = os.path.join(os.path.dirname(ref), "optical2sar_rdr")
            if dem_dir is None:
                dem_dir = os.path.join(os.path.dirname(ref), "optical2sar_dem")
            dem_base = _download_dem_for_target(
                target=target,
                dem_path=dem,
                dem_dir=dem_dir,
                zrange=zrange,
                use_high_res_only=bool(dem_highres_only),
            )
            stem = "xml"
            if meta.get("target_index") is not None:
                stem = "burst%03d" % int(meta["target_index"])
            latp, lonp = _generate_rdr_from_xml_target(
                target=target,
                width=meta["width"],
                length=meta["length"],
                dem_base=dem_base,
                rdr_dir=rdr_dir,
                stem=stem,
            )
            print("INFO: generated lat/lon lookup: %s , %s / 已自动生成 lat/lon 查找表: %s , %s" % (latp, lonp, latp, lonp))
            auto_generated = True

        try:
            provider = RasterLookupProvider(latp, lonp, meta["width"], meta["length"])
        except Exception as err:
            if explicit_lookup or auto_generated:
                raise
            print("WARNING: discovered lat/lon lookup invalid (%s), trying DEM+topo generation. / 发现 lat/lon 查找表无效，尝试 DEM+topo 自动生成。" % str(err))
            zrange = (float(zmin), float(zmax))
            if rdr_dir is None:
                rdr_dir = os.path.join(os.path.dirname(ref), "optical2sar_rdr")
            if dem_dir is None:
                dem_dir = os.path.join(os.path.dirname(ref), "optical2sar_dem")
            dem_base = _download_dem_for_target(
                target=target,
                dem_path=dem,
                dem_dir=dem_dir,
                zrange=zrange,
                use_high_res_only=bool(dem_highres_only),
            )
            stem = "xml"
            if meta.get("target_index") is not None:
                stem = "burst%03d" % int(meta["target_index"])
            latp, lonp = _generate_rdr_from_xml_target(
                target=target,
                width=meta["width"],
                length=meta["length"],
                dem_base=dem_base,
                rdr_dir=rdr_dir,
                stem=stem,
            )
            provider = RasterLookupProvider(latp, lonp, meta["width"], meta["length"])
            print("INFO: regenerated lat/lon lookup: %s , %s / 已重新生成 lat/lon 查找表: %s , %s" % (latp, lonp, latp, lonp))
        print("INFO: reference from ISCE product xml: %s / 使用 ISCE 产品 XML 作为参考: %s" % (ref, ref))
        return meta, provider

    raise RuntimeError("Unsupported reference format. Use .zip or .xml")


def _bilinear_sample(src, px, py, nodata, fill_value):
    out = np.full(px.shape, fill_value, dtype=np.float32)
    if src.size == 0:
        return out

    h, w = src.shape
    finite = np.isfinite(px) & np.isfinite(py)
    if not np.any(finite):
        return out

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    valid = finite & (x0 >= 0) & (x0 < (w - 1)) & (y0 >= 0) & (y0 < (h - 1))
    if not np.any(valid):
        return out

    rr, cc = np.where(valid)
    x0v = x0[rr, cc]
    y0v = y0[rr, cc]
    x1v = x0v + 1
    y1v = y0v + 1

    fx = (px[rr, cc] - x0v).astype(np.float32)
    fy = (py[rr, cc] - y0v).astype(np.float32)

    q00 = src[y0v, x0v].astype(np.float32, copy=False)
    q10 = src[y0v, x1v].astype(np.float32, copy=False)
    q01 = src[y1v, x0v].astype(np.float32, copy=False)
    q11 = src[y1v, x1v].astype(np.float32, copy=False)

    if nodata is not None:
        if math.isnan(nodata):
            nod = np.isnan(q00) | np.isnan(q10) | np.isnan(q01) | np.isnan(q11)
        else:
            nod = (q00 == nodata) | (q10 == nodata) | (q01 == nodata) | (q11 == nodata)
    else:
        nod = np.zeros(q00.shape, dtype=bool)

    vals = (q00 * (1.0 - fx) * (1.0 - fy) +
            q10 * fx * (1.0 - fy) +
            q01 * (1.0 - fx) * fy +
            q11 * fx * fy)

    good = ~nod
    if np.any(good):
        out[rr[good], cc[good]] = vals[good]

    return out


def _build_transformer(src_ds, optical_epsg):
    proj = src_ds.GetProjection()
    if proj:
        target_crs = CRS.from_wkt(proj)
    elif optical_epsg:
        target_crs = CRS.from_epsg(int(optical_epsg))
    else:
        raise RuntimeError("Optical raster has no projection; provide --optical-epsg.")
    return Transformer.from_crs(CRS.from_epsg(4326), target_crs, always_xy=True)


def run(optical_tif, reference, out_tif, product_type=None, swath=None, pol=None,
        burst_index=-1, lat_rdr=None, lon_rdr=None, optical_epsg=None,
        dem=None, dem_dir=None, rdr_dir=None, dem_highres_only=False,
        zmin=-500.0, zmax=9000.0, block_lines=256, fill_value=-9999.0):
    gdal.UseExceptions()

    meta, geo_provider = _build_reference(
        reference=reference,
        product_type=product_type,
        swath=swath,
        pol=pol,
        burst_index=burst_index,
        lat_rdr=lat_rdr,
        lon_rdr=lon_rdr,
        dem=dem,
        dem_dir=dem_dir,
        rdr_dir=rdr_dir,
        dem_highres_only=dem_highres_only,
        zmin=zmin,
        zmax=zmax,
    )

    width = int(meta["width"])
    length = int(meta["length"])
    rg_spacing = float(meta["range_spacing"])
    az_spacing = float(meta["az_spacing"])

    src_ds = gdal.Open(optical_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError("Cannot open optical image: %s" % optical_tif)

    gt = src_ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise RuntimeError("Optical image has no geotransform.")
    inv_gt = gdal.InvGeoTransform(gt)
    if inv_gt is None:
        raise RuntimeError("Failed to invert optical geotransform.")

    transformer = _build_transformer(src_ds, optical_epsg)

    nb = src_ds.RasterCount
    if nb < 1:
        raise RuntimeError("Optical image has no bands.")

    src_arrays = []
    src_nodata = []
    for b in range(1, nb + 1):
        band = src_ds.GetRasterBand(b)
        arr = band.ReadAsArray()
        if arr is None:
            raise RuntimeError("Failed reading optical band %d." % b)
        src_arrays.append(arr.astype(np.float32, copy=False))
        nd = band.GetNoDataValue()
        src_nodata.append(float(nd) if nd is not None else None)

    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_tif,
        width,
        length,
        nb,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    if out_ds is None:
        raise RuntimeError("Cannot create output: %s" % out_tif)

    # SAR local grid: x -> range direction, y -> azimuth direction.
    out_ds.SetGeoTransform((0.0, rg_spacing, 0.0, 0.0, 0.0, az_spacing))
    srs_local = osr.SpatialReference()
    srs_local.SetLocalCS("SAR_RangeAzimuth")
    out_ds.SetProjection(srs_local.ExportToWkt())
    out_ds.SetMetadataItem("SAR_NUMBER_OF_SAMPLES", str(width))
    out_ds.SetMetadataItem("SAR_NUMBER_OF_LINES", str(length))
    out_ds.SetMetadataItem("SAR_RANGE_PIXEL_SIZE", str(rg_spacing))
    out_ds.SetMetadataItem("SAR_AZIMUTH_PIXEL_SIZE", str(az_spacing))
    out_ds.SetMetadataItem("SAR_REFERENCE", os.path.abspath(reference))

    for ib in range(1, nb + 1):
        out_ds.GetRasterBand(ib).SetNoDataValue(float(fill_value))

    for y0 in range(0, length, int(block_lines)):
        y1 = min(length, y0 + int(block_lines))
        lat_blk, lon_blk = geo_provider.read_block(y0, y1)

        xx, yy = transformer.transform(lon_blk, lat_blk)
        px = inv_gt[0] + inv_gt[1] * xx + inv_gt[2] * yy
        py = inv_gt[3] + inv_gt[4] * xx + inv_gt[5] * yy

        for ib in range(nb):
            out_blk = _bilinear_sample(
                src_arrays[ib],
                px,
                py,
                nodata=src_nodata[ib],
                fill_value=float(fill_value),
            )
            out_ds.GetRasterBand(ib + 1).WriteArray(out_blk, 0, y0)

        print("INFO: wrote lines %d-%d / %d / 已写入行 %d-%d / %d" % (y0, y1 - 1, length - 1, y0, y1 - 1, length - 1))

    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    print("DONE: %s / 处理完成: %s" % (os.path.abspath(out_tif), os.path.abspath(out_tif)))


def cmdline():
    parser = argparse.ArgumentParser(
        description="Project UTM optical raster to SAR range/azimuth grid using ISCE-compatible metadata. "
                    "基于 ISCE 元数据将 UTM 光学影像投影到 SAR 距离-方位坐标。"
    )
    parser.add_argument("--optical", required=True, help="Input optical GeoTIFF (typically UTM projected). 输入光学 GeoTIFF（通常为 UTM 投影）。")
    parser.add_argument("--reference", required=True, help="SAR reference (.zip SAFE or .xml). SAR 参考数据（.zip SAFE 或 .xml）。")
    parser.add_argument("--out", required=True, help="Output GeoTIFF on SAR grid. 输出到 SAR 网格的 GeoTIFF。")

    parser.add_argument("--product-type", choices=["slc", "grd"], default=None,
                        help="For zip input, optional product type filter. zip 输入时可选：产品类型过滤。")
    parser.add_argument("--swath", default=None, help="For zip input, optional swath filter (e.g. iw1, iw2). zip 输入时可选：波束过滤。")
    parser.add_argument("--pol", default=None, help="For zip input, optional polarization filter (e.g. vv, vh). zip 输入时可选：极化过滤。")
    parser.add_argument("--burst-index", type=int, default=-1,
                        help="For ISCE TOPS XML with bursts, burst index to use. -1 means middle burst. "
                             "ISCE TOPS XML 含 bursts 时使用的索引，-1 表示中间 burst。")

    parser.add_argument("--lat-rdr", default=None,
                        help="Lat lookup raster for ISCE product XML (if annotation geogrid is absent). "
                             "ISCE 产品 XML 使用的纬度查找表（无 annotation geogrid 时）。")
    parser.add_argument("--lon-rdr", default=None,
                        help="Lon lookup raster for ISCE product XML (if annotation geogrid is absent). "
                             "ISCE 产品 XML 使用的经度查找表（无 annotation geogrid 时）。")
    parser.add_argument("--dem", default=None,
                        help="DEM base path (without .xml). Used when lat/lon rdr are missing and need topo generation. "
                             "DEM 基路径（不含 .xml），在缺少 lat/lon rdr 时用于 topo 生成。")
    parser.add_argument("--dem-dir", default=None,
                        help="Directory for DEM download/cache when --dem is not provided. 未提供 --dem 时的 DEM 下载/缓存目录。")
    parser.add_argument("--rdr-dir", default=None,
                        help="Directory to write generated lat/lon/hgt/los rdr files. 生成 lat/lon/hgt/los rdr 文件的目录。")
    parser.add_argument("--dem-highres-only", action="store_true",
                        help="Use only high-resolution DEM source when auto downloading DEM. 自动下载 DEM 时仅使用高分辨率源。")
    parser.add_argument("--zmin", type=float, default=-500.0,
                        help="Minimum terrain height (m) for bbox/DEM estimation. bbox/DEM 估计使用的最小地形高程（米）。")
    parser.add_argument("--zmax", type=float, default=9000.0,
                        help="Maximum terrain height (m) for bbox/DEM estimation. bbox/DEM 估计使用的最大地形高程（米）。")
    parser.add_argument("--optical-epsg", type=int, default=None,
                        help="Only used when optical TIFF has no projection. 仅在光学 TIFF 无投影信息时使用。")

    parser.add_argument("--block-lines", type=int, default=256, help="Processing block size in SAR lines. SAR 行方向分块处理行数。")
    parser.add_argument("--fill-value", type=float, default=-9999.0, help="Output nodata fill value. 输出无效值填充值。")

    args = parser.parse_args()
    run(
        optical_tif=args.optical,
        reference=args.reference,
        out_tif=args.out,
        product_type=args.product_type,
        swath=args.swath,
        pol=args.pol,
        burst_index=args.burst_index,
        lat_rdr=args.lat_rdr,
        lon_rdr=args.lon_rdr,
        dem=args.dem,
        dem_dir=args.dem_dir,
        rdr_dir=args.rdr_dir,
        dem_highres_only=args.dem_highres_only,
        zmin=args.zmin,
        zmax=args.zmax,
        optical_epsg=args.optical_epsg,
        block_lines=args.block_lines,
        fill_value=args.fill_value,
    )


if __name__ == "__main__":
    cmdline()
