#!/usr/bin/env python3
"""Convert ISCE-style rasters to PNG and GeoTIFF products.

Features:
- Supports complex / integer / float rasters via GDAL-openable inputs.
- 98% contrast stretch (or configurable) to uint8 for view products.
- Complex rasters export both phase (HSV) and intensity visualizations.
- Exports both "view" GeoTIFF(s) and "raw" GeoTIFF preserving data type.
"""

import argparse
import os
import sys

import numpy as np
from osgeo import gdal


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ISCE raster to PNG and GeoTIFF (raw + view)."
    )
    p.add_argument("input", help="Input raster path (file/.vrt/.xml).")
    p.add_argument(
        "--outdir", default=".", help="Output directory (default: current directory)."
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Output prefix (default: basename of input without extension).",
    )
    p.add_argument(
        "--percent",
        type=float,
        default=98.0,
        help="Percentile stretch span for view outputs, default 98.0 (1%%-99%%).",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "phase", "intensity"],
        default="auto",
        help="For real-valued rasters: force phase/intensity view mode. Default auto.",
    )
    p.add_argument(
        "--band",
        type=int,
        default=1,
        help="Band index for non-complex data (1-based), default 1.",
    )
    p.add_argument(
        "--phase-cmap",
        default="hsv",
        help="Phase colormap name (default: hsv, gamma-like cyclic).",
    )
    return p.parse_args()


def resolve_input(path):
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


def base_prefix(path):
    name = os.path.basename(path)
    for ext in (".vrt", ".xml", ".tif", ".tiff"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def percentile_clip_to_u8(data, span_percent=98.0):
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8), 0.0, 1.0

    tail = max((100.0 - span_percent) / 2.0, 0.0)
    pmin, pmax = np.percentile(arr[finite], [tail, 100.0 - tail])

    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        pmin = float(np.min(arr[finite]))
        pmax = float(np.max(arr[finite]))
        if pmax <= pmin:
            return np.zeros(arr.shape, dtype=np.uint8), pmin, pmax

    scaled = (arr - pmin) * (255.0 / (pmax - pmin))
    scaled[~finite] = 0
    return np.clip(scaled, 0, 255).astype(np.uint8), pmin, pmax


def phase_to_rgb(phase_rad, cmap_name="hsv"):
    import matplotlib.cm as cm

    wrapped = np.mod(phase_rad + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(wrapped)
    rgb = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return rgb


def write_png(path, arr):
    from PIL import Image

    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        raise ValueError(f"Unsupported PNG array shape: {arr.shape}")


def numpy_to_gdal_dtype(arr):
    dt = arr.dtype
    if np.issubdtype(dt, np.uint8):
        return gdal.GDT_Byte
    if np.issubdtype(dt, np.int16):
        return gdal.GDT_Int16
    if np.issubdtype(dt, np.uint16):
        return gdal.GDT_UInt16
    if np.issubdtype(dt, np.int32):
        return gdal.GDT_Int32
    if np.issubdtype(dt, np.uint32):
        return gdal.GDT_UInt32
    if np.issubdtype(dt, np.float32):
        return gdal.GDT_Float32
    if np.issubdtype(dt, np.float64):
        return gdal.GDT_Float64
    if np.issubdtype(dt, np.complex64):
        return gdal.GDT_CFloat32
    if np.issubdtype(dt, np.complex128):
        return gdal.GDT_CFloat64
    return gdal.GDT_Float32


def write_tiff(path, arr, ds_template=None, gdal_dtype=None):
    arr = np.asarray(arr)
    if gdal_dtype is None:
        gdal_dtype = numpy_to_gdal_dtype(arr)

    if arr.ndim == 2:
        ny, nx = arr.shape
        bands = 1
    elif arr.ndim == 3 and arr.shape[2] >= 1:
        ny, nx, bands = arr.shape
    else:
        raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")

    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(path, nx, ny, bands, gdal_dtype, options=["COMPRESS=LZW"])
    if out is None:
        raise RuntimeError(f"Failed to create {path}")

    if ds_template is not None:
        gt = ds_template.GetGeoTransform(can_return_null=True)
        if gt:
            out.SetGeoTransform(gt)
        proj = ds_template.GetProjection()
        if proj:
            out.SetProjection(proj)

    if bands == 1:
        out.GetRasterBand(1).WriteArray(arr)
    else:
        for i in range(bands):
            out.GetRasterBand(i + 1).WriteArray(arr[..., i])

    out.FlushCache()
    out = None


def main():
    args = parse_args()
    in_path = resolve_input(args.input)

    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"ERROR: Failed to open input raster: {args.input}", file=sys.stderr)
        return 2

    if args.band < 1 or args.band > ds.RasterCount:
        print(
            f"ERROR: band {args.band} is out of range [1, {ds.RasterCount}]",
            file=sys.stderr,
        )
        return 2

    band = ds.GetRasterBand(args.band)
    arr = band.ReadAsArray()
    if arr is None:
        print("ERROR: Failed to read raster data.", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or base_prefix(in_path)

    out_files = []
    raw_tif = os.path.join(args.outdir, f"{prefix}.raw.tif")

    # Raw GeoTIFF preserving original values/type
    write_tiff(raw_tif, arr, ds_template=ds)
    out_files.append(raw_tif)

    is_complex = np.iscomplexobj(arr)

    if is_complex:
        phase = np.angle(arr)
        intensity = np.abs(arr)

        phase_rgb = phase_to_rgb(phase, cmap_name=args.phase_cmap)
        intensity_u8, pmin, pmax = percentile_clip_to_u8(intensity, args.percent)

        phase_png = os.path.join(args.outdir, f"{prefix}.phase.png")
        intensity_png = os.path.join(args.outdir, f"{prefix}.intensity.png")
        phase_view_tif = os.path.join(args.outdir, f"{prefix}.phase.view.tif")
        intensity_view_tif = os.path.join(args.outdir, f"{prefix}.intensity.view.tif")

        write_png(phase_png, phase_rgb)
        write_png(intensity_png, intensity_u8)
        write_tiff(phase_view_tif, phase_rgb, ds_template=ds, gdal_dtype=gdal.GDT_Byte)
        write_tiff(
            intensity_view_tif,
            intensity_u8,
            ds_template=ds,
            gdal_dtype=gdal.GDT_Byte,
        )

        out_files.extend([phase_png, intensity_png, phase_view_tif, intensity_view_tif])
        print(f"INFO: Complex intensity stretch: min={pmin:.6g}, max={pmax:.6g}")

    else:
        real_mode = args.mode
        if real_mode == "auto":
            real_mode = "intensity"

        if real_mode == "phase":
            phase_rgb = phase_to_rgb(arr, cmap_name=args.phase_cmap)
            png = os.path.join(args.outdir, f"{prefix}.phase.png")
            view_tif = os.path.join(args.outdir, f"{prefix}.phase.view.tif")
            write_png(png, phase_rgb)
            write_tiff(view_tif, phase_rgb, ds_template=ds, gdal_dtype=gdal.GDT_Byte)
            out_files.extend([png, view_tif])
        else:
            u8, pmin, pmax = percentile_clip_to_u8(arr, args.percent)
            png = os.path.join(args.outdir, f"{prefix}.png")
            view_tif = os.path.join(args.outdir, f"{prefix}.view.tif")
            write_png(png, u8)
            write_tiff(view_tif, u8, ds_template=ds, gdal_dtype=gdal.GDT_Byte)
            out_files.extend([png, view_tif])
            print(f"INFO: Stretch: min={pmin:.6g}, max={pmax:.6g}")

    print("DONE. Outputs:")
    for p in out_files:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
