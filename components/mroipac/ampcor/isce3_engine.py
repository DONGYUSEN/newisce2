#!/usr/bin/env python3

import os
import shutil
import tempfile

import numpy as np


ENGINE_LEGACY = "legacy"
ENGINE_ISCE3_CPU = "isce3_cpu"
ENGINE_ISCE3_GPU = "isce3_gpu"
VALID_ENGINES = (ENGINE_LEGACY, ENGINE_ISCE3_CPU, ENGINE_ISCE3_GPU)


def normalize_engine(engine):
    """Normalize engine selector tokens to canonical values."""
    token = str(engine if engine is not None else ENGINE_LEGACY).strip().lower()
    aliases = {
        "legacy": ENGINE_LEGACY,
        "isce2": ENGINE_LEGACY,
        "cpu": ENGINE_ISCE3_CPU,
        "isce3": ENGINE_ISCE3_CPU,
        "isce3cpu": ENGINE_ISCE3_CPU,
        "isce3_cpu": ENGINE_ISCE3_CPU,
        "gpu": ENGINE_ISCE3_GPU,
        "isce3gpu": ENGINE_ISCE3_GPU,
        "isce3_gpu": ENGINE_ISCE3_GPU,
    }
    if token not in aliases:
        raise ValueError(
            "Unsupported ampcor engine '{0}'. Valid values: {1}".format(
                engine, ", ".join(VALID_ENGINES)
            )
        )
    return aliases[token]


def resolve_image_name(image_or_name):
    """
    Resolve image filename for isce3 ampcor inputs.
    Prefer existing .vrt when available, then raw filename.
    """
    if image_or_name is None:
        return None

    candidates = []
    if isinstance(image_or_name, str):
        candidates.append(image_or_name)
    else:
        for attr in ("filename", "_filename"):
            value = getattr(image_or_name, attr, None)
            if value:
                candidates.append(value)
        getter = getattr(image_or_name, "getFilename", None)
        if callable(getter):
            value = getter()
            if value:
                candidates.append(value)

    seen = set()
    ordered = []
    for item in candidates:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
        if item and (not item.endswith(".vrt")):
            vrt = item + ".vrt"
            if vrt not in seen:
                ordered.insert(0, vrt)
                seen.add(vrt)

    for path in ordered:
        if os.path.exists(path):
            return path

    # Fall back to the first candidate if file is not materialized yet.
    return ordered[-1] if ordered else None


def _import_isce3_ampcor(engine):
    """
    Import isce3 and return selected ampcor implementation class.
    If external isce3 bindings are unavailable, fall back to in-tree bundled
    implementations so isce2 can run without an installed isce3 package.
    """
    ext_err = None
    try:
        import isce3
        if engine == ENGINE_ISCE3_CPU and hasattr(isce3, "matchtemplate") \
                and hasattr(isce3.matchtemplate, "PyCPUAmpcor"):
            return isce3.matchtemplate.PyCPUAmpcor
        if engine == ENGINE_ISCE3_GPU and hasattr(isce3, "cuda") \
                and hasattr(isce3.cuda, "matchtemplate") \
                and hasattr(isce3.cuda.matchtemplate, "PyCuAmpcor"):
            return isce3.cuda.matchtemplate.PyCuAmpcor
        ext_err = RuntimeError(
            "isce3 package is present but required class for engine '{0}' is unavailable.".format(engine)
        )
    except Exception as err:
        ext_err = err

    try:
        from mroipac.ampcor import isce3_bundled
        if engine == ENGINE_ISCE3_CPU:
            return isce3_bundled.BundledPyCPUAmpcor
        if engine == ENGINE_ISCE3_GPU:
            return isce3_bundled.BundledPyCuAmpcor
    except Exception as bund_err:
        raise ImportError(
            "Failed to load engine '{0}' from both external isce3 and bundled implementations. "
            "External error: {1}; bundled error: {2}".format(engine, ext_err, bund_err)
        )

    raise ValueError("Legacy engine should not call _import_isce3_ampcor().")


def _create_empty_envi_dataset(path, width, length, bands):
    try:
        from osgeo import gdal
    except Exception as err:
        raise ImportError(
            "isce3 ampcor adapter requires osgeo.gdal to create output datasets: {0}".format(err)
        )

    drv = gdal.GetDriverByName("ENVI")
    if drv is None:
        raise RuntimeError("GDAL ENVI driver is unavailable.")

    ds = drv.Create(
        path,
        xsize=int(width),
        ysize=int(length),
        bands=int(bands),
        eType=gdal.GDT_Float32,
        options=["INTERLEAVE=BIP"],
    )
    if ds is None:
        raise RuntimeError("Failed to create ENVI dataset: {0}".format(path))
    ds = None


def _require_positive_int(value, label):
    try:
        ivalue = int(value)
    except Exception:
        raise ValueError("{0} must be an integer. Got: {1}".format(label, value))
    if ivalue <= 0:
        raise ValueError("{0} must be > 0. Got: {1}".format(label, value))
    return ivalue


def _set_optional_attr(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)
        return True
    return False


def run_ampcor_engine(
    engine,
    reference_path,
    secondary_path,
    reference_width,
    reference_length,
    secondary_width,
    secondary_length,
    first_sample_across,
    first_sample_down,
    skip_sample_across,
    skip_sample_down,
    number_window_across,
    number_window_down,
    window_size_width,
    window_size_height,
    search_window_size_width,
    search_window_size_height,
    across_gross_offset=0,
    down_gross_offset=0,
    oversampling_factor=16,
    zoom_window_size=8,
    corr_surface_oversampling_method=0,
    raw_data_oversampling_factor=1,
    corr_stat_window_size=21,
    deramp_method=1,
    deramp_axis=0,
    device_id=0,
    n_streams=2,
    chunk_window_across=64,
    chunk_window_down=1,
    mmap_size_gb=8,
    use_mmap=1,
    output_prefix=None,
    cleanup_workdir=True,
):
    """
    Run isce3 CPU/GPU ampcor and map outputs to isce2-compatible 1D arrays.
    """
    engine = normalize_engine(engine)
    if engine == ENGINE_LEGACY:
        raise ValueError("run_ampcor_engine() does not handle legacy engine.")

    ref_path = resolve_image_name(reference_path)
    sec_path = resolve_image_name(secondary_path)
    if not ref_path:
        raise ValueError("Reference image path is undefined for isce3 ampcor engine.")
    if not sec_path:
        raise ValueError("Secondary image path is undefined for isce3 ampcor engine.")

    ref_w = _require_positive_int(reference_width, "reference_width")
    ref_l = _require_positive_int(reference_length, "reference_length")
    sec_w = _require_positive_int(secondary_width, "secondary_width")
    sec_l = _require_positive_int(secondary_length, "secondary_length")

    first_ac = int(first_sample_across)
    first_dn = int(first_sample_down)
    skip_ac = _require_positive_int(skip_sample_across, "skip_sample_across")
    skip_dn = _require_positive_int(skip_sample_down, "skip_sample_down")

    n_ac = _require_positive_int(number_window_across, "number_window_across")
    n_dn = _require_positive_int(number_window_down, "number_window_down")

    win_w = _require_positive_int(window_size_width, "window_size_width")
    win_h = _require_positive_int(window_size_height, "window_size_height")
    srh_w = _require_positive_int(search_window_size_width, "search_window_size_width")
    srh_h = _require_positive_int(search_window_size_height, "search_window_size_height")

    AmpcorImpl = _import_isce3_ampcor(engine)
    obj = AmpcorImpl()

    _set_optional_attr(obj, "algorithm", 0)
    _set_optional_attr(obj, "derampMethod", int(deramp_method))
    _set_optional_attr(obj, "derampAxis", int(deramp_axis))

    obj.referenceImageName = str(ref_path)
    obj.referenceImageWidth = ref_w
    obj.referenceImageHeight = ref_l
    obj.secondaryImageName = str(sec_path)
    obj.secondaryImageWidth = sec_w
    obj.secondaryImageHeight = sec_l

    obj.windowSizeWidth = win_w
    obj.windowSizeHeight = win_h
    obj.halfSearchRangeAcross = srh_w
    obj.halfSearchRangeDown = srh_h

    obj.skipSampleAcross = skip_ac
    obj.skipSampleDown = skip_dn
    obj.referenceStartPixelAcrossStatic = first_ac
    obj.referenceStartPixelDownStatic = first_dn
    obj.numberWindowAcross = n_ac
    obj.numberWindowDown = n_dn

    obj.corrSurfaceOverSamplingMethod = int(corr_surface_oversampling_method)
    obj.corrSurfaceOverSamplingFactor = _require_positive_int(
        oversampling_factor, "oversampling_factor"
    )
    obj.corrSurfaceZoomInWindow = _require_positive_int(
        zoom_window_size, "zoom_window_size"
    )
    obj.rawDataOversamplingFactor = _require_positive_int(
        raw_data_oversampling_factor, "raw_data_oversampling_factor"
    )
    obj.corrStatWindowSize = _require_positive_int(
        corr_stat_window_size, "corr_stat_window_size"
    )

    obj.useMmap = int(use_mmap)
    obj.mmapSize = _require_positive_int(mmap_size_gb, "mmap_size_gb")
    obj.numberWindowAcrossInChunk = max(1, min(n_ac, int(chunk_window_across)))
    obj.numberWindowDownInChunk = max(1, min(n_dn, int(chunk_window_down)))
    obj.deviceID = int(device_id)
    obj.nStreams = max(1, int(n_streams))

    scratch = None
    remove_scratch = False
    if output_prefix is None:
        scratch = tempfile.mkdtemp(prefix="isce3_ampcor_")
        remove_scratch = bool(cleanup_workdir)
        base = os.path.join(scratch, "ampcor")
    else:
        base = os.path.abspath(str(output_prefix))
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)

    outputs = {
        "offset": base + "_dense_offsets",
        "gross": base + "_gross_offsets",
        "snr": base + "_snr",
        "cov": base + "_covariance",
        "corr": base + "_correlation_peak",
    }

    obj.offsetImageName = outputs["offset"]
    obj.grossOffsetImageName = outputs["gross"]
    obj.snrImageName = outputs["snr"]
    obj.covImageName = outputs["cov"]
    obj.corrImageName = outputs["corr"]
    obj.mergeGrossOffset = 1

    try:
        obj.setupParams()
        obj.setConstantGrossOffset(int(down_gross_offset), int(across_gross_offset))
        obj.checkPixelInImageRange()

        _create_empty_envi_dataset(outputs["offset"], n_ac, n_dn, 2)
        _create_empty_envi_dataset(outputs["gross"], n_ac, n_dn, 2)
        _create_empty_envi_dataset(outputs["snr"], n_ac, n_dn, 1)
        _create_empty_envi_dataset(outputs["cov"], n_ac, n_dn, 3)
        _create_empty_envi_dataset(outputs["corr"], n_ac, n_dn, 1)

        obj.runAmpcor()

        expected_2b = n_dn * n_ac * 2
        expected_1b = n_dn * n_ac
        expected_3b = n_dn * n_ac * 3

        offset_raw = np.fromfile(outputs["offset"], dtype=np.float32)
        snr_raw = np.fromfile(outputs["snr"], dtype=np.float32)
        cov_raw = np.fromfile(outputs["cov"], dtype=np.float32)

        if offset_raw.size != expected_2b:
            raise RuntimeError(
                "isce3 ampcor offset output size mismatch: got {0}, expected {1}".format(
                    offset_raw.size, expected_2b
                )
            )
        if snr_raw.size != expected_1b:
            raise RuntimeError(
                "isce3 ampcor snr output size mismatch: got {0}, expected {1}".format(
                    snr_raw.size, expected_1b
                )
            )
        if cov_raw.size != expected_3b:
            raise RuntimeError(
                "isce3 ampcor covariance output size mismatch: got {0}, expected {1}".format(
                    cov_raw.size, expected_3b
                )
            )

        offsets_bip = offset_raw.reshape(n_dn, n_ac, 2)
        cov_bip = cov_raw.reshape(n_dn, n_ac, 3)
        snr_img = snr_raw.reshape(n_dn, n_ac)

        # isce3 dense_offsets BIP band order: [azimuth, range].
        down_offset = offsets_bip[:, :, 0]
        across_offset = offsets_bip[:, :, 1]

        cov_azaz = cov_bip[:, :, 0]
        cov_rgrg = cov_bip[:, :, 1]
        cov_azrg = cov_bip[:, :, 2]

        across_grid = first_ac + np.arange(n_ac, dtype=np.int32) * skip_ac
        down_grid = first_dn + np.arange(n_dn, dtype=np.int32) * skip_dn
        location_across = np.tile(across_grid, n_dn)
        location_down = np.repeat(down_grid, n_ac)

        return {
            "engine": engine,
            "num_rows": int(n_dn * n_ac),
            "location_across": location_across,
            "location_down": location_down,
            "location_across_offset": across_offset.astype(np.float32, copy=False).reshape(-1),
            "location_down_offset": down_offset.astype(np.float32, copy=False).reshape(-1),
            "snr": snr_img.astype(np.float32, copy=False).reshape(-1),
            "cov1": cov_rgrg.astype(np.float32, copy=False).reshape(-1),
            "cov2": cov_azaz.astype(np.float32, copy=False).reshape(-1),
            "cov3": cov_azrg.astype(np.float32, copy=False).reshape(-1),
            "output_paths": outputs,
        }
    finally:
        if remove_scratch and scratch:
            shutil.rmtree(scratch, ignore_errors=True)
