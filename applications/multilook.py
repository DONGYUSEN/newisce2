#!/usr/bin/env python3

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("isce.multilook")


def multilook(
    data: np.ndarray,
    nalks: int,
    nrlks: int,
    boundary: str = 'crop'
) -> np.ndarray:
    """
    多视处理（通用复数数据）

    Args:
        data: 输入复数数组 (rows, cols)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪不能整除部分, 'pad' 填充到可整除

    Returns:
        多视后复数数组
    """
    rows, cols = data.shape
    out_rows = rows // nalks
    out_cols = cols // nrlks

    if boundary == 'crop':
        data = data[:out_rows * nalks, :out_cols * nrlks]
    elif boundary == 'pad':
        pad_rows = (nalks - rows % nalks) % nalks
        pad_cols = (nrlks - cols % nrlks) % nrlks
        if pad_rows > 0 or pad_cols > 0:
            data = np.pad(data, ((0, pad_rows), (0, pad_cols)), mode='edge')
        out_rows = data.shape[0] // nalks
        out_cols = data.shape[1] // nrlks

    amplitude = np.abs(data)
    amp = amplitude.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3)) / (nalks * nrlks)

    phase_complex = np.exp(1j * np.angle(data))
    phase_complex = phase_complex.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3))
    phase_ml = np.angle(phase_complex)

    result = amp * np.exp(1j * phase_ml)
    return result.astype(np.complex64)


def multilook_phase(
    phase_data: np.ndarray,
    nalks: int,
    nrlks: int,
    boundary: str = 'crop'
) -> np.ndarray:
    """
    对相位数据进行多视处理（矢量平均法）

    Args:
        phase_data: 相位数据数组，形状为 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪, 'pad' 填充

    Returns:
        多视处理后的相位数据数组
    """
    rows, cols = phase_data.shape
    out_rows = rows // nalks
    out_cols = cols // nrlks

    if boundary == 'crop':
        phase_data = phase_data[:out_rows * nalks, :out_cols * nrlks]
    elif boundary == 'pad':
        pad_rows = (nalks - rows % nalks) % nalks
        pad_cols = (nrlks - cols % nrlks) % nrlks
        if pad_rows > 0 or pad_cols > 0:
            phase_data = np.pad(phase_data, ((0, pad_rows), (0, pad_cols)), mode='edge')
        out_rows = phase_data.shape[0] // nalks
        out_cols = phase_data.shape[1] // nrlks

    complex_repr = np.exp(1j * phase_data)
    azimuth_sum = complex_repr.reshape(out_rows, nalks, out_cols * nrlks).sum(axis=1)
    total_sum = azimuth_sum.reshape(out_rows, out_cols, nrlks).sum(axis=2)
    mean_phase = np.angle(total_sum)

    return mean_phase


def multilook_preserve_phase(
    data: np.ndarray,
    nalks: int,
    nrlks: int,
    boundary: str = 'crop'
) -> np.ndarray:
    """
    相位保持多视处理 - 振幅和相位分别处理再合并

    Args:
        data: 复数数据数组，形状为 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪, 'pad' 填充

    Returns:
        多视处理后的复数数据数组
    """
    amp = np.abs(data)
    phase = np.angle(data)

    amp_ml = multilook(amp.astype(np.float32), nalks, nrlks, boundary=boundary)
    phase_ml = multilook_phase(phase, nalks, nrlks, boundary=boundary)

    result = amp_ml * np.exp(1j * phase_ml)
    return result.astype(np.complex64)


def multilook_with_stats(
    data: np.ndarray,
    nalks: int,
    nrlks: int,
    preserve_phase: bool = True,
    boundary: str = 'crop'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    多视处理并返回统计信息

    Returns:
        (result, stats): 多视结果和统计信息
    """
    amplitude_orig = np.abs(data)
    mean_amp_orig = np.mean(amplitude_orig)
    std_amp_orig = np.std(amplitude_orig)
    snr_orig = (mean_amp_orig / std_amp_orig) ** 2 if std_amp_orig > 0 else float('inf')

    original_stats = {
        'mean_amplitude': float(mean_amp_orig),
        'std_amplitude': float(std_amp_orig),
        'enl_estimate': float(snr_orig),
        'shape': data.shape
    }

    if preserve_phase:
        result = multilook_preserve_phase(data, nalks, nrlks, boundary=boundary)
    else:
        result = multilook(data, nalks, nrlks, boundary=boundary)

    amplitude_ml = np.abs(result)
    mean_amp_ml = np.mean(amplitude_ml)
    std_amp_ml = np.std(amplitude_ml)
    snr_ml = (mean_amp_ml / std_amp_ml) ** 2 if std_amp_ml > 0 else float('inf')

    multilook_stats = {
        'mean_amplitude': float(mean_amp_ml),
        'std_amplitude': float(std_amp_ml),
        'enl_estimate': float(snr_ml),
        'shape': result.shape
    }

    theoretical_enl_gain = nalks * nrlks
    actual_enl_gain = snr_ml / snr_orig if snr_orig > 0 and snr_orig != float('inf') else 1.0

    stats = {
        'original': original_stats,
        'multilook': multilook_stats,
        'improvement': {
            'enl_gain': float(actual_enl_gain),
            'enl_gain_db': float(10 * np.log10(actual_enl_gain)) if actual_enl_gain > 0 else 0.0,
            'theoretical_enl_gain': int(theoretical_enl_gain),
            'theoretical_enl_gain_db': float(10 * np.log10(theoretical_enl_gain)),
            'noise_reduction_percent': float((std_amp_orig - std_amp_ml) / std_amp_orig * 100) if std_amp_orig > 0 else 0.0,
            'efficiency': float(actual_enl_gain / theoretical_enl_gain * 100)
        },
        'parameters': {
            'nalks': nalks,
            'nrlks': nrlks,
            'total_looks': nalks * nrlks,
            'preserve_phase': preserve_phase
        }
    }

    return result, stats


def multilook_chunked(
    input_path: str,
    output_path: str,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000,
    preserve_phase: bool = True
) -> bool:
    """
    分块多视处理（文件到文件）

    Args:
        input_path: 输入TIFF文件路径
        output_path: 输出TIFF文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数（输出行数）
        preserve_phase: 是否使用相位保持多视处理

    Returns:
        处理是否成功
    """
    from osgeo import gdal
    gdal.UseExceptions()

    input_path = Path(input_path)
    output_path = Path(output_path)

    in_ds = gdal.Open(str(input_path))
    if in_ds is None:
        raise ValueError(f"无法打开文件: {input_path}")

    in_rows = in_ds.RasterYSize
    in_cols = in_ds.RasterXSize
    geotransform = in_ds.GetGeoTransform()
    projection = in_ds.GetProjection()

    out_rows = in_rows // nalks
    out_cols = in_cols // nrlks

    new_geotransform = list(geotransform)
    new_geotransform[1] *= nrlks
    new_geotransform[5] *= nalks

    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        str(output_path), out_cols, out_rows, 1, gdal.GDT_CFloat32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    out_ds.SetGeoTransform(tuple(new_geotransform))
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)

    for out_row in range(0, out_rows, chunk_lines):
        chunk_size = min(chunk_lines, out_rows - out_row)
        read_rows = chunk_size * nalks

        data = in_ds.GetRasterBand(1).ReadAsArray(0, out_row * nalks, in_cols, read_rows)
        if data is None:
            raise ValueError(f"读取分块失败 at row {out_row * nalks}")

        if preserve_phase:
            ml_data = multilook_preserve_phase(data.astype(np.complex64), nalks, nrlks)
        else:
            ml_data = multilook(data.astype(np.complex64), nalks, nrlks)
        out_band.WriteArray(ml_data, 0, out_row)

    in_ds = None
    out_ds = None
    return True


def update_image_xml(
    xml_path: str,
    nalks: int,
    nrlks: int,
    out_xml_path: Optional[str] = None
) -> bool:
    """
    更新ISCE XML文件中的图像参数

    Args:
        xml_path: 输入XML文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        out_xml_path: 输出XML路径，若为None则覆盖原文件

    Returns:
        是否成功
    """
    import xml.etree.ElementTree as ET

    if not os.path.exists(xml_path):
        logger.warning(f"XML文件不存在: {xml_path}")
        return False

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get_value(prop):
        val_node = prop.find('value')
        if val_node is not None and val_node.text:
            return val_node.text.strip()
        return None

    def set_value(prop, value):
        val_node = prop.find('value')
        if val_node is not None:
            val_node.text = str(value)

    changes = {}

    for prop in root.iter('property'):
        name_node = prop.find('name')
        if name_node is None or not name_node.text:
            continue

        name = name_node.text.strip()
        value = get_value(prop)

        if value is None:
            continue

        if name in ['width', 'Width']:
            try:
                new_val = int(int(value) / nrlks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['length', 'Length', 'numberOfRows', 'NumberOfRows']:
            try:
                new_val = int(int(value) / nalks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['azimuthSpacing', 'AzimuthSpacing', 'rowSpacing', 'rowSpacing']:
            try:
                new_val = float(float(value) * nalks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['rangeSpacing', 'RangeSpacing', 'columnSpacing', 'columnSpacing']:
            try:
                new_val = float(float(value) * nrlks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['prf', 'PRF', 'pulseRepetitionFrequency']:
            try:
                new_val = float(float(value) / nalks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['azimuthResolution', 'AzimuthResolution']:
            try:
                new_val = float(float(value) * nalks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['rangeResolution', 'RangeResolution', 'groundRangeResolution']:
            try:
                new_val = float(float(value) * nrlks)
                set_value(prop, new_val)
                changes[name] = f"{value} -> {new_val}"
            except:
                pass

        elif name in ['azimuthLooks', 'AzimuthLooks']:
            try:
                orig = int(value) * nalks
                set_value(prop, orig)
                changes[name] = f"{value} -> {orig}"
            except:
                pass

        elif name in ['rangeLooks', 'RangeLooks']:
            try:
                orig = int(value) * nrlks
                set_value(prop, orig)
                changes[name] = f"{value} -> {orig}"
            except:
                pass

        elif name in ['rangeTimeFirstPixel', 'rangeTimeLastPixel']:
            try:
                if name == 'rangeTimeFirstPixel':
                    continue
                first_pixel = 0
                for sibling in root.iter('property'):
                    n = sibling.find('name')
                    if n is not None and n.text == 'rangeTimeFirstPixel':
                        first_pixel = float(get_value(sibling) or 0)
                        break
                if first_pixel > 0:
                    col_spacing = 0
                    for sibling in root.iter('property'):
                        n = sibling.find('name')
                        if n is not None and n.text == 'rangeSpacing':
                            col_spacing = float(get_value(sibling) or 0)
                            break
                    if col_spacing > 0:
                        out_cols = int(int(value) / nrlks)
                        new_last = first_pixel + (out_cols - 1) * col_spacing * nrlks
                        set_value(prop, new_last)
                        changes[name] = f"{value} -> {new_last}"
            except:
                pass

    for name, change in changes.items():
        logger.debug(f"  {name}: {change}")

    if out_xml_path is None:
        out_xml_path = xml_path

    tree.write(out_xml_path, encoding='UTF-8', xml_declaration=True)
    return True


def multilook_isce_image(
    input_path: str,
    output_path: str,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000,
    preserve_phase: bool = True,
    update_xml: bool = True
) -> bool:
    """
    对ISCE格式图像进行多视处理

    Args:
        input_path: 输入图像路径（不含.xml后缀）
        output_path: 输出图像路径（不含.xml后缀）
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数
        preserve_phase: 是否使用相位保持多视
        update_xml: 是否更新XML参数

    Returns:
        是否成功
    """
    from osgeo import gdal
    gdal.UseExceptions()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        input_path_slc = Path(str(input_path) + '.slc')
        if input_path_slc.exists():
            input_path = input_path_slc
        else:
            logger.error(f"输入文件不存在: {input_path}")
            return False

    input_xml = input_path.with_suffix('.xml')
    if not input_xml.exists():
        input_xml = input_path.parent / (input_path.name + '.xml')

    logger.info(f"多视处理: {input_path.name}")
    logger.info(f"  视数: az={nalks}, rg={nrlks}")

    in_ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    if in_ds is None:
        logger.error(f"无法打开文件: {input_path}")
        return False

    in_rows = in_ds.RasterYSize
    in_cols = in_ds.RasterXSize
    geotransform = in_ds.GetGeoTransform()
    projection = in_ds.GetProjection()

    out_rows = in_rows // nalks
    out_cols = in_cols // nrlks

    logger.info(f"  输入: {in_rows}x{in_cols} -> 输出: {out_rows}x{out_cols}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        str(output_path), out_cols, out_rows, 1, gdal.GDT_CFloat32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )

    new_gt = list(geotransform)
    new_gt[1] *= nrlks
    new_gt[5] *= nalks
    out_ds.SetGeoTransform(tuple(new_gt))
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)

    for start_row in range(0, out_rows, chunk_lines):
        chunk_size = min(chunk_lines, out_rows - start_row)
        read_rows = chunk_size * nalks

        data = in_ds.GetRasterBand(1).ReadAsArray(0, start_row * nalks, in_cols, read_rows)
        if data is None:
            logger.warning(f"读取失败 at row {start_row * nalks}")
            continue

        data = data.astype(np.complex64)

        if preserve_phase:
            ml_data = multilook_preserve_phase(data, nalks, nrlks)
        else:
            ml_data = multilook(data, nalks, nrlks)

        out_band.WriteArray(ml_data, 0, start_row)

        if (start_row + chunk_size) % 1000 == 0 or start_row + chunk_size >= out_rows:
            logger.info(f"  进度: {min(start_row + chunk_size, out_rows)}/{out_rows} 行")

    in_ds = None
    out_ds = None

    if update_xml and input_xml.exists():
        output_xml = output_path.with_suffix('.xml')
        update_image_xml(str(input_xml), nalks, nrlks, str(output_xml))
        logger.info(f"  XML: {output_xml}")

    logger.info(f"  输出: {output_path}")
    return True


def multilook_from_xml(
    input_xml: str,
    output_path: str,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000,
    preserve_phase: bool = True
) -> bool:
    """
    从ISCE XML配置文件执行多视处理

    Args:
        input_xml: 输入XML文件路径
        output_path: 输出图像路径（不含.xml后缀）
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数
        preserve_phase: 是否使用相位保持多视

    Returns:
        是否成功
    """
    input_path = Path(input_xml).with_suffix('')
    return multilook_isce_image(
        str(input_path),
        output_path,
        nalks,
        nrlks,
        chunk_lines,
        preserve_phase,
        update_xml=True
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ISCE 多视处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  multilook.py -i input.slc -o output.slc -az 2 -rg 4
  multilook.py --input input.xml --output output --azimuth 1 --range 2
  multilook.py -i input.slc -o output.slc -az 2 -rg 4 --stats
        """
    )
    parser.add_argument('-i', '--input', required=True, help='输入图像或XML路径')
    parser.add_argument('-o', '--output', required=True, help='输出图像路径（不含.xml后缀）')
    parser.add_argument('-az', '--azimuth', type=int, required=True, help='方位向视数')
    parser.add_argument('-rg', '--range', type=int, required=True, help='距离向视数')
    parser.add_argument('--chunk', type=int, default=1000, help='分块行数 (默认: 1000)')
    parser.add_argument('--no-phase-preserve', action='store_true', help='不使用相位保持多视')
    parser.add_argument('--stats', action='store_true', help='输出统计信息')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    input_path = Path(args.input)
    if not input_path.exists() and not Path(str(input_path) + '.xml').exists():
        logger.error(f"输入文件不存在: {input_path}")
        return 1

    if str(input_path).endswith('.xml'):
        input_path = input_path.with_suffix('')

    success = multilook_isce_image(
        str(input_path),
        args.output,
        args.azimuth,
        args.range,
        args.chunk,
        preserve_phase=not args.no_phase_preserve,
        update_xml=True
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main() or 0)