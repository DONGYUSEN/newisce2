# StripInSAR 使用说明

## 概述

`strip_insar.py` 是 ISCE2 中简化的条带InSAR处理工具，基于 `stripmapApp.py` 改进而来：
- **简化日志输出**：统一 logging 格式，减少终端冗余信息
- **规范输出路径**：预设目录结构，便于数据管理
- **新增多视步骤**：可在配准前对SLC进行多视处理，节省存储空间

## 输入文件格式 (input.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<insarApp>
  <component name="insar">

    <!-- ========== 1. 必填参数 ========== -->

    <!-- 传感器类型 (LUTAN1, GF3, DJ1, ALOS, ALOS2, Sentinel1, etc.) -->
    <property name="sensor name">LUTAN1</property>

    <!-- 参考影像元数据 -->
    <component name="reference">
      <catalog>reference.xml</catalog>
    </component>

    <!-- 辅影像元数据 -->
    <component name="secondary">
      <catalog>secondary.xml</catalog>
    </component>

    <!-- DEM文件路径 -->
    <property name="demFilename">/path/to/dem.dem.wgs84</property>

    <!-- ========== 2. 多视参数 (新增) ========== -->

    <!-- 是否启用多视处理 -->
    <property name="do multilook">True</property>

    <!-- 多视方位向视数 -->
    <property name="multilook azimuth looks">2</property>

    <!-- 多视距离向视数 -->
    <property name="multilook range looks">4</property>

    <!-- 若未指定multilook参数，将使用numberAzimuthLooks/numberRangeLooks -->

    <!-- ========== 3. 处理范围 ========== -->

    <!-- 处理区域 (南, 北, 西, 东) 单位: 度 -->
    <!-- <property name="regionOfInterest">30.10,30.90,113.20,114.10</property> -->

    <!-- 地理编码范围 -->
    <!-- <property name="geocode bounding box">30.10,30.90,113.20,114.10</property> -->

    <!-- 地理编码输出分辨率 (度) -->
    <!-- <property name="geoPosting">0.00009</property> -->

    <!-- 干涉图目标像元间距 (米) -->
    <property name="posting">30</property>

    <!-- 多视因素 (用于干涉图生成) -->
    <property name="range looks">8</property>
    <property name="azimuth looks">8</property>

    <!-- ========== 4. 传感器/多普勒配置 ========== -->

    <!-- 多普勒计算方法: useDOPIQ, useDefault -->
    <property name="reference doppler method">useDEFAULT</property>
    <property name="secondary doppler method">useDEFAULT</property>

    <!-- 轨道插值方法: HERMITE (默认), SCH, LEGENDRE -->
    <property name="orbit interpolation method">HERMITE</property>

    <!-- 混合传感器时指定 -->
    <!-- <property name="reference sensor name">LUTAN1</property> -->
    <!-- <property name="secondary sensor name">LUTAN1</property> -->

    <!-- ========== 5. 配准控制 ========== -->

    <!-- 是否使用GPU -->
    <property name="use GPU">True</property>

    <!-- 外部配准辅助 -->
    <property name="use external coregistration">False</property>

    <!-- 几何闭合循环开关 -->
    <property name="enable rdrdem offset loop">True</property>

    <!-- 密集偏移控制 -->
    <property name="do denseoffsets">True</property>
    <property name="do rubbersheetingAzimuth">False</property>
    <property name="do rubbersheetingRange">False</property>
    <property name="rubber sheet SNR Threshold">5.0</property>
    <property name="rubber sheet filter size">9</property>

    <!-- 时序精化参数 -->
    <property name="refine timing azimuth-azimuth order">0</property>
    <property name="refine timing azimuth-range order">0</property>
    <property name="refine timing range-azimuth order">0</property>
    <property name="refine timing range-range order">0</property>
    <property name="refine timing SNR threshold">18.0</property>

    <!-- ========== 6. 干涉图/滤波/解缠 ========== -->

    <!-- Goldstein滤波强度 -->
    <property name="filter strength">0.5</property>

    <!-- 分频谱/电离层校正 -->
    <property name="do split spectrum">False</property>
    <property name="do dispersive">False</property>

    <!-- 解缠控制 -->
    <property name="do unwrap">True</property>
    <property name="unwrapper name">snaphu</property>
    <property name="snaphu gmtsar preprocess">True</property>
    <property name="snaphu coherence threshold">0.20</property>
    <property name="snaphu interpolate masked phase">False</property>
    <property name="snaphu interpolation radius">300</property>
    <property name="snaphu tile nrow">2</property>
    <property name="snaphu tile ncol">2</property>
    <property name="snaphu row overlap">400</property>
    <property name="snaphu col overlap">400</property>

    <!-- 二阶段解缠 -->
    <!-- <property name="do unwrap 2 stage">False</property> -->
    <!-- <property name="unwrapper 2stage name">REDARC0</property> -->
    <!-- <property name="SOLVER_2STAGE">pulp</property> -->

  </component>
</insarApp>
```

## 参数说明

### 1. 核心参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `sensor name` | str | 必填 | 传感器类型 |
| `demFilename` | str | 必填 | DEM文件路径 |
| `reference` | component | 必填 | 参考影像catalog |
| `secondary` | component | 必填 | 辅影像catalog |

### 2. 多视参数 (strip_insar特有)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `do multilook` | bool | False | 是否启用多视处理 |
| `multilook azimuth looks` | int | None | 多视方位向视数 |
| `multilook range looks` | int | None | 多视距离向视数 |

**多视逻辑说明**：
- `do_multilook=False` 或 `az=rg=1`：创建符号链接到原始SLC，节省空间
- `do_multilook=True` 且 `az>1 or rg>1`：执行多视处理，输出到 `02_ml_slc/`

### 3. 处理范围参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `regionOfInterest` | list[float] | None | 处理区域 [S,N,W,E] 度 |
| `geocode bounding box` | list[float] | None | 地理编码范围 [S,N,W,E] |
| `geoPosting` | float | None | 地理编码分辨率 (度) |
| `posting` | int | 30 | 干涉图目标像元间距 (米) |
| `range looks` | int | None | 距离向多视数 |
| `azimuth looks` | int | None | 方位向多视数 |

### 4. 传感器/多普勒参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `reference doppler method` | str | None | useDOPIQ / useDefault |
| `secondary doppler method` | str | None | useDOPIQ / useDefault |
| `orbit interpolation method` | str | HERMITE | HERMITE / SCH / LEGENDRE |
| `reference sensor name` | str | None | 参考传感器类型 |
| `secondary sensor name` | str | None | 辅传感器类型 |
| `rangeCropFarPixels` | int | None | 远距裁剪像素数 |

### 5. 配准参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use GPU` | bool | True | 优先使用GPU |
| `use external coregistration` | bool | False | 外部配准辅助 |
| `enable rdrdem offset loop` | bool | True | 几何闭合循环 |
| `do denseoffsets` | bool | False | 密集偏移计算 |
| `do rubbersheetingAzimuth` | bool | False | 方位向橡皮贴 |
| `do rubbersheetingRange` | bool | False | 距离向橡皮贴 |
| `rubber sheet SNR Threshold` | float | 5.0 | 橡皮贴SNR阈值 |
| `rubber sheet filter size` | int | 9 | 橡皮贴滤波大小 |
| `refine timing SNR threshold` | float | 1.2 | 时序精化SNR阈值 |
| `refine timing azimuth-azimuth order` | int | 0 | 方位-方位多项式阶数 |
| `refine timing azimuth-range order` | int | 0 | 方位-距离多项式阶数 |
| `refine timing range-azimuth order` | int | 0 | 距离-方位多项式阶数 |
| `refine timing range-range order` | int | 0 | 距离-距离多项式阶数 |

### 6. 干涉图/滤波/解缠参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `filter strength` | float | 0.5 | Goldstein滤波强度 (0-1) |
| `do split spectrum` | bool | False | 分频谱处理 |
| `do dispersive` | bool | False | 电离层校正 |
| `do unwrap` | bool | True | 是否解缠 |
| `unwrapper name` | str | grass | snaphu / icu / grass |
| `snaphu coherence threshold` | float | 0.20 | snaphu相干性阈值 |
| `snaphu gmtsar preprocess` | bool | True | GMTSAR风格预处理 |
| `snaphu tile nrow` | int | 2 | snaphu Tile行数 |
| `snaphu tile ncol` | int | 2 | snaphu Tile列数 |
| `snaphu row overlap` | int | 400 | Tile行重叠 (像素) |
| `snaphu col overlap` | int | 400 | Tile列重叠 (像素) |

### 7. 其他参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `correlation_method` | str | cchz_wave | 相干性估计方法 |
| `useHighResolutionDemOnly` | bool | False | 仅使用最高分辨率DEM |
| `geocode list` | list[str] | None | 地理编码产品列表 |
| `offset geocode list` | list[str] | None | 偏移地理编码列表 |
| `pickle dump directory` | str | PICKLE | pickle缓存目录 |
| `renderer` | str | xml | 序列化格式 (xml/pickle) |

## 输出目录结构

```
workdir/
├── 01_raw_data/              # 原始RAW数据
├── 02_slc/                   # 全分辨率SLC
├── 02_ml_slc/                # 多视后SLC (启用多视时)
│   ├── reference.ml.slc      # 多视参考SLC
│   ├── reference.ml.slc.xml
│   ├── secondary.ml.slc       # 多视辅SLC
│   └── secondary.ml.slc.xml
├── 03_coregistered_slc/       # 配准后SLC
├── 04_geometry/              # 几何产品 (lat, lon, los, hgt)
├── 05_interferogram/         # 干涉图和相干性
│   ├── topophase.flat        # 干涉图
│   ├── filt_topophase.flat   # 滤波后干涉图
│   ├── phsig.cor             # 相干性
│   └── topophase.cor         # 相关系数
├── 06_unwrapped/             # 解缠结果
│   └── filt_topophase.unw    # 解缠相位
├── 07_geocoded/              # 地理编码产品
├── 08_dense_offsets/         # 密集偏移
├── 09_ionosphere/            # 电离层校正 (启用时)
├── PICKLE/                   # 处理状态缓存
└── strip_insar.log           # 日志文件
```

## 命令行用法

```bash
# 标准运行
strip_insar.py input.xml

# 分步骤运行 (便于调试)
strip_insar.py --steps input.xml

# 指定输出目录
strip_insar.py -o /path/to/output input.xml

# 详细输出模式
strip_insar.py -v input.xml

# 查看可用步骤
strip_insar.py --help-steps
```

## 处理流程

```
01: startup          - 初始化
02: preprocess       - 预处理原始数据
03: cropraw          - 裁剪原始数据
04: formslc          - SLC成像
05: cropslc          - 裁剪SLC
06: multilook        - 多视处理 [新增]
07: verifyDEM        - 验证/下载DEM
08: topo             - 地形计算
09: normalize_secondary_sampling - 归一化辅影像采样
10: geo2rdr          - 地理到雷达坐标转换
11: rdrdem_offset    - RADAR-DEM偏移计算
12: rect_rgoffset     - 距离偏移校正
13: coarse_resample  - 粗配准重采样
14: misregistration  - 精化配准
15: refined_resample - 精化重采样
16: dense_offsets     - 密集偏移计算
17: rubber_sheet_range - 距离向橡皮贴
18: rubber_sheet_azimuth - 方位向橡皮贴
19: fine_resample     - 最终重采样
20: split_range_spectrum - 分频谱
21: sub_band_resample - 子带重采样
22: interferogram     - 干涉图生成
23: filter            - 滤波
24: unwrap            - 解缠
25: geocode           - 地理编码
26: geocodeoffsets    - 偏移地理编码
27: endup             - 完成
```

## 多视功能详解

### 多视参数优先级

1. 若设置 `multilookAz` / `multilookRg`，使用该值
2. 若未设置，使用 `numberAzimuthLooks` / `numberRangeLooks`
3. 若均为 None，使用 1 (不进行多视)

### 多视对后续处理的影响

- 多视后SLC将替换 `referenceSlcCropProduct` 和 `secondarySlcCropProduct`
- 后续配准、解缠等步骤均使用多视后的数据
- XML参数（width, length, spacing, prf等）会自动更新

### 存储节省策略

| 条件 | 行为 | 存储 |
|------|------|------|
| `do_multilook=False` | 符号链接到02_slc | 几乎为零 |
| `az=1, rg=1` | 符号链接到02_slc | 几乎为零 |
| `az>1 or rg>1` | 实际多视处理 | 原始/(az*rg) |

## 完整配置示例

### 示例1: 启用多视

```xml
<insarApp>
  <component name="insar">
    <property name="sensor name">GF3</property>
    <component name="reference">
      <catalog>reference.xml</catalog>
    </component>
    <component name="secondary">
      <catalog>secondary.xml</catalog>
    </component>
    <property name="demFilename">/data/dem.wgs84</property>

    <!-- 多视配置 -->
    <property name="do multilook">True</property>
    <property name="multilook azimuth looks">2</property>
    <property name="multilook range looks">4</property>

    <!-- 处理参数 -->
    <property name="posting">30</property>
    <property name="range looks">8</property>
    <property name="azimuth looks">8</property>
    <property name="filter strength">0.5</property>
    <property name="do unwrap">True</property>
    <property name="unwrapper name">snaphu</property>
    <property name="snaphu coherence threshold">0.20</property>
    <property name="use GPU">True</property>
    <property name="do denseoffsets">True</property>
  </component>
</insarApp>
```

### 示例2: 不启用多视（符号链接模式）

```xml
<insarApp>
  <component name="insar">
    <property name="sensor name">LUTAN1</property>
    <component name="reference">
      <catalog>reference.xml</catalog>
    </component>
    <component name="secondary">
      <catalog>secondary.xml</catalog>
    </component>
    <property name="demFilename">/data/dem.wgs84</property>

    <!-- 多视关闭，使用符号链接节省空间 -->
    <property name="do multilook">False</property>

    <property name="posting">30</property>
    <property name="range looks">8</property>
    <property name="azimuth looks">8</property>
    <property name="filter strength">0.5</property>
    <property name="do unwrap">True</property>
    <property name="unwrapper name">grass</property>
    <property name="use GPU">True</property>
  </component>
</insarApp>
```

## 常见问题

### Q1: 多视参数如何选择？
A: 多视视数应根据目标分辨率和存储条件选择：
- 低分辨率（~30m）: az=2, rg=4
- 中分辨率（~10m）: az=1, rg=2
- 高分辨率（~3m）: az=1, rg=1 或不启用多视

### Q2: 多视后为什么还要设置range looks/azimuth looks？
A: `multilookAz/Rg` 用于多视SLC（影响配准精度），`range looks/azimuth looks` 用于干涉图多视（影响分辨率和噪声）。两者可以不同。

### Q3: 如何查看处理进度？
A: 使用 `--steps` 模式运行，会逐步输出每个步骤的状态，便于定位问题。

### Q4: 如何跳过某些步骤？
A: 目前strip_insar不支持跳过特定步骤，如需调整流程建议修改源码或使用 `--steps` 手动控制。