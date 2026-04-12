# NISAR数据导入ISCE2可行性分析与实施路线图

> 日期: 2026-04-12
> 状态: 分析完成，待实施
> 前置工作: ISCE3 backprojection C++库已完整集成到ISCE2（stripmapApp + topsApp）

---

## 一、背景

NISAR（NASA-ISRO SAR Mission）是NASA和ISRO联合开发的L-band + S-band双频SAR卫星。
ISCE3已有完整的NISAR支持（从RSLC到GUNW的全流程），但ISCE2目前不支持NISAR数据。

本分析评估在ISCE2中添加NISAR数据导入功能的可行性和实施方案。

---

## 二、NISAR数据格式概述

### 2.1 NISAR RSLC产品特征

NISAR的Level-1 RSLC（Range-Doppler SLC）产品是**HDF5格式**的聚焦SLC数据，已在零多普勒几何下。

| 特性 | NISAR RSLC | Sentinel-1 (TOPS) | ALOS-2 |
|------|-----------|-------------------|--------|
| **数据格式** | HDF5 (.h5) | TIFF + XML | CEOS二进制 |
| **频段** | 双频: L-band (LSAR) + S-band (SSAR) | C-band | L-band |
| **观测模式** | Stripmap + ScanSAR (非TOPS) | TOPS (burst模式) | Stripmap + ScanSAR |
| **极化** | 多极化 (HH/HV/VH/VV, 按频率组织) | 通常VV或VV+VH | HH/HV/VH/VV |
| **几何** | 零多普勒 (已聚焦SLC) | 零多普勒 (已聚焦SLC) | 零多普勒 (SLC) |
| **轨道** | HDF5内嵌 (`metadata/orbit/`) | 外部EOF轨道文件 | CEOS leader文件 |
| **多普勒** | 2D LUT (`processingInformation/parameters/`) | XML中多项式 | leader文件系数 |
| **数据类型** | complex32 (float16对) 或 complex64 | complex (int16对) | complex (多种) |

### 2.2 NISAR RSLC HDF5结构

从ISCE3代码（`nisar/products/readers/SLC/RSLC.py`, `Base/Base.py`, `Base/Identification.py`）推断的完整结构：

```
/science/
  LSAR/                          ← L-band频段（SSAR为S-band）
    RSLC/
      identification/            ← 产品标识
        missionId                ← "NISAR"
        productType              ← "RSLC" (或旧版 "SLC")
        absoluteOrbitNumber
        lookDirection            ← "right" / "left"
        orbitPassDirection       ← "ascending" / "descending"
        zeroDopplerStartTime     ← ISO 8601 时间戳
        zeroDopplerEndTime
        listOfFrequencies        ← ["A"] 或 ["A", "B"]
        diagnosticModeFlag       ← 0=Science/DBF, 1=DM1, 2=DM2
        plannedDatatakeId
        plannedObservationId
        isUrgentObservation
        isJointObservation
      swaths/
        zeroDopplerTime          ← 方位时间向量（共享）
        frequencyA/
          slantRange             ← 距离向向量
          listOfPolarizations    ← ["HH", "HV"] 等
          HH                     ← complex SLC数据
          HV
          processedCenterFrequency
          processedRangeBandwidth
          acquiredCenterFrequency
          acquiredRangeBandwidth
          nominalAcquisitionPRF
          sceneCenterAlongTrackSpacing
          sceneCenterGroundRangeSpacing
        frequencyB/              ← 可选第二子频段
          ...
      metadata/
        orbit/                   ← 状态向量（时间, 位置XYZ, 速度XYZ）
        attitude/                ← 姿态四元数
        calibrationInformation/
          geometry/
            zeroDopplerTime
            slantRange
          frequencyA/
            noiseEquivalentBackscatter/
      processingInformation/
        parameters/
          frequencyA/
            dopplerCentroid      ← 2D LUT（方位时间 × 斜距）
            zeroDopplerTime
            slantRange
```

### 2.3 根路径发现逻辑

ISCE3的 `get_hdf5_file_root_path()` 按以下顺序搜索：
```python
SCIENCE_PATH = '/science/'
NISAR_SENSOR_LIST = ['SSAR', 'LSAR']  # S-band优先? 实际上按字母序
# 实际逻辑: 遍历 ['SSAR', 'LSAR'], 找到第一个存在的
# 对于L-band产品: root = '/science/LSAR'
# 对于S-band产品: root = '/science/SSAR'
```

产品路径 = `{root}/RSLC/` (或 `{root}/SLC/` 为旧格式，自动映射为RSLC)

---

## 三、ISCE2 Sensor框架分析

### 3.1 Sensor基类接口

文件: `components/isceobj/Sensor/Sensor.py` (128行)

```python
class Sensor(Component):
    def parse(self): ...                    # 解析元数据入口
    def _populatePlatform(self, **kwargs): ... # 卫星名、任务、pointing direction
    def _populateInstrument(self, **kwargs): ... # 波长、PRF、chirp参数、采样率
    def _populateFrame(self, **kwargs): ...  # sensingStart/Stop, startingRange, numberOfLines/Samples
    def _populateOrbit(self, **kwargs): ...  # 轨道状态向量列表
    def extractImage(self): ...              # 提取数据到二进制文件
    def extractDoppler(self): ...            # 多普勒质心多项式
```

### 3.2 Sensor注册机制

文件: `components/isceobj/Sensor/__init__.py`

```python
# SENSORS字典映射大写传感器名到工厂函数
createNISAR_RSLC = partial(factory_template, 'NISAR_RSLC')
SENSORS['NISAR_RSLC'] = createNISAR_RSLC
```

### 3.3 参考实现

- **ALOS2.py** (546行): CEOS二进制格式的L-band SLC。继承Sensor基类，展示了完整的parse→populate→extract流程。
- **TOPS/Sentinel1.py** (1406行): TOPS burst模式。继承Component（非Sensor），使用独立的BurstSLC对象，**不适合NISAR参考**。

**关键参考: ALOS2.py 的 `extractImage()` 实现模式:**
```python
def extractImage(self):
    self.parse()  # 确保已解析
    out = open(self.output, 'wb')
    self.imageFile.extractImage(output=out)
    out.close()
    # 设置 sensingStart/Stop/Mid, startingRange, farRange
    # 创建 SlcImage 对象
    rawImage = isceobj.createSlcImage()
    rawImage.setFilename(self.output)
    rawImage.setWidth(self.imageFile.width)
    rawImage.renderHdr()
    self.frame.setImage(rawImage)
```

---

## 四、NISAR_RSLC Sensor 设计方案

### 4.1 类结构

```python
class NISAR_RSLC(Sensor):
    """NISAR Level-1 RSLC (HDF5) reader for ISCE2."""
    
    family = 'nisar_rslc'
    
    # 输入参数
    parameter_list = (
        HDF5FILE,       # str: RSLC HDF5文件路径
        FREQUENCY,      # str: "A" 或 "B"（默认"A"）
        POLARIZATION,   # str: "HH", "HV", etc.（默认"HH"）
    ) + Sensor.parameter_list
```

### 4.2 方法映射表

| ISCE2方法 | NISAR HDF5数据源 | 实现要点 | 难度 |
|-----------|-----------------|---------|------|
| `parse()` | 打开HDF5, 验证productType, 发现频段和极化 | 参考ISCE3的`GenericProduct.parsePolarizations()` | ⭐⭐ |
| `_populatePlatform()` | `identification/missionId` → "NISAR"<br>`identification/lookDirection` → pointingDirection (right=-1, left=1) | 直接映射 | ⭐ |
| `_populateInstrument()` | `processedCenterFrequency` → 波长 (λ=c/f)<br>`nominalAcquisitionPRF` → PRF<br>`processedRangeBandwidth` → chirp带宽<br>`slantRange` → 采样率 (推算) | 需要从HDF5字段计算ISCE2期望的参数 | ⭐⭐ |
| `_populateFrame()` | `zeroDopplerStartTime` → sensingStart<br>`zeroDopplerEndTime` → sensingStop<br>`slantRange[0]` → startingRange<br>SLC数据shape → numberOfLines/Samples | 时间字符串解析 + 距离向参数推算 | ⭐⭐ |
| `_populateOrbit()` | `metadata/orbit/` → time[], positionX/Y/Z[], velocityX/Y/Z[] | 遍历HDF5 orbit组，创建StateVector → Orbit | ⭐⭐ |
| `extractImage()` | 从HDF5读取complex SLC → 写入二进制文件 | complex32→complex64转换 + 分块I/O | ⭐⭐⭐ |
| `extractDoppler()` | `dopplerCentroid` 2D LUT → 1D多项式 | 沿方位向取中值/平均，拟合距离向多项式 | ⭐⭐⭐ |

### 4.3 关键技术挑战

#### Challenge 1: complex32数据类型转换

NISAR RSLC可能使用complex32（两个float16），ISCE2期望complex64。

```python
def _read_complex_slc(self, dataset, out_file, chunk_lines=512):
    """分块读取NISAR SLC，转换为complex64并写入二进制文件。"""
    nlines, nsamples = dataset.shape
    
    with open(out_file, 'wb') as fout:
        for i0 in range(0, nlines, chunk_lines):
            i1 = min(i0 + chunk_lines, nlines)
            chunk = dataset[i0:i1, :]
            
            # NISAR complex32: structured dtype with 'r' and 'i' float16 fields
            if chunk.dtype.names and 'r' in chunk.dtype.names:
                real = chunk['r'].astype(np.float32)
                imag = chunk['i'].astype(np.float32)
                slc_chunk = real + 1j * imag
            else:
                slc_chunk = chunk.astype(np.complex64)
            
            slc_chunk.tofile(fout)
```

#### Challenge 2: 2D Doppler LUT → 1D多项式

ISCE2的extractDoppler期望返回距离向多项式系数。NISAR提供2D LUT (azimuth × range)。

```python
def extractDoppler(self):
    """将NISAR的2D Doppler LUT降维为1D距离向多项式。"""
    # 读取2D LUT
    doppler_2d = h5f[f'{proc_path}/dopplerCentroid'][:]
    slant_range = h5f[f'{proc_path}/slantRange'][:]
    
    # 沿方位向取中值（中间时刻的Doppler）
    doppler_1d = np.median(doppler_2d, axis=0)
    
    # 拟合距离向多项式（2阶）
    range_norm = slant_range - slant_range[len(slant_range)//2]
    coeffs = np.polyfit(range_norm, doppler_1d, 2)
    
    # 返回ISCE2期望的格式
    quadratic = {'a': coeffs[2] / prf, 'b': coeffs[1] / prf, 'c': coeffs[0] / prf}
    
    # 同时设置 _dopplerVsPixel (更精确，供roiApp使用)
    self.frame._dopplerVsPixel = [coeffs[2], coeffs[1], coeffs[0]]
    
    return quadratic
```

#### Challenge 3: 内存管理

NISAR全分辨率SLC可能非常大（L-band 80MHz带宽、20000+距离样本、数万行）。必须分块读取（如512行/块），不能一次性加载整个数据集到内存。

---

## 五、处理流程评估

### 5.1 NISAR Stripmap → stripmapApp ✅

NISAR的主要科学观测使用Stripmap模式。RSLC产品是**连续的、非burst的SLC数据**，与stripmapApp的假设完全匹配：

- 连续方位时间采样 ✅
- 零多普勒几何 ✅
- 单一PRF ✅
- 标准距离向采样 ✅

**结论**: 编写NISAR_RSLC sensor后，可直接用stripmapApp处理NISAR stripmap数据。

### 5.2 NISAR ScanSAR → 需要额外评估

NISAR的ScanSAR模式与Sentinel-1 TOPS有本质区别：
- **Sentinel-1 TOPS**: burst结构 + 方位频谱调制 → 需要topsApp
- **NISAR ScanSAR**: RSLC已经是完整聚焦SLC，无burst结构 → 不需要topsApp

对于NISAR ScanSAR RSLC：
- 每个子swath是独立的连续SLC
- 可以视为多个独立stripmap处理
- 可能需要后续的子swath拼接

### 5.3 是否需要 `nisarApp`？

**短期: 不需要。**

| 方案 | 工作量 | 功能覆盖 | 推荐度 |
|------|--------|---------|-------|
| **A: NISAR_RSLC sensor + stripmapApp** | 1-1.5周 | 基本InSAR（单频单极化） | ⭐⭐⭐ 推荐起步 |
| **B: 方案A + 双频脚本** | 2-3周 | 双频独立处理 + 简单电离层校正 | ⭐⭐ 第二阶段 |
| **C: 完整nisarApp** | 4-8周+ | 全部NISAR特性 | ⭐ 投入产出比低，建议直接用ISCE3 |

### 5.4 双频处理策略

**起步方案（独立处理）：**
```bash
# L-band InSAR
stripmapApp.py --steps --sensor.name=NISAR_RSLC \
  --sensor.hdf5file=nisar_reference.h5 --sensor.frequency=A --sensor.polarization=HH

# S-band InSAR
stripmapApp.py --steps --sensor.name=NISAR_RSLC \
  --sensor.hdf5file=nisar_reference.h5 --sensor.frequency=B --sensor.polarization=HH
```

高级双频电离层校正建议直接使用ISCE3的 `nisar/workflows/ionosphere.py`。

---

## 六、与ISCE3 NISAR模块的复用关系

### 6.1 可复用组件

| ISCE3组件 | 路径 | 可复用性 | ISCE2等价物 |
|-----------|------|---------|-----------|
| HDF5路径逻辑 | `Base/Base.py` | ✅ 高 | 直接参考路径字符串 |
| 根路径发现 | `Base/Base.get_hdf5_file_root_path()` | ✅ 高 | 重写为纯h5py版本 |
| Identification解析 | `Base/Identification.py` | ✅ 高 | 参考字段映射 |
| 极化发现 | `GenericProduct.parsePolarizations()` | ✅ 高 | 参考遍历逻辑 |
| 产品类型验证 | `GenericProduct.get_hdf5_file_product_type()` | ✅ 高 | SLC→RSLC兼容逻辑 |
| ComplexFloat16解码 | RSLC reader | ✅ 高 | numpy重写 |

### 6.2 不可复用组件

| ISCE3组件 | 原因 |
|-----------|------|
| pyre框架 | ISCE2不使用pyre |
| journal日志 | ISCE2使用Python logging |
| isce3.core对象 (Orbit, LUT2d, RadarGridParameters) | ISCE2有自己的对象体系 |
| isce3.product.Swath | ISCE2使用Frame/Instrument |
| nisar/workflows/insar.py | 完全不同的工作流框架 |

---

## 七、ISCE3 InSAR工作流参考

从 `nisar/workflows/insar.py` 提取的NISAR InSAR处理步骤：

```
1. bandpass_insar    → 带通滤波
2. rdr2geo           → 雷达→地理坐标（topo）
3. geo2rdr           → 地理→雷达坐标（几何配准）
4. prepare_insar_hdf5 → 准备输出HDF5
5. coarse_resample   → 粗配准重采样
6. dense_offsets     → 密集偏移量估计（可选）
7. offsets_product   → 偏移量产品（可选）
8. rubbersheet       → 非刚性配准（可选）
9. fine_resample     → 精配准重采样（可选）
10. crossmul         → 干涉图生成
11. filter_interferogram → 干涉图滤波
12. unwrap           → 相位解缠
13. ionosphere       → 电离层校正（需split_spectrum）
14. geocode          → 地理编码（RIFG, RUNW, ROFF → GUNW, GOFF）
15. troposphere      → 对流层校正
16. solid_earth_tides → 固体潮校正
17. baseline         → 基线计算
```

对应ISCE2 stripmapApp的步骤映射：

| ISCE3步骤 | ISCE2 stripmapApp等价步骤 | 兼容性 |
|-----------|--------------------------|-------|
| rdr2geo | topo | ✅ |
| geo2rdr | geo2rdr | ✅ |
| coarse_resample | resampleSlc | ✅ |
| crossmul | interferogram | ✅ |
| filter_interferogram | filter | ✅ |
| unwrap | unwrap | ✅ |
| geocode | geocode | ✅ |
| dense_offsets | denseOffsets | ✅ |
| ionosphere | ❌ 不支持 | 需要ISCE3 |
| troposphere | ❌ 不支持 | 需要外部工具 |
| solid_earth_tides | ❌ 不支持 | 需要外部工具 |

---

## 八、实施路线图

### Phase 1: NISAR_RSLC Sensor 插件 (1-1.5周)

**目标**: 最小可行的NISAR数据导入，可通过stripmapApp运行InSAR。

**文件清单:**
```
components/isceobj/Sensor/
  NISAR_RSLC.py          (~350-400行) 新增
  __init__.py             修改（添加SENSORS注册）

tests/                    (或同目录)
  test_nisar_rslc.py     (~200行) 新增
```

**任务分解:**

| # | 任务 | 工时 | 依赖 |
|---|------|------|------|
| 1.1 | 创建NISAR_RSLC类骨架 + Component参数定义 | 2h | - |
| 1.2 | 实现`parse()`: HDF5打开、productType验证、频段/极化发现 | 2h | 1.1 |
| 1.3 | 实现`_populatePlatform()`: missionId, lookDirection | 1h | 1.2 |
| 1.4 | 实现`_populateInstrument()`: 波长、PRF、带宽、采样率 | 2h | 1.2 |
| 1.5 | 实现`_populateFrame()`: sensing时间、距离参数、图像尺寸 | 2h | 1.2 |
| 1.6 | 实现`_populateOrbit()`: HDF5轨道组→StateVector→Orbit | 3h | 1.2 |
| 1.7 | 实现`extractImage()`: 分块读取+complex32转换+二进制写入 | 4h | 1.2 |
| 1.8 | 实现`extractDoppler()`: 2D LUT→1D多项式 | 3h | 1.2 |
| 1.9 | 注册到`__init__.py`的SENSORS字典 | 0.5h | 1.1 |
| 1.10 | 单元测试（模拟NISAR HDF5） | 3h | 1.7, 1.8 |
| 1.11 | stripmapApp集成测试 | 3h | 1.10 |

**预计总工时: 25-30小时**

### Phase 2: 端到端验证 (0.5周)

| # | 任务 | 说明 |
|---|------|------|
| 2.1 | 获取NISAR样例RSLC数据 | JPL公开数据或模拟数据 |
| 2.2 | 运行stripmapApp处理NISAR对 | 验证完整InSAR流程 |
| 2.3 | 与ISCE3处理结果对比 | 确认几何精度和相位一致性 |
| 2.4 | 修复边界情况 | complex32、多频、不同极化等 |

### Phase 3: 高级功能 (可选, 2-4周)

| # | 功能 | 复杂度 | 推荐 |
|---|------|--------|------|
| 3.1 | ScanSAR子swath处理 | 中 | 如有需求 |
| 3.2 | 双频联合处理脚本 | 中 | 如有需求 |
| 3.3 | 辐射定标(NESZ, calibration LUT) | 高 | 建议用ISCE3 |
| 3.4 | 电离层双频校正 | 高 | 建议用ISCE3 |
| 3.5 | 与backprojection集成（NISAR数据用backproject重聚焦） | 中 | 有研究价值 |

---

## 九、结论

| 问题 | 答案 |
|------|------|
| **能否将NISAR数据导入ISCE2？** | ✅ 完全可行。核心是编写NISAR_RSLC sensor类。 |
| **是否需要新的topsApp风格脚本？** | ❌ 不需要。NISAR RSLC是连续SLC，stripmapApp即可。 |
| **主要技术难点** | ① complex32→complex64转换 ② 2D Doppler→1D多项式 ③ 大文件分块I/O |
| **工作量** | Phase 1: ~25-30h (1-1.5周) |
| **推荐路径** | Phase 1先行 → Phase 2验证 → Phase 3按需扩展 |
| **高级NISAR特性** | 电离层/对流层校正等建议直接使用ISCE3 |

---

## 附录A: ISCE2已完成的改进工作

### A.1 RCMC地形校正 (Phase 1)
- 在mroipac formslc中加入地形相关RCMC
- 修改10个文件

### A.2 ISCE3 Backprojection集成 (Phase 2-3)
- 22个C++源文件迁移到 `contrib/isce3_backproject/`
- pybind11绑定 + OpenMP优化
- stripmapApp: FOCUSER参数 + `runBackproject.py`
- topsApp: REFOCUS_BURSTS参数 + `runRefocusBursts.py`
- 15/15测试通过

### A.3 关键文件索引

```
# Backprojection模块
contrib/isce3_backproject/
  __init__.py
  build.py                    # 独立构建脚本
  SConscript                  # SCons集成
  backproject.cpython-312-x86_64-linux-gnu.so
  bindings/pyBackproject.cpp  # pybind11绑定
  adapters/
    __init__.py
    dem_adapter.py            # DEM适配器
    isce2_adapter.py          # ISCE2↔ISCE3对象转换
    range_compress.py         # FFT距离压缩
    tops_adapter.py           # TOPS burst适配器
  include/                    # 22个C++头文件
  vendor/Eigen/               # Eigen 3.4.0

# 修改的应用文件
applications/stripmapApp.py   # FOCUSER参数
applications/topsApp.py       # REFOCUS_BURSTS参数
components/isceobj/StripmapProc/
  Factories.py                # createFormSLCBackproject
  runBackproject.py           # backprojection聚焦步骤
components/isceobj/TopsProc/
  Factories.py                # createRefocusBursts
  runRefocusBursts.py         # burst重聚焦步骤
```

## 附录B: NISAR_RSLC Sensor 代码骨架

```python
#!/usr/bin/env python3
"""
NISAR Level-1 RSLC (HDF5) reader for ISCE2.

Reads NISAR RSLC products and populates ISCE2 Frame/Instrument/Orbit objects
for use with stripmapApp InSAR processing.

References:
  - ISCE3: nisar/products/readers/SLC/RSLC.py
  - ISCE3: nisar/products/readers/Base/Base.py
  - ISCE3: nisar/products/readers/Base/Identification.py
"""

import os
import datetime
import h5py
import numpy as np
import logging

from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from .Sensor import Sensor

# HDF5 root paths for NISAR frequency bands
NISAR_SENSOR_LIST = ['LSAR', 'SSAR']
SCIENCE_PATH = '/science/'

HDF5FILE = Component.Parameter(
    'hdf5file',
    public_name='hdf5 file',
    default=None,
    type=str,
    mandatory=True,
    doc='Path to NISAR RSLC HDF5 file'
)

FREQUENCY = Component.Parameter(
    'frequency',
    public_name='frequency',
    default='A',
    type=str,
    mandatory=False,
    doc='Frequency band: A or B'
)

POLARIZATION = Component.Parameter(
    'polarization',
    public_name='polarization',
    default='HH',
    type=str,
    mandatory=False,
    doc='Polarization channel: HH, HV, VH, VV'
)


class NISAR_RSLC(Sensor):
    """NISAR Level-1 RSLC (HDF5) reader for ISCE2."""
    
    family = 'nisar_rslc'
    logging_name = 'isce.sensor.NISAR_RSLC'
    
    parameter_list = (HDF5FILE, FREQUENCY, POLARIZATION) + Sensor.parameter_list
    
    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._root_path = None      # e.g., '/science/LSAR'
        self._product_path = None   # e.g., '/science/LSAR/RSLC'
        self._swath_path = None     # e.g., '/science/LSAR/RSLC/swaths'
        self._metadata_path = None  # e.g., '/science/LSAR/RSLC/metadata'
        self._proc_path = None      # e.g., '/science/LSAR/RSLC/processingInformation'
    
    def getFrame(self):
        return self.frame
    
    def parse(self):
        """Open HDF5 and discover root path, product type, frequencies, polarizations."""
        # TODO: 实现
        pass
    
    def _populatePlatform(self):
        """Set platform info from identification group."""
        # TODO: missionId, lookDirection → pointingDirection
        pass
    
    def _populateInstrument(self):
        """Set instrument params from swath metadata."""
        # TODO: centerFrequency → wavelength, PRF, bandwidth, sampling rate
        pass
    
    def _populateFrame(self):
        """Set frame timing and geometry from swath data."""
        # TODO: zeroDopplerStart/EndTime, slantRange, image dimensions
        pass
    
    def _populateOrbit(self):
        """Extract orbit state vectors from metadata/orbit group."""
        # TODO: time, posX/Y/Z, velX/Y/Z → StateVector → Orbit
        pass
    
    def extractImage(self):
        """Read SLC data from HDF5 and write to binary file."""
        # TODO: chunked read, complex32→complex64 conversion
        pass
    
    def extractDoppler(self):
        """Extract Doppler centroid and convert 2D LUT to 1D polynomial."""
        # TODO: read 2D LUT, median along azimuth, polyfit along range
        pass
```

---

*本文档由 ISCE2 backprojection 集成项目的分析阶段生成，作为 NISAR 支持实施的参考依据。*
