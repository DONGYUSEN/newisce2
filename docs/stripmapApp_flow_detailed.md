# stripmapApp 详细数据处理流程图（含判断与转换）

本文基于 `isce2/applications/stripmapApp.py` 与 `components/isceobj/StripmapProc/*` 实际代码路径整理，重点覆盖：
- 原始/已聚焦数据分支（RAW/SLC）
- Doppler 处理与几何系统（Native/Zero Doppler）
- GPU/CPU 与失败回退分支
- misregistration 的 PRF/采样率比例补偿
- dense offsets / rubbersheet / fine resample 的触发条件
- split-spectrum / 子带干涉 / dispersive 处理链

## 0. Usage 速查（含 backproject 集成）

### 0.1 命令行

```bash
stripmapApp.py <input-file.xml>
stripmapApp.py --steps
stripmapApp.py --help
stripmapApp.py --help --steps
```

### 0.2 分步运行与续跑

```bash
# 查看所有步骤
stripmapApp.py stripmapApp.xml --steps

# 从指定步骤到指定步骤
stripmapApp.py stripmapApp.xml --steps --start=geo2rdr --end=geocode
```

常见步骤名（与代码一致）：
`startup, preprocess, cropraw, formslc, cropslc, verifyDEM, topo, normalize_secondary_sampling, geo2rdr, rdrdem_offset, rect_rgoffset, coarse_resample, misregistration, refined_resample, dense_offsets, rubber_sheet_range, rubber_sheet_azimuth, fine_resample, split_range_spectrum, sub_band_resample, interferogram, filter, unwrap, sub_band_interferogram, filter_low_band, filter_high_band, unwrap_low_band, unwrap_high_band, ionosphere, geocode, geocodeoffsets`

### 0.3 配置 backproject focuser

`stripmapApp.py` 现支持两种聚焦器：
- `formslc`（默认，ROI range-Doppler）
- `backproject`（ISCE3 时域后向投影）

在 XML 中设置：

```xml
<property name="focuser">backproject</property>
```

默认值：

```xml
<property name="focuser">formslc</property>
```

## 1. 主流程图（端到端）

```mermaid
flowchart TD
    A[Start: stripmapApp.run] --> B[Configure parameters and facilities]
    B --> C[preprocess: runPreprocessor]

    C --> C1{reference/secondary raw?}
    C1 -->|RAW| C2[make_raw + Doppler method]
    C1 -->|SLC| C3[Read SLC metadata + geometry system]
    C2 --> C4[Save referenceRawProduct/secondaryRawProduct]
    C3 --> C5[Save referenceSlcProduct/secondarySlcProduct]
    C4 --> D[cropraw: runCrop(True)]
    C5 --> D

    D --> D1{regionOfInterest set?}
    D1 -->|No| D2[Skip raw crop]
    D1 -->|Yes| D3[geo bbox -> az/range box + cropFrame]
    D2 --> E[formslc: runFormSLC]
    D3 --> E

    E --> E1{raw products exist?}
    E1 -->|Yes| E2[Focus RAW to SLC]
    E1 -->|No| E3[Skip focusing]
    E2 --> F[cropslc: runCrop(False)]
    E3 --> F

    F --> F1{regionOfInterest set?}
    F1 -->|No| F2[Skip SLC crop]
    F1 -->|Yes| F3[geo bbox -> az/range box + cropFrame]
    F2 --> G[verifyDEM]
    F3 --> G

    G --> G1{demFilename provided?}
    G1 -->|Yes| G2[Load/convert DEM to WGS84]
    G1 -->|No| G3[Estimate bbox from ref/sec geometry and download DEM]
    G2 --> H[topo]
    G3 --> H

    H --> H1{useGPU and GPU topo available?}
    H1 -->|Yes| H2[runTopoGPU]
    H1 -->|No or fail| H3[runTopoCPU]
    H2 --> I[geo2rdr]
    H3 --> I

    I --> I1{useGPU and GPU geo2rdr available?}
    I1 -->|Yes| I2[runGeo2rdrGPU]
    I1 -->|No or fail| I3[runGeo2rdrCPU]
    I2 --> J[coarse_resample]
    I3 --> J

    J --> K[misregistration: runRefineSecondaryTiming]
    K --> L[refined_resample]

    L --> M[dense_offsets]
    M --> M1{doDenseOffsets?}
    M1 -->|No| M2[Skip dense offsets]
    M1 -->|Yes| M3[GPU -> External CPU -> DenseAmpcor fallback]

    M2 --> N[rubber_sheet_range]
    M3 --> N

    N --> N1{doRubbersheetingRange and doDenseOffsets?}
    N1 -->|Yes| N2[Filter dense range offsets + add to geometric offsets]
    N1 -->|No| N3[Skip range rubbersheet]

    N2 --> O[rubber_sheet_azimuth]
    N3 --> O

    O --> O1{doRubbersheetingAzimuth and doDenseOffsets?}
    O1 -->|Yes| O2[Filter dense az offsets + add to geometric offsets]
    O1 -->|No| O3[Skip azimuth rubbersheet]

    O2 --> P[fine_resample]
    O3 --> P

    P --> P1{any rubbersheet enabled?}
    P1 -->|No| P2[Skip fine resample]
    P1 -->|Yes| P3[Resample with rubber-sheeted offsets]

    P2 --> Q[split_range_spectrum]
    P3 --> Q

    Q --> Q1{doSplitSpectrum?}
    Q1 -->|No| Q2[Skip split-spectrum]
    Q1 -->|Yes| Q3[Generate low/high subband SLCs]

    Q2 --> R[sub_band_resample]
    Q3 --> R

    R --> R1{doSplitSpectrum?}
    R1 -->|No| R2[Skip sub-band resample]
    R1 -->|Yes| R3[Resample secondary low/high subbands]

    R2 --> S[interferogram(full)]
    R3 --> S

    S --> T[sub_band_interferogram]
    T --> T1{doDispersive?}
    T1 -->|No| T2[Skip sub-band interferograms]
    T1 -->|Yes| T3[Generate low/high interferograms]

    T2 --> U[filter full/low/high]
    T3 --> U

    U --> V[unwrap full/low/high]
    V --> W[ionosphere: runDispersive]
    W --> W1{doDispersive?}
    W1 -->|No| W2[Skip dispersive estimation]
    W1 -->|Yes| W3[Estimate dispersive/non-dispersive + filtering + unwrap-error correction]

    W2 --> X[geocode]
    W3 --> X

    X --> Y[geocodeoffsets]
    Y --> Y1{is offset mode and doDenseOffsets?}
    Y1 -->|No| Y2[Skip offset geocode]
    Y1 -->|Yes| Y3[Geocode offsets]

    Y2 --> Z[End]
    Y3 --> Z
```

## 1.1 misregistration 与 Dense/Rubbersheet 细分支

```mermaid
flowchart TD
    A[runRefineSecondaryTiming] --> B[Compute azratio/rgratio]
    B --> C{useGPU?}

    C -->|Yes| D{GPU Ampcor available and success?}
    D -->|Yes| E[Save azpoly/rgpoly (scaled by ratios)]
    D -->|No| F{ISCE_ALLOW_CPU_AMPCOR_FALLBACK=1 ?}
    F -->|Yes| G[CPU Ampcor fallback -> fit poly -> scale by ratios]
    F -->|No| H[Write zero misregistration polynomials]

    C -->|No| I[Integrated external registration]
    I --> J{Quality gates pass?}
    J -->|Yes| K[Accept external azpoly/rgpoly (already ratio-scaled)]
    J -->|No but keep_on_quality_fail| L[Accept external result + mark gate failure]
    J -->|No and strict reject| M{allow CPU fallback?}
    M -->|Yes| G
    M -->|No| H

    E --> N[runResampleSlc('refined')]
    G --> N
    H --> N
    K --> N
    L --> N

    N --> O[runDenseOffsets]
    O --> P{doDenseOffsets?}
    P -->|No| Q[Skip dense offsets; rubbersheet steps will also skip]
    P -->|Yes| R{Dense path}
    R -->|GPU success| S[Dense offsets from PyCuAmpcor]
    R -->|GPU fail| T{ISCE_DENSE_EXTERNAL_ENABLED?}
    T -->|Yes and external success| U[Dense offsets from external CPU]
    T -->|No or external fail| V[DenseAmpcor CPU fallback]

    S --> W[runRubbersheetRange/Azimuth if enabled]
    U --> W
    V --> W
    Q --> X[runResampleSlc('fine') -> skip if no rubbersheet]
    W --> X
```

## 2. 关键判断与“转换”节点说明

### 2.1 Doppler 方法选择与几何系统
- `referenceDopplerMethod/secondaryDopplerMethod` 未指定时，按传感器自动选（`useDEFAULT` 或 `useDOPIQ`）。
- `useDEFAULT` 路径下，会调用传感器自身 `extractDoppler()` 读取/拟合 DC，并写入 `frame._dopplerVsPixel`。
- 对 SLC 还会判定几何系统：`Zero Doppler` 或 `Native Doppler`，后续 `topo/geo2rdr/geocode` 都依赖该标志选择 Doppler 模型。

### 2.2 RAW -> SLC 转换
- 若输入是 RAW：`runFormSLC` 执行聚焦，输出 `*_slc` 产品。
- 若输入是 SLC：跳过聚焦，直接进入裁剪/几何处理。

### 2.3 坐标转换链
- `runTopo`: 雷达坐标 -> 地理坐标（生成 `lat/lon/hgt/los`）。
- `runGeo2rdr`: 地理坐标 -> 雷达坐标（生成 `range/azimuth offset` 几何偏移）。
- `runGeocode`: 将雷达产品映射回地理坐标（最终地理编码输出）。

### 2.4 PRF 比例补偿（misregistration 核心）
- 在 `runRefineSecondaryTiming` 中计算：
  - `azratio = secondary.PRF / reference.PRF`
  - `rgratio = dR_reference / dR_secondary`
- 对估计出的 `azpoly/rgpoly` 系数按上述比例缩放后再保存。
- 这一步是 stripmapApp 中“完整 PRF/采样率比例补偿”的核心实现位点。

### 2.5 misregistration 计算分支（重要）
- 若 `useGPU=True`：
  1. 先尝试 GPU Ampcor；
  2. 失败时不会自动切到 integrated external（当前策略）；
  3. 后续按环境变量决定 CPU Ampcor fallback 或零多项式。
- 若 `useGPU=False`：
  1. 强制 integrated external registration（coarse+fine）；
  2. 带质量门限（coarse valid/spread, az RMS）；
  3. 失败后再看是否允许 CPU Ampcor fallback。

### 2.6 dense offsets / rubbersheet / fine resample 联动
- `doDenseOffsets=False`：dense offsets 不跑，rubbersheet 即使打开也会跳过。
- `doRubbersheetingRange` 或 `doRubbersheetingAzimuth` 为真时，才有必要执行 `fine_resample`。
- `fine_resample` 本身在“两个 rubbersheet 都为假”时会直接跳过。

### 2.7 split-spectrum 与 dispersive 链
- `doSplitSpectrum=False`：子带 SLC 与子带重采样全部跳过。
- `doDispersive=False`：子带 interferogram/filter/unwrap 以及 `runDispersive` 都会跳过。
- 两者是独立开关：`doDispersive=True` 并不会自动把 `doSplitSpectrum` 改成 `True`。

## 3. 产物流转（关键文件）

| 阶段 | 关键输入 | 关键输出 |
|---|---|---|
| preprocess | reference/secondary sensor config | `referenceRawProduct` / `secondaryRawProduct` 或 `referenceSlcProduct` / `secondarySlcProduct` |
| formslc | RAW product | `*_slc` 产品 |
| cropslc | SLC product | `referenceSlcCropProduct` / `secondarySlcCropProduct` |
| topo | reference SLC crop + DEM | `geometry/*.full`（lat/lon/hgt/los） |
| geo2rdr | secondary SLC crop + geometry | `offsets/range.off`, `offsets/azimuth.off` |
| coarse/refined/fine resample | SLC + offsets + misreg poly | `coregisteredSlc/coarse|refined|fine_coreg.slc` |
| misregistration | reference SLC + coarse coreg SLC | `misreg/misreg_az.xml`, `misreg/misreg_rg.xml` |
| dense/rubbersheet | refined coreg SLC + offsets | dense offset 栅格 + rubber-sheeted offsets |
| interferogram | reference SLC + coreg secondary SLC | `ifg/topophase.flat/int` + coherence |
| split spectrum | reference/secondary SLC | low/high subband SLC |
| dispersive | low/high filtered-unwrapped ifg | `ionosphere/dispersive`, `nondispersive` (+ filtered/修正版本) |
| geocode | 产品列表 + DEM + orbit | `*.geo` |

## 4. 建议分析顺序（你可按此对日志排错）

1. 先看 `preprocess` 后的几何系统与 Doppler 方法是否符合预期（Native/Zero, useDEFAULT/useDOPIQ）。
2. 再看 `misregistration` 是否成功生成非零 `misreg_az/rg` 多项式。
3. 若开启 rubbersheet，确认 dense offsets 有效点数量与过滤后偏移场是否合理。
4. 最后检查 full/sub-band interferogram、unwrap、dispersive 输出链是否一致。

## 5. 代码锚点（便于你继续追）
- 主步骤编排：`applications/stripmapApp.py`（`_steps`, `main`）
- 预处理与 Doppler 入口：`components/isceobj/StripmapProc/runPreprocessor.py`
- 几何链：`runTopo.py`, `runGeo2rdr.py`, `runGeocode.py`
- 配准链：`runResampleSlc.py`, `runRefineSecondaryTiming.py`, `externalRegistration.py`
- 稠密偏移与橡皮片：`runDenseOffsets.py`, `runRubbersheetAzimuth.py`, `runRubbersheetRange.py`
- 干涉/滤波/解缠/电离层：`runInterferogram.py`, `runFilter.py`, `runUnwrap*.py`, `runDispersive.py`
