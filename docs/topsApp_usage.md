# topsApp 使用说明（含 backprojection burst refocus）

本文对应 `isce2/applications/topsApp.py` 当前实现，覆盖：
- 基本运行方式
- 分步运行/续跑
- 新增参数 `refocus bursts with backprojection`
- 常用参数（`offset SNR threshold`、`useGPU`）

## 1. 命令行

```bash
topsApp.py <input-file.xml>
topsApp.py --steps
topsApp.py --help
topsApp.py --help --steps
```

## 2. 分步运行

```bash
# 列出步骤并执行
topsApp.py topsApp.xml --steps

# 从某一步到某一步（示例）
topsApp.py topsApp.xml --steps --start=prepesd --end=rangecoreg
topsApp.py topsApp.xml --steps --start=fineoffsets --end=mergebursts
```

步骤名（与代码一致）：
`startup, preprocess, refocusBursts(条件步骤), computeBaselines, verifyDEM, topo, subsetoverlaps, coarseoffsets, coarseresamp, overlapifg, prepesd, esd, rangecoreg, fineoffsets, fineresamp, ion, burstifg, mergebursts, filter, unwrap, unwrap2stage, geocode, denseoffsets, filteroffsets, geocodeoffsets, endup`

## 3. XML 关键参数

### 3.1 启用 burst 回聚焦（ISCE3 backprojection）

```xml
<property name="refocus bursts with backprojection">True</property>
```

说明：
- 默认 `False`
- 当为 `True` 时，会在 `preprocess` 后插入 `refocusBursts` 步骤
- 该步骤会重写 burst SLC，建议在新输出目录中运行，或从 `preprocess` 开始重跑

### 3.2 `runRangeCoreg` 相关阈值

```xml
<property name="offset SNR threshold">2.5</property>
```

说明：
- 默认值为 `8.0`
- 数据相干较弱时可下调（如 `2.5` 或 `2.0`）以避免 `number of coherent points = 0`

### 3.3 GPU 开关

```xml
<property name="useGPU">True</property>
```

说明：
- 当前 `topsApp.py` 默认值为 `True`
- 即使开启 GPU，`runRangeCoreg` 的核心筛选仍由 `offset SNR threshold` 决定

## 4. 最小配置片段示例

```xml
<topsApp>
  <component name="topsinsar">
    <property name="useGPU">True</property>
    <property name="offset SNR threshold">2.5</property>
    <property name="refocus bursts with backprojection">True</property>
  </component>
</topsApp>
```
