# rtcApp 使用说明

本文对应 `isce2/applications/rtcApp.py` 当前实现。

## 1. 命令行

```bash
rtcApp.py <input-file.xml>
rtcApp.py --steps
rtcApp.py --help
rtcApp.py --help --steps
```

## 2. 分步运行

```bash
# 逐步执行
rtcApp.py rtcApp.xml --steps

# 指定起止步骤（示例）
rtcApp.py rtcApp.xml --steps --start=verifyDEM --end=geocode
```

步骤名（与代码一致）：
`startup, preprocess, verifyDEM, multilook, topo, normalize, geocode, endup`

## 3. 常用 XML 参数

```xml
<property name="useHighResolutionDemOnly">False</property>
<property name="demFilename">/Work/xxx/dem/dem.wgs84</property>
<property name="apply water mask">False</property>
<property name="geocode bounding box">[S, N, W, E]</property>
<property name="epsg id">32647</property>
<property name="geocode spacing">30</property>
<property name="geocode interpolation method">bilinear</property>
```

说明：
- `demFilename` 未设置时会按范围自动下载/拼接 DEM
- `epsg id`、`geocode spacing` 用于控制输出投影与分辨率
- `normalize` 步骤负责归一化（例如 gamma0）

## 4. 最小配置片段示例

```xml
<rtcApp>
  <component name="grdsar">
    <property name="sensor name">SENTINEL1</property>
    <property name="demFilename">/Work/rtc/dem/dem.wgs84</property>
    <property name="epsg id">32647</property>
    <property name="geocode spacing">30</property>
  </component>
</rtcApp>
```
