# image2sar 使用说明（对应 `optical2sar.py`）

说明：
- 代码文件为 `isce2/applications/optical2sar.py`
- 这里按你提出的名称统一称为 `image2sar`
- 作用：把 UTM 光学影像重采样到 SAR 距离-方位网格

## 1. 基本命令

```bash
python3 applications/optical2sar.py \
  --optical <optical.tif> \
  --reference <SAFE.zip|reference.xml> \
  --out <out_on_sar_grid.tif>
```

## 2. 常见输入模式

### 2.1 参考为 Sentinel-1 SAFE zip

```bash
python3 applications/optical2sar.py \
  --optical optical_utm.tif \
  --reference S1A_xxx.SAFE.zip \
  --product-type slc \
  --swath iw2 \
  --pol vv \
  --out optical_on_sar_iw2.tif
```

### 2.2 参考为 ISCE XML（已存在 lat/lon 查找表）

```bash
python3 applications/optical2sar.py \
  --optical optical_utm.tif \
  --reference reference.xml \
  --lat-rdr geometry/lat.rdr \
  --lon-rdr geometry/lon.rdr \
  --out optical_on_sar.tif
```

### 2.3 参考为 ISCE XML（自动生成查找表）

```bash
python3 applications/optical2sar.py \
  --optical optical_utm.tif \
  --reference reference.xml \
  --dem /Work/dem/demLat_N28_N31_Lon_E093_E097.dem.wgs84 \
  --rdr-dir ./geometry \
  --out optical_on_sar.tif
```

## 3. 参数速查

必选参数：
- `--optical`：输入光学 GeoTIFF
- `--reference`：SAR 参考（SAFE zip 或 XML）
- `--out`：输出到 SAR 网格的 GeoTIFF

常用可选参数：
- `--product-type {slc,grd}`、`--swath`、`--pol`（zip 参考时筛选）
- `--burst-index`（TOPS bursts XML 时选择 burst，`-1` 为中间）
- `--lat-rdr`、`--lon-rdr`（显式指定查找表）
- `--dem`、`--dem-dir`、`--rdr-dir`（自动生成查找表时）
- `--zmin`、`--zmax`（地形高程范围）
- `--optical-epsg`（光学图像缺少投影时）
- `--block-lines`（分块大小，默认 `256`）
- `--fill-value`（输出无效值，默认 `-9999`）
