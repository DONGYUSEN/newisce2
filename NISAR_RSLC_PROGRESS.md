# NISAR RSLC Implementation Progress

> Last updated: 2026-04-12
> Status: Phase 2.2a paused — waiting to install build deps

---

## 1. User Requests (As-Is)

1. "请从目录下的isce_ 文件恢复resume信息" — 恢复之前的session上下文
2. "从isce2目录中读取'/home/ysdong/Software/isce/isce2/NISAR_SUPPORT_ANALYSIS.md'" — 读取NISAR分析报告
3. "根据上述实施路线图，逐步开展工作，直至完成任务" — 按照NISAR_SUPPORT_ANALYSIS.md中的实施路线图执行Phase 1全部任务
4. "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed." — 继续执行剩余的Phase 1任务 (1.10, 1.11)
5. "继续phase2" — 开始Phase 2端到端验证
6. "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed." — 继续Phase 2工作，用户选择"Build ISCE2 from source"

## 2. Final Goal

实现NISAR_RSLC Sensor插件，使ISCE2可以通过stripmapApp处理NISAR RSLC数据，并完成端到端验证（Phase 2）。长期目标是之后做topsApp的改进工作。

## 3. Work Completed

### Phase 1: ALL COMPLETE ✅

#### Tasks 1.1-1.8: NISAR_RSLC.py 创建 ✅
- **文件**: `/home/ysdong/Software/isce/isce2/components/isceobj/Sensor/NISAR_RSLC.py` (~610 lines)
- 完整实现了所有核心方法:
  - `parse()`: HDF5打开, root path自动发现(LSAR/SSAR), product type验证(RSLC/SLC兼容), 频段/极化验证
  - `_populatePlatform()`: missionId, lookDirection → pointingDirection, Planet
  - `_populateInstrument()`: centerFrequency→wavelength, zeroDopplerTimeSpacing→PRF, slantRangeSpacing→rangePixelSize, rangeBandwidth
  - `_populateFrame()`: zeroDopplerStart/EndTime, slantRange, image dimensions, farRange, passDirection, orbitNumber
  - `_populateOrbit()`: orbit/time + position + velocity → StateVector → Orbit
  - `extractImage()`: 分块读取(512行/块), complex32(float16对)→complex64转换, 二进制写入, createSlcImage
  - `extractDoppler()`: 2D LUT→median→1D, UnivariateSpline插值到image grid (adaptive k=3/k=1/constant fallback), polyfit(capped order), _dopplerVsPixel + quadratic dict返回
  - `_parseEpochFromUnits()`: 从HDF5 dataset units属性解析'seconds since YYYY-MM-DD HH:MM:SS'格式的epoch

#### Task 1.9: 注册到SENSORS字典 ✅
- **文件**: `/home/ysdong/Software/isce/isce2/components/isceobj/Sensor/__init__.py` (修改)
- 添加了 `createNISAR_RSLC = partial(factory_template, 'NISAR_RSLC')` 和 `"NISAR_RSLC": createNISAR_RSLC`

#### Task 1.10: 单元测试 ✅
- **文件**: `/home/ysdong/Software/isce/isce2/components/isceobj/Sensor/test/test_NISAR_RSLC.py` (~690行)
- 37 tests total: 17 pass (standalone), 20 skipped (need ISCE2 installed)

#### Task 1.11: stripmapApp集成 ✅
- **文件**: `contrib/stack/stripmapStack/unpackFrame_NISAR_RSLC.py` (~73行)
- **文件**: `contrib/stack/stripmapStack/prepareNISAR_RSLCStack.py` (~110行)

### Phase 2.1: Integration Validation ✅ COMPLETE

- **文件**: `/home/ysdong/Software/isce/isce2/components/isceobj/Sensor/test/test_NISAR_RSLC_integration.py` (~683行, 新建)
- **20/20 tests ALL PASS** against real NISAR-format ISCE3 test data:
  - `SanAnd_129.h5`: NISAR LSAR/SLC format, complex64, 150x200 (freqA) / 150x50 (freqB), dual-frequency
  - `SanAnd_138.h5`: Same track pair for InSAR, 150x400 (freqA)
  - `REE_RSLC_out17.h5`: complex32 (float16 pairs), 129x129, single freq
- Tests cover: root path discovery, product group discovery, epoch parsing, full parse (freqA/B/invalid), pair consistency, extractImage (complex64 + complex32 + freqB), extractDoppler (all 3 files), sensing time range, orbit coverage, slant range geometry

**Bugs found and fixed during Phase 2.1:**
1. `extractDoppler()`: `UnivariateSpline` crashed with `m<=k` when Doppler LUT had ≤3 clipped range samples — fixed by adaptive spline order (k=3→k=1→constant fallback)
2. `extractDoppler()`: `np.polyfit` RankWarning when fit_order exceeded meaningful LUT samples — fixed by capping `fit_order = min(min(41, len(pix) - 1), max(1, n_pts - 1))`

### Phase 2.2: Building ISCE2 — PAUSED ⏸️

System environment checked:
- ✅ cmake 3.31.6, gcc 14.2.0, g++, python3.12, numpy 2.4.2, scipy 1.17.1, h5py 3.16.0
- ✅ Runtime libs: libgdal36, libgfortran5, libmotif-common, libxft2, libxt6t64, python3-gdal (system)
- ❌ **gfortran compiler NOT installed** (only libgfortran5 runtime)
- ❌ **libgdal-dev NOT installed** (no gdal-config, no headers in /usr/include/)
- ❌ **libmotif-dev, libxft-dev, libxt-dev NOT installed** (no dev headers)
- ❌ **No sudo access** — cannot `apt-get install`
- ✅ **mamba dry-run succeeded** — `mamba install -c conda-forge gfortran_linux-64 gdal` resolves (37 packages, 233MB)

## 4. Remaining Tasks (Resumption Checklist)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.2a | Install build deps via mamba | **NEXT** | `mamba install -c conda-forge gfortran_linux-64 gdal` (need libmotif/Xft/Xt too, or disable mdx/insarApp GUI) |
| 2.2b | CMake configure ISCE2 | PENDING | `cmake -DCMAKE_INSTALL_PREFIX=... -DPYTHON_MODULE_DIR=... ../` |
| 2.2c | CMake build ISCE2 | PENDING | `make -j$(nproc)` |
| 2.2d | Install ISCE2 and verify `import isce` | PENDING | `make install`, set ISCE_HOME, PYTHONPATH, PATH |
| 2.2e | Run stripmapApp end-to-end with SanAnd_129 + SanAnd_138 | PENDING | |
| 2.3 | Compare with ISCE3 processing results | PENDING | |
| 2.4 | Test edge cases (multi-frequency, different polarizations) | PENDING | |
| Phase 3 | topsApp改进工作 (future) | NOT STARTED | |

## 5. Key Technical Details

### NISAR RSLC HDF5 path structure:
```
/science/{LSAR|SSAR}/
  identification/  → missionId, productType, lookDirection, orbitPassDirection, absoluteOrbitNumber
  {RSLC|SLC}/
    swaths/
      zeroDopplerTime (attrs: units='seconds since YYYY-MM-DD HH:MM:SS')
      zeroDopplerTimeSpacing
      frequency{A|B}/
        slantRange, slantRangeSpacing, processedCenterFrequency
        processedRangeBandwidth, {HH|HV|VH|VV} (complex SLC data)
    metadata/
      orbit/ → time, position[N,3], velocity[N,3]
      processingInformation/parameters/
        frequency{A|B}/dopplerCentroid (2D LUT)
        slantRange (shared coordinate vector)
```

### Usage pattern:
```python
obj = createSensor("NISAR_RSLC")
obj.configure()
obj.hdf5file = "/path/to/nisar.h5"
obj.frequency = "A"
obj.polarization = "HH"
obj.output = "/path/to/output.slc"
obj.extractImage()
obj.extractDoppler()
```

### Test Data Files (ISCE3 test data, NISAR format):
- `/home/ysdong/Software/isce/isce3/tests/data/SanAnd_129.h5` — LSAR/SLC, complex64, 150x200(A)/150x50(B)
- `/home/ysdong/Software/isce/isce3/tests/data/SanAnd_138.h5` — Same track pair, 150x400(A)/150x50(B)
- `/home/ysdong/Software/isce/isce3/tests/data/REE_RSLC_out17.h5` — LSAR/SLC, complex32, 129x129

### Build Environment:
- Python: `/home/ysdong/miniforge3/bin/python3` (3.12, base conda env)
- ISCE2 source: `/home/ysdong/Software/isce/isce2/`
- CMake build system
- No sudo — use mamba/conda for deps

## 6. Discoveries

1. ISCE3 test data directory has NISAR-format test files perfect for validation.
2. SanAnd files use productType="RSLC" but product path is "SLC" — our `_get_product_group()` correctly falls back.
3. NISAR RSLC is continuous SLC (not burst) — stripmapApp is appropriate.
4. Small Doppler LUTs can break cubic spline and polyfit — both fixed with adaptive fallbacks.
5. No sudo access on this machine — must use conda/mamba for build deps.

## 7. Files Created/Modified

### Created:
- `components/isceobj/Sensor/NISAR_RSLC.py` — Main sensor plugin (~610 lines)
- `components/isceobj/Sensor/test/__init__.py` — Empty init
- `components/isceobj/Sensor/test/test_NISAR_RSLC.py` — Unit tests (~690 lines)
- `components/isceobj/Sensor/test/test_NISAR_RSLC_integration.py` — Integration tests (~683 lines)
- `contrib/stack/stripmapStack/unpackFrame_NISAR_RSLC.py` — Unpack script (~73 lines)
- `contrib/stack/stripmapStack/prepareNISAR_RSLCStack.py` — Stack prep script (~110 lines)

### Modified:
- `components/isceobj/Sensor/__init__.py` — Added NISAR_RSLC to SENSORS dict

## 8. Explicit Constraints (Verbatim)

- "注意代码也要集成到isce2中" — 代码要集成进ISCE2代码库
- "首先从stripmapApp开始，做完后开始topsApp的改进工作"

## 9. Resumption Instructions

To resume in a new session:
1. Read this file: `/home/ysdong/Software/isce/isce2/NISAR_RSLC_PROGRESS.md`
2. Read the master plan: `/home/ysdong/Software/isce/isce2/NISAR_SUPPORT_ANALYSIS.md`
3. Continue from Phase 2.2a: install build deps via mamba, then cmake configure/build/install
4. After ISCE2 is built, run the 20 skipped unit tests and the stripmapApp end-to-end test
