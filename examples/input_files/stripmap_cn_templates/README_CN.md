# StripmapApp 中文模板说明（天仪 / Lutan1）

目录下模板文件：

- `stripmapApp_LUTAN1_cn.xml`
- `reference_LUTAN1_cn.xml`
- `secondary_LUTAN1_cn.xml`
- `stripmapApp_TIANYI_cn.xml`
- `reference_TIANYI_cn.xml`
- `secondary_TIANYI_cn.xml`

## 使用方式

1. 复制对应模板到你的处理目录。  
2. 按注释修改数据路径与 DEM 路径。  
3. 运行：

```bash
stripmapApp.py stripmapApp_LUTAN1_cn.xml --steps
# 或
stripmapApp.py stripmapApp_TIANYI_cn.xml --steps
```

## 与 GPU/Fallback 相关的建议

- 模板默认 `<property name="use GPU">True</property>`。  
- 你当前代码分支已实现：可用时优先 GPU，GPU 失败自动回退 CPU/常规路径。  
- 对 stripmap 细配准（PyCuAmpcor）可通过环境变量控制：

```bash
# 1) 优先使用 GPU ampcor（默认就是 True）
export ISCE_PREFER_GPU_AMPCOR=1

# 2) 若 GPU 失败，是否允许回退到 CPU ampcor（默认 False）
#    你当前策略是 GPU 失败优先走 external registration，所以通常保持 0
export ISCE_ALLOW_CPU_AMPCOR_FALLBACK=0
```

## 多普勒说明

- 当前代码里 `LUTAN1` / `TIANYI` 已按 Native Doppler 流程处理。  
- `LUTAN1` 模板默认用于 Native 处理。若做 Native vs Zero-Doppler 对照，请仅改对照组关键项并保持其他参数一致。
