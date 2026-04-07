# ISCE in docker

## Build

1. Clone repo:
   ```
   git clone https://github.com/isce-framework/isce2.git
   ```
1. Change directory:
   ```
   cd isce2
   ```
1. Build image:
   ```
   docker build --rm --force-rm -t hysds/isce2:latest -f docker/Dockerfile .
   ```
   For cuda version:
   ```
   docker build --rm --force-rm -t hysds/isce2:latest-cuda -f docker/Dockerfile.cuda .
   ```

## APT mirror in Docker image

`docker/Dockerfile`, `docker/Dockerfile.cuda`, and `MintPy/Dockerfile` now
default to a mainland China apt mirror (`mirrors.aliyun.com`).

Override examples:

- Disable CN mirror and use upstream source:
  ```bash
  docker build -f docker/Dockerfile \
    --build-arg USE_CN_MIRROR=false \
    -t hysds/isce2:latest .
  ```
- Keep CN mirror but switch to another host (e.g., TUNA):
  ```bash
  docker build -f docker/Dockerfile.cuda \
    --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
    -t hysds/isce2:latest-cuda .
  ```

## Python package mirror in Docker image

Both `docker/Dockerfile` and `docker/Dockerfile.cuda` now set:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

This is useful in mainland China for faster and more stable `pip install`.

If needed, override at runtime:

```bash
docker run --rm \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  -e PIP_TRUSTED_HOST=pypi.org \
  <image> <cmd>
```

## SNAPHU in CUDA image

`docker/Dockerfile.cuda` installs `snaphu` from Ubuntu apt packages and sets:

```bash
ISCE_SNAPHU_RUN_MODE=external
ISCE_SNAPHU_BIN=/usr/bin/snaphu
ISCE_SNAPHU_NPROC=8
ISCE_SNAPHU_NTILEROW=2
ISCE_SNAPHU_NTILECOL=2
ISCE_SNAPHU_ROWOVRLP=400
ISCE_SNAPHU_COLOVRLP=400
```

You can override them in `docker run -e ...` as needed.

## Automatic postprocess hook before `endup`

Main ISCE2 apps (`topsApp`, `insarApp`, `stripmapApp`, `alos2App`,
`alos2burstApp`, `rtcApp`, `isceApp`) run an auto postprocess command before
`endup()`.

Environment controls:

```bash
export ISCE_AUTO_POSTPROCESS=1
export ISCE_AUTO_POSTPROCESS_STRICT=0
export ISCE_AUTO_POSTPROCESS_CMD=isce2-tops-postprocess
export ISCE_AUTO_POSTPROCESS_ARGS=""
```

Defaults:
- `ISCE_AUTO_POSTPROCESS=1`
- `ISCE_AUTO_POSTPROCESS_STRICT=0`
- `ISCE_AUTO_POSTPROCESS_CMD=isce2-tops-postprocess`

Set `ISCE_AUTO_POSTPROCESS=0` to disable.

## External Registration Quality-Fail Policy

Stripmap `runRefineSecondaryTiming` now supports a non-fallback mode for
integrated external registration quality-gate failures:

- Keep external-registration polynomials (no Ampcor fallback).
- Do **not** auto-enable `doDenseOffsets` / rubbersheet flags.
  Dense offsets only run when XML explicitly sets `doDenseOffsets=True`.

Environment controls:

```
export ISCE_EXTERNAL_REGISTRATION_NO_AMPCOR_FALLBACK_ON_QUALITY_FAIL=1
export ISCE_EXTERNAL_REGISTRATION_FORCE_DENSE_RUBBERSHEET_ON_QUALITY_FAIL=1
export ISCE_EXTERNAL_REGISTRATION_NO_AMPCOR_FALLBACK_ON_ERROR=1
```

Defaults are `1` (enabled) for all three.
Note: `ISCE_EXTERNAL_REGISTRATION_FORCE_DENSE_RUBBERSHEET_ON_QUALITY_FAIL`
is kept for compatibility, but dense/rubbersheet is no longer auto-enabled.

Note:
- `ISCE_EXTERNAL_REGISTRATION_ENABLED=0` means external registration is disabled and
  processing follows the Ampcor fallback path.
