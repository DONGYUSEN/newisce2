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
