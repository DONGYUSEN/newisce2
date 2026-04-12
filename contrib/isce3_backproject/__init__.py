#!/usr/bin/env python3
"""
ISCE3 backprojection SAR focusing algorithm, migrated to ISCE2.

Provides time-domain backprojection (TDBP) for SAR image formation.

Usage
-----
>>> from isce3_backproject.backproject import (
...     DateTime, Orbit, RadarGridParameters, RadarGeometry,
...     DEMInterpolator, KnabKernel, TabulatedKernel,
...     backproject, DryTroposphereModel, LookSide,
... )
"""

try:
    from .backproject import *  # noqa: F401,F403
    from .backproject import backproject  # explicit re-export
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import backproject C++ extension: {e}. "
        "Make sure the module was built (scons install).",
        ImportWarning,
        stacklevel=2,
    )
