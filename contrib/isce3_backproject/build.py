#!/usr/bin/env python3
"""
Standalone Makefile-like build script for isce3_backproject.

Usage:
    python build.py              # build everything
    python build.py clean        # remove build artifacts
    python build.py test         # build + run smoke test

Can be used independently from the ISCE2 SCons build system.
"""

import os
import sys
import subprocess
import sysconfig
import glob

THISDIR = os.path.dirname(os.path.abspath(__file__))
INCDIR = os.path.join(THISDIR, "include")
VENDORDIR = os.path.join(THISDIR, "vendor")
BINDINGSDIR = os.path.join(THISDIR, "bindings")
BUILDDIR = os.path.join(THISDIR, "build")

CXX = os.environ.get("CXX", "g++")
CXXFLAGS = ["-std=c++17", "-O2", "-fopenmp", "-fPIC"]
INCLUDES = [f"-I{INCDIR}", f"-I{VENDORDIR}"]


def find_sources():
    srcs = []
    for root, _, files in os.walk(os.path.join(INCDIR, "isce3")):
        for f in files:
            if f.endswith(".cpp"):
                srcs.append(os.path.join(root, f))
    return sorted(srcs)


def obj_name(cpp_path):
    base = os.path.splitext(os.path.basename(cpp_path))[0]
    return os.path.join(BUILDDIR, base + ".o")


def run(cmd, **kw):
    print(f"  {' '.join(cmd)}")
    subprocess.check_call(cmd, **kw)


def compile_one(src):
    obj = obj_name(src)
    if os.path.exists(obj) and os.path.getmtime(obj) > os.path.getmtime(src):
        return obj
    run([CXX] + CXXFLAGS + INCLUDES + ["-c", src, "-o", obj])
    return obj


def build_lib(objects):
    target = os.path.join(BUILDDIR, "libisce3_backproject.so")
    run([CXX, "-shared", "-fopenmp", "-o", target] + objects)
    return target


def build_pymod(objects):
    try:
        import pybind11

        pb11_inc = pybind11.get_include()
    except ImportError:
        print("WARNING: pybind11 not found, skipping Python binding")
        return None

    py_inc = sysconfig.get_path("include")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    binding_src = os.path.join(BINDINGSDIR, "pyBackproject.cpp")
    binding_obj = os.path.join(BUILDDIR, "pyBackproject.o")

    run(
        [CXX]
        + CXXFLAGS
        + INCLUDES
        + [f"-I{pb11_inc}", f"-I{py_inc}", "-c", binding_src, "-o", binding_obj]
    )

    target = os.path.join(BUILDDIR, "backproject" + ext_suffix)
    run([CXX, "-shared", "-fopenmp", "-o", target, binding_obj] + objects)
    symlink = os.path.join(THISDIR, "backproject" + ext_suffix)
    if os.path.islink(symlink):
        os.unlink(symlink)
    os.symlink(target, symlink)
    return target


def clean():
    import shutil

    if os.path.isdir(BUILDDIR):
        shutil.rmtree(BUILDDIR)
        print(f"Removed {BUILDDIR}")
    for f in glob.glob(os.path.join(THISDIR, "backproject.*.so")):
        if os.path.islink(f):
            os.unlink(f)
            print(f"Removed symlink {f}")


def smoke_test():
    sys.path.insert(0, BUILDDIR)
    import backproject as bp
    import numpy as np

    import math

    dt = bp.DateTime(2023, 6, 15, 10, 30, 0)
    svecs = []
    for i in range(20):
        sv = bp.StateVector()
        sv.datetime = bp.DateTime(f"2023-06-15T10:30:{i:02d}.000000000")
        angle = 0.001 * i
        r = 7.071e6
        sv.position = bp.Vec3(r * math.cos(angle), r * math.sin(angle), 0.0)
        sv.velocity = bp.Vec3(
            -r * 0.001 * math.sin(angle), r * 0.001 * math.cos(angle), 0.0
        )
        svecs.append(sv)
    orbit = bp.Orbit(svecs)
    lut = bp.LUT2d(0.0)
    in_grid = bp.RadarGridParameters(
        0.0, 0.0556, 1000.0, 850000.0, 7.5, bp.LookSide.Right, 64, 128, dt
    )
    out_grid = bp.RadarGridParameters(
        0.01, 0.0556, 500.0, 855000.0, 15.0, bp.LookSide.Right, 8, 16, dt
    )

    in_geom = bp.RadarGeometry(in_grid, orbit, lut)
    out_geom = bp.RadarGeometry(out_grid, orbit, lut)
    dem = bp.DEMInterpolator(0.0)
    kernel = bp.TabulatedKernel(bp.KnabKernel(8.0, 0.9), 10000)

    rng = np.random.default_rng(42)
    in_data = (
        rng.standard_normal((64, 128)) + 1j * rng.standard_normal((64, 128))
    ).astype(np.complex64)

    out, height, ec = bp.backproject(
        in_data, in_geom, out_geom, dem, fc=5.405e9, ds=5.0, kernel=kernel
    )

    assert ec == bp.ErrorCode.Success, f"backproject returned {ec}"
    assert out.shape == (8, 16), f"unexpected shape {out.shape}"
    print(f"Smoke test PASSED: output {out.shape}, ErrorCode.Success")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "build"

    if action == "clean":
        clean()
        sys.exit(0)

    os.makedirs(BUILDDIR, exist_ok=True)
    sources = find_sources()
    print(f"Found {len(sources)} C++ source files")

    objects = [compile_one(s) for s in sources]
    print(f"\nLinking shared library...")
    lib = build_lib(objects)
    print(f"  -> {lib}")

    print(f"\nBuilding pybind11 module...")
    pymod = build_pymod(objects)
    if pymod:
        print(f"  -> {pymod}")

    if action == "test":
        print(f"\nRunning smoke test...")
        smoke_test()

    print("\nDone.")
