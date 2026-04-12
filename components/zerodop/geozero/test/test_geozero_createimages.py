#!/usr/bin/env python3

import logging
import sys
import types
import unittest
from unittest import mock

# The source-tree test run does not provide the install-time top-level "isce" package.
# Geozero imports "from isce import logging", so provide a compatible shim.
if "isce" not in sys.modules:
    shim = types.ModuleType("isce")
    shim.logging = logging
    sys.modules["isce"] = shim

from zerodop.geozero.Geozero import Geocode


class _DummyInputImage:
    def __init__(self):
        self.imageType = "cpx"
        self.dataType = "CFLOAT"


class _DummyGeoImage:
    def __init__(self):
        self.imageType = None
        self.dataType = "CFLOAT"
        self.filename = None
        self.access_mode = None
        self.width = None

    def setFilename(self, filename):
        self.filename = filename

    def setAccessMode(self, mode):
        self.access_mode = mode

    def setWidth(self, width):
        self.width = width

    def setCaster(self, access_mode, dtype):
        self.caster = (access_mode, dtype)

    def createImage(self):
        pass

    def getImagePointer(self):
        return 12345


class GeozeroCreateImagesTest(unittest.TestCase):
    @mock.patch("zerodop.geozero.Geozero.IU.copyAttributes")
    @mock.patch("zerodop.geozero.Geozero.createIntImage")
    def test_create_images_without_dem_crop_still_sets_geo_width(
        self, mock_create_int_image, mock_copy_attributes
    ):
        geo_image = _DummyGeoImage()
        mock_create_int_image.return_value = geo_image

        obj = Geocode()
        obj.demCropFilename = ""
        obj.geoFilename = "dummy.geo"
        obj.inputImage = _DummyInputImage()
        obj.computeGeoImageWidth = mock.Mock(return_value=1111)
        obj.polyDoppler = mock.Mock()
        obj.polyDoppler.exportToC.return_value = 777

        obj.createImages()

        self.assertEqual(obj.demCropAccessor, 0)
        self.assertEqual(geo_image.width, 1111)
        self.assertEqual(obj.geoAccessor, 12345)
        self.assertEqual(obj.polyDopplerAccessor, 777)
        obj.computeGeoImageWidth.assert_called_once_with()
        mock_copy_attributes.assert_called_once_with(obj.inputImage, geo_image)


if __name__ == "__main__":
    unittest.main()
