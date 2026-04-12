// Minimal DEMInterpolator for ISCE3 backprojection migrated to ISCE2.
// Stripped version: no GDAL/io::Raster, no loadDEM, no interpolateXY.
// Provides interpolateLonLat() via a user-supplied callback or constant height.

#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include "forward.h"

#include <isce3/core/Constants.h>
#include <isce3/error/ErrorCode.h>

class isce3::geometry::DEMInterpolator {

public:
    using HeightFunc = std::function<double(double lon, double lat)>;

    DEMInterpolator(float height = 0.0, int epsg = 4326)
        : _refHeight{height},
          _minValue{height}, _meanValue{height}, _maxValue{height},
          _epsgcode{epsg},
          _heightFunc{nullptr} {}

    DEMInterpolator(HeightFunc func, float refHeight, int epsg = 4326)
        : _refHeight{refHeight},
          _minValue{refHeight}, _meanValue{refHeight}, _maxValue{refHeight},
          _epsgcode{epsg},
          _heightFunc{std::move(func)} {}

    double interpolateLonLat(double lon, double lat) const {
        if (_heightFunc) {
            return _heightFunc(lon, lat);
        }
        return static_cast<double>(_refHeight);
    }

    double refHeight() const { return _refHeight; }
    void refHeight(double h) {
        _refHeight = h;
        if (!_heightFunc) {
            _minValue = h;
            _meanValue = h;
            _maxValue = h;
        }
    }

    float meanHeight() const { return _meanValue; }
    float maxHeight() const { return _maxValue; }
    float minHeight() const { return _minValue; }

    int epsgCode() const { return _epsgcode; }
    void epsgCode(int epsgcode) { _epsgcode = epsgcode; }

    void setStats(float minVal, float meanVal, float maxVal) {
        _minValue = minVal;
        _meanValue = meanVal;
        _maxValue = maxVal;
    }

private:
    float _refHeight;
    float _minValue;
    float _meanValue;
    float _maxValue;
    int _epsgcode;
    HeightFunc _heightFunc;
};
