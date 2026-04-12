#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <complex>
#include <vector>
#include <string>
#include <omp.h>

#include <isce3/core/DateTime.h>
#include <isce3/core/TimeDelta.h>
#include <isce3/core/Vector.h>
#include <isce3/core/StateVector.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Kernels.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/container/RadarGeometry.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/focus/Backproject.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/error/ErrorCode.h>

namespace py = pybind11;
using namespace isce3;

PYBIND11_MODULE(backproject, m)
{
    m.doc() = "ISCE3 backprojection SAR focusing (migrated to ISCE2)";

    // --- DateTime ---
    py::class_<core::DateTime>(m, "DateTime")
        .def(py::init<int,int,int>(), py::arg("y"), py::arg("m"), py::arg("d"))
        .def(py::init<int,int,int,int,int,int>())
        .def(py::init<int,int,int,int,int,double>())
        .def(py::init<const std::string&>())
        .def_readwrite("year",    &core::DateTime::year)
        .def_readwrite("months",  &core::DateTime::months)
        .def_readwrite("days",    &core::DateTime::days)
        .def_readwrite("hours",   &core::DateTime::hours)
        .def_readwrite("minutes", &core::DateTime::minutes)
        .def_readwrite("seconds", &core::DateTime::seconds)
        .def_readwrite("frac",    &core::DateTime::frac)
        .def("__repr__", [](const core::DateTime& dt) {
            return std::string(dt);
        });

    // --- TimeDelta ---
    py::class_<core::TimeDelta>(m, "TimeDelta")
        .def(py::init<double>(), py::arg("seconds"))
        .def("getTotalSeconds", &core::TimeDelta::getTotalSeconds);

    // --- LookSide ---
    py::enum_<core::LookSide>(m, "LookSide")
        .value("Left",  core::LookSide::Left)
        .value("Right", core::LookSide::Right);

    // --- Vec3 ---
    py::class_<core::Vec3>(m, "Vec3")
        .def(py::init([](double x, double y, double z) {
            return core::Vec3{x, y, z};
        }), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("__getitem__", [](const core::Vec3& v, int i) { return v[i]; })
        .def("__setitem__", [](core::Vec3& v, int i, double val) { v[i] = val; });

    // --- StateVector ---
    py::class_<core::StateVector>(m, "StateVector")
        .def(py::init<>())
        .def_readwrite("datetime", &core::StateVector::datetime)
        .def_readwrite("position", &core::StateVector::position)
        .def_readwrite("velocity", &core::StateVector::velocity);

    // --- OrbitInterpMethod ---
    py::enum_<core::OrbitInterpMethod>(m, "OrbitInterpMethod")
        .value("Hermite",  core::OrbitInterpMethod::Hermite)
        .value("Legendre", core::OrbitInterpMethod::Legendre);

    // --- Orbit ---
    py::class_<core::Orbit>(m, "Orbit")
        .def(py::init<const std::vector<core::StateVector>&,
                       core::OrbitInterpMethod>(),
             py::arg("statevecs"),
             py::arg("interp_method") = core::OrbitInterpMethod::Hermite)
        .def("referenceEpoch",
             static_cast<const core::DateTime& (core::Orbit::*)() const>(
                 &core::Orbit::referenceEpoch))
        .def("size", &core::Orbit::size);

    // --- LUT2d<double> ---
    py::class_<core::LUT2d<double>>(m, "LUT2d")
        .def(py::init<const double&>(), py::arg("ref_value") = 0.0)
        .def("eval",
             static_cast<double (core::LUT2d<double>::*)(double, double) const>(
                 &core::LUT2d<double>::eval));

    // --- RadarGridParameters ---
    py::class_<product::RadarGridParameters>(m, "RadarGridParameters")
        .def(py::init<double, double, double, double, double,
                       core::LookSide, size_t, size_t, core::DateTime>(),
             py::arg("sensing_start"),
             py::arg("wavelength"),
             py::arg("prf"),
             py::arg("starting_range"),
             py::arg("range_pixel_spacing"),
             py::arg("look_side"),
             py::arg("length"),
             py::arg("width"),
             py::arg("ref_epoch"))
        .def("sensingStart",
             static_cast<double (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::sensingStart))
        .def("wavelength",
             static_cast<double (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::wavelength))
        .def("prf",
             static_cast<double (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::prf))
        .def("startingRange",
             static_cast<double (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::startingRange))
        .def("rangePixelSpacing",
             static_cast<double (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::rangePixelSpacing))
        .def("lookSide",          static_cast<core::LookSide (product::RadarGridParameters::*)() const>(&product::RadarGridParameters::lookSide))
        .def("length",
             static_cast<size_t (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::length))
        .def("width",
             static_cast<size_t (product::RadarGridParameters::*)() const>(
                 &product::RadarGridParameters::width));

    // --- RadarGeometry ---
    py::class_<container::RadarGeometry>(m, "RadarGeometry")
        .def(py::init<const product::RadarGridParameters&,
                       const core::Orbit&,
                       const core::LUT2d<double>&>(),
             py::arg("radar_grid"),
             py::arg("orbit"),
             py::arg("doppler"))
        .def("radarGrid", &container::RadarGeometry::radarGrid,
             py::return_value_policy::reference_internal)
        .def("orbit",     &container::RadarGeometry::orbit,
             py::return_value_policy::reference_internal)
        .def("doppler",   &container::RadarGeometry::doppler,
             py::return_value_policy::reference_internal)
        .def("gridLength", &container::RadarGeometry::gridLength)
        .def("gridWidth",  &container::RadarGeometry::gridWidth);

    // --- DEMInterpolator ---
    py::class_<geometry::DEMInterpolator>(m, "DEMInterpolator")
        .def(py::init<float, int>(),
             py::arg("height") = 0.0f,
             py::arg("epsg")   = 4326)
        .def(py::init([](py::function func, float ref_height, int epsg) {
            auto py_func = std::make_shared<py::function>(std::move(func));
            geometry::DEMInterpolator::HeightFunc wrapped =
                [py_func](double lon, double lat) -> double {
                    py::gil_scoped_acquire gil;
                    return (*py_func)(lon, lat).cast<double>();
                };
            return geometry::DEMInterpolator(std::move(wrapped), ref_height, epsg);
        }),
             py::arg("func"),
             py::arg("ref_height"),
             py::arg("epsg") = 4326)
        .def("interpolateLonLat", &geometry::DEMInterpolator::interpolateLonLat)
        .def("refHeight", static_cast<double (geometry::DEMInterpolator::*)() const>(&geometry::DEMInterpolator::refHeight))
        .def("epsgCode",  static_cast<int (geometry::DEMInterpolator::*)() const>(&geometry::DEMInterpolator::epsgCode))
        .def("setStats", &geometry::DEMInterpolator::setStats,
             py::arg("min_val"), py::arg("mean_val"), py::arg("max_val"))
        .def("meanHeight", &geometry::DEMInterpolator::meanHeight)
        .def("maxHeight", &geometry::DEMInterpolator::maxHeight)
        .def("minHeight", &geometry::DEMInterpolator::minHeight);

    // --- DryTroposphereModel ---
    py::enum_<focus::DryTroposphereModel>(m, "DryTroposphereModel")
        .value("NoDelay", focus::DryTroposphereModel::NoDelay)
        .value("TSX",     focus::DryTroposphereModel::TSX);

    // --- Kernel<float> (abstract base) ---
    py::class_<core::Kernel<float>>(m, "Kernel")
        .def("width", &core::Kernel<float>::width);

    // --- KnabKernel ---
    py::class_<core::KnabKernel<float>, core::Kernel<float>>(m, "KnabKernel")
        .def(py::init<double, double>(),
             py::arg("width"),
             py::arg("bandwidth"));

    // --- LinearKernel ---
    py::class_<core::LinearKernel<float>, core::Kernel<float>>(m, "LinearKernel")
        .def(py::init<>());

    // --- TabulatedKernel ---
    py::class_<core::TabulatedKernel<float>, core::Kernel<float>>(m, "TabulatedKernel")
        .def(py::init([](const core::Kernel<float>& k, int n) {
            return core::TabulatedKernel<float>(k, n);
        }), py::arg("kernel"), py::arg("n"));

    // --- ChebyKernel ---
    py::class_<core::ChebyKernel<float>, core::Kernel<float>>(m, "ChebyKernel")
        .def(py::init([](const core::Kernel<float>& k, int n) {
            return core::ChebyKernel<float>(k, n);
        }), py::arg("kernel"), py::arg("n"));

    // --- ErrorCode ---
    py::enum_<error::ErrorCode>(m, "ErrorCode")
        .value("Success",              error::ErrorCode::Success)
        .value("OrbitInterpSizeError", error::ErrorCode::OrbitInterpSizeError)
        .value("FailedToConverge",     error::ErrorCode::FailedToConverge)
        .value("WrongLookSide",        error::ErrorCode::WrongLookSide);

    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"));
    m.def("get_max_threads", []() { return omp_get_max_threads(); });
    m.def("get_num_procs", []() { return omp_get_num_procs(); });

    // --- backproject (numpy interface) ---
    m.def("backproject",
        [](py::array_t<std::complex<float>, py::array::c_style> in_data,
           const container::RadarGeometry& in_geometry,
           const container::RadarGeometry& out_geometry,
           const geometry::DEMInterpolator& dem,
           double fc, double ds,
           const core::Kernel<float>& kernel,
           focus::DryTroposphereModel dry_tropo_model,
           bool return_height)
        -> py::tuple
        {
            auto in_buf = in_data.request();
            size_t out_len = out_geometry.gridLength();
            size_t out_wid = out_geometry.gridWidth();

            auto out_data = py::array_t<std::complex<float>>({out_len, out_wid});
            auto out_buf = out_data.request();

            py::array_t<float> out_height;
            float* height_ptr = nullptr;
            if (return_height) {
                out_height = py::array_t<float>({out_len, out_wid});
                height_ptr = static_cast<float*>(out_height.request().ptr);
            }

            error::ErrorCode ec;
            {
                py::gil_scoped_release release;
                ec = focus::backproject(
                    static_cast<std::complex<float>*>(out_buf.ptr),
                    out_geometry,
                    static_cast<const std::complex<float>*>(in_buf.ptr),
                    in_geometry,
                    dem, fc, ds, kernel, dry_tropo_model,
                    {}, {},
                    height_ptr
                );
            }

            if (return_height) {
                return py::make_tuple(out_data, out_height, py::cast(ec));
            }
            return py::make_tuple(out_data, py::none(), py::cast(ec));
        },
        py::arg("in_data"),
        py::arg("in_geometry"),
        py::arg("out_geometry"),
        py::arg("dem"),
        py::arg("fc"),
        py::arg("ds"),
        py::arg("kernel"),
        py::arg("dry_tropo_model") = focus::DryTroposphereModel::TSX,
        py::arg("return_height") = false,
        R"doc(
Time-domain backprojection SAR focusing.

Parameters
----------
in_data : np.ndarray[complex64]
    Range-compressed input signal (2D, azimuth x range).
in_geometry : RadarGeometry
    Input data geometry (grid, orbit, doppler).
out_geometry : RadarGeometry
    Output focused grid geometry.
dem : DEMInterpolator
    DEM for terrain height.
fc : float
    Center frequency in Hz.
ds : float
    Desired azimuth resolution in meters.
kernel : Kernel
    Range interpolation kernel (e.g. KnabKernel).
dry_tropo_model : DryTroposphereModel
    Tropospheric delay model (default: TSX).
return_height : bool
    If True, also return per-pixel height array.

Returns
-------
tuple of (out_data, height, error_code)
    out_data: np.ndarray[complex64] focused output
    height: np.ndarray[float32] or None
    error_code: ErrorCode
)doc"
    );
}
