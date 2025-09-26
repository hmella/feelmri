#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

// Horner's method: vector version
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1> faster_polyval(
    const Eigen::Array<T, Eigen::Dynamic, 1>& x,
    const Eigen::Array<T, Eigen::Dynamic, 1>& p)
{
    Eigen::Array<T, Eigen::Dynamic, 1> y = Eigen::Array<T, Eigen::Dynamic, 1>::Zero(x.size());
    for (int i = 0; i < p.size(); ++i) {
        y = y * x + p[i];
    }
    return y;
}

// Horner's method: scalar version
template <typename T>
T faster_polyval(
    const T x,
    const Eigen::Array<T, Eigen::Dynamic, 1>& p)
{
    T y = 0;
    for (int i = 0; i < p.size(); ++i) {
        y = y * x + p[i];
    }
    return y;
}


PYBIND11_MODULE(MathCpp, m) {
    m.def(
        "faster_polyval",
        py::overload_cast<const Eigen::ArrayXf&, const Eigen::ArrayXf&>(&faster_polyval<float>),
        "Evaluate polynomial at array x (float)"
    );
    // m.def(
    //     "faster_polyval",
    //     py::overload_cast<const Eigen::ArrayXd&, const Eigen::ArrayXd&>(&faster_polyval<double>),
    //     "Evaluate polynomial at array x (double)"
    // );
    m.def(
        "faster_polyval",
        py::overload_cast<const float, const Eigen::ArrayXf&>(&faster_polyval<float>),
        "Evaluate polynomial at scalar x (float)"
    );
    // m.def(
    //     "faster_polyval",
    //     py::overload_cast<const double, const Eigen::ArrayXd&>(&faster_polyval<double>),
    //     "Evaluate polynomial at scalar x (double)"
    // );
}