#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

template<typename T>
py::array_t<T> tensordot_modes_weights(
    const py::array_t<T, py::array::c_style> modes,  // (P,C,M)
    const py::array_t<T, py::array::c_style> weights // (M,)
){
    namespace py = pybind11;
    using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vec    = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const Eigen::Index P = static_cast<Eigen::Index>(modes.shape(0));
    const Eigen::Index C = static_cast<Eigen::Index>(modes.shape(1));
    const Eigen::Index M = static_cast<Eigen::Index>(modes.shape(2));

    const T* A_ptr = modes.data();
    const T* w_ptr = weights.data();

    // Map A as (P*C × M) row-major; w as (M)
    Eigen::Map<const RowMat> A(A_ptr, P*C, M);
    Eigen::Map<const Vec>    w(w_ptr, M);

    // Allocate output (P, C) row-major
    py::array_t<T> out({P, C});
    T* out_ptr = out.mutable_data();
    Eigen::Map<RowMat> O(out_ptr, P, C);

    // Release the GIL during compute
    py::gil_scoped_release nogil;

    // Portable (Eigen 3.3+): write result directly into output buffer as a vector
    Eigen::Map<Vec> y(out_ptr, P*C);
    y.noalias() = A * w;

    return out;
}


PYBIND11_MODULE(POD, m) {
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<float>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float32)");
}