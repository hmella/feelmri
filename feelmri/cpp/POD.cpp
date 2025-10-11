#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

template<typename T>
py::array_t<T> tensordot_modes_weights(
    const py::array_t<T, py::array::c_style> modes,   // (P,C,M), C-contiguous, dtype T
    const py::array_t<T, py::array::c_style> weights) // (M,),   C-contiguous, dtype T
{
    using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vec    = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    // ---- shapes (no copies) ----
    const Eigen::Index P = static_cast<Eigen::Index>(modes.shape(0));
    const Eigen::Index C = static_cast<Eigen::Index>(modes.shape(1));
    const Eigen::Index M = static_cast<Eigen::Index>(modes.shape(2));

    // ---- raw pointers (no temporaries) ----
    const T* A_ptr = modes.data();   // points into the NumPy buffer
    const T* w_ptr = weights.data(); // points into the NumPy buffer

    // Map modes as (P*C × M) row-major without copying
    Eigen::Map<const RowMat> A(A_ptr, P*C, M);
    Eigen::Map<const Vec>    w(w_ptr, M);

    // Allocate output (P,C) NumPy array; we will write directly into its buffer
    py::array_t<T> out({P, C});
    T* out_ptr = out.mutable_data();

    // Release the GIL while doing the GEMV
    py::gil_scoped_release nogil;

    // Write the result directly to the output buffer as a vector view (P*C,)
    Eigen::Map<Vec> y(out_ptr, P*C);
    y.noalias() = A * w;            // no temporaries created

    return out; // (P,C) view over the same buffer
}


PYBIND11_MODULE(POD, m) {
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<float>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float32)");
}