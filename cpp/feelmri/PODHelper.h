/**
 * @file PODHelper.h
 * @brief POD mode-weight tensor contraction exposed to Python via pybind11.
 */
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

/**
 * @brief Contract a POD mode tensor with a weight vector: ``(P,C,M) · (M,) → (P,C)``.
 *
 * Reshapes the C-contiguous @p modes array to a ``(P*C, M)`` matrix, then
 * performs a GIL-released Eigen GEMV against @p weights.  The result is
 * written directly into a freshly allocated ``(P,C)`` NumPy array.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 * @param modes   C-contiguous NumPy array of shape ``(P, C, M)`` and dtype @p T.
 * @param weights C-contiguous NumPy array of shape ``(M,)`` and dtype @p T.
 * @return        NumPy array of shape ``(P, C)`` containing the contraction result.
 */
template<typename T>
py::array_t<T> tensordot_modes_weights(
    const py::array_t<T, py::array::c_style> modes,   // (P,C,M), C-contiguous, dtype T
    const py::array_t<T, py::array::c_style> weights  // (M,),   C-contiguous, dtype T
);


PYBIND11_MODULE(PODHelper, m) {
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<float>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float32)");
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<double>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float64)");        
}