#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

template<typename T>
py::array_t<T> tensordot_modes_weights(
    const py::array_t<T, py::array::c_style> modes,   // (P,C,M), C-contiguous, dtype T
    const py::array_t<T, py::array::c_style> weights  // (M,),   C-contiguous, dtype T
);


PYBIND11_MODULE(POD, m) {
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<float>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float32)");
  m.def("tensordot_modes_weights",
        &tensordot_modes_weights<double>,
        py::arg("modes").noconvert(), py::arg("weights").noconvert(),
        "Compute (P,C,M)·(M,) -> (P,C) (float64)");        
}