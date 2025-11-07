#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

namespace py = pybind11;

// MassAssemble
template <typename T>
Eigen::SparseMatrix<T> MassAssemble(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const py::object &finite_element,
    const py::object &quadrature_rule
);


// PYBIND11 module
PYBIND11_MODULE(Assemble, m)
{
    m.doc() = "Templated finite element assembly functions";

    m.def("MassAssemble", py::overload_cast<
        const Eigen::MatrixXi &,
        const Eigen::MatrixXf &,
        const py::object &,
        const py::object &
    >(&MassAssemble<float>));

    m.def("MassAssemble", py::overload_cast<
        const Eigen::MatrixXi &,
        const Eigen::MatrixXd &,
        const py::object &,
        const py::object &
    >(&MassAssemble<double>));
}
