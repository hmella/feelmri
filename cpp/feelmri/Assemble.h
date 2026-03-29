#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/polyset.h>

namespace py = pybind11;

// MassAssemble
template <typename T>
Eigen::SparseMatrix<T> MassAssemble(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const py::object &finite_element,
    const py::object &quadrature_rule
);

// MassAssemble
template <typename T>
Eigen::SparseMatrix<T> basixMassAssemble(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const std::string &meshio_type,
    const std::string &quadrature_variant,
    const std::string &quadrature_rule,
    const int quadrature_degree
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

    m.def("basixMassAssemble", py::overload_cast<
        const Eigen::MatrixXi &,
        const Eigen::MatrixXf &,
        const std::string &,
        const std::string &,
        const std::string &,
        const int
    >(&basixMassAssemble<float>));

    m.def("basixMassAssemble", py::overload_cast<
        const Eigen::MatrixXi &,
        const Eigen::MatrixXd &,
        const std::string &,
        const std::string &,
        const std::string &,
        const int
    >(&basixMassAssemble<double>));
}
