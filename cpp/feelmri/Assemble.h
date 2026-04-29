/**
 * @file Assemble.h
 * @brief Finite element global mass matrix assembly exposed to Python via pybind11.
 *
 * Provides two assembly routines:
 * - @ref MassAssemble   : uses legacy pybind11 FE / quadrature-rule objects.
 * - @ref basixMassAssemble : optimised path using the Basix library with
 *   pre-tabulated basis functions and optional OpenMP parallelism.
 */
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

/**
 * @brief Assemble the global sparse mass matrix using Python FE/quadrature objects.
 *
 * Loops over elements, evaluates basis functions and Jacobians via isoparametric
 * mapping, integrates the local mass matrix at each quadrature point, and
 * assembles into the global sparse matrix using triplet insertion.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 *
 * @param elems          Element connectivity array, shape ``(n_elem, n_nodes_per_elem)``.
 * @param nodes          Node coordinates, shape ``(n_nodes, 3)`` (m).
 * @param finite_element Python FE object providing basis-function evaluation.
 * @param quadrature_rule Python quadrature object providing points and weights.
 *
 * @return Sparse mass matrix of shape ``(n_nodes, n_nodes)``.
 */
template <typename T>
Eigen::SparseMatrix<T> MassAssemble(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const py::object &finite_element,
    const py::object &quadrature_rule
);

/**
 * @brief Assemble the global sparse mass matrix using the Basix library (optimised).
 *
 * Pre-tabulates basis functions at quadrature points once per mesh-type, then
 * assembles with thread-local buffers (OpenMP) and pre-allocated triplet
 * vectors to avoid memory reallocation.  Prefer this over @ref MassAssemble
 * for production use.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 *
 * @param elems              Element connectivity, shape ``(n_elem, n_nodes_per_elem)``.
 * @param nodes              Node coordinates, shape ``(n_nodes, 3)`` (m).
 * @param meshio_type        Meshio element-type string.
 *                           Supported: ``"triangle"``, ``"tetra"``, ``"tetra10"``,
 *                           ``"wedge"``, ``"hexahedron"``.
 * @param quadrature_variant Basix quadrature-variant string (e.g. ``"default"``).
 * @param quadrature_rule    Basix quadrature-rule string (e.g. ``"default"``).
 * @param quadrature_degree  Polynomial degree of exactness for the quadrature rule.
 *
 * @return Sparse mass matrix of shape ``(n_nodes, n_nodes)``.
 */
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
