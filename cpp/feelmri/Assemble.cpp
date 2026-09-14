#include "Assemble.h"
#include "FEUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <span>

namespace py = pybind11;


template <typename T, std::size_t d>
using mdspan_t = basix::md::mdspan<T, basix::md::dextents<std::size_t, d>>;

// -----------------------------------------------------------------------------
// Local mass assembly using *pre-tabulated* basis values/derivatives
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Optimized Local mass assembly
// -----------------------------------------------------------------------------
template <typename T>
static inline void basixLocalMassAssemble(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Me,       // Passed by reference (no alloc)
    const Eigen::Matrix<T, Eigen::Dynamic, 3>& elem_nodes,      // Forced 3 columns (no alloc)
    const std::vector<T>& wts,
    const mdspan_t<const T, 4>& tab,
    int nb_dofs,
    int nq)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    Me.setZero(); // Reset thread-local buffer

    for (int q = 0; q < nq; ++q)
    {
        // Zero-copy mapping directly from Basix tabulation. 
        // Assumes C-style contiguous memory layout where 'dofs' vary fastest after 'value_size' (1).
        Eigen::Map<const Vec> S   (&tab(0, q, 0, 0), nb_dofs);
        Eigen::Map<const Vec> dSdr(&tab(1, q, 0, 0), nb_dofs);
        Eigen::Map<const Vec> dSds(&tab(2, q, 0, 0), nb_dofs);
        Eigen::Map<const Vec> dSdt(&tab(3, q, 0, 0), nb_dofs);

        // Build Jacobian
        Mat3 J;
        J.row(0) = dSdr.transpose() * elem_nodes;
        J.row(1) = dSds.transpose() * elem_nodes;
        J.row(2) = dSdt.transpose() * elem_nodes;

        const T detJ = std::abs(J.determinant());
        const T c = detJ * wts[q];

        // Rank-1 update. .noalias() prevents Eigen from creating a temporary matrix
        Me.noalias() += c * (S * S.transpose());
    }
}

// -----------------------------------------------------------------------------
// Optimized Global Mass assemble
// -----------------------------------------------------------------------------
template <typename T>
Eigen::SparseMatrix<T> basixMassAssemble(
    const Eigen::MatrixXi& elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& nodes,
    const std::string& meshio_type,
    const std::string& quadrature_variant,
    const std::string& quadrature_rule,
    const int quadrature_degree)
{
    const int nb_elems   = elems.rows();
    const int nb_nodes   = nodes.rows();
    const int nb_nodes_e = elems.cols();

    // Reorder to Basix DOF order; used for both element geometry and triplet indices.
    const Eigen::MatrixXi elems_b = permute_meshio_to_basix(elems, meshio_type);

    // Basix setup
    const auto fe_info = get_fe_info(meshio_type);
    const auto variant = basix::element::lagrange_variant::equispaced;

    const basix::FiniteElement<T> finite_element =
        basix::create_element<T>(
            fe_info.family, fe_info.cell, fe_info.degree, variant,
            basix::element::dpc_variant::unset, false);

    const int nb_dofs = finite_element.dim();

    auto qw = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::Default, fe_info.cell,
        basix::polyset::type::standard, quadrature_degree);

    const std::vector<T>& qpts_flat = qw[0];
    const std::vector<T>& wts       = qw[1];

    const std::size_t gdim = 3;
    const int nq = static_cast<int>(wts.size());

    auto [tab_data, tab_shape] =
        finite_element.tabulate(
            1, std::span<const T>(qpts_flat.data(), qpts_flat.size()),
            {static_cast<std::size_t>(nq), gdim});

    mdspan_t<const T, 4> tab(tab_data.data(), tab_shape);

    Eigen::SparseMatrix<T> M(nb_nodes, nb_nodes);
    using TripletType = Eigen::Triplet<T>;

    // Pre-allocate the EXACT size required to avoid resizing and allow threaded assignment
    const std::size_t total_triplets = static_cast<std::size_t>(nb_elems) * nb_nodes_e * nb_nodes_e;
    std::vector<TripletType> coefficients(total_triplets);

    // Assembly loop
    Eigen::Matrix<T, Eigen::Dynamic, 3> elem_nodes(nb_nodes_e, 3);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Me(nb_dofs, nb_dofs);

    for (int e = 0; e < nb_elems; ++e)
    {
        const auto elem = elems_b.row(e);

        for (int i = 0; i < nb_nodes_e; ++i)
            elem_nodes.row(i) = nodes.row(elem(i));

        // Populate thread-local Me
        basixLocalMassAssemble<T>(Me, elem_nodes, wts, tab, nb_dofs, nq);

        // Compute precise offset for lock-free parallel insertion
        const std::size_t offset = static_cast<std::size_t>(e) * nb_nodes_e * nb_nodes_e;
        std::size_t idx = 0;

        for (int i = 0; i < nb_nodes_e; ++i)
        {
            for (int j = 0; j < nb_nodes_e; ++j)
            {
                coefficients[offset + idx++] = TripletType(elem(i), elem(j), Me(i, j));
            }
        }
    }

    // Build sparse matrix
    M.setFromTriplets(coefficients.begin(), coefficients.end());
    return M;
}