#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <string>
#include <unordered_map>
#include <stdexcept>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>


// -----------------------------------------------------------------------------
// 1. FE LOOKUP UTILITIES
// -----------------------------------------------------------------------------
struct FEInfo {
    basix::cell::type cell;
    int degree;
    basix::element::family family;
};

static const std::unordered_map<std::string, FEInfo> fe_from_meshio = {
    {"triangle",   {basix::cell::type::triangle,     1, basix::element::family::P}},
    {"tetra",      {basix::cell::type::tetrahedron,  1, basix::element::family::P}},
    {"tetra10",    {basix::cell::type::tetrahedron,  2, basix::element::family::P}},
    {"wedge",      {basix::cell::type::prism,        1, basix::element::family::P}},
    {"hexahedron", {basix::cell::type::hexahedron,   1, basix::element::family::P}},
};

inline const FEInfo& get_fe_info(const std::string& meshio_type) {
    auto it = fe_from_meshio.find(meshio_type);
    if (it == fe_from_meshio.end()) throw std::runtime_error("Unsupported element type: " + meshio_type);
    return it->second;
}

template <typename T, std::size_t d>
using mdspan_t = basix::md::mdspan<T, basix::md::dextents<std::size_t, d>>;

// -----------------------------------------------------------------------------
// 2. QUADRATURE CACHE STRUCT & BUILDER
// -----------------------------------------------------------------------------
template <typename T>
struct FEQuadratureCache
{
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 3>> xq;      
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> wq;      
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Sq; 
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SqT; 
};

template <typename T>
FEQuadratureCache<T> BuildFEQuadratureCache(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const std::string& meshio_type,
    int quadrature_degree)
{
    FEQuadratureCache<T> cache;
    const int nelem = elems.rows();
    const int nn_e  = elems.cols();

    const auto fe_info = get_fe_info(meshio_type);
    auto fe = basix::create_element<T>(
        fe_info.family, fe_info.cell, fe_info.degree,
        basix::element::lagrange_variant::equispaced,
        basix::element::dpc_variant::unset, false);

    auto qw = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::Default, fe_info.cell,
        basix::polyset::type::standard, quadrature_degree);

    const std::vector<T>& qpts_flat = qw[0];
    const std::vector<T>& qweights  = qw[1];
    const int nq = static_cast<int>(qweights.size());
    const int ndofs_e = fe.dim();
    const std::size_t gdim = 3;

    auto [tab_data, tab_shape] = fe.tabulate(1, std::span<const T>(qpts_flat.data(), qpts_flat.size()), {static_cast<std::size_t>(nq), gdim});
    mdspan_t<const T, 4> tab(tab_data.data(), tab_shape);

    cache.xq.resize(nelem);
    cache.wq.resize(nelem);
    cache.Sq.resize(nelem);
    cache.SqT.resize(nelem);

    Eigen::Matrix<T, Eigen::Dynamic, 3> elem_nodes(nn_e, 3);

    for (int e = 0; e < nelem; ++e)
    {
        for (int i = 0; i < nn_e; ++i) elem_nodes.row(i) = nodes.row(elems(e, i));

        cache.xq[e].resize(nq, 3);
        cache.wq[e].resize(nq);
        cache.Sq[e].resize(ndofs_e, nq);
        cache.SqT[e].resize(nq, ndofs_e);

        for (int q = 0; q < nq; ++q)
        {
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> S   (&tab(0, q, 0, 0), ndofs_e);
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> dSdr(&tab(1, q, 0, 0), ndofs_e);
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> dSds(&tab(2, q, 0, 0), ndofs_e);
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> dSdt(&tab(3, q, 0, 0), ndofs_e);

            cache.Sq[e].col(q) = S;
            cache.xq[e].row(q).noalias() = S.transpose() * elem_nodes;

            Eigen::Matrix<T, 3, 3> J;
            J.row(0) = dSdr.transpose() * elem_nodes;
            J.row(1) = dSds.transpose() * elem_nodes;
            J.row(2) = dSdt.transpose() * elem_nodes;

            cache.wq[e](q) = std::abs(J.determinant()) * qweights[q];
        }
        cache.SqT[e].noalias() = cache.Sq[e].transpose();
    }
    return cache;
}