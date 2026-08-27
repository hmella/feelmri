/**
 * @file FEUtils.h
 * @brief Shared finite element lookup tables and per-element quadrature cache.
 *
 * Provides:
 * - @ref FEInfo / @ref fe_from_meshio : map meshio element-type strings to
 *   Basix cell types, polynomial degrees, and element families.
 * - @ref FEQuadratureCache : per-element pre-tabulated basis functions,
 *   physical quadrature points, and Jacobian-weighted quadrature weights.
 * - @ref BuildFEQuadratureCache : factory that populates the cache.
 */
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
#include <array>
#include <vector>
#include <mutex>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>


// -----------------------------------------------------------------------------
// 1. FE LOOKUP UTILITIES
// -----------------------------------------------------------------------------

/// Basix element descriptor associated with a meshio element-type string.
struct FEInfo {
    basix::cell::type      cell;    ///< Basix cell type (e.g. tetrahedron).
    int                    degree;  ///< Polynomial degree of the Lagrange basis.
    basix::element::family family;  ///< Element family (Lagrange ``P``).
};

/// Maps meshio element-type strings to their Basix FEInfo descriptors.
/// Supported keys: ``"triangle"``, ``"tetra"``, ``"tetra10"``, ``"wedge"``, ``"hexahedron"``.
static const std::unordered_map<std::string, FEInfo> fe_from_meshio = {
    {"triangle",   {basix::cell::type::triangle,     1, basix::element::family::P}},
    {"tetra",      {basix::cell::type::tetrahedron,  1, basix::element::family::P}},
    {"tetra10",    {basix::cell::type::tetrahedron,  2, basix::element::family::P}},
    {"wedge",      {basix::cell::type::prism,        1, basix::element::family::P}},
    {"hexahedron", {basix::cell::type::hexahedron,   1, basix::element::family::P}},
};

/**
 * @brief Look up the Basix FEInfo for a meshio element-type string.
 *
 * @param meshio_type Meshio element-type string (e.g. ``"tetra"``).
 * @return            Const reference to the corresponding @ref FEInfo entry.
 * @throws std::runtime_error If @p meshio_type is not in @ref fe_from_meshio.
 */
inline const FEInfo& get_fe_info(const std::string& meshio_type) {
    auto it = fe_from_meshio.find(meshio_type);
    if (it == fe_from_meshio.end()) throw std::runtime_error("Unsupported element type: " + meshio_type);
    return it->second;
}

// -----------------------------------------------------------------------------
// 1b. MESHIO -> BASIX DOF ORDERING
// -----------------------------------------------------------------------------

/**
 * @brief Reference-cell coordinates of each element node, in meshio/VTK order.
 *
 * meshio and Basix both list vertex DOFs before edge/face DOFs, but order them
 * differently within those blocks: VTK walks the bottom face of a hexahedron
 * cyclically where Basix uses a tensor-product lattice, and the two number
 * tetrahedron edges differently.
 *
 * The permutation is derived from these coordinates at run time (see
 * @ref meshio_to_basix_permutation) rather than hard-coded as indices, so it remains
 * valid if Basix changes its DOF numbering; supporting a new cell type requires only
 * adding its reference coordinates here. Components beyond the cell's topological
 * dimension are ignored.
 */
static const std::unordered_map<std::string, std::vector<std::array<double, 3>>>
meshio_reference_points = {
    {"triangle",   {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}},
    {"tetra",      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
    // 4 vertices, then edge midpoints (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
    {"tetra10",    {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                    {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0},
                    {0, 0, 0.5}, {0.5, 0, 0.5}, {0, 0.5, 0.5}}},
    // bottom face walked cyclically, then the top face
    {"hexahedron", {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}},
    {"wedge",      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}},
};

/**
 * @brief Permutation taking meshio DOF order to Basix DOF order.
 *
 * ``perm[j]`` is the meshio-order index of the node that Basix expects at its own
 * DOF ``j``, i.e. ``elems_basix.col(j) == elems_meshio.col(perm[j])``.
 *
 * Each Basix interpolation point is matched against @ref meshio_reference_points by
 * position. Reference nodes are separated by far more than @c tol, so a match within
 * @c tol is unique; the bijectivity check guards against a malformed table.
 *
 * @param meshio_type Meshio element-type string (see @ref fe_from_meshio).
 * @return            Const reference to the cached permutation, length ``ndofs``.
 * @throws std::runtime_error If the type is unregistered, the DOF counts disagree,
 *                            or any Basix point fails to match a reference node.
 */
inline const std::vector<int>& meshio_to_basix_permutation(const std::string& meshio_type)
{
    static std::unordered_map<std::string, std::vector<int>> cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);

    if (auto it = cache.find(meshio_type); it != cache.end()) return it->second;

    auto ref_it = meshio_reference_points.find(meshio_type);
    if (ref_it == meshio_reference_points.end())
        throw std::runtime_error(
            "No meshio reference coordinates registered for element type: " + meshio_type);
    const std::vector<std::array<double, 3>>& ref = ref_it->second;

    const FEInfo& fe_info = get_fe_info(meshio_type);
    auto fe = basix::create_element<double>(
        fe_info.family, fe_info.cell, fe_info.degree,
        basix::element::lagrange_variant::equispaced,
        basix::element::dpc_variant::unset, false);

    const auto& [pts, shape] = fe.points();
    const std::size_t ndofs = shape[0];
    const std::size_t tdim  = shape[1];

    if (ndofs != ref.size())
        throw std::runtime_error(
            "Element type '" + meshio_type + "': Basix reports " + std::to_string(ndofs)
            + " DOFs but the meshio reference table lists " + std::to_string(ref.size()));

    constexpr double tol = 1e-10;
    std::vector<int>  perm(ndofs, -1);
    std::vector<bool> used(ndofs, false);

    for (std::size_t j = 0; j < ndofs; ++j)          // Basix DOF index
    {
        int match = -1;
        for (std::size_t i = 0; i < ndofs; ++i)      // meshio node index
        {
            double d2 = 0.0;
            for (std::size_t k = 0; k < tdim; ++k)
            {
                const double d = ref[i][k] - static_cast<double>(pts[j * tdim + k]);
                d2 += d * d;
            }
            if (d2 <= tol * tol) { match = static_cast<int>(i); break; }
        }
        if (match < 0)
            throw std::runtime_error(
                "Element type '" + meshio_type + "': Basix DOF " + std::to_string(j)
                + " matches no node in the meshio reference table");
        if (used[match])
            throw std::runtime_error(
                "Element type '" + meshio_type + "': meshio node " + std::to_string(match)
                + " matched by more than one Basix DOF (duplicate reference coordinates)");
        used[match] = true;
        perm[j] = match;
    }

    return cache.emplace(meshio_type, std::move(perm)).first->second;
}

/**
 * @brief Reorder element connectivity columns from meshio DOF order to Basix order.
 *
 * Apply once, before any Basix tabulation, and use the returned connectivity for the
 * global assembly indices as well, so the geometry map and the node numbering agree.
 * Python-side connectivity stays in meshio order, as XDMF output expects.
 *
 * @param elems       Connectivity in meshio order, shape ``(n_elem, n_nodes_per_elem)``.
 * @param meshio_type Meshio element-type string.
 * @return            Connectivity in Basix order (the input itself when they agree).
 */
inline Eigen::MatrixXi permute_meshio_to_basix(const Eigen::MatrixXi& elems,
                                               const std::string& meshio_type)
{
    const std::vector<int>& perm = meshio_to_basix_permutation(meshio_type);

    if (static_cast<std::size_t>(elems.cols()) != perm.size())
        throw std::runtime_error(
            "Connectivity has " + std::to_string(elems.cols()) + " nodes per element but '"
            + meshio_type + "' expects " + std::to_string(perm.size()));

    bool identity = true;
    for (std::size_t j = 0; j < perm.size(); ++j)
        if (perm[j] != static_cast<int>(j)) { identity = false; break; }
    if (identity) return elems;

    Eigen::MatrixXi out(elems.rows(), elems.cols());
    for (std::size_t j = 0; j < perm.size(); ++j)
        out.col(static_cast<Eigen::Index>(j)) = elems.col(perm[j]);
    return out;
}

template <typename T, std::size_t d>
using mdspan_t = basix::md::mdspan<T, basix::md::dextents<std::size_t, d>>;

// -----------------------------------------------------------------------------
// 2. QUADRATURE CACHE STRUCT & BUILDER
// -----------------------------------------------------------------------------

/**
 * @brief Per-element cache of pre-tabulated FE basis functions and quadrature data.
 *
 * All vectors are indexed by element index ``e``.  Pre-computing and storing
 * these quantities avoids repeated Basix calls during signal assembly.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 */
template <typename T>
struct FEQuadratureCache
{
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 3>> xq;     ///< Physical quadrature points per element, each ``(nq, 3)`` (m).
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> wq;     ///< Jacobian-weighted quadrature weights per element, each ``(nq,)``.
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Sq;  ///< Basis-function matrix per element, shape ``(n_dofs, nq)``.
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> SqT; ///< Transpose of @ref Sq, shape ``(nq, n_dofs)``, row-major for fast row access.
};

/**
 * @brief Build a @ref FEQuadratureCache by pre-tabulating basis functions and weights.
 *
 * For each element, maps the reference quadrature points to physical space,
 * evaluates the Lagrange basis functions, and stores the Jacobian-scaled
 * quadrature weights.  The resulting cache can be used for repeated signal
 * assembly without incurring Basix overhead at runtime.
 *
 * @tparam T Floating-point scalar type (``float`` or ``double``).
 *
 * @param elems             Element connectivity, shape ``(n_elem, n_nodes_per_elem)``,
 *                          already in **Basix** DOF order (see @ref permute_meshio_to_basix).
 * @param nodes             Node coordinates, shape ``(n_nodes, 3)`` (m).
 * @param meshio_type       Meshio element-type string (see @ref fe_from_meshio).
 * @param quadrature_degree Polynomial degree of exactness for the quadrature rule.
 *
 * @return Populated @ref FEQuadratureCache<T>.
 */
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