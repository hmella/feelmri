#include "Assemble.h"

namespace py = pybind11;

// Template type alias
template <typename T>
using IsoMap = std::tuple<
    Eigen::Matrix<T, Eigen::Dynamic, 1>, // S
    Eigen::Matrix<T, Eigen::Dynamic, 1>, // dSdx
    Eigen::Matrix<T, Eigen::Dynamic, 1>, // dSdy
    Eigen::Matrix<T, Eigen::Dynamic, 1>, // dSdz
    T>;                                  // detJ

// IsoMapping
template <typename T>
IsoMap<T> IsoMapping(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &qpoint,
    const py::object &finite_element)
{
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    const auto values = py::cast<Eigen::Tensor<T, 4>>(finite_element.attr("tabulate")(1, qpoint));
    const int nb_dofs = py::cast<int>(finite_element.attr("dimension"));

    Vec S(nb_dofs), dSdr(nb_dofs), dSds(nb_dofs), dSdt(nb_dofs);
    for (int i = 0; i < nb_dofs; i++)
    {
        S(i) = values(0, 0, i, 0);
        dSdr(i) = values(1, 0, i, 0);
        dSds(i) = values(2, 0, i, 0);
        dSdt(i) = values(3, 0, i, 0);
    }

    Mat3 J;
    J(0, 0) = (dSdr.transpose() * nodes.col(0)).sum();
    J(0, 1) = (dSdr.transpose() * nodes.col(1)).sum();
    J(0, 2) = (dSdr.transpose() * nodes.col(2)).sum();
    J(1, 0) = (dSds.transpose() * nodes.col(0)).sum();
    J(1, 1) = (dSds.transpose() * nodes.col(1)).sum();
    J(1, 2) = (dSds.transpose() * nodes.col(2)).sum();
    J(2, 0) = (dSdt.transpose() * nodes.col(0)).sum();
    J(2, 1) = (dSdt.transpose() * nodes.col(1)).sum();
    J(2, 2) = (dSdt.transpose() * nodes.col(2)).sum();

    const T detJ = std::abs(J.determinant());
    const Mat3 invJ = J.inverse();

    Vec dSdx = invJ(0, 0) * dSdr + invJ(0, 1) * dSds + invJ(0, 2) * dSdt;
    Vec dSdy = invJ(1, 0) * dSdr + invJ(1, 1) * dSds + invJ(1, 2) * dSdt;
    Vec dSdz = invJ(2, 0) * dSdr + invJ(2, 1) * dSds + invJ(2, 2) * dSdt;

    return std::make_tuple(S, dSdx, dSdy, dSdz, detJ);
}

// LocalMassAssemble
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> LocalMassAssemble(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &qpoints,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &qweights,
    const py::object &finite_element)
{
    const int nb_dofs = py::cast<int>(finite_element.attr("dimension"));
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Me(nb_dofs, nb_dofs);
    Me.setZero();

    for (int q = 0; q < qweights.size(); q++)
    {
        const auto [S, dSdx, dSdy, dSdz, detJ] = IsoMapping<T>(nodes, qpoints.row(q), finite_element);
        Me += (S * S.transpose()) * detJ * qweights(q);
    }
    return Me;
}

// MassAssemble
template <typename T>
Eigen::SparseMatrix<T> MassAssemble(
    const Eigen::MatrixXi &elems,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
    const py::object &finite_element,
    const py::object &quadrature_rule)
{
    using TripletType = Eigen::Triplet<T>;

    const int nb_elems = elems.rows();
    const int nb_nodes = nodes.rows();
    const int nb_nodes_e = elems.row(0).size();

    Eigen::SparseMatrix<T> M(nb_nodes, nb_nodes);
    std::vector<TripletType> coefficients;

    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> qpoints =
        py::cast<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(quadrature_rule.attr("points"));
    const Eigen::Matrix<T, Eigen::Dynamic, 1> qweights =
        py::cast<Eigen::Matrix<T, Eigen::Dynamic, 1>>(quadrature_rule.attr("weights"));

    for (int e = 0; e < nb_elems; e++)
    {
        Eigen::VectorXi elem = elems.row(e);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> elem_nodes(nb_nodes_e, 3);
        for (int i = 0; i < nb_nodes_e; ++i)
            elem_nodes.row(i) = nodes.row(elem(i));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Me =
            LocalMassAssemble<T>(elem_nodes, qpoints, qweights, finite_element);

        for (int i = 0; i < nb_nodes_e; i++)
            for (int j = 0; j < nb_nodes_e; j++)
                coefficients.emplace_back(elem(i), elem(j), Me(i, j));
    }

    M.setFromTriplets(coefficients.begin(), coefficients.end());
    return M;
}
