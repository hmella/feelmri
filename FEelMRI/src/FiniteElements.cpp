#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// using namespace Eigen;
namespace py = pybind11;

// Custom types
using ShapeFunctions = std::tuple<Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>>;
using QuadRule = std::tuple<Eigen::Matrix<float, 4, 3>, Eigen::Vector<float, 4>>;
using IsoMap = std::tuple<Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>,
  Eigen::Matrix<float, 4, 1>,
  float>;

// Triplets
using TripletType = Eigen::Triplet<float>;


// Quadrature rule
QuadRule QuadratureRule(){

  // Points
  Eigen::Matrix<float, 4, 3> x = (Eigen::Matrix<float, 4, 3>() <<
    0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
    0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
    0.1381966011250105, 0.1381966011250105, 0.5854101966249685,
    0.1381966011250105, 0.1381966011250105, 0.1381966011250105
  ).finished();

  // Weights
  Eigen::Vector<float, 4> w;
  w << 0.25/6.0, 0.25/6.0, 0.25/6.0, 0.25/6.0;

  return std::make_tuple(x, w);
}


// Computes the shape functions and their derivatives for a tetrahedral element
ShapeFunctions P1Tetra(float r, float s, float t){
  // Shape functions
  Eigen::Matrix<float, 4, 1> S;
  S << 1-r-s-t, r, s, t;

  // Derivatives
  Eigen::Matrix<float, 4, 1> dSdr, dSds, dSdt;
  dSdr << -1, 1, 0, 0;
  dSds << -1, 0, 1, 0;
  dSdt << -1, 0, 0, 1;

  return std::make_tuple(S, dSdr, dSds, dSdt);
}


// Isoparametric mapping
IsoMap IsoMapping(const Eigen::MatrixXf &nodes, const Eigen::VectorXf &qpoint){

  // Evaluate shape functions
  auto [S, dSdr, dSds, dSdt] = P1Tetra(qpoint(0), qpoint(1), qpoint(2));

  // Jacobian matrix
  Eigen::Matrix3f J;
  J(0, 0) = (dSdr.transpose() * nodes.col(0)).sum();
  J(0, 1) = (dSdr.transpose() * nodes.col(1)).sum();
  J(0, 2) = (dSdr.transpose() * nodes.col(2)).sum();
  J(1, 0) = (dSds.transpose() * nodes.col(0)).sum();
  J(1, 1) = (dSds.transpose() * nodes.col(1)).sum();
  J(1, 2) = (dSds.transpose() * nodes.col(2)).sum();
  J(2, 0) = (dSdt.transpose() * nodes.col(0)).sum();
  J(2, 1) = (dSdt.transpose() * nodes.col(1)).sum();
  J(2, 2) = (dSdt.transpose() * nodes.col(2)).sum();

  // Jacobian determinant
  const float detJ = J.determinant();

  // Jacobian inverse matrix
  const Eigen::Matrix3f invJ = J.inverse();

  // Shape functions in the global element
  const Eigen::Matrix<float, 4, 1> dSdx = invJ(0,0)*dSdr + invJ(0,1)*dSds + invJ(0,2)*dSdt;
  const Eigen::Matrix<float, 4, 1> dSdy = invJ(1,0)*dSdr + invJ(1,1)*dSds + invJ(1,2)*dSdt;
  const Eigen::Matrix<float, 4, 1> dSdz = invJ(2,0)*dSdr + invJ(2,1)*dSds + invJ(2,2)*dSdt;

  return std::make_tuple(S, dSdx, dSdy, dSdz, detJ);
}


// Assemble local mass matrix
Eigen::MatrixXf LocalMassAssemble(
  const Eigen::MatrixXf &nodes,
  const Eigen::MatrixXf &qpoints,
  const Eigen::VectorXf &qweights){

  // Assembled local matrix
  Eigen::Matrix4f Me = Eigen::Matrix4f::Zero();

  // Assembling
  for (int q = 0; q < qweights.size(); q++){ 

    // Isoparametric mapping
    const IsoMap isoMap = IsoMapping(nodes, qpoints.row(q));
    const auto& [S, dSdx, dSdy, dSdz, detJ] = isoMap;

    // Matrix assembling (should the determinant be adjusted?)
    Me += (S*S.transpose())*detJ*qweights(q);

  }

  return Me;
}


// Assemble mass matrix
Eigen::SparseMatrix<float> MassAssemble(const Eigen::MatrixXi &elems,
  const Eigen::MatrixXf &nodes){

  // Number of elements and nodes
  const int nb_elems = elems.rows();
  const int nb_nodes = nodes.rows();
  const int nb_nodes_e = elems.row(0).size();
  const int el_mat_size = nb_nodes_e*nb_nodes_e;

  // Quadrature rule
  const QuadRule quadRule = QuadratureRule();
  const Eigen::MatrixXf qpoints  = std::get<0>(quadRule);
  const Eigen::VectorXf qweights = std::get<1>(quadRule);

  // Global and local mass matrices
  Eigen::SparseMatrix<float> M(nb_nodes, nb_nodes);
  Eigen::MatrixXf Me = Eigen::MatrixXf::Zero(nb_nodes_e, nb_nodes_e);

  // Indices
  std::vector<TripletType> coefficients;

  // Element and nodes
  Eigen::VectorXi elem(el_mat_size);
  Eigen::MatrixXf elem_nodes(el_mat_size, 3);

  // Loop over elements for assembling
  for (int e = 0; e < nb_elems; e++){

    // Nodes in the element
    elem = elems(e, Eigen::indexing::all);
    elem_nodes = nodes(elem, Eigen::indexing::all);

    // Local assemble
    Me = LocalMassAssemble(elem_nodes, qpoints, qweights);

    // Fill triplets
    for (int i = 0; i < nb_nodes_e; i++){
      for (int j = 0; j < nb_nodes_e; j++){
        coefficients.push_back(TripletType(elem(i), elem(j), Me(i,j)));
      }
    }        
  }

  // Sparse matrix filling
  M.setFromTriplets(coefficients.begin(), coefficients.end());

  return M;
}


PYBIND11_MODULE(FiniteElements, m) {
    m.doc() = "Finite elements functions"; // optional module docstring
    m.def("MassAssemble", &MassAssemble, py::return_value_policy::reference);
}
