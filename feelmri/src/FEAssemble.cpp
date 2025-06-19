#include <pybind11/pybind11.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

// using namespace Eigen;
namespace py = pybind11;

using IsoMap = std::tuple<Eigen::VectorXf,
  Eigen::VectorXf,
  Eigen::VectorXf,
  Eigen::VectorXf,
  float>;

// Triplets
using TripletType = Eigen::Triplet<float>;


// Isoparametric mapping
IsoMap IsoMapping(
  const Eigen::MatrixXf &nodes, 
  const Eigen::MatrixXf &qpoint, 
  const py::object &finite_element){

  // Evaluate shape functions
  const auto values = py::cast<Eigen::Tensor<float, 4>>(finite_element.attr("tabulate")(1, qpoint));

  // Number of degrees of freedom
  const auto nb_dofs = py::cast<int>(finite_element.attr("dimension"));

  // Create shape functions
  Eigen::VectorXf S(nb_dofs), dSdr(nb_dofs), dSds(nb_dofs), dSdt(nb_dofs);
  for (int i=0; i<nb_dofs; i++){
    S(i) = values(0,0,i,0); 
    dSdr(i) = values(1,0,i,0); 
    dSds(i) = values(2,0,i,0); 
    dSdt(i) = values(3,0,i,0); 
  }

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
  Eigen::VectorXf dSdx(nb_dofs), dSdy(nb_dofs), dSdz(nb_dofs);
  dSdx = invJ(0,0)*dSdr + invJ(0,1)*dSds + invJ(0,2)*dSdt;
  dSdy = invJ(1,0)*dSdr + invJ(1,1)*dSds + invJ(1,2)*dSdt;
  dSdz = invJ(2,0)*dSdr + invJ(2,1)*dSds + invJ(2,2)*dSdt;

  return std::make_tuple(S, dSdx, dSdy, dSdz, detJ);
}


// Assemble local mass matrix
Eigen::MatrixXf LocalMassAssemble(
  const Eigen::MatrixXf &nodes,
  const Eigen::MatrixXf &qpoints,
  const Eigen::VectorXf &qweights,
  const py::object &finite_element){

  // Number of degrees of freedom
  const auto nb_dofs = py::cast<int>(finite_element.attr("dimension"));

  // Assembled local matrix
  Eigen::MatrixXf Me(nb_dofs, nb_dofs);
  Me.setZero();

  // Assembling
  for (int q = 0; q < qweights.size(); q++){ 

    // Isoparametric mapping
    const IsoMap tmp = IsoMapping(nodes, qpoints.row(q), finite_element);
    const auto& [S, dSdx, dSdy, dSdz, detJ] = tmp;

    // Matrix assembling (should the determinant be adjusted?)
    Me += (S*S.transpose())*detJ*qweights(q);

  }

  return Me;
}


// Assemble mass matrix
Eigen::SparseMatrix<float> MassAssemble(
  const Eigen::MatrixXi &elems,
  const Eigen::MatrixXf &nodes,
  const py::object &finite_element,
  const py::object &quadrature_rule){

  // Number of elements and nodes
  const int nb_elems = elems.rows();
  const int nb_nodes = nodes.rows();
  const int nb_nodes_e = elems.row(0).size();
  const int el_mat_size = nb_nodes_e*nb_nodes_e;

  // Global and local mass matrices
  Eigen::SparseMatrix<float> M(nb_nodes, nb_nodes);
  Eigen::MatrixXf Me = Eigen::MatrixXf::Zero(nb_nodes_e, nb_nodes_e);

  // Indices
  std::vector<TripletType> coefficients;

  // Element and nodes
  Eigen::VectorXi elem(el_mat_size);
  Eigen::MatrixXf elem_nodes(el_mat_size, 3);

  // Quadrature rule
  const Eigen::MatrixXf quadrature_points = py::cast<Eigen::MatrixXf>(quadrature_rule.attr("points"));
  const Eigen::VectorXf quadrature_weights = py::cast<Eigen::VectorXf>(quadrature_rule.attr("weights"));

  // Loop over elements for assembling
  for (int e = 0; e < nb_elems; e++){

    // Nodes in the element
    elem = elems(e, Eigen::indexing::all);
    elem_nodes = nodes(elem, Eigen::indexing::all);

    // Local assemble
    Me = LocalMassAssemble(elem_nodes, quadrature_points, quadrature_weights, finite_element);

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


// Assemble local stiffness matrix
Eigen::MatrixXf LocalGradAssemble(
  const Eigen::MatrixXf &nodes,
  const Eigen::MatrixXf &qpoints,
  const Eigen::VectorXf &qweights,
  const py::object &finite_element,
  const int &dim){

  // Number of degrees of freedom
  const auto nb_dofs = py::cast<int>(finite_element.attr("dimension"));

  // Assembled local matrix
  Eigen::MatrixXf Ce(nb_dofs, nb_dofs);
  Ce.setZero();

  // Assembling
  for (int q = 0; q < qweights.size(); q++){ 

    // Isoparametric mapping
    const IsoMap tmp = IsoMapping(nodes, qpoints.row(q), finite_element);
    const auto& [S, dSdx, dSdy, dSdz, detJ] = tmp;

    Eigen::Matrix<float, 4, 1> dSdi;
    if (dim == 0){
      dSdi = dSdx;
    }
    else if (dim == 1){
      dSdi = dSdy;
    }
    else if (dim == 2){
      dSdi = dSdz;
    }

    // Matrix assembling (should the determinant be adjusted?)
    Ce += (dSdi*S.transpose())*detJ*qweights(q);
  }

  return Ce;
}


// Assemble mass matrix
Eigen::MatrixXf GradProjection(
  const Eigen::VectorXf &u,
  const Eigen::MatrixXi &elems,
  const Eigen::MatrixXf &nodes,
  const py::object &finite_element,
  const py::object &quadrature_rule){

  // Geometric dimension
  const int dim = nodes.cols();

  // Number of elements, nodes, and dofs
  const int nb_elems    = elems.rows();
  const int nb_nodes    = nodes.rows();
  const int nb_nodes_e  = elems.row(0).size();
  const int el_mat_size = nb_nodes_e*nb_nodes_e;

  // Quadrature rule
  const Eigen::MatrixXf quadrature_points = py::cast<Eigen::MatrixXf>(quadrature_rule.attr("points"));
  const Eigen::VectorXf quadrature_weights = py::cast<Eigen::VectorXf>(quadrature_rule.attr("weights"));

  // Global and local mass matrices
  Eigen::SparseMatrix<float> M(nb_nodes, nb_nodes);
  Eigen::SparseMatrix<float> C(nb_nodes, nb_nodes);
  Eigen::MatrixXf Me = Eigen::MatrixXf::Zero(nb_nodes_e, nb_nodes_e);
  Eigen::MatrixXf Ce = Eigen::MatrixXf::Zero(nb_nodes_e, nb_nodes_e);

  // Indices
  std::vector<TripletType> M_coefficients;

  // Element and nodes
  Eigen::VectorXi elem(el_mat_size);
  Eigen::MatrixXf elem_nodes(el_mat_size, 3);

  // Assemble mass matrix
  for (int e = 0; e < nb_elems; e++){
  
    // Nodes in the element
    elem = elems(e, Eigen::indexing::all);
    elem_nodes = nodes(elem, Eigen::indexing::all);

    // Local mass assemble
    Me = LocalMassAssemble(elem_nodes, quadrature_points, quadrature_weights, finite_element);
  
    // Fill triplets
    for (int i = 0; i < nb_nodes_e; i++){
      for (int j = 0; j < nb_nodes_e; j++){
        M_coefficients.push_back(TripletType(elem(i), elem(j), Me(i,j)));
      }
    }        
  }
  M.setFromTriplets(M_coefficients.begin(), M_coefficients.end());

  // Solver
  Eigen::VectorXf b = Eigen::VectorXf::Zero(nb_nodes);
  Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  solver.compute(M);
  Eigen::MatrixXf w = Eigen::MatrixXf::Zero(nb_nodes, dim);

  // Assemble LHS matrix
  for (int d = 0; d < dim; d++){

    // Indices
    std::vector<TripletType> C_coefficients;

    for (int e = 0; e < nb_elems; e++){
  
      // Nodes in the element
      elem = elems(e, Eigen::indexing::all);
      elem_nodes = nodes(elem, Eigen::indexing::all);

      // Local assemble
      Ce = LocalGradAssemble(elem_nodes, quadrature_points, quadrature_weights, finite_element, d);

      // Fill triplets
      for (int i = 0; i < nb_nodes_e; i++){
        for (int j = 0; j < nb_nodes_e; j++){
          C_coefficients.push_back(TripletType(elem(i), elem(j), Ce(i,j)));
        }
      }        
    }

    // Sparse matrix filling
    C.setFromTriplets(C_coefficients.begin(), C_coefficients.end());

    // Solve
    b = C*u;
    w(Eigen::indexing::all, d) = solver.solve(b);
  }

  return w;
}

PYBIND11_MODULE(FEAssemble, m) {
    m.doc() = "Finite elements functions"; // optional module docstring
    m.def("MassAssemble", &MassAssemble, py::return_value_policy::reference);
    m.def("GradProjection", &GradProjection, py::return_value_policy::reference);
}
