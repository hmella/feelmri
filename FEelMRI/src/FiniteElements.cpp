#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
namespace py = pybind11;

// Custom types
typedef std::tuple<Matrix<double, 4, 1>,
  Matrix<double, 4, 1>,
  Matrix<double, 4, 1>,
  Matrix<double, 4, 1>> shapeFunctions;
typedef std::tuple<Matrix<double, 4, 3>, std::array<double, 4>> quadRule;
typedef std::tuple<Matrix<double, 4, 1>,
  Matrix<double, 4, 1>,
  Matrix<double, 4, 1>,
  Matrix<double, 4, 1>,
  double> isopMap;

// Triplets
typedef Triplet<double> T;


// Quadrature rule
quadRule quadratureRule(int degree){
  // // Points
  // double a = 0.585410196624969;
  // double b = 0.138196601125011;
  // Matrix<double, 4, 3> x = (Matrix<double, 4, 3>() <<
  // a, b, b,
  // b, a, b,
  // b, b, a,
  // b, b, b).finished();

  // // Weights
  // std::array<double, 4> w;
  // w[0] = 1.0/24.0;
  // w[1] = 1.0/24.0;
  // w[2] = 1.0/24.0;
  // w[3] = 1.0/24.0;

  // Points
  Matrix<double, 4, 3> x = (Matrix<double, 4, 3>() <<
    0.5854101966249685,  0.1381966011250105,  0.1381966011250105,
    0.1381966011250105,  0.5854101966249685,  0.1381966011250105,
    0.1381966011250105,  0.1381966011250105,  0.5854101966249685,
    0.1381966011250105,  0.1381966011250105,  0.1381966011250105
  ).finished();

  // Weights
  std::array<double, 4> w;
  w[0] = 0.25/6.0;
  w[1] = 0.25/6.0;
  w[2] = 0.25/6.0;
  w[3] = 0.25/6.0;

  return std::make_tuple(x, w);
}


// Shape functions and derivatives
shapeFunctions P1ShapeTetra(double r, double s, double t){
  // Shape functions
  Matrix<double, 4, 1> S;
  S(0,0) = 1-r-s-t;
  S(1,0) = r;
  S(2,0) = s;
  S(3,0) = t;

  // Derivatives
  Matrix<double, 4, 1> dSdr;
  dSdr(0,0) = -1;
  dSdr(1,0) = 1;
  dSdr(2,0) = 0;
  dSdr(3,0) = 0;
  Matrix<double, 4, 1> dSds;
  dSds(0,0) = -1;
  dSds(1,0) = 0;
  dSds(2,0) = 1;
  dSds(3,0) = 0;
  Matrix<double, 4, 1> dSdt;
  dSdt(0,0) = -1;
  dSdt(1,0) = 0;
  dSdt(2,0) = 0;
  dSdt(3,0) = 1;

  return std::make_tuple(S, dSdr, dSds, dSdt);
}


// Isoparametric mapping
isopMap isopMapping(const MatrixXd &x,
  const MatrixXd &y,
  const MatrixXd &z,
  const double &r,
  const double &s,
  const double &t){
  // Evaluate shape functions
  const shapeFunctions sss = P1ShapeTetra(r,s,t);
  const Matrix<double, 4, 1> S = std::get<0>(sss);
  const Matrix<double, 4, 1> dSdr = std::get<1>(sss);
  const Matrix<double, 4, 1> dSds = std::get<2>(sss);
  const Matrix<double, 4, 1> dSdt = std::get<3>(sss);

  // Jacobian matrix
  const Matrix3d J = (Matrix3d() << (dSdr.transpose()*x).sum(),
    (dSdr.transpose()*y).sum(),
    (dSdr.transpose()*z).sum(),
    (dSds.transpose()*x).sum(),
    (dSds.transpose()*y).sum(),
    (dSds.transpose()*z).sum(),
    (dSdt.transpose()*x).sum(),
    (dSdt.transpose()*y).sum(),
    (dSdt.transpose()*z).sum()).finished();

  // Jacobian determinant
  const double detJ = J.determinant();

  // Jacobian inverse matrix
  const Matrix3d invJ = J.inverse();

  // Shape functions in the global element
  const Matrix<double, 4, 1> dSdx = invJ(0,0)*dSdr + invJ(0,1)*dSds + invJ(0,2)*dSdt;
  const Matrix<double, 4, 1> dSdy = invJ(1,0)*dSdr + invJ(1,1)*dSds + invJ(1,2)*dSdt;
  const Matrix<double, 4, 1> dSdz = invJ(2,0)*dSdr + invJ(2,1)*dSds + invJ(2,2)*dSdt;

  return std::make_tuple(S, dSdx, dSdy, dSdz, detJ);
}


// Assemble integral locally
Vector3d local_assemble(const Matrix<double, 4, 3> &xe,
  const Matrix<double, 4, 3> &values,
  const Matrix<double, 4, 3> &qpoints,
  const std::array<double, 4> &qweights){

  // Assembled integral
  Vector3d integral;

  // Assembling
  for (int q=0; q<4; q++){ 
    // Isoparametric mapping
    const isopMap iso = isopMapping(xe.col(0),xe.col(1),xe.col(2),
                                    qpoints(q,0),qpoints(q,1),qpoints(q,2));
    const Matrix<double, 4, 1> S = std::get<0>(iso);
    const double detJ = std::get<4>(iso);

    // Matrix assembling (should the determinant be adjusted?)
    integral(0) = (S.array()*values.col(0).array()).sum()*detJ*qweights[q];
    integral(1) = (S.array()*values.col(1).array()).sum()*detJ*qweights[q];
    integral(2) = (S.array()*values.col(2).array()).sum()*detJ*qweights[q];
  }

  return integral;
}


// Assemble integral over the whole domain
Vector3d assemble(const MatrixXi &elems,
  const MatrixXd &nodes,
  const MatrixXd &values){

  // Number of elements
  const int nel = elems.rows();

  // Quadrature rule
  const quadRule quadrature = quadratureRule(0);
  const Matrix<double, 4, 3> qpoints  = std::get<0>(quadrature);
  const std::array<double, 4> qweights = std::get<1>(quadrature);

  // Assembled integral
  Vector3d integral = (Vector3d() << 0, 0, 0).finished();
  Vector3d local_integral = (Vector3d() << 0, 0, 0).finished();

  // Assembling
  Vector<int, 4> elem;
  Matrix<double, 4, 3> xe;
  Matrix<double, 4, 3> val;
  for (int e=0; e<nel; e++){ // loop over elements
    for (int n=0; n<4; n++){ // loop over nodes in the element e
      // Nodes in the element
      elem = elems(e, indexing::all);
      xe = nodes(elem, indexing::all);

      // Nodal values to be integrated
      val = values(elem, indexing::all);

      // Local assemble
      local_integral = local_assemble(xe, val, qpoints, qweights);
      integral += local_integral;
         
    }
  }

  return integral;
}


// Assemble local mass matrix
std::tuple<VectorXd, Matrix4i> localMassAssemble(
  const Matrix<double, 4, 3> &xe,
  const Matrix<double, 4, 3> &qpoints,
  const std::array<double, 4> &qweights){

  // Assembled local matrix
  Matrix4d Me;

  // Local indices
  Matrix4i idx;

  // Assembling
  for (int q=0; q<4; q++){ 
    // Isoparametric mapping
    const isopMap iso = isopMapping(xe.col(0),xe.col(1),xe.col(2),
                                    qpoints(q,0),qpoints(q,1),qpoints(q,2));
    const Matrix<double, 4, 1> S = std::get<0>(iso);
    const double detJ = std::abs(std::get<4>(iso));

    // Matrix assembling (should the determinant be adjusted?)
    Me = (S*S.transpose())*detJ*qweights[q];

    // Indices
    idx(q, indexing::all) = (Array<int, 1, 4>() << 0, 1, 2, 3).finished();
  }

  return std::make_tuple(Me.reshaped(), idx);
}


// Assemble mass matrix
SparseMatrix<double> massAssemble(const MatrixXi &elems,
  const MatrixXd &nodes){

  // Number of elements and nodes
  const int nel = elems.rows();
  const int nno = nodes.rows();

  // Quadrature rule
  const quadRule quadrature = quadratureRule(0);
  const Matrix<double, 4, 3> qpoints  = std::get<0>(quadrature);
  const std::array<double, 4> qweights = std::get<1>(quadrature);

  // Mass matrix
  SparseMatrix<double> M(nno, nno);
  Matrix<double, 1, 16> Me;
  Matrix4i idx;

  // Indices
  Vector<int,16> rows;
  Vector<int,16> cols;
  std::vector<T> coefficients;

  // Assembling
  Vector<int, 4> elem;
  Matrix<double, 4, 3> xe;
  for (int e=0; e<nel; e++){ // loop over elements
    // Nodes in the element
    elem = elems(e, indexing::all);
    xe = nodes(elem, indexing::all);

    // Local assemble
    auto [Me, idx] = localMassAssemble(xe, qpoints, qweights);

    // Assign triplets
    rows = elem(idx.reshaped());
    cols = elem(idx.transpose().reshaped());

    // Fill triplets
    for (int i=0; i<16; i++){
      coefficients.push_back(T(rows(i),cols(i),Me(i)));
    }        
  }

  // Sparse matrix filling
  M.setFromTriplets(coefficients.begin(), coefficients.end());

  return M;
}


// Assemble integral over the whole domain using the mass matrix
VectorXd massAssembleInt(const MatrixXi &elems,
  const MatrixXd &nodes,
  const MatrixXd &values){

  // Number of nodes
  const int nno = nodes.rows();

  // Number of components to integrate
  const int nva = values.cols();

  // Function of ones
  MatrixXd ones = MatrixXd::Constant(nva,nno,1.0);

  // Mass matrix
  SparseMatrix<double> M = massAssemble(elems, nodes);

  // Calculate integral
  VectorXd integral = ones*(M*values);

  return integral;
}


PYBIND11_MODULE(FiniteElements, m) {
    m.doc() = "Finite elements functions"; // optional module docstring
    m.def("assemble", &assemble, py::return_value_policy::reference);
    m.def("massAssemble", &massAssemble, py::return_value_policy::reference);
    m.def("massAssembleInt", &massAssembleInt, py::return_value_policy::reference);
}
