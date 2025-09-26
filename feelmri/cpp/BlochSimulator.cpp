#pragma once
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

template <typename T>
using Magnetization = std::pair<
    Matrix<std::complex<T>, Dynamic, Dynamic>,
    Matrix<T, Dynamic, Dynamic>
    >;

template <typename T>
Magnetization<T> solve_mri(
    const py::EigenDRef<Matrix<T, Dynamic, 3>> r0,
    const py::EigenDRef<Array<T, Dynamic, 1>> T1,
    const py::EigenDRef<Array<T, Dynamic, 1>> T2,
    const py::EigenDRef<Array<T, Dynamic, 1>> delta_B,
    const T M0,
    const T gamma,
    const py::EigenDRef<Array<std::complex<T>, Dynamic, 1>> rf_all,
    const py::EigenDRef<Matrix<T, Dynamic, 3>> G_all,
    const py::EigenDRef<Array<T, Dynamic, 1>> dt,
    const py::EigenDRef<VectorXi> regime_idx,
    const py::EigenDRef<Array<std::complex<T>, Dynamic, 1>> Mxy_initial,
    const py::EigenDRef<Array<T, Dynamic, 1>> Mz_initial,
    const py::function &pod_trajectory
    ){
    // Complex unit and data type
    using C = std::complex<T>;
    const C i1(0.0, 1.0);

    // Number of nodes and time points
    const int n_pos = r0.rows();
    const int n_time = rf_all.size();

    // Position at current time
    Matrix<T, Dynamic, Dynamic> r(n_pos, 3);

    // Initialize matrices for Mxy and Mz and set initial conditions
    Matrix<C, Dynamic, Dynamic> Mxy(n_pos, n_time);
    Matrix<T, Dynamic, Dynamic> Mz(n_pos, n_time);
    Mxy.col(0) = Mxy_initial;
    Mz.col(0) = Mz_initial;

    // Initialize arrays for solving loop
    Array<T, Dynamic, 1> Bz(n_pos);
    Array<T, Dynamic, 1> Bnorm(n_pos);
    Array<T, Dynamic, 1> half_phi(n_pos);
    Array<C, Dynamic, 1> nz(n_pos);
    Array<C, Dynamic, 1> nxy(n_pos);
    Array<C, Dynamic, 1> cos_phi2(n_pos);
    Array<C, Dynamic, 1> sin_phi2(n_pos);
    Array<C, Dynamic, 1> alpha(n_pos);
    Array<C, Dynamic, 1> beta(n_pos);
    Array<C, Dynamic, 1> conj_a(n_pos);

    // The ‘template’ disambiguator is REQUIRED here:
    auto arr = pod_trajectory(0.0).template cast<py::array_t<T, py::array::c_style>>();
    Map<const Matrix<T, Dynamic, 3, RowMajor>> traj(nullptr, n_pos, 3);

    // Arrays for precomputed reciprocals 
    const Array<T,Dynamic,1> invT1 = T1.cwiseInverse();  // do once before loop
    const Array<T,Dynamic,1> invT2 = T2.cwiseInverse();
    Array<T,Dynamic,1> e1(n_pos);
    Array<T,Dynamic,1> e2(n_pos);

    // Solve the Bloch equations
    T ti = 0.0;
    T dt_i = -1.0;
    for (int i = 0; i < n_time - 1; ++i) {

        // Precompute exponentials
        if (dt_i != dt[i + 1]) {
            e1 = (-dt[i + 1] * invT1).exp();
            e2 = (-dt[i + 1] * invT2).exp();
        }

        // Update time
        dt_i = dt[i + 1];
        ti += dt_i;

        // Update position
        {
          // py::gil_scoped_acquire gil;
          arr = pod_trajectory(ti).template cast<py::array_t<T, py::array::c_style>>();

          // Rebind Map to the new buffer (no allocations)
          new (&traj) Eigen::Map<const Matrix<T, Dynamic, 3, RowMajor>>(arr.data(), n_pos, 3);
        }

        // // Map without copy (NumPy is row-major/C-contiguous)
        // r.noalias() = r0;
        // r.noalias() += traj;

        // rf parts (scalars)
        const T rf_r = rf_all[i+1].real();
        const T rf_i = rf_all[i+1].imag();
        const T rf2  = rf_r*rf_r + rf_i*rf_i;

        // gz term per position
        Bz = ((r0 + traj) * G_all.row(i+1).transpose()).array() + delta_B;

        // Bnorm = sqrt( rf^2 + z^2 )
        Bnorm = (Bz.square() + rf2).sqrt().cwiseMax(T(1e-12));

        nz = Bz / Bnorm;

        half_phi = -T(0.5) * gamma * Bnorm * dt_i;
        cos_phi2 = half_phi.cos();
        sin_phi2 = half_phi.sin();

        alpha = cos_phi2 - i1 * nz * sin_phi2;
        conj_a = alpha.conjugate();

        // Rotation regime
        // if (regime_idx[i+1]){
          nxy = (rf_r + i1 * rf_i) / Bnorm;
          beta = - i1 * nxy * sin_phi2;
          Mxy.col(i + 1) = (T(2.0) * conj_a * beta * Mz.col(i).array()
                          + conj_a.square() * Mxy.col(i).array()
                          - beta.square() * Mxy.col(i).array().conjugate()).matrix();

          Mz.col(i + 1) = ((alpha.abs2() - beta.abs2()) * Mz.col(i).array()
                          - T(2.0) * (alpha * beta * Mxy.col(i).array().conjugate()).real()).matrix();
        // }
        // // Precession regime
        // else {
        //   Mxy.col(i + 1) = (conj_a.square() * Mxy.col(i).array()).matrix();

        //   Mz.col(i + 1) = (alpha.abs2() * Mz.col(i).array()).matrix();
        // }

        // Apply T2 and T1 relaxation
        Mxy.col(i + 1) = (Mxy.col(i + 1).array() * e2).matrix();
        Mz.col(i + 1) = (Mz.col(i + 1).array() * e1
                      + (T(1.0) - e1) * M0).matrix();        

    }

    return {Mxy, Mz};

}


template <typename T>
Magnetization<T> solve_mri(
    const Matrix<T, Dynamic, 3>& r0,             // (n_pos, 3)
    const Array<T, Dynamic, 1>& T1,              // (n_pos,)
    const Array<T, Dynamic, 1>& T2,              // (n_pos,)
    const Array<T, Dynamic, 1>& delta_B,         // (n_pos,)
    const T &M0,                                 // Scalar
    const T &gamma,                              // rad/ms/mT
    const Array<std::complex<T>, Dynamic, 1>& rf_all,       // (n_time,)
    const Matrix<T, Dynamic, 3>& G_all,          // (n_time, 3)
    const Array<T, Dynamic, 1>& dt,              // (n_time,)
    const VectorXi& regime_idx,                  // (n_time,)
    const Array<std::complex<T>, Dynamic, 1> &Mxy_initial,
    const Array<T, Dynamic, 1> &Mz_initial,
    const py::none &pod_trajectory
    ){
    // Complex unit and data type
    using C = std::complex<T>;
    const C i1(0.0, 1.0);

    // Number of nodes and time points
    const int n_pos = r0.rows();
    const int n_time = rf_all.size();

    // Initialize matrices for Mxy and Mz and set initial conditions
    Matrix<C, Dynamic, Dynamic> Mxy(n_pos, n_time);
    Matrix<T, Dynamic, Dynamic> Mz(n_pos, n_time);
    Mxy.col(0) = Mxy_initial;
    Mz.col(0) = Mz_initial;

    // Initialize arrays for solving loop
    Array<T, Dynamic, 1> Bz(n_pos);
    Array<T, Dynamic, 1> Bnorm(n_pos);
    Array<T, Dynamic, 1> half_phi(n_pos);
    Array<C, Dynamic, 1> nz(n_pos);
    Array<C, Dynamic, 1> nxy(n_pos);
    Array<C, Dynamic, 1> cos_phi2(n_pos);
    Array<C, Dynamic, 1> sin_phi2(n_pos);
    Array<C, Dynamic, 1> alpha(n_pos);
    Array<C, Dynamic, 1> beta(n_pos);
    Array<C, Dynamic, 1> conj_a(n_pos);

    // Arrays for precomputed reciprocals 
    const Array<T,Dynamic,1> invT1 = T1.cwiseInverse();  // do once before loop
    const Array<T,Dynamic,1> invT2 = T2.cwiseInverse();
    Array<T,Dynamic,1> e1(n_pos);
    Array<T,Dynamic,1> e2(n_pos);

    // Solve the Bloch equations
    T ti = 0.0;
    T dt_i = -1.0;
    for (int i = 0; i < n_time - 1; ++i) {

        // Precompute exponentials
        if (dt_i != dt[i + 1]) {
            e1 = (-dt[i + 1] * invT1).exp();
            e2 = (-dt[i + 1] * invT2).exp();
        }

        // Update time
        dt_i = dt[i + 1];
        ti += dt_i;

        // rf parts (scalars)
        const T rf_r = rf_all[i+1].real();
        const T rf_i = rf_all[i+1].imag();
        const T rf2  = rf_r*rf_r + rf_i*rf_i;

        // gz term per position
        Bz = (r0 * G_all.row(i+1).transpose()).array() - delta_B;

        // Bnorm = sqrt( rf^2 + z^2 )
        Bnorm = (Bz.square() + rf2).sqrt().cwiseMax(T(1e-12));

        nz = Bz / Bnorm;
        nxy = (rf_r + i1 * rf_i) / Bnorm;

        half_phi = -T(0.5) * gamma * Bnorm * dt_i;
        cos_phi2 = half_phi.cos();
        sin_phi2 = half_phi.sin();

        alpha = cos_phi2 - i1 * nz * sin_phi2;
        beta = - i1 * nxy * sin_phi2;
        conj_a = alpha.conjugate();

        // Rotation regime
        Mxy.col(i + 1) = (T(2.0) * conj_a * beta * Mz.col(i).array()
                        + conj_a.square() * Mxy.col(i).array()
                        - beta.square() * Mxy.col(i).array().conjugate()).matrix();

        Mz.col(i + 1) = ((alpha.abs2() - beta.abs2()) * Mz.col(i).array()
                        - T(2.0) * (alpha * beta * Mxy.col(i).array().conjugate()).real()).matrix();

        // Apply T2 and T1 relaxation
        Mxy.col(i + 1) = (Mxy.col(i + 1).array() * e2).matrix();
        Mz.col(i + 1) = (Mz.col(i + 1).array() * e1
                      + (T(1.0) - e1) * M0).matrix();
  }

    return {Mxy, Mz};
}


PYBIND11_MODULE(BlochSimulator, m) {
  // Overload for the case without pod_trajectory
  m.def("solve_mri", py::overload_cast<const Matrix<float, Dynamic, 3>&,
    const Array<float, Dynamic, 1>& ,
    const Array<float, Dynamic, 1>& ,
    const Array<float, Dynamic, 1>& ,
    const float & ,
    const float & ,
    const Array<std::complex<float>, Dynamic, 1>& ,
    const Matrix<float, Dynamic, 3>& ,
    const Array<float, Dynamic, 1>& ,
    const VectorXi& ,
    const Array<std::complex<float>, Dynamic, 1> &,
    const Array<float, Dynamic, 1> &,
    const py::none &>(&solve_mri<float>));
  // Overload for the case with pod_trajectory
  m.def("solve_mri", py::overload_cast<
    const py::EigenDRef<Matrix<float, Dynamic, 3>>,
    const py::EigenDRef<Array<float, Dynamic, 1>> ,
    const py::EigenDRef<Array<float, Dynamic, 1>>,
    const py::EigenDRef<Array<float, Dynamic, 1>>,
    const float ,
    const float ,
    const py::EigenDRef<Array<std::complex<float>, Dynamic, 1>>,
    const py::EigenDRef<Matrix<float, Dynamic, 3>>,
    const py::EigenDRef<Array<float, Dynamic, 1>> ,
    const py::EigenDRef<VectorXi> ,
    const py::EigenDRef<Array<std::complex<float>, Dynamic, 1>>,
    const py::EigenDRef<Array<float, Dynamic, 1>>,
    const py::function &>(&solve_mri<float>));
}