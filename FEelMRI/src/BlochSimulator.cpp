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
    const py::function &pod_trajectory
    ){

    // Number of nodes and time points
    const int n_pos = r0.rows();
    const int n_time = rf_all.size();

    // Position at current time
    Matrix<T, Dynamic, Dynamic> r(n_pos, 3);

    // Initialize matrices for Mxy and Mz and set initial conditions
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy = Matrix<std::complex<T>, Dynamic, Dynamic>::Zero(n_pos, n_time);
    Matrix<T, Dynamic, Dynamic> Mz = Matrix<T, Dynamic, Dynamic>::Zero(n_pos, n_time);
    Mxy.col(0) = Mxy_initial;
    Mz.col(0) = Mz_initial;

    // Initialize arrays for solving loop
    Array<T, Dynamic, 1> Bz(n_pos);
    Array<T, Dynamic, 1> B1_real(n_pos);
    Array<T, Dynamic, 1> B1_imag(n_pos);
    Matrix<T, Dynamic, 3> B(n_pos, 3);
    Array<T, Dynamic, 1> Bnorm(n_pos);
    Array<T, Dynamic, 1> phi(n_pos);
    Array<T, Dynamic, 1> nz(n_pos);
    Array<std::complex<T>, Dynamic, 1> nxy(n_pos);
    Array<T, Dynamic, 1> a(n_pos);
    Array<std::complex<T>, Dynamic, 1> cos_phi2(n_pos);
    Array<std::complex<T>, Dynamic, 1> sin_phi2(n_pos);
    Array<std::complex<T>, Dynamic, 1> alpha(n_pos);
    Array<std::complex<T>, Dynamic, 1> beta(n_pos);
    Array<std::complex<T>, Dynamic, 1> conj_a(n_pos);

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Solve the Bloch equations
    T t = 0.0;
    for (int i = 0; i < n_time - 1; ++i) {

        // Update position
        t += dt[i + 1];
        // r.noalias() = r0 + t*py::cast<Matrix<T, Dynamic, Dynamic>>(pod_trajectory(0.0));
        r.noalias() = r0 + py::cast<Matrix<T, Dynamic, Dynamic>>(pod_trajectory(t));
        // r.noalias() = r0;

        // External magnetic fields
        B1_real = rf_all[i + 1].real();
        B1_imag = rf_all[i + 1].imag();
        Bz = (r * G_all.row(i + 1).transpose()).array() + delta_B;
        B.col(0) = B1_real;
        B.col(1) = B1_imag;
        B.col(2) = Bz;

        Bnorm = B.rowwise().norm();
        phi = -gamma * Bnorm * dt[i + 1];

        nz = B.col(2).array() / Bnorm.max(T(1e-12));
        nxy = (B.col(0).array() + i1 * B.col(1).array()) / Bnorm.max(T(1e-12));

        a = phi * T(0.5);
        cos_phi2 = a.cos();
        sin_phi2 = a.sin();

        alpha = cos_phi2 - i1 * nz * sin_phi2;
        beta = - i1 * nxy * sin_phi2;

        // if (regime_idx[i + 1] == 0) {
        //     // Precession regime
        //     Mxy.col(i + 1) = Mxy.col(i).array() * (phi * Complex(0, 1)).exp();
        //     Mz.col(i + 1) = Mz.col(i);
        // } else {
            // Rotation regime
            conj_a = alpha.conjugate();
            Mxy.col(i + 1) = (2.0 * conj_a * beta * Mz.col(i).array()
                            + conj_a.square() * Mxy.col(i).array()
                            - beta.square() * Mxy.col(i).array().conjugate()).matrix();

            Mz.col(i + 1) = ((alpha.abs2() - beta.abs2()) * Mz.col(i).array()
                            - 2.0 * (alpha * beta * Mxy.col(i).array().conjugate()).real()).matrix();
        // }

        // Apply T2 and T1 relaxation
        Mxy.col(i + 1) = Mxy.col(i + 1).array() * (-dt[i + 1] / T2).exp();
        Mz.col(i + 1) = Mz.col(i + 1).array() * (-dt[i + 1] / T1).exp()
                      + (T(1.0) - (-dt[i + 1] / T1).exp()) * M0;
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

    // Number of nodes and time points
    const int n_pos = r0.rows();
    const int n_time = rf_all.size();

    // Initialize matrices for Mxy and Mz and set initial conditions
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy = Matrix<std::complex<T>, Dynamic, Dynamic>::Zero(n_pos, n_time);
    Matrix<T, Dynamic, Dynamic> Mz = Matrix<T, Dynamic, Dynamic>::Zero(n_pos, n_time);
    Mxy.col(0) = Mxy_initial;
    Mz.col(0) = Mz_initial;

    // Initialize arrays for solving loop
    Array<T, Dynamic, 1> Bz(n_pos);
    Array<T, Dynamic, 1> B1_real(n_pos);
    Array<T, Dynamic, 1> B1_imag(n_pos);
    Matrix<T, Dynamic, 3> B(n_pos, 3);
    Array<T, Dynamic, 1> Bnorm(n_pos);
    Array<T, Dynamic, 1> phi(n_pos);
    Array<T, Dynamic, 1> nz(n_pos);
    Array<std::complex<T>, Dynamic, 1> nxy(n_pos);
    Array<T, Dynamic, 1> a(n_pos);
    Array<std::complex<T>, Dynamic, 1> cos_phi2(n_pos);
    Array<std::complex<T>, Dynamic, 1> sin_phi2(n_pos);
    Array<std::complex<T>, Dynamic, 1> alpha(n_pos);
    Array<std::complex<T>, Dynamic, 1> beta(n_pos);
    Array<std::complex<T>, Dynamic, 1> conj_a(n_pos);

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Solve the Bloch equations
    for (int i = 0; i < n_time - 1; ++i) {

        // External magnetic fields
        B1_real = rf_all[i + 1].real();
        B1_imag = rf_all[i + 1].imag();
        Bz = (r0 * G_all.row(i + 1).transpose()).array() + delta_B;
        B.col(0) = B1_real;
        B.col(1) = B1_imag;
        B.col(2) = Bz;

        Bnorm = B.rowwise().norm();
        phi = -gamma * Bnorm * dt[i + 1];

        nz = B.col(2).array() / Bnorm.max(T(1e-12));
        nxy = (B.col(0).array() + i1 * B.col(1).array()) / Bnorm.max(T(1e-12));

        a = phi * T(0.5);
        cos_phi2 = a.cos();
        sin_phi2 = a.sin();

        alpha = cos_phi2 - i1 * nz * sin_phi2;
        beta = - i1 * nxy * sin_phi2;

        // if (regime_idx[i + 1] == 0) {
        //     // Precession regime
        //     Mxy.col(i + 1) = Mxy.col(i).array() * (phi * Complex(0, 1)).exp();
        //     Mz.col(i + 1) = Mz.col(i);
        // } else {
            // Rotation regime
            conj_a = alpha.conjugate();
            Mxy.col(i + 1) = (2.0 * conj_a * beta * Mz.col(i).array()
                            + conj_a.square() * Mxy.col(i).array()
                            - beta.square() * Mxy.col(i).array().conjugate()).matrix();

            Mz.col(i + 1) = ((alpha.abs2() - beta.abs2()) * Mz.col(i).array()
                            - 2.0 * (alpha * beta * Mxy.col(i).array().conjugate()).real()).matrix();
        // }

        // Apply T2 and T1 relaxation
        Mxy.col(i + 1) = Mxy.col(i + 1).array() * (-dt[i + 1] / T2).exp();
        Mz.col(i + 1) = Mz.col(i + 1).array() * (-dt[i + 1] / T1).exp()
                      + (T(1.0) - (-dt[i + 1] / T1).exp()) * M0;
  }

    return {Mxy, Mz};

}


PYBIND11_MODULE(BlochSimulator, m) {
  // Overload for the case without pod_trajectory
  m.def("solve_mri", py::overload_cast<const Matrix<double, Dynamic, 3>&,
    const Array<double, Dynamic, 1>& ,
    const Array<double, Dynamic, 1>& ,
    const Array<double, Dynamic, 1>& ,
    const double & ,
    const double & ,
    const Array<std::complex<double>, Dynamic, 1>& ,
    const Matrix<double, Dynamic, 3>& ,
    const Array<double, Dynamic, 1>& ,
    const VectorXi& ,
    const Array<std::complex<double>, Dynamic, 1> &,
    const Array<double, Dynamic, 1> &,
    const py::none &>(&solve_mri<double>));
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
  m.def("solve_mri", py::overload_cast<const Matrix<double, Dynamic, 3>&,
    const Array<double, Dynamic, 1>& ,
    const Array<double, Dynamic, 1>& ,
    const Array<double, Dynamic, 1>& ,
    const double & ,
    const double & ,
    const Array<std::complex<double>, Dynamic, 1>& ,
    const Matrix<double, Dynamic, 3>& ,
    const Array<double, Dynamic, 1>& ,
    const VectorXi& ,
    const Array<std::complex<double>, Dynamic, 1> &,
    const Array<double, Dynamic, 1> &,
    const py::function &>(&solve_mri<double>));
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
    const py::function &>(&solve_mri<float>));
}