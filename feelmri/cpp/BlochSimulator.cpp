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
    const Matrix<T, Dynamic, 3, RowMajor> &r0,
    const Matrix<T, Dynamic, 1> &T1,
    const Matrix<T, Dynamic, 1> &T2,
    const Matrix<T, Dynamic, 1> &delta_B,
    const T &M0,
    const T &gamma,
    const Matrix<std::complex<T>, Dynamic, 1> &rf_all,
    const Matrix<T, Dynamic, 3> &G_all,
    const Matrix<T, Dynamic, 1> &dt,
    const Matrix<bool, Dynamic, 1> &regime_idx,
    const Matrix<std::complex<T>, Dynamic, 1> &Mxy_initial,
    const Matrix<T, Dynamic, 1> &Mz_initial,
    const py::function &pod_trajectory
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

    // The ‘template’ disambiguator is REQUIRED here:
    auto arr = pod_trajectory(0.0).template cast<py::array_t<T, py::array::c_style>>();
    Map<const Matrix<T, Dynamic, 3, RowMajor>> traj(nullptr, n_pos, 3);

    // Matrixs for precomputed reciprocals (do once before loop)
    const Matrix<T, Dynamic, 1> invT1 = T1.cwiseInverse();
    const Matrix<T, Dynamic, 1> invT2 = T2.cwiseInverse();
    Matrix<T, Dynamic, 1> e1(n_pos);
    Matrix<T, Dynamic, 1> e2(n_pos);

    // Variables for gradients and RF pulses
    T rf2;
    C rf_complex;
    T Gx, Gy, Gz;
    T kappa;

    // thread-local scalars
    T Bz_p, Bnorm_p, invB_p, half_phi_p, c, s;
    C alpha_p, beta_p, conj_a_p, nxy_p;
    C Mxy_prev_p, Mxy_new_p; 
    T Mz_prev_p, Mz_new_p;
    T nz, a2, b2;

    // Solve the Bloch equations
    T ti = 0.0;
    T dt_i = -1.0;
    for (int i = 0; i < n_time - 1; ++i) {
      
        // Precompute exponentials
        if (dt_i != dt[i + 1]) {
            for (int p = 0; p < n_pos; ++p) {
                e1(p) = std::exp(-dt[i + 1] * invT1(p));
                e2(p) = std::exp(-dt[i + 1] * invT2(p));
            }
            // e1 = (-dt[i + 1] * invT1).exp();
            // e2 = (-dt[i + 1] * invT2).exp();
        }

        // Update time
        dt_i = dt[i + 1];
        ti += dt_i;

        // Get trajectory at time ti
        arr = pod_trajectory(ti).template cast<py::array_t<T, py::array::c_style>>();

        // Map the data (no copy)
        Eigen::Map<const Matrix<T, Dynamic, 3, RowMajor>> traj(arr.data(), n_pos, 3);

        // rf parts (scalars)
        rf_complex = rf_all[i + 1];
        rf2 = std::norm(rf_complex);
        kappa = -T(0.5) * gamma * dt_i;   // half-phi scale

        // Gradients
        Gx = G_all(i + 1, 0);
        Gy = G_all(i + 1, 1);
        Gz = G_all(i + 1, 2);

        // Solve for each FE dof position
        for (int p = 0; p < n_pos; ++p) {
            // gz = (r0+traj)[p,:] · G
            Bz_p = (r0(p,0) + traj(p,0))*Gx 
                + (r0(p,1) + traj(p,1))*Gy 
                + (r0(p,2) + traj(p,2))*Gz 
                + delta_B(p);

            Bnorm_p = std::sqrt(Bz_p*Bz_p + rf2);
            if (Bnorm_p < T(1e-12)) Bnorm_p = T(1e-12);
            nz = Bz_p / Bnorm_p;

            half_phi_p = Bnorm_p * kappa;
            c = std::cos(half_phi_p);
            s = std::sin(half_phi_p);
            alpha_p = C(c, -nz*s);
            conj_a_p = std::conj(alpha_p);

            Mxy_prev_p = Mxy(p, i);
            Mz_prev_p  = Mz(p, i);

            a2 = std::norm(alpha_p);

            // if (regime_idx[i+1]) {
              nxy_p = rf_complex / Bnorm_p;
              beta_p = - i1 * nxy_p * s;

              Mxy_new_p = T(2)*conj_a_p*beta_p*Mz_prev_p
                        + conj_a_p*conj_a_p * Mxy_prev_p
                        - beta_p*beta_p * std::conj(Mxy_prev_p);

              b2 = std::norm(beta_p);
              Mz_new_p = (a2 - b2) * Mz_prev_p
                      - T(2) * std::real(alpha_p * beta_p * std::conj(Mxy_prev_p));
            // } else {
            //   Mxy_new_p = conj_a_p*conj_a_p * Mxy_prev_p;
            //   Mz_new_p  = a2 * Mz_prev_p;
            // }

            // Relaxation (e1/e2 precomputed scalars per position)
            Mxy(p, i+1) = Mxy_new_p * e2(p);
            Mz (p, i+1) = Mz_new_p  * e1(p) + (T(1) - e1(p)) * M0;
        }

    }

    return {Mxy, Mz};
}


template <typename T>
Magnetization<T> solve_mri(
    const Matrix<T, Dynamic, 3, RowMajor> &r0,
    const Matrix<T, Dynamic, 1> &T1,
    const Matrix<T, Dynamic, 1> &T2,
    const Matrix<T, Dynamic, 1> &delta_B,
    const T &M0,
    const T &gamma,
    const Matrix<std::complex<T>, Dynamic, 1> &rf_all,
    const Matrix<T, Dynamic, 3> &G_all,
    const Matrix<T, Dynamic, 1> &dt,
    const Matrix<bool, Dynamic, 1> &regime_idx,
    const Matrix<std::complex<T>, Dynamic, 1> &Mxy_initial,
    const Matrix<T, Dynamic, 1> &Mz_initial,
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

    // Matrixs for precomputed reciprocals (do once before loop)
    const Matrix<T, Dynamic, 1> invT1 = T1.cwiseInverse();
    const Matrix<T, Dynamic, 1> invT2 = T2.cwiseInverse();
    Matrix<T, Dynamic, 1> e1(n_pos);
    Matrix<T, Dynamic, 1> e2(n_pos);

    // Variables for gradients and RF pulses
    T rf2;
    C rf_complex;
    T Gx, Gy, Gz;
    T kappa;

    // thread-local scalars
    T Bz_p, Bnorm_p, invB_p, half_phi_p, c, s;
    C alpha_p, beta_p, conj_a_p, nxy_p;
    C Mxy_prev_p, Mxy_new_p; 
    T Mz_prev_p, Mz_new_p;
    T nz, a2, b2;

    // Solve the Bloch equations
    T ti = 0.0;
    T dt_i = -1.0;
    for (int i = 0; i < n_time - 1; ++i) {
      
        // Precompute exponentials
        if (dt_i != dt[i + 1]) {
            for (int p = 0; p < n_pos; ++p) {
                e1(p) = std::exp(-dt[i + 1] * invT1(p));
                e2(p) = std::exp(-dt[i + 1] * invT2(p));
            }
            // e1 = (-dt[i + 1] * invT1).exp();
            // e2 = (-dt[i + 1] * invT2).exp();
        }

        // Update time
        dt_i = dt[i + 1];
        ti += dt_i;

        // rf parts (scalars)
        rf_complex = rf_all[i + 1];
        rf2 = std::norm(rf_complex);
        kappa = -T(0.5) * gamma * dt_i;   // half-phi scale

        // Gradients
        Gx = G_all(i + 1, 0);
        Gy = G_all(i + 1, 1);
        Gz = G_all(i + 1, 2);

        // Solve for each FE dof position
        for (int p = 0; p < n_pos; ++p) {
            // gz = (r0+traj)[p,:] · G
            Bz_p = r0(p,0)*Gx 
                + r0(p,1)*Gy 
                + r0(p,2)*Gz 
                + delta_B(p);

            Bnorm_p = std::sqrt(Bz_p*Bz_p + rf2);
            if (Bnorm_p < T(1e-12)) Bnorm_p = T(1e-12);
            nz = Bz_p / Bnorm_p;

            half_phi_p = Bnorm_p * kappa;
            c = std::cos(half_phi_p);
            s = std::sin(half_phi_p);
            alpha_p = C(c, -nz*s);
            conj_a_p = std::conj(alpha_p);

            Mxy_prev_p = Mxy(p, i);
            Mz_prev_p  = Mz(p, i);

            a2 = std::norm(alpha_p);

            // if (regime_idx[i+1]) {
              nxy_p = rf_complex / Bnorm_p;
              beta_p = - i1 * nxy_p * s;

              Mxy_new_p = T(2)*conj_a_p*beta_p*Mz_prev_p
                        + conj_a_p*conj_a_p * Mxy_prev_p
                        - beta_p*beta_p * std::conj(Mxy_prev_p);

              b2 = std::norm(beta_p);
              Mz_new_p = (a2 - b2) * Mz_prev_p
                      - T(2) * std::real(alpha_p * beta_p * std::conj(Mxy_prev_p));
            // } else {
            //   Mxy_new_p = conj_a_p*conj_a_p * Mxy_prev_p;
            //   Mz_new_p  = a2 * Mz_prev_p;
            // }

            // Relaxation (e1/e2 precomputed scalars per position)
            Mxy(p, i+1) = Mxy_new_p * e2(p);
            Mz (p, i+1) = Mz_new_p  * e1(p) + (T(1) - e1(p)) * M0;
        }

    }

    return {Mxy, Mz};
}


PYBIND11_MODULE(BlochSimulator, m) {
  // Define function overloads
  // Overload for the case with pod_trajectory
  m.def("solve_mri", py::overload_cast<
    const Matrix<float, Dynamic, 3, RowMajor> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const float & ,
    const float & ,
    const Matrix<std::complex<float>, Dynamic, 1> &,
    const Matrix<float, Dynamic, 3> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<bool, Dynamic, 1> & ,
    const Matrix<std::complex<float>, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const py::function &>(&solve_mri<float>));
  m.def("solve_mri", py::overload_cast<
    const Matrix<float, Dynamic, 3, RowMajor> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const float & ,
    const float & ,
    const Matrix<std::complex<float>, Dynamic, 1> &,
    const Matrix<float, Dynamic, 3> & ,
    const Matrix<float, Dynamic, 1> & ,
    const Matrix<bool, Dynamic, 1> & ,
    const Matrix<std::complex<float>, Dynamic, 1> & ,
    const Matrix<float, Dynamic, 1> & ,
    const py::none &>(&solve_mri<float>));
}