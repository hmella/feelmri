#include <Signal.h>
#include <PhaseContrast.h>

/**
 * @brief Computes the 3D flow image from k-space data.
 *
 * This function calculates the 3D flow image by simulating the k-space 
 * acquisition process. It takes into account the mass matrix, k-space 
 * trajectory, k-space timings, object initial position and velocity, 
 * encoded velocities, field inhomogeneity, T2 time of the blood, velocity 
 * encoding, and slice profile.
 *
 * @param MPI_rank The rank of the MPI process.
 * @param M The mass matrix.
 * @param kloc The k-space trajectory.
 * @param t The k-space timings.
 * @param r0 The initial position of the object.
 * @param v The initial velocity of the object.
 * @param phi_v The encoded velocities.
 * @param phi_dB0 The field inhomogeneity.
 * @param T2 The T2 time of the blood.
 * @param profile The slice profile.
 * @return A 4D tensor representing the k-space data.
 */
template <typename T>
Tensor<std::complex<T>, 4> PC(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Matrix<T, Dynamic, Dynamic> &v,
  const Matrix<T, Dynamic, Dynamic> &phi_v,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2,
  const Vector<std::complex<T>, Dynamic> &profile
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    // For non-cartesian acquisitions, this number should be 1
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r0.rows();

    // Nb of encoded velocities
    const uint nb_vencs = phi_v.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_nodes, nb_vencs);
    Mxy = 1.0e+3 * nb_nodes
        * (i1 * phi_v).array().exp() 
        * profile.array().replicate(1, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic>  phi_off(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_nodes);

    // Iterate over kspace phase lines
    uint i, j, k, l;
    for (j = 0; j < nb_lines; j++){

      // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (i = 0; i < nb_meas; i++){

        // Iterate over slices
        for (k = 0; k < nb_kz; k++){

          // Update blood position at time t(i,j)
          r.noalias() = UpdatePosition(r0, v, t(i,j,k));

          // Update off-resonance phase
          phi_off.noalias() = phi_dB0 * t(i,j,k);

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate T2/T2* decay
          T2_decay = (-t(i,j,k) / T2.array()).exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_vencs; l++){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay, l);
          }
        }
      }
    }

    return kspace;
}


/**
 * @brief Computes the 3D flow image from k-space data.
 *
 * This function calculates the 3D flow image by simulating the k-space 
 * acquisition process. It takes into account the mass matrix, k-space 
 * trajectory, k-space timings, object initial position and velocity, 
 * encoded velocities, field inhomogeneity, T2 time of the blood, velocity 
 * encoding, and slice profile.
 *
 * @param MPI_rank The rank of the MPI process.
 * @param M The mass matrix.
 * @param kloc The k-space trajectory.
 * @param t The k-space timings.
 * @param r0 The initial position of the object.
 * @param v The initial velocity of the object.
 * @param phi_v The encoded velocities.
 * @param phi_dB0 The field inhomogeneity.
 * @param T2 The T2 time of the blood.
 * @param profile The slice profile.
 * @return A 4D tensor representing the k-space data.
 */
template <typename T>
Tensor<std::complex<T>, 4> PC(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    // For non-cartesian acquisitions, this number should be 1
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r0.rows();

    // Nb of encoded velocities
    const uint nb_vencs = Mxy.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);
    r.noalias() = r0;

    // Kspace, Fourier exponential, and off-resonance phase
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic>  phi_off(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_nodes);

    // Iterate over kspace phase lines
    uint i, j, k, l;
    for (j = 0; j < nb_lines; j++){

      // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (i = 0; i < nb_meas; i++){

        // Iterate over slices
        for (k = 0; k < nb_kz; k++){

          // // Update blood position at time t(i,j)
          // r.noalias() = UpdatePosition(r0, v, t(i,j,k));

          // Update off-resonance phase
          phi_off.noalias() = phi_dB0 * t(i,j,k);

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate T2/T2* decay
          T2_decay = (-t(i,j,k) / T2.array()).exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_vencs; l++){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay, l);
          }
        }
      }
    }

    return kspace;
}


/**
 * @brief Computes the 3D flow image from k-space data.
 *
 * This function calculates the 3D flow image by simulating the k-space 
 * acquisition process. It takes into account the mass matrix, k-space 
 * trajectory, k-space timings, object initial position and velocity, 
 * encoded velocities, field inhomogeneity, T2 time of the blood, velocity 
 * encoding, and slice profile.
 *
 * @param MPI_rank The rank of the MPI process.
 * @param M The mass matrix.
 * @param kloc The k-space trajectory.
 * @param t The k-space timings.
 * @param r0 The initial position of the object.
 * @param v The initial velocity of the object.
 * @param phi_v The encoded velocities.
 * @param phi_dB0 The field inhomogeneity.
 * @param T2 The T2 time of the blood.
 * @param profile The slice profile.
 * @return A 4D tensor representing the k-space data.
 */
template <typename T>
Tensor<std::complex<T>, 4> PC(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy,
  const py::function &pod_trajectory
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    // For non-cartesian acquisitions, this number should be 1
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r0.rows();

    // Nb of encoded velocities
    const uint nb_vencs = Mxy.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic>  phi_off(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    float t_old = -1.0;

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_nodes);

    // Iterate over kspace readout points
    uint i, j, k, l;
    for (i = 0; i < nb_meas; i++){

      // Iterate over kspace phase lines
      for (j = 0; j < nb_lines; j++){

          // Iterate over slices
          for (k = 0; k < nb_kz; k++){

          if (t_old != t(i,j,k)){
            // Update position
            auto traj = py::cast<Matrix<T, Dynamic, Dynamic>>(pod_trajectory(t(i,j,k)));
            if (traj.rows() == 1) {
                r.noalias() = r0 + traj.colwise().replicate(r0.rows());
            } else {
                r.noalias() = r0 + traj;
            }

            // Update off-resonance phase
            phi_off.noalias() = phi_dB0 * t(i,j,k);  

            // Calculate T2/T2* decay
            T2_decay = (-t(i,j,k) / T2.array()).exp();

            // Update time
            t_old = t(i,j,k);
          } 

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_vencs; l++){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay, l);
          }
        }
      }
    }

    return kspace;
}


/**
 * @brief Computes the 3D flow image with fat signal contribution.
 * 
 * This function calculates the k-space values for a 3D flow image, taking into account the fat signal contribution. 
 * It uses the input parameters to compute the Fourier exponential, off-resonance phase, and T2* decay, and iterates 
 * over the k-space phase lines, readout points, and slices to generate the final k-space tensor.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M Sparse matrix used in the computation.
 * @param kloc Vector of 3D tensors representing k-space locations.
 * @param t Matrix of time points.
 * @param r0 Initial positions of spins.
 * @param v Velocities of spins.
 * @param phi_v Matrix of velocity encoding phases.
 * @param phi_dB0 Vector of off-resonance frequencies.
 * @param T2 T2 relaxation time.
 * @param profile Vector of profile values.
 * @param rho_w Vector of water proton densities.
 * @param rho_f Vector of fat proton densities.
 * @param chemical_shift Vector of chemical shift values.
 * @return Tensor<std::complex<T>, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
template <typename T>
Tensor<std::complex<T>, 4> PCFat(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Matrix<T, Dynamic, Dynamic> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Matrix<T, Dynamic, Dynamic> &v,
  const Matrix<T, Dynamic, Dynamic> &phi_v,
  const Vector<T, Dynamic> &phi_dB0,
  const T &T2,
  const Vector<std::complex<T>, Dynamic> &profile,
  const Vector<T, Dynamic> &rho_w,
  const Vector<T, Dynamic> &rho_f,
  const Vector<T, Dynamic> &chemical_shift  //
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r0.rows();

    // Nb of encoded velocities
    const uint nb_vencs = phi_v.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w = 1.0e+3 * nb_nodes * rho_w.replicate(1, nb_vencs).array() * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_vencs);
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_nodes, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic>  phi_off(nb_nodes);
    Vector<std::complex<T>, Dynamic> phi_fat(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    const Matrix<T, Dynamic, Dynamic> T2_decay = (-t / T2).array().exp();

    // Iterate over kspace phase lines
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (uint i = 0; i < nb_meas; ++i){

        // Update blood position at time t(i,j)
        r.noalias() =UpdatePosition(r0, v, t(i,j));

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0*t(i,j);
        phi_fat.noalias() = chemical_shift*t(i,j);
        Mxy = Mxy_w.array() + 1.0e+3 * nb_nodes * rho_f.replicate(1, nb_vencs).array() * (i1 * phi_fat.replicate(1, nb_vencs)).array().exp() * profile.replicate(1, nb_vencs).array();

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < nb_vencs; ++l){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
          }
        }
      }
    }

    return kspace;
}


/**
 * @brief Computes the 3D flow image with fat signal contribution.
 * 
 * This function calculates the k-space values for a 3D flow image, taking into account the fat signal contribution. 
 * It uses the input parameters to compute the Fourier exponential, off-resonance phase, and T2* decay, and iterates 
 * over the k-space phase lines, readout points, and slices to generate the final k-space tensor.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M Sparse matrix used in the computation.
 * @param kloc Vector of 3D tensors representing k-space locations.
 * @param t Matrix of time points.
 * @param r0 Initial positions of spins.
 * @param v Velocities of spins.
 * @param phi_v Matrix of velocity encoding phases.
 * @param phi_dB0 Vector of off-resonance frequencies.
 * @param T2 T2 relaxation time.
 * @param VENC Matrix of velocity encoding values.
 * @param profile Vector of profile values.
 * @param rho_w Vector of water proton densities.
 * @param rho_f Vector of fat proton densities.
 * @param chemical_shift Vector of chemical shift values.
 * @return Tensor<std::complex<T>, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
template <typename T>
Tensor<std::complex<T>, 4> PCFat2(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Matrix<T, Dynamic, Dynamic> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Matrix<T, Dynamic, Dynamic> &v,
  const Matrix<T, Dynamic, Dynamic> &phi_v,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &M0,
  const Vector<T, Dynamic> &T2,
  const Vector<std::complex<T>, Dynamic> &profile,
  const Vector<T, Dynamic> &rho_w,
  const Vector<T, Dynamic> &rho_f,
  const Vector<T, Dynamic> &chemical_shift  //
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r0.rows();

    // Nb of encoded velocities
    const uint nb_vencs = phi_v.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w(nb_nodes, nb_vencs), Mxy(nb_nodes, nb_vencs);
    Mxy_w = 1e+3 * nb_nodes * M0.replicate(1, nb_vencs).array() 
          * rho_w.replicate(1, nb_vencs).array()
          * (i1 * phi_v).array().exp()
          * profile.array().replicate(1, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic> phi_off(nb_nodes), phi_fat(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    Vector<T, Dynamic> T2_decay;

    // Iterate over kspace phase linesa
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (uint i = 0; i < nb_meas; ++i){

        T2_decay = (-t(i,j) / T2.array()).exp();

        // Update blood position at time t(i,j)
        r.noalias() = r0; // UpdatePosition(r0, v, t(i,j));

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0*t(i,j);
        phi_fat.noalias() = chemical_shift*t(i,j);
        Mxy = Mxy_w.array() 
            + 1.0e+3 * nb_nodes * M0.replicate(1, nb_vencs).array()
            * rho_f.replicate(1, nb_vencs).array()
            * (i1 * phi_fat.replicate(1, nb_vencs)).array().exp()
            * profile.replicate(1, nb_vencs).array();

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < nb_vencs; ++l){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
          }
        }
      }
    }

    return kspace;
}