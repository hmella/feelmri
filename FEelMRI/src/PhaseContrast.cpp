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
Tensor<cfloat, 4> PC(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const VectorXf &T2,
  const VectorXcf &profile
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r0.rows();

    // Nb of encoded velocities
    const uint nb_velocities = phi_v.cols();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy = 1.0e+3 * nb_spins * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_velocities);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);

    // kspace
    Tensor<cfloat, 4> kspace(nb_meas, nb_lines, nb_kz, nb_velocities);

    // T2* decay
    // const MatrixXf T2_decay = (-t / T2).array().exp();
    VectorXf T2_decay;

    // Iterate over kspace phase lines
    uint i, j, k, l;
    for (j = 0; j < nb_lines; j++){

      // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (i = 0; i < nb_meas; i++){

        // Update blood position at time t(i,j)
        r.noalias() = UpdatePosition(r0, v, t(i,j));

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0 * t(i,j);

        // Iterate over slices
        for (k = 0; k < nb_kz; k++){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate T2/T2* decay
          T2_decay = (-t(i,j) / T2.array()).exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_velocities; l++){
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
 * @return Tensor<cfloat, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
Tensor<cfloat, 4> PCFat(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile,
  const VectorXf &rho_w,
  const VectorXf &rho_f,
  const VectorXf &chemical_shift  //
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r0.rows();

    // Nb of encoded velocities
    const uint nb_velocities = phi_v.cols();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy_w = 1.0e+3 * nb_spins * rho_w.replicate(1, nb_velocities).array() * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_velocities);
    MatrixXcf Mxy(nb_spins, nb_velocities);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);
    VectorXcf phi_fat(nb_spins);

    // kspace
    Tensor<cfloat, 4> kspace(nb_meas, nb_lines, nb_kz, nb_velocities);

    // T2* decay
    const MatrixXf T2_decay = (-t / T2).array().exp();

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
        Mxy = Mxy_w.array() + 1.0e+3 * nb_spins * rho_f.replicate(1, nb_velocities).array() * (i1 * phi_fat.replicate(1, nb_velocities)).array().exp() * profile.replicate(1, nb_velocities).array();

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < nb_velocities; ++l){
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
 * @return Tensor<cfloat, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
 */
Tensor<cfloat, 4> PCFat2(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const VectorXf &M0,
  const VectorXf &T2,
  const VectorXcf &profile,
  const VectorXf &rho_w,
  const VectorXf &rho_f,
  const VectorXf &chemical_shift  //
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r0.rows();

    // Nb of encoded velocities
    const uint nb_velocities = phi_v.cols();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy_w = 1.0e+3 * nb_spins * M0.replicate(1, nb_velocities).array() * rho_w.replicate(1, nb_velocities).array() * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_velocities);
    MatrixXcf Mxy(nb_spins, nb_velocities);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);
    VectorXcf phi_fat(nb_spins);

    // kspace
    Tensor<cfloat, 4> kspace(nb_meas, nb_lines, nb_kz, nb_velocities);

    // T2* decay
    VectorXf T2_decay;

    // Iterate over kspace phase linesa
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (uint i = 0; i < nb_meas; ++i){

        T2_decay = (-t(i,j) / T2.array()).exp();

        // Update blood position at time t(i,j)
        r.noalias() = r0; // UpdatePosition(r0, v, t(i,j));

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0*t(i,j);
        phi_fat.noalias() = chemical_shift*t(i,j);
        Mxy = Mxy_w.array() + 1.0e+3 * nb_spins * M0.replicate(1, nb_velocities).array() * rho_f.replicate(1, nb_velocities).array() * (i1 * phi_fat.replicate(1, nb_velocities)).array().exp() * profile.replicate(1, nb_velocities).array();

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < nb_velocities; ++l){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
          }
        }
      }
    }

    return kspace;
}

/**
 * @brief Computes the 1D flow image for a given slice and phase line.
 * 
 * This function calculates the k-space values for a given slice and phase line
 * by updating the blood position, off-resonance phase, and Fourier exponential
 * at each readout point. It also applies T2* decay to the computed k-space values.
 * 
 * @param MPI_rank The rank of the MPI process.
 * @param M A sparse matrix representing the encoding matrix.
 * @param kloc A vector of 3D tensors representing the k-space locations.
 * @param t A matrix representing the time points.
 * @param r0 A matrix representing the initial positions of the spins.
 * @param v A matrix representing the velocities of the spins.
 * @param phi_v A matrix representing the phase encoding velocities.
 * @param phi_dB0 A vector representing the off-resonance frequencies.
 * @param T2 The T2 relaxation time constant.
 * @param VENC A matrix representing the velocity encoding.
 * @param profile A vector representing the profile.
 * @param j The phase line index.
 * @param k The slice index.
 * @return MatrixXcf The computed k-space values.
 */
MatrixXcf PC1D(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
  const MatrixXf &v,
  const MatrixXf &phi_v,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile,
  const int &j,
  const int &k
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r0.rows();

    // Nb of encoded velocities
    const uint nb_velocities = phi_v.cols();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy = 1.0e+3 * nb_spins * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_velocities);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);

    // ComplexTensor2D kspace(nb_meas, 3);
    MatrixXcf kspace(nb_meas, nb_velocities);

    // T2* decay
    const MatrixXf T2_decay = (-t / T2).array().exp();

    // // Debugging
    // if (MPI_rank == 0){ py::print("  ky location ", j); }

    // Iterate over kspace readout points
    for (uint i = 0; i < nb_meas; ++i){

      // Update blood position at time t(i,j)
      r.noalias() = UpdatePosition(r0, v, t(i,j));

      // Update off-resonance phase
      phi_off.noalias() = phi_dB0*t(i,j);

      // Calculate Fourier exponential
      fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

      // Calculate k-space values, add T2* decay, and assign value to output array
      for (uint l = 0; l < nb_velocities; ++l){
        kspace(i,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
      }
    }

    return kspace;
}
