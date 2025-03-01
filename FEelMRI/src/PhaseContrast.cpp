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
    const uint nb_spins = r0.rows();

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
    Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_spins, nb_vencs);
    Mxy = 1.0e+3 * nb_spins
        * (i1 * phi_v).array().exp() 
        * profile.array().replicate(1, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic>  phi_off(nb_spins);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_spins);

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
 * @param distance Map indicating the distance from every node to the walls
 * @param outward Unit vector pointing outward to the wall.
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
  const Vector<std::complex<T>, Dynamic> &profile,
  const Vector<T, Dynamic> &distance,
  const Matrix<T, Dynamic, Dynamic> &outward
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    // For non-cartesian acquisitions, this number should be 1
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r0.rows();

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
    Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_spins, nb_vencs);
    Mxy = 1.0e+3 * nb_spins
        * (i1 * phi_v).array().exp() 
        * profile.array().replicate(1, nb_vencs);
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_(nb_spins, nb_vencs);
    Mxy_.noalias() = Mxy;
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic>  phi_off(nb_spins);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_vencs);

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_spins);

    // Displacement
    Vector<T, Dynamic> u(nb_spins);
    
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

          // Check if blood flow is outside the domain
          u = (v.array() * t(i,j,k)  * outward.array()).rowwise().sum().abs() - distance.array();
          Mxy_ = (u.array() > 0.0).select(0.0, Mxy);
          // py::print(Mxy_.array());

          // Update off-resonance phase
          phi_off.noalias() = phi_dB0 * t(i,j,k);

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate T2/T2* decay
          T2_decay = (-t(i,j,k) / T2.array()).exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_vencs; l++){
            kspace(i,j,k,l) = signal(M, Mxy_, fourier, T2_decay, l);
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
    const uint nb_spins = r0.rows();

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
    Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w = 1.0e+3 * nb_spins * rho_w.replicate(1, nb_vencs).array() * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_vencs);
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_spins, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic>  phi_off(nb_spins);
    Vector<std::complex<T>, Dynamic> phi_fat(nb_spins);

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
        Mxy = Mxy_w.array() + 1.0e+3 * nb_spins * rho_f.replicate(1, nb_vencs).array() * (i1 * phi_fat.replicate(1, nb_vencs)).array().exp() * profile.replicate(1, nb_vencs).array();

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
    const uint nb_spins = r0.rows();

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
    Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w(nb_spins, nb_vencs), Mxy(nb_spins, nb_vencs);
    Mxy_w = 1e+3 * nb_spins * M0.replicate(1, nb_vencs).array() 
          * rho_w.replicate(1, nb_vencs).array()
          * (i1 * phi_v).array().exp()
          * profile.array().replicate(1, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic> phi_off(nb_spins), phi_fat(nb_spins);

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
            + 1.0e+3 * nb_spins * M0.replicate(1, nb_vencs).array()
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


// /**
//  * @brief Computes the 3D flow image with fat signal contribution.
//  * 
//  * This function calculates the k-space values for a 3D flow image, taking into account the fat signal contribution. 
//  * It uses the input parameters to compute the Fourier exponential, off-resonance phase, and T2* decay, and iterates 
//  * over the k-space phase lines, readout points, and slices to generate the final k-space tensor.
//  * 
//  * @param MPI_rank The rank of the MPI process.
//  * @param M Sparse matrix used in the computation.
//  * @param kloc Vector of 3D tensors representing k-space locations.
//  * @param t Matrix of time points.
//  * @param r0 Initial positions of spins.
//  * @param v Velocities of spins.
//  * @param phi_v Matrix of velocity encoding phases.
//  * @param phi_dB0 Vector of off-resonance frequencies.
//  * @param T2 T2 relaxation time.
//  * @param VENC Matrix of velocity encoding values.
//  * @param profile Vector of profile values.
//  * @param rho_w Vector of water proton densities.
//  * @param rho_f Vector of fat proton densities.
//  * @param chemical_shift Vector of chemical shift values.
//  * @return Tensor<std::complex<T>, 4> The computed k-space tensor with dimensions (nb_meas, nb_lines, nb_kz, 3).
//  */
// template <typename T>
// Tensor<std::complex<T>, 5> PCFatMultiEcho(
//   const int &MPI_rank,
//   const SparseMatrix<T> &M,
//   const std::vector<Tensor<T, 3>> &kloc,
//   const std::vector<Matrix<T, Dynamic, Dynamic>> &t,
//   const Matrix<T, Dynamic, Dynamic> &r0,
//   const Matrix<T, Dynamic, Dynamic> &v,
//   const Matrix<T, Dynamic, Dynamic> &phi_v,
//   const Vector<T, Dynamic> &phi_dB0,
//   const Vector<T, Dynamic> &M0,
//   const Vector<T, Dynamic> &T2,
//   const Vector<std::complex<T>, Dynamic> &profile,
//   const Vector<T, Dynamic> &rho_w,
//   const Vector<T, Dynamic> &rho_f,
//   const Vector<T, Dynamic> &chemical_shift
//   ){

//     // Nb of measurements in the readout direction
//     const uint nb_meas = kloc[0].dimension(0);
  
//     // Nb of kspace lines/spokes/interleaves
//     const uint nb_lines = kloc[0].dimension(1);

//     // Nb of measurements in the kz direction
//     const uint nb_kz = kloc[0].dimension(2);
    
//     // Nb of spins    
//     const uint nb_spins = r0.rows();

//     // Nb of encoded velocities
//     const uint nb_vencs = phi_v.cols();

//     // Nb of echo times
//     const uint nb_echoes = t.size();

//     // Complex unit
//     const std::complex<T> i1(0.0, 1.0);

//     // Get the equivalent gradient needed to go from the center of the kspace
//     // to each location
//     const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
//     const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
//     const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

//     // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
//     Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

//     // Kspace, Fourier exponential, and off-resonance phase
//     Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w(nb_spins, nb_vencs), Mxy(nb_spins, nb_vencs);
//     Mxy_w = 1e+3 * nb_spins * M0.replicate(1, nb_vencs).array() 
//           * rho_w.replicate(1, nb_vencs).array()
//           * (i1 * phi_v).array().exp()
//           * profile.array().replicate(1, nb_vencs);
//     Vector<std::complex<T>, Dynamic> fourier(nb_spins);
//     Vector<T, Dynamic> phi_off(nb_spins), phi_fat(nb_spins);

//     // kspace
//     Tensor<std::complex<T>, 5> kspace(nb_meas, nb_lines, nb_kz, nb_vencs, nb_echoes);

//     // T2* decay
//     Vector<T, Dynamic> T2_decay;

//     // Iterate over kspace phase lines
//     for (uint j = 0; j < nb_lines; ++j){

//       // Debugging
//       // if (MPI_rank == 0){ py::print("  ky location ", j); }

//       // Iterate over kspace readout points
//       for (uint i = 0; i < nb_meas; ++i){

//         // Iterate over slices
//         for (uint k = 0; k < nb_kz; ++k){

//           T2_decay = (-t[e](i,j) / T2.array()).exp();

//           // Update blood position at time t(i,j)
//           r.noalias() = r0; // UpdatePosition(r0, v, t[e](i,j));

//           // Update off-resonance phase
//           phi_off.noalias() = phi_dB0 * t[e](i,j);
//           phi_fat.noalias() = chemical_shift * t[e](i,j);
//           Mxy = Mxy_w.array() 
//               + 1.0e+3 * nb_spins * M0.replicate(1, nb_vencs).array()
//               * rho_f.replicate(1, nb_vencs).array()
//               * (i1 * phi_fat.replicate(1, nb_vencs)).array().exp()
//               * profile.replicate(1, nb_vencs).array();

//           // Calculate Fourier exponential
//           fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

//           // Iterate over echo times
//           for (uint e = 0; e < nb_echoes; ++e){

//             // Calculate k-space values, add T2* decay, and assign value to output array
//             for (uint l = 0; l < nb_vencs; ++l){
//               kspace(i,j,k,l,e) = signal(M, Mxy, fourier, T2_decay(i,j), l);
//             }
//           }
//         }
//       }
//     }

//     return kspace;
// }


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
 * @return Matrix<std::complex<T>, Dynamic, Dynamic> The computed k-space values.
 */
template <typename T>
Matrix<std::complex<T>, Dynamic, Dynamic> PC1D(
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
    const uint nb_vencs = phi_v.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy = 1.0e+3 * nb_spins * (i1 * phi_v).array().exp() * profile.array().replicate(1, nb_vencs);
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic>  phi_off(nb_spins);

    // ComplexTensor2D kspace(nb_meas, 3);
    Matrix<std::complex<T>, Dynamic, Dynamic> kspace(nb_meas, nb_vencs);

    // T2* decay
    const Matrix<T, Dynamic, Dynamic> T2_decay = (-t / T2).array().exp();

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
      for (uint l = 0; l < nb_vencs; ++l){
        kspace(i,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
      }
    }

    return kspace;
}
