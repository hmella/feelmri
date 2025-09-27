#include <MRI.h>
#include <pybind11/numpy.h>


/**
 * @brief Simulate and assemble the MRI k-space signal for a set of spins and encoding schemes.
 *
 * @tparam T Numeric type for spin properties and computations (e.g., float or double).
 *
 * @param MPI_rank          Rank of the current MPI process
 *                          reserved for parallel extensions).
 * @param M                 Sparse encoding matrix of size (nb_nodes × nb_vencs), 
 *                          mapping spin magnetizations to velocity encodings.
 * @param kloc              Vector of three 3D tensors (kx, ky, kz) specifying k-space 
 *                          locations for each readout, phase, and slice index.
 * @param t                 3D tensor of acquisition times (in seconds) for each 
 *                          k-space sample point [nb_meas × nb_lines × nb_kz].
 * @param r0                Initial spin positions [nb_nodes × 3] in spatial coordinates.
 * @param phi_dB0           Off‐resonance frequency per spin (in Hz), length nb_nodes.
 * @param T2                T2* relaxation time constants per spin (in seconds), length nb_nodes.
 * @param Mxy               Complex transverse magnetization for each spin and encoding 
 *                          [nb_nodes × nb_vencs].
 * @param pod_trajectory    Python callable: given a time scalar, returns a matrix [nb_nodes × 3]
 *                          of position offsets for each spin.
 *
 * @return A 4D tensor of complex k-space data with dimensions 
 *         [nb_meas × nb_lines × nb_kz × nb_vencs].
 *
 * @details 
 * For each k-space sample (i,j,k):
 *   - Updates spin positions via r = r0 + pod_trajectory(t[i,j,k])
 *   - Computes off-resonance phase term φ_off = phi_dB0 * t[i,j,k]
 *   - Applies T2* decay: exp(-t[i,j,k] / T2)
 *   - Forms the Fourier kernel exp(-i·(kx·x + ky·y + kz·z) + i·φ_off)
 *   - Multiplies by the precomputed M·Mxy encoding and accumulates into the k-space matrix
 *   - Finally maps the 2D k-space matrix into a 4D tensor for output
 */
template <typename T>
Tensor<std::complex<T>, 4, RowMajor> Signal(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Array<T, Dynamic, 1> &phi_dB0,
  const Array<T, Dynamic, 1> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy,
  const py::function &pod_trajectory
  ){

    // Define complex template datatype and complex unit
    using C = std::complex<T>;
    const C i1(0.0, 1.0);

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

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const T two_pi = 2.0 * PI;
    T kx_, ky_, kz_;

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Matrix<T, Dynamic, Dynamic> r(nb_nodes, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    Array<T, Dynamic, 1> phase(nb_nodes, 1), phi_off(nb_nodes, 1);
    Vector<C, Dynamic> fourier(nb_nodes);

    // The ‘template’ disambiguator is REQUIRED here:
    auto arr = pod_trajectory(t(0,0,0)).template cast<py::array_t<T, py::array::c_style>>();
    Map<const Matrix<T, Dynamic, 3, RowMajor>> traj(nullptr, nb_nodes, 3);

    // T2* decay
    Array<T, Dynamic, 1> T2_decay(nb_nodes, 1);
    Array<T, Dynamic, 1> inv_T2 = T2.inverse();

    // Precompute M * Mxy product
    Matrix<C, Dynamic, Dynamic> M_Mxy = M * Mxy;
    Matrix<C, Dynamic, Dynamic> M_Mxy_T2(nb_nodes, nb_vencs);

    // kspace in matrix format
    const uint S = nb_meas * nb_lines * nb_kz;
    Matrix<C, Dynamic, Dynamic, RowMajor> kspace_mat(S, nb_vencs);

    // // Release GIL for the whole numeric loop
    // py::gil_scoped_release nogil;

    // Iterate over kspace readout points
    T t_old = -1.0;
    for (uint i = 0, row = 0; i < nb_meas; i++){

      // Iterate over kspace phase lines
      for (uint j = 0; j < nb_lines; j++){

          // Iterate over slices
          for (uint k = 0; k < nb_kz; k++, row++){

            if (t_old == t(i,j,k)){
              
            } else {

              // Update position
              {
                // py::gil_scoped_acquire gil;
                arr = pod_trajectory(t(i,j,k)).template cast<py::array_t<T, py::array::c_style>>();

                // Rebind Map to the new buffer (no allocations)
                new (&traj) Eigen::Map<const Matrix<T, Dynamic, 3, RowMajor>>(arr.data(), nb_nodes, 3);
              }

              // Map without copy (NumPy is row-major/C-contiguous)
              r.noalias() = r0;
              r.noalias() += traj;

              // Update off-resonance phase
              phi_off = phi_dB0.array() * t(i,j,k);  

              // Calculate T2/T2* decay
              T2_decay = (-t(i,j,k) * inv_T2).exp();

              // M_Mxy_T2 = M_Mxy .* T2_decay (broadcast down columns)
              M_Mxy_T2.noalias() = (M_Mxy.array().colwise() * T2_decay.template cast<C>()).matrix();

              // Update time
              t_old = t(i,j,k); 

            }           

            // Scalars for this (i,j,k)
            kx_ = two_pi * kloc[0](i,j,k), ky_ = two_pi * kloc[1](i,j,k), kz_ = two_pi * kloc[2](i,j,k);

            // r: nb_nodes x 3  (column 0:x, 1:y, 2:z)
            phase = -kx_ * r.col(0).array()
                  - ky_ * r.col(1).array()
                  - kz_ * r.col(2).array()
                  + phi_off;

            // e^{i*phase} = cos + i sin
            fourier.noalias() = (phase.cos() + i1 * phase.sin()).matrix();

            // Calculate k-space values, add T2* decay, and assign value to output array
            kspace_mat.row(row).noalias() = fourier.transpose() * M_Mxy_T2;
        }
      }
    }

  // py::print(MPI_rank, 6);
  return TensorMap<Tensor<C, 4, RowMajor>>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, nb_vencs);
}



/**
 * @brief Simulate and assemble the MRI k-space signal for a set of spins and encoding schemes.
 *
 * @tparam T Numeric type for spin properties and computations (e.g., float or double).
 *
 * @param MPI_rank          Rank of the current MPI process
 *                          reserved for parallel extensions).
 * @param M                 Sparse encoding matrix of size (nb_nodes × nb_vencs), 
 *                          mapping spin magnetizations to velocity encodings.
 * @param kloc              Vector of three 3D tensors (kx, ky, kz) specifying k-space 
 *                          locations for each readout, phase, and slice index.
 * @param t                 3D tensor of acquisition times (in seconds) for each 
 *                          k-space sample point [nb_meas × nb_lines × nb_kz].
 * @param r0                Initial spin positions [nb_nodes × 3] in spatial coordinates.
 * @param phi_dB0           Off‐resonance frequency per spin (in Hz), length nb_nodes.
 * @param T2                T2* relaxation time constants per spin (in seconds), length nb_nodes.
 * @param Mxy               Complex transverse magnetization for each spin and encoding 
 *                          [nb_nodes × nb_vencs].
 * @param pod_trajectory    Python callable: given a time scalar, returns a matrix [nb_nodes × 3]
 *                          of position offsets for each spin.
 *
 * @return A 4D tensor of complex k-space data with dimensions 
 *         [nb_meas × nb_lines × nb_kz × nb_vencs].
 *
 * @details 
 * For each k-space sample (i,j,k):
 *   - Updates spin positions via r = r0 + pod_trajectory(t[i,j,k])
 *   - Computes off-resonance phase term φ_off = phi_dB0 * t[i,j,k]
 *   - Applies T2* decay: exp(-t[i,j,k] / T2)
 *   - Forms the Fourier kernel exp(-i·(kx·x + ky·y + kz·z) + i·φ_off)
 *   - Multiplies by the precomputed M·Mxy encoding and accumulates into the k-space matrix
 *   - Finally maps the 2D k-space matrix into a 4D tensor for output
 */
template <typename T>
Tensor<std::complex<T>, 4, RowMajor> Signal(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
  const Array<T, Dynamic, 1> &phi_dB0,
  const Array<T, Dynamic, 1> &T2,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy,
  const py::none &pod_trajectory
  ){

    // Define complex template datatype and complex unit
    using C = std::complex<T>;
    const C i1(0.0, 1.0);

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

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const T two_pi = 2.0 * PI;
    T kx_, ky_, kz_;

    // Kspace, Fourier exponential, and off-resonance phase
    Array<T, Dynamic, 1> phase(nb_nodes, 1), phi_off(nb_nodes, 1);
    Vector<C, Dynamic> fourier(nb_nodes);

    // T2* decay
    Array<T, Dynamic, 1> T2_decay(nb_nodes, 1);
    Array<T, Dynamic, 1> inv_T2 = T2.inverse();

    // Precompute M * Mxy product
    Matrix<C, Dynamic, Dynamic> M_Mxy = M * Mxy;
    Matrix<C, Dynamic, Dynamic> M_Mxy_T2(nb_nodes, nb_vencs);

    // kspace in matrix format
    const uint S = nb_meas * nb_lines * nb_kz;
    Matrix<C, Dynamic, Dynamic, RowMajor> kspace_mat(S, nb_vencs);

    // Iterate over kspace readout points
    T t_old = -1.0;
    for (uint i = 0, row = 0; i < nb_meas; i++){

      // Iterate over kspace phase lines
      for (uint j = 0; j < nb_lines; j++){

          // Iterate over slices
          for (uint k = 0; k < nb_kz; k++, row++){

            if (t_old == t(i,j,k)){
              
            } else {

              // Update off-resonance phase
              phi_off = phi_dB0.array() * t(i,j,k);  

              // Calculate T2/T2* decay
              T2_decay = (-t(i,j,k) * inv_T2).exp();

              // M_Mxy_T2 = M_Mxy .* T2_decay (broadcast down columns)
              M_Mxy_T2.noalias() = (M_Mxy.array().colwise() * T2_decay.template cast<C>()).matrix();

              // Update time
              t_old = t(i,j,k); 

            }           

            // Scalars for this (i,j,k)
            kx_ = two_pi * kloc[0](i,j,k), ky_ = two_pi * kloc[1](i,j,k), kz_ = two_pi * kloc[2](i,j,k);

            // r: nb_nodes x 3  (column 0:x, 1:y, 2:z)
            phase = -kx_ * r0.col(0).array()
                  - ky_ * r0.col(1).array()
                  - kz_ * r0.col(2).array()
                  + phi_off;

            // e^{i*phase} = cos + i sin
            fourier.noalias() = (phase.cos() + i1 * phase.sin()).matrix();

            // Calculate k-space values, add T2* decay, and assign value to output array
            kspace_mat.row(row).noalias() = fourier.transpose() * M_Mxy_T2;
        }
      }
    }

  // py::print(MPI_rank, 6);
  return TensorMap<Tensor<C, 4, RowMajor>>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, nb_vencs);
}