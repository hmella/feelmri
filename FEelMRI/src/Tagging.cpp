#include <Signal.h>
#include <Tagging.h>

template <typename T>
Tensor<std::complex<T>, 4> SPAMM(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r,
  const Matrix<std::complex<T>, Dynamic, Dynamic> &M_spamm,
  const Vector<T, Dynamic> &phi_dB0,
  const Vector<T, Dynamic> &T2,
  const Vector<std::complex<T>, Dynamic> &profile
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_nodes = r.rows();

    // Nb of encoded velocities
    const uint nb_enc = M_spamm.cols();

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy;
    Mxy = M_spamm.array(); // * profile.array().replicate(1, nb_enc);
    Vector<std::complex<T>, Dynamic> fourier(nb_nodes);
    Vector<T, Dynamic>  phi_off(nb_nodes);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_enc);

    // T2* decay
    Vector<T, Dynamic> T2_decay(nb_nodes);

    // Iterate over kspace phase lines
    uint i, j, k, l;
    for (j = 0; j < nb_lines; j++){

      // // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (i = 0; i < nb_meas; i++){

        // Iterate over slices
        for (k = 0; k < nb_kz; k++){

          // Update off-resonance phase
          phi_off.noalias() = phi_dB0 * t(i,j,k);

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate T2/T2* decay
          T2_decay = (-t(i,j,k) / T2.array()).exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_enc; l++){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay, l);
          }
        }
      }
    }

    return kspace;
}
