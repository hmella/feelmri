#include <Signal.h>
#include <Tagging.h>


Tensor<std::complex<float>, 4> SPAMM(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r,
  const MatrixXcf &M_spamm,
  const VectorXf &phi_dB0,
  const float &T2,
  const VectorXcf &profile
  ){

    // Nb of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);
  
    // Nb of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);

    // Nb of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);
    
    // Nb of spins    
    const uint nb_spins = r.rows();

    // Nb of encoded velocities
    const uint nb_enc = M_spamm.cols();

    // Complex unit
    const std::complex<float> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy = 1.0e+3 * nb_spins * M_spamm.array() * profile.array().replicate(1, nb_enc);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);

    // kspace
    Tensor<std::complex<float>, 4> kspace(nb_meas, nb_lines, nb_kz, nb_enc);

    // T2* decay
    const MatrixXf T2_decay = (-t / T2).array().exp();

    // Iterate over kspace phase lines
    uint i, j, k, l;
    for (j = 0; j < nb_lines; j++){

      // // Debugging
      // if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (i = 0; i < nb_meas; i++){

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0 * t(i,j);

        // Iterate over slices
        for (k = 0; k < nb_kz; k++){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (l = 0; l < nb_enc; l++){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
          }
        }
      }
    }

    return kspace;
}
