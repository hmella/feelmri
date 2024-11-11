#include <Signal.h>
#include <MRIEncoding.h>


Tensor<cfloat, 4> WaterFat(
  const int &MPI_rank,
  const SparseMatrix<float> &M,
  const std::vector<Tensor<float, 3>> &kloc,
  const MatrixXf &t,
  const MatrixXf &r0,
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

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<float, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<float, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<float, 3> kz = 2.0 * PI * kloc[2];

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy_w = 1.0e+3 * nb_spins * M0.array() * rho_w.array() * profile.array();
    MatrixXcf Mxy(nb_spins, 1);
    VectorXcf fourier(nb_spins);
    VectorXf  phi_off(nb_spins);
    VectorXcf phi_fat(nb_spins);

    // kspace
    Tensor<cfloat, 4> kspace(nb_meas, nb_lines, nb_kz, 1);

    // T2* decay
    VectorXf T2_decay;

    // Iterate over kspace phase linesa
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (uint i = 0; i < nb_meas; ++i){

        T2_decay = (-t(i,j) / T2.array()).exp();

        // Update off-resonance phase
        phi_off.noalias() = phi_dB0*t(i,j);
        phi_fat.noalias() = chemical_shift*t(i,j);
        Mxy = Mxy_w.array() + 1.0e+3 * nb_spins * M0.array() * rho_f.array() * (i1 * phi_fat).array().exp() * profile.array();

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          // Calculate Fourier exponential
          fourier = FourierEncoding(r0, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < 1; ++l){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay(i,j), l);
          }
        }
      }
    }

    return kspace;
}