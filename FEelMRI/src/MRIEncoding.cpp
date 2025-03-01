#include <Signal.h>
#include <MRIEncoding.h>


template <typename T>
Tensor<std::complex<T>, 4> WaterFat(
  const int &MPI_rank,
  const SparseMatrix<T> &M,
  const std::vector<Tensor<T, 3>> &kloc,
  const Tensor<T, 3> &t,
  const Matrix<T, Dynamic, Dynamic> &r0,
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

    // Complex unit
    const std::complex<T> i1(0.0, 1.0);

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Tensor<T, 3> kx = 2.0 * PI * kloc[0];
    const Tensor<T, 3> ky = 2.0 * PI * kloc[1];
    const Tensor<T, 3> kz = 2.0 * PI * kloc[2];

    // Kspace, Fourier exponential, and off-resonance phase
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy_w = 1.0e+3 * nb_spins * M0.array() * rho_w.array() * profile.array();
    Matrix<std::complex<T>, Dynamic, Dynamic> Mxy(nb_spins, 1);
    Vector<std::complex<T>, Dynamic> fourier(nb_spins);
    Vector<T, Dynamic>  phi_off(nb_spins);
    Vector<T, Dynamic> phi_fat(nb_spins);

    // kspace
    Tensor<std::complex<T>, 4> kspace(nb_meas, nb_lines, nb_kz, 1);

    // T2* decay
    Vector<T, Dynamic> T2_decay;

    // Iterate over kspace phase linesa
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over kspace readout points
      for (uint i = 0; i < nb_meas; ++i){

        // Iterate over slices
        for (uint k = 0; k < nb_kz; ++k){

          T2_decay = (-t(i,j,k) / T2.array()).exp();

          // Update off-resonance phase
          phi_off.noalias() = phi_dB0*t(i,j,k);
          phi_fat.noalias() = chemical_shift*t(i,j,k);
          Mxy = Mxy_w.array() + 1.0e+3 * nb_spins * M0.array() * rho_f.array() * (i1 * phi_fat).array().exp() * profile.array();

          // Calculate Fourier exponential
          fourier = FourierEncoding(r0, kx, ky, kz, phi_off, i, j, k);

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < 1; ++l){
            kspace(i,j,k,l) = signal(M, Mxy, fourier, T2_decay, l);
          }
        }
      }
    }

    return kspace;
}