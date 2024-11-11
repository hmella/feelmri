import cupy as cp


def PC(M, kxyz, t ,v, r0, gamma_x_delta_B0, T2, VENC, profile):
  # Number of kspace lines/spokes/interleaves
  nb_lines = kxyz[0].shape[1] # kxy[0].cols()
  
  # Number of measurements in the readout direction
  nb_meas = kxyz[0].shape[0] # kxy[0].rows()

  # Number of measurements in the kz direction
  nb_kz = kxyz[0].shape[2]

  # Number of spins
  nb_spins = r0.shape[0]

  # Get the equivalent gradient needed to go from the center of the kspace
  # to each location
  kx = 2.0 * cp.pi * kxyz[0]
  ky = 2.0 * cp.pi * kxyz[1]
  kz = 2.0 * cp.pi * kxyz[2]

  # Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
  r = cp.zeros([nb_spins, 3], dtype=cp.float32)

  # Kspace and Fourier exponential
  Mxy = 1.0e+3 * nb_spins * cp.exp(1j * cp.pi / VENC * v) * profile
  fe = cp.zeros([nb_spins, 1], dtype=cp.complex64)
  phi_off = cp.zeros([nb_spins, 1], dtype=cp.float32)

  # kspace
  kspace = cp.zeros([nb_meas, nb_lines, nb_kz, 3], dtype=cp.complex64)

  # T2* decay
  T2_decay = cp.exp(-t / T2)

  # Iterate over kspace measurements/kspace points
  for j in range(nb_lines):

    # Debugging
    print("  ky location ", j)

    # Iterate over slice kspace measurements
    for i in range(nb_meas):

      # Update blood position at time t(i,j)
      r[:,:] = r0 + v*t[i,j]

      # Update off-resonance phase
      phi_off[:,0] = gamma_x_delta_B0*t[i,j]

      for k in range(nb_kz):

        # Update Fourier exponential
        fe[:,0] = cp.exp(1j * (-r[:,0] * kx[i,j,k] - r[:,1] * ky[i,j,k] - r[:,2] * kz[i,j,k] - phi_off[:,0]))

        # Calculate k-space values, add T2* decay, and assign value to output array
        for l in range(3):
          kspace[i,j,k,l] = M.dot(Mxy[:,l]).dot(fe[:,0]) * T2_decay[i,j]

  return kspace