import pickle
from pathlib import Path

import numpy as np
from skimage.transform import resize

from FEelMRI.Filters import Tukey_filter
from FEelMRI.IO import VTIFile, XDMFFile
from FEelMRI.KSpaceTraj import Cartesian, Radial, Spiral
from FEelMRI.Math import Rx, Ry, Rz, itok, ktoi
from FEelMRI.Noise import add_cpx_noise
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.Plotter import MRIPlotter
from FEelMRI.Parameters import ParameterHandler

if __name__ == '__main__':

 # Import imaging parameters
  parameters = ParameterHandler('Parameters/PARAMETERS_wf.yaml')

  # Imaging orientation paramters
  theta_x = parameters.theta_x
  theta_y = parameters.theta_y
  theta_z = parameters.theta_z
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC

  # Import generated data
  im_file = Path('MRImages/{:s}_WF.pkl'.format(parameters.Sequence))
  with open(im_file, 'rb') as f:
    data = pickle.load(f)

  # Extract information from data
  K = data['kspace']

  # Fix the direction of kspace lines measured in the opposite direction
  if isinstance(data['traj'], Cartesian) and data['traj'].lines_per_shot > 1:   
    # Reorder lines depending of their readout direction
    for shot in data['traj'].shots:
      for idx, ph in enumerate(shot):

        # Readout direction
        ro = (-1)**idx

        # Reverse orientations (only when ro=-1)
        K[::ro,ph,...] = K[::1,ph,...]

  # Zero padding in the dimensions with even measurements to avoid shifts in 
  # the image domain
  if data['traj'].res[0] % 2 == 0:
    pad_width = ((0, 1), (0, 0), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # data['traj'].res[0] += 1
  if data['traj'].res[1] % 2 == 0:
    pad_width = ((0, 0), (0, 1), (0, 0), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # data['traj'].res[1] += 1
  if data['traj'].res[2] % 2 == 0:
    pad_width = ((0, 0), (0, 0), (0, 1), (0, 0))
    K = np.pad(K, pad_width, mode='constant')
    # data['traj'].res[2] += 1

  # Add noise
  K = itok(add_cpx_noise(ktoi(K, [0,1,2]), relative_std=0.01, mask=1), [0,1,2])

  # Kspace filtering (as the scanner would do)
  h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
  h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3]))
  K_fil = H*K

  # Apply the inverse Fourier transform to obtain the image
  I = ktoi(K_fil[::1,...], [0,1,2])

  # The final image can resized to achieve the desired resolution
  resized_shape = np.hstack((data['traj'].oversampling_arr*data['traj'].res, I.shape[3:]))  
  I = resize(np.real(I), resized_shape) + 1j*resize(np.imag(I), resized_shape)

  # Chop if needed
  enc_Nx = K.shape[0]
  rec_Nx = data['traj'].res[0]
  if (enc_Nx == rec_Nx):
      I = I
  else:
      ind1 = (enc_Nx - rec_Nx) // 2 #+ (data['traj'].res[0]-1 % 2 != 0)
      ind2 = (enc_Nx - rec_Nx) // 2 + rec_Nx #+ (data['traj'].res[0]-1 % 2 != 0)
      print(ind1)
      print(ind2)
      I = I[ind1:ind2,...]
  print('Image shape after correcting oversampling: ',I.shape)

  # Plot image using matplotlib plotter
  plotter = MRIPlotter(images=[np.abs(I[...,0,:]), np.angle(I[...,0,:])], FOV=parameters.FOV, title=['Magnitude', 'Phase'])
  plotter.show()

  # # Origin and pixel spacing of the generated image
  # # spacing = (data['traj'].pxsz).tolist()
  # spacing = (data['traj'].FOV/data['traj'].res).tolist()
  # origin  = (MPS_ori@(-0.5*data['traj'].FOV) + LOC).tolist()


  # # #########################################################
  # # #   Export images to vti
  # # #########################################################

  # # Create VTIFile
  # vti_file = im_file.parents[0]/('vti/' + im_file.stem + '.pvd')
  # file = VTIFile(filename=str(vti_file), origin=origin, spacing=spacing, direction=MPS_ori.flatten().tolist(), nbFrames=K.shape[-1])

  # # Get velocity and magnitude
  # v_factor = (VENC/100.0)*(1/np.pi)
  # vx = v_factor*np.angle(I[...,0,:]).copy()
  # vy = v_factor*np.angle(I[...,1,:]).copy()
  # vz = v_factor*np.angle(I[...,2,:]).copy()
  # mx = np.abs(I[...,0,:]).copy()
  # my = np.abs(I[...,1,:]).copy()
  # mz = np.abs(I[...,2,:]).copy()

  # # Estimate angiographic image
  # velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
  # angio = ( (mx + my + mz)/3 )*velocity_magnitude/K.shape[-1]

  # # Write VTI
  # file.write(cellData={'velocity_x': vx, 'velocity_y': vy, 'velocity_z': vz, 'angiography': angio, 'magnitude': mx})


  # #########################################################
  # #   Export scaled phantom to xdmf
  # #########################################################
  # # Create phantom object
  # sim_file = Path('phantoms/aorta_CFD.xdmf')
  # phantom = FEMPhantom(path=str(sim_file), scale_factor=0.01)

  # # Create XDMFFile to export scaled data
  # xdmf_file = im_file.parents[0]/'xdmf/aorta_CFD.xdmf'
  # file = XDMFFile(filename=str(xdmf_file), nodes=phantom.mesh['nodes'], elements=phantom.mesh['all_elems'])

  # # Write data
  # for fr in range(phantom.Nfr):

  #   # Read velocity at current timestep
  #   phantom.read_data(fr)

  #   # Get information from phantom
  #   velocity = phantom.velocity

  #   # Export data in the registered frame
  #   # file.write(cellData={'velocity': phantom.velocity, 'pressure': phantom.pressure}, time=fr*dt)
  #   file.write(pointData={'velocity': phantom.velocity, 'pressure': phantom.pressure}, time=fr)

  # # Close XDMFFile
  # file.close()