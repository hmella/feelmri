import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feelmri.IO import VTIFile, XDMFFile
from feelmri.Math import Rx, Ry, Rz
from feelmri.Noise import add_cpx_noise
from feelmri.Parameters import ParameterHandler
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import NUFFTRecon

if __name__ == '__main__':

 # Import imaging parameters
  parameters = ParameterHandler('parameters/aorta_slice_radial.yaml')

  # Import generated data
  im_file = Path('MRImages/{:s}_V{:.0f}_RADIAL.pkl'.format(parameters.Sequence, 100*parameters.VENC))
  with open(im_file, 'rb') as f:
    data = pickle.load(f)

  # Extract information from data and add noise
  # K = add_cpx_noise(data['kspace'], relative_std=0.01, mask=1)
  K = data['kspace']

  # Reconstruction
  I = NUFFTRecon(K, data['traj'], Jd=(4, 4, 1), iterative=True)

  d = 1
  plt.scatter(data['traj'].points[0][::d,::d,0].flatten(), data['traj'].points[1][::d,::d,0].flatten(), c=np.abs(K[::d,::d,0,0,0]).flatten(), s=1)
  plt.show()

  # Plot image using matplotlib plotter
  phi_x = np.angle(I[...,0,:] * np.conj(I[...,3,:]))
  phi_y = np.angle(I[...,1,:] * np.conj(I[...,3,:]))
  phi_z = np.angle(I[...,2,:] * np.conj(I[...,3,:]))
  phi_ref = np.angle(I[...,3,:])
  plotter = MRIPlotter(images=[np.abs(I[...,0,:]), phi_x, phi_y, phi_z, phi_ref], FOV=parameters.FOV, title=['Magnitude', '$\phi_x-\phi_{ref}$', '$\phi_y-\phi_{ref}$', '$\phi_z-\phi_{ref}$', '$\phi_{ref}$'])
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
