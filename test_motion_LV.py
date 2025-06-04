import time
from pathlib import Path

import numpy as np

from FEelMRI.Math import Rx, Ry, Rz
from FEelMRI.Motion import PODTrajectory
from FEelMRI.Parameters import ParameterHandler
from FEelMRI.Phantom import FEMPhantom
from FEelMRI.IO import XDMFFile

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/PARAMETERS_LV.yaml')

  # Imaging orientation paramters
  theta_x = np.deg2rad(parameters.theta_x)
  theta_y = np.deg2rad(parameters.theta_y)
  theta_z = np.deg2rad(parameters.theta_z)
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = parameters.LOC


  # Create FEM phantom object
  phantom = FEMPhantom(path='phantoms/beating_heart.xdmf', scale_factor=1.0)

  # Translate phantom to obtain the desired slice location
  phantom.orient(MPS_ori, LOC)

  # # Distribute phantom across MPI ranks
  # phantom.distribute_mesh()

  # Iterate over cardiac phases
  t0 = time.time()
  trajectory = np.zeros([phantom.nodes.shape[0], phantom.nodes.shape[1],phantom.Nfr], dtype=np.float32)
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr
    phantom.read_data(fr)
    displacement = phantom.point_data['displacement'] @ MPS_ori
    trajectory[..., fr] = displacement

  # Define POD object
  times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, phantom.Nfr)
  pod_trajectory = PODTrajectory(time_array=times,
                                 data=trajectory,
                                 n_modes=10,
                                 taylor_order=10)

  # Test class
  times = np.linspace(0, (phantom.Nfr-1)*parameters.TimeSpacing, 100)
  file = XDMFFile('test_motion.xdmf', nodes=phantom.nodes, elements=phantom.all_elements)
  for i, t in enumerate(times):
    ut = pod_trajectory(t)
    file.write(pointData={'displacement': ut}, time=t)
  file.close()