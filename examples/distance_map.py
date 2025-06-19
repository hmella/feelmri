import cupy as cp
import numpy as np
from fimpy.solver import create_fim_solver

from feelmri.IO import XDMFFile
from feelmri.Phantom import FEMPhantom
from feelmri.FiniteElements import GradProjection

if __name__ == '__main__':

  stream = cp.cuda.stream.Stream(non_blocking=False)
  cp.show_config()

  # Navier-Stokes simulation data to be used
  phantom_file = 'phantoms/aorta_CFD.xdmf'

  # Create FEM phantom object
  phantom = FEMPhantom(path=phantom_file, velocity_label='velocity', scale_factor=0.01)

  # Read first time step
  phantom.read_data(0)

  # Get mesh points and elements
  points = phantom.mesh['nodes']
  elems = phantom.mesh['elems']
  elem_centers = np.mean(points[elems], axis=1)

  #The domain will have a small spot where movement will be slow
  ones = np.ones([elems.shape[0],])
  D = np.eye(3, dtype=np.float32)[np.newaxis] * ones[..., np.newaxis, np.newaxis]

  # Boundary points
  velocity = phantom.point_data['velocity']
  x0 = np.where(np.linalg.norm(velocity, axis=-1) == 0.0)[0]
  x0_vals = np.zeros(x0.shape[0], dtype=np.float32)

  # Create a FIM solver, by default the GPU solver will be called with the active list
  fim = create_fim_solver(points, elems, D)
  phi = fim.comp_fim(x0, x0_vals)
 
  # Distance map
  distance = phi.reshape((points.shape[0], -1))

  # Calculate gradient
  grad = GradProjection(distance, elems, points)
  grad /= np.linalg.norm(grad, axis=-1)[:, np.newaxis] 

  # Export data
  file = XDMFFile(filename='distance_map.xdmf', nodes=points, elements=({'tetra': elems}))
  file.write(pointData={'distance': distance, 'grad': grad})
  file.close()