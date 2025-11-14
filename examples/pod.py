import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.IO import XDMFFile
from feelmri.Motion import POD
from feelmri.MPIUtilities import MPI_rank
from feelmri.MRImaging import VelocityEncoding
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

if FAST_MODE:
    Nb_frames = 1
else:
    Nb_frames = -1

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/phase_contrast.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                      box_name='Box1',
                      transform_name='Transform1',
                      length_units=parameters.Formatting.units)

  # Create FEM phantom object
  phantom = FEMPhantom(script_path/'phantoms/aorta_P1_tetra.xdmf', velocity_label='velocity', scale_factor=0.01)

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # Create array to store displacements
  v = Q_(np.zeros([phantom.global_shape[0], 3, phantom.Nfr], dtype=np.float32), 'm/s')
  for fr in range(phantom.Nfr):
    # Read displacement data in frame fr and interpolate to the submesh
    phantom.read_data(fr)
    v[..., fr] = Q_(phantom.point_data['velocity'], 'm/s')

  # Define POD object
  dt = parameters.Imaging.TimeSpacing
  times = np.linspace(0, (phantom.Nfr-1)*dt, phantom.Nfr)
  pod_velocity = POD(times=times.m_as('s'),
                    data=v.m_as('m/s'),
                    global_to_local=phantom.local_to_global_nodes,
                    n_modes=30,
                    is_periodic=True)

  # Export pod and phantom velocities to XDMF file
  Nb_frames = phantom.Nfr if FAST_MODE==False else 1
  file = XDMFFile(f'pod_{MPI_rank}.xdmf', nodes=phantom.local_nodes, elements={phantom.cell_type: phantom.local_elements})
  for fr in range(Nb_frames):
    t = fr * parameters.Imaging.TimeSpacing.m_as('s')
    file.write(pointData={'pod_velocity': pod_velocity(t), 'phantom_velocity': v[phantom.local_to_global_nodes, :, fr]}, time=t)
  file.close()