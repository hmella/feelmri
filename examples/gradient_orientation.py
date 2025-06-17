import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from FEelMRI.Bloch import BlochSolver, Sequence, SequenceBlock
from FEelMRI.IO import XDMFFile
from FEelMRI.KSpaceTraj import CartesianStack
from FEelMRI.Motion import PODVelocity
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, gather_data
from FEelMRI.MRImaging import SliceProfile, VelocityEncoding
from FEelMRI.MRObjects import RF, Gradient, Scanner
from FEelMRI.Noise import add_cpx_noise
from FEelMRI.Parameters import ParameterHandler, PVSMParser

if __name__ == '__main__':

  # Import imaging parameters
  parameters = ParameterHandler('parameters/gradient_direction.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Create velocity encoding gradients for each direction
  t_ref = Q_(0.0, 'ms')  # Reference time for gradients
  for d in range(enc.nb_directions):

    # Create sequence object and Bloch solver
    seq = Sequence()

    # Bipolar gradients
    bp1 = Gradient(scanner=scanner, t_ref=t_ref)
    bp2 = bp1.make_bipolar(parameters.VelocityEncoding.VENC.to('m/s'))

    # Rotate the bipolar gradients to the desired direction
    bp1_rotated = bp1.rotate(enc.directions[d])
    bp2_rotated = bp2.rotate(enc.directions[d])

    # Update reference time for second lobe
    [g.update_reference(g.t_ref - (bp1.dur - g.dur)) for g in bp2_rotated]

    # Imaging block
    imaging = SequenceBlock(gradients=bp1_rotated + bp2_rotated,
                            dt_gr=Q_(1e-2, 'ms'), 
                            dt=Q_(1, 'ms'), 
                            store_magnetization=False)

    # Plot sequence block
    imaging.plot()