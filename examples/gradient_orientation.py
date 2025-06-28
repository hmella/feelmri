from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import Sequence, SequenceBlock
from feelmri.MRImaging import VelocityEncoding
from feelmri.MRObjects import Gradient, Scanner
from feelmri.Parameters import ParameterHandler

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/gradient_orientation.yaml')

  # Velocity encoding parameters
  venc_dirs = list(parameters.VelocityEncoding.Directions.values())
  enc = VelocityEncoding(parameters.VelocityEncoding.VENC, np.array(venc_dirs))

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Create sequence object to show original and rotated gradients
  seq = Sequence()

  # Calculate bipolar gradients for the VENC given in the paramters file in the measurement direction (axis=0)
  time = Q_(1.0, 'ms')  # Reference time wrt the sequence
  bp1 = Gradient(scanner=scanner, time=time, axis=0)
  bp2 = bp1.make_bipolar(parameters.VelocityEncoding.VENC.to('m/s'))

  # Original block of gradients
  original = SequenceBlock([bp1, bp2])

  # Add original block to the sequence
  seq.add_block(original)

  # Add a delay between the original and rotated gradients
  seq.add_block(Q_(0.25, 'ms'))

  # Rotate the bipolar gradients to the desired direction given in the parameters file. Directions can have a norm bigger than 1 but it is normalized inside the rotate function.
  bp1r = bp1.rotate(enc.directions)
  bp2r = bp2.rotate(enc.directions)

  # Create velocity encoding gradients and rotate them
  for d in range(enc.nb_directions):

    # Update reference time for second lobe (rotate function keep the time reference of the original gradient)
    [g.change_time(bp1r[d][0].time + bp1r[d][0].dur) for g in bp2r[d]]

    # Rotated block of gradients
    rotated = SequenceBlock(gradients=bp1r[d] + bp2r[d])

    # Add rotated block to the sequence
    seq.add_block(rotated)

    # Add a delay between the original and rotated gradients
    seq.add_block(Q_(0.25, 'ms'))

  # Plot the sequence
  seq.plot()