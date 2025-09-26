import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
import pickle
import time
from pathlib import Path

import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.IO import XDMFFile
from feelmri.KSpaceTraj import CartesianStack
from feelmri.Motion import POD, RespiratoryMotion
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRImaging import PositionEncoding, SliceProfile
from feelmri.MRObjects import RF, Gradient, Scanner
from feelmri.Parameters import ParameterHandler, PVSMParser
from feelmri.Phantom import FEMPhantom
from feelmri.Plotter import MRIPlotter
from feelmri.Recon import CartesianRecon
from feelmri.Tagging import SPAMM

if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # Import imaging parameters
  parameters = ParameterHandler(script_path/'parameters/trajectories.yaml')

  # Import PVSM file to get the FOV, LOC and MPS orientation
  planning = PVSMParser(script_path/parameters.Formatting.planning,
                          box_name='Box1',
                          transform_name='Transform1',
                          length_units=parameters.Formatting.units)

  # Create scanner object defining the gradient strength, slew rate and giromagnetic ratio
  scanner = Scanner(gradient_strength=parameters.Hardware.G_max,
                    gradient_slew_rate=parameters.Hardware.G_sr)

  # Slice profile
  # The slice profile prepulse is calculated based on a reference RF pulse with
  # user-defined characteristics. The slice profile object allows accessing the calculated adjusted RF pulse and dephasing and rephasing gradients
  rf = RF(scanner=scanner, 
          NbLobes=[4, 4], 
          alpha=0.46, 
          shape='apodized_sinc', 
          flip_angle=parameters.Imaging.FlipAngle.to('rad'))
  sp = SliceProfile(delta_z=planning.FOV[2].to('m'), 
    profile_samples=100,
    rf=rf,
    dt=Q_(1e-2, 'ms'), 
    plot=False, 
    bandwidth='maximum')
  # sp.optimize(frac_start=0.7, frac_end=0.8, N=100, profile_samples=100)

  # Create sequence object
  seq = Sequence()

  # Imaging block
  imaging = SequenceBlock(gradients=[sp.dephasing, sp.rephasing], rf_pulses=[sp.rf], dt_rf=Q_(1e-2, 'ms'), dt_gr=Q_(1e-2, 'ms'), dt=Q_(1, 'ms'), store_magnetization=True)

  # Create dummy block to reach steady state
  dummy = imaging.copy()
  dummy.store_magnetization = False

  # Add dummy blocks to the sequence to reach steady state
  time_spacing = parameters.Imaging.TimeSpacing - imaging.dur

  # Add blocks to the sequence
  seq.add_block(Q_(0.25, 'ms'))
  for fr in range(1):
    seq.add_block(imaging)
    seq.add_block(time_spacing, dt=Q_(1, 'ms'))  # Time spacing between frames
  seq.plot()

  # # Bloch solver
  # solver = BlochSolver(seq, phantom, 
  #                       scanner=scanner, 
  #                       M0=1e+9, 
  #                       T1=Q_(parameters.Phantom.T1, 'ms'), 
  #                       T2=Q_(parameters.Phantom.T2star, 'ms'), 
  #                       delta_B=delta_B0.reshape((-1, 1)),
  #                       pod_trajectory=pod_sum)

  # # Solve for x and y directions
  # Mxy, Mz = solver.solve() 

  # Generate kspace trajectory
  traj = CartesianStack(FOV=planning.FOV.to('m'),
                      t_start=sp.rephasing.dur + sp.rf.dur2,
                      res=parameters.Imaging.RES, 
                      oversampling=parameters.Imaging.Oversampling, 
                      lines_per_shot=parameters.Imaging.LinesPerShot, 
                      MPS_ori=planning.MPS,
                      LOC=planning.LOC.m,
                      receiver_bw=parameters.Hardware.r_BW.to('Hz'), 
                      plot_seq=True)
  traj.plot_trajectory()