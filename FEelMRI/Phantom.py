import meshio
import numpy as np

from FEelMRI.MPIUtilities import MPI_rank
from FEelMRI.FiniteElements import massAssemble
import warnings

class FEMPhantom:
  def __init__(self, path='', scale_factor=1.0, displacement_label='displacement', velocity_label='velocity', acceleration_label='acceleration', pressure_label='pressure'):
    self.path = path
    self.scale_factor = scale_factor
    self.displacement_label = displacement_label
    self.velocity_label = velocity_label
    self.acceleration_label = acceleration_label
    self.pressure_label = pressure_label
    self.mesh, self.reader, self.Nfr = self._prepare_reader()
    self.bbox = self.bounding_box()
    self.point_data = None
    self.cell_data = None

  def _prepare_reader(self):
    try:
      # Define reader from time series to import data
      reader = meshio.xdmf.TimeSeriesReader(self.path)

      # Import mesh
      nodes, all_elems = reader.read_points_cells()
      elems = all_elems[0].data

      # Scale mesh
      nodes *= self.scale_factor

      # Number of timesteps
      Nfr = reader.num_steps

    except Exception as e: 
      # Import mesh
      mesh = meshio.read(self.path)
      nodes = mesh.points
      elems = mesh.cells[0].data
      reader = None

      # Scale mesh
      nodes *= self.scale_factor

      # Number of timesteps
      Nfr = 1

    return {'nodes': nodes, 'elems': elems, 'all_elems': all_elems}, reader, Nfr

  def bounding_box(self):
    ''' Calculate bounding box of the FEM geometry '''
    bmin = np.min(self.mesh['nodes'], axis=0)
    bmax = np.max(self.mesh['nodes'], axis=0)
    if MPI_rank == 0:
      print('Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))
    return (bmin, bmax)

  def read_data(self, fr):
    ''' Read data at frame fr '''
    d, self.point_data, self.cell_data = self.reader.read_data(fr)

    # Displacement
    try:
      self.point_data[self.displacement_label] *= self.scale_factor
    except KeyError:
      if MPI_rank == 0:
        warnings.warn(f"Displacement label '{self.displacement_label}' not found in point data.")

    # Velocity
    try:
      self.point_data[self.velocity_label] *= self.scale_factor
    except KeyError:
      if MPI_rank == 0:
        warnings.warn(f"Velocity label '{self.velocity_label}' not found in point data.")

    # Acceleration
    try:
      self.point_data[self.acceleration_label] *= self.scale_factor**2
    except KeyError:
      if MPI_rank == 0:
        warnings.warn(f"Acceleration label '{self.acceleration_label}' not found in point data.")

    # Pressure
    try:
      self.point_data[self.pressure_label] *= 1.0
    except KeyError:
      if MPI_rank == 0:
        warnings.warn(f"Pressure label '{self.pressure_label}' not found in point data.")

  def translate(self, MPS_ori, LOC):
    ''' Translate phantom to obtain the desired slice location '''
    translated_nodes = (self.mesh['nodes'] - LOC) @ MPS_ori
    return translated_nodes

  def mass_matrix(self):
    ''' Assemble mass matrix for integrals '''
    M = massAssemble(self.mesh['elems'], self.mesh['nodes'])
    return M