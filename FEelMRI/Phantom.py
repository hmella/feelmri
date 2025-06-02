import time
import warnings

import meshio
import numpy as np
import pymetis
from scipy.interpolate import RBFInterpolator

from FEelMRI.FEAssemble import MassAssemble
from FEelMRI.FiniteElements import FiniteElement, QuadratureRule
from FEelMRI.MeshRefinement import refine_mesh
from FEelMRI.MPIUtilities import MPI_print, MPI_rank, MPI_size

# Define a dictionary for the element types
element_dict = {
    'triangle': 'triangle',
    'tetra': 'tetrahedron',
    'tetra10': 'quadrilateral',
}
degree_dict = {
    'triangle': 1,
    'tetra': 1,
    'tetra10': 2,
}
family_dict = {
    'triangle': 'P',
    'tetra': 'P',
    'tetra10': 'P',
}


class FEMPhantom:
  def __init__(self, path='', scale_factor=1.0, displacement_label='displacement', velocity_label='velocity', acceleration_label='acceleration', pressure_label='pressure', dtype=np.float32):
    self.path = path
    self.scale_factor = scale_factor
    self.displacement_label = displacement_label
    self.velocity_label = velocity_label
    self.acceleration_label = acceleration_label
    self.pressure_label = pressure_label
    self.dtype = dtype
    mesh, self.reader, self.Nfr = self._prepare_reader()
    self.elements = mesh['elements']
    self.all_elements = mesh['all_elements']
    self.nodes = mesh['nodes']
    self.local_elements = mesh['local_elements']
    self.local_nodes = mesh['local_nodes']
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

    # Convert nodes to given dtype
    nodes = nodes.astype(self.dtype)

    # Mesh dictionary
    mesh = {'nodes': nodes, 
            'elements': elems, 
            'all_elements': all_elems, 
            'local_elements': elems,
            'local_nodes': nodes}

    return mesh, reader, Nfr

  def distribute_mesh(self):
    # Mesh partitioning
    if MPI_size > 1:

      # Mesh partitioning
      connectivity = self.elements.tolist()
      num_parts    = MPI_size
      tpwgts = [1.0/MPI_size] * MPI_size  # Equal weights for each partition
      n_cuts, membership, vert_part = pymetis.part_mesh(num_parts, connectivity, None, tpwgts, pymetis.GType.DUAL)

      # Local elements
      local_elements_idx = np.argwhere(np.array(membership) == MPI_rank).ravel()
      local_elems = self.elements[local_elements_idx, :]

      # Local nodes
      local_nodes_idx = np.unique(local_elems)
      local_nodes = self.nodes[local_nodes_idx, :]

      # Create a mapping from old node indices to new indices
      mapped_nodes = -np.ones(self.nodes.shape[0], dtype=int)
      mapped_nodes[local_nodes_idx] = np.arange(len(local_nodes_idx))

      # Remap the element node indices to the new submesh node indices
      local_elems = mapped_nodes[local_elems]

    else:
      local_elems = self.elements
      local_nodes = self.nodes
      membership = np.zeros(self.elements.shape[0], dtype=int)

    # Debugging info
    print("Process {:d} has {:d} elements and {:d} nodes".format(MPI_rank, len(local_elems), local_nodes.shape[0]))

    # Update mesh parameters
    self.local_elements = local_elems
    self.local_nodes = local_nodes

    # Add mesh partition 
    self.partitioning = {'partitioning': membership.reshape(-1, 1)}

  def create_submesh(self, markers, refine=False, element_size=0.01):
    # Get element indexes where profile is non-zero (given a tolerance)
    submesh_elems = self.elements[markers, :]

    # Get nodes contained in profile elements 
    submesh_nodes_map = np.sort(np.unique(submesh_elems))
    submesh_nodes = self.nodes[submesh_nodes_map, :]

    # Create a mapping from old node indices to new indices
    mapped_nodes = -np.ones(self.nodes.shape[0], dtype=int)
    mapped_nodes[submesh_nodes_map] = np.arange(len(submesh_nodes_map))

    # Remap the element node indices to the new submesh node indices
    submesh_elems = mapped_nodes[submesh_elems]

    # Mesh refinement
    if refine:
      # Machine time
      t0 = time.time()

      # Debugging info
      MPI_print("[MeshRefinement] Number of elements before refining: {:d}".format(len(submesh_elems)))

      # Refine mesh
      submesh_nodes, submesh_elems = refine_mesh(submesh_nodes, submesh_elems, element_size)

      # Debugging info
      MPI_print("[MeshRefinement] Number of elements after refining: {:d}".format(len(submesh_elems)))
      MPI_print("[MeshRefinement] Elapsed time for refinement: {:.2f} s".format(time.time()-t0))

    # Update mesh parameters and backup original mesh
    self.nodes_ = self.nodes.copy()
    self.elements_ = self.elements.copy()
    self.all_elements_ = self.all_elements.copy()
    self.submesh_markers_ = markers
    self.nodes = submesh_nodes
    self.elements = submesh_elems
    # self.all_elements = {self.all_elements_[0].type: submesh_elems}
    self.submesh_nodes_map = submesh_nodes_map

  def bounding_box(self):
    ''' Calculate bounding box of the FEM geometry '''
    bmin = np.min(self.nodes, axis=0)
    bmax = np.max(self.nodes, axis=0)
    if MPI_rank == 0:
      print('Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))
    return (bmin, bmax)

  def read_data(self, fr):
    ''' Read data at frame fr '''
    d, self.point_data, self.cell_data = self.reader.read_data(fr)

    # Convert point and cell data to given dtype
    for key in self.point_data:
      self.point_data[key] = self.point_data[key].astype(self.dtype)

    # Displacement
    if self.displacement_label in self.point_data:
      self.point_data[self.displacement_label] *= self.scale_factor

    # Velocity
    if self.velocity_label in self.point_data:
      self.point_data[self.velocity_label] *= self.scale_factor

    # Acceleration
    if self.acceleration_label in self.point_data:
      self.point_data[self.acceleration_label] *= self.scale_factor**2

    # Pressure
    if self.pressure_label in self.point_data:
      self.point_data[self.pressure_label] *= 1.0


  def interpolate_to_submesh(self, data, kernel='linear', neighbors=25):
      """
      Interpolate a velocity field defined on the main mesh nodes to the submesh nodes using RBF.

      Parameters:
      - velocity_field: numpy.ndarray of shape (N, 3), velocity field on the main mesh nodes.
      - kernel: str, type of RBF kernel to use ('linear', 'thin_plate_spline', 'cubic', etc.).
      - epsilon: float, shape parameter for the RBF. If None, it will be estimated automatically.

      Returns:
      - submesh_velocity: numpy.ndarray of shape (M, 3), interpolated velocity field on the submesh nodes.
      """
      try:
        self.nodes_
      except KeyError:
        raise ValueError("Submesh not created. Please create a submesh first using `create_submesh`.")

      # Main mesh nodes and submesh nodes
      idx = self.submesh_nodes_map
      mesh_nodes = self.nodes_[idx, :]

      # Stacked data
      if data.shape[1] > 0:
        data = np.column_stack(tuple([data[...,i].flatten()[idx] for i in range(data.shape[1])]))

      # Define dummy interpolator to save time
      if hasattr(self, 'submesh_interp'):
        d_dtype = complex if np.iscomplexobj(data) else float
        data = np.asarray(data, dtype=d_dtype, order="C")
        self.submesh_interp.d = data
      else:
        self.submesh_interp = RBFInterpolator(mesh_nodes, data, neighbors=neighbors, kernel=kernel, degree=1)

      # Interpolate data
      interp_data = self.submesh_interp(self.local_nodes)

      return interp_data


  def orient(self, MPS_ori, LOC):
    ''' Translate phantom to obtain the desired slice location '''
    # Get orientation
    MPS_ori = MPS_ori.astype(self.dtype)
    LOC = LOC.astype(self.dtype)

    # Translate and rotate
    self.nodes = (self.nodes - LOC) @ MPS_ori


  def mass_matrix(self, lumped=False, use_submesh=False, quadrature_order=2):
    ''' Assemble mass matrix for integrals '''
    # Create finite element and quadrature rule according to the mesh type
    cell_type = self.all_elements[0].type
    fe = FiniteElement(family=family_dict[cell_type], 
                       cell_type=element_dict[cell_type], 
                       degree=degree_dict[cell_type], 
                       variant='equispaced')
    qr = QuadratureRule(cell_type=element_dict[cell_type], 
                        order=quadrature_order, 
                        rule='default')

    # Assemble mass matrix
    M = MassAssemble(self.local_elements, self.local_nodes, fe, qr)

    # Make matrix lumped if requested
    if lumped:
      try:
        from scipy.sparse import lil_matrix
        diag = M.sum(axis=1)
        M = lil_matrix(M.shape, dtype=M.dtype)
        M.setdiag(diag)
      except ImportError:
        if MPI_rank == 0:
          warnings.warn('Lumped mass matrix not available. Please install scipy.')

    return M