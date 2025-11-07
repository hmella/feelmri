import time
import warnings

import meshio
import numpy as np
import pymetis
from pint import Quantity
from scipy.interpolate import RBFInterpolator
from scipy.sparse import lil_matrix

from feelmri.Assemble import MassAssemble
from feelmri.FiniteElements import FiniteElement, QuadratureRule
from feelmri.MPIUtilities import MPI_print, MPI_rank, MPI_size

# Define a dictionary for the element types
element_dict = {
    'triangle': 'triangle',
    'tetra': 'tetrahedron',
    'tetra10': 'tetrahedron',
    'wedge': 'prism',
    'hexahedron': 'hexahedron'
}
degree_dict = {
    'triangle': 1,
    'tetra': 1,
    'tetra10': 2,
    'wedge': 1,
    'hexahedron': 1
}
family_dict = {
    'triangle': 'P',
    'tetra': 'P',
    'tetra10': 'P',
    'wedge': 'P',
    'hexahedron': 'P'
}


class FEMPhantom:
  def __init__(self, path: str = '', 
               scale_factor: float = 1.0, 
               displacement_label: str = 'displacement', 
               velocity_label: str = 'velocity', 
               acceleration_label: str = 'acceleration',
               pressure_label:str = 'pressure', 
               dtype: np.dtype = np.float32):
    self.path = path
    self.scale_factor = scale_factor
    self.displacement_label = displacement_label
    self.velocity_label = velocity_label
    self.acceleration_label = acceleration_label
    self.pressure_label = pressure_label
    self.dtype = dtype
    self.point_data = None
    self.cell_data = None
    mesh, self.reader, self.Nfr = self._prepare_reader()
    self.cell_type = mesh['cell_type']
    self.global_elements = mesh['elements']
    self.global_nodes = mesh['nodes']
    self.global_shape = self.global_nodes.shape
    self.local_elements = mesh['elements']
    self.local_nodes = mesh['nodes']
    self.local_shape = self.global_nodes.shape
    self.bbox = self.bounding_box()
    self.distribute_mesh()

  def _prepare_reader(self):
    try:
      # Define reader from time series to import data
      reader = meshio.xdmf.TimeSeriesReader(self.path)

      # Import mesh
      nodes, all_elems = reader.read_points_cells()
      elems = all_elems[0].data
      elems_type = all_elems[0].type

      # Scale mesh
      nodes *= self.scale_factor

      # Number of timesteps
      Nfr = reader.num_steps

    except Exception as e: 
      # Import mesh
      mesh = meshio.read(self.path)
      nodes = mesh.points
      elems = mesh.cells[0].data
      elems_type = mesh.cells[0].type
      reader = None

      # Cell and point data
      self.point_data = mesh.point_data
      self.cell_data = mesh.cell_data

      # Scale mesh
      nodes *= self.scale_factor

      # Number of timesteps
      Nfr = 1

    # Convert nodes to given dtype
    nodes = nodes.astype(self.dtype)

    # Mesh dictionary
    mesh = {'nodes': nodes, 
            'elements': elems, 
            'cell_type': elems_type}

    return mesh, reader, Nfr


  def create_submesh(self, markers, refine=False):
    '''
    Create a submesh from the global mesh based on the markers provided.
    Parameters
    ----------
    markers: np.ndarray
        A boolean array indicating which elements to include in the submesh.
    refine: bool, optional
        If True, the submesh will be refined. Default is False.
    element_size: float, optional
        The target size of the elements in the refined mesh. Default is 0.01.
    Returns
    -------
    None 

    Notes
    -----   
    This method modifies the global mesh to create a submesh based on the markers provided (this means that the original mesh is stored in ).
    '''

    # Get element indexes where profile is non-zero (given a tolerance)
    submesh_elems = self.global_elements[markers, :]

    # Get nodes contained in profile elements 
    submesh_nodes_map = np.unique(submesh_elems)
    submesh_nodes = self.global_nodes[submesh_nodes_map, :]

    # Create a mapping from old node indices to new indices
    mapped_nodes = -np.ones(self.global_nodes.shape[0], dtype=int)
    mapped_nodes[submesh_nodes_map] = np.arange(len(submesh_nodes_map))

    # Remap the element node indices to the new submesh node indices
    submesh_elems = mapped_nodes[submesh_elems]

    # Backup original mesh
    self._global_nodes = self.global_nodes
    self._global_elements = self.global_elements
    self._global_shape = self.global_shape

    # Update mesh parameters and backup original mesh
    self.global_nodes = submesh_nodes
    self.global_elements = submesh_elems
    self.mesh_to_submesh_nodes = submesh_nodes_map
    self.global_shape = submesh_nodes.shape

    # Debugging info
    MPI_print("[FEMPhantom] Submesh created with {:d} elements and {:d} nodes".format(len(self.global_elements), len(self.global_nodes)))

    # Submesh distribution
    self.distribute_mesh()


  def distribute_mesh(self):
    ''' Distribute mesh across MPI processes '''
    # Mesh partitioning
    connectivity = self.global_elements.tolist()
    num_parts = MPI_size
    _, membership, _ = pymetis.part_mesh(num_parts, connectivity, None, None, pymetis.GType.DUAL)

    # Map between local and global indices for cells. Given the a local index, it provides the corresponding global index
    l2g_cells_idx = np.argwhere(np.array(membership) == MPI_rank).ravel()

    # Local cells
    local_elems = self.global_elements[l2g_cells_idx, :]

    # Local nodes
    l2g_nodes_idx = np.unique(local_elems.flatten())
    local_nodes = self.global_nodes[l2g_nodes_idx, :]
    nb_local_nodes = local_nodes.shape[0]

    # Build global to local mapping for cells and nodes
    g2l_nodes_idx = -np.ones(self.global_nodes.shape[0], dtype=np.int32)
    g2l_nodes_idx[l2g_nodes_idx] = np.arange(nb_local_nodes)

    # Remap the element node indices to the new submesh node indices
    local_elems = g2l_nodes_idx[local_elems.flatten()].reshape((-1, local_elems.shape[1]))

    # Debugging info
    print("[FEMPhantom] Process {:d} has {:d} elements and {:d} nodes after mesh distribution".format(MPI_rank, len(local_elems), local_nodes.shape[0]))

    # Update mesh parameters
    self.local_elements = local_elems
    self.local_nodes = local_nodes
    self.local_shape = local_nodes.shape

    # Add global to local mapping
    self.local_to_global_nodes = l2g_nodes_idx
    self.local_to_global_elems = l2g_cells_idx

    # Add mesh partition 
    self.partitioning = np.array(membership).reshape(-1, 1)


  def bounding_box(self):
    ''' Calculate bounding box of the FEM geometry '''
    bmin = np.min(self.global_nodes, axis=0)
    bmax = np.max(self.global_nodes, axis=0)
    if MPI_rank == 0:
      print('[FEMPhantom] Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(bmin[0],bmin[1],bmin[2],bmax[0],bmax[1],bmax[2]))
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

  def to_submesh(self, data, global_mesh=False):
    """
    Convert data defined on the global mesh to the submesh nodes.

    Parameters
    ----------
    data: np.ndarray
        The data to convert, shape (N, M) where N is the number of nodes and M is the number of channels.

    Returns
    -------
    np.ndarray
        The data converted to the submesh nodes.
    """
    try:
      self.mesh_to_submesh_nodes
    except KeyError:
      raise ValueError("Submesh not created. Please create a submesh first using `create_submesh`.")

    # Verify data shape
    point_data = True
    cell_data = False
    if data.shape[0] != self._global_nodes.shape[0]:
      point_data = False
      cell_data = True
      if data.shape[0] != self.global_cells_.shape[0]:
        raise ValueError("Data shape does not match the mesh nodes or cells.")

    # Main mesh nodes and submesh nodes
    if point_data and global_mesh:
      idx = self.mesh_to_submesh_nodes
    elif point_data and not global_mesh:
      idx = self.mesh_to_submesh_nodes[self.local_to_global_nodes]
    elif cell_data:
      raise NotImplementedError("Cell data to submesh conversion is not implemented yet.")

    return data[idx,...]    

  def interpolate_to_submesh(self, data, local=True, kernel='linear', neighbors=25):
      """
      Interpolate a velocity field defined on the main mesh nodes to the submesh nodes using RBF.

      Parameters
      ----------
      data: np.ndarray
          The data to interpolate, shape (N, M) where N is the number of nodes and M is the number of channels.
      local: bool, optional
          If True, the interpolation is done only on the local CPU nodes. If False, it uses the full mesh nodes.
          Default is True.
      kernel: str, optional
          The kernel to use for the RBF interpolation. Default is 'linear'.
      neighbors: int, optional
          The number of nearest neighbors to use for the interpolation. Default is 25.
      """
      try:
        self.mesh_to_submesh_nodes
      except KeyError:
        raise ValueError("Submesh not created. Please create a submesh first using `create_submesh`.")

      # Main mesh nodes and submesh nodes
      idx = self.mesh_to_submesh_nodes
      mesh_nodes = self._global_nodes[idx, :]

      # Stacked data
      if data.shape[1] > 0:
        data = np.column_stack(tuple([data[..., i].flatten()[idx] for i in range(data.shape[1])]))

      # Define dummy interpolator to save time
      if hasattr(self, 'submesh_interp'):
        d_dtype = complex if np.iscomplexobj(data) else float
        data = np.asarray(data, dtype=d_dtype, order="C")
        self.submesh_interp.d = data
      else:
        self.submesh_interp = RBFInterpolator(mesh_nodes, data, neighbors=neighbors, kernel=kernel, degree=1)

      # Interpolate data
      if local:
        interp_data = self.submesh_interp(self.local_nodes)
      else:
        interp_data = self.submesh_interp(self.global_nodes)

      return interp_data


  def orient(self, MPS_ori: np.ndarray, LOC: Quantity):
    ''' Translate phantom image coordinate system to obtain the desired slice location '''
    # Get orientation
    MPS_ori = MPS_ori.astype(self.dtype)
    LOC = LOC.astype(self.dtype)

    # Translate and rotate
    self.global_nodes = (self.global_nodes - LOC.m) @ MPS_ori
    self.local_nodes = (self.local_nodes - LOC.m) @ MPS_ori

  def reorient(self, MPS_ori: np.ndarray, LOC: Quantity):
    ''' Translate phantom ot its original coordinate system (undo orient operation) '''
    # Get orientation
    MPS_ori = MPS_ori.astype(self.dtype)
    LOC = LOC.astype(self.dtype)

    # Translate and rotate
    self.global_nodes = self.global_nodes @ MPS_ori.T + LOC.m
    self.local_nodes = self.local_nodes @ MPS_ori.T + LOC.m


  def mass_matrix(self, lumped=False, use_submesh=False, quadrature_order=2):
    ''' Assemble mass matrix for integrals '''
    # Create finite element and quadrature rule according to the mesh type
    cell_type = self.cell_type
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
      diag = M.sum(axis=1)
      M = lil_matrix(M.shape, dtype=M.dtype)
      M.setdiag(diag)
    return M
  
  def moving_mass_matrix(self, local_nodes, lumped=False, use_submesh=False, quadrature_order=2):
    ''' Assemble mass matrix for integrals '''
    # Create finite element and quadrature rule according to the mesh type
    cell_type = self.cell_type
    fe = FiniteElement(family=family_dict[cell_type], 
                       cell_type=element_dict[cell_type], 
                       degree=degree_dict[cell_type], 
                       variant='equispaced')
    qr = QuadratureRule(cell_type=element_dict[cell_type], 
                        order=quadrature_order, 
                        rule='default')

    # Assemble mass matrix
    M = MassAssemble(self.local_elements, local_nodes, fe, qr)

    # Make matrix lumped if requested
    if lumped:
      diag = M.sum(axis=1)
      M = lil_matrix(M.shape, dtype=M.dtype)
      M.setdiag(diag)

    return M
