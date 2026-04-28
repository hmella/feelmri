"""
Finite element mesh phantom for MRI signal simulation.

:class:`FEMPhantom` loads an FE mesh (XDMF time-series or static meshio
format), partitions it across MPI ranks using pymetis, and exposes
signal-assembly methods that call into the C++ :mod:`feelmri.MRIAssemble`
extension.
"""
import time
import warnings

import meshio
import numpy as np
import pymetis
from pint import Quantity
from scipy.interpolate import RBFInterpolator
from scipy.sparse import lil_matrix

from feelmri.Assemble import basixMassAssemble as bMassAssemble
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank, MPI_size
from feelmri.MRIAssemble import SignalAssembler

# Dictionary for pymetis ncommon (the number of common nodes that two elements must
# have in order to put an edge between them in the dual graph)
pymetis_ncommon = {
    'triangle': 2,
    'tetra': 3,
    'tetra10': 3,
    'wedge': 3,
    'hexahedron': 4
}


class FEMPhantom:
    """Finite element mesh phantom for MRI signal simulation.

    Loads an FEM geometry from an XDMF time-series file or a static meshio-
    compatible file, partitions the mesh across MPI ranks with pymetis, and
    provides methods for assembling MRI k-space signals via the C++ backend.

    Parameters
    ----------
    path : str, optional
        Path to the mesh file (XDMF time-series or any meshio-readable format).
    scale_factor : float, optional
        Uniform scale applied to node coordinates and displacement/velocity
        fields after loading. Default is 1.0.
    displacement_label : str, optional
        Key for the displacement field in the mesh point data. Default is
        ``'displacement'``.
    velocity_label : str, optional
        Key for the velocity field in the mesh point data. Default is
        ``'velocity'``.
    acceleration_label : str, optional
        Key for the acceleration field in the mesh point data. Default is
        ``'acceleration'``.
    pressure_label : str, optional
        Key for the pressure field in the mesh point data. Default is
        ``'pressure'``.
    dtype : np.dtype, optional
        Floating-point dtype used for node coordinates and field data.
        Default is ``np.float32``.
    """

    def __init__(self, path: str = '',
                 scale_factor: float = 1.0,
                 displacement_label: str = 'displacement',
                 velocity_label: str = 'velocity',
                 acceleration_label: str = 'acceleration',
                 pressure_label: str = 'pressure',
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

        # Calculate element size
        self._element_size_assembler = SignalAssembler(self.global_elements, self.global_nodes, self.cell_type, 1)
        self.global_elem_size = self._element_size_assembler.estimate_element_sizes()
        self.distribute_mesh()

    def _prepare_reader(self):
        """Open the mesh file and return the mesh dictionary, reader, and frame count.

        Returns
        -------
        tuple
            ``(mesh, reader, Nfr)`` where ``mesh`` is a dict with keys
            ``'nodes'``, ``'elements'``, and ``'cell_type'``; ``reader`` is a
            :class:`meshio.xdmf.TimeSeriesReader` or ``None`` for static
            meshes; and ``Nfr`` is the number of time steps.
        """
        try:
            # Define reader from time series to import data
            reader = meshio.xdmf.TimeSeriesReader(self.path)
            nodes, all_elems = reader.read_points_cells()
            elems = all_elems[0].data

            # Element type
            elems_type = all_elems[0].type

            # Scale mesh
            nodes *= self.scale_factor

            # Number of timesteps
            Nfr = reader.num_steps

        except Exception as e:
            # Import mesh
            if MPI_rank == 0:
                mesh = meshio.read(self.path)
                nodes = mesh.points
                elems = mesh.cells[0].data
                elems_type = mesh.cells[0].type

                # Cell and point data
                self.point_data = mesh.point_data
                self.cell_data = mesh.cell_data
            else:
                mesh = None
                nodes = None
                elems = None
                elems_type = None

                # Cell and point data
                self.point_data = None
                self.cell_data = None

            # Broadcast nodes and elements to all processes
            mesh = MPI_comm.bcast(mesh, root=0)
            nodes = MPI_comm.bcast(nodes, root=0)
            elems = MPI_comm.bcast(elems, root=0)
            elems_type = MPI_comm.bcast(elems_type, root=0)
            self.point_data = MPI_comm.bcast(self.point_data, root=0)
            self.cell_data = MPI_comm.bcast(self.cell_data, root=0)
            MPI_comm.Barrier()

            # No reader available
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
                'cell_type': elems_type}

        return mesh, reader, Nfr

    def bounding_box(self):
        """Compute the axis-aligned bounding box of the global mesh.

        Returns
        -------
        tuple
            ``(bmin, bmax)`` where each is a 3-element array of node coordinate
            extremes.
        """
        bmin = np.min(self.global_nodes, axis=0)
        bmax = np.max(self.global_nodes, axis=0)
        if MPI_rank == 0:
            print('[FEMPhantom] Bounding box: ({:f},{:f},{:f}), ({:f},{:f},{:f})'.format(
                bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]))
        return (bmin, bmax)

    def create_submesh(self, markers, refine=False):
        """Restrict the global mesh to a subset of elements defined by ``markers``.

        The original mesh is backed up and can be restored. After this call
        all operations (signal assembly, distribution) operate on the submesh.

        Parameters
        ----------
        markers : np.ndarray
            Boolean array of length ``N_elements`` selecting which elements
            to include in the submesh.
        refine : bool, optional
            Reserved for future use. Default is False.

        Notes
        -----
        This method modifies the global mesh in place. The original mesh is
        preserved in ``_global_nodes`` and ``_global_elements``.
        """
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
        self._global_elem_size = self.global_elem_size
        self._global_shape = self.global_shape

        # Update mesh parameters and backup original mesh
        self.global_nodes = submesh_nodes
        self.global_elements = submesh_elems
        self.mesh_to_submesh_nodes = submesh_nodes_map
        self.global_shape = submesh_nodes.shape

        self._element_size_assembler = SignalAssembler(self.global_elements, self.global_nodes, self.cell_type, 1)
        self.global_elem_size = self._element_size_assembler.estimate_element_sizes()

        MPI_print("[FEMPhantom] Submesh created with {:d} elements and {:d} nodes".format(
            len(self.global_elements), len(self.global_nodes)))

        # Submesh distribution
        self.distribute_mesh()

    def distribute_mesh(self):
        """Partition the global mesh across MPI ranks using pymetis.

        Partitioning is performed on rank 0 and broadcast to all ranks.
        Local node and element arrays together with local-to-global index
        maps are stored as instance attributes.
        """
        # Mesh partitioning
        connectivity = self.global_elements
        num_parts = MPI_size
        ncommon = pymetis_ncommon[self.cell_type]
        if MPI_rank == 0:
            _, membership, _ = pymetis.part_mesh(num_parts, connectivity, None, None, pymetis.GType.DUAL, ncommon)
        else:
            membership = None

        # Broadcast partitioning to all processes
        membership = MPI_comm.bcast(membership, root=0)

        # Map between local and global indices for cells. Given the a local index, it provides the corresponding global index
        l2g_cells_idx = np.argwhere(np.array(membership) == MPI_rank).ravel()

        # Local cells
        local_elems = self.global_elements[l2g_cells_idx, :]
        local_elem_size = self.global_elem_size[l2g_cells_idx]

        # Local nodes
        l2g_nodes_idx = np.unique(local_elems.flatten())
        local_nodes = self.global_nodes[l2g_nodes_idx, :]
        nb_local_nodes = local_nodes.shape[0]

        # Build global to local mapping for cells and nodes
        g2l_nodes_idx = -np.ones(self.global_nodes.shape[0], dtype=np.int32)
        g2l_nodes_idx[l2g_nodes_idx] = np.arange(nb_local_nodes)

        # Remap the element node indices to the new submesh node indices
        local_elems = g2l_nodes_idx[local_elems.flatten()].reshape((-1, local_elems.shape[1]))

        print("[FEMPhantom] Process {:d} has {:d} elements and {:d} nodes after mesh distribution".format(
            MPI_rank, len(local_elems), local_nodes.shape[0]))

        # Update mesh parameters
        self.local_elements = local_elems
        self.local_elem_size = local_elem_size
        self.local_nodes = local_nodes
        self.local_shape = local_nodes.shape

        # Add global to local mapping
        self.local_to_global_nodes = l2g_nodes_idx
        self.local_to_global_elems = l2g_cells_idx

        # Add mesh partition
        self.partitioning = np.array(membership).reshape(-1, 1)

    def gather_to_global(self, local_point_data=None, local_cell_data=None):
        """Gather local point/cell data from all MPI ranks into global arrays on rank 0.

        Parameters
        ----------
        local_point_data : dict, optional
            Dictionary of local point arrays keyed by field name.
        local_cell_data : dict, optional
            Dictionary of local cell arrays keyed by field name.

        Returns
        -------
        tuple
            ``(global_pd, global_cd)`` where each is a dict of global arrays
            on rank 0, or ``(None, None)`` on ranks > 0.
        """
        global_pd = None
        global_cd = None

        # Process Point Data
        if local_point_data is not None:
            global_pd = {}
            for key, local_array in local_point_data.items():
                # Gather all local arrays and their global indices to Rank 0
                gathered_data = MPI_comm.gather(local_array, root=0)
                gathered_indices = MPI_comm.gather(self.local_to_global_nodes, root=0)

                if MPI_rank == 0:
                    # Initialize an empty global array.
                    # E.g., if local_array is (N_local, 3), global is (N_global, 3)
                    shape = list(local_array.shape)
                    shape[0] = self.global_nodes.shape[0]
                    global_array = np.zeros(shape, dtype=local_array.dtype)

                    # Stitch the chunks into the global array using the mapping
                    for rank_data, rank_indices in zip(gathered_data, gathered_indices):
                        global_array[rank_indices] = rank_data

                    global_pd[key] = global_array

        # Process Cell Data
        if local_cell_data is not None:
            global_cd = {}
            for key, local_array in local_cell_data.items():
                gathered_data = MPI_comm.gather(local_array, root=0)
                gathered_indices = MPI_comm.gather(self.local_to_global_elems, root=0)

                if MPI_rank == 0:
                    shape = list(local_array.shape)
                    shape[0] = self.global_elements.shape[0]
                    global_array = np.zeros(shape, dtype=local_array.dtype)

                    for rank_data, rank_indices in zip(gathered_data, gathered_indices):
                        global_array[rank_indices] = rank_data

                    global_cd[key] = global_array

        # Ranks > 0 will return (None, None)
        return global_pd, global_cd

    def read_data(self, fr):
        """Read and broadcast time-series point/cell data for frame ``fr``.

        Parameters
        ----------
        fr : int
            Frame index to read from the XDMF time-series reader.
        """
        if MPI_rank == 0:
            d, p_data, c_data = self.reader.read_data(fr)
        else:
            d, p_data, c_data = None, None, None

        d = MPI_comm.bcast(d, root=0)
        p_data = MPI_comm.bcast(p_data, root=0)
        c_data = MPI_comm.bcast(c_data, root=0)

        self.point_data = p_data
        self.cell_data = c_data

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
        """Map data defined on the global mesh onto submesh nodes.

        Parameters
        ----------
        data : np.ndarray
            Array of shape ``(N_nodes, ...)`` defined on the global mesh.
        global_mesh : bool, optional
            If True, return values for all submesh nodes; if False, return
            values only for the local (MPI-rank-owned) submesh nodes.
            Default is False.

        Returns
        -------
        np.ndarray
            Data restricted to the requested submesh nodes.

        Raises
        ------
        ValueError
            If no submesh has been created, or if ``data`` shape is
            inconsistent with the mesh dimensions.
        """
        if not hasattr(self, 'mesh_to_submesh_nodes'):
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

        return data[idx, ...]

    def interpolate_to_submesh(self, data, local=True, kernel='linear', neighbors=25):
        """Interpolate a field from the main mesh to submesh nodes using RBF.

        Parameters
        ----------
        data : np.ndarray
            Source data of shape ``(N_nodes, M)`` defined on the main mesh.
        local : bool, optional
            If True, interpolate to the local (rank-owned) submesh nodes;
            if False, to all global submesh nodes. Default is True.
        kernel : str, optional
            RBF kernel passed to :class:`scipy.interpolate.RBFInterpolator`.
            Default is ``'linear'``.
        neighbors : int, optional
            Number of nearest neighbors used in the RBF. Default is 25.

        Returns
        -------
        np.ndarray
            Interpolated data at the target submesh nodes.
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
        """Transform node coordinates from phantom space to scanner image space.

        Parameters
        ----------
        MPS_ori : np.ndarray
            3×3 machine-to-patient-space rotation matrix.
        LOC : pint.Quantity
            3-element location (translation) vector with physical units.
        """
        # Get orientation
        MPS_ori = MPS_ori.astype(self.dtype)
        LOC = LOC.astype(self.dtype)

        # Translate and rotate
        self.global_nodes = (self.global_nodes - LOC.m) @ MPS_ori
        self.local_nodes = (self.local_nodes - LOC.m) @ MPS_ori

    def reorient(self, MPS_ori: np.ndarray, LOC: Quantity):
        """Undo :meth:`orient`, restoring nodes to the original phantom coordinate system.

        Parameters
        ----------
        MPS_ori : np.ndarray
            3×3 rotation matrix previously passed to :meth:`orient`.
        LOC : pint.Quantity
            3-element location vector previously passed to :meth:`orient`.
        """
        # Get orientation
        MPS_ori = MPS_ori.astype(self.dtype)
        LOC = LOC.astype(self.dtype)

        # Translate and rotate
        self.global_nodes = self.global_nodes @ MPS_ori.T + LOC.m
        self.local_nodes = self.local_nodes @ MPS_ori.T + LOC.m

    def mass_matrix(self, lumped=False, quadrature_order=2):
        """Assemble the finite element mass matrix on the local mesh partition.

        Parameters
        ----------
        lumped : bool, optional
            If True, lump the mass matrix to a diagonal by row-summing.
            Default is False.
        quadrature_order : int, optional
            Quadrature rule order for numerical integration. Default is 2.

        Returns
        -------
        scipy.sparse matrix
            Assembled (optionally lumped) mass matrix.
        """
        # Assemble mass matrix
        M = bMassAssemble(self.local_elements, self.local_nodes, self.cell_type, 'equispaced', 'default', quadrature_order)

        # Make matrix lumped if requested
        if lumped:
            diag = M.sum(axis=1)
            M = lil_matrix(M.shape, dtype=M.dtype)
            M.setdiag(diag)
        return M

    def moving_mass_matrix(self, local_nodes, lumped=False, quadrature_order=2):
        """Assemble the mass matrix using externally provided (deformed) node positions.

        Parameters
        ----------
        local_nodes : np.ndarray
            Node coordinate array for the current deformed configuration.
        lumped : bool, optional
            If True, lump the mass matrix to a diagonal. Default is False.
        quadrature_order : int, optional
            Quadrature rule order. Default is 2.

        Returns
        -------
        scipy.sparse matrix
            Assembled mass matrix for the given node positions.
        """
        # Assemble mass matrix
        M = bMassAssemble(self.local_elements, local_nodes, self.cell_type, 'equispaced', 'default', quadrature_order)

        # Make matrix lumped if requested
        if lumped:
            diag = M.sum(axis=1)
            M = lil_matrix(M.shape, dtype=M.dtype)
            M.setdiag(diag)

        return M

    def set_assembler(self, voxel_size, lorder=1, horder=1, nodal_approximation=False, lumped=True):
        """Configure the signal assembler for small and large elements separately.

        Elements smaller than ``voxel_size`` use quadrature order ``lorder``;
        larger elements use ``horder``. When ``nodal_approximation`` is True,
        a lumped mass matrix is also pre-assembled for the small elements.

        Parameters
        ----------
        voxel_size : float
            Threshold element size separating the low- and high-order groups.
        lorder : int, optional
            Quadrature order for elements with size < ``voxel_size``.
            Default is 1.
        horder : int, optional
            Quadrature order for elements with size >= ``voxel_size``.
            Default is 1.
        nodal_approximation : bool, optional
            If True, assemble and store a (optionally lumped) mass matrix for
            the small-element group. Default is False.
        lumped : bool, optional
            Lump the small-element mass matrix. Default is True.
        """
        # TODO: add option to define low order (for small elements) and high order (for large elements) quadrature rules
        small = np.where(self.local_elem_size < voxel_size)[0]
        large = np.where(self.local_elem_size >= voxel_size)[0]
        print("[Assembler] Rank {:d} has {:d}/{:d} elements with size < {:f}".format(
            MPI_rank, len(small), len(self.local_elem_size), voxel_size))
        self.assembler = []

        for d in [(small, lorder), (large, horder)]:
            size, order = d
            if np.size(size) == 0:
                continue
            self.assembler.append(SignalAssembler(self.local_elements[size, :], self.local_nodes, self.cell_type, order))

        self.nodal_approximation__ = False

        # Create mass matrix if nodal_approximation is True
        if nodal_approximation and small.size > 0:
            self.nodal_approximation__ = True

            # Assemble mass matrix
            self.M_ = bMassAssemble(self.local_elements[small, :], self.local_nodes, self.cell_type, 'equispaced', 'default', lorder)

            # Make matrix lumped if requested
            if lumped:
                diag = self.M_.sum(axis=1)
                self.M_ = lil_matrix(self.M_.shape, dtype=self.M_.dtype)
                self.M_.setdiag(diag)

            # Convert to CSR so Pybind11 can map it cleanly to Eigen::SparseMatrix in C++
            self.M_ = self.M_.tocsr()

    def update_magnetization(self, Mxy):
        """Push a new transverse magnetization array into all assembler instances."""
        for i, a in enumerate(self.assembler):
            if i == 0 and self.nodal_approximation__:
                a.update_nodal_magnetization(self.M_, Mxy)
            else:
                a.update_magnetization(Mxy)
                a.update_full_magnetization(Mxy)

    def precompute_trajectory(self, pod):
        """Pre-compute the motion trajectory for all assembler instances.

        Parameters
        ----------
        pod : POD or callable
            Motion trajectory object.

        Returns
        -------
        list
            List of pre-computed trajectory objects, one per assembler.
        """
        return [a.precompute_trajectory(pod) for a in self.assembler]

    def set_static_fields(self, T2, phi_dB0):
        """Set static relaxation and field-map arrays in all assembler instances.

        Parameters
        ----------
        T2 : np.ndarray
            T2 relaxation time map (nodal values).
        phi_dB0 : np.ndarray
            B0 field inhomogeneity phase map (nodal values).
        """
        [a.set_static_fields(T2, phi_dB0) for a in self.assembler]

    def mri_signal(self, kspace_points, kspace_times, pod=None):
        """Compute the MRI k-space signal using the configured assembler(s).

        Uses nodal integration for the first assembler group when
        ``nodal_approximation`` is active, and full Gauss integration
        otherwise.

        Parameters
        ----------
        kspace_points : np.ndarray
            K-space sample coordinates.
        kspace_times : np.ndarray
            Acquisition times corresponding to each k-space sample.
        pod : POD, list of POD, or None
            Motion trajectory. Pass a list when multiple assembler groups
            each have a pre-computed trajectory.

        Returns
        -------
        np.ndarray
            Complex k-space signal summed over all assembler groups.
        """
        # Create help to call the correct function
        eval_helper = []

        # Use enumerate to safely get the index and the object
        for i, a in enumerate(self.assembler):
            if i == 0 and self.nodal_approximation__:
                eval_helper.append(a.signal_nodal)
            else:
                eval_helper.append(a.signal)

        if isinstance(pod, list):
            return sum([signal(kspace_points, kspace_times, p) for (signal, p) in zip(eval_helper, pod)])
        else:
            return sum([signal(kspace_points, kspace_times, pod) for signal in eval_helper])

    def signal(self, kspace_points, kspace_times, pod=None):
        """Compute the k-space signal using full Gauss integration.

        Parameters
        ----------
        kspace_points : np.ndarray
            K-space sample coordinates.
        kspace_times : np.ndarray
            Acquisition times.
        pod : POD, list of POD, or None
            Motion trajectory.

        Returns
        -------
        np.ndarray
            Complex k-space signal.
        """
        # Added isinstance check to prevent crashes when passing precomputed lists
        if isinstance(pod, list):
            return sum([a.signal(kspace_points, kspace_times, p) for (a, p) in zip(self.assembler, pod)])
        else:
            return sum([a.signal(kspace_points, kspace_times, pod) for a in self.assembler])

    def signal_full(self, kspace_points, kspace_times, pod=None):
        """Compute the k-space signal using full magnetization integration.

        Parameters
        ----------
        kspace_points : np.ndarray
            K-space sample coordinates.
        kspace_times : np.ndarray
            Acquisition times.
        pod : POD, list of POD, or None
            Motion trajectory.

        Returns
        -------
        np.ndarray
            Complex k-space signal.
        """
        # Added isinstance check to prevent crashes when passing precomputed lists
        if isinstance(pod, list):
            return sum([a.signal_full(kspace_points, kspace_times, p) for (a, p) in zip(self.assembler, pod)])
        else:
            return sum([a.signal_full(kspace_points, kspace_times, pod) for a in self.assembler])

    def signal_nodal(self, kspace_points, kspace_times, pod=None):
        """Compute the k-space signal using ultra-fast nodal mass matrix integration.

        Parameters
        ----------
        kspace_points : np.ndarray
            K-space sample coordinates.
        kspace_times : np.ndarray
            Acquisition times.
        pod : POD, list of POD, or None
            Motion trajectory.

        Returns
        -------
        np.ndarray
            Complex k-space signal.
        """
        if isinstance(pod, list):
            return sum([a.signal_nodal(kspace_points, kspace_times, p) for (a, p) in zip(self.assembler, pod)])
        else:
            return sum([a.signal_nodal(kspace_points, kspace_times, pod) for a in self.assembler])

    def signal_sum(self, kspace_points, kspace_times, pod=None):
        """Compute the k-space signal by summing nodal contributions.

        Parameters
        ----------
        kspace_points : np.ndarray
            K-space sample coordinates.
        kspace_times : np.ndarray
            Acquisition times.
        pod : POD, list of POD, or None
            Motion trajectory.

        Returns
        -------
        np.ndarray
            Complex k-space signal.
        """
        if isinstance(pod, list):
            return sum([a.signal_sum(kspace_points, kspace_times, p) for (a, p) in zip(self.assembler, pod)])
        else:
            return sum([a.signal_sum(kspace_points, kspace_times, pod) for a in self.assembler])
