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

from contextlib import contextmanager
from feelmri.Assemble import basixMassAssemble as bMassAssemble
from feelmri.MRIAssemble import quadrature_npoints
from mpi4py import MPI
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

    def _cached_mode_arrays(self, pod):
        """C-contiguous ``modes_x/y/z`` for this trajectory and active partition.

        The cache holds a reference to ``pod`` so its ``id`` cannot be reused, and
        holds the arrays themselves so their addresses stay stable for the phantom's
        lifetime, which the assembler's pointer-keyed quadrature-mode cache requires.
        """
        cache = self.__dict__.setdefault('_mode_array_cache', {})
        n_nodes = self.local_nodes.shape[0]
        key = (id(pod), getattr(self, '_active_partition', None), n_nodes)
        hit = cache.get(key)
        if hit is None:
            if getattr(self, '_dual', False) and self._active_partition == 'signal':
                modes = self._signal_modes(pod)
            else:
                modes = pod.get_modes(n_nodes)
            hit = (pod,
                   np.ascontiguousarray(modes[:, 0, :]),
                   np.ascontiguousarray(modes[:, 1, :]),
                   np.ascontiguousarray(modes[:, 2, :]))
            cache[key] = hit
        return hit[1], hit[2], hit[3]

    def _prepare_pod_data(self, kspace_times, pod):
        """Helper to extract and format POD data for zero-copy C++ execution."""
        has_traj = pod is not None
        
        if has_traj:
            # Flatten time and find unique timestamps to avoid redundant evaluations
            t_flat = kspace_times.flatten()
            unique_times, inv_indices = np.unique(t_flat, return_inverse=True)
            unique_weights = pod.get_weights(unique_times)
            weights = unique_weights[inv_indices] 
            
            # Static modes, cached per (trajectory, partition).
            #
            # The assembler caches the S_global_ * modes product keyed on
            # modes_x.data(). A freshly allocated array here can reuse a previously
            # freed address, which either invalidates the cache on every call or
            # matches it against different modes. Holding the buffers keeps the
            # address stable and avoids three array copies per call.
            modes_x, modes_y, modes_z = self._cached_mode_arrays(pod)

            # Force 2D C-contiguous layout for PyBind11
            total_modes = modes_x.shape[1]
            weights = np.ascontiguousarray(weights.reshape(-1, total_modes), dtype=np.float32)
        else:
            # Empty dummies
            weights = np.empty((0, 0), dtype=np.float32)
            modes_x = np.empty((0, 0), dtype=np.float32)
            modes_y = np.empty((0, 0), dtype=np.float32)
            modes_z = np.empty((0, 0), dtype=np.float32)

        t_array_cpp = np.ascontiguousarray(kspace_times, dtype=np.float32)
        
        return t_array_cpp, modes_x, modes_y, modes_z, weights, has_traj

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

    def distribute_mesh(self, graph_type: str = 'nodal', elem_weights=None,
                        overdecompose: int = 2, node_weight: float = 1.0):
        """Partition the global mesh across MPI ranks using pymetis.

        Partitioning is performed on rank 0 and broadcast to all ranks.
        Local node and element arrays together with local-to-global index
        maps are stored as instance attributes.

        Parameters
        ----------
        graph_type : {'nodal', 'dual'}, optional
            Which METIS graph to balance when ``elem_weights`` is None.
            ``'nodal'`` (default) balances the nodal graph, ``'dual'``
            balances the element graph.
        elem_weights : array_like of float, optional
            Per-element cost, in the same units as a node, for joint-cost
            partitioning. When given, the mesh is overdecomposed and the
            chunks are bin-packed so as to minimise the *maximum* of
            ``n_local_nodes + sum(elem_weights)`` over ranks, rather than
            balancing either quantity on its own. See
            :meth:`quadrature_cost_weights` and :meth:`enable_dual_partition`.
        overdecompose : int, optional
            Number of chunks per rank in the joint-cost path. Default 2;
            larger values balance better in ratio but duplicate more boundary
            nodes and raise the absolute cost. Ignored when
            ``elem_weights`` is None.

        Notes
        -----
        The Bloch solve costs O(nodes) per rank, and a rank owns every node
        touched by any of its elements -- so nodes on a partition boundary
        are carried by every rank that touches them, and a partition with a
        large surface-to-volume ratio ends up with more nodes per element.
        Balancing the DUAL graph equalises *elements* and lets that spread
        through: on the 173 475-node heart submesh at 8 ranks it gives
        ``max/mean = 1.107`` in nodes. Balancing the NODAL graph instead
        gives ``1.020`` (max 24 875 -> 23 479 nodes, a 5.6% shorter critical
        path) at the same partitioning cost and with element balance
        essentially unchanged (1.021 -> 1.028), so it is the default.

        Rebalancing the DUAL partition afterwards by moving boundary
        elements does *not* work and was tried: every element moved across a
        boundary tends to add as many nodes to the receiving rank as it frees
        from the donor, so the total node count inflates and ``max`` barely
        moves. METIS's own partition is already a local optimum for
        ``max(node_count)``; the fix has to be in the objective it optimises,
        which is what ``graph_type`` selects.
        """
        # Mesh partitioning
        connectivity = self.global_elements
        num_parts = MPI_size
        gtype = str(graph_type).lower()
        if gtype not in ('nodal', 'dual'):
            raise ValueError(
                f"distribute_mesh: graph_type must be 'nodal' or 'dual'; got {graph_type!r}")
        if MPI_rank == 0:
            if elem_weights is not None:
                membership = self._joint_cost_partition(
                    num_parts, np.asarray(elem_weights, dtype=np.float64),
                    int(overdecompose), node_weight=float(node_weight))
            elif gtype == 'nodal':
                _, membership, _ = pymetis.part_mesh(
                    num_parts, connectivity, None, None, pymetis.GType.NODAL)
            else:
                ncommon = pymetis_ncommon[self.cell_type]
                _, membership, _ = pymetis.part_mesh(
                    num_parts, connectivity, None, None, pymetis.GType.DUAL, ncommon)
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
        self._local_to_global_nodes = l2g_nodes_idx
        self.local_to_global_elems = l2g_cells_idx

        # Add mesh partition
        self.partitioning = np.array(membership).reshape(-1, 1)

        # The global -> local node map is the scatter map any redistribution between
        # two partitions needs, so keep it rather than discarding it.
        self._g2l_nodes = g2l_nodes_idx

        # What objective this partition optimises, so callers can reuse it instead
        # of recomputing an identical one.
        self._partition_kind = 'weighted' if elem_weights is not None else gtype

        # Nothing is bound to this partition yet. Anything that captures
        # `local_to_global_nodes` or `local_nodes` must set this, so a later
        # repartition can refuse rather than silently invalidate it.
        self._partition_bound = False

    # ------------------------------------------------------------------
    # Multiple simultaneous partitions
    # ------------------------------------------------------------------

    _PARTITION_ATTRS = ('local_elements', 'local_elem_size', 'local_nodes',
                        'local_shape', '_local_to_global_nodes',
                        'local_to_global_elems', 'partitioning', '_g2l_nodes')

    def _capture_partition(self):
        """Snapshot the flat attributes that constitute the current partition."""
        return {a: getattr(self, a) for a in self._PARTITION_ATTRS}

    def _install_partition(self, state):
        for a, v in state.items():
            object.__setattr__(self, a, v)

    def activate(self, name):
        """Make a named partition the live one.

        The mesh state is eight flat attributes rather than an object, so switching
        partitions means swapping all of them at once. Everything downstream keeps
        reading ``self.local_nodes`` and friends unchanged.
        """
        if name not in self._partitions:
            raise KeyError(f"activate: no partition named {name!r}; "
                           f"have {sorted(self._partitions)}")
        if getattr(self, '_active_partition', None) == name:
            return
        if getattr(self, '_active_partition', None) is not None:
            self._partitions[self._active_partition] = self._capture_partition()
        self._install_partition(self._partitions[name])
        self._active_partition = name

    @contextmanager
    def _using(self, name):
        """Temporarily activate a partition, restoring the previous one after."""
        if not getattr(self, '_partitions', None) or name not in self._partitions:
            yield
            return
        previous = self._active_partition
        self.activate(name)
        try:
            yield
        finally:
            if previous is not None:
                self.activate(previous)

    @property
    def local_to_global_nodes(self):
        """Global index of each local node.

        Reading this binds the caller to the current partition -- it is what a POD
        trajectory captures via ``global_to_local`` -- so a later repartition would
        silently invalidate whatever was built from it. Accessing it therefore marks
        the partition as in use, and :meth:`enable_dual_partition` then refuses.
        """
        self._partition_bound = True
        return self._local_to_global_nodes

    @local_to_global_nodes.setter
    def local_to_global_nodes(self, value):
        self._local_to_global_nodes = value

    # ------------------------------------------------------------------
    # Redistribution of nodal data between two partitions
    # ------------------------------------------------------------------

    def _canonical_owner(self, partitioning=None):
        """Global node -> the single rank responsible for it: the lowest-numbered
        rank whose elements touch it.

        Node ownership is non-exclusive -- a rank owns every node its elements
        touch -- so interface nodes live on several ranks. Picking one canonical
        owner is exact wherever the duplicated values are identical, which they are:
        the Bloch kernel is a pure per-node function of per-node inputs.
        """
        part = (np.asarray(self.partitioning).ravel() if partitioning is None
                else np.asarray(partitioning).ravel())
        elems = self.global_elements
        owner = np.full(self.global_nodes.shape[0], MPI_size, dtype=np.int32)
        for r in range(MPI_size):
            g = np.unique(elems[part == r].ravel())
            np.minimum.at(owner, g, r)
        return owner

    def _node_ownership_mask(self):
        """1.0 for nodes this rank owns, 0.0 for those it merely touches.

        Cached per partition -- it depends only on the partition, not on the data.
        """
        cache = self.__dict__.setdefault('_own_mask_cache', {})
        key = getattr(self, '_active_partition', None)
        if key not in cache:
            owner = self._canonical_owner()
            mask = (owner[self._local_to_global_nodes] == MPI_rank)
            cache[key] = np.ascontiguousarray(mask.astype(np.float32))
        return cache[key]

    def _redistribution_schedule(self, src, dst):
        """Send/receive plan moving nodal data from partition ``src`` to ``dst``.

        Built once per pair and cached. Needs **no communication**: ``partitioning``
        is broadcast to every rank, so each rank can derive the whole plan locally.

        Node ownership is non-exclusive -- a rank owns every node its elements touch --
        so ``sum_r N_r > N_global`` and this is not a permutation. Each global node is
        assigned a single canonical sender (the lowest-ranked owner in ``src``), which
        is exact because the Bloch kernel is a pure per-node function, so every rank
        holding a duplicated node holds bit-identical values.
        """
        key = (src, dst)
        cache = self.__dict__.setdefault('_redist_cache', {})
        if key in cache:
            return cache[key]

        part_src = np.asarray(self._partitions[src]['partitioning']).ravel()
        part_dst = np.asarray(self._partitions[dst]['partitioning']).ravel()
        l2g_src = self._partitions[src]['_local_to_global_nodes']
        l2g_dst = self._partitions[dst]['_local_to_global_nodes']
        elems = self.global_elements
        n_global = self.global_nodes.shape[0]

        owner = self._canonical_owner(part_src)

        mine = l2g_src[owner[l2g_src] == MPI_rank]     # nodes this rank is responsible for
        send_idx = []
        for r in range(MPI_size):
            needed = np.unique(elems[part_dst == r].ravel())
            send_idx.append(np.intersect1d(mine, needed, assume_unique=False))

        scounts = np.array([len(x) for x in send_idx], dtype=np.int32)
        rcounts = np.empty(MPI_size, dtype=np.int32)
        MPI_comm.Alltoall(scounts, rcounts)
        sdisp = np.insert(np.cumsum(scounts), 0, 0)[:-1].astype(np.int32)
        rdisp = np.insert(np.cumsum(rcounts), 0, 0)[:-1].astype(np.int32)

        send_g = (np.concatenate(send_idx) if send_idx else
                  np.empty(0, dtype=np.int64)).astype(np.int64)
        recv_g = np.empty(int(rcounts.sum()), dtype=np.int64)
        MPI_comm.Alltoallv([send_g, (scounts, sdisp), MPI.INT64_T],
                           [recv_g, (rcounts, rdisp), MPI.INT64_T])

        src_rows = self._partitions[src]['_g2l_nodes'][send_g].astype(np.int64)
        dst_rows = self._partitions[dst]['_g2l_nodes'][recv_g].astype(np.int64)

        # A destination node no canonical sender covers would be a silent wrong
        # answer, not a crash -- so check rather than trust.
        if dst_rows.size != l2g_dst.size or np.unique(dst_rows).size != l2g_dst.size:
            raise RuntimeError(
                f"redistribution {src!r} -> {dst!r} does not cover rank {MPI_rank}: "
                f"{np.unique(dst_rows).size} of {l2g_dst.size} destination nodes filled")
        if src_rows.size and src_rows.min() < 0:
            raise RuntimeError(f"redistribution {src!r} -> {dst!r}: sender does not own "
                               "a node it was scheduled to send")

        cache[key] = (scounts, sdisp, rcounts, rdisp, src_rows, dst_rows)
        return cache[key]

    def redistribute_nodal(self, array, src='bloch', dst='signal'):
        """Move a per-node array from one partition's layout to another's.

        Parameters
        ----------
        array : np.ndarray
            ``(n_local_src,)`` or ``(n_local_src, ncols)``, any dtype MPI can map.

        Returns
        -------
        np.ndarray
            The same data in ``dst``'s node ordering, shape ``(n_local_dst, ...)``.
        """
        scounts, sdisp, rcounts, rdisp, src_rows, dst_rows = \
            self._redistribution_schedule(src, dst)

        a = np.asarray(array)
        flat = a.reshape(a.shape[0], -1)
        ncols = flat.shape[1]
        mpi_t = MPI._typedict[np.dtype(a.dtype).char]

        sendbuf = np.ascontiguousarray(flat[src_rows])
        recvbuf = np.empty((dst_rows.size, ncols), dtype=a.dtype)
        MPI_comm.Alltoallv(
            [sendbuf, (scounts * ncols, sdisp * ncols), mpi_t],
            [recvbuf, (rcounts * ncols, rdisp * ncols), mpi_t])

        n_dst = self._partitions[dst]['_local_to_global_nodes'].size
        out = np.empty((n_dst, ncols), dtype=a.dtype)
        out[dst_rows] = recvbuf
        return out.reshape((n_dst,) + a.shape[1:])

    def _dual_graph(self):
        """Element adjacency CSR: elements sharing at least ``ncommon`` nodes.

        Built generically from the element-node incidence (``A = C @ C.T``) so it
        works for every supported cell type, reusing the same ``ncommon`` values
        METIS uses for its own dual graph.
        """
        from scipy.sparse import csr_matrix

        ne = self.global_elements.shape[0]
        nn = self.global_nodes.shape[0]
        ncommon = pymetis_ncommon[self.cell_type]

        rows = np.repeat(np.arange(ne), self.global_elements.shape[1])
        C = csr_matrix((np.ones(self.global_elements.size, np.int8),
                        (rows, self.global_elements.ravel())), shape=(ne, nn))
        A = (C @ C.T).tocoo()
        keep = (A.data >= ncommon) & (A.row != A.col)
        A = csr_matrix((np.ones(int(keep.sum()), np.int8), (A.row[keep], A.col[keep])),
                       shape=(ne, ne))
        return A.indptr.astype(np.int64), A.indices.astype(np.int64)

    def _chunk_weights(self, elem_weights, node_weight=1.0):
        """Integer per-element weights for the overdecomposition step.

        The blended cost is O(1) in node-equivalents and METIS takes only integers,
        so rounding it directly collapses the cheap and expensive element classes
        onto the same value and discards the cost distinction. Rescale first.
        """
        nodes_per_elem = self.global_elements.shape[1] / (self.global_elements.size
                                                          / self.global_nodes.shape[0])
        w = node_weight * nodes_per_elem + np.asarray(elem_weights, dtype=np.float64)
        w_min = float(w.min())
        scale = 1024.0 / w_min if w_min > 0 else 1024.0
        return np.maximum(np.round(w * scale), 1).astype(np.int64)

    def _joint_cost_partition(self, num_parts, elem_weights, overdecompose,
                              node_weight=1.0):
        """Partition minimising ``max_r (n_nodes_r + sum(elem_weights)_r)``.

        Balancing nodes and element cost as two separate constraints -- whether by
        two partitions or by METIS multi-constraint -- minimises a *sum of maxima*.
        The quantity that actually sets the wall time is the *maximum of sums*, which
        is never larger and is strictly smaller when the two costs anti-correlate
        across the mesh. So the mesh is overdecomposed and the chunks are packed
        directly against that objective.

        Node counts are unioned, never summed: a rank owns every node its elements
        touch, so two adjacent chunks share boundary nodes and cost less together
        than two distant ones. That is what keeps the packing spatially coherent.

        ``node_weight`` scales the node term. It is 1.0 for a joint partition, and
        **0.0 when the phase being balanced does no per-node work** -- the dual
        scheme's signal layout with ``nodal_approximation=False``, where the cost is
        purely ``O(quadrature points)``. Leaving it at 1.0 there lets the node term
        dominate and the quadrature balance barely improves.
        """
        if num_parts == 1:
            return np.zeros(self.global_elements.shape[0], dtype=np.int64)

        ne = self.global_elements.shape[0]
        xadj, adjncy = self._dual_graph()
        adj = pymetis.CSRAdjacency(xadj, adjncy)

        # Chunk with the blended weight so the pieces are already roughly right;
        # the packing below fixes what a scalar weight cannot express.
        _, sub = pymetis.part_graph(num_parts * overdecompose, adjacency=adj,
                                    vweights=self._chunk_weights(elem_weights,
                                                                 node_weight))
        sub = np.asarray(sub)

        chunks = []
        for c in range(num_parts * overdecompose):
            idx = np.flatnonzero(sub == c)
            if idx.size:
                chunks.append((np.unique(self.global_elements[idx].ravel()),
                               float(elem_weights[idx].sum()), idx))
        # Heaviest first: a first-fit descending pass, standard for bin packing.
        chunks.sort(key=lambda c: node_weight * len(c[0]) + c[1], reverse=True)

        rank_nodes = [set() for _ in range(num_parts)]
        rank_cost = np.zeros(num_parts)
        membership = np.zeros(ne, dtype=np.int64)
        for nodes, w, idx in chunks:
            nodeset = set(nodes.tolist())
            best, best_cost = 0, None
            for r in range(num_parts):
                cost = node_weight * len(rank_nodes[r] | nodeset) + rank_cost[r] + w
                if best_cost is None or cost < best_cost:
                    best, best_cost = r, cost
            rank_nodes[best] |= nodeset
            rank_cost[best] += w
            membership[idx] = best
        return membership

    def enable_dual_partition(self, voxel_size, lorder=1, horder=1,
                              nodal_approximation=False, lumped=True,
                              cost_ratio=47.0, overdecompose=2):
        """Build two partitions once: one for the Bloch solve, one for k-space.

        ``'bloch'`` balances nodes (the Bloch solve and the nodal signal paths cost
        O(nodes)); ``'signal'`` balances quadrature cost. The assembler groups are
        built on ``'signal'`` only -- ``'bloch'`` never needs them -- and ``'bloch'``
        is left active, so the solver and the POD bind to it as usual.

        Nodal data is moved between the two by :meth:`redistribute_nodal`, which the
        signal path invokes automatically. Static fields and POD modes move once at
        setup; only ``Mxy`` moves per handoff.

        Call this **instead of** :meth:`set_assembler`, and before constructing the
        POD trajectory or the BlochSolver.

        """
        if getattr(self, '_partition_bound', False):
            raise RuntimeError(
                "enable_dual_partition: the current partition is already in use. "
                "Call it BEFORE constructing the POD trajectory and the BlochSolver.")

        self._partitions = {}
        self._active_partition = None
        self.__dict__.pop('_redist_cache', None)

        # 'bloch': node-balanced. create_submesh() and __init__ both end in a NODAL
        # distribute_mesh, so the live partition usually already IS this one --
        # recomputing it would run METIS again for an identical result.
        if getattr(self, '_partition_kind', None) != 'nodal':
            self.distribute_mesh(graph_type='nodal')
        self._partitions['bloch'] = self._capture_partition()

        # 'signal': quadrature-cost-balanced.
        weights = self.quadrature_cost_weights(
            voxel_size, lorder, horder, cost_ratio,
            nodal_approximation=nodal_approximation)
        # With nodal_approximation the small-element group runs through signal_nodal,
        # whose cost is O(nodes) -- so the node term belongs in the objective. Without
        # it the signal phase is purely O(quadrature points) and including the node
        # term lets it dominate, leaving the quadrature barely balanced at all.
        self.distribute_mesh(elem_weights=weights, overdecompose=overdecompose,
                             node_weight=1.0 if nodal_approximation else 0.0)
        self._partitions['signal'] = self._capture_partition()
        self._active_partition = 'signal'

        # Assembler groups live on the signal partition only.
        self.set_assembler(voxel_size, lorder=lorder, horder=horder,
                           nodal_approximation=nodal_approximation, lumped=lumped)
        self._partitions['signal'] = self._capture_partition()

        self.activate('bloch')
        self._partition_bound = False
        self._dual = True
        MPI_print("[FEMPhantom] Dual partitioning enabled: 'bloch' (node-balanced) "
                  "+ 'signal' (quadrature-balanced).")

    def _signal_modes(self, pod):
        """POD modes in the *signal* partition's node ordering.

        Redistributed once and cached. Building a second ``POD`` object instead would
        re-run the full global SVD on every rank (``Motion.calculate_pod``), and
        ``get_modes`` hard-checks the node count, so it cannot simply be re-sliced.
        """
        cache = self.__dict__.setdefault('_signal_modes_cache', {})
        key = id(pod)
        if key not in cache:
            n_bloch = self._partitions['bloch']['_local_to_global_nodes'].size
            modes = np.asarray(pod.get_modes(n_bloch), dtype=np.float32)
            cache[key] = np.ascontiguousarray(
                self.redistribute_nodal(modes, 'bloch', 'signal'))
        return cache[key]

    def quadrature_cost_weights(self, voxel_size, lorder=1, horder=1, cost_ratio=47.0,
                                nodal_approximation=False):
        """Per-element integration cost, in node-equivalents.

        Mirrors the ``local_elem_size < voxel_size`` split :meth:`set_assembler`
        applies, so the weights cannot disagree with the grouping actually used.

        Parameters
        ----------
        cost_ratio : float, optional
            How many quadrature points cost as much as one node, for *your*
            sequence: (Bloch seconds per node) / (quadrature seconds per
            quadrature point). It varies with the ratio of Bloch blocks to
            readouts.
        nodal_approximation : bool, optional
            Must match what will be passed to :meth:`set_assembler`. When True the
            small-element group is integrated through ``signal_nodal``, whose cost
            is O(nodes) and is therefore already counted by the node term -- so
            those elements carry **zero** quadrature weight. Charging them
            ``nq(lorder)`` as well double-counts them and measurably degrades the
            partition.

        Returns
        -------
        np.ndarray
            ``(n_global_elements,)`` float cost per element.
        """
        nq_lo = 0 if nodal_approximation else quadrature_npoints(self.cell_type, int(lorder))
        nq_hi = quadrature_npoints(self.cell_type, int(horder))
        size = np.asarray(self.global_elem_size, dtype=np.float64)
        return np.where(size < voxel_size, nq_lo, nq_hi) / float(cost_ratio)

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
                gathered_indices = MPI_comm.gather(self._local_to_global_nodes, root=0)

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
            idx = self.mesh_to_submesh_nodes[self._local_to_global_nodes]
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

    def set_assembler(self, voxel_size, lorder=1, horder=1, nodal_approximation=False,
                      lumped=True):
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
        Notes
        -----
        This does **not** repartition. It builds the assembler groups on whatever
        partition is live -- the NODAL one from ``distribute_mesh``, or the signal
        layout when called from :meth:`enable_dual_partition`.
        """
        # Under dual partitioning the assembler groups belong to the SIGNAL layout,
        # but 'bloch' is the resting one -- so a second set_assembler call (which any
        # script comparing assembler configurations makes) would otherwise build them
        # on the wrong partition. The signal layout itself is not rebuilt, so the
        # partition-keyed caches stay valid.
        if getattr(self, '_dual', False) and self._active_partition != 'signal':
            with self._using('signal'):
                return self.set_assembler(voxel_size, lorder=lorder, horder=horder,
                                          nodal_approximation=nodal_approximation,
                                          lumped=lumped)

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

        # Tell the assemblers which nodes this rank is responsible for, so the raw
        # nodal sum does not count interface nodes once per owning rank. Every group
        # carries the full local node set, so they all get the same mask.
        if MPI_size > 1:
            owned = self._node_ownership_mask()
            for a in self.assembler:
                a.set_node_ownership(owned)

        # The assembler groups now hold this partition's elements and nodes.
        self._partition_bound = True

    def update_magnetization(self, Mxy):
        """Push a new transverse magnetization array into all assembler instances.

        Under dual partitioning ``Mxy`` arrives in the Bloch layout (it is what
        ``BlochSolver.solve()`` returns) and is redistributed into the signal layout
        here -- the only per-handoff communication in the scheme.
        """
        if getattr(self, '_dual', False):
            Mxy = self.redistribute_nodal(np.ascontiguousarray(Mxy), 'bloch', 'signal')
            with self._using('signal'):
                return self._update_magnetization_local(Mxy)
        return self._update_magnetization_local(Mxy)

    def _update_magnetization_local(self, Mxy):
        for i, a in enumerate(self.assembler):
            if i == 0 and self.nodal_approximation__:
                a.update_nodal_magnetization(self.M_, Mxy)
            else:
                # Nodal store only. The quadrature projection is deferred to the
                # first `signal` / `signal_full` call on this group, so workloads
                # that use only signal_sum / signal_nodal never pay it.
                a.update_magnetization(Mxy)

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
        """Push per-node relaxation and off-resonance into the assembler groups.

        Under dual partitioning these arrive in the Bloch layout -- scripts build
        them from ``phantom.local_nodes`` -- and are redistributed once into the
        signal layout, where the assemblers live. Same contract as
        :meth:`update_magnetization`, so call sites need no dual-specific variant.
        """
        if getattr(self, '_dual', False) and self._active_partition != 'signal':
            T2 = self.redistribute_nodal(np.ascontiguousarray(T2), 'bloch', 'signal')
            phi_dB0 = self.redistribute_nodal(np.ascontiguousarray(phi_dB0),
                                              'bloch', 'signal')
            with self._using('signal'):
                return self._set_static_fields_local(T2, phi_dB0)
        return self._set_static_fields_local(T2, phi_dB0)

    def _set_static_fields_local(self, T2, phi_dB0):
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
        if isinstance(pod, list):
            raise NotImplementedError("Lists of trajectories must be combined using PODSum before evaluation.")
            
        with self._using('signal'):
            t_cpp, m_x, m_y, m_z, w, has_traj = self._prepare_pod_data(kspace_times, pod)

            eval_helper = []
            for i, a in enumerate(self.assembler):
                if i == 0 and self.nodal_approximation__:
                    eval_helper.append(a.signal_nodal)
                else:
                    eval_helper.append(a.signal)

            return sum([signal(kspace_points, t_cpp, m_x, m_y, m_z, w, has_traj)
                        for signal in eval_helper])

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
            raise NotImplementedError("Lists of trajectories must be combined using PODSum before evaluation.")
        with self._using('signal'):
            t_cpp, m_x, m_y, m_z, w, has_traj = self._prepare_pod_data(kspace_times, pod)
            return sum([a.signal(kspace_points, t_cpp, m_x, m_y, m_z, w, has_traj)
                        for a in self.assembler])

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
            raise NotImplementedError("Lists of trajectories must be combined using PODSum before evaluation.")
        # Evaluated on ONE group only, for the same reason as ``signal_sum``: every
        # group carries the full node set. Note the mass matrix ``M_`` is assembled
        # from the *small*-element group alone, so on a mesh that splits, this
        # integrates only that group -- use ``mri_signal``, which routes the
        # large-element group through the quadrature path, for the whole mesh.
        with self._using('signal'):
            t_cpp, m_x, m_y, m_z, w, has_traj = self._prepare_pod_data(kspace_times, pod)
            return self.assembler[0].signal_nodal(kspace_points, t_cpp, m_x, m_y, m_z,
                                                  w, has_traj)

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
            raise NotImplementedError("Lists of trajectories must be combined using PODSum before evaluation.")
        # Evaluated on ONE group only. Every assembler group is constructed with the
        # rank's *entire* node set (only the element subset differs), so summing this
        # nodal quantity over groups would count each node once per group.
        with self._using('signal'):
            t_cpp, m_x, m_y, m_z, w, has_traj = self._prepare_pod_data(kspace_times, pod)
            return self.assembler[0].signal_sum(kspace_points, t_cpp, m_x, m_y, m_z,
                                                w, has_traj)