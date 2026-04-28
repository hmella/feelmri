"""
MPI utility helpers for parallel MR simulation runs.
"""
import numpy as np
from mpi4py import MPI


MPI_comm = MPI.COMM_WORLD
MPI_size = MPI_comm.Get_size()
MPI_rank = MPI_comm.Get_rank()


def gather_data(data):
    """Gather data from all MPI processes by summing them into the root process.

    Parameters
    ----------
    data : np.ndarray
        Local array held by each process.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``data`` containing the element-wise sum
        across all processes (only meaningful on rank 0).
    """
    # Get the local data type
    dtype = data.dtype
    mpi_type = MPI._typedict[dtype.char]

    # Empty image
    gathered_data = np.zeros_like(data)

    # Reduced image
    MPI_comm.Reduce([data, mpi_type], [gathered_data, mpi_type], op=MPI.SUM, root=0)

    return gathered_data


def MPI_print(*args, **kwargs):
    """Print a message only from the root process (rank 0).

    Parameters
    ----------
    *args : any
        Positional arguments forwarded to :func:`print`.
    **kwargs : any
        Keyword arguments forwarded to :func:`print`.
    """
    if MPI_rank == 0:
        print(*args, **kwargs)

    # Synchronize all processes
    MPI_comm.Barrier()
