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


def collective_raise(message, exc_type=ValueError):
  """Raise on EVERY rank if ANY rank supplies a message.

  Per-node validation inspects local data, so a bad entry can exist on one rank
  only. A bare ``raise`` there aborts that rank while the others walk on into
  the next collective and block forever, an un-debuggable hang in place of a
  one-line traceback, for what is usually a one-line input mistake.

  **Call it unconditionally.** Putting it inside the branch that found the
  problem reproduces the very bug it exists to prevent, which has happened
  here more than once: compute a message (empty when clean) and pass that.

  ``exc_type`` keeps a guard's own exception class, the sub-ensemble
  preconditions promise ``NotImplementedError``, not ``ValueError``.
  """
  if MPI_comm.Get_size() == 1:
    if message:
      raise exc_type(message)
    return
  gathered = MPI_comm.allgather(str(message or ''))
  offenders = [r for r, m in enumerate(gathered) if m]
  if offenders:
    first = offenders[0]
    raise exc_type(
      f"{gathered[first]} [reported by rank {first}"
      + (f" and {len(offenders) - 1} other(s)" if len(offenders) > 1 else "")
      + f" of {len(gathered)}]")
