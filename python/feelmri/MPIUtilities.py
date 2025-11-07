import numpy as np
from mpi4py import MPI


# MPI
MPI_comm = MPI.COMM_WORLD
MPI_size = MPI_comm.Get_size()
MPI_rank = MPI_comm.Get_rank()


# Sum images obtained in each processor into a single
# array
def gather_data(data):
  ''' Gather data from all MPI processes by summing them up '''

  # Get the local data type
  dtype = data.dtype
  mpi_type = MPI._typedict[dtype.char]

  # Empty image
  gathered_data = np.zeros_like(data)

  # Reduced image
  MPI_comm.Reduce([data, mpi_type], [gathered_data, mpi_type], op=MPI.SUM, root=0)

  return gathered_data


# Printing function for parallel processing
def MPI_print(*args, **kwargs):
  ''' Print only from the root process (rank 0) '''
  if MPI_rank == 0:
    print(*args, **kwargs)

  # Synchronize all processes
  MPI_comm.Barrier()