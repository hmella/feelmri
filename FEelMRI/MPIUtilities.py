import numpy as np
from mpi4py import MPI

# MPI
MPI_comm = MPI.COMM_WORLD
MPI_size = MPI_comm.Get_size()
MPI_rank = MPI_comm.Get_rank()


# Scatter array
def scatterKspace(kspace, times):
  if MPI_rank==0:

    # Number of readout points
    ro_samples = kspace[0].shape[0]

    # Phase line indices
    idx = np.linspace(0, ro_samples-1, ro_samples, dtype=np.int64)

    # Split arrays
    local_idx = [[a.astype(int)] for a in np.array_split(idx, MPI_size)]

  else:
    #Create variables on other cores
    local_idx = None

  # Scatter local arrays to other cores
  local_idx   = MPI_comm.scatter(local_idx, root=0)[0]
  local_kspace = (kspace[0][local_idx,...],
                  kspace[1][local_idx,...],
                  kspace[2][local_idx,...])
  local_times  = times[local_idx,...]

  print("Number of readout points in process {:d}: {:d}".format(MPI_rank, len(local_idx
  )))

  return local_kspace, local_times, local_idx


# Sum images obtained in each processor into a single
# array
def gather_data(data):

  # Get the local data type
  dtype = data.dtype
  mpi_type = MPI._typedict[dtype.char]

  # Empty image
  gathered_data = np.zeros_like(data)

  # Reduced image
  # MPI_comm.Reduce([data, mpi_type], [gathered_data, mpi_type], op=MPI.SUM, root=0)
  MPI_comm.Reduce([data, mpi_type], [gathered_data, mpi_type], op=MPI.SUM, root=0)

  return gathered_data


# Printing function for parallel processing
def MPI_print(*args, **kwargs):
  if MPI_rank == 0:
    print(*args, **kwargs)