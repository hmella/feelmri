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
def gather_image(image):

  # Check input array dtype
  if image.dtype == np.complex128:
      MPI_TYPE = MPI.C_DOUBLE_COMPLEX
  elif image.dtype == np.complex64:
      MPI_TYPE = MPI.C_FLOAT_COMPLEX
  elif image.dtype == np.float:
      MPI_TYPE = MPI.DOUBLE
  elif image.dtype == np.float32:
      MPI_TYPE = MPI.FLOAT
  elif image.dtype == np.int:
        MPI_TYPE = MPI.LONG
  elif image.dtype == np.int32:
      MPI_TYPE = MPI.INT

  # Empty image
  total = np.zeros_like(image)

  # Reduced image
  MPI_comm.Reduce([image, MPI_TYPE], [total, MPI_TYPE], op=MPI.SUM, root=0)

  return total


# Scatter spins across processess
def ScatterSpins(coordinates):
  ''' Scatter dofmap and coordinate spins to local processes

  Input:
  -----------
    spins:        numpy ndarray of shape [n, d] with d the dimension of the function space
    coordinates: numpy ndarray spins coordinates

  Output:
  -----------
    local_spins: distributed spins along all processes
    local_coords: distributed coordinates along all processes
  '''
  if MPI_rank==0:

    # Number of spins
    nr_spins = coordinates.shape[0]
    geo_dim  = coordinates.shape[1]

    # Spins indices
    spins = np.linspace(0, nr_spins-1, nr_spins, dtype=np.int64)

    # Number of spins
    arr_size = int(nr_spins/MPI_size)
    if arr_size % geo_dim != 0:
      arr_size = arr_size - 1
    sections = [int(arr_size*i) for i in range(1,MPI_size)]

    # Split arrays
    local_spins = [[a] for a in np.split(spins, sections, axis=0)]

  else:
    #Create variables on other cores
    local_spins = None

  # Scatter local arrays to other cores
  local_spins   = MPI_comm.scatter(local_spins, root=0)[0]
  local_coords = coordinates[local_spins]

  # # Make spins local
  # local_spins  = local_spins# - local_spins.min()

  return local_coords, local_spins


# Printing function for parallel processing
def MPI_print(*args, **kwargs):
  if MPI_rank == 0:
    print(*args, **kwargs)