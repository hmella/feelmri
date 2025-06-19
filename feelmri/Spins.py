import meshio
import numpy as np

from feelmri.MPIUtilities import MPI_rank, MPI_size


class Spins:
    def __init__(self, Nb_samples=1000, parameters=[]):
        self.Nb_samples = np.ceil(Nb_samples/MPI_size).astype(int)
        self.static = parameters.static
        self.R_en = parameters.R_en
        self.R_ep = parameters.R_ep
        self.R_inner = parameters.R_inner
        self.R_outer = parameters.R_outer
        self.h = parameters.h
        self.center = parameters.center
        self.samples, self.regions = self.generate_samples()
        self.mesh = self.build_mesh()

    # Generate spins using cylindrical coordinates
    def generate_samples(self):
        # Number of samples in each process
        if self.static:
          Ns  = int(self.Nb_samples/2)
        else:
          Ns  = self.Nb_samples

        # Radius, angle and height of the left ventricle
        r_lv = np.sqrt(np.random.uniform(self.R_en**2, self.R_ep**2, size=[Ns,]))
        t_lv = np.random.uniform(0.0, 2*np.pi, size=[Ns,])
        z_lv = np.random.uniform(-0.5*self.h, 0.5*self.h, size=[Ns,])

        # Radius, angle and height of the statit tissue
        if self.static:
          r_st = np.sqrt(np.random.uniform(self.R_inner**2, self.R_outer**2, size=[Ns,]))
          t_st = np.random.uniform(-np.pi/4, np.pi/4, size=[Ns,])
          z_st = np.random.uniform(-0.5*self.h, 0.5*self.h, size=[Ns,])

        # Concatenate values
        if self.static:
          r = np.concatenate([r_lv, r_st])
          t = np.concatenate([t_lv, t_st])
          z = np.concatenate([z_lv, z_st])
        else:
          r = r_lv
          t = t_lv
          z = z_lv          

        # CartesianStack coordinates
        x = r*np.cos(t) + self.center[0]
        y = r*np.sin(t) + self.center[1]
        z = z + self.center[2]

        # Separate regions
        inner = r < self.R_en
        outer = r > self.R_ep
        ventricle = (~inner)*(~outer)
        static = ~ventricle
        print('Number of spins in process {:d}: {:d}'.format(MPI_rank, x.shape[0]))

        return np.column_stack((x,y,z)), np.column_stack((ventricle,static))

    def build_mesh(self):
        N = self.Nb_samples
        vertices = np.linspace(0,N-1,N).reshape((N,1))
        return meshio.Mesh(points=self.samples, cells={'vertex': vertices})


# Function
class Function:
    def __init__(self, spins, dim=3, type=float):
        self.spins = spins
        self.dim = dim
        self.x = spins.samples
        self.type = type
        self.array = np.zeros([self.x.shape[0], dim], dtype=self.type)

    # Assign method
    def assign(self, u):
        self.array[:] = u

    # Vector
    def vector(self):
        return self.array

    # Copy vector values
    def copy_values(self):
      return np.copy(self.array)