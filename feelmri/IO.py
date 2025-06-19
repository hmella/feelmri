import os
import pickle
from pathlib import Path
from subprocess import run

import meshio
import numpy as np
from scipy.io import savemat

from feelmri.MPIUtilities import MPI_comm, MPI_rank

try:
  from pyevtk.hl import imageToVTK
  from pyevtk.vtk import VtkGroup  
except ImportError:
  print("feelmri import error: pyevtk python module not available")


# Save Python objects
def save_pyobject(obj, filename, sep_proc=False):
    # Write file
    if not sep_proc:
        if MPI_rank == 0:
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, -1)
    else:
        # Split filename path
        root, ext = os.path.splitext(filename)
        with open(root+'_{:d}'.format(MPI_rank)+ext, 'wb') as output:
            pickle.dump(obj, output, -1)

# Load Python objects
def load_pyobject(filename, sep_proc=False):

    # Load files
    if not sep_proc:
        if MPI_rank==0:
            with open(filename, 'rb') as output:
                obj = pickle.load(output)
        else:
            obj = None

        # Broadcast object
        obj = MPI_comm.bcast(obj, root=0)
    else:
        # Split filename path
        root, ext = os.path.splitext(filename)
        with open(root+'_{:d}'.format(MPI_rank)+ext, 'rb') as output:
            obj = pickle.load(output)

    return obj


# Write Functions to vtk
def write_vtk(functions, path=None, name=None):

    # Verify if output folder exists
    directory = os.path.dirname(path)
    if directory != []:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get mesh
    try:
        mesh = functions.spins.mesh
    except:
        mesh = functions[0].spins.mesh

    # Prepare data as a dictionary
    point_data = {}
    for i, (u, n) in enumerate(zip(functions,name)):

        # Element shape
        element_shape = u.dim

        if element_shape == 2:
          d = u.vector()
          data = np.zeros([d.shape[0], d.shape[1]+1], dtype=d.dtype)
          data[:,:-1] = d
        else:
          data = u.vector()

        point_data[n] = data

    mesh.point_data = point_data
    meshio.write(path, mesh)


# Export images
def export_image(data, path=None, name=None):

    if name is None:
        name = 'I'

    # Export data
    savemat(path+'.mat',{name: data})


# Scale images
def scale_data(I, mag=True, pha=False, real=False, imag=False, dtype=np.uint64):

    # slope and intercept
    ScaleIntercept = np.ceil(np.abs(I).max())
    ScaleSlope =  np.iinfo(dtype).max/(2*ScaleIntercept)

    # Data extraction
    if mag:
        mag = (ScaleSlope*np.abs(I)+ScaleIntercept).astype(dtype)
    if pha:
        pha = (ScaleSlope*1000*(np.angle(I)+np.pi)).astype(dtype)
    if real:
        real = (ScaleSlope*(np.real(I)+ScaleIntercept)).astype(dtype)
    if imag:
        imag = (ScaleSlope*(np.imag(I)+ScaleIntercept)).astype(dtype)

    # Rescaling parameters
    RescaleSlope = 1.0/ScaleSlope
    RescaleIntercept = -ScaleIntercept

    # output
    output = {}
    if mag:
        output["mag"] = {"Image": mag,
                        "RescaleSlope": RescaleSlope,
                        "RescaleIntercept": RescaleIntercept}
    if pha:
        output["pha"] = {"Image": pha,
                        "RescaleSlope": RescaleSlope,
                        "RescaleIntercept": RescaleIntercept}
    if real:
        output["real"] = {"Image": real,
                        "RescaleSlope": RescaleSlope,
                        "RescaleIntercept": RescaleIntercept}
    if imag:
        output["imag"] = {"Image": imag,
                        "RescaleSlope": RescaleSlope,
                        "RescaleIntercept": RescaleIntercept}

    return output


# Rescale image
def rescale_image(I):

    # Get images, slope and intercept
    Im = dict()
    for (key, value) in I.items():
        Im[key] = I[key]['Image']*I[key]['RescaleSlope'] + I[key]['RescaleIntercept']

    return Im

# File class to write individual or cine VTI files
class VTIFile:
  def __init__(self, filename='image.pvd', origin=np.zeros([3,]), spacing=np.ones([3,]), direction=np.eye(3).flatten(), nbFrames=1, dt=1):
    self.filename = Path(filename) if '.pvd' in filename else Path(filename+'.pvd')
    self.origin = origin
    self.spacing = spacing
    self.direction = direction
    self.nbFrames = nbFrames
    self.dt = dt

  def write(self, cellData=None, pointData=None):

    # Make sure the containing folder exists
    self.filename.parent.mkdir(parents=True, exist_ok=True)

    # Create pvd object to group all files
    pvd = VtkGroup(str(self.filename.parent/self.filename.stem))

    # Check if nbFrames match the last dimension of each cellData
    if self.nbFrames > 1:

      # Write cine images
      print("Writing vti files...")
      for fr in range(self.nbFrames):
        print("    Writing fame {:d}".format(fr))

        # cellData at frame fr
        if cellData != None:
          cell_data = [self.make_contiguous(cellData[key][...,fr]) for key in cellData.keys()]
          cdfr = dict(zip(cellData.keys(), cell_data))
        if pointData != None:
          point_data = [self.make_contiguous(pointData[key][...,fr]) for key in pointData.keys()]    
          ptfr =  dict(zip(pointData.keys(), point_data))   

        # Frame path
        frame_path = str(self.filename.parent / (self.filename.stem + '_{:04d}'.format(fr)))

        # Write VTI
        if cellData != None and pointData == None:
          imageToVTK(frame_path, cellData=cdfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
        elif cellData == None and pointData != None:
          imageToVTK(frame_path, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)        
        else:
          imageToVTK(frame_path, cellData=cdfr, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)        

        # Add VTI files to pvd group
        pvd.addFile(filepath=frame_path+'.vti', sim_time=fr*self.dt)

      # Save PVD
      pvd.save()

    else:
      # Make sure data is ordered as contiguous arrays
      if cellData != None:
        cell_data = [self.make_contiguous(cellData[key]) for key in cellData.keys()]
        cdfr = dict(zip(cellData.keys(), data))
      if pointData != None:
        point_data = [self.make_contiguous(cellData[key]) for key in pointData.keys()]
        ptfr = dict(zip(pointData.keys(), data))

      # Write data
      pvd = VtkGroup(self.filename.as_posix())
      frame_path = str(self.filename.parent/self.filename.stem)
      if cellData != None and pointData == None:
        imageToVTK(frame_path, cellData=cdfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      elif cellData == None and pointData != None:
        imageToVTK(frame_path, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      else:
        imageToVTK(frame_path, cellData=cdfr, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      pvd.addFile(filepath=frame_path+'.vti', sim_time=0)
      pvd.save()

  def make_contiguous(self, A):
    if not A.data.contiguous:
      return np.ascontiguousarray(A)
    else:
      return A


# File class to write individual or cine VTI files
class XDMFFile:
  def __init__(self, filename='phantom.xdmf', nodes=None, elements=None):
    self.filename = Path(filename) if '.xdmf' in filename else Path(filename+'.xdmf')
    self.nodes = nodes
    self.elements = elements
    self.__firstwrite__ = True

  def write(self, pointData=None, cellData=None, time=0.0):

    if self.__firstwrite__:
      print("Writing XDMF file...")

      # Make sure containing folder exists
      self.filename.parent.mkdir(parents=True, exist_ok=True)

      # Create writer
      self.writer = meshio.xdmf.TimeSeriesWriter(str(self.filename))
      self.writer.__enter__()

      # Store mesh
      self.writer.write_points_cells(self.nodes, self.elements)
      self.__firstwrite__ = False
      
    # Write data
    print("    Writing time {:.2f}".format(time))
    self.writer.write_data(time, point_data=pointData, cell_data=cellData)

  def close(self):
    # Close h5 files
    self.writer.__exit__()

    # Move generated HDF5 file to the right folder
    run(['mv',self.filename.stem+'.h5',str(self.filename.parent)+'/'])


# File class to write individual or cine TXT files
class TXTFile:
  def __init__(self, filename='image.txt', nodes=None, metadata=None):
    self.filename = Path(filename) if '.txt' in filename else Path(filename+'.txt')
    self.nodes = nodes        # numpy ndarray
    self.metadata = metadata  # dictionary
    self._idx = 0
    self.__firstwrite__ = True

  def write(self, pointData=None, time=0.0):
    # Make sure containing folder exists
    if self.__firstwrite__:
      self.filename.parent.mkdir(parents=True, exist_ok=True)
      self.__firstwrite__ = False

    destination = str(self.filename.parent/(self.filename.stem+'_{:04d}'.format(self._idx)+self.filename.suffix))
    with open(destination, "w") as f:
      # Write time
      f.write("{:s}\n".format("Time"))
      f.write("{:s}\n".format(str(time)))

      # Write metadata
      if self.metadata != None:
        for key in self.metadata.keys():
          # Write key
          f.write("{:s}\n".format(key))

          # Write metadata
          if np.isscalar(self.metadata[key]):
            f.write("{:s}\n".format(str(self.metadata[key])))
          elif isinstance(self.metadata[key],  str):
            f.write("{:s}\n".format(self.metadata[key]))
          elif isinstance(self.metadata[key], np.ndarray):
            np.savetxt(f, self.metadata[key])   

    # Write data
    data = self.nodes
    for d in pointData:
      data = np.column_stack((data, d))
    with open(destination,'ab') as f:
      np.savetxt(f, data)   

    # Update file counter
    self._idx += 1