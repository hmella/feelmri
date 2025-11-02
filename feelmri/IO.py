from pathlib import Path
from subprocess import run

import meshio
import numpy as np

from feelmri.MPIUtilities import MPI_comm, MPI_rank, MPI_print

try:
  from pyevtk.hl import imageToVTK
  from pyevtk.vtk import VtkGroup  
except ImportError:
  MPI_print("feelmri import error: pyevtk python module not available")


# File class to write individual or cine VTI files
class VTIFile:
  def __init__(self, filename : str = 'image.pvd', 
               origin : np.ndarray = np.zeros([3,]), 
               spacing : np.ndarray = np.ones([3,]), 
               direction : np.ndarray = np.eye(3).flatten(), 
               nbFrames : int = 1, 
               dt : float = 1):
    self.filename = Path(filename) if '.pvd' in filename else Path(filename+'.pvd')
    self.origin = origin
    self.spacing = spacing
    self.direction = direction
    self.nbFrames = nbFrames
    self.dt = dt

  def write(self, cellData=None, pointData=None):
    '''Write VTI or PVD files.
    Parameters
    ----------
    cellData : dict, optional
        Dictionary of cell data to write. The default is None.
    pointData : dict, optional
        Dictionary of point data to write. The default is None.
    '''

    # Make sure the containing folder exists
    self.filename.parent.mkdir(parents=True, exist_ok=True)

    # Create pvd object to group all files
    pvd = VtkGroup(str(self.filename.parent/self.filename.stem))

    # Check if nbFrames match the last dimension of each cellData
    if self.nbFrames > 1:

      # Write cine images
      MPI_print("Writing vti files...")
      for fr in range(self.nbFrames):
        MPI_print("    Writing fame {:d}".format(fr))

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
      if cellData is not None:
        cell_data = [self.make_contiguous(cellData[key]) for key in cellData.keys()]
        cdfr = dict(zip(cellData.keys(), cell_data))
      if pointData is not None:
        point_data = [self.make_contiguous(pointData[key]) for key in pointData.keys()]
        ptfr = dict(zip(pointData.keys(), point_data))

      # Write data
      frame_path = str(self.filename.parent/self.filename.stem)
      if cellData is not None and pointData is None:
        imageToVTK(frame_path, cellData=cdfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      elif cellData is None and pointData is not None:
        imageToVTK(frame_path, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      else:
        imageToVTK(frame_path, cellData=cdfr, pointData=ptfr, origin=self.origin, spacing=self.spacing, direction=self.direction)
      pvd.addFile(filepath=frame_path+'.vti', sim_time=0)
      pvd.save()

  def make_contiguous(self, A):
    '''Make array contiguous in memory.'''
    if not A.data.contiguous:
      return np.ascontiguousarray(A)
    else:
      return A


# File class to write individual or cine VTI files
class XDMFFile:
  def __init__(self, filename: Path | str = 'phantom.xdmf', 
               nodes: np.ndarray = None, 
               elements: np.ndarray = None):
    if isinstance(filename, str):
      self.filename = Path(filename) if '.xdmf' in filename else Path(filename+'.xdmf')
    elif isinstance(filename, Path):
      self.filename = filename if '.xdmf' in filename.name else filename.with_suffix('.xdmf')
    self.nodes = nodes
    self.elements = elements
    self.__firstwrite__ = True

  def write(self, pointData=None, cellData=None, time=0.0):
    '''Write XDMF files.
    Parameters
    ----------
    pointData : dict of np.ndarray, optional
        Dictionary of point data arrays to write. The default is None.
    cellData : dict of np.ndarray, optional
        Dictionary of cell data arrays to write. The default is None.
    time : float, optional
        Time value to write. The default is 0.0.
    '''

    # First write: create file and store mesh
    if self.__firstwrite__:
      MPI_print("Writing XDMF file...")

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
    '''Close XDMF file and move to the right folder.'''
    MPI_print("Closing XDMF file...")

    # Close h5 files
    self.writer.__exit__()

    # Move generated HDF5 file to the right folder
    run(['mv',self.filename.stem+'.h5',str(self.filename.parent)+'/'])


# File class to write individual or cine TXT files
class TXTFile:
  def __init__(self, filename: str | Path = 'image.txt', 
               nodes : np.ndarray = None, 
               metadata: dict = None):
    self.filename = Path(filename) if '.txt' in filename else Path(filename+'.txt')
    self.nodes = nodes        # numpy ndarray
    self.metadata = metadata  # dictionary
    self._idx = 0
    self.__firstwrite__ = True

  def write(self, pointData: dict = None, time: float = 0.0):
    '''Write TXT files.
    Parameters
    ----------
    pointData : dict of np.ndarray, optional
        Dictionary of point data arrays to write. The default is None.
    time : float, optional
        Time value to write. The default is 0.0.
    '''
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