import shutil
import traceback
from pathlib import Path

import meshio
import numpy as np
from pyevtk.hl import imageToVTK
from pyevtk.vtk import VtkGroup

from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank


class VTIFile:
  def __init__(self, filename: str = 'image.pvd', 
               origin: np.ndarray = np.zeros([3,]), 
               spacing: np.ndarray = np.ones([3,]), 
               direction: np.ndarray = np.eye(3).flatten(), 
               nbFrames: int = 1, 
               dt: float = 1):
    if isinstance(filename, str):
      self.filename = Path(filename) if filename.endswith('.pvd') else Path(filename+'.pvd')
    elif isinstance(filename, Path):
      self.filename = filename if filename.suffix == '.pvd' else filename.with_suffix('.pvd')
    self.origin = origin
    self.spacing = spacing
    self.direction = direction
    self.nbFrames = nbFrames
    self.dt = dt

  def write(self, cellData=None, pointData=None):
    # Guard: Only Rank 0 talks to the disk
    if MPI_rank == 0:
      try:
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        pvd = VtkGroup(str(self.filename.parent / self.filename.stem))

        if self.nbFrames > 1:
          print("Writing vti files...", flush=True)
          for fr in range(self.nbFrames):
            print(f"  Writing frame {fr:d}", flush=True)

            cdfr, ptfr = None, None
            if cellData is not None:
              cdfr = {k: self.make_contiguous(v[..., fr]) for k, v in cellData.items()}
            if pointData is not None:
              ptfr = {k: self.make_contiguous(v[..., fr]) for k, v in pointData.items()}   

            frame_path = str(self.filename.parent / f"{self.filename.stem}_{fr:04d}")

            imageToVTK(frame_path, cellData=cdfr, pointData=ptfr, 
                       origin=self.origin, spacing=self.spacing, direction=self.direction)  

            pvd.addFile(filepath=frame_path+'.vti', sim_time=fr*self.dt)
          pvd.save()
            
        else:
          print("Writing vti files...", flush=True)
          cdfr, ptfr = None, None
          
          if cellData is not None:
            # Automatically drop the 4th dimension if Nb_frames == 1 but array is 4D
            cdfr = {k: self.make_contiguous(v[..., 0] if v.ndim == 4 else v) for k, v in cellData.items()}
          if pointData is not None:
            # Automatically drop the 4th dimension if Nb_frames == 1 but array is 4D
            ptfr = {k: self.make_contiguous(v[..., 0] if v.ndim == 4 else v) for k, v in pointData.items()}

          frame_path = str(self.filename.parent / self.filename.stem)
          imageToVTK(frame_path, cellData=cdfr, pointData=ptfr, 
                     origin=self.origin, spacing=self.spacing, direction=self.direction)
          
          pvd.addFile(filepath=frame_path+'.vti', sim_time=0)
          pvd.save()
            
      except Exception as e:
        # Catch silent crashes and force them to print
        print(f"\n--- CRITICAL ERROR ON RANK 0 ---\n{traceback.format_exc()}", flush=True)
        MPI_comm.Abort(1)

    # Keep all ranks perfectly synchronized so Rank 1 doesn't race ahead
    MPI_comm.Barrier()

  def make_contiguous(self, A):
    # Safely cast to a pure numpy array (strips Pint units)
    A_np = np.asarray(A)
    return np.ascontiguousarray(A_np) if not A_np.flags.c_contiguous else A_np


class XDMFFile:
  def __init__(self, filename: Path | str = 'phantom.xdmf', 
         nodes: np.ndarray = None, 
         elements = None):
    if isinstance(filename, str):
      self.filename = Path(filename) if filename.endswith('.xdmf') else Path(filename+'.xdmf')
    elif isinstance(filename, Path):
      self.filename = filename if filename.suffix == '.xdmf' else filename.with_suffix('.xdmf')
      
    self.nodes = nodes
    
    # Safeguard: Convert dictionary to the list of tuples meshio expects
    if isinstance(elements, dict):
      self.elements = list(elements.items())
    else:
      self.elements = elements
      
    self.__firstwrite__ = True
    self.writer = None

  def write(self, pointData=None, cellData=None, time=0.0):
    if MPI_rank == 0:
      try:
        if self.__firstwrite__:
          # Use standard print with flush to ensure it bypasses MPI buffers
          print("Writing XDMF file...", flush=True)
          self.filename.parent.mkdir(parents=True, exist_ok=True)
          
          self.writer = meshio.xdmf.TimeSeriesWriter(str(self.filename))
          self.writer.__enter__()
          self.writer.write_points_cells(self.nodes, self.elements)
          self.__firstwrite__ = False
          
        print(f"  Writing time {time:.2f}", flush=True)
        self.writer.write_data(time, point_data=pointData, cell_data=cellData)
      
      except Exception as e:
        # If meshio crashes, print the exact error and immediately kill the MPI job
        print(f"\n--- CRITICAL ERROR ON RANK 0 ---\n{traceback.format_exc()}", flush=True)
        MPI_comm.Abort(1)

    # THE FIX: Force all ranks to wait here until Rank 0 finishes disk I/O.
    # This prevents Rank 1 from racing ahead and deadlocking the communicator.
    MPI_comm.Barrier()

  def close(self):
    if MPI_rank == 0:
      try:
        print("Closing XDMF file...", flush=True)
        if self.writer is not None:
          self.writer.__exit__(None, None, None)
        
        # The file meshio creates in the current working directory
        h5_file_cwd = Path(self.filename.stem + '.h5')
        target_dir = self.filename.parent
        target_file_path = target_dir / h5_file_cwd.name
        
        # Only attempt to move if the target path is different from where the file currently is
        if h5_file_cwd.exists() and h5_file_cwd.resolve() != target_file_path.resolve():
          # We specify the full destination file path to safely overwrite if needed
          shutil.move(str(h5_file_cwd), str(target_file_path))
          
      except Exception as e:
        print(f"\n--- CRITICAL ERROR CLOSING ON RANK 0 ---\n{traceback.format_exc()}", flush=True)
        MPI_comm.Abort(1)
    
    # Sync before final exit
    MPI_comm.Barrier()


class TXTFile:
  def __init__(self, filename: str | Path = 'image.txt', 
         nodes: np.ndarray = None, 
         metadata: dict = None):
    self.filename = Path(filename) if str(filename).endswith('.txt') else Path(str(filename)+'.txt')
    self.nodes = nodes    
    self.metadata = metadata  
    self._idx = 0
    self.__firstwrite__ = True

  def write(self, pointData: dict = None, time: float = 0.0):
    if MPI_rank == 0:
      if self.__firstwrite__:
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.__firstwrite__ = False

      destination = self.filename.parent / f"{self.filename.stem}_{self._idx:04d}{self.filename.suffix}"
      
      with open(destination, "w") as f:
        f.write("Time\n")
        f.write(f"{time}\n")

        if self.metadata is not None:
          for key, val in self.metadata.items():
            f.write(f"{key}\n")
            if np.isscalar(val) or isinstance(val, str):
              f.write(f"{val}\n")
            elif isinstance(val, np.ndarray):
              np.savetxt(f, val)   

      # Write data (fixed memory bloat and dictionary iteration bug)
      if pointData is not None:
        # Stack all arrays in one go, combining nodes and all pointData values
        data_to_write = np.column_stack([self.nodes] + list(pointData.values()))
      else:
        data_to_write = self.nodes

      with open(destination, 'ab') as f:
        np.savetxt(f, data_to_write)   

      self._idx += 1