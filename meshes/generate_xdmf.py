from pathlib import Path

import meshio
import numpy as np
import yaml
from FEelMRI.IO import XDMFFile

# Cell type
cell_type = 'wedge'

# Cell labels
domains = [i for i in range(10)]

# Path to meshes
path_to_mesh = Path('water_and_fat.msh')

# Import mesh with walls in VTU format
mesh = meshio.read(path_to_mesh)
nodes = mesh.points

# Get cell data from CellBlocks
cells = [cell.data for cell in mesh.cells if cell.type == cell_type]
cell_markers = [mesh.cell_data['gmsh:physical'][i] for i in range(len(mesh.cell_data['gmsh:physical']))]

# Concatenate data
cells = np.concatenate(cells)
cell_markers = np.concatenate(cell_markers).reshape(-1, 1)

print(cells.shape)
print(cell_markers.shape)

# Generate point markers
point_markers = np.zeros([nodes.shape[0], 1])
for i, markers in enumerate(mesh.cell_data['gmsh:physical']):
  print(mesh.cells[i].type)
  if mesh.cells[i].type == cell_type:
    point_markers[mesh.cells[i].data.flatten()] = markers[i]

# Interpolated phantom with walls
xdmf_file = 'xdmf/water_and_fat.xdmf'
file = XDMFFile(filename=xdmf_file, nodes=nodes, elements=({cell_type: cells}))

# Write timeseries
file.write(pointData={'point_markers': point_markers}, cellData={'cell_markers': cell_markers})

# Close file
file.close()