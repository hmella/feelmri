import subprocess
import os
from pathlib import Path

import meshio

from feelmri.Phantom import FEMPhantom

# Script path
script_path = Path(__file__).parent.resolve()

# Import the phantom
phantom = FEMPhantom("beating_heart.xdmf")
nodes = phantom.global_nodes
elems = phantom.global_elements
print(elems)
print(nodes)

# Create meshio mesh object
mesh = meshio.Mesh(points=nodes, cells={"tetra": elems})

# Export to Gmsh format
meshio.write("beating_heart.msh", mesh, file_format="gmsh22")

# # Run script to generate 3D hexahedral mesh with Gmsh
# os.system(f"gmsh {script_path}/convert_to_hex.geo -3 -o {script_path}/heart_P1_hex.msh")