# FEelMRI

FEelMRI is a open source generator of synthetic Complementary SPAMM and DENSE images.

## Get started
To generate CSPAMM images, in Python simply do
```python
from FEelMRI import *
import numpy as np

# Parameters
p = Parameters(time_steps=18)

# Encoding frequency
ke = 0.07               # encoding frequency [cycles/mm]
ke = 1000*2*np.pi*ke    # encoding frequency [rad/m]

# Create complimentary image
I = CSPAMMImage(FOV=np.array([0.2, 0.2, 0.04]),
          center=np.array([0.0,0.0,0.03]),
          resolution=np.array([100, 100, 1]),
          encoding_frequency=np.array([ke,ke,0]),
          T1=0.85,
          flip_angle=15*np.pi/180,
          encoding_angle=90*np.pi/180,
          off_resonance=phi,
          kspace_factor=15,
          slice_thickness=0.008,
          oversampling_factor=2,
          phase_profiles=50)

# Spins
spins = Spins(Nb_samples=1000000, parameters=p)

# Create phantom object
phantom = Phantom(spins, p, patient=False, write_vtk=False)

# EPI acquisiton object
artifact = EPI(receiver_bw=128*1000,
               echo_train_length=10,
               off_resonance=200,
               acq_matrix=I.acq_matrix,
               spatial_shift='top-down')

# Generate images
kspace_0, kspace_1, kspace_in, mask = I.generate(artifact, phantom, p)

# Complementary images
u = u0 - u1

# Plots
if MPI_rank==0:
    multi_slice_viewer(np.abs(u[:,:,0,0,:]))
    multi_slice_viewer(np.abs(u[:,:,0,1,:]))
    multi_slice_viewer(np.abs(kspace_0.k[:,:,0,0,:]))
    multi_slice_viewer(np.abs(kspace_0.k[:,:,0,1,:]))
```

and run from a terminal with
```bash
mpirun -n nb_proc python3 foo.py
```
Resulting images look like this:
| CSPAMM image with epi-like artifacts  | CSPAMM kspace with epi-like artifacts |
| ------------- | ------------- |
| ![CSPAMM image](/screenshots/Figure_1.png "CSPAMM image with epi-like artifacts")  | ![CSPAMM image](/screenshots/Figure_2.png "CSPAMM image with epi-like artifacts")  |

## Installation
```bash
git clone https://github.com/hernanmella/FEelMRI
cd FEelMRI
python3 setup.py install
```

## Installation with Docker
To build the docker image run the following instruction in the command line:
```bash
docker build --rm . -t FEelMRI
```

Once the building process has finished, run the following to start the FEelMRI Docker container:
```bash
docker run -ti -v "path/to/folder":"/home/FEelMRI" FEelMRI
```

If you wish to run the container with plotting support, i.e., allowing to the container to show images, first run:
```bash
sudo apt-get install x11-xserver-utils && xhost +
```
and finally, use the following instruction to start the container:
```bash
docker run --rm -it \
   --user="FEelMRI" \
   --env="DISPLAY" \
   --volume="/etc/group:/etc/group:ro" \
   --volume="/etc/passwd:/etc/passwd:ro" \
   --volume="/etc/shadow:/etc/shadow:ro" \
   --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -v "/home/hernan/Git/FEelMRI/demo":"/home/FEelMRI" FEelMRI
```

## Dependencies
The dependencies of the project are listed in the files ```install_dependencies.sh``` and ```Dockerfile```.