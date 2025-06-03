
<!-- <p align="center"> -->
  <img height="150" src="gifs/logo.png" alt="color picker" />
<!-- </p> -->

**FEelMRI** is an open-source library for generating synthetic magnetic resonance images from finite-element (FE) simulations. The library is designed to handle complex phenomena whose behavior is described by partial differential equations and approximated using FEM. **FEelMRI** supports FE meshes with arbitrary cell geometries and simulations performed in any discrete function space.

<p align="center">
  <img height="216" src="gifs/spamm.gif" alt="color picker" /> <img height="216" src="gifs/aorta.gif" alt="color picker" />
</p>

## Installation instructions
To install the library, first clone the repository and navigate to the ```FEelMRI/``` folder:
```bash
git clone https://github.com/hernanmella/FEelMRI && cd FEelMRI/
```

### Dependencies
Dependencies can be installed via the ```install_dependencies.sh``` script. You can run:
```bash
source install_dependencies.sh  # be careful with the sudo commands inside this script
```
or
```bash
chmod a+x install_dependencies.sh && ./install_dependencies.sh
```

### Installing FEelMRI
To install the library, run:
```bash
pip3 install .
```
You may need to prepend ```sudo```, though this is not recommended. To install without ```sudo```, add the ```--user``` flag:
```bash
pip3 install . --user
```


### Docker images
Two Dockerfiles are provided in the ```docker/``` folder: one for CPU parallelization and one for GPU parallelization using CUDA. To build either image, run:
```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -f docker/Dockerfile_foo -t image_name
```
Here, ```foo``` can be either ```cpu``` or ```gpu```, and ```image_name``` will be the tag for your image. The ```--build-arg UID=$(id -u)``` and ```--build-arg GID=$(id -g)``` arguments ensure that files created inside the container are owned by your user, avoiding permission issues.

#### Starting a FEelMRI Docker Container
* **CPU container**:
  ```bash
  docker run --name container_name --shm-size 256m -ti -v $(pwd):/home/FEelMRI/ image_name
  ```
* **GPU container**:  
  ```bash
  docker run --name container_name  --runtime=nvidia --gpus all --shm-size 256m -ti -v $(pwd):/home/FEelMRI/ image_name
  ```

#### Allowing plots inside containers
To enable plotting within Docker containers, run (in place of the above `docker run` commands):
```bash
docker run -it \
    --name container_name \
    --user=$(id -u $USER):$(id -g $USER) \
    --env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $(pwd):/home/FEelMRI/ image_name
```