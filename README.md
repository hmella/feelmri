# FEelMRI

FEelMRI is an open source library for the generation of synthetic Magnetic Resonance images from Finite Elements simulations.

<img width="480" height="216" src="gifs/spamm.gif" alt="color picker" /> <img width="287" height="216" src="gifs/aorta.gif" alt="color picker" />

## Installation instructions
The first step to install the library is to clone the repository and navigate to the ```FEelMRI/``` folder:
```bash
git clone https://github.com/hernanmella/FEelMRI && cd FEelMRI/
```

### Dependencies
```FEelMRI``` dependencies can be installed through the ```install_dependencies.sh``` script. To do this, you should be able to run the script through:
```bash
source install_dependencies.sh  # be careful with the sudo commands inside this script
```
or
```bash
chmod a+x install_dependencies.sh && ./install_dependencies.sh
```

### Installing the library
To install ```FEelMRI``` simply run:
```bash
pip3 install .  # install
```
You might need to run this with ```sudo``` although not recommended. To avoid this, add the ```--user``` flag.


### Docker images
To avoid compiling the source code directly on your machine and allow interoperability, two ```Dockerfiles``` are provided inside the ```docker/``` folder: one for CPU parallelization and other for GPU parallelization using CUDA. To build any of the docker images, run the following instruction in the terminal:
```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -f docker/Dockerfile_foo -t image_name
```
where ```foo``` can be either ```cpu``` or ```gpu```, and ```image_name``` denotes the tag of yout image. The ```--build-arg UID=$(id -u)``` and ```--build-arg GID=$(id -g)``` arguments are used to tell Docker who owns the users and groups inside the image and avoid running the containers with inadequate permissions.

Once the building process has finished, run the following to start a FEelMRI Docker container:
```bash
docker run --name container_name --shm-size 256m -ti -v /path/to/host/folder:/home/FEelMRI/ image_name
```

#### Allowing plots inside containers
To allow plotting inside docker containers, run the following:
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

<!-- If you wish to run the container with plotting support, i.e., allowing to the container to show images, first run:
```bash
sudo apt-get install x11-xserver-utils && xhost + -->
<!-- ```

## Examples

and run from a terminal with
```bash
mpirun -n nb_proc python3 foo.py
```
Resulting images look like this:
| CSPAMM image with epi-like artifacts  | CSPAMM kspace with epi-like artifacts |
| ------------- | ------------- |
| ![CSPAMM image](/screenshots/Figure_1.png "CSPAMM image with epi-like artifacts")  | ![CSPAMM image](/screenshots/Figure_2.png "CSPAMM image with epi-like artifacts")  | -->
