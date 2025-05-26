#!/bin/bash

# Install FEelMRI dependencies
sudo apt-get update -y
sudo apt-get -y install build-essential python3-dev python3-pip python3-tk \
                 python3-setuptools libopenmpi-dev mpich cmake
sudo pip3 install pybind11 pyyaml mpi4py h5py meshio scipy matplotlib scikit-image pint

# Clone, build and install Eigen3-dev
git clone https://gitlab.com/libeigen/eigen.git && \
cd eigen && mkdir build && cd build/ && \
cmake -DCMAKE_INSTALL_PREFIX=/usr .. && sudo make install && \
cd .. && cd .. && rm -rf eigen
