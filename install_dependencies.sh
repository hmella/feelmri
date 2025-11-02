#!/bin/bash

# Install feelmri dependencies
sudo apt-get -y install build-essential python3-dev python3-pip python3-tk python3-setuptools libopenmpi-dev cmake

# Eigen3 installation
git clone https://gitlab.com/libeigen/eigen.git && \
cd eigen && mkdir build && cd build/ && \
cmake -DCMAKE_INSTALL_PREFIX=/usr .. && sudo make install && \
cd .. && cd .. && rm -rf eigen
