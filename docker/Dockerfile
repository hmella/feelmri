# Download base image ubuntu 22.04
FROM ubuntu:22.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
LABEL org.opencontainers.image.authors="hernan.mella@pucv.cl"

# Create new user
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} -r feelmri \
  && useradd --uid ${UID} -g feelmri --no-log-init --shell /bin/bash feelmri
RUN mkdir -p /home/feelmri
WORKDIR /home/feelmri

# Change to root user for installation
USER root

# Update Ubuntu Software repository
RUN apt-get update && apt-get install -y apt-transport-https && apt-get install -y sudo && apt-get -y upgrade

# Install FEelMRI dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      python3 \
      python3-dev \
      python3-pip \
      python3-tk \
      cmake \
      ninja-build \
      git \
      libopenmpi-dev \
      screen \
      nano \
      libblas-dev \
      liblapack-dev \
      ccache
 RUN rm -rf /var/lib/apt/lists/*

# Copy FEelMRI source code to temporary build directory
RUN mkdir /tmp/feelmri_build/
COPY pyproject.toml /tmp/feelmri_build/
COPY CMakeLists.txt /tmp/feelmri_build/
COPY README.md /tmp/feelmri_build/
COPY cpp/ /tmp/feelmri_build/cpp/
COPY python/ /tmp/feelmri_build/python/

# Install FEelMRI
RUN cd /tmp/feelmri_build/ && pip3 install .

# Clean up temporary build files
RUN rm -rf build/ tmp/ dist/ feelmri.egg-info/
RUN rm -rf /tmp/feelmri_build/

# Change default user
USER feelmri
WORKDIR /home/feelmri