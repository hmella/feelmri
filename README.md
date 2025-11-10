<p align="center">
  <img height="180" src="images/logo.png" alt="FEelMRI logo" />
</p>

<p align="center">
  <b>A fast, simple, and extensible Python library for transforming Finite Element simulations into MR Images</b>
</p>

<p align="center">
  <a href="https://github.com/hernanmella/feelmri/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-linux%20%7C%20docker-lightgrey.svg" alt="Platforms"></a>
</p>

---

**FEelMRI** is an open-source, cross-platform library designed to **generate synthetic magnetic resonance (MR) images** from **finite element (FE) simulations**.
It can handle complex physical phenomena governed by partial differential equations (PDEs) and supports **arbitrary cell geometries** and **discrete function spaces**.

---

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img height="200" src="images/spamm.gif" alt="SPAMM MRI"/><br>
        <b>Orthogonal-CSPAMM</b>
      </td>
      <td align="center">
        <img height="200" src="images/aorta.gif" alt="Aorta MRI"/><br>
        <b>4D Flow</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img height="140" src="images/abdomen.png" alt="Abdominal FFE"/><br>
        <b>Abdominal FFE</b>
      </td>
      <td align="center">
        <img height="140" src="images/free_running.png" alt="Free Running"/><br>
        <b>Free Running</b>
      </td>
    </tr>
  </table>
</p>

---

## 📚 Table of Contents

* [Installation](#-installation)
* [Quick Start Example](#-quick-start-example)
* [First Run](#-first-run)
* [Docker Setup (Cross-Platform)](#-docker-setup-cross-platform)
* [How to Contribute](#-how-to-contribute)
* [Citation](#-citation)
* [License](#-license)

---

## 🚀 Installation

> These steps were tested primarily on **Linux** systems.
> For Windows and macOS, please refer to [Docker Setup](#-docker-setup-cross-platform).

### 1️⃣ System Dependencies

You’ll need some basic system libraries and build tools. On Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    python3 python3-dev python3-pip python3-tk \
    cmake ninja-build git libopenmpi-dev
```

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/hernanmella/feelmri.git
cd feelmri
```

### 3️⃣ Install FEelMRI

```bash
pip install .
```

Verify installation ✅:

```bash
python3 -c "import feelmri; print(feelmri.__version__)"
```

### 4️⃣ (Optional) Unzip Example Phantoms

```bash
7z x examples/phantoms/phantoms_compressed.zip -o examples/phantoms/
```

---

## 🐍 First Run

Example scripts are provided in the `examples/` directory.
To run an example using multiple cores:

```bash
cd examples/
mpirun -n <nb_cores> python3 <example_script>.py
```

Replace `<nb_cores>` with the number of CPU cores to use.

### MRI planning with ParaView
To plan and position the FOV position and orientation, you are referred to [examples/planning/](examples/planning/) directory.

---

## 🐳 Docker Setup (Cross-Platform)

If you prefer an isolated or multi-platform setup, you can use Docker.

### Build the Docker Image

```bash
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -f docker/Dockerfile \
  -t feelmri:latest .
```

### Run the Container

```bash
docker run --name feelmri_container \
  --shm-size 256m -ti \
  -v $(pwd):/home/feelmri/ \
  feelmri:latest
```

### Enable GUI/Plots in Docker

If you want to visualize plots (e.g., matplotlib):

```bash
xhost +local:root
docker run -it \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $(pwd):/home/feelmri/ \
  feelmri:latest
```

---

## 🤝 How to Contribute

We welcome community contributions!
To get started:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Submit a pull request 🎉

More details are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📖 Citation

The related research article is currently under review.
Citation details will be added soon.

---

## 📜 License

This project is distributed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

<p align="center"><i>© 2025 Hernán Mella — FEelMRI Project</i></p>
