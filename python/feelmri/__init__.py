from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("feelmri")
except PackageNotFoundError:
    __version__ = "0.0.0"

# MR objects (scanner hardware, gradients, RF pulses)
from feelmri.MRObjects import Gradient, RF, Scanner

# Bloch simulation
from feelmri.Bloch import (
    ADC,
    BlochSolver,
    Sequence,
    SequenceBlock,
    collapse_isochromats,
    create_multi_isochromats,
    plot_isochromat_voxel,
    spoiling_residual,
)

# Slice profile and encoding
from feelmri.MRImaging import PositionEncoding, SliceProfile, VelocityEncoding

# k-space trajectories
from feelmri.KSpaceTraj import CartesianStack, RadialStack, SpiralStack, Trajectory

# Finite element phantom
from feelmri.Phantom import FEMPhantom

# Pulseq adapter and end-to-end simulator
from feelmri.PulseqAdapter import (
    PulseqImport,
    ReadoutWindow,
    as_signal_inputs,
    import_pulseq,
    kspace_trajectory,
    read_seq_feelmri,
)

# Motion models
from feelmri.Motion import POD, PODSum, PODVelocity, RespiratoryMotion

# Image reconstruction
from feelmri.Recon import (
    CartesianRecon,
    dcf_local_speed_readout,
    dcf_pipe_menon,
    dcf_radial_stack,
    reconstruct_nufft,
)

# I/O
from feelmri.IO import TXTFile, VTIFile, XDMFFile

# Parameters and configuration
from feelmri.Parameters import ParameterHandler, PVSMParser

# Plotting
from feelmri.Plotter import MRIPlotter

# k-space filters
from feelmri.Filters import Riesz, Tukey

# Math utilities
from feelmri.Math import Rx, Ry, Rz, itok, ktoi

# Noise
from feelmri.Noise import add_cpx_noise

# MPI utilities
from feelmri.MPIUtilities import MPI_comm, MPI_print, MPI_rank

__all__ = [
    # Version
    "__version__",
    # MR objects
    "Scanner", "Gradient", "RF",
    # Bloch simulation
    "ADC", "SequenceBlock", "Sequence", "BlochSolver",
    "create_multi_isochromats", "collapse_isochromats",
    "plot_isochromat_voxel", "spoiling_residual",
    # Slice profile and encoding
    "SliceProfile", "VelocityEncoding", "PositionEncoding",
    # k-space trajectories
    "Trajectory", "CartesianStack", "RadialStack", "SpiralStack",
    # Phantom
    "FEMPhantom",
    # Pulseq adapter / simulator
    "PulseqImport", "ReadoutWindow",
    "import_pulseq", "read_seq_feelmri", "kspace_trajectory",
    "as_signal_inputs",
    "SimulationResult", "simulate_pulseq",
    # Motion
    "RespiratoryMotion", "POD", "PODVelocity", "PODSum",
    # Reconstruction
    "CartesianRecon", "reconstruct_nufft",
    "dcf_pipe_menon", "dcf_radial_stack", "dcf_local_speed_readout",
    # I/O
    "VTIFile", "XDMFFile", "TXTFile",
    # Parameters
    "ParameterHandler", "PVSMParser",
    # Plotting
    "MRIPlotter",
    # Filters
    "Riesz", "Tukey",
    # Math utilities
    "itok", "ktoi", "Rx", "Ry", "Rz",
    # Noise
    "add_cpx_noise",
    # MPI utilities
    "MPI_print", "MPI_rank", "MPI_comm",
]
