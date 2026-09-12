import os

os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity as Q_

from feelmri.Bloch import BlochSolver, Sequence, SequenceBlock
from feelmri.KSpaceTraj import CartesianStack
from feelmri.MPIUtilities import MPI_print, MPI_rank, gather_data
from feelmri.MRObjects import Scanner
from feelmri.Phantom import FEMPhantom
from feelmri.Recon import reconstruct_nufft

# Receive coil sensitivity. `BlochSolver(b1_map=...)` is TRANSMIT: it scales
# the RF a spin sees and so changes the magnetization itself. This is the other
# half -- what a coil HEARS from a magnetization that is already fixed. The two
# are unrelated: a body coil transmitting while an array receives is the
# ordinary arrangement.
#
# It rides the assembler's `nv` axis, which is free on the signal side, so a
# magnetization of `n_enc` columns and a map of `n_coils` produce
# `nv = n_enc * n_coils` with COILS VARYING FASTEST -- one C-order reshape to
# `(n_enc, n_coils)` recovers both.
#
# The point of the example is the COMBINE. Root-sum-of-squares needs no map,
# which is its whole appeal, and it destroys the phase. The matched filter
#
#     I = sum_c conj(S_c) I_c / sum_c |S_c|^2
#
# needs the map and keeps the phase, which is what anything read out of the
# argument -- velocity, off-resonance, fat/water -- depends on. In a simulation
# the map is known by construction rather than estimated, so the filter is
# exact: the same field handed to `set_receive_sensitivity`, sampled on the
# image grid instead of on the mesh.

# Enable fast mode for testing if the environment variable is set
FAST_MODE = os.getenv("FEELMRI_FAST_TEST", "0") == "1"

resolution = 48 if FAST_MODE else 96

# Four coils on a ring around the object, each a smooth complex field: a
# Gaussian falling away from its own centre, times a constant phase that
# differs from coil to coil. Nothing here is fitted or estimated -- it is
# written down, which is what lets the matched filter be exact.
coil_radius = 0.16
# The width matters, and not only for appearances. The reconstruction is
# band-limited while the map divided out is not, so recon(S*M) is not
# S*recon(M) and the matched filter cannot be exact for a map that varies on
# the scale of the point-spread function. Measured residual shading after the
# combine: 1.08x at 0.13 m, 1.32x at 0.09 m. Real arrays are smooth on the
# voxel scale, which is what makes the filter -- and every scheme that
# estimates a map -- work at all.
coil_width = 0.13


def coil_sensitivity(points):
  """``(n_points, n_coils)`` complex sensitivities at `points` (m)."""
  pts = np.asarray(points, dtype=float)
  maps = []
  for c in range(4):
    angle = 2.0 * np.pi * c / 4
    centre = coil_radius * np.array([np.cos(angle), np.sin(angle), 0.0])
    r2 = np.sum((pts - centre) ** 2, axis=1)
    maps.append(np.exp(-r2 / (2.0 * coil_width ** 2)) * np.exp(1j * angle))
  return np.stack(maps, axis=1).astype(np.complex64)


if __name__ == '__main__':

  # Get path of this script to allow running from any directory
  script_path = Path(__file__).parent

  # 1. A wide flat slab, imaged as a projection through its thickness.
  phantom = FEMPhantom(path=script_path/'phantoms/water_fat_P1_prism.xdmf',
                       scale_factor=0.01)
  phantom.set_assembler(voxel_size=1e3, lorder=1, horder=1,
                        nodal_approximation=True, lumped=True)
  nodes = phantom.local_nodes
  n_local = nodes.shape[0]
  phantom.set_static_fields(T2=np.full(n_local, 1e9, dtype=np.float32),
                            phi_dB0=np.zeros(n_local, dtype=np.float32))

  scanner = Scanner()

  # 2. A magnetization with a spatially varying PHASE. A uniform one would let
  # root-sum-of-squares look correct: it discards a phase that was not there.
  # A linear ramp of about a radian across the object is what separates the two
  # combines below.
  # 0.6*pi across the half-width, so the ramp spans 1.2*pi end to end and
  # never wraps -- a ramp that wraps is still recovered correctly but the
  # comparison below would have to unwrap it to say so.
  reach, phase_gain = 0.10, 0.6*np.pi
  true_phase = (phase_gain * nodes[:, 0] / reach).astype(np.float32)
  initial_Mxy = np.exp(1j * true_phase).astype(np.complex64)

  # 3. Hand the map to the phantom. It arrives in the layout the caller is
  # under and is redistributed into the signal layout once; the fold into the
  # magnetization happens AFTER that redistribution, so the per-handoff
  # Alltoallv still moves one column rather than four copies of it.
  phantom.set_receive_sensitivity(coil_sensitivity(nodes))

  # 4. One block, no gradients, no relaxation: the magnetization above is what
  # reaches the readout. The excitation is not the subject here.
  seq = Sequence()
  seq.add_block(SequenceBlock(dur=Q_(0.5, 'ms'), dt=Q_(0.5, 'ms'),
                              empty=True, store_magnetization=True))
  solver = BlochSolver(seq, phantom, scanner=scanner,
                       T1=Q_(1e9, 'ms'), T2=Q_(1e9, 'ms'),
                       initial_Mxy=initial_Mxy.reshape((-1, 1)),
                       initial_Mz=0.0, perfect_spoiling=False,
                       dtype='float64')
  Mxy, _ = solver.solve()

  # 5. Image it. `nv` comes back as 4 because the magnetization has one column
  # and the map has four.
  FOV = Q_(np.array([0.24, 0.24, 0.05]), 'm')
  traj = CartesianStack(FOV=FOV, res=np.array([resolution, resolution, 1]),
                        oversampling=1, lines_per_shot=1, scanner=scanner)
  phantom.update_magnetization(Mxy[:, 0])
  K = phantom.mri_signal(traj.points,
                         traj.times.m_as('ms') - traj.t_start.m_as('ms'))
  K = gather_data(K)
  MPI_print('k-space shape {} -> nv = {} (1 encoding x 4 coils)'
            .format(K.shape, K.shape[-1]))

  # The same acquisition with the map CLEARED: one uniform channel, which is
  # the image the matched filter is supposed to give back. Comparing against it
  # is what turns "the shading looks flatter" into a number, and it costs no
  # second solve -- the magnetization is unchanged.
  phantom.set_receive_sensitivity(None)
  phantom.update_magnetization(Mxy[:, 0])
  K_ref = gather_data(phantom.mri_signal(
      traj.points, traj.times.m_as('ms') - traj.t_start.m_as('ms')))

  if MPI_rank == 0:
    # A single slice, so the reconstruction is 2-D and carries no z axis.
    img_shape = (resolution, resolution)

    # The SAME field, on the image grid rather than the mesh. This is the whole
    # reason the matched filter is exact here instead of estimated.
    vox = FOV.m_as('m')[:2] / np.array(img_shape)
    grid = [(np.arange(n) - 0.5*(n - 1))*h for n, h in zip(img_shape, vox)]
    gx, gy = np.meshgrid(*grid, indexing='ij')
    gz = np.zeros_like(gx)
    sens = coil_sensitivity(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))
    # reconstruct_nufft puts channels FIRST, so the map has to match.
    sens = sens.reshape(img_shape + (4,)).transpose(2, 0, 1)

    common = dict(ktraj=traj.points, img_shape=img_shape, fov=FOV.m_as('m'),
                  auto_dcw=None, mode='adjoint')
    channels = reconstruct_nufft(kdata=K, combine=None, **common)
    rss = reconstruct_nufft(kdata=K, combine='rss', **common)
    roemer = reconstruct_nufft(kdata=K, combine='roemer', sensitivities=sens,
                               **common)

    # 6. What separates the two combines, measured rather than asserted by eye.
    # Over the object only: outside it there is no signal and no phase.
    mag = np.abs(roemer)
    inside = mag > 0.3 * mag.max()
    expected = phase_gain * gx / reach
    # Referenced to the object centre, since a reconstruction carries an
    # arbitrary global phase that neither combine is meant to remove.
    centre = int(np.argmin(np.abs(expected[inside])))
    def relative(field):
      p = np.angle(field)[inside]
      return np.angle(np.exp(1j*(p - p[centre])))
    target = expected[inside] - expected[inside][centre]

    # Compared on the circle: both sides are phases, so a bare difference
    # would read 2*pi wherever one of them happens to sit across a branch cut.
    def phase_error(field):
      return float(np.median(
          np.abs(np.angle(np.exp(1j*(relative(field) - target))))))
    err_roemer, err_rss = phase_error(roemer), phase_error(rss)
    MPI_print('phase error against the truth over the object (median): '
              'Roemer {:.4f} rad, RSS {:.4f} rad'.format(err_roemer, err_rss))
    # The MEDIAN, not the worst voxel. The residual is the forward model, not
    # the combine: a 48x48 truncated reconstruction of a phantom full of
    # circular vials rings at every internal edge, which reads 0.45 rad at the
    # worst voxel while the median sits at 0.05. RSS returns a real magnitude,
    # so its phase is identically zero and it cannot reproduce a ramp at all.
    assert err_roemer < 0.15, (
      f'the matched filter should recover the magnetization phase; got '
      f'{err_roemer:.4f} rad')
    assert err_rss > 0.5, (
      f'this case no longer separates the two combines: RSS reproduced the '
      f'phase to {err_rss:.4f} rad')

    # Shading, against the uniform-channel image rather than against itself.
    uniform = reconstruct_nufft(kdata=K_ref, combine=None, **common)
    ratio_single = np.abs(channels[0])[inside] / np.abs(uniform)[inside]
    ratio_roemer = np.abs(roemer)[inside] / np.abs(uniform)[inside]
    MPI_print('magnitude against the no-coil image: one coil varies {:.1f}x '
              'over the object, the matched filter {:.2f}x'
              .format(float(ratio_single.max()/ratio_single.min()),
                      float(ratio_roemer.max()/ratio_roemer.min())))
    assert ratio_roemer.max()/ratio_roemer.min() < 1.25, (
      'the matched filter did not divide the sensitivity shading out')

    # 7. Panels: each coil sees a different part of the object, RSS puts them
    # back together without the phase, the matched filter puts them back
    # together with it.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # A COMMON grey scale across the four coils. Autoscaling each panel would
    # normalise away exactly what they are here to show -- every coil would
    # look equally bright and the shading would be invisible.
    vmax = float(np.abs(channels).max())
    for c in range(4):
      ax = axes[0, c]
      im = ax.imshow(np.abs(channels[c]).T, origin='lower', cmap='Greys_r',
                     vmin=0.0, vmax=vmax)
      ax.set_title('coil {} magnitude'.format(c))
      ax.set_xticks([]); ax.set_yticks([])
      fig.colorbar(im, ax=ax, fraction=0.046)

    for ax, (title, field, cmap) in zip(
        axes[1],
        [('RSS magnitude', np.abs(rss), 'Greys_r'),
         ('Roemer magnitude', np.abs(roemer), 'Greys_r'),
         ('Roemer phase', np.angle(roemer) * inside, 'RdBu_r'),
         ('true phase', expected * inside, 'RdBu_r')]):
      im = ax.imshow(field.T, origin='lower', cmap=cmap,
                     **({'vmin': -np.pi, 'vmax': np.pi} if cmap == 'RdBu_r' else {}))
      ax.set_title(title)
      ax.set_xticks([]); ax.set_yticks([])
      fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('RSS discards the phase; the matched filter keeps it and '
                 'divides the shading out')
    fig.tight_layout()
    plt.show()
