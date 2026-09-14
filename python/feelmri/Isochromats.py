"""
Sub-voxel spin ensembles: spectral bins and spatial isochromats.

Two different mechanisms, deliberately kept apart. A FREQUENCY spread
(:func:`lineshape_bins`) models intra-voxel T2' and is rewound by a 180; a
SPATIAL spread (:func:`create_multi_isochromats`) models what a gradient
spoiler dephases and is rewound by an opposite gradient. Sampling one with the
other is a category error.

:func:`spoiling_residual` sizes ``K`` before a phantom run costs anything.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from feelmri.MPIUtilities import MPI_comm, MPI_rank, collective_raise


LINESHAPES = ('gaussian', 'uniform', 'lorentzian')

# Worst error of the lorentzian rule against exp(-t/T2') over tau in
# [0, 3*T2'], as a least-squares fit to eleven measured K from 8 to 256. Within
# 8% of the measurement everywhere; the earlier K^-0.55 form was 29% optimistic
# by K=256, i.e. wrong in the reassuring direction at exactly the K someone
# picks after reading the warning.
def _LORENTZIAN_ERR(K):
  return 1.34 * float(K)**-0.638



def lineshape_bins(K, lineshape='gaussian'):
  """Quadrature nodes and weights for an intra-voxel field distribution.

  Returns ``(z, w)``: ``K`` dimensionless frequency offsets and ``K``
  non-negative weights summing to exactly 1. Scaled by ``1/T2'`` the offsets
  are rad/ms, and the ensemble average

      F(tau) = sum_k w_k * exp(-i * z_k / T2' * tau)

  is the decay the sub-ensemble produces. The weights are a genuine probability
  distribution: an unconstrained least-squares fit reaches 4e-11 on the
  exponential but needs ``sum|w| = 1534``, i.e. cancellation that a spin
  ensemble cannot represent and float32 cannot carry.

  Measured worst error against the lineshape each rule claims, over
  ``tau`` in ``[0, 3*T2']``:

  ============  ========  ========  ========  ========
  K                    8        16        64       256
  ============  ========  ========  ========  ========
  gaussian       9.7e-03   1.7e-08   3.3e-16   5.0e-16
  uniform        2.2e-07   6.1e-16   3.1e-16   8.3e-16
  lorentzian     3.3e-01   2.4e-01   1.0e-01   3.8e-02
  ============  ========  ========  ========  ========

  * ``'gaussian'`` -- Gauss-Hermite. Decay ``exp(-tau^2 / 2 T2'^2)``.
    Machine precision by K=24.
  * ``'uniform'`` -- Gauss-Legendre on a half-width of ``sqrt(3)/T2'``, chosen
    so the variance matches the Gaussian of the same ``T2'``. Decay is a sinc,
    so the signal has true zero crossings and partial recoveries -- the right
    model for a linear susceptibility gradient across the voxel.
  **A finite bin set is quasi-periodic, so the decay REVIVES at long tau.**
  With K discrete frequencies the ensemble cannot stay cancelled forever; it
  recurs once the accumulated phase spread wraps. Largest ``tau/T2'`` at which
  each rule still tracks its lineshape to 1e-3:

  ============  ======  ======  ======  ======
  K                  8      16      32      64
  ============  ======  ======  ======  ======
  gaussian        2.50    4.67    7.86   12.47
  uniform         5.37   13.40     inf     inf
  ============  ======  ======  ======  ======

  So for the gaussian rule size ``K >= 5 * tau_max / T2'``, where ``tau_max``
  is the longest time coherence survives WITHOUT a refocusing pulse -- a 180
  restarts the clock, so echo-based sequences are far less demanding than the
  bound suggests. The uniform rule barely needs sizing.

  * ``'lorentzian'`` -- equal-probability quantile midpoints. The only rule
    that targets the conventional ``exp(-t/T2*)``, and the only inaccurate one:
    it converges roughly as ``K^-0.55``, so 3.8e-2 at K=256. That is not an
    implementation limit. ``exp(-t/T2*)`` is the Fourier transform of a
    Lorentzian, which has infinite variance, so reproducing it needs
    arbitrarily far off-resonance spins and no finite ensemble gets there.
  """
  K = int(K)
  if K < 1:
    raise ValueError(f"lineshape_bins: K must be >= 1; got {K}")
  key = str(lineshape).lower()
  if key not in LINESHAPES:
    raise ValueError(
      f"lineshape_bins: lineshape must be one of {list(LINESHAPES)}; "
      f"got {lineshape!r}")
  if key == 'gaussian':
    z, w = np.polynomial.hermite_e.hermegauss(K)
  elif key == 'uniform':
    z, w = np.polynomial.legendre.leggauss(K)
    z = z * np.sqrt(3.0)
  else:
    u = (np.arange(K) + 0.5) / K
    z, w = np.tan(np.pi * (u - 0.5)), np.ones(K)
  z = np.asarray(z, dtype=np.float64)
  w = np.asarray(w, dtype=np.float64)
  # The rules break down at large K and numpy does not say so. hermegauss
  # returns NaN weights somewhere between K=320 and K=400, and `spectral_bins`
  # has no natural ceiling, so without this check a large K silently produced
  # NaN magnetization for the whole phantom.
  if not (np.all(np.isfinite(z)) and np.all(np.isfinite(w)) and w.sum() > 0.0):
    raise ValueError(
      f"lineshape_bins: the {key} rule loses all precision at K = {K} "
      f"(non-finite nodes or weights). Use a smaller K; the gaussian rule is "
      f"reliable to about K = 320, and its weights are already negligible far "
      f"below that.")
  # Normalise in float64. The T1 recovery term (1 - e1) * M0 is AFFINE, so the
  # collapsed equilibrium is M0 * sum(w): raw Gauss-Hermite weights sum to
  # 2.5066 and Gauss-Legendre to 2.0, either of which would put the whole
  # phantom at the wrong M0.
  w = w / w.sum()
  # Drop sub-spins that cannot contribute. Gauss-Hermite spends its extreme
  # abscissae on weights far below machine epsilon -- 4 of 32, 70 of 128 --
  # and each is a full sub-spin carried through every time step and then
  # multiplied by nothing. Pruning at float64 epsilon changes no result
  # float64 can represent.
  #
  # >= 2, not >= 1: a single surviving bin is not an ensemble and would make
  # _n_bins == 1, routing the solver down the no-ensemble path while
  # t2_prime still reads back as configured.
  keep = w >= 1e-16
  if keep.sum() >= 2:
    z, w = z[keep], w[keep]
    w = w / w.sum()
  return z, w


def _draw_in_sphere_offsets(M, R, distribution='uniform', seed=None):
  """Draw ``M`` offset vectors uniformly distributed inside a 3-sphere of radius ``R``.

  Three samplers are supported. All three use the same inverse-CDF
  mapping from the unit cube to the sphere — ``r = R * u^(1/3)``,
  ``cos(theta) = 1 - 2v``, ``phi = 2*pi*w`` — and differ only in how
  ``(u, v, w) in [0, 1)^3`` is drawn:

  * ``'uniform'`` — i.i.d. ``Uniform([0, 1])`` via
    ``numpy.random.default_rng(seed)``. Monte-Carlo residual rate
    :math:`\\rho \\sim K^{-1/2}`.
  * ``'sobol'`` — :class:`scipy.stats.qmc.Sobol`, a low-discrepancy
    sequence. Quasi-Monte-Carlo residual rate
    :math:`\\rho = \\mathcal O((\\log K)^d / K)`.
  * ``'halton'`` — :class:`scipy.stats.qmc.Halton`, same QMC class as
    Sobol; cheaper to seed but empirically slightly weaker in 3-D
    due to higher-prime axis correlations.

  Parameters
  ----------
  M : int
      Number of points to draw.
  R : float
      Sphere radius (m, but the function is unit-agnostic).
  distribution : {'uniform', 'sobol', 'halton'}
  seed : int or None
      Forwarded to the underlying RNG / QMC engine. ``None`` retains
      the pre-refactor non-deterministic behaviour.

  Returns
  -------
  np.ndarray
      Float32 C-contiguous array of shape ``(M, 3)``.
  """
  dist = str(distribution).lower()
  if dist == 'uniform':
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=M)
    v = rng.uniform(0.0, 1.0, size=M)
    w = rng.uniform(0.0, 1.0, size=M)
  elif dist in ('sobol', 'halton'):
    from scipy.stats.qmc import Halton, Sobol
    M_int = int(M)
    if dist == 'sobol':
      # Sobol's (t, m, s)-net balance properties hold exactly when n
      # is a power of 2. Generate the smallest 2**m >= M and slice
      # rather than calling random(M) — strictly higher-quality, and
      # avoids the scipy UserWarning about non-power-of-2 sample counts.
      qmc = Sobol(d=3, seed=seed)
      m_exp = int(np.ceil(np.log2(max(M_int, 1))))
      pts = qmc.random_base2(m_exp)[:M_int]
    else:
      qmc = Halton(d=3, seed=seed)
      pts = qmc.random(M_int)
    u, v, w = pts[:, 0], pts[:, 1], pts[:, 2]
  else:
    raise ValueError(
      f"unknown distribution {distribution!r}; expected one of "
      f"'uniform', 'sobol', 'halton'"
    )
  radius = R * np.cbrt(u)
  cos_theta = 1.0 - 2.0 * v
  sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
  phi = 2.0 * np.pi * w
  out = np.empty((int(M), 3), dtype=np.float32)
  out[:, 0] = (radius * sin_theta * np.cos(phi)).astype(np.float32)
  out[:, 1] = (radius * sin_theta * np.sin(phi)).astype(np.float32)
  out[:, 2] = (radius * cos_theta).astype(np.float32)
  return out


def create_multi_isochromats(x, T1, T2, delta_B, Mxy0, Mz0,
                             K=100, pos_jitter=0.2e-3,
                             distribution='uniform', seed=None):
  """Replicate every node K times and offset by an in-sphere jitter.

  Every input array is repeated K times along axis 0 with
  :func:`numpy.repeat` (so node ``n`` produces the contiguous range
  ``[n*K : (n+1)*K]``). The positions ``x_big`` are then perturbed by
  in-sphere offsets drawn from ``distribution`` with radius
  ``pos_jitter``.

  Parameters
  ----------
  x : np.ndarray
      Node positions of shape ``(N, 3)``.
  T1, T2, delta_B, Mxy0, Mz0 : np.ndarray
      Nodal arrays repeated K times along axis 0.
  K : int, optional
      Number of isochromats per node. Default 100.
  pos_jitter : float, optional
      Radius of the in-sphere offset (m). Default 0.2 mm.
  distribution : {'uniform', 'sobol', 'halton'}, optional
      Sampler for the offsets — see :func:`_draw_in_sphere_offsets`.
      Default ``'uniform'`` preserves the pre-refactor behaviour.
  seed : int or None, optional
      RNG / QMC seed forwarded to the sampler. ``None`` is
      non-deterministic; ``BlochSolver`` defaults to ``0`` so the
      spoiler is reproducible.
  """
  x_big      = np.repeat(x, K, axis=0)
  T1_big     = np.repeat(T1, K, axis=0)
  T2_big     = np.repeat(T2, K, axis=0)
  deltaB_big = np.repeat(delta_B, K, axis=0)
  Mxy_big    = np.repeat(Mxy0, K, axis=0)
  Mz_big     = np.repeat(Mz0, K, axis=0)

  N = x.shape[0]
  jitter = _draw_in_sphere_offsets(N * K, pos_jitter,
                                   distribution=distribution, seed=seed)
  if x.shape[1] == 2:
    jitter = jitter[:, :2]
  x_big = x_big + jitter.astype(x_big.dtype, copy=False)

  return x_big, T1_big, T2_big, deltaB_big, Mxy_big, Mz_big


def collapse_isochromats(Mxy_big, Mz_big, K, mode="mean"):
    Mxy_big = np.asarray(Mxy_big)
    Mz_big  = np.asarray(Mz_big)

    if Mxy_big.ndim == 1:
        Mxy_big = Mxy_big.reshape(-1, 1)
    if Mz_big.ndim == 1:
        Mz_big = Mz_big.reshape(-1, 1)

    N_big = Mxy_big.shape[0]
    N = N_big // K

    # Reshape arrays to isolate the K isochromats for each node
    # Shapes become (N, K, 1)
    Mxy_reshaped = Mxy_big.reshape(N, K, -1)
    Mz_reshaped  = Mz_big.reshape(N, K, -1)

    # Compute mean or sum across the K axis (axis=1)
    if mode == "mean":
        Mxy_out = np.mean(Mxy_reshaped, axis=1)
        Mz_out  = np.mean(Mz_reshaped, axis=1)
    else:
        Mxy_out = np.sum(Mxy_reshaped, axis=1)
        Mz_out  = np.sum(Mz_reshaped, axis=1)

    return Mxy_out, Mz_out


def plot_isochromat_voxel(positions, *, R=None, ax=None,
                          color='steelblue', alpha=0.7, s=8,
                          title=None, show=True, export_to=None):
  """3-D scatter of K isochromat positions inside a voxel.

  Parameters
  ----------
  positions : np.ndarray
      Array of shape ``(K, 3)`` with the isochromat coordinates (m).
  R : float, optional
      Voxel radius. When supplied, a translucent reference sphere of
      that radius is drawn at the origin for spatial context.
  ax : matplotlib 3-D axis, optional
      Pre-existing axes to draw into. When ``None``, a fresh figure
      is created.
  color : str, optional
      Scatter colour.
  alpha : float, optional
      Scatter alpha.
  s : int or float, optional
      Scatter marker size.
  title : str, optional
      Axes title.
  show : bool, optional
      Call ``plt.show()`` after rendering. Default True.
  export_to : str or path-like, optional
      When supplied, save the figure to this path before showing.

  Notes
  -----
  Rank-0 guarded: on non-zero MPI ranks the function is a no-op and
  returns ``None`` to mirror the convention used by
  :meth:`SequenceBlock.plot` and :meth:`Sequence.plot`.
  """
  # Validated BEFORE the rank split and through the collective, because
  # `positions` is per-rank data: a bare raise on rank 0 left every other rank
  # already waiting in the Barrier below, turning a one-line shape mistake
  # into a hang.
  positions = np.asarray(positions)
  collective_raise(
      '' if positions.ndim == 2 and positions.shape[1] == 3 else
      f'plot_isochromat_voxel: rank {MPI_rank} was given positions of shape '
      f'{positions.shape}; it must be (K, 3).')

  if MPI_rank != 0:
    MPI_comm.Barrier()
    return None

  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D proj)
  if ax is None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
  else:
    fig = ax.figure

  ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
             c=color, s=s, alpha=alpha, depthshade=True,
             label=f'K = {positions.shape[0]}')

  if R is not None and R > 0:
    n = 24
    u, v = np.meshgrid(
      np.linspace(0.0, 2.0 * np.pi, n),
      np.linspace(0.0, np.pi, n // 2 + 1),
    )
    xs = R * np.sin(v) * np.cos(u)
    ys = R * np.sin(v) * np.sin(u)
    zs = R * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='gray', linewidth=0.3, alpha=0.4)

  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  ax.set_title(title or f'Isochromat voxel scatter (K = {positions.shape[0]})')
  ax.legend(loc='upper right')
  try:
    ax.set_aspect('equal')
  except (NotImplementedError, ValueError):
    pass

  # The other ranks are already in this Barrier, so rank 0 must reach it
  # whatever the drawing does -- a savefig onto an unwritable path would
  # otherwise strand them.
  try:
    if export_to is not None:
      fig.savefig(export_to, bbox_inches='tight')
    if show:
      plt.show()
  finally:
    MPI_comm.Barrier()
  return ax


def spoiling_residual(K, k_sp, voxel_size, *,
                      distribution='sobol', seed=0, n_trials=1):
  """Numerical residual of the K-isochromat spoiling sum.

  Computes

  .. math::

     \\rho(K) \\;=\\; \\left|
       \\frac{1}{K}\\sum_{k=1}^{K}
         \\exp\\big(\\,i\\, 2\\pi\\, \\vec k_{\\rm sp}\\cdot \\vec r_k\\big)
     \\right|

  where :math:`\\vec r_k` are K isochromat offsets drawn from
  ``distribution`` inside a sphere of radius ``voxel_size`` (m).
  Useful for sizing ``BlochSolver.isochromat_K`` before launching a
  real simulation.

  Parameters
  ----------
  K : int
      Number of isochromats.
  k_sp : array-like of shape (3,)
      Spoiler wavenumber :math:`\\vec k_{\\rm sp} = \\gamma/(2\\pi) \\cdot \\int_0^T G(t)\\, dt`
      in 1/m. The phase per isochromat is :math:`2\\pi \\vec k_{\\rm sp}\\cdot\\vec r_k`.
  voxel_size : float
      Sphere radius for the isochromat draw (m).
  distribution : {'uniform', 'sobol', 'halton'}
      Sampler — see :func:`_draw_in_sphere_offsets`.
  seed : int or None
      Base seed; trial ``t`` uses ``seed + t``.
  n_trials : int
      Independent repeats for mean/std estimation.

  Returns
  -------
  (mean, std) : tuple of float
      Mean and sample standard deviation of :math:`\\rho(K)` across
      ``n_trials`` independent draws.
  """
  k_sp = np.asarray(k_sp, dtype=np.float64).reshape(3)
  rhos = np.empty(int(n_trials), dtype=np.float64)
  for t in range(int(n_trials)):
    s = None if seed is None else int(seed) + t
    r = _draw_in_sphere_offsets(int(K), float(voxel_size),
                                distribution=distribution, seed=s)
    phase = 2.0 * np.pi * (r.astype(np.float64) @ k_sp)
    rhos[t] = np.abs(np.mean(np.exp(1j * phase)))
  if n_trials == 1:
    return float(rhos[0]), 0.0
  return float(rhos.mean()), float(rhos.std(ddof=0))


def plot_multi_isochromat_dephasing(
        idx,
        x_big,
        Mxy_big,
        Mxy_hist,
        K,
        x_original=None,
        elem_radius=None,
        t_index=None,
        show_positions=True,
        title_prefix="Isochromat Dephasing",
        show=True,
        export_to=None):
    """
    Visualizes the K isochromats from original FE node idx in the complex plane,
    together with the original node and element radius.

    Parameters
    ----------
    idx : int
        FE node index to inspect.
    x_big : array (N_big, dim)
        Enlarged coordinates from create_multi_isochromats().
    Mxy_big : array (N_big, 1)
        Initial transverse magnetization.
    Mxy_hist : array (N_big, n_time)
        Time-history of Mxy for all isochromats (complex).
    K : int
        Number of sub-isochromats per original node.
    x_original : array (N, dim), optional
        Original node coordinates. Only used for plotting reference.
    elem_radius : float, optional
        Radius for element visualization around original node.
    show : bool, optional
        Call ``plt.show()`` when done.
    export_to : str or path-like, optional
        Save the figure to this path.
    """

    # Rank-0 guarded, like `plot_isochromat_voxel`: without it every rank
    # opened its own window. There is no collective in this body, so no
    # Barrier is needed -- and adding one would create the hazard rather
    # than remove it.
    if MPI_rank != 0:
        return None

    # Determine which rows in x_big / Mxy_big correspond to node idx
    start = idx * K
    end   = start + K
    iso_slice = slice(start, end)

    # Pick the magnetizations to plot
    if t_index is None:
        M = Mxy_big[iso_slice, 0]
        title_t = "(initial)"
    else:
        if t_index >= Mxy_hist.shape[1]:
            raise IndexError(
                f"t_index={t_index} exceeds number of time points {Mxy_hist.shape[1]}"
            )
        M = Mxy_hist[iso_slice, t_index]
        title_t = f"(t index = {t_index})"

    # Prepare complex-plane coordinates
    Re = np.real(M)
    Im = np.imag(M)

    # Plot
    fig = plt.figure(figsize=(11, 5))

    # --- complex plane ---
    ax1 = fig.add_subplot(1, 2 if show_positions else 1, 1)
    ax1.scatter(Re, Im, s=60, c='blue', label='Isochromats')

    # Draw mean magnetization vector (spoiled result)
    M_mean = np.mean(M)
    ax1.scatter(np.real(M_mean), np.imag(M_mean),
                s=120, c='red', marker='x', label='Mean Mxy')

    ax1.arrow(0, 0, np.real(M_mean), np.imag(M_mean),
              head_width=0.02 * np.max(np.abs(Re + 1j * Im)),
              color='red', linewidth=1.8)

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Real(Mxy)")
    ax1.set_ylabel("Imag(Mxy)")
    ax1.set_aspect("equal", "box")
    ax1.set_title(f"{title_prefix} for node {idx} {title_t}\nComplex plane")
    ax1.legend()

    # Arrows for each isochromat
    rmax = np.max(np.abs(Re + 1j * Im))
    for r, im in zip(Re, Im):
        ax1.arrow(0, 0, r, im, head_width=0.02 * rmax,
                  length_includes_head=True, color="gray", alpha=0.4)

    # --- jittered positions (2nd subplot) ---
    if show_positions:
        x_node = x_big[iso_slice]  # (K, dim)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot jittered isochromats
        ax2.scatter(x_node[:, 0], x_node[:, 1], c='blue', s=50, label="Isochromats")

        # Plot original node
        if x_original is not None:
            x0 = x_original[idx]
            ax2.scatter([x0[0]], [x0[1]], c='black', s=80, marker='*', label="Original node")

            # Draw element radius as circle
            if elem_radius is not None:
                circle = Circle((x0[0], x0[1]), elem_radius,
                                fill=False, linestyle='--', edgecolor='red', linewidth=1.2)
                ax2.add_patch(circle)
                ax2.set_xlim(x0[0] - elem_radius * 1.5, x0[0] + elem_radius * 1.5)
                ax2.set_ylim(x0[1] - elem_radius * 1.5, x0[1] + elem_radius * 1.5)

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Isochromat jittered positions\n(with original node + element radius)")
        ax2.set_aspect("equal", "box")
        ax2.legend()

    plt.tight_layout()
    if export_to is not None:
        fig.savefig(export_to, bbox_inches='tight')
    if show:
        plt.show()
    return fig
