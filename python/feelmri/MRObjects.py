"""
MR hardware object definitions: scanner, gradient, and RF pulse.

:class:`Scanner` holds hardware limits (gradient strength, slew rate,
gyromagnetic ratio). :class:`Gradient` represents a trapezoidal or
user-defined gradient waveform. :class:`RF` generates analytic or
user-defined RF excitation pulses with flip-angle normalization.
"""
import collections
import copy
import itertools
import warnings

import matplotlib.pyplot as plt

import numpy as np
from pint import Quantity
from scipy.interpolate import interp1d

from feelmri.MPIUtilities import MPI, MPI_print, MPI_rank


class Scanner:
    """MRI scanner hardware specification.

    Parameters
    ----------
    field_strength : Quantity, optional
        Static field strength (T). Default is 1.5 T.
    gradient_strength : Quantity, optional
        Maximum gradient amplitude (mT/m). Default is 33 mT/m.
    gradient_slew_rate : Quantity, optional
        Maximum gradient slew rate (mT/m/ms). Default is 180 mT/m/ms.
    b1_max : Quantity, optional
        Peak transmit amplitude (mT). Default 0.025 mT (25 uT), a typical
        whole-body limit. Nothing enforces it implicitly -- it exists so
        :meth:`feelmri.Bloch.Sequence.check_hardware` has something to compare
        an imported pulse against.

    Attributes
    ----------
    gammabar : Quantity
        Gyromagnetic ratio for protons (42.58 MHz/T).
    gamma : Quantity
        Angular gyromagnetic ratio (2π × gammabar, rad·Hz/T).
    """
    def __init__(self, 
                field_strength: Quantity = Quantity(1.5, 'T'), 
                gradient_strength: Quantity = Quantity(33,'mT/m'),
                gradient_slew_rate: Quantity = Quantity(180,'mT/m/ms'),
                b1_max: Quantity = Quantity(0.060, 'mT'),
                rf_dead_time: Quantity = Quantity(0.0, 'ms'),
                rf_ringdown_time: Quantity = Quantity(0.0, 'ms'),
                adc_dead_time: Quantity = Quantity(0.0, 'ms')):
        self.field_strength = field_strength
        self.gradient_strength = gradient_strength
        self.gradient_slew_rate = gradient_slew_rate
        self.b1_max = b1_max
        # Transmit/receive dead times. A .seq file does not record them -- they are
        # a property of the scanner, not of the sequence -- so they arrive here.
        # They default to zero, which preserves the adapter's behaviour before they
        # existed; set them to a real spec and pypulseq's four dead-time checks go
        # live on import. Do not default them non-zero: the analytical fixtures are
        # authored with all three at zero on purpose.
        self.rf_dead_time = rf_dead_time
        self.rf_ringdown_time = rf_ringdown_time
        self.adc_dead_time = adc_dead_time
        self.gammabar = Quantity(42.58e6, 'Hz/T')
        self.gamma = Quantity(42.58e6*2*np.pi, 'rad*Hz/T')


#: What `BlochSolver` needs from a :class:`B0Field`. A NamedTuple rather than a
#: bare tuple so a consumer that has not been taught about a new channel fails
#: with an AttributeError instead of silently unpacking the wrong thing.
SolverTerms = collections.namedtuple(
    'SolverTerms', 'offset_mT delta_B gradient quadratic node_gradient')

#: The same for the signal evaluator, with the rates already in rad/ms.
ReadoutTerms = collections.namedtuple(
    'ReadoutTerms', 'phi_uniform phi_nodal gradient quadratic node_gradient')


class B0Field:
    """Scanner-fixed B0 inhomogeneity, sampled at the spin's CURRENT position.

    This is the lab-frame counterpart of ``BlochSolver(delta_B=)`` and
    ``FEMPhantom.set_static_fields(phi_dB0=)``. Those are one value per mesh
    node, frozen onto the node, so they travel with the tissue -- right for
    chemical shift and local susceptibility, wrong for a shim residual or a
    main-field imperfection, which stay where the magnet put them. A spin that
    moves through this field samples a different value; a spin that moves
    through ``delta_B`` does not.

    The field is given as an expression of position and carried as a
    low-order expansion:

        dB0(x) = offset + gradient . x        [mT]

    with ``x`` in metres, in the SCANNER frame, measured from isocentre -- not
    the imaging frame, and not relative to the slice location.
    :meth:`in_frame` converts to whatever frame a caller works in.

    Why an expansion is not a compromise here: a static field in a
    current-free bore satisfies Laplace's equation, so it IS a solid-harmonic
    series, and that is the basis shim hardware is specified in. Truncation
    order is the only approximation, and :attr:`residual_rms` reports it.
    Anything the series cannot represent is not a smooth scanner field -- it is
    tissue structure, and it belongs on ``delta_B``.

    Parameters
    ----------
    offset : Quantity
        Uniform part, mT.
    gradient : Quantity
        Linear part, mT/m, a 3-vector in the scanner frame.
    order : int, optional
        Expansion order this object carries (0 or 1).
    residual_rms : Quantity, optional
        What :meth:`fit` could not represent, mT. Zero for a hand-built field.
    """

    #: The highest degree the SOLVER AND READOUT can carry, not the highest the
    #: fit could find. The kernel's quadratic channel is six coefficients and
    #: the assembler's `maxwell` channel is six monomials, so degree 2 is the
    #: ceiling; detecting a cubic here would only let it be truncated to a
    #: quadratic silently, which measured as losing 100% of a Z3 shim. A field
    #: above the cap falls through to the per-node expansion instead, which is
    #: exact on a static phantom and first order on a moving one.
    MAX_ORDER = 2
    #: A residual this far below the field RMS means the expression IS a
    #: polynomial of that degree, not merely well approximated by one, so the
    #: search stops there rather than at `rtol`. The floor is sqrt(eps), not
    #: eps: the normal equations square the condition number, the same reason
    #: `pod-motion.md` records a 1.5e-8 floor for the method of snapshots.
    #: Measured separation is seven orders -- a true polynomial lands at 0 to
    #: 1e-8, the nearest miss (a degree-4 field fitted at degree 3) at 4.7e-01.
    EXACT_RTOL = 1.0e-7

    def __init__(self, offset=Quantity(0.0, 'mT'),
                 gradient=Quantity(np.zeros(3), 'mT/m'),
                 *, order=None, residual_rms=Quantity(0.0, 'mT'),
                 coefficients=None):
        self.offset_mT = float(Quantity(offset).m_as('mT'))
        g = np.asarray(Quantity(gradient).m_as('mT/m'), dtype=np.float64).reshape(-1)
        if g.size != 3:
            raise ValueError(
                f"B0Field: gradient must be a 3-vector in mT/m, got {g.size} "
                f"entries.")
        self.gradient_mT_per_m = g
        self.residual_rms_mT = float(Quantity(residual_rms).m_as('mT'))
        if order is None:
            order = 1 if np.any(g) else 0
        self.order = int(order)
        # The whole coefficient vector, laid out by `_monomial_exponents`. The
        # constant and linear parts are mirrored on `offset_mT` and
        # `gradient_mT_per_m` because those two are what the free kernel path
        # consumes; everything above degree 1 is only readable here.
        if coefficients is None:
            coefficients = np.concatenate(([self.offset_mT], g))[
                :len(self._monomial_exponents(self.order))]
        self.coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        # Per-node fallback, set by `on_phantom` when no polynomial fits. The
        # stamp records which node set it was built on -- a per-node array
        # means nothing under a different partition or ordering.
        self._nodal_mT = None
        self._nodal_grad = None
        self._nodal_stamp = None
        self._expression = None
        self._mesh_residual_mT = 0.0

    @property
    def kind(self):
        """``'uniform'``, ``'linear'``, ``'polynomial'`` or ``'nodal'``.

        Names what a consumer has to be able to carry, not how the field was
        written: the first two ride channels that cost nothing, the third needs
        the monomial form, and the fourth is per node.
        """
        if self._nodal_mT is not None:
            return 'nodal'
        if self.order >= 2:
            return 'polynomial'
        return 'linear' if np.any(self.gradient_mT_per_m) else 'uniform'

    def nodal_mT(self, phantom):
        """The per-node field values, checked against the node set they describe.

        Raises rather than returning a stale array: a per-node quantity paired
        with the wrong partition is a plausible wrong answer with no symptom,
        which is the failure this whole class exists to avoid.
        """
        if self._nodal_mT is None:
            raise TypeError(
                f"B0Field.nodal_mT: this field is a {self.kind} expansion, not "
                f"a per-node one; read `coefficients` or `in_frame` instead.")
        stamp = self._node_stamp(phantom)
        if stamp != self._nodal_stamp:
            raise ValueError(
                f"B0Field.nodal_mT: this field was sampled on a different node "
                f"set ({self._nodal_stamp}) from the one asking for it "
                f"({stamp}). Repartitioning or a second phantom invalidates a "
                f"per-node array; rebuild it with `B0Field.on_phantom`.")
        return self._nodal_mT

    def quadratic_mT_per_m2(self):
        """The six degree-2 coefficients as ``(xx, yy, zz, xy, xz, yz)``.

        Zeros when the field carries no quadratic part, so a caller may add
        them unconditionally.
        """
        out = np.zeros(6, dtype=np.float64)
        if self.order >= 2:
            out[:] = self.coefficients[4:10]
        return out

    def __repr__(self):
        return (f"B0Field(offset={self.offset_mT:.6g} mT, "
                f"gradient={np.round(self.gradient_mT_per_m, 9)} mT/m, "
                f"order={self.order})")

    @property
    def is_zero(self):
        """True when the field is identically zero, so callers can skip it.

        Every representation has to be checked, not just the constant and the
        gradient: a purely quadratic field has both of those zero, and reading
        it as absent would skip the whole channel and silently return the
        no-field answer.
        """
        if self._nodal_mT is not None:
            return not np.any(self._nodal_mT)
        return (self.offset_mT == 0.0 and not np.any(self.gradient_mT_per_m)
                and not np.any(self.coefficients))

    def __call__(self, points_m):
        """Evaluate the expansion at scanner-frame positions, in mT."""
        p = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        return self.offset_mT + p @ self.gradient_mT_per_m

    def in_frame(self, rotation=None, location=None, physical=False):
        """The same field rewritten for the coordinates a caller works in.

        ``FEMPhantom.orient`` leaves ``x_used = M.T @ (x_scanner - LOC)``, and
        ``BlochSolver`` rotates back to ``x_scanner - LOC`` when it evaluates
        the concomitant term. Substituting either into ``b + g . x_scanner``
        gives the same constant and a rotated gradient:

            b_used = b + g . LOC
            g_used = g           (physical=True, x_used = x_scanner - LOC)
                   = M.T @ g     (physical=False, x_used = M.T (x_scanner - LOC))

        So the slice offset lands entirely in the constant -- a linear field is
        origin-correct without any change to the node array.

        Returns ``(offset_mT, gradient_mT_per_m)``.
        """
        b, g, q = self.in_frame_full(rotation=rotation, location=location,
                                     physical=physical)
        if np.any(q):
            raise NotImplementedError(
                f"B0Field.in_frame: this field is degree {self.order}, and the "
                f"constant and linear parts alone are not it -- returning them "
                f"would silently truncate the quadratic part. Use "
                f"`in_frame_full`, which returns all three, or the "
                f"`solver_terms` / `readout_terms` accessors.")
        return b, g

    @staticmethod
    def _quad_matrix(v):
        """(xx, yy, zz, xy, xz, yz) -> the symmetric matrix of the same form.

        The off-diagonals are halved because the vector spells the field as
        ``Qxx x^2 + ... + Qxy xy``, so ``x^T M x`` only reproduces it when the
        cross terms are split between the two symmetric slots.
        """
        xx, yy, zz, xy, xz, yz = (float(c) for c in v)
        return np.array([[xx, 0.5 * xy, 0.5 * xz],
                         [0.5 * xy, yy, 0.5 * yz],
                         [0.5 * xz, 0.5 * yz, zz]], dtype=np.float64)

    @staticmethod
    def _quad_vector(m):
        """Inverse of :meth:`_quad_matrix`."""
        return np.array([m[0, 0], m[1, 1], m[2, 2],
                         2.0 * m[0, 1], 2.0 * m[0, 2], 2.0 * m[1, 2]],
                        dtype=np.float64)

    def in_frame_full(self, rotation=None, location=None, physical=False):
        """Constant, gradient and quadratic form in the caller's frame.

        Writing the scanner position as ``x_s = R x + L`` and expanding,

            b' = b + g.L + L^T Q L
            g' = R^T (g + 2 Q L)
            Q' = R^T Q R

        so the slice offset moves down into the lower orders rather than being
        lost -- the same algebra that re-centres the concomitant term, and the
        reason a degree-2 field cannot simply reuse the linear adapter. ``R^T``
        is dropped when ``physical``, because the caller's frame is then already
        the scanner's apart from the translation.

        Returns ``(offset_mT, gradient_mT_per_m, quadratic_mT_per_m2)`` with the
        quadratic as ``(xx, yy, zz, xy, xz, yz)``, all zeros below degree 2.
        """
        if self.order >= 3:
            raise NotImplementedError(
                f"B0Field.in_frame_full: this field is degree {self.order}, and "
                f"a constant, a gradient and a quadratic form are not it -- "
                f"returning them would drop every higher monomial. Measured on "
                f"a Z3 shim, that is 100% of the field. Build it with "
                f"`B0Field.on_phantom`, which falls back to the per-node "
                f"expansion above the degree the channels can carry.")
        b = self.offset_mT
        g = np.asarray(self.gradient_mT_per_m, dtype=np.float64).copy()
        Q = self._quad_matrix(self.quadratic_mT_per_m2())
        if location is not None:
            L = np.asarray(location, dtype=np.float64).reshape(3)
            b = b + float(g @ L) + float(L @ Q @ L)
            g = g + 2.0 * (Q @ L)
        if not physical and rotation is not None:
            R = np.asarray(rotation, dtype=np.float64)
            g = R.T @ g
            Q = R.T @ Q @ R
        return (b, np.ascontiguousarray(g, dtype=np.float64),
                self._quad_vector(Q))

    def solver_terms(self, phantom, moving, rotation=None, location=None,
                     physical=False):
        """What `BlochSolver` needs from this field, in the frame it works in.

        Returns ``(offset_mT, delta_B_mT, gradient, quadratic)``: a uniform
        part to fold into `delta_B`, a per-node array to fold into it instead, a
        3-vector for the hoisted gradient scalars, and the six quadratic
        coefficients. `delta_B_mT` and `gradient` are never both set.

        `moving` is the caller's, not this object's: the solver knows whether
        it has a `pod_trajectory` and the readout knows separately whether it
        was given a `pod`, and the two may legitimately disagree. A phantom
        that does not move samples the field at `x0` for the whole solve, so a
        per-node value IS the Eulerian answer there -- exact, at no cost,
        however rough the field.
        """
        if self.kind != 'nodal':
            b, g, q = self.in_frame_full(rotation=rotation, location=location,
                                         physical=physical)
            if np.any(q) and not moving:
                # A phantom that does not move samples the field at x0 for the
                # whole solve, so the quadratic part is a CONSTANT per node and
                # folds into `delta_B` -- exact, and it needs no kernel channel
                # at all. Only the linear case is left on the hoisted gradient
                # scalars, so nothing that works today changes path.
                x = np.asarray(phantom.local_nodes, dtype=np.float64)
                if physical and rotation is not None:
                    x = x @ np.asarray(rotation, dtype=np.float64).T
                M = self._quad_matrix(q)
                vals = (b + x @ g
                        + np.einsum('ij,jk,ik->i', x, M, x))
                return SolverTerms(0.0, vals.reshape(-1, 1), None, None, None)
            return SolverTerms(b, None, (g if np.any(g) else None),
                               (q if np.any(q) else None), None)
        nodal = self.nodal_mT(phantom)
        if not moving:
            return SolverTerms(0.0, nodal.reshape(-1, 1), None, None, None)
        # Eulerian, to first order in the displacement. Writing
        #   dB0(x0 + u) ~= [dB0(x0) - g.x0] + curr . g
        # puts the bracket on `delta_B`, where it costs nothing, and leaves a
        # per-node vector the kernel adds to the gradient scalars it already
        # hoists. `g` follows the same frame rule as the polynomial gradient.
        g = self.node_gradient(phantom, rotation=rotation, physical=physical)
        x = np.asarray(phantom.local_nodes, dtype=np.float64)
        if physical and rotation is not None:
            x = x @ np.asarray(rotation, dtype=np.float64).T
        bracket = nodal - np.einsum('ij,ij->i', x, g)
        return SolverTerms(0.0, bracket.reshape(-1, 1), None, None, g)

    def readout_terms(self, phantom, scanner, moving, rotation=None,
                      location=None):
        """What the signal evaluator needs, in the imaging frame it works in.

        Returns ``(phi_uniform, phi_nodal, gradient, quadratic)`` with the
        off-resonance rates in rad/ms and the quadratic still in mT/m^2. The
        assembler's nodes are always the imaging ones, so unlike the solver
        there is no physical-frame case here.
        """
        gamma = scanner.gamma.m_as('rad/ms/mT')
        if self.kind != 'nodal':
            b, g, q = self.in_frame_full(rotation=rotation, location=location)
            return ReadoutTerms(gamma * b, None, (g if np.any(g) else None),
                                (q if np.any(q) else None), None)
        nodal = self.nodal_mT(phantom)
        if not moving:
            return ReadoutTerms(0.0, gamma * nodal, None, None, None)
        # Unlike the solver's, this expansion is written against the
        # DISPLACEMENT rather than the absolute position:
        #   dB0(x0 + u) ~= dB0(x0) + g . u
        # so `phi_nodal` is the field itself and the gradient rides its own
        # channel. The solver's kernel has no choice -- its gradient scalars
        # multiply the absolute `curr` -- but the assembler interpolates both
        # arrays through the element shape functions, and interpolating
        # `g . x0` separately from `g` leaves an artifact that does not vanish
        # at rest. The assembler's nodes are always the imaging ones, so
        # `physical` has no meaning here and the gradient always takes R^T.
        g = self.node_gradient(phantom, rotation=rotation, physical=False)
        return ReadoutTerms(0.0, gamma * nodal, None, None, gamma * g)

    def phi_offset(self, scanner, location=None):
        """The uniform part as an off-resonance rate in rad/ms, to add to
        ``phi_dB0``.

        Uniform in space, so it carries no rotation -- but it DOES depend on
        where isocentre is: a slice offset turns part of the gradient into a
        constant, so pass the same ``location`` :meth:`in_frame` gets or a
        shifted slice loses that term.
        """
        b, _g = self.in_frame(location=location)
        return float(b * scanner.gamma.m_as('rad/ms/mT'))

    @classmethod
    def fit(cls, expression, points_m, *, rtol=1.0e-3, max_order=None,
            collective=True):
        """Expand ``expression`` into the lowest order that represents it.

        ``expression`` takes an ``(N, 3)`` array of SCANNER-frame positions in
        metres and returns ``(N,)`` in mT, or a pint Quantity. Orders 0, 1, 2
        are tried in turn and the first whose residual RMS falls below
        ``rtol`` times the field RMS is kept.

        Under MPI the normal equations are accumulated locally and reduced, so
        every rank solves the same system and gets a bit-identical field.
        Fitting a gathered map per rank would let ranks disagree and the solve
        would be silently inconsistent.
        """
        from feelmri.MPIUtilities import MPI_comm

        p = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        values = expression(p)
        if isinstance(values, Quantity):
            values = values.m_as('mT')
        f = np.asarray(values, dtype=np.float64).reshape(-1)
        if f.size != p.shape[0]:
            raise ValueError(
                f"B0Field.fit: the expression returned {f.size} values for "
                f"{p.shape[0]} points; it must map (N, 3) positions to (N,).")

        max_order = cls.MAX_ORDER if max_order is None else int(max_order)
        n_tot = float(p.shape[0])
        ff = float(f @ f)
        if collective:
            n_tot = MPI_comm.allreduce(n_tot, op=MPI.SUM)
            ff = MPI_comm.allreduce(ff, op=MPI.SUM)
        rms = np.sqrt(ff / n_tot) if n_tot else 0.0

        best = None
        resid_by_order = {}
        for order in range(0, max_order + 1):
            # An underdetermined fit is exact and meaningless: 20 monomials
            # through 5 points reproduces them all and says nothing about the
            # field anywhere else. The count that matters is the GLOBAL one,
            # since the normal equations are reduced across ranks.
            n_terms = len(cls._monomial_exponents(order))
            if order > 0 and n_terms >= n_tot:
                break
            basis = cls._design(p, order)
            # Normal equations, reduced BEFORE the solve so every rank solves
            # an identical system.
            A = basis.T @ basis
            b = basis.T @ f
            if collective:
                A = MPI_comm.allreduce(A, op=MPI.SUM)
                b = MPI_comm.allreduce(b, op=MPI.SUM)
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
            # |f - Phi c|^2 = f.f - 2 c.b + c.A.c, from the reduced pieces.
            resid = max(ff - 2.0 * float(coef @ b) + float(coef @ A @ coef), 0.0)
            resid_rms = np.sqrt(resid / n_tot) if n_tot else 0.0
            resid_by_order[order] = resid_rms
            best = (order, coef, resid_rms)
            # Two stopping rules, and the first is the one that matters. A
            # residual at round-off means the expression IS this polynomial --
            # every shim is -- so it is carried exactly and there is nothing to
            # gain from a higher degree. `rtol` is the weaker rule for a field
            # that is only well approximated.
            if (resid_rms <= cls.EXACT_RTOL * rms or rms == 0.0
                    or resid_rms <= rtol * rms):
                break

        order, coef, resid_rms = best
        if resid_rms > rtol * rms and rms > 0.0:
            raise ValueError(
                f"B0Field.fit: degree {order} still leaves a residual of "
                f"{resid_rms:.4g} mT against a field RMS of {rms:.4g} mT "
                f"({resid_rms / rms:.1%}), above rtol={rtol:g}. No polynomial "
                f"up to the degree {int(n_tot):d} points can support "
                f"represents this field, so it needs the per-node expansion "
                f"instead of the global one -- build it with "
                f"`B0Field.on_phantom`, which falls back to that.")

        gradient = np.zeros(3)
        if order >= 1:
            gradient = coef[1:4]
        return cls(Quantity(float(coef[0]), 'mT'),
                   Quantity(gradient, 'mT/m'),
                   order=order, residual_rms=Quantity(resid_rms, 'mT'),
                   coefficients=coef)

    @classmethod
    def on_phantom(cls, expression, phantom, *, nodal='auto',
                   gradient=None, fd_step=None, **kwargs):
        """Build the cheapest representation of `expression` this phantom needs.

        The phantom's `local_nodes` are imaging-frame and measured from the
        slice centre once `orient` has run, so they are mapped back with the
        stored `_orientation` and `_location` before the expression sees them --
        the expression always works in SCANNER coordinates.

        A global polynomial is tried first, because it costs nothing per node
        and is exact for anything polynomial, which every shim is. Only when no
        polynomial up to the cap represents the field does this fall back to
        sampling it per node. `nodal=False` restores the older behaviour of
        refusing instead.
        """
        nodes = cls._scanner_nodes(phantom)
        try:
            return cls.fit(expression, nodes, **kwargs)
        except ValueError:
            if not nodal:
                raise
        field = cls(coefficients=np.zeros(1))
        field._nodal_mT = cls._sample(expression, nodes)
        field._nodal_grad = cls._sample_gradient(
            expression, nodes, gradient=gradient, fd_step=fd_step,
            collective=kwargs.get('collective', True))
        field._nodal_stamp = cls._node_stamp(phantom)
        field._expression = expression
        field._mesh_residual_mT = cls._mesh_residual(expression, phantom, nodes)
        return field

    @staticmethod
    def _scanner_nodes(phantom):
        """`phantom.local_nodes` mapped back into scanner coordinates."""
        nodes = np.asarray(phantom.local_nodes, dtype=np.float64)
        rotation = getattr(phantom, '_orientation', None)
        location = getattr(phantom, '_location', None)
        if rotation is not None:
            nodes = nodes @ np.asarray(rotation, dtype=np.float64).T
        if location is not None:
            nodes = nodes + np.asarray(location, dtype=np.float64).reshape(3)
        return nodes

    @staticmethod
    def _node_stamp(phantom):
        """Identifies the node set a per-node array was built on.

        A per-node array is only meaningful against the partition and the node
        ordering that produced it, and `distribute_mesh` / `enable_dual_partition`
        change both. Cheap enough to re-check at every use.
        """
        nodes = np.asarray(phantom.local_nodes)
        return (getattr(phantom, '_active_partition', None), nodes.shape,
                float(nodes[0, 0]), float(nodes[-1, -1]))

    @staticmethod
    def _sample(expression, points):
        """`expression` at `points`, in mT, with the same contract `fit` uses."""
        values = expression(points)
        if isinstance(values, Quantity):
            values = values.m_as('mT')
        f = np.asarray(values, dtype=np.float64).reshape(-1)
        if f.size != points.shape[0]:
            raise ValueError(
                f"B0Field: the expression returned {f.size} values for "
                f"{points.shape[0]} points; it must map (N, 3) positions to "
                f"(N,).")
        return f

    def node_gradient(self, phantom, rotation=None, physical=False):
        """The per-node field gradient, in the frame the caller works in.

        Built once and cached on the field, and checked against the node set it
        was sampled on for the same reason :meth:`nodal_mT` is.
        """
        if self._nodal_grad is None:
            raise TypeError(
                f"B0Field.node_gradient: this field is a {self.kind} expansion, "
                f"which has no per-node gradient.")
        stamp = self._node_stamp(phantom)
        if stamp != self._nodal_stamp:
            raise ValueError(
                f"B0Field.node_gradient: sampled on a different node set "
                f"({self._nodal_stamp}) from the one asking for it ({stamp}).")
        g = self._nodal_grad
        if not physical and rotation is not None:
            g = g @ np.asarray(rotation, dtype=np.float64)
        return np.ascontiguousarray(g, dtype=np.float64)

    @classmethod
    def _sample_gradient(cls, expression, points, gradient=None, fd_step=None,
                         collective=True):
        """grad(expression) at `points`, in mT/m, in SCANNER coordinates.

        An analytic `gradient` is used when given. Otherwise central
        differences, with the step CHECKED rather than assumed: a finite
        difference of a callable that is not differentiable produces a
        confident number that is entirely wrong, and that is exactly the input
        this class refuses in prose. Comparing the estimate at `h` against the
        one at `2h` separates the two -- a smooth field moves by its O(h^2)
        truncation, a kink or a lookup moves by the order of the gradient
        itself.
        """
        from feelmri.MPIUtilities import MPI_comm, collective_raise

        if gradient is not None:
            g = gradient(points)
            if isinstance(g, Quantity):
                g = g.m_as('mT/m')
            g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
            if g.shape[0] != points.shape[0]:
                raise ValueError(
                    f"B0Field: the gradient returned {g.shape[0]} rows for "
                    f"{points.shape[0]} points; it must map (N, 3) to (N, 3).")
            return g

        h = 1.0e-4 if fd_step is None else float(fd_step)

        def central(step):
            out = np.empty(points.shape, dtype=np.float64)
            for axis in range(3):
                e = np.zeros(3, dtype=np.float64)
                e[axis] = step
                out[:, axis] = (cls._sample(expression, points + e)
                                - cls._sample(expression, points - e)) / (2 * step)
            return out

        g = central(h)
        drift = float(np.abs(g - central(2.0 * h)).max())
        scale = float(np.abs(g).max())
        if collective:
            drift = MPI_comm.allreduce(drift, op=MPI.MAX)
            scale = MPI_comm.allreduce(scale, op=MPI.MAX)
        problem = None
        if scale > 0.0 and drift > 0.01 * scale:
            problem = (
                f"B0Field: the finite-difference gradient of this expression "
                f"moves by {drift / scale:.1%} between a step of {h:g} m and "
                f"one of {2 * h:g} m. A smooth field moves by its truncation "
                f"error, which is far smaller; this much means the expression "
                f"is not differentiable -- a lookup, a step, or interpolated "
                f"data. Pass an analytic `gradient=`, or put the field on "
                f"`delta_B`, which needs no derivative.")
        collective_raise(problem, ValueError)
        return g

    @classmethod
    def _mesh_residual(cls, expression, phantom, nodes):
        """How much of the field varies WITHIN one element, in mT.

        A per-node field reaches the readout through the shape functions, so
        whatever it does between nodes is not represented at all. Comparing the
        expression at each element centroid against the mean of that element's
        nodal values measures exactly that -- and unlike an interpolant it needs
        no per-cell-type basis, so it is valid for every element the mesh may
        hold. Returns 0.0 when the connectivity is not available.
        """
        elems = getattr(phantom, 'local_elements', None)
        if elems is None or len(elems) == 0:
            return 0.0
        elems = np.asarray(elems)
        centroids = nodes[elems].mean(axis=1)
        nodal = cls._sample(expression, nodes)
        return float(np.abs(cls._sample(expression, centroids)
                            - nodal[elems].mean(axis=1)).max())

    @staticmethod
    def _monomial_exponents(order):
        """Exponent triples up to `order`, in the order the columns are laid out.

        Degrees 0-2 are spelled out rather than generated, so the layout stays
        ``1 | x y z | x^2 y^2 z^2 xy xz yz`` -- `coefficients[1:4]` is the
        gradient and the six quadratic slots match the order the assembler's
        `maxwell` channel expects. Degree 3 and up are generated.
        """
        exps = [(0, 0, 0)]
        if order >= 1:
            exps += [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        if order >= 2:
            exps += [(2, 0, 0), (0, 2, 0), (0, 0, 2),
                     (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        for d in range(3, order + 1):
            for combo in itertools.combinations_with_replacement(range(3), d):
                e = [0, 0, 0]
                for a in combo:
                    e[a] += 1
                exps.append(tuple(e))
        return exps

    @classmethod
    def _design(cls, points, order):
        """Monomials up to `order`, as columns."""
        cols = []
        for ex, ey, ez in cls._monomial_exponents(order):
            cols.append(points[:, 0] ** ex * points[:, 1] ** ey
                        * points[:, 2] ** ez)
        return np.stack(cols, axis=1)


class Gradient:
    """Gradient waveform for trapezoidal, triangular, or user-supplied shapes.

    The gradient may be defined either analytically (via slope, strength, and
    plateau length) or explicitly (via supplied ``timings`` and ``amplitudes``).
    If both ``timings`` and ``amplitudes`` are provided, the constructor
    immediately builds the interpolator and **skips all analytic gradient
    construction**.

    Parameters
    ----------
    slope : Quantity or None, optional
        Duration of each ramp (ms). If None, derived from gradient amplitude
        and slew rate.
    lenc : Quantity, optional
        Duration of the flat-top portion of the gradient (ms). Default is 1 ms.
    strength : Quantity or None, optional
        Gradient amplitude (mT/m). If None, the maximum scanner gradient is used.
    scanner : Scanner, optional
        Scanner object containing hardware limits.
    ref : Quantity, optional
        Reference time (ms) relative to the sequence block. Default is 0 ms.
    time : Quantity, optional
        Absolute start time (ms) of the gradient within the sequence. Default is 0 ms.
    axis : int, optional
        Gradient axis (0 = M, 1 = P, 2 = S). Default is 0.
    timings : Quantity or None, optional
        Explicit waveform timing samples. If provided together with ``amplitudes``,
        analytic construction is bypassed.
    amplitudes : Quantity or None, optional
        Explicit waveform amplitude samples corresponding to ``timings``.

    Notes
    -----
    When ``timings`` and ``amplitudes`` are supplied, all analytic parameters
    (slope, lenc, strength) are stored but *not used* for waveform generation.
    The total duration is inferred from the last timing sample.
    """

    def __init__(
        self,
        slope=None,
        lenc=Quantity(1.0, "ms"),
        strength=None,
        scanner=None,
        ref=Quantity(0.0, "ms"),
        time=Quantity(0.0, "ms"),
        axis=0,
        timings=None,
        amplitudes=None,
    ):

        # A default-constructed Scanner() in the signature would be ONE shared
        # instance for every gradient built without an explicit scanner, so
        # mutating one gradient's scanner would change the hardware limits seen
        # by all of them.
        self.scanner = Scanner() if scanner is None else scanner
        scanner = self.scanner
        self.Gr_max = scanner.gradient_strength      # [mT/m]
        self.Gr_sr = scanner.gradient_slew_rate      # [mT/m/ms]

        self.ref = ref.to("ms")
        self.time = time.to("ms")
        self.axis = axis
        self.user_defined = False

        # ------------------------------------------------------------------
        # USER-SUPPLIED TIMINGS (override analytic construction)
        # ------------------------------------------------------------------
        if timings is not None and amplitudes is not None:
            """
            When the user directly supplies timing and amplitude samples,
            the gradient shape is defined entirely by these arrays.

            The duration is inferred from the last timepoint.
            All analytic construction (slope/lenc/strength) is bypassed.
            """

            self.timings = timings
            self.amplitudes = amplitudes

            # Duration of the gradient relative to start time and reference
            self.dur = (self.timings[-1] - self.time + self.ref).to("ms")
            self.dur2 = (self.dur - self.ref).to("ms")

            # Build interpolator from supplied samples
            self.interpolator = interp1d(
                self.timings.m,
                self.amplitudes.m,
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )

            # Flag: user-defined gradient parameters
            self.user_defined = True

            # Preserve user-provided parameters but do not use them
            self.slope = slope
            self.lenc = lenc
            self.strength = strength
            return

        # ------------------------------------------------------------------
        # ANALYTIC GRADIENT CONSTRUCTION
        # ------------------------------------------------------------------
        self.strength = self.Gr_max if strength is None else strength
        self.lenc = lenc

        # Determine slope from amplitude and slew rate if not provided
        self.slope = (
            np.abs(self.strength) / self.Gr_sr if slope is None else slope
        )

        # Duration of full trapezoid
        if self.lenc <= 0.0:
            self.dur = (2 * self.slope).to("ms")
        else:
            self.dur = (self.slope + self.lenc + self.slope).to("ms")

        self.dur2 = (self.dur - self.ref).to("ms")

        # Compute default timing/amplitude arrays
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Representations and arithmetic
    # ======================================================================
    def __copy__(self):
        """Return a deep copy of the gradient object."""
        return copy.deepcopy(self)

    def __repr__(self):
        """String representation including major gradient parameters."""
        return (
            f"Gradient(slope={self.slope}, lenc={self.lenc}, strength={self.strength}, "
            f"Gr_max={self.Gr_max}, Gr_sr={self.Gr_sr}, ref={self.ref}, "
            f"time={self.time}, dur={self.dur}, axis={self.axis})"
        )

    def __call__(self, t):
        """Evaluate the gradient at time `t` using the internal interpolator."""
        return self.interpolator(t)

    def __mul__(self, other):
        """Multiply the gradient amplitude by a scalar.

        Parameters
        ----------
        other : float
            Scalar multiplier.

        Returns
        -------
        Gradient
            A new gradient object with scaled amplitude.

        Notes
        -----
        Only numerical scalars are permitted. This returns a *new* object.
        """
        if isinstance(other, bool):
            raise TypeError("Gradient cannot be multiplied by a bool.")
        if isinstance(other, (int, float, np.number)):
            if self.user_defined:
                # A user-supplied gradient has no slope/lenc/strength to
                # rebuild from -- scale the samples it actually carries.
                return Gradient(
                    timings=self.timings,
                    amplitudes=self.amplitudes * other,
                    scanner=self.scanner,
                    ref=self.ref,
                    time=self.time,
                    axis=self.axis,
                )
            return Gradient(
                slope=self.slope,
                lenc=self.lenc,
                strength=self.strength * other,
                scanner=self.scanner,
                ref=self.ref,
                time=self.time,
                axis=self.axis,
            )
        raise TypeError("Gradient can only be multiplied by a scalar (int or float).")

    # ======================================================================
    # Gradient Construction Helpers
    # ======================================================================
    def evaluate(self, t):
        """Evaluate the gradient interpolator at a given time point.

        Parameters
        ----------
        t : float
            Time at which to evaluate the gradient (ms).

        Returns
        -------
        float
            Gradient amplitude at time ``t`` (mT/m).
        """
        return self.interpolator(t)

    def group_timings(self):
        """Generate timing and amplitude arrays for a trapezoidal or triangular gradient.

        Returns
        -------
        tuple
            ``(timings, amplitudes, interpolator)`` where timings are offset
            by ``(time - ref)`` and the interpolator uses linear interpolation.
        """

        if self.lenc <= 0.0:
            # Triangular gradient
            timings = Quantity(
                np.array(
                    [0.0, self.slope.m, self.slope.m + self.slope.m],
                    dtype=np.float64,
                ),
                self.slope.u,
            )
            amplitudes = Quantity(
                np.array([0.0, self.strength.m, 0.0], dtype=np.float64),
                self.strength.u,
            )
        else:
            # Trapezoidal gradient
            timings = Quantity(
                np.array(
                    [
                        0.0,
                        self.slope.m,
                        self.slope.m + self.lenc.m,
                        self.slope.m + self.lenc.m + self.slope.m,
                    ],
                    dtype=np.float64,
                ),
                self.slope.u,
            )
            amplitudes = Quantity(
                np.array(
                    [0.0, self.strength.m, self.strength.m, 0.0],
                    dtype=np.float64,
                ),
                self.strength.u,
            )

        # Shift timing by sequence offsets. NOT in place: `timings` was float32
        # until 2026-09-10 and `+=` kept that dtype under numpy's same-kind
        # casting, quantising the absolute sequence time -- 2.7e-5 ms of corner
        # error at t = 1000 ms, enough to put a raster point past the corner it
        # was meant to name. The arrays are four entries long, so float64 costs
        # nothing worth counting.
        timings = timings + (self.time - self.ref)

        interpolator = interp1d(
            timings.m,
            amplitudes.m,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )

        return timings, amplitudes, interpolator

    # ======================================================================
    # Time & Reference Adjustment
    # ======================================================================
    def change_ref(self, ref):
        """Update the reference time of the gradient.

        Parameters
        ----------
        ref : Quantity
            New reference time (ms).
        """
        self.ref = ref.to("ms")
        self.dur2 = (self.dur - self.ref).to("ms")
        # group_timings places the waveform at (time - ref), so a changed
        # reference must rebuild it -- otherwise only dur2 moves and the
        # interpolator keeps sampling the old timeline. RF.change_ref has
        # always rebuilt; this is the same contract.
        if not self.user_defined:
            self.timings, self.amplitudes, self.interpolator = self.group_timings()

    def change_time(self, time):
        """Update the absolute time of the gradient and rebuild timing arrays.

        For analytic gradients the trapezoid is regenerated via
        :meth:`group_timings`. For user-supplied gradients the existing
        ``timings`` array is shifted by the delta between the new and
        previous absolute time (preserving the original raster shape),
        and the interpolator is rebuilt explicitly so the shifted
        timeline is authoritatively used regardless of the scipy
        version's input-handling convention.

        Parameters
        ----------
        time : Quantity
            New absolute start time (ms).
        """
        new_time = time.to("ms")
        if self.user_defined:
          delta = new_time - self.time
          self.time = new_time
          self.timings = self.timings + delta
          self.interpolator = interp1d(
            self.timings.m,
            self.amplitudes.m,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
          )
        else:
          self.time = new_time
          self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Gradient Calculation Based on Bandwidth (original code preserved)
    # ======================================================================
    def calculate(self, k_bw, receiver_bw=None, ro_samples=None, ofac=None):
        """Calculate gradient shape from k-space bandwidth and scanner constraints.

        Parameters
        ----------
        k_bw : Quantity
            k-space bandwidth (1/m).
        receiver_bw : Quantity, optional
            Receiver bandwidth (Hz). If provided, the flat-top duration is
            fixed to accommodate the ADC window.
        ro_samples : int, optional
            Number of readout samples (required when ``receiver_bw`` is set).
        ofac : float, optional
            Oversampling factor (required when ``receiver_bw`` is set).

        Notes
        -----
        Updates ``slope``, ``lenc``, ``strength``, and the total ``dur`` in place.
        """

        if receiver_bw is not None:
            # Fixed flat-top duration from receiver bandwidth
            self.lenc = (ro_samples / ofac) / receiver_bw.to("1/ms")
            self.strength = (
                k_bw.to("1/m") /
                (self.scanner.gammabar.to("1/mT/ms") * self.lenc.to("ms"))
            )

            # Enforce hardware amplitude limit
            if self.strength > self.Gr_max:
                self.strength = self.Gr_max
                self.lenc = (
                    k_bw.to("1/m") /
                    (
                        self.scanner.gammabar.to("1/mT/s") *
                        self.strength.to("mT/m")
                    )
                )
                receiver_bw = ((ro_samples / ofac) / self.lenc).to("Hz")
                warnings.warn(
                    "Required gradient amplitude exceeds maximum. "
                    f"Adjusted receiver BW to {receiver_bw.m_as('Hz'):.0f} Hz."
                )

            self.slope = np.abs(self.strength) / self.Gr_sr

        else:
            # Compute triangular ramps only
            slope_req = np.sqrt(
                np.abs(k_bw.to("1/m")) /
                (
                    self.scanner.gammabar.to("1/mT/ms") *
                    self.Gr_sr.to("mT/m/ms")
                )
            )
            slope_max = self.Gr_max / self.Gr_sr

            if slope_req < slope_max:
                self.slope = slope_req
                self.strength = self.Gr_sr * slope_req
                self.lenc = self.slope - slope_req
            else:
                self.slope = slope_max
                self.strength = self.Gr_max

                k_slopes = (
                    self.scanner.gammabar.to("1/mT/ms") *
                    self.Gr_sr.to("mT/m/ms") *
                    slope_max.to("ms")**2
                )
                self.lenc = (
                    (np.abs(k_bw.to("1/m")) - k_slopes.to("1/m")) /
                    (self.strength.to("mT/m") *
                     self.scanner.gammabar.to("1/mT/ms"))
                )

            self.strength *= np.sign(k_bw)

        # Update total duration
        if self.lenc < 0:
            self.dur = self.slope + self.slope
        else:
            self.dur = self.slope + self.lenc + self.slope

        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Bipolar Gradient Construction
    # ======================================================================
    def make_bipolar(self, VENC):
        """Construct a bipolar velocity-encoding gradient lobe pair.

        Modifies this gradient in place to become the first lobe and returns
        the second (inverted) lobe shifted in time.

        Parameters
        ----------
        VENC : Quantity
            Velocity encoding value (m/s).

        Returns
        -------
        Gradient
            Second gradient lobe (inverted, shifted by this lobe's duration).

        Notes
        -----
        The duration is chosen to produce the desired first-moment (velocity
        phase sensitivity π/VENC) using the scanner slew rate and amplitude
        limits.
        """

        VENC_sign = np.sign(VENC)
        VENC = np.abs(VENC)

        slope_max = (self.Gr_max / self.Gr_sr).to("ms")

        # Required slope for pure triangular VENC lobe
        slope_req = np.cbrt(
            Quantity(np.pi, "rad") /
            (
                2 *
                self.scanner.gamma.to("rad/ms/mT") *
                self.Gr_sr.to("mT/m/ms") *
                VENC.to("m/ms")
            )
        )

        if slope_req <= slope_max:
            self.slope = slope_req.to("ms")
            self.strength = -self.Gr_sr.to("mT/m/ms") * slope_req.to("ms")
            self.lenc = self.slope - slope_req
        else:
            a = (
                self.scanner.gamma.to("rad/ms/mT") *
                VENC.to("m/ms") *
                self.Gr_max.to("mT/m")
            )
            b = 3 * a * slope_max.to("ms")
            c = 2 * a * slope_max.to("ms")**2 - np.pi

            lenc_req = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            self.slope = slope_max.to("ms")
            self.strength = -self.Gr_max.to("mT/m") * VENC_sign
            self.lenc = lenc_req.to("ms")

        # Update duration
        if self.lenc < 0:
            self.dur = (self.slope + self.slope).to("ms")
        else:
            self.dur = (self.slope + self.lenc + self.slope).to("ms")

        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

        # Second lobe (inverted)
        g = self.__copy__()
        g *= -1.0
        g.change_time(self.time + self.dur)

        return g

    # ======================================================================
    # Area Computation
    # ======================================================================
    def area(self, t0=None, nb_samples=1000):
        """Compute the zeroth moment (area) of the gradient waveform.

        Parameters
        ----------
        t0 : Quantity or None, optional
            Integration start time (ms). Default is ``time - ref``.
        nb_samples : int, optional
            Number of samples for trapezoidal integration. Default is 1000.

        Returns
        -------
        Quantity
            Gradient area (mT·ms/m).
        """
        if t0 is None:
            t0 = self.time - self.ref

        t = np.linspace(
            t0.m_as("ms"),
            (self.time - self.ref + self.dur).m_as("ms"),
            nb_samples,
        )
        a = np.trapezoid(self.interpolator(t), t)
        return Quantity(a, "mT*ms/m")

    # ======================================================================
    # Area Matching (unchanged)
    # ======================================================================
    def match_area(self, area, dur=None):
        """Adjust slope, lenc, and strength to achieve a target gradient area.

        Parameters
        ----------
        area : Quantity
            Desired zeroth moment (mT·ms/m).
        dur : Quantity or None, optional
            Desired total duration (ms). If None, uses the minimal achievable
            duration given the scanner limits.

        Notes
        -----
        Sign of ``area`` is preserved; magnitude is used internally for
        calculations and restored at the end.
        """

        sign = np.sign(area)
        area = abs(area).to("mT*ms/m")

        slope_max = (self.Gr_max / self.Gr_sr).to("ms")

        if dur is not None:
            dur = dur.to("ms")

            # Case A: triangular only
            if dur < 2 * slope_max:
                self.slope = dur / 2
                self.lenc = Quantity(0.0, "ms")
                self.strength = (area / self.slope).to(self.Gr_max.u)

                if self.strength > self.Gr_max:
                    raise ValueError(
                        f"Cannot achieve area={area} in dur={dur}: "
                        f"G={self.strength} > Gmax={self.Gr_max}"
                    )

                self.dur = dur

            # Case B: plateau needed
            else:
                self.slope = slope_max
                self.lenc = dur - 2 * slope_max
                self.strength = (area / (self.slope + self.lenc)).to(
                    self.Gr_max.u
                )

                if self.strength > self.Gr_max:
                    raise ValueError(
                        f"Cannot achieve area={area} in dur={dur}: "
                        f"G={self.strength} > Gmax={self.Gr_max}"
                    )

                self.dur = dur

        else:
            # Minimal duration case
            slope_min = slope_max
            area_max = slope_min * self.Gr_max.to("mT/m")

            # Pure triangular
            if area <= area_max:
                ratio = (area / area_max).m
                self.slope = (slope_min * np.sqrt(ratio)).to("ms")
                self.strength = (
                    self.Gr_max * np.sqrt(ratio)
                ).to(self.Gr_max.u)
                self.lenc = (
                    self.slope - slope_min * np.sqrt(ratio)
                ).to("ms")

            # Plateau required
            else:
                area_needed = area - area_max
                self.slope = slope_min
                self.strength = self.Gr_max
                self.lenc = (area_needed / self.Gr_max).to("ms")

            # Duration update
            if self.lenc.m <= 0:
                self.dur = 2 * self.slope
            else:
                self.dur = 2 * self.slope + self.lenc

        # Restore sign
        self.strength *= sign

        # Recompute waveform
        self.dur2 = (self.dur - self.ref).to("ms")
        self.timings, self.amplitudes, self.interpolator = self.group_timings()

    # ======================================================================
    # Gradient Rotation
    # ======================================================================
    def rotate(self, directions, normalize_dirs=False):
        """Decompose the gradient into axis components along given direction(s).

        Parameters
        ----------
        directions : np.ndarray
            Direction vector(s), shape ``(N, 3)`` in MPS coordinates.
        normalize_dirs : bool, optional
            If True, each direction vector is normalized before decomposition.
            Default is False.

        Returns
        -------
        list of Gradient or list of list of Gradient
            For a single direction, a list of up to 3 axis-specific gradients.
            For multiple directions, a list of such lists.
        """
        directions = directions.reshape((-1, 3))
        if directions.shape[1] != 3:
            raise ValueError("Direction must be a 3-element vector [M,P,S].")

        nb_dirs = directions.shape[0]
        gradients = [[] for _ in range(nb_dirs)]

        for d in range(nb_dirs):
            direction = directions[d, :]
            norm = np.linalg.norm(direction)

            if norm != 0 and normalize_dirs:
                direction = direction / norm

            area_val = self.area()

            for i, fraction in enumerate(direction):
                if fraction != 0.0:
                    g = self.__copy__()
                    g.axis = i
                    g.match_area(fraction * area_val)
                    gradients[d].append(g)
                elif fraction == 0.0 and norm == 0.0:
                    g = self.__copy__()
                    g *= 0.0
                    gradients[d].append(g)
                    break

        max_dur = max(
            [
                g.dur if g.strength != 0.0 else Quantity(0.0, "ms")
                for d in range(nb_dirs)
                for g in gradients[d]
            ]
        )

        for d in range(nb_dirs):
            for g in gradients[d]:
                g.match_area(g.area(), max_dur)

        return gradients[0] if nb_dirs == 1 else gradients

    # ======================================================================
    # Plotting
    # ======================================================================
    def plot(self, linestyle="-"):
        """Plot the gradient waveform.

        Parameters
        ----------
        linestyle : str, optional
            Matplotlib line style string. Default is ``'-'``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        plt.plot(self.timings, self.amplitudes, linestyle)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mT/m)")
        plt.title("Gradient waveform")
        plt.grid(True)
        plt.show()
        return fig


class RF:
    """
    Radiofrequency (RF) excitation pulse class.

    This class generates RF waveforms commonly used in MRI imaging and
    simulation. It supports both *analytic pulse generation* (sinc, apodized
    sinc, hard pulses), and *user-provided custom waveforms*. All timing,
    phase, magnitude, and flip-angle normalization behavior is preserved
    exactly as in the original implementation.

    Key Features
    ------------
    - Supports analytical "hard", "sinc", and "apodized_sinc" RF shapes.
    - Maintains full Quantity support for physical units (ms, rad, Hz, etc.).
    - Preserves historical behavior: windowing, t-shifting, apodization,
      and flip-angle normalization.
    - Flip-angle specification is enforced by integrating the amplitude
      and applying a normalization factor.
    - Fully complex-valued interpolation, avoiding the real/imag duplication.
    - Optional custom waveform definition via `timings` + `waveform`.
    - Stores computed `timings` and complex `waveform` automatically.
    - Safe default for `scanner` (no mutable default instances).

    Parameters
    ----------
    scanner : Scanner or None
        MRI system definition containing at least `gamma`.
        If None, a default Scanner() is created.

    NbLobes : list or tuple of int, default=[2, 2]
        Number of lobes on the left and right sides of the sinc pulse.

    alpha : float, default=0.46
        Apodization factor for the apodized sinc. Ignored if shape='sinc'.

    shape : {'sinc', 'apodized_sinc', 'hard'}, default='apodized_sinc'
        RF pulse shape to generate.

    flip_angle : Quantity, default=pi/2 rad
        Desired flip angle of the RF pulse.

    dur : Quantity, default=2 ms
        Total duration of the RF pulse.

    ref : Quantity, default=0 ms
        Reference time for phase and time-shifting.

    time : Quantity, default=0 ms
        Time origin of the pulse.

    nb_samples : int, default=200
        Number of time samples used to generate the interpolated waveform.

    phase_offset : Quantity, default=0 rad
        Constant phase offset to apply to the entire RF waveform.

    frequency_offset : Quantity, default=0 Hz
        Frequency offset of the RF waveform (modulates B1 via exp(i 2π f t)).

    timings : array-like of floats or Quantities, optional
        Custom time vector in ms (or convertible to ms). If provided together
        with `waveform`, the analytic pulse generator is bypassed.

    waveform : array-like of complex, optional
        Custom RF waveform values corresponding to `timings`.

    Notes
    -----
    - Interpolation uses linear complex interpolation.
    - The internal pulse is always generated in milliseconds.
    """
    def __init__(self,
                 scanner=None,
                 NbLobes=None,
                 alpha=0.46,
                 shape='apodized_sinc',
                 flip_angle=Quantity(np.pi/2, 'rad'),
                 dur=Quantity(2.0, 'ms'),
                 ref=Quantity(0.0, 'ms'),
                 time=Quantity(0.0, 'ms'),
                 nb_samples=200,
                 phase_offset=Quantity(0.0, 'rad'),
                 frequency_offset=Quantity(0.0, 'Hz'),
                 timings=None,
                 waveform=None,
                 use='undefined'):

        # Safe default for scanner (prevents mutable default hazards)
        self.scanner = scanner if scanner is not None else Scanner()

        self.NbLobes = [2, 2] if NbLobes is None else list(NbLobes)
        self.alpha = alpha
        self.shape = shape

        # Functional label inherited from a Pulseq v1.5 'use' tag. Carried
        # through verbatim so partition-aware code (e.g. PulseqAdapter) can
        # identify excitation / refocusing / preparation pulses.
        self.use = use

        # Shape selection (exact original behavior)
        if self.shape == 'sinc':
            self._pulse = self._unit_sinc
            if self.alpha != 0.0 and MPI_rank == 0:
                warnings.warn("For 'sinc' shape, the alpha parameter is automatically set to 0.0")
            self.alpha = 0.0
        elif self.shape == 'apodized_sinc':
            self._pulse = self._unit_sinc
        elif self.shape == 'hard':
            self._pulse = self._unit_hard
        else:
            # 'custom' (and anything else) carries its waveform explicitly, so
            # there is no analytic generator. Leaving the attribute unset made
            # plot() raise AttributeError on every imported pulse, since
            # PulseqAdapter._convert_rf builds them all with shape='custom'.
            self._pulse = None

        # Physical parameters with units
        self.flip_angle = flip_angle.to('rad')
        self.ref = ref.to('ms')
        self.time = time.to('ms')
        self.dur = dur.to('ms')
        self.dur2 = (self.dur - self.ref).to('ms')
        self.nb_samples = nb_samples

        # Lobe durations for “sinc-like” pulse shapes
        self.half1 = (self.NbLobes[0] + 1)/(np.sum(self.NbLobes) + 2)*self.dur.to('ms')
        self.half2 = (self.NbLobes[1] + 1)/(np.sum(self.NbLobes) + 2)*self.dur.to('ms')

        self.phase_offset = phase_offset.to('rad')
        self.frequency_offset = frequency_offset.to('Hz')

        # Public waveform storage
        self.timings = None    # in ms
        self.waveform = None   # complex RF samples
        self._custom_waveform = False

        # Complex-valued interpolator
        self.interp = None

        # If user provides custom waveform → bypass analytic generator
        if timings is not None and waveform is not None:
            self._init_from_user_waveform(timings, waveform)
            self._custom_waveform = True
        else:
            self._build_interpolator()

    # ======================================================================
    # Internal time helpers
    # ======================================================================
    def _window(self, t):
        """Return a rectangular window selecting times between (time-ref) and (time-ref+dur)."""
        start = (self.time - self.ref).m_as('ms')
        end   = (self.time - self.ref + self.dur).m_as('ms')
        return (t >= start)*(t <= end)

    def _t_shift(self, t):
        """Return shifted local time used inside the RF excitation model."""
        return t - (self.time - self.ref).m_as('ms') - self.half1.m_as('ms')

    # ======================================================================
    # Analytic pulse definitions (bit-for-bit identical to original)
    # ======================================================================
    def _unit_sinc(self, t):
        """
        Generate an (apodized) sinc pulse.

        This method preserves the exact amplitude, windowing, phase modulation,
        and apodization behavior of the original implementation.
        """
        N = max(self.NbLobes)
        t_shift = self._t_shift(t)

        bw = (self.NbLobes[0] + self.NbLobes[1] + 2)/self.dur.to('ms')

        B1e = (1/bw.m)
        B1e *= (1 - self.alpha) + self.alpha*np.cos(np.pi*bw.m*t_shift/N)
        B1e *= np.sinc(bw.m*t_shift)
        B1e *= self._window(t)

        # Construct complex B1 with real magnitude (exact original)
        B1 = B1e + 1j*0

        # Apply phase + frequency offsets
        if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
            # The frequency term is NEGATED: the solver precesses as
            # exp(-i*gamma*Bz*t), so an RF modulated as exp(+i*2*pi*df*t)
            # resonates at z = -df/(gammabar*Gz), the mirror of the slice
            # Pulseq means. Every writer sets freq_offset = gammabar*Gz*z to
            # select the slice at +z, so a positive offset selects positive z.
            B1 *= np.exp(1j*(self.phase_offset.m_as('rad')
                            - 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

        return B1

    def _unit_hard(self, t):
        """Generate a hard (rectangular) RF pulse."""
        t_shift = self._t_shift(t)
        B1e = 1.0 * self._window(t)
        B1 = B1e + 1j*0

        if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
            # The frequency term is NEGATED: the solver precesses as
            # exp(-i*gamma*Bz*t), so an RF modulated as exp(+i*2*pi*df*t)
            # resonates at z = -df/(gammabar*Gz), the mirror of the slice
            # Pulseq means. Every writer sets freq_offset = gammabar*Gz*z to
            # select the slice at +z, so a positive offset selects positive z.
            B1 *= np.exp(1j*(self.phase_offset.m_as('rad')
                            - 2*np.pi*self.frequency_offset.m_as('kHz')*t_shift))

        return B1

    # ======================================================================
    # Flip-angle normalization
    # ======================================================================
    def _flip_angle_factor(self, t):
        """
        Compute normalization factor so that the final RF pulse integrates to
        the desired flip angle.

        Flip angle = γ ∫ B1(t) dt
        """
        dt = t[1] - t[0]
        amp = self._pulse(t)
        unit_FA = np.sum((amp[1:] + amp[:-1])/2)*dt*self.scanner.gamma.m_as('rad/mT/ms')
        return self.flip_angle.m_as('rad')/unit_FA

    # ======================================================================
    # Interpolator constructors
    # ======================================================================
    def _init_from_user_waveform(self, timings, waveform):
        """
        Initialize the RF using a custom user-supplied waveform.
        """
        self.timings = timings
        self.waveform = waveform

        # Dimensionless arrays for interpolation, in float64/complex128 rather than
        # float32/complex64. A Quantity keeps its own precision, so narrowing a
        # bare ndarray would make the same pulse depend on how it was spelled.
        # Absolute time is the worse half: float32 ms quantises to 2.7e-5 ms at
        # t = 1 s, above the raster tolerance it has to sit under.
        tt = timings.m_as('ms') if isinstance(timings, Quantity) else np.array(timings, dtype=np.float64)
        ww = waveform.m_as('mT') if isinstance(waveform, Quantity) else np.array(waveform, dtype=np.complex128)

        # Apply phase + frequency offsets, as the analytic generators do. This is
        # the only path an imported pulse takes -- PulseqAdapter builds them all
        # with shape='custom'.
        #
        # Two deliberate choices:
        #  * The ramp is referenced to the pulse's OWN START (tt[0]), matching
        #    pypulseq (`rf.signal * exp(1j*(phase + 2*pi*freq*rf.t))`, with rf.t
        #    starting at 0). The analytic path references the pulse CENTRE via
        #    _t_shift, a standing 2*pi*f*half1 disagreement left alone here.
        #  * It is applied to the INTERPOLATOR only, never to self.waveform, so
        #    `waveform` keeps matching pypulseq's bare `rf.signal` and the
        #    flip-angle round trip in test_pulseq_invariants stays meaningful.
        if self.phase_offset.m != 0.0 or self.frequency_offset.m != 0.0:
            ww = ww * np.exp(1j*(self.phase_offset.m_as('rad')
                                 - 2*np.pi*self.frequency_offset.m_as('kHz')
                                 * (tt - tt[0])))

        # Duration of the gradient relative to start time and reference
        self.dur = (self.timings[-1] - self.timings[0] + self.ref).to("ms")
        self.dur2 = (self.dur - self.ref).to("ms")

        self.interp = interp1d(
            tt, ww,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def _build_interpolator(self):
        """
        Build the analytic RF waveform, scale it to the correct flip angle,
        and construct the complex interpolating function.
        """
        # print("_build_interpolator: Generating analytic RF waveform.")
        start = (self.time - self.ref).m_as('ms')
        end   = (self.time - self.ref + self.dur).m_as('ms')

        t = np.linspace(start, end, self.nb_samples)

        # Flip-angle controlled scaling
        scaling = self._flip_angle_factor(t)
        wf = np.abs(scaling) * self._pulse(t)

        self.timings = t
        self.waveform = wf

        self.interp = interp1d(
            t, wf,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    # ======================================================================
    # Public Methods
    # ======================================================================
    def __call__(self, t):
        """
        Evaluate the complex RF pulse at time `t`.

        Parameters
        ----------
        t : float, ndarray, or Quantity
            Time at which to evaluate the RF (in ms or convertible to ms).

        Returns
        -------
        complex or ndarray of complex
            Complex RF field B1(t).
        """
        if isinstance(t, Quantity):
            t = t.to('ms').m_as('ms')
        return self.interp(t)


    def change_ref(self, ref):
        """Change the reference time of the RF pulse."""
        self.ref = ref.to('ms')
        self.dur2 = (self.dur - self.ref).to('ms')
        if self._custom_waveform:
            self._init_from_user_waveform(self.timings, self.waveform)
        else:
            self._build_interpolator()


    def change_time(self, time):
        """Change the absolute timing of the RF pulse.

        For custom (user-supplied) waveforms the existing ``timings``
        array is shifted rigidly by the delta from the previous absolute
        time, preserving any non-uniform raster produced by the Pulseq
        adapter; the interpolator is rebuilt via
        :meth:`_init_from_user_waveform`. For analytic shapes the
        waveform is regenerated by :meth:`_build_interpolator`.
        """
        new_time = time.to('ms')
        delta = new_time - self.time
        self.time = new_time
        if self._custom_waveform:
            self.timings = self.timings + delta
            self._init_from_user_waveform(self.timings, self.waveform)
        else:
            self._build_interpolator()


    def plot(self, linestyle='-'):
        """
        Plot the RF pulse waveform.

        For an analytic shape this draws the internal (unscaled) generator, for
        visual inspection of the pulse design. A custom waveform -- which is
        what every imported Pulseq pulse is -- has no analytic generator, so its
        stored samples are drawn instead. Calling the generator unconditionally
        used to raise on every imported pulse.
        """
        if self._pulse is None:
            t = (self.timings.m_as('ms') if isinstance(self.timings, Quantity)
                 else np.asarray(self.timings, dtype=float))
            wf = (self.waveform.m_as('mT') if isinstance(self.waveform, Quantity)
                  else np.asarray(self.waveform))
        else:
            start = (self.time - self.ref).m_as('ms')
            end   = (self.time - self.ref + self.dur).m_as('ms')
            t = np.linspace(start, end, self.nb_samples)
            wf = np.asarray(self._pulse(t))

        plt.figure()
        plt.plot(t, np.real(wf), linestyle)
        plt.plot(t, np.imag(wf), linestyle)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mT)")
        plt.legend(["Real", "Imag"])
        plt.show()