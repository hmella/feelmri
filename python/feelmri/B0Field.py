"""
The scanner-fixed B0 field.

:class:`B0Field` is main-field inhomogeneity: it belongs to the bore, so a
moving spin samples it at its CURRENT position. That is what separates it from
``delta_B`` / ``phi_dB0``, which are one value per NODE and travel with the
material point -- right for chemical shift and susceptibility, wrong for a
shim.

The field is given as an expression in SCANNER coordinates and expanded once,
onto the lowest polynomial order that represents it within ``rtol``, falling
back to a per-node Taylor expansion for anything a polynomial cannot carry.
:data:`SolverTerms` and :data:`ReadoutTerms` are what the two halves of the
simulation take from it.

The module and the class share a name, as ``datetime`` does. ``feelmri.B0Field``
resolves to the CLASS -- ``feelmri/__init__.py`` binds it after the submodule
exists -- so ``from feelmri import B0Field`` and
``from feelmri.B0Field import B0Field`` both give you the class.
"""
import collections
import itertools

import numpy as np
from pint import Quantity

from feelmri.MPIUtilities import MPI, MPI_print, MPI_rank


#: What `BlochSolver` needs from a :class:`B0Field`. A NamedTuple rather than a
#: bare tuple so a consumer that has not been taught about a new channel fails
#: with an AttributeError instead of silently unpacking the wrong thing.
SolverTerms = collections.namedtuple(
    'SolverTerms', 'offset_mT delta_B gradient quadratic node_gradient')

#: The same for the signal evaluator, with the rates already in rad/ms.
#: What the readout needs from a PER-NODE field. Two fields, not five: unlike
#: the solver, which has one entry point and carries every representation
#: through it, the readout has a dedicated function per trajectory type for
#: anything a polynomial can carry (`Trajectory.b0_terms`,
#: `PulseqAdapter.b0_readout_terms`). The five-field version carried a
#: `gradient` in mT/m and a `quadratic` in mT/m^2 inside a tuple whose other
#: entries were rad/ms, and no caller anywhere read either.
ReadoutTerms = collections.namedtuple(
    'ReadoutTerms', 'phi_nodal node_gradient')


class NoPolynomialFits(ValueError):
    """Raised by :meth:`B0Field.fit` when no polynomial up to the cap
    represents the expression. Its own exception type, so `on_phantom` falls
    back to the per-node rung for THAT and not for a `LinAlgError` (which is a
    `ValueError` subclass) or for the expression-shape contract error."""


class B0Field:
    """Scanner-fixed B0 inhomogeneity, sampled at the spin's CURRENT position.

    This is the lab-frame counterpart of ``BlochSolver(delta_B=)`` and
    ``FEMPhantom.set_static_fields(phi_dB0=)``. Those are one value per mesh
    node, frozen onto the node, so they travel with the tissue -- right for
    chemical shift and local susceptibility, wrong for a shim residual or a
    main-field imperfection, which stay where the magnet put them. A spin that
    moves through this field samples a different value; a spin that moves
    through ``delta_B`` does not.

    The field is given as an expression of position and carried as whichever
    of three representations it actually needs -- :attr:`kind` says which:

        'uniform' / 'linear'  dB0(x) = c0 + g . x                    [mT]
        'polynomial'          every monomial up to :attr:`MAX_ORDER`
        'nodal'               one value and one gradient PER NODE

    with ``x`` in metres, in the SCANNER frame, measured from isocentre -- not
    the imaging frame, and not relative to the slice location.
    :meth:`in_frame` converts to whatever frame a caller works in, and refuses
    a per-node field, which has no closed form to convert.

    Why a polynomial is not a compromise here: a static field in a
    current-free bore satisfies Laplace's equation, so it IS a solid-harmonic
    series, and that is the basis shim hardware is specified in. Every shim is
    a polynomial, so the fit is exact and costs no per-node memory --
    :attr:`residual_rms_mT` reports what it could not represent. A field no
    polynomial up to the cap represents falls back to the per-node expansion,
    which is EXACT on a phantom that does not move (the nodal value is the
    Eulerian answer there) and first order in the displacement on one that
    does.

    Parameters
    ----------
    offset : Quantity
        Uniform part, mT.
    gradient : Quantity
        Linear part, mT/m, a 3-vector in the scanner frame.
    order : int, optional
        Polynomial degree this object carries, 0 to :attr:`MAX_ORDER`. The
        coefficient vector must match it; see :meth:`fit`.
    coefficients : np.ndarray, optional
        Every monomial in the order `_monomial_exponents` lays out, which for
        degree 2 is ``(1 | x y z | xx yy zz xy xz yz)``. Defaults to the
        constant and the gradient, zero-padded to the declared order.
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
    # Kept as the documented floor of the fit rather than as a stopping rule:
    # the normal equations square the condition number, so a residual below
    # `sqrt(eps)` of the variation means the expression IS that polynomial and
    # nothing finer is measurable. It was written as a second `or` clause
    # beside `rtol` and was unreachable for any `rtol >= 1e-7`.
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
        n_terms = len(self._monomial_exponents(self.order))
        if coefficients is None:
            # Only the constant and the gradient are known here, so a declared
            # order above 1 is padded rather than truncated: truncating built a
            # length-4 vector that called itself degree 2, and
            # `quadratic_mT_per_m2` then raised a broadcast error from inside
            # `BlochSolver` rather than at the constructor.
            coefficients = np.zeros(n_terms, dtype=np.float64)
            coefficients[:min(4, n_terms)] = np.concatenate(
                ([self.offset_mT], g))[:min(4, n_terms)]
        self.coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if self.coefficients.size != n_terms:
            raise ValueError(
                f"B0Field: order {self.order} has {n_terms} monomials but "
                f"{self.coefficients.size} coefficients were given. A vector "
                f"that does not match its order is not a polynomial, and it "
                f"fails later inside the solver rather than here.")
        # The mirrors have to AGREE with the vector, not merely be the right
        # length. `__call__` evaluates the coefficients while `in_frame_full`
        # and `phi_offset` read the mirrors, so a hand-built field could answer
        # 0 at a point and still report a 5 mT offset -- two different fields
        # from one object, each self-consistent on its own accessors.
        keep = min(4, n_terms)
        mirrors = np.concatenate(([self.offset_mT], g))[:keep]
        if not np.array_equal(self.coefficients[:keep], mirrors):
            raise ValueError(
                f"B0Field: the coefficient vector's constant and linear terms "
                f"{self.coefficients[:keep]} disagree with the offset and "
                f"gradient given alongside them {mirrors}. They describe the "
                f"same field and are read by different callers.")
        # Per-node fallback, set by `on_phantom` when no polynomial fits. The
        # stamp records which node set it was built on -- a per-node array
        # means nothing under a different partition or ordering.
        self._nodal_mT = None
        self._nodal_grad = None
        self._nodal_stamp = None
        # How much of the field varies WITHIN one element, in mT. A per-node
        # field reaches the readout through the shape functions, so whatever it
        # does between nodes is not represented at all -- a separate limit from
        # the Taylor one, and the only error left on the static-exact path.
        # Set by `on_phantom`; compare it against the field's own amplitude.
        self.mesh_residual_mT = 0.0

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
        if self.order >= 3:
            raise NotImplementedError(
                f"B0Field.quadratic_mT_per_m2: this field is degree "
                f"{self.order}, and monomials 4:10 of it are not its quadratic "
                f"form -- returning them would drop every higher term "
                f"silently. Read `coefficients` instead.")
        out = np.zeros(6, dtype=np.float64)
        if self.order >= 2:
            out[:] = self.coefficients[4:10]
        return out

    def __repr__(self):
        # A per-node field carries a zero coefficient vector, so the polynomial
        # spelling below renders it as an indistinguishable null field.
        if self.kind == 'nodal':
            # A rank can own no nodes, and `max()` on an empty array raises.
            # This fires on exactly those ranks, so a log line on all of them
            # aborts a strict subset and the rest block at the next collective.
            peak = (float(np.abs(self._nodal_mT).max())
                    if self._nodal_mT.size else 0.0)
            resid = ('not measured -- no element connectivity'
                     if self.mesh_residual_mT is None
                     else f"{self.mesh_residual_mT:.3g} mT")
            return (f"B0Field(kind='nodal', nodes={self._nodal_mT.size}, "
                    f"peak={peak:.6g} mT, within-element residual={resid})")
        return (f"B0Field(kind={self.kind!r}, order={self.order}, "
                f"offset={self.offset_mT:.6g} mT, "
                f"gradient={np.round(self.gradient_mT_per_m, 9)} mT/m)")

    def polynomial_readout_terms(self, times_ms, scanner, rotation=None,
                                 location=None):
        """The three readout channels a POLYNOMIAL field needs, in one place.

        Returns ``(dk, phi_rate, maxwell)``: the k-space offset to ADD to each
        sample, the uniform part as an off-resonance RATE in rad/ms, and
        ``(N, 6)`` quadratic coefficients to add to whatever the concomitant
        term contributes -- ``None`` below degree 2.

        `Trajectory.b0_terms` and `PulseqAdapter.b0_readout_terms` are the two
        callers and both used to spell this out themselves. Two copies of one
        algebra can drift in the sign, the frame, the time origin or the dtype
        with nothing to notice, and they had already drifted in what the second
        return value MEANS -- a rate in one and a phase in the other. The
        shared form returns the rate; a caller wanting the phase multiplies by
        its own `t`.
        """
        if self.kind == 'nodal':
            raise TypeError(
                "polynomial_readout_terms: this field is a per-node "
                "expansion; none of these three channels can carry one -- a "
                "k-space shift is linear in position and the six maxwell "
                "coefficients are quadratic. It rides the phantom instead: "
                "add `readout_terms(...).phi_nodal` to `phi_dB0` and pass "
                "`.node_gradient` to `FEMPhantom.set_b0_gradient`.")
        b, g, q = self.in_frame_full(rotation=rotation, location=location,
                                     physical=False)
        t = np.asarray(times_ms, dtype=np.float64)
        gammabar = scanner.gammabar.m_as('1/ms/mT')
        gamma = scanner.gamma.m_as('rad/ms/mT')
        dk = (gammabar * t)[..., None] * np.asarray(g).reshape(
            (1,) * t.ndim + (3,))
        maxwell = None
        if np.any(q):
            # The assembler ADDS `m . monomials` to the phase, and the phase a
            # static field accrues by time t is `-gamma * dB0 * t`. Laid out
            # (xx, yy, zz, xy, xz, yz), the order the assembler reads.
            maxwell = (-gamma * t.reshape(-1, 1)) * np.asarray(q).reshape(1, 6)
        return dk, gamma * float(b), maxwell

    @staticmethod
    def is_live(field):
        """Whether ``field`` is a non-zero ``B0Field``, agreed across ranks.

        Both reductions are reached unconditionally, which is the whole point.
        Spelling this at the call site as ``field is not None and not
        field.is_zero_everywhere()`` short-circuits, so a rank whose field is
        ``None`` never enters the ``allreduce`` the others are inside and they
        block there forever. A field present on some ranks only is an error in
        its own right -- it describes one scanner -- so it is refused by name
        rather than left to deadlock at the next collective.
        """
        from feelmri.MPIUtilities import MPI_comm, collective_raise

        here = field is not None
        present = (MPI_comm.allgather(here) if MPI_comm.Get_size() > 1
                   else [here])
        mismatch = ''
        if any(present) and not all(present):
            absent = [r for r, ok in enumerate(present) if not ok]
            mismatch = (f"BlochSolver: a `b0_field` was given on some ranks "
                        f"and not on others (absent on rank(s) {absent}). It "
                        f"describes one scanner, so every rank must build it.")
        # Uniform by construction -- every rank formed it from the same
        # gathered list -- so this is safe to reach from every rank.
        collective_raise(mismatch)
        if not all(present):
            return False
        return not field.is_zero_everywhere()

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

    def is_zero_everywhere(self, collective=True):
        """:attr:`is_zero`, agreed across every rank.

        For a per-node field :attr:`is_zero` inspects the LOCAL slice, so a
        rank whose own nodes all sit where the field vanishes answers True
        while its peers answer False -- measured on a rod split four ways under
        a shim residual confined to half of it, 1 of 4 ranks disagreed. That
        predicate decides which kernel channels exist, so it has to be reduced
        before anything branches on it.
        """
        local = bool(self.is_zero)
        if not collective:
            return local
        from feelmri.MPIUtilities import MPI_comm
        from mpi4py import MPI as _MPI
        return bool(MPI_comm.allreduce(local, op=_MPI.LAND))

    def __call__(self, points_m):
        """Evaluate the expansion at scanner-frame positions, in mT.

        The WHOLE expansion, not its linear part. Reading `offset + g . x` off
        an object that also carries a quadratic form returns a number that is
        not the field anywhere: measured on a degree-2 fixture, 4.8e-20 mT
        where the field is 2.5e-06. A per-node field is refused rather than
        answered with the zeros it was constructed with -- every other accessor
        on this class refuses, and this is the one a caller reaches for first.
        """
        if self.kind == 'nodal':
            raise TypeError(
                "B0Field.__call__: this field is a per-node expansion, which "
                "has no closed form to evaluate at arbitrary points. Read it "
                "at the nodes it was sampled on with `nodal_mT(phantom)`.")
        p = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        return self._design(p, self.order) @ self.coefficients

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
        if self.kind == 'nodal':
            raise TypeError(
                "B0Field.in_frame_full: this field is a per-node expansion, "
                "which has no constant, gradient or quadratic form -- "
                "returning the zeros it was constructed with would drop the "
                "whole field with no symptom. Use `solver_terms` / "
                "`readout_terms`, which hand each consumer what it can carry.")
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

        Returns ``(offset_mT, delta_B_mT, gradient, quadratic, node_gradient)``:
        a uniform
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

    def readout_terms(self, phantom, scanner, moving, rotation=None):
        """What the signal evaluator needs, in the imaging frame it works in.

        Returns ``(phi_nodal, node_gradient)`` in rad/ms and rad/ms/m: the
        field at each node, and its gradient there, which
        :meth:`FEMPhantom.set_b0_gradient` takes. ``node_gradient`` is ``None``
        on a phantom that does not move, where the nodal value IS the Eulerian
        answer. A field a polynomial can carry is REFUSED -- see the message.
        The
        assembler's nodes are always the imaging ones, so unlike the solver
        there is no physical-frame case here.
        """
        gamma = scanner.gamma.m_as('rad/ms/mT')
        if self.kind != 'nodal':
            raise TypeError(
                f"B0Field.readout_terms: this is a {self.kind} field, which "
                f"the readout carries as a k-space shift, a uniform phase and "
                f"the six `maxwell` coefficients rather than per node. Use "
                f"`Trajectory.b0_terms` for a native trajectory or "
                f"`PulseqAdapter.b0_readout_terms` for an imported one; both "
                f"return all three together, because a field split across "
                f"channels is a field that can be half-applied.")
        nodal = self.nodal_mT(phantom)
        if not moving:
            return ReadoutTerms(gamma * nodal, None)
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
        return ReadoutTerms(gamma * nodal, gamma * g)

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
        are tried in turn and the first whose residual RMS falls below ``rtol``
        times the RMS of the field's VARIATION is kept -- the mean is removed
        first, so a large uniform offset cannot flatter the fit. A shim spelled
        as a big constant plus a small spatial term is therefore judged on the
        spatial term alone, and reaches the per-node fallback more readily than
        a ratio against the raw values would suggest.

        Under MPI the normal equations are accumulated locally and reduced, so
        every rank solves the same system and gets a bit-identical field.
        Fitting a gathered map per rank would let ranks disagree and the solve
        would be silently inconsistent.
        """
        from feelmri.MPIUtilities import MPI_comm, collective_raise

        if max_order is not None and int(max_order) < 0:
            raise ValueError(
                f"B0Field.fit: max_order={max_order} tries no degree at all. "
                f"Pass 0 or more, or leave it unset for the class cap.")
        p = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        values = expression(p)
        if isinstance(values, Quantity):
            values = values.m_as('mT')
        f = np.asarray(values, dtype=np.float64).reshape(-1)
        # Collected, not raised on the spot: both counts are rank-local, so an
        # expression that returns a fixed length matches on one rank only and
        # a bare raise there strands the others in the reductions below.
        rows = ("" if f.size == p.shape[0] else
                f"B0Field.fit: the expression returned {f.size} values for "
                f"{p.shape[0]} points; it must map (N, 3) positions to (N,).")
        # Gated on `collective`, which promises this call makes none: an
        # ungated `collective_raise` allgathers whenever the size is > 1, so a
        # rank that fits locally would pair that allgather with whatever the
        # others reach next and every later collective reads corrupted data.
        if collective:
            collective_raise(rows)
        elif rows:
            raise ValueError(rows)

        max_order = cls.MAX_ORDER if max_order is None else int(max_order)
        # The fit is done on the field with its MEAN REMOVED, and the mean is
        # put back on the constant coefficient at the end. Two things follow,
        # and both are needed for a field written the way a B0 map is written
        # -- a large uniform offset plus a small spatial term:
        #
        #  * the yardstick becomes the RMS of the VARIATION. Scored against
        #    the RMS of the VALUES, a small spatial term sits under `rtol`
        #    purely because the offset is large, and the fit discards it. The
        #    constant monomial carries the offset exactly at every order, so
        #    it cannot belong in the measure of what is left to fit.
        #  * the residual stops cancelling. It is formed from the reduced
        #    `f.f - 2 c.b + c.A.c`, and against a large uncentred offset those
        #    terms dwarf the answer, so what comes back is round-off.
        n_tot = float(p.shape[0])
        fs = float(f.sum())
        pp = float((p * p).sum())
        if collective:
            n_tot = MPI_comm.allreduce(n_tot, op=MPI.SUM)
            fs = MPI_comm.allreduce(fs, op=MPI.SUM)
            pp = MPI_comm.allreduce(pp, op=MPI.SUM)
        mean = fs / n_tot if n_tot else 0.0
        # The design is built on positions scaled to unit RMS radius, and the
        # coefficients are scaled back at the end. Without it the monomial
        # columns span `L^degree`, so on a cloud a millimetre across the
        # quadratic columns are ~1e-6 of the constant one and the NORMAL
        # matrix squares that: the degree-2 fit of an exact degree-2 field
        # over a 0.1 mm box came back with a residual equal to its own
        # variation. The fit is a property of the field, not of the units the
        # geometry happens to be in.
        radius = np.sqrt(pp / (3.0 * n_tot)) if n_tot and pp > 0.0 else 1.0
        p = p / radius
        f = f - mean
        ff = float(f @ f)
        if collective:
            ff = MPI_comm.allreduce(ff, op=MPI.SUM)
        scale = np.sqrt(ff / n_tot) if n_tot else 0.0

        best = None
        # Why the search stopped early, for the message below: a higher degree
        # was refused by the point count or by the rank of its design.
        blocked = None
        for order in range(0, max_order + 1):
            # An underdetermined fit is exact and meaningless: 20 monomials
            # through 5 points reproduces them all and says nothing about the
            # field anywhere else. The count that matters is the GLOBAL one,
            # since the normal equations are reduced across ranks.
            n_terms = len(cls._monomial_exponents(order))
            if order > 0 and n_terms >= n_tot:
                blocked = (order, f"{n_terms} monomials through "
                                  f"{int(n_tot):d} points is exact and says "
                                  f"nothing about the field between them")
                break
            basis = cls._design(p, order)
            # Normal equations, reduced BEFORE the solve so every rank solves
            # an identical system.
            A = basis.T @ basis
            b = basis.T @ f
            if collective:
                A = MPI_comm.allreduce(A, op=MPI.SUM)
                b = MPI_comm.allreduce(b, op=MPI.SUM)
            coef, _res, _rank_A, _sv = np.linalg.lstsq(A, b, rcond=None)
            # The rank is taken from the DESIGN's spectrum, not the normal
            # matrix's. `A` is symmetric positive semidefinite, so its singular
            # values are the SQUARED singular values of the design -- testing
            # `_rank_A` refuses at `cond(design) ~ 1.5e7`, about half the
            # decades a rank test on the design allows, and calls a
            # sub-millimetre cloud degenerate when it is merely small.
            sv = np.sqrt(np.maximum(np.asarray(_sv, dtype=np.float64), 0.0))
            rank = (int(np.count_nonzero(
                sv > sv[0] * max(A.shape) * np.finfo(np.float64).eps))
                if sv.size and sv[0] > 0.0 else 0)
            # The point count is not enough: monomials that are linearly
            # dependent ON THESE POINTS make the fit exact and arbitrary off
            # the sampled manifold -- a coplanar cloud fits `x^2 + z^2` at
            # order 2 with a zero residual and a zz coefficient of zero.
            # `order > 0` because degree 0 is the one fit that must always
            # produce an answer: it is a single constant monomial, so a
            # rank-deficient design there means there is nothing to fit at all
            # -- an empty point cloud -- and the honest result is the uniform
            # field at the mean, not an unpack of `best` that is still None.
            if order > 0 and rank < n_terms:
                blocked = (order, f"the degree-{order} monomials are linearly "
                                  f"dependent on these points (rank {rank} of "
                                  f"{n_terms}), so the fit would be exact on "
                                  f"them and arbitrary anywhere else")
                break
            # |f - Phi c|^2 = f.f - 2 c.b + c.A.c, from the reduced pieces.
            resid = max(ff - 2.0 * float(coef @ b) + float(coef @ A @ coef), 0.0)
            resid_rms = np.sqrt(resid / n_tot) if n_tot else 0.0
            best = (order, coef, resid_rms)
            # Two stopping rules, and the first is the one that matters. A
            # residual at round-off means the expression IS this polynomial --
            # every shim is -- so it is carried exactly and there is nothing to
            # gain from a higher degree. `rtol` is the weaker rule for a field
            # that is only well approximated.
            if (scale == 0.0 or resid_rms <= rtol * scale):
                break

        order, coef, resid_rms = best
        if resid_rms > rtol * scale and scale > 0.0:
            raise NoPolynomialFits(
                f"B0Field.fit: degree {order} still leaves a residual of "
                f"{resid_rms:.4g} mT against a spatial variation of "
                f"{scale:.4g} mT ({resid_rms / scale:.1%}), above "
                f"rtol={rtol:g}. " +
                (f"Degree {blocked[0]} was not attempted: {blocked[1]}. "
                 if blocked else "") +
                f"This field needs the per-node expansion instead of the "
                f"global one -- build it with `B0Field.on_phantom`, which "
                f"falls back to that.")

        # Back into physical units: the monomial `x^a y^b z^c` was fitted on
        # `p / radius`, so its coefficient carries `radius^(a+b+c)`.
        coef = np.asarray(coef, dtype=np.float64).copy()
        degrees = np.array([sum(e) for e in cls._monomial_exponents(order)],
                           dtype=np.float64)
        coef = coef / (radius ** degrees)
        coef[0] += mean
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
            fitted = cls.fit(expression, nodes, **kwargs)
            # A caller-supplied `max_order` above the class cap produces a fit
            # nothing downstream can consume: `quadratic_mT_per_m2` and
            # `in_frame_full` both refuse degree 3, so the field would raise
            # from inside `BlochSolver` instead of falling back here -- which
            # is the outcome the docstring promises and the constructor check
            # exists to prevent.
            if fitted.order > cls.MAX_ORDER:
                raise NoPolynomialFits(
                    f"B0Field.on_phantom: the expression fits at degree "
                    f"{fitted.order}, above the cap of {cls.MAX_ORDER} that "
                    f"the solver and the readout can carry. It needs the "
                    f"per-node expansion.")
            gap, span = cls._holdout_residual(expression, phantom, nodes, fitted,
                                              kwargs.get('collective', True))
            rtol = float(kwargs.get('rtol', 1.0e-3))
            # `span is None` means no rank had the connectivity to check with,
            # so the fit is UNVERIFIED rather than verified clean.
            if span is not None and span > 0.0 and gap > rtol * span:
                # The fit reproduces every node and does not generalise. A
                # coarse structured mesh is the ordinary way in: 27 nodes on a
                # 3 x 3 x 3 lattice carry only THREE distinct values per axis,
                # and a degree-2 polynomial passes through any three points
                # exactly -- so `sin(x / 0.02)` over 10 radians fitted with a
                # residual of 0.0. Exact at the nodes, wrong everywhere else,
                # which costs nothing while the phantom is still and is the
                # whole answer once it moves.
                raise NoPolynomialFits(
                    f"B0Field.on_phantom: the degree-{fitted.order} fit "
                    f"reproduces every node but leaves {gap:.4g} mT at the "
                    f"element centroids, against a field variation of "
                    f"{span:.4g} mT. It is interpolating this node set rather "
                    f"than representing the field, so the per-node expansion "
                    f"is used instead.")
            return fitted
        except NoPolynomialFits:
            # Only this one. A bare `except ValueError` also swallows the
            # expression-shape contract error and `numpy.linalg.LinAlgError`,
            # which is a ValueError subclass -- turning a caller's bug into a
            # silent per-node fallback.
            if not nodal:
                raise
        field = cls(coefficients=np.zeros(1))
        field._nodal_mT = cls._sample(expression, nodes)
        field._nodal_grad = cls._sample_gradient(
            expression, nodes, gradient=gradient, fd_step=fd_step,
            collective=kwargs.get('collective', True))
        field._nodal_stamp = cls._node_stamp(phantom)
        field.mesh_residual_mT = cls._mesh_residual(
            expression, phantom, nodes, collective=kwargs.get('collective', True))
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
        nodes = np.asarray(phantom.local_nodes, dtype=np.float64)
        part = getattr(phantom, '_active_partition', None)
        if nodes.size == 0:
            # A rank may legitimately own no nodes; indexing one here would
            # raise from inside every per-node accessor instead.
            return (part, nodes.shape, 0.0, 0.0, 0.0)
        # The first and last coordinate alone miss a rigid translation -- a
        # pure shift along y left the stamp bit-identical while every node had
        # moved, which is exactly the pairing the stamp exists to refuse.
        #
        # The two MOMENTS catch a translation and a change of shape and neither
        # can see a REORDERING: both are symmetric functions of the rows, so a
        # permuted node set gives a bit-identical stamp -- and a reordering is
        # exactly the pairing this exists to refuse, since every value stays
        # valid while the node it belongs to moves. The index-weighted sum is
        # the term that sees it. (`flat @ flat` is also invariant under a
        # rotation about the origin, which cannot arise here: `orient` after
        # `set_assembler` is refused.)
        flat = nodes.reshape(-1)
        idx = np.arange(1, nodes.shape[0] + 1, dtype=np.float64)
        return (part, nodes.shape, float(nodes.sum(axis=0).sum()),
                float(flat @ flat), float(idx @ nodes.sum(axis=1)))

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
        from feelmri.MPIUtilities import collective_raise
        stamp = self._node_stamp(phantom)
        # The stamp is built from this rank's own nodes, so a repartition can
        # move it on some ranks only. Collected, for the same reason as above.
        collective_raise(
            "" if stamp == self._nodal_stamp else
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

        # Which BRANCH is taken decides how many collectives this call makes
        # -- the analytic one below makes none, the finite-difference path
        # three -- so the ranks have to agree on it before either runs.
        analytic = gradient is not None
        if collective and MPI_comm.Get_size() > 1:
            seen = MPI_comm.allgather(analytic)
            collective_raise(
                "" if all(seen) or not any(seen) else
                "B0Field: an analytic `gradient=` was supplied on some ranks "
                "and not on others; it describes one field and must be the "
                "same callable everywhere.")
        if analytic:
            g = gradient(points)
            if isinstance(g, Quantity):
                g = g.m_as('mT/m')
            g = np.asarray(g, dtype=np.float64).reshape(-1, 3)
            # Collected: `points.shape[0]` is this rank's node count, so a
            # gradient that returns a fixed length matches on exactly one rank
            # and the rest walk on into the next collective.
            rows = ("" if g.shape[0] == points.shape[0] else
                    f"B0Field: the gradient returned {g.shape[0]} rows for "
                    f"{points.shape[0]} points; it must map (N, 3) to (N, 3).")
            if collective:
                collective_raise(rows)
            elif rows:
                raise ValueError(rows)
            return g

        # Scaled to the cloud, not absolute. A fixed 1e-4 m is a tenth of a
        # sub-millimetre domain and a millionth of a bore-sized one, and at a
        # boundary node the perturbed points always sit outside the mesh --
        # harmless for an analytic expression, meaningless for anything with a
        # support.
        span = float(np.abs(points).max()) if points.size else 0.0
        if collective:
            span = MPI_comm.allreduce(span, op=MPI.MAX)
        h = (max(1.0e-6 * span, 1.0e-9) if fd_step is None else float(fd_step))

        def central(step):
            out = np.empty(points.shape, dtype=np.float64)
            for axis in range(3):
                e = np.zeros(3, dtype=np.float64)
                e[axis] = step
                out[:, axis] = (cls._sample(expression, points + e)
                                - cls._sample(expression, points - e)) / (2 * step)
            return out

        g = central(h)
        coarse = central(2.0 * h)
        # POINTWISE, against a floor tied to the global peak. Comparing the two
        # global maxima instead hides a kink wherever the gradient is small
        # against the largest gradient anywhere -- which is most of a localised
        # defect, and exactly the input this class refuses in prose.
        scale = float(np.abs(g).max()) if g.size else 0.0
        if collective:
            scale = MPI_comm.allreduce(scale, op=MPI.MAX)
        # A CONSTANT field has `g == 0` everywhere, so `scale` and the floor
        # are both zero and the ratio below is 0/0: a RuntimeWarning and a NaN
        # drift, which is an exception under `-W error` and, at `collective=
        # False` with more than one rank, only on the flat ranks.
        floor = 1.0e-3 * scale if scale > 0.0 else 1.0
        local = np.abs(g - coarse) / np.maximum(np.abs(g), floor)
        drift = float(local.max()) * scale if local.size else 0.0
        if collective:
            drift = MPI_comm.allreduce(drift, op=MPI.MAX)
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
        # Gated, like every other guard here: `collective=False` promises this
        # call makes none, and `collective_raise` allgathers whenever the size
        # is > 1. Ungated it desynchronises the ranks that took this branch
        # from the ones that did not, and the NEXT allgather then returns each
        # rank its own value paired with someone else's -- silently, so the
        # corruption surfaces somewhere unrelated.
        if collective:
            collective_raise(problem, ValueError)
        elif problem:
            raise ValueError(problem)
        return g

    @classmethod
    def _holdout_residual(cls, expression, phantom, nodes, fitted, collective):
        """How badly a fit does at points it was NOT fitted on.

        The element centroids are the cheapest such set: they are inside the
        mesh, they need no new geometry, and on a structured mesh they are
        exactly the places the nodes cannot see. Counting points is not enough
        to catch this -- the design can be full rank and the fit exact while
        the polynomial is interpolating the lattice.

        Returns ``(worst residual, field variation)``, both in mT and both
        reduced across ranks so every rank reaches the same verdict.
        """
        from feelmri.MPIUtilities import MPI_comm

        # The local values are computed first and the reduction is
        # UNCONDITIONAL. A rank that owns no elements returning early here
        # would skip the allreduce its peers are already inside, and the others
        # block in it for ever.
        elems = getattr(phantom, 'local_elements', None)
        have = elems is not None and len(elems) > 0
        # IDENTITIES for the reductions, not zeros. A rank that owns no
        # elements has no opinion about the field's range, and contributing
        # 0.0 to a MIN and a MAX makes the span straddle zero: on a field that
        # does not -- a shim written as a large uniform offset plus a small
        # spatial term, which is how this class documents them -- the span is
        # then inflated by the whole offset and the threshold below cannot
        # fire. Measured on a 1 ppm shim at 1.5 T, 1.8e-04 mT of real
        # variation was reported as 5.0 mT.
        gap, lo, hi = 0.0, np.inf, -np.inf
        if have:
            centroids = nodes[np.asarray(elems)].mean(axis=1)
            truth = cls._sample(expression, centroids)
            gap = float(np.abs(
                truth - np.asarray(fitted(centroids)).reshape(-1)).max())
            if truth.size:
                lo, hi = float(truth.min()), float(truth.max())
        if collective:
            gap = MPI_comm.allreduce(gap, op=MPI.MAX)
            lo = MPI_comm.allreduce(lo, op=MPI.MIN)
            hi = MPI_comm.allreduce(hi, op=MPI.MAX)
            have = bool(MPI_comm.allreduce(have, op=MPI.LOR))
        # No connectivity ANYWHERE means the guard could not run, which is not
        # the same as having run and found nothing. `on_phantom` is told so
        # rather than reading a zero span as a clean bill of health.
        if not have:
            return gap, None
        return gap, hi - lo

    @classmethod
    def _mesh_residual(cls, expression, phantom, nodes, collective=True):
        """How much of the field varies WITHIN one element, in mT.

        A per-node field reaches the readout through the shape functions, so
        whatever it does between nodes is not represented at all. Comparing the
        expression at each element centroid against the mean of that element's
        nodal values measures exactly that -- and unlike an interpolant it needs
        no per-cell-type basis, so it is valid for every element the mesh may
        hold.

        Returns ``None`` when NO rank has the connectivity to measure it. That
        is not the same as zero, and zero is the most reassuring answer this
        function can give -- "the per-node representation loses nothing" --
        which is exactly the wrong thing to return for a check that could not
        run. Same distinction `_holdout_residual` draws.

        REDUCED across ranks, like every other error metric on this class. A
        per-rank figure is worse than none: it was reported once, from
        whichever rank happened to ask, and that rank's slice may be the
        smoothest part of the field. Measured on `spamm.py` at 8 ranks, the
        unreduced form read 4.370e-07 mT against a true 7.483e-07 -- an
        UNDER-report, which is the dangerous direction for a number whose job
        is to say how much of the field is being dropped.
        """
        from feelmri.MPIUtilities import MPI_comm

        elems = getattr(phantom, 'local_elements', None)
        have = elems is not None and len(elems) > 0
        worst = 0.0
        if have:
            elems = np.asarray(elems)
            centroids = nodes[elems].mean(axis=1)
            nodal = cls._sample(expression, nodes)
            worst = float(np.abs(cls._sample(expression, centroids)
                                 - nodal[elems].mean(axis=1)).max())
        # Unconditional, for the reason `_holdout_residual` gives. A rank with
        # no elements contributing 0.0 to a MAX over non-negative residuals is
        # harmless, unlike the MIN/MAX pair there.
        if collective:
            worst = MPI_comm.allreduce(worst, op=MPI.MAX)
            have = bool(MPI_comm.allreduce(have, op=MPI.LOR))
        return worst if have else None

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
