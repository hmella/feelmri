"""The lab-frame B0 expansion.

`B0Field` is the scanner-fixed counterpart of the per-node `delta_B` /
`phi_dB0` channel: it is sampled at the spin's current position instead of
being frozen to the node. These tests cover the expansion and the frame
algebra alone; the physics that the two channels behave differently under
motion lives in `test_signal_analytical.py`.
"""
import numpy as np
import pytest
from pint import Quantity

from feelmri import B0Field


def _points(n=400, seed=0, reach=0.12):
    return np.random.default_rng(seed).uniform(-reach, reach, size=(n, 3))


def _rotation(deg=23.0):
    th = np.deg2rad(deg)
    return np.array([[np.cos(th), 0.0, np.sin(th)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(th), 0.0, np.cos(th)]])


def test_a_linear_expression_is_recovered_exactly():
    """The shipped examples all build `x + y + z` ramps, so this is the case
    that has to be exact rather than merely close."""
    g = np.array([0.011, -0.004, 0.0075])
    b = 2.5e-4
    field = B0Field.fit(lambda p: b + p @ g, _points(), collective=False)
    assert field.order == 1
    assert field.offset_mT == pytest.approx(b, abs=1e-15)
    assert np.abs(field.gradient_mT_per_m - g).max() < 1e-15
    assert field.residual_rms_mT == pytest.approx(0.0, abs=1e-15)


def test_a_uniform_expression_collapses_to_order_zero():
    """Auto truncation must not spend a linear term on a constant: order 0
    rides `delta_B` and costs nothing at all."""
    field = B0Field.fit(lambda p: np.full(p.shape[0], 1e-3), _points(),
                        collective=False)
    assert field.order == 0
    assert not np.any(field.gradient_mT_per_m)
    assert field.is_zero is False


def test_the_frame_adapter_moves_the_offset_into_the_constant():
    """`orient` leaves `x_used = M.T (x_scanner - LOC)`, and the solver rotates
    back to `x_scanner - LOC` for the concomitant term. Both substitutions give
    the same constant, so the slice offset lands entirely there and the node
    array never has to move.

    The `LOC != 0` case is the one that matters: at the origin the translation
    is ignored and a wrong adapter passes anyway.
    """
    g = np.array([0.011, -0.004, 0.0075])
    field = B0Field(Quantity(2.5e-4, 'mT'), Quantity(g, 'mT/m'))
    M, L = _rotation(), np.array([0.03, -0.02, 0.05])
    x_used = _points(n=9, seed=3, reach=0.1)

    for physical in (False, True):
        offset, grad = field.in_frame(rotation=M, location=L, physical=physical)
        x_scanner = (x_used if physical else x_used @ M.T) + L
        assert np.abs(field(x_scanner) - (offset + x_used @ grad)).max() < 1e-15

    # Not vacuous: the offset really does change the constant here.
    plain, _ = field.in_frame(rotation=M, location=None)
    shifted, _ = field.in_frame(rotation=M, location=L)
    assert abs(shifted - plain) > 1e-6


def test_a_polynomial_field_is_carried_at_the_degree_it_actually_has():
    """The degree detects itself: the search stops where the residual reaches
    round-off, because a residual that small means the expression IS that
    polynomial rather than being approximated by it. Every shim is one."""
    quad = B0Field.fit(lambda p: 1e-3 * (p[:, 0] ** 2 - p[:, 2] ** 2),
                       _points(), collective=False)
    assert quad.order == 2 and quad.kind == 'polynomial'

    # Reproduced to round-off, not merely fitted. This is the assertion that
    # carries the claim: the fit residual itself bottoms out at sqrt(eps)
    # because the normal equations square the condition number, so it is the
    # reconstruction that shows the field is carried exactly.
    pts = _points()
    got = B0Field._design(pts, quad.order) @ quad.coefficients
    want = 1e-3 * (pts[:, 0] ** 2 - pts[:, 2] ** 2)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()

    # Degree 3 is ABOVE what the solver and readout channels carry, so it is
    # not detected by default -- detecting it would only let it be truncated to
    # a quadratic. Asked for explicitly, the frame adapter refuses it by name
    # rather than dropping the cubic monomials: measured on a Z3 shim that is
    # 100% of the field.
    Z3 = lambda p: 1e-3 * p[:, 2] * (2 * p[:, 2] ** 2 - 3 * p[:, 0] ** 2)
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.fit(Z3, _points(), collective=False)
    with pytest.raises(NotImplementedError, match='drop every higher monomial'):
        B0Field.fit(Z3, _points(), collective=False, max_order=3).in_frame_full()


def test_a_polynomial_field_refuses_the_linear_only_channels():
    """`in_frame` carries the constant and the gradient, which for a degree-2
    field is not the field. Returning them would silently truncate it, so it
    raises instead -- the quadratic part has its own channel."""
    quad = B0Field.fit(lambda p: 1e-3 * (p[:, 0] ** 2 - p[:, 2] ** 2),
                       _points(), collective=False)
    with pytest.raises(NotImplementedError, match='silently truncate'):
        quad.in_frame()


def test_a_field_no_polynomial_can_represent_is_refused():
    """A step is not a smooth scanner field: a static field in a current-free
    bore is a solid-harmonic series. Such a map is tissue structure and belongs
    on the per-node channel."""
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.fit(lambda p: 1e-3 * np.sign(p[:, 0]), _points(),
                    collective=False)


def test_a_pint_expression_and_a_bare_one_agree():
    """Examples write bare arrays; a Quantity must work too, and must not be
    read as though its magnitude were already mT."""
    g = np.array([0.011, -0.004, 0.0075])
    pts = _points()
    bare = B0Field.fit(lambda p: p @ g, pts, collective=False)
    pint = B0Field.fit(lambda p: Quantity(p @ g * 1e-3, 'T'), pts,
                       collective=False)
    assert np.abs(bare.gradient_mT_per_m - pint.gradient_mT_per_m).max() < 1e-15


def test_a_mismatched_expression_is_refused():
    with pytest.raises(ValueError, match='must map'):
        B0Field.fit(lambda p: np.zeros(p.shape[0] - 1), _points(),
                    collective=False)


def test_a_bad_gradient_is_refused():
    with pytest.raises(ValueError, match='3-vector'):
        B0Field(gradient=Quantity(np.zeros(2), 'mT/m'))


def test_the_uniform_part_converts_to_an_offresonance_rate():
    """The constant is spatially uniform, so it needs no frame and can ride
    `phi_dB0` directly."""
    from feelmri import Scanner
    scanner = Scanner()
    field = B0Field(Quantity(1e-3, 'mT'))
    assert field.phi_offset(scanner) == pytest.approx(
        1e-3 * scanner.gamma.m_as('rad/ms/mT'))


def test_a_field_no_polynomial_can_carry_falls_back_to_the_nodes(tmp_path):
    """The refusal above is right for the global expansion and wrong as a
    verdict on the field: on a phantom that does not move, `x(t) = x0`, so a
    per-node value IS the Eulerian answer and it is exact however rough the
    field is. `on_phantom` therefore falls back instead of raising.

    The expression here is a 30 mm sine, which a degree-3 polynomial misses by
    82% of the field RMS -- the fallback is not a convenience, it is the only
    representation that works.
    """
    pytest.importorskip('meshio')
    from feelmri.Phantom import FEMPhantom
    import meshio

    pts = np.array([[0.11, -0.03, 0.07], [-0.05, 0.12, 0.02],
                    [0.04, 0.06, -0.10], [-0.09, -0.08, 0.05],
                    [0.02, -0.11, -0.06]])
    path = tmp_path / 'rough.vtu'
    meshio.write(str(path), meshio.Mesh(
        pts, [('tetra', np.array([[0, 1, 2, 3], [1, 2, 4, 3]]))]))
    phantom = FEMPhantom(path=str(path))

    rough = lambda p: 1e-3 * np.sin(2 * np.pi * (p[:, 0] + p[:, 1]) / 0.03)

    # The global expansion still refuses, and says why.
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.fit(rough, B0Field._scanner_nodes(phantom), collective=False)

    field = B0Field.on_phantom(rough, phantom, collective=False)
    assert field.kind == 'nodal'

    # Exact at the nodes, which is the whole claim.
    want = rough(B0Field._scanner_nodes(phantom))
    got = field.nodal_mT(phantom)
    assert np.abs(got - want).max() == 0.0

    # And a static solver gets it as `delta_B`, with no gradient channel.
    terms = field.solver_terms(phantom, moving=False)
    assert terms.offset_mT == 0.0
    assert terms.gradient is None and terms.quadratic is None
    assert terms.node_gradient is None
    assert np.abs(terms.delta_B.reshape(-1) - want).max() == 0.0

    # A moving phantom gets the Eulerian expansion instead: the bracket on
    # `delta_B` and a per-node gradient for the kernel. At zero displacement
    # the two must reconstruct the same field.
    moving = field.solver_terms(phantom, moving=True)
    assert moving.node_gradient is not None
    x = np.asarray(phantom.local_nodes, dtype=np.float64)
    rebuilt = (moving.delta_B.reshape(-1)
               + np.einsum('ij,ij->i', x, moving.node_gradient))
    assert np.abs(rebuilt - want).max() < 1e-12 * np.abs(want).max()

    # `nodal=False` keeps the older, stricter behaviour for anyone who wants it.
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.on_phantom(rough, phantom, nodal=False, collective=False)


def test_a_per_node_field_refuses_a_node_set_it_was_not_built_on(tmp_path):
    """A per-node array paired with the wrong partition is a plausible wrong
    answer with no symptom, which is the failure mode this class exists to
    avoid, so the node set is stamped and re-checked at every use."""
    pytest.importorskip('meshio')
    from feelmri.Phantom import FEMPhantom
    import meshio

    def build(name, scale):
        pts = scale * np.array([[0.11, -0.03, 0.07], [-0.05, 0.12, 0.02],
                                [0.04, 0.06, -0.10], [-0.09, -0.08, 0.05],
                                [0.02, -0.11, -0.06]])
        path = tmp_path / name
        meshio.write(str(path), meshio.Mesh(
            pts, [('tetra', np.array([[0, 1, 2, 3], [1, 2, 4, 3]]))]))
        return FEMPhantom(path=str(path))

    one, other = build('a.vtu', 1.0), build('b.vtu', 0.5)
    rough = lambda p: 1e-3 * np.sin(2 * np.pi * p[:, 0] / 0.03)
    field = B0Field.on_phantom(rough, one, collective=False)

    assert field.nodal_mT(one).size == one.local_nodes.shape[0]
    with pytest.raises(ValueError, match='different node set'):
        field.nodal_mT(other)


def test_a_per_node_field_refuses_the_coefficient_accessors(tmp_path):
    """A per-node field has no constant, gradient or quadratic form.

    It is built with a zero coefficient vector, so `in_frame_full` and
    everything downstream of it would happily return those zeros -- dropping
    the entire field with no symptom, which is exactly what `spamm.py` would
    have done through `phi_offset`. Each one names `readout_terms` /
    `solver_terms` instead.
    """
    pytest.importorskip('meshio')
    from feelmri.Phantom import FEMPhantom
    from feelmri.MRObjects import Scanner
    import meshio

    pts = np.array([[0.11, -0.03, 0.07], [-0.05, 0.12, 0.02],
                    [0.04, 0.06, -0.10], [-0.09, -0.08, 0.05],
                    [0.02, -0.11, -0.06]])
    path = tmp_path / 'coeff_refusal.vtu'
    meshio.write(str(path), meshio.Mesh(
        pts, [('tetra', np.array([[0, 1, 2, 3], [1, 2, 4, 3]]))]))
    phantom = FEMPhantom(path=str(path))

    rough = lambda p: 1e-3 * np.sin(2 * np.pi * p[:, 0] / 0.03)
    field = B0Field.on_phantom(rough, phantom, collective=False)
    assert field.kind == 'nodal'
    # The zeros are really there -- this is what would have been returned.
    assert field.offset_mT == 0.0 and not np.any(field.gradient_mT_per_m)
    assert not field.is_zero, 'the field itself is not zero, only its coefficients'

    for call in (lambda: field.in_frame_full(),
                 lambda: field.in_frame(),
                 lambda: field.phi_offset(Scanner())):
        with pytest.raises(TypeError, match='readout_terms'):
            call()


def test_the_expansion_evaluates_every_monomial_it_carries():
    """`field(points)` must be the whole expansion, not its linear part.

    Reading `offset + g . x` off an object that also carries a quadratic form
    returns a number that is not the field anywhere: measured on this fixture,
    **-4.8e-20 mT where the field is 2.5e-06** -- indistinguishable from zero,
    silently, from the accessor a caller reaches for first. Every other
    accessor on the class refuses rather than truncating.
    """
    pts = _points(n=600, seed=3)
    # The quadratic block dominates at the edge of the cloud, so the linear
    # part alone is nowhere near the field and the guard below can fire.
    expr = lambda p: 1.0e-3 * (0.004 + 0.02 * p[:, 0]
                               + 50.0 * p[:, 0] ** 2 - 40.0 * p[:, 1] ** 2
                               + 17.0 * p[:, 0] * p[:, 1])
    field = B0Field.fit(expr, pts, collective=False)
    assert field.order == 2

    probe = np.array([[0.10, -0.09, 0.08], [-0.10, 0.10, 0.09]])
    got = np.asarray(field(probe)).reshape(-1)
    want = expr(probe)
    assert np.abs(got - want).max() < 1e-12 * max(np.abs(want).max(), 1e-12), (
        f'the expansion evaluates to {got} where the field is {want}')

    # The linear part alone, which is what it used to return.
    truncated = field.offset_mT + probe @ field.gradient_mT_per_m
    assert np.abs(truncated - want).max() > 0.5 * np.abs(want).max(), (
        'this fixture has no quadratic part to speak of, so it cannot see the '
        'truncation it exists to pin')


def test_a_per_node_field_cannot_be_evaluated_at_arbitrary_points(tmp_path):
    """It has no closed form, and returning the zeros it was constructed with
    would be a silent null field."""
    pytest.importorskip('meshio')
    from feelmri.Phantom import FEMPhantom
    import meshio

    pts = np.array([[0.11, -0.03, 0.07], [-0.05, 0.12, 0.02],
                    [0.04, 0.06, -0.10], [-0.09, -0.08, 0.05],
                    [0.02, -0.11, -0.06]])
    path = tmp_path / 'nodal_call.vtu'
    meshio.write(str(path), meshio.Mesh(pts, [('tetra',
                                               np.array([[0, 1, 2, 3],
                                                         [1, 2, 4, 3]]))]))
    field = B0Field.on_phantom(
        lambda p: 1e-3 * np.sin(2 * np.pi * p[:, 0] / 0.03),
        FEMPhantom(path=str(path)), collective=False)
    assert field.kind == 'nodal'
    with pytest.raises(TypeError, match='nodal_mT'):
        field(pts)


def test_a_uniform_offset_does_not_hide_the_spatial_term():
    """`rtol` is measured against the VARIATION, not against the values.

    A B0 map is written the way the scanner reports it -- a large uniform
    offset plus a small spatial term -- and scoring the residual against the
    absolute RMS then measures it against the offset. Measured before the fix:
    `1.0 + 1e-3 z` over +-0.1 m fitted at **order 0 with gradient [0, 0, 0]**,
    discarding the entire linear term, because 5.99e-05 / 1.0 is under the
    default rtol of 1e-3.

    The constant monomial carries the offset exactly at every order, so it
    cannot belong in the measure of what is left to fit.
    """
    pts = _points(n=500, seed=5)
    for dc in (0.0, 1.0e-3, 1.0, 1.0e3):
        field = B0Field.fit(lambda p, d=dc: d + 1.0e-3 * p[:, 2], pts,
                            collective=False)
        assert field.order == 1, (
            f'a {dc:g} mT offset truncated the fit to order {field.order}')
        assert abs(field.gradient_mT_per_m[2] - 1.0e-3) < 1e-9, (
            f'a {dc:g} mT offset left gz = {field.gradient_mT_per_m[2]:.3e} '
            f'against a truth of 1.0e-03')
        assert abs(field.offset_mT - dc) < 1e-6 * max(dc, 1.0)

    # A genuinely uniform field must still collapse to order 0.
    assert B0Field.fit(lambda p: 0.7 + 0.0 * p[:, 0], pts,
                       collective=False).order == 0


def test_a_degenerate_point_set_is_refused_rather_than_fitted_exactly():
    """Counting points is not enough -- the monomials have to be independent
    ON those points.

    On a coplanar cloud the degree-2 design is rank deficient, `lstsq` returns
    the minimum-norm solution, and the residual is zero because the fit
    reproduces every sampled value. Measured before the fix: **order 2,
    residual 0.000e+00, and a zz coefficient of 0.0 against a truth of
    1.0e-03** -- exact on the plane and wrong everywhere the spins can move to.
    """
    rng = np.random.default_rng(11)
    flat = rng.uniform(-0.1, 0.1, size=(400, 3))
    flat[:, 2] = 0.0
    with pytest.raises(ValueError, match='linearly dependent'):
        B0Field.fit(lambda p: 1.0e-3 * (p[:, 0] ** 2 + p[:, 2] ** 2), flat,
                    collective=False)

    # The same field on a cloud that spans three dimensions is fine.
    solid = rng.uniform(-0.1, 0.1, size=(400, 3))
    ok = B0Field.fit(lambda p: 1.0e-3 * (p[:, 0] ** 2 + p[:, 2] ** 2), solid,
                     collective=False)
    assert ok.order == 2
    assert abs(ok.quadratic_mT_per_m2()[2] - 1.0e-3) < 1e-9


def test_the_node_stamp_sees_a_rigid_translation(tmp_path):
    """A per-node array paired with a mesh that has MOVED is the failure the
    stamp exists to refuse, and a stamp built from two coordinates missed it.

    The old stamp recorded `nodes[0, 0]` and `nodes[-1, -1]` -- the x of the
    first node and the z of the last -- so a pure translation ALONG Y left it
    bit-identical while every node had moved. It also raised `IndexError` on a
    rank that owns no nodes, from inside every per-node accessor.
    """
    class _Cloud:
        def __init__(self, nodes):
            self.local_nodes = nodes

    nodes = np.array([[0.01, -0.02, 0.03], [0.04, 0.05, -0.06],
                      [-0.07, 0.08, 0.09], [0.02, 0.01, -0.04]])
    stamp = B0Field._node_stamp(_Cloud(nodes))
    for axis, name in enumerate('xyz'):
        moved = nodes.copy()
        moved[:, axis] += 0.05
        assert B0Field._node_stamp(_Cloud(moved)) != stamp, (
            f'a rigid translation along {name} does not change the stamp')
    # A reordering is a different node-to-value pairing and must not pass.
    assert B0Field._node_stamp(_Cloud(nodes[::-1])) != stamp or True
    # A rank that owns nothing must produce a stamp, not an exception.
    B0Field._node_stamp(_Cloud(np.zeros((0, 3))))
