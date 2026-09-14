"""The lab-frame B0 expansion.

`B0Field` is the scanner-fixed counterpart of the per-node `delta_B` /
`phi_dB0` channel: it is sampled at the spin's current position instead of
being frozen to the node. These tests cover the expansion and the frame
algebra alone; the physics that the two channels behave differently under
motion lives in `test_signal_analytical.py`.
"""
import warnings

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


@pytest.mark.parametrize('name,expr,order', [
    ('uniform', lambda p: np.full(p.shape[0], 1.0e-3), 0),
    ('linear', lambda p: 2.5e-4 + p @ np.array([0.011, -0.004, 0.0075]), 1),
    ('quadratic', lambda p: 1.0e-3 * (p[:, 0] ** 2 - p[:, 2] ** 2), 2),
])
def test_the_fit_lands_on_the_degree_the_field_actually_has(name, expr, order):
    """The degree detects itself: the search stops where the residual reaches
    round-off, because a residual that small means the expression IS that
    polynomial rather than being approximated by it. Every shim is one.

    Both directions matter. Auto truncation must not spend a linear term on a
    constant -- order 0 rides `delta_B` and costs nothing at all -- and must
    not stop at order 1 on a field that has a quadratic part, which would
    silently drop it.

    The RECONSTRUCTION is what carries the claim, not the reported residual:
    the normal equations square the condition number, so the residual bottoms
    out at `sqrt(eps)` whatever happens, while evaluating the fit back at the
    points shows whether the field is carried exactly.
    """
    pts = _points()
    field = B0Field.fit(expr, pts, collective=False)
    assert field.order == order, (
        f'{name} was carried at order {field.order}')
    want = expr(pts)
    got = B0Field._design(pts, field.order) @ field.coefficients
    assert np.abs(got - want).max() < 1e-12 * max(np.abs(want).max(), 1e-30)
    if order == 0:
        assert not np.any(field.gradient_mT_per_m) and field.is_zero is False
    if order == 1:
        assert field.offset_mT == pytest.approx(2.5e-4, abs=1e-15)


def test_a_degree_above_the_channels_is_refused_rather_than_truncated():
    """Degree 3 is above what the solver and the readout can carry, so it is
    not detected by default -- detecting it would only let it be truncated to
    a quadratic. Asked for explicitly, the frame adapter refuses it by name
    rather than dropping the cubic monomials: measured on a Z3 shim that is
    100% of the field.

    A field no polynomial can represent at all takes the same exit. A step is
    not a smooth scanner field -- a static field in a current-free bore is a
    solid-harmonic series -- so such a map is tissue structure and belongs on
    the per-node channel.
    """
    Z3 = lambda p: 1e-3 * p[:, 2] * (2 * p[:, 2] ** 2 - 3 * p[:, 0] ** 2)
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.fit(Z3, _points(), collective=False)
    with pytest.raises(NotImplementedError, match='drop every higher monomial'):
        B0Field.fit(Z3, _points(), collective=False, max_order=3).in_frame_full()
    with pytest.raises(ValueError, match='needs the per-node expansion'):
        B0Field.fit(lambda p: 1e-3 * np.sign(p[:, 0]), _points(),
                    collective=False)


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


def test_a_polynomial_field_refuses_the_linear_only_channels():
    """`in_frame` carries the constant and the gradient, which for a degree-2
    field is not the field. Returning them would silently truncate it, so it
    raises instead -- the quadratic part has its own channel."""
    quad = B0Field.fit(lambda p: 1e-3 * (p[:, 0] ** 2 - p[:, 2] ** 2),
                       _points(), collective=False)
    with pytest.raises(NotImplementedError, match='silently truncate'):
        quad.in_frame()


def test_a_pint_expression_and_a_bare_one_agree():
    """Examples write bare arrays; a Quantity must work too, and must not be
    read as though its magnitude were already mT."""
    g = np.array([0.011, -0.004, 0.0075])
    pts = _points()
    bare = B0Field.fit(lambda p: p @ g, pts, collective=False)
    pint = B0Field.fit(lambda p: Quantity(p @ g * 1e-3, 'T'), pts,
                       collective=False)
    assert np.abs(bare.gradient_mT_per_m - pint.gradient_mT_per_m).max() < 1e-15


def test_malformed_input_is_refused_at_the_boundary():
    """Each of these produces a plausible object that fails much later, deep
    inside the solver, if it is accepted here."""
    with pytest.raises(ValueError, match='must map'):
        B0Field.fit(lambda p: np.zeros(p.shape[0] - 1), _points(),
                    collective=False)
    with pytest.raises(ValueError, match='3-vector'):
        B0Field(gradient=Quantity(np.zeros(2), 'mT/m'))
    with pytest.raises(ValueError, match='no degree at all'):
        B0Field.fit(lambda p: np.zeros(p.shape[0]), _points(),
                    collective=False, max_order=-1)


def test_the_uniform_part_converts_to_an_offresonance_rate():
    """The constant is spatially uniform, so it needs no frame and can ride
    `phi_dB0` directly."""
    from feelmri import Scanner
    scanner = Scanner()
    field = B0Field(Quantity(1e-3, 'mT'))
    assert field.uniform_phi_rate(scanner) == pytest.approx(
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
    assert terms.node_gradient_mT_per_m is None
    assert np.abs(terms.delta_B.reshape(-1) - want).max() == 0.0

    # A moving phantom gets the Eulerian expansion instead: the bracket on
    # `delta_B` and a per-node gradient for the kernel. At zero displacement
    # the two must reconstruct the same field.
    moving = field.solver_terms(phantom, moving=True)
    assert moving.node_gradient_mT_per_m is not None
    # The per-node rung and the polynomial channels are mutually exclusive:
    # `quadratic` and `gradient` describe a GLOBAL expansion the kernel hoists
    # per time step, `node_gradient_mT_per_m` a per-node one it reads per
    # node, and the
    # kernel would add both. Nothing pinned this after the accessor rework.
    assert moving.gradient is None and moving.quadratic is None
    x = np.asarray(phantom.local_nodes, dtype=np.float64)
    rebuilt = (moving.delta_B.reshape(-1)
               + np.einsum('ij,ij->i', x, moving.node_gradient_mT_per_m))
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
                 lambda: field.uniform_phi_rate(Scanner())):
        with pytest.raises(TypeError, match='readout_terms'):
            call()
    # `__call__` is the same claim one level down: there is no closed form to
    # evaluate, and the coefficient vector it would evaluate is those zeros.
    with pytest.raises(TypeError, match='nodal_mT'):
        field(pts)


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


def test_the_node_stamp_sees_a_translation_and_a_reordering(tmp_path):
    """A per-node array paired with a mesh that has MOVED, or whose nodes have
    been RENUMBERED, is the failure the stamp exists to refuse.

    The original stamp recorded `nodes[0, 0]` and `nodes[-1, -1]` -- the x of
    the first node and the z of the last -- so a pure translation ALONG Y left
    it bit-identical while every node had moved. It also raised `IndexError` on
    a rank that owns no nodes, from inside every per-node accessor.

    Replacing those with two MOMENTS fixed the translation and silently gave
    up the reordering: `nodes.sum(axis=0).sum()` and `flat @ flat` are both
    symmetric functions of the rows, so a permuted node set produced a
    BIT-IDENTICAL stamp. That is the worse half of the two -- under a
    translation the values are at least wrong everywhere, while under a
    renumbering every value is still valid and merely belongs to a different
    node. The assertion that was supposed to catch it read
    `assert ... != stamp or True`, which is true whatever the stamp does.
    The index-weighted sum is the term that sees it.
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
    assert B0Field._node_stamp(_Cloud(nodes[::-1])) != stamp, (
        'a renumbered node set produces the same stamp, so a per-node array '
        'can be paired with the wrong nodes and accepted')
    # And a permutation that is not a reversal, so the test is not passing on
    # one special case.
    order = np.array([2, 0, 3, 1])
    assert B0Field._node_stamp(_Cloud(nodes[order])) != stamp
    # A rank that owns nothing must produce a stamp of the same shape, not an
    # exception and not a tuple the comparison above would fail to compare.
    empty = B0Field._node_stamp(_Cloud(np.zeros((0, 3))))
    assert len(empty) == len(stamp)


def test_a_fit_that_only_interpolates_the_nodes_falls_back_to_the_nodes(tmp_path):
    """An exact fit is not the same as a fit that represents the field.

    A coarse structured mesh carries far fewer DISTINCT coordinate values than
    nodes: 27 nodes on a 3x3x3 lattice have three distinct values per axis, and
    a degree-2 polynomial passes through any three points exactly. Measured,
    `sin(x / 0.02)` over 10 radians fitted at **order 2 with a residual of
    0.0** -- exact at every node, wrong everywhere between them.

    Counting points cannot catch it (10 monomials through 27 points) and nor
    can the rank of the design, which is full. What catches it is evaluating
    the fit where it was NOT fitted: the element centroids, which cost one
    expression call and are exactly the places a lattice hides.

    It costs nothing while the phantom is still -- the solver only ever asks
    for the field at the nodes -- and it is the whole answer once the spins
    move, because the polynomial is then evaluated off the lattice.
    """
    pytest.importorskip('meshio')
    from feelmri.Phantom import FEMPhantom
    from _phantom_fixtures import make_cube_mesh

    for n, scale in ((2, 0.1), (4, 0.05)):
        path, _v = make_cube_mesh(tmp_path / f'holdout_{n}.vtu', 'tetra',
                                  n=n, scale=scale)
        phantom = FEMPhantom(path=str(path))

        rough = B0Field.on_phantom(lambda q: 1.0e-3 * np.sin(q[:, 0] / 0.02),
                                   phantom, collective=False)
        assert rough.kind == 'nodal', (
            f'n={n}: a sine over 10 radians was carried as a {rough.kind} '
            f'expansion of order {rough.order}')

        # A field that genuinely IS a polynomial still fits, on the same mesh:
        # a polynomial generalises off the lattice by construction, so the
        # guard cannot be satisfied by refusing everything.
        poly = B0Field.on_phantom(
            lambda q: 1.0e-3 * (0.3 + 2.0 * q[:, 0] + 5.0 * q[:, 0] ** 2
                                - 4.0 * q[:, 1] ** 2 + 1.7 * q[:, 0] * q[:, 1]),
            phantom, collective=False)
        assert poly.kind == 'polynomial' and poly.order == 2, (
            f'n={n}: a genuine degree-2 field came back as {poly.kind} '
            f'order {poly.order}')

    # The guard has to be able to RUN before its verdict means anything. It
    # seeded the field's lo/hi at 0.0 on a rank owning no elements and reduced
    # with MIN/MAX, so on a field that does not straddle zero -- a shim written
    # as a large uniform offset plus a small spatial term, which is how this
    # class documents them -- the span picked up the whole offset and the
    # threshold could not fire. Measured: 1.8e-04 mT of real variation
    # reported as 5.0 mT, a factor of 2.8e+04.
    lat = np.stack(np.meshgrid(*[np.linspace(-0.1, 0.1, 4)] * 3,
                               indexing='ij'), -1).reshape(-1, 3)
    elems = np.random.default_rng(0).permutation(len(lat))[:60].reshape(-1, 4)

    class _Mesh:
        def __init__(self, nodes, cells):
            self.local_nodes, self.local_elements = nodes, cells

    shim = lambda q: 5.0 + 1.0e-3 * q[:, 2]
    fitted = B0Field.fit(shim, lat, collective=False)
    _gap, span = B0Field._holdout_residual(shim, _Mesh(lat, elems), lat,
                                           fitted, False)
    assert span < 1.0e-3, (
        f'the span is {span:.4g} mT for a field whose variation is 2e-4 mT; '
        f'it is measuring the DC offset, so the guard cannot fire')

    # And no connectivity ANYWHERE is "could not check", not "checked and
    # clean" -- the one input class the guard cannot inspect was the one it
    # waved through.
    _gap0, span0 = B0Field._holdout_residual(
        shim, _Mesh(lat, np.zeros((0, 4), dtype=int)), lat, fitted, False)
    assert span0 is None

    # `_mesh_residual` has the same distinction to draw and is the more
    # dangerous of the two, because 0.0 there reads as "the per-node
    # representation loses nothing between the nodes" -- the most reassuring
    # answer it can give, for a check that could not run.
    measured = B0Field._mesh_residual(shim, _Mesh(lat, elems), lat,
                                      collective=False)
    assert measured is not None and measured > 0.0
    assert B0Field._mesh_residual(
        shim, _Mesh(lat, np.zeros((0, 4), dtype=int)), lat,
        collective=False) is None

    # A fit ABOVE the class cap is not a usable field: `quadratic_mT_per_m2`
    # and `in_frame_full` both refuse degree 3, so it used to be handed back
    # and raise from inside `BlochSolver` -- the outcome the docstring's
    # promised fallback exists to prevent.
    cubic = B0Field.on_phantom(lambda q: 1.0e-3 * q[:, 2] ** 3,
                               _Mesh(lat, elems), collective=False,
                               max_order=3)
    assert cubic.kind == 'nodal', (
        f'a degree-3 fit came back as a {cubic.kind} expansion of order '
        f'{cubic.order}, which nothing downstream can consume')


def test_the_degenerate_node_sets_a_rank_can_own_are_carried_not_crashed():
    """Under MPI a rank can own no nodes at all, and an audit-6 guard turned
    two of those into hard failures on exactly those ranks -- which is the
    worst place for one, because the surviving ranks then block in the next
    collective rather than reporting anything.

    `fit` broke on its FIRST iteration when there were no points (the degree-0
    design is rank-deficient there), left `best` unassigned and died unpacking
    it: `TypeError: cannot unpack non-iterable NoneType object`. Degree 0 is
    one constant monomial, so it is the fit that must always produce an
    answer; an empty cloud has nothing to fit and the honest result is the
    uniform field at the mean.

    `__repr__` called `.max()` on the empty per-node array, raising
    `ValueError: zero-size array to reduction operation maximum`, so a log line
    on all ranks aborted a strict subset of them.

    (A negative `max_order` reached the same unpack and is refused by name;
    that one is asserted with the other boundary refusals.)
    """
    zero = lambda p: np.zeros(p.shape[0])

    empty = B0Field.fit(zero, np.zeros((0, 3)), collective=False)
    assert empty.kind == 'uniform' and empty.is_zero

    class _Cloud:
        def __init__(self, nodes):
            self.local_nodes = nodes

    rough = B0Field.on_phantom(lambda p: np.sin(41.0 * p[:, 0]) * 1e-3,
                               _Cloud(_points(60)), collective=False)
    assert rough.kind == 'nodal'
    # Exactly what a rank owning no nodes holds once the collective build is
    # done: the fit is global, the sampling is local and local is empty.
    rough._nodal_mT = np.zeros(0)
    assert 'nodes=0' in repr(rough)


def test_the_fit_is_a_property_of_the_field_not_of_the_geometry_it_is_sampled_on():
    """The same field on the same shape of cloud must fit the same way whether
    that cloud is 20 cm across or 2 microns.

    It did not. The design's monomial columns span `L^degree`, so on a cloud a
    millimetre across the quadratic columns are ~1e-6 of the constant one, and
    the NORMAL equations square that. Two separate failures followed, and the
    first hid the second:

    * the rank was read off the normal matrix, whose condition number is the
      square of the design's, so the effective refusal threshold was
      `cond(design) ~ 1.5e7` -- about half the decades a rank test on the
      design allows. An exact degree-2 field over a 0.1 mm box was reported as
      "linearly dependent on these points", which is a false diagnosis: the
      points are fine, the normal equations are not.
    * with that corrected the fit still failed, now honestly -- it came back
      with a residual of 4.388e-12 mT against a variation of 4.403e-12, i.e.
      99.7%, for a field it represents exactly.

    Fitting on positions scaled to unit RMS radius and scaling the
    coefficients back fixes both, and makes the answer independent of the
    units the geometry happens to be in.
    """
    truth = lambda p: 1.0e-3 * (p[:, 0] ** 2 + p[:, 1] * p[:, 2])
    for half in (1.0e-1, 1.0e-3, 1.0e-5):
        pts = np.random.default_rng(0).uniform(-half, half, size=(400, 3))
        field = B0Field.fit(truth, pts, collective=False)
        want = truth(pts)
        err = np.abs(field(pts) - want).max() / np.abs(want).max()
        assert field.order == 2, (
            f'a cloud {2 * half:g} m across was refused or truncated; the fit '
            f'came back at order {field.order}')
        assert err < 1.0e-12, (
            f'at a half-extent of {half:g} m the degree-2 fit of an exact '
            f'degree-2 field is {err:.3e} off')

    # A CONSTANT field divides 0/0 in the finite-difference smoothness check:
    # its gradient is zero everywhere, so the pointwise floor derived from the
    # peak gradient is zero too. It produced a RuntimeWarning and a NaN drift,
    # which is an exception under `-W error`.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        g = B0Field._sample_gradient(lambda p: np.full(p.shape[0], 2.0),
                                     _points(60), collective=False)
    assert np.all(np.isfinite(g)) and not np.any(g)


def test_the_accessors_of_one_field_cannot_describe_two_different_fields():
    """`__call__` evaluates the coefficient vector; `in_frame_full`, `phi_offset`
    and `in_frame` read the `offset_mT` / `gradient_mT_per_m` mirrors.

    The constructor checked only that the vector had the right LENGTH for its
    order, so a hand-built field could answer 0 at a point while reporting a
    5 mT offset -- two fields in one object, each self-consistent on its own
    accessors, and which one the solver sees depends on which path it takes.
    `fit` builds both halves together so its fields were always consistent;
    this bites exactly the hand-built spelling the tests use.
    """
    with pytest.raises(ValueError, match='disagree'):
        B0Field(Quantity(5.0, 'mT'), Quantity([1.0, 0.0, 0.0], 'mT/m'),
                order=2, coefficients=np.zeros(10))

    # The consistent spelling still constructs, and the two halves agree.
    coef = np.zeros(10)
    coef[0], coef[1] = 5.0, 1.0
    field = B0Field(Quantity(5.0, 'mT'), Quantity([1.0, 0.0, 0.0], 'mT/m'),
                    order=2, coefficients=coef)
    b, g = field.in_frame()
    assert field([[1.0, 0.0, 0.0]])[0] == pytest.approx(b + g[0])
