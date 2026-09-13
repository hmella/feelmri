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


def test_an_expression_needing_a_quadratic_is_refused_with_what_it_would_lose():
    """Silently truncating a field the expansion cannot carry would be the
    worst outcome, so it raises and reports what order 1 leaves behind."""
    with pytest.raises(NotImplementedError, match='needs order 2'):
        B0Field.fit(lambda p: 1e-3 * (p[:, 0] ** 2 - p[:, 2] ** 2), _points(),
                    collective=False)


def test_a_field_no_polynomial_can_represent_is_refused():
    """A step is not a smooth scanner field: a static field in a current-free
    bore is a solid-harmonic series. Such a map is tissue structure and belongs
    on the per-node channel."""
    with pytest.raises(ValueError, match='not a smooth scanner field'):
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
