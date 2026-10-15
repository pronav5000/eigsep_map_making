import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eigsep_mapmaking import alm, instrument, perturbations as pt, sky
from eigsep_mapmaking.forward import ForwardModel
from conftest import FREQS, NSIDE


def _check_basis(basis, beam_maps):
    w = pt.quad_weights(beam_maps.shape[1:], basis.sampling)
    # orthogonal to the scale direction at every frequency
    c = pt._inner(basis.modes, beam_maps[None], w) / pt._inner(beam_maps, beam_maps, w)
    assert np.max(np.abs(c)) < 1e-10
    # orthogonal, each with the beam's RMS
    K = basis.k
    G = (basis.modes * w).reshape(K, -1) @ basis.modes.reshape(K, -1).T
    norm2 = pt._inner(beam_maps, beam_maps, w).sum()
    np.testing.assert_allclose(G, norm2 * np.eye(basis.k), atol=1e-8 * norm2)


@pytest.fixture(scope="module")
def band():
    """The synthetic beam over a band of 8 channels; FREQS are channels 0 and 1."""
    from eigsep_mapmaking import simulate
    return simulate.synthetic_beam_maps(NSIDE, np.arange(8) * 5.0 + 50.0)


def test_svd_basis(band, beam_maps):
    np.testing.assert_allclose(band[:2], beam_maps)
    b = pt.svd_basis(band, k=3, select=[0, 1])
    assert b.k == 3 and b.modes.shape == (3,) + beam_maps.shape
    _check_basis(b, beam_maps)
    one = pt.svd_basis(band, k=3, select=[1])  # a single channel, as in the notebooks
    assert one.k == 3
    _check_basis(one, beam_maps[1:])


def test_svd_raw_basis_is_mostly_scale(band, beam_maps):
    """Without removal, the leading SVD mode is mostly the beam itself (review 2.2.3)."""
    raw = pt.svd_basis(band, k=1, select=[0, 1], remove=False)
    w = pt.quad_weights(beam_maps.shape[1:], "healpix")
    m0 = raw.modes[0]
    along = pt._inner(m0, beam_maps, w) ** 2 / pt._inner(beam_maps, beam_maps, w)
    assert along.sum() / pt._inner(m0, m0, w).sum() > 0.95


def test_lowl_basis(beam_maps):
    b = pt.lowl_basis(beam_maps, lmax_pert=2, n_freq_funcs=2)
    assert b.k == 2 * (9 - 1)
    _check_basis(b, beam_maps)


def test_real_harmonics_orthonormal():
    Y = pt.real_harmonics(3, (12 * 16**2,), "healpix")
    G = Y @ Y.T * 4 * np.pi / Y.shape[-1]
    np.testing.assert_allclose(G, np.eye(16), atol=0.02)


def test_linear_model_equals_recompute(obs, band, beam_maps, sky_maps):
    basis = pt.svd_basis(band, k=2, select=[0, 1])
    lin = pt.LinearBeamModel.build(beam_maps, basis, obs)
    beta = np.array([0.03, -0.05])
    fm_lin = lin.forward_model(beta)
    pert = pt.inject_error(beam_maps, basis, beta)
    fm_full = ForwardModel.from_response(instrument.beam_response(pert, obs), obs)
    x = alm.pack(sky.sky_alm_sim(sky_maps, obs))
    np.testing.assert_allclose(fm_lin.predict(x), fm_full.predict(x), rtol=1e-11)
    # differentiable and jit-able in beta
    g = jax.jit(jax.grad(lambda b: jnp.sum(lin.forward_model(b).predict(x))))(jnp.zeros(2))
    assert np.all(np.isfinite(g)) and np.any(g != 0)


def test_scale_direction_is_invisible(obs, beam_maps, sky_maps):
    """A 'mode' equal to the beam itself changes nothing in the data."""
    basis = pt.BeamBasis(beam_maps[None], np.ones(1), "healpix")
    lin = pt.LinearBeamModel.build(beam_maps, basis, obs)
    x = alm.pack(sky.sky_alm_sim(sky_maps, obs))
    np.testing.assert_allclose(lin.forward_model([0.3]).predict(x),
                               lin.forward_model([0.0]).predict(x), rtol=1e-11)
