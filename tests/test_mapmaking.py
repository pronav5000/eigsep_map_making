import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eigsep_mapmaking import alm, mapmaking, simulate
from eigsep_mapmaking.sky import SkyPrior


@pytest.fixture(scope="module")
def data(obs, beam_maps, sky_maps, response):
    return simulate.simulate(beam_maps, sky_maps, obs, noise_scale=300.0, response=response)


@pytest.fixture(scope="module")
def prior(data):
    return SkyPrior.oracle(alm.unpack(data.x_true), floor=1e-3)


@pytest.fixture(scope="module")
def dense(data, prior):
    return mapmaking.wiener_dense(data.fm, prior, data.sigma, data.y)


def test_cg_equals_dense(data, prior, dense):
    x_cg, resid = mapmaking.wiener_cg(data.fm, prior, data.sigma, data.y, tol=1e-12)
    assert resid < 1e-9
    np.testing.assert_allclose(x_cg, dense[0], atol=1e-7 * float(jnp.max(jnp.abs(dense[0]))))


def test_mspace_equals_dense(data, prior, dense):
    x_m = mapmaking.wiener_mspace(data.fm, prior, data.sigma, data.y)
    np.testing.assert_allclose(x_m, dense[0], atol=1e-6 * float(jnp.max(jnp.abs(dense[0]))))
    var = mapmaking.posterior_var_mspace(data.fm, prior, data.sigma)
    dvar = jnp.diag(dense[1]).reshape(var.shape)
    np.testing.assert_allclose(var, dvar, rtol=1e-5, atol=1e-8 * float(jnp.max(dvar)))


def test_mspace_rejects_nonstationary_noise(data, prior):
    with pytest.raises(ValueError):
        mapmaking.wiener_mspace(data.fm, prior, jnp.ones(data.y.shape), data.y)


def test_constrained_realizations_have_posterior_covariance(data, prior, dense):
    keys = jax.random.split(jax.random.PRNGKey(7), 400)
    draw = jax.jit(jax.vmap(lambda k: mapmaking.constrained_realization(
        data.fm, prior, data.sigma, data.y, k)))
    xs = np.asarray(draw(keys)).reshape(400, -1)
    mean, cov = dense[0].ravel(), np.asarray(dense[1])
    sd = np.sqrt(np.diag(cov))
    z = (xs.mean(0) - mean) / (sd / np.sqrt(400))
    assert np.mean(np.abs(z) < 3) > 0.97
    ratio = xs.var(0) / np.diag(cov)
    assert 0.85 < np.median(ratio) < 1.15


def test_clip_is_display_only(data):
    maps = mapmaking.clip_positive_for_display(data.x_true - 2 * data.x_true.mean(), nside=4)
    assert maps.shape == (2, 192) and np.all(maps >= 0)
