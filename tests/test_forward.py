import croissant
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eigsep_mapmaking import alm, instrument, sky
from eigsep_mapmaking.forward import ForwardModel, mspace_alm
from conftest import FREQS, LMAX


@pytest.fixture(scope="module")
def fm(response, obs):
    return ForwardModel.from_response(response, obs)


@pytest.fixture(scope="module")
def x_sky(sky_maps, obs):
    return alm.pack(sky.sky_alm_sim(sky_maps, obs))


def test_predict_matches_croissant_convolve(fm, response, obs, x_sky):
    s = alm.unpack(x_sky)
    for p in range(obs.n_pointings):
        v = croissant.simulator.convolve(response.alm[p], s, obs.phases).real  # (T, F)
        want = (v + (response.norm[p] - response.above[p]) * obs.Tgnd_f) / response.norm[p]
        np.testing.assert_allclose(fm.predict(x_sky)[:, :, p], want.T, rtol=1e-12)


def test_adjoint_dot_product(fm):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, fm.shape_sky)
    y = jax.random.normal(k2, fm.shape_data)
    np.testing.assert_allclose(jnp.vdot(fm.apply(x), y), jnp.vdot(x, fm.adjoint(y)), rtol=1e-12)


def test_beam_scale_invariance(obs, beam_maps, x_sky, fm):
    """V is unchanged by B -> c B (per frequency), ground term included (review 2.2.1)."""
    c = np.array([3.0, 0.2])[:, None]
    fm2 = ForwardModel.from_response(instrument.beam_response(beam_maps * c, obs), obs)
    np.testing.assert_allclose(fm2.predict(x_sky), fm.predict(x_sky), rtol=1e-11)


def test_mspace_matches_time_domain(fm, x_sky):
    """Y = R conj(s) + ground for a uniform sidereal day (snapshots)."""
    y = fm.predict(x_sky)
    Y = fm.to_mspace(y)
    R = fm.mspace_response()
    pred = jnp.einsum("fmpl,fml->fmp", R, mspace_alm(alm.unpack(x_sky))) + fm.mspace_ground()
    np.testing.assert_allclose(Y, pred, atol=1e-8 * float(jnp.max(jnp.abs(Y))))  # JD rounding (~40 us)
    np.testing.assert_allclose(fm.window_factor(), 1.0, atol=1e-10)


def test_mspace_with_integration_windows(beam_maps, sky_maps, obs):
    times, ends = instrument.uniform_times(obs.n_times, integrate=True)
    o = instrument.Observation(times_jd=times, times_end_jd=ends, freqs=FREQS, lmax=LMAX,
                               pointings=obs.pointings)
    fm = ForwardModel.from_response(instrument.beam_response(beam_maps, o), o)
    x = alm.pack(sky.sky_alm_sim(sky_maps, o))
    Y = fm.to_mspace(fm.predict(x))
    pred = jnp.einsum("fmpl,fml->fmp", fm.mspace_response(), mspace_alm(alm.unpack(x)))
    pred = pred + fm.mspace_ground()
    np.testing.assert_allclose(Y, pred, atol=1e-8 * float(jnp.max(jnp.abs(Y))))  # JD rounding (~40 us)
    m = np.arange(LMAX + 1)
    dphi = 2 * np.pi / o.n_times
    np.testing.assert_allclose(fm.window_factor(),
                               np.exp(-1j * m * dphi / 2) * np.sinc(m * dphi / (2 * np.pi)),
                               atol=1e-9)
