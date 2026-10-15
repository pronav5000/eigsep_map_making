import jax
import jax.numpy as jnp
import numpy as np

from eigsep_mapmaking import instrument, noise, simulate
from conftest import FREQS, LMAX


def test_radiometer_and_units(obs):
    np.testing.assert_allclose(noise.radiometer_sigma(1000.0, 1e6, 100.0), 0.1)
    assert noise.channel_width_hz(FREQS) == 5e6
    dt = noise.integration_time_s(obs)
    day = 86164.0905  # ~ sidereal day in s
    np.testing.assert_allclose(dt, day / obs.n_times, rtol=1e-4)  # seconds, not days*86400^2
    times, ends = instrument.uniform_times(10, integrate=True)
    o = instrument.Observation(times_jd=times, times_end_jd=ends, freqs=FREQS, lmax=2)
    np.testing.assert_allclose(noise.integration_time_s(o), day / 10, rtol=1e-4)


def test_sigma_shapes():
    y = 100.0 + jnp.arange(24.0).reshape(2, 4, 3)
    s = noise.sigma_from_model(y, 1e6, np.full(4, 100.0))
    assert s.shape == (2, 1, 3)
    np.testing.assert_allclose(s[:, 0], y.mean(1) / 1e4)
    s2 = noise.sigma_from_model(y, 1e6, np.full(4, 100.0), stationary=False, floor=0.01)
    np.testing.assert_allclose(s2, jnp.sqrt((y / 1e4) ** 2 + (0.01 * y) ** 2))


def test_realize_statistics():
    n = noise.realize(jnp.array([[[2.0]], [[0.5]]]), (2, 20000, 1), jax.random.PRNGKey(1))
    np.testing.assert_allclose(n.std(axis=(1, 2)), [2.0, 0.5], rtol=0.03)


def test_simulate(obs, beam_maps, sky_maps, response):
    d = simulate.simulate(beam_maps, sky_maps, obs, noise_scale=100.0, response=response)
    assert d.y.shape == (2, obs.n_times, 3) and d.sigma.shape == (2, 1, 3)
    r = (d.y - d.y_clean) / d.sigma
    assert 0.6 < float(jnp.std(r)) < 1.4  # noisy data, not noiseless (review 2.1.4)
    np.testing.assert_allclose(d.fm.predict(d.x_true), d.y_clean)
    assert np.all(d.y_clean > 0)
    # noise level: ~T_ant / sqrt(5 MHz * 4100 s) * 100
    np.testing.assert_allclose(d.sigma[:, 0], 100 * d.y_clean.mean(1) /
                               np.sqrt(5e6 * noise.integration_time_s(obs)[0]), rtol=1e-6)
