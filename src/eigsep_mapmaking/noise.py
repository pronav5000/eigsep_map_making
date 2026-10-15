"""Radiometer noise.

sigma = T_ant / sqrt(df * dt) with df the channel width in Hz and dt the integration time
in seconds. The notebooks had two unit errors here (review 2.1.3): seconds added to Julian
days made dt 86400x too large (sigma ~294x too small), and df in Hz x 1e6 made sigma 1000x
too small. Both are avoided by computing dt from the ``Observation`` in seconds.
"""

import jax
import jax.numpy as jnp
import numpy as np


def radiometer_sigma(T_ant, df_hz, dt_s):
    """sigma = T_ant / sqrt(df_hz * dt_s)."""
    return jnp.asarray(T_ant) / jnp.sqrt(df_hz * dt_s)


def channel_width_hz(freqs_mhz):
    """Channel width in Hz from evenly spaced channel centres in MHz."""
    freqs_mhz = np.atleast_1d(np.asarray(freqs_mhz, float))
    if len(freqs_mhz) < 2:
        raise ValueError("one channel: pass the channel width explicitly")
    d = np.diff(freqs_mhz)
    if not np.allclose(d, d[0]):
        raise ValueError("channels are not evenly spaced: pass the widths explicitly")
    return float(d[0]) * 1e6


def integration_time_s(obs):
    """Integration time per sample in seconds, (T,): the window length when the observation
    has windows, else the spacing between samples (the last sample gets the previous one)."""
    if obs.times_end_jd is not None:
        return (obs.times_end_jd - obs.times_jd) * 86400.0
    if obs.n_times < 2:
        raise ValueError("one snapshot: no integration time")
    dt = np.diff(obs.times_jd) * 86400.0
    return np.append(dt, dt[-1])


def sigma_from_model(y_model, df_hz, dt_s, stationary=True, floor=0.0):
    """Noise level for data y (F, T, P) given the model antenna temperature ``y_model``.

    stationary=True (plan D8 default): one sigma per (frequency, pointing) from the
    time-averaged antenna temperature, shape (F, 1, P). Otherwise sigma(t) proportional to
    T_ant(t), shape (F, T, P). ``floor`` adds a fractional model-error term in quadrature
    (off by default; the notebooks used 1.5 %).
    """
    y_model = jnp.asarray(y_model)
    dt_s = jnp.asarray(dt_s)[None, :, None] if jnp.ndim(dt_s) == 1 else dt_s
    df = jnp.reshape(jnp.asarray(df_hz, float), (-1, 1, 1)) if jnp.ndim(df_hz) else df_hz
    if stationary:
        T = jnp.mean(y_model, axis=1, keepdims=True)
        sig = radiometer_sigma(T, df, jnp.mean(dt_s, axis=1, keepdims=True)
                               if jnp.ndim(dt_s) else dt_s)
        mod = floor * jnp.mean(jnp.abs(y_model), axis=1, keepdims=True)
    else:
        sig = radiometer_sigma(y_model, df, dt_s)
        mod = floor * jnp.abs(y_model)
    return jnp.sqrt(sig**2 + mod**2)


def realize(sigma, shape, key):
    """A Gaussian noise realization with standard deviation ``sigma`` (broadcast to shape)."""
    return jnp.broadcast_to(sigma, shape) * jax.random.normal(key, shape)
