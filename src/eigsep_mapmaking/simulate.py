"""Synthetic beams and skies, and mock datasets.

Everything here runs without data files, so tests and examples can use it.
"""

from dataclasses import dataclass

import healpy as hp
import jax
import numpy as np

from . import alm, instrument, noise, sky
from .forward import ForwardModel


def synthetic_beam_maps(nside, freqs, fwhm0_deg=60.0, nu0=50.0, ellipticity=0.3,
                        backlobe=0.02, seed=None):
    """A smooth, chromatic, full-sphere power beam in the beam frame, (F, Npix).

    Main lobe at the zenith with width ~ fwhm0 * (nu0 / nu), a cos(2 phi) ellipticity that
    rotates slowly with frequency (so the beam has azimuthal structure the m-decomposition
    can see), and a uniform back lobe of relative level ``backlobe`` below the horizon.
    ``seed`` adds a small random distortion.
    """
    freqs = np.atleast_1d(np.asarray(freqs, float))
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    out = []
    rng = np.random.default_rng(seed)
    wobble = 0.05 * rng.normal(size=4) if seed is not None else np.zeros(4)
    for nu in freqs:
        sigma = np.radians(fwhm0_deg) * (nu0 / nu) / 2.3548
        psi = 0.2 * (nu - nu0) / nu0
        shape = 1 + ellipticity * np.cos(2 * (phi - psi)) + wobble[0] * np.cos(phi + wobble[1])
        main = np.exp(-0.5 * (theta / sigma) ** 2 / shape) * (1 + wobble[2] * np.cos(3 * phi))
        back = backlobe * (1 + 0.5 * np.cos(theta)) * (theta > np.pi / 2)
        out.append(main + back)
    return np.array(out)


def synthetic_sky_maps(nside, freqs, nu0=50.0, T0=3000.0, beta=2.5, dbeta=0.1, lmax=None,
                       seed=0):
    """A positive galactic-like sky in K, (F, Npix): a power-law-in-frequency random field
    with a spatially varying spectral index (so frequencies are not proportional)."""
    freqs = np.atleast_1d(np.asarray(freqs, float))
    lmax = 3 * nside - 1 if lmax is None else lmax
    rng = np.random.default_rng(seed)
    ell = np.arange(lmax + 1)
    cl = np.where(ell > 0, (np.maximum(ell, 1) / 1.0) ** -2.5, 0.0)
    np.random.seed(rng.integers(2**31))
    base = hp.synfast(cl, nside, lmax=lmax)
    base = T0 * (1 + 0.4 * base / base.std())
    base = np.maximum(base, 0.2 * T0)
    db = hp.synfast(cl, nside, lmax=lmax)
    index = beta + dbeta * db / db.std()
    return base[None] * (freqs[:, None] / nu0) ** (-index[None])


@dataclass(frozen=True, eq=False)
class Dataset:
    """A mock observation. ``y`` = ``y_clean`` + noise, all (F, T, P); ``sigma`` is (F, 1, P)
    (stationary) or (F, T, P). ``x_true`` is the packed true sky (F, L**2) and ``fm`` the
    true forward model."""

    obs: object
    fm: object
    x_true: jax.Array
    y_clean: jax.Array
    y: jax.Array
    sigma: jax.Array
    response: object = None


def simulate(beam_maps, sky_maps, obs, beam_sampling="healpix", sky_sampling="healpix",
             df_hz=None, stationary=True, noise_scale=1.0, floor=0.0, seed=0, response=None):
    """Simulate drift-scan data with radiometer noise.

    ``df_hz`` defaults to the channel spacing; the integration time comes from ``obs`` in
    seconds. ``noise_scale`` multiplies sigma (e.g. to stand in for less integration).
    ``response`` reuses a precomputed ``instrument.BeamResponse``.
    """
    if response is None:
        response = instrument.beam_response(beam_maps, obs, sampling=beam_sampling)
    fm = ForwardModel.from_response(response, obs)
    x_true = alm.pack(sky.sky_alm_sim(sky_maps, obs, sampling=sky_sampling))
    y_clean = fm.predict(x_true)
    if df_hz is None:
        df_hz = noise.channel_width_hz(obs.freqs)
    sigma = noise_scale * noise.sigma_from_model(
        y_clean, df_hz, noise.integration_time_s(obs), stationary=stationary, floor=floor)
    y = y_clean + noise.realize(sigma, y_clean.shape, jax.random.PRNGKey(seed))
    return Dataset(obs, fm, x_true, y_clean, y, sigma, response)
