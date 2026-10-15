"""Synthetic beams and skies, and mock datasets.

Everything here runs without data files, so tests and examples can use it.
"""

import healpy as hp
import numpy as np


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
