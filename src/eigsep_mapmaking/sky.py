"""Sky models: GSM maps, sky alm in the simulation frame, and Gaussian sky priors.

The sky is a real field T(n, nu) in K. Its alm in croissant's simulation frame come from
``croissant.Sky.compute_alm_eq(world, et=obs.et_ref)``, the frame of the beam alm from
``instrument.beam_response`` (CIRS at ``times_jd[0]`` on Earth).
"""

from dataclasses import dataclass

import croissant
import healpy as hp
import jax.numpy as jnp
import numpy as np

from . import alm as alm_mod


def gsm_maps(freqs_mhz, nside, model="gsm16", resolution="lo"):
    """Galactic HEALPix maps (RING) of a diffuse sky model in K (Rayleigh-Jeans), (F, Npix).

    Needs ``pygdsm`` (``pip install .[gsm]``) and network access to its data hosts the first
    time (zenodo.org / apps.datacentral.org.au). ``model`` is "gsm16" (GlobalSkyModel16,
    plan §8.2 Q3) or "gsm08"; ``resolution="lo"`` is GSM2016's native nside 64.
    """
    import pygdsm

    if model == "gsm16":
        gsm = pygdsm.GlobalSkyModel16(freq_unit="MHz", data_unit="TRJ", resolution=resolution)
    elif model == "gsm08":
        gsm = pygdsm.GlobalSkyModel(freq_unit="MHz")
    else:
        raise ValueError(f"unknown model {model!r}")
    maps = np.atleast_2d(gsm.generate(np.asarray(freqs_mhz, float)))
    if hp.npix2nside(maps.shape[-1]) != nside:
        maps = hp.ud_grade(maps, nside, order_in="RING", order_out="RING")
    return np.asarray(maps, float)


def save_sky(path, maps, freqs, coord="galactic"):
    """Write HEALPix sky maps to ``.npz`` (keys ``maps``, ``freqs``, ``nside``, ``coord``)."""
    maps = np.atleast_2d(maps)
    np.savez(path, maps=maps, freqs=np.asarray(freqs, float),
             nside=hp.npix2nside(maps.shape[-1]), coord=coord)


def load_sky(path):
    """Read a file written by ``save_sky``; returns ``(maps, freqs, coord)``."""
    with np.load(path) as f:
        return f["maps"], f["freqs"], str(f["coord"]) if "coord" in f else "galactic"


def sky_alm_sim(maps, obs, sampling="healpix", coord="galactic"):
    """Sky alm in the simulation frame of ``obs``, (F, L, 2L-1)."""
    maps = jnp.asarray(maps, float)
    kw = dict(lmax=obs.lmax) if sampling == "healpix" else {}
    sky = croissant.Sky(maps, obs.freqs, sampling=sampling, coord=coord, engine=obs.engine, **kw)
    out = sky.compute_alm_eq(world=obs.world, et=obs.et_ref)
    return alm_mod.reduce_lmax(out, obs.lmax)


def sky_maps_from_alm(alm, nside):
    """HEALPix maps (in the frame of the alm) from s2fft-layout alm (F, L, 2L-1) or (L, 2L-1).
    s2fft needs L >= 2 nside."""
    import s2fft

    alm = jnp.asarray(alm)
    single = alm.ndim == 2
    alm = alm[None] if single else alm
    L = alm.shape[-2]
    maps = np.stack([np.asarray(s2fft.inverse(a, L=L, nside=nside, sampling="healpix",
                                              reality=True, method="jax")).real for a in alm])
    return maps[0] if single else maps


@dataclass(frozen=True, eq=False)
class SkyPrior:
    """Gaussian, isotropic sky prior: diagonal covariance S in the packed basis, (F, L**2).

    Isotropy makes S diagonal with S_i = C_l(i) for every packed entry (``alm`` module).
    """

    S: jnp.ndarray

    @property
    def lmax(self):
        return alm_mod.lmax_from_n_real(self.S.shape[-1])

    @property
    def cl(self):
        """C_l per frequency, (F, L) (read off the m = 0 entries)."""
        return self.S[..., : self.lmax + 1]

    @classmethod
    def from_cl(cls, cl):
        """From C_l, (F, L) or (L,)."""
        cl = jnp.atleast_2d(jnp.asarray(cl, float))
        return cls(alm_mod.cl_to_packed(cl))

    @classmethod
    def oracle(cls, true_alm, floor=1e-4):
        """C_l of the true sky (an oracle prior, as in the notebooks), floored at
        ``floor`` times the largest C_l of each frequency. ``floor=None`` disables it."""
        cl = jnp.atleast_2d(alm_mod.cl_from_alm(true_alm))
        if floor is not None:
            cl = jnp.maximum(cl, floor * cl.max(axis=-1, keepdims=True))
        return cls.from_cl(cl)

    @classmethod
    def power_law(cls, lmax, freqs, amp, alpha=2.5, beta=2.5, nu0=50.0, monopole_sigma=None):
        """A non-oracle prior: C_l(nu) = amp * (l / 1)^(-alpha) * (nu / nu0)^(-2 beta) for
        l >= 1. The monopole gets C_0 = 4 pi monopole_sigma^2 (sigma in K on the mean sky
        temperature; default: the l = 1 value)."""
        freqs = np.atleast_1d(np.asarray(freqs, float))
        ell = np.arange(lmax + 1, dtype=float)
        shape = np.where(ell > 0, np.maximum(ell, 1.0) ** (-alpha), 1.0)
        cl = amp * shape[None, :] * (freqs[:, None] / nu0) ** (-2 * beta)
        if monopole_sigma is not None:
            mono = 4 * np.pi * (np.asarray(monopole_sigma) * np.ones_like(freqs)) ** 2
            cl[:, 0] = mono * (freqs / nu0) ** (-2 * beta) if np.ndim(monopole_sigma) == 0 else mono
        return cls.from_cl(cl)

    def sample(self, key):
        """A packed sky realization, (F, L**2)."""
        import jax

        return jnp.sqrt(self.S) * jax.random.normal(key, self.S.shape)
