"""Observation set-up and the instrument response, built on croissant.

Frames (see docs/math.md, section 2):

* the *beam frame* is topocentric for a zenith-pointing antenna: z up, phi = 0 East,
  phi = pi/2 North (croissant's convention);
* a *pointing* is an active rotation R of the beam in the topocentric frame,
  given as ZYZ Euler angles; ``pointing_euler`` builds R from a tilt and an azimuth;
* the *simulation frame* is croissant's: CIRS at ``times_jd[0]`` on Earth (MEPA on
  the Moon). The sky rotates in it as alm -> alm * exp(-i m phi(t)).

For every pointing the beam pattern is rotated with ``croissant.rotations.rotate_to_grid``
onto an MWSS grid (exact for band-limited input), weighted by the ground-fixed horizon,
transformed with ``croissant.Beam``, and rotated to the simulation frame with
``Simulator.topo_to_sim``. The result, ``BeamResponse``, is linear in the input beam
pattern, which ``perturbations`` relies on.
"""

from dataclasses import dataclass, field
from functools import cached_property

import croissant
import jax
import jax.numpy as jnp
import numpy as np
from astropy.time import Time

# Plan §8.2 Q5: the notebooks' setup.
DEFAULT_SITE = dict(lat=39.2477, lon=-113.4033, alt=0.0)
DEFAULT_START = "2025-12-01 09:00:00"


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pointing_matrix(tilt_deg, az_deg=0.0, rot_deg=0.0):
    """Active rotation matrix of a pointing in the topocentric frame (x East, y North, z up).

    The antenna is first turned about the zenith by ``rot_deg`` (clockwise seen from above,
    croissant's ``beam_rot`` convention), then its boresight is tilted from the zenith by
    ``tilt_deg`` towards compass azimuth ``az_deg`` (North = 0, East = 90).
    """
    phi0 = np.pi / 2 - np.radians(az_deg)  # grid longitude of the tilt direction
    return _rz(phi0) @ _ry(np.radians(tilt_deg)) @ _rz(-phi0) @ _rz(-np.radians(rot_deg))


def pointing_euler(tilt_deg, az_deg=0.0, rot_deg=0.0):
    """ZYZ Euler angles (radians) of ``pointing_matrix``, as ``rotate_to_grid`` takes them."""
    return np.array(croissant.rotations.rotmat_to_eulerZYZ(pointing_matrix(tilt_deg, az_deg, rot_deg)))


def default_pointings(n=20, max_tilt_deg=90.0, az_deg=0.0):
    """The notebooks' pointings: ``n`` tilts from 0 to ``max_tilt_deg`` about one horizontal
    axis. The notebooks used healpy ``Rotator(rot=(0, 0, el))``, a rotation about the
    East axis, which tilts the boresight along the North-South line; here it tilts
    towards ``az_deg`` (default North)."""
    return np.stack([pointing_euler(t, az_deg) for t in np.linspace(0.0, max_tilt_deg, n)])


def uniform_times(n_times, start=DEFAULT_START, world="earth", n_days=1.0, integrate=False):
    """``n_times`` samples evenly covering ``n_days`` sidereal days from ``start``.

    Returns ``(times_jd, times_end_jd)``; with ``integrate=True`` each sample integrates
    over its whole slot (no gaps), otherwise ``times_end_jd`` is None (snapshots).
    """
    day = croissant.constants.sidereal_day[world] / 86400.0  # in days
    t0 = Time(start).jd if isinstance(start, str) else float(start)
    dt = n_days * day / n_times
    times = t0 + np.arange(n_times) * dt
    return times, (times + dt if integrate else None)


@dataclass(frozen=True, eq=False)
class Observation:
    """Everything about an observation except the beam pattern and the sky.

    Parameters
    ----------
    times_jd : (T,) Julian days of the samples (window starts if ``times_end_jd``).
    freqs : (F,) MHz.
    lmax : band limit of the simulation.
    pointings : (P, 3) ZYZ Euler angles in radians (``pointing_euler``).
    lat, lon, alt : site, degrees and metres.
    world : "earth" or "moon".
    Tgnd : ground temperature in K, scalar or (F,).
    times_end_jd : (T,) or None, end of each integration window.
    horizon_theta : horizon colatitude in radians, scalar or callable of compass azimuth
        (as ``croissant.Beam(horizon_theta=...)``), fixed to the ground.
    engine : croissant SHT engine.
    """

    times_jd: np.ndarray
    freqs: np.ndarray
    lmax: int
    pointings: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    lat: float = DEFAULT_SITE["lat"]
    lon: float = DEFAULT_SITE["lon"]
    alt: float = DEFAULT_SITE["alt"]
    world: str = "earth"
    Tgnd: object = 300.0
    times_end_jd: np.ndarray | None = None
    horizon_theta: object = np.pi / 2
    engine: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "times_jd", np.atleast_1d(np.asarray(self.times_jd, float)))
        object.__setattr__(self, "freqs", np.atleast_1d(np.asarray(self.freqs, float)))
        p = np.asarray(self.pointings, float).reshape(-1, 3)
        object.__setattr__(self, "pointings", p)
        if self.times_end_jd is not None:
            object.__setattr__(self, "times_end_jd", np.asarray(self.times_end_jd, float))

    @classmethod
    def default(cls, freqs, lmax, n_times=500, n_pointings=20, **kw):
        """Plan §8.2 Q5: one sidereal day from 2025-12-01 09:00 UTC at the notebooks' site,
        ``n_pointings`` tilts 0-90 deg."""
        times, ends = uniform_times(n_times, integrate=kw.pop("integrate", False))
        return cls(times_jd=times, times_end_jd=ends, freqs=freqs, lmax=lmax,
                   pointings=default_pointings(n_pointings), **kw)

    @property
    def n_times(self):
        return len(self.times_jd)

    @property
    def n_freqs(self):
        return len(self.freqs)

    @property
    def n_pointings(self):
        return len(self.pointings)

    @property
    def Tgnd_f(self):
        """Ground temperature per frequency, (F,)."""
        return jnp.broadcast_to(jnp.asarray(self.Tgnd, float), (self.n_freqs,))

    @cached_property
    def et_ref(self):
        """Epoch of the simulation frame (SPICE ET), as ``croissant.simulation_epoch``."""
        return croissant.simulation_epoch(self.times_jd)

    @cached_property
    def frame(self):
        """A croissant ``Simulator`` with an empty beam and sky. Used for its frame
        rotation (``topo_to_sim``) and phases, which do not depend on beam or sky."""
        L = self.lmax + 1
        grid = jnp.zeros((self.n_freqs, L + 1, 2 * L))
        beam = croissant.Beam(grid, self.freqs, sampling="mwss", engine="s2fft")
        sky = croissant.Sky(grid, self.freqs, sampling="mwss", engine="s2fft")
        return croissant.Simulator(
            beam, sky, jnp.asarray(self.times_jd), self.freqs, self.lon, self.lat,
            alt=self.alt, lmax=self.lmax, world=self.world, Tgnd=self.Tgnd_f,
            times_end_jd=None if self.times_end_jd is None else jnp.asarray(self.times_end_jd),
        )

    @property
    def phases(self):
        """exp(-i m phi(t)), window-averaged if ``times_end_jd``; (T, 2L-1)."""
        return self.frame.phases

    @property
    def rotation_angle(self):
        """Sky rotation angle phi(t) of each sample start, radians, (T,)."""
        day = croissant.constants.sidereal_day[self.world]
        return 2 * np.pi * (self.times_jd - self.times_jd[0]) * 86400.0 / day

    def is_uniform_full_day(self, atol_s=1e-3):
        """True if the samples evenly tile exactly one sidereal day and there are more than
        2 lmax of them: the condition for the per-m decoupling (docs/math.md, section 3).
        Julian days carry ~40 us of rounding, hence the tolerance in seconds."""
        day = croissant.constants.sidereal_day[self.world]
        dt = np.diff(self.times_jd) * 86400.0
        n = self.n_times
        return bool(np.allclose(dt, day / n, rtol=0.0, atol=atol_s) and n > 2 * self.lmax)


@dataclass(frozen=True, eq=False)
class BeamResponse:
    """The beam as the data see it, for every pointing and frequency.

    alm   : (P, F, L, 2L-1) horizon-weighted beam alm in the simulation frame (not normalized)
    norm  : (P, F) integral of the beam over the whole sphere
    above : (P, F) integral of the horizon-weighted beam

    All three are linear in the input beam pattern. Ground fraction: 1 - above / norm.
    """

    alm: jax.Array
    norm: jax.Array
    above: jax.Array

    @property
    def fgnd(self):
        return 1.0 - self.above / self.norm

    def __add__(self, other):
        return BeamResponse(self.alm + other.alm, self.norm + other.norm, self.above + other.above)

    def scale(self, c):
        return BeamResponse(c * self.alm, c * self.norm, c * self.above)


def beam_alm_topo(beam_maps, obs, sampling="healpix"):
    """Full-sphere alm of the beam in the beam frame, (F, L, 2L-1).

    ``beam_maps`` is (F, Npix) for HEALPix or (F, Ntheta, Nphi) on an s2fft grid, and must
    cover the whole sphere (below the horizon too, plan §8.2 Q2).
    """
    beam_maps = jnp.asarray(beam_maps, float)
    spatial = beam_maps.shape[1:]
    kw = dict(lmax=obs.lmax) if sampling == "healpix" else {}
    beam = croissant.Beam(beam_maps, obs.freqs, sampling=sampling, horizon=jnp.ones(spatial),
                          engine=obs.engine, **kw)
    return beam.compute_alm()


def pointed_beam(alm_topo, euler, obs):
    """The beam turned to one pointing, on an MWSS grid, as a horizon-weighted croissant Beam."""
    grid = croissant.rotations.rotate_to_grid(alm_topo, jnp.asarray(euler), sampling="mwss")
    return croissant.Beam(grid, obs.freqs, sampling="mwss", horizon_frame="topocentric",
                          horizon_theta=obs.horizon_theta, engine=obs.engine)


def beam_response(beam_maps, obs, sampling="healpix"):
    """``BeamResponse`` of a beam pattern (in the beam frame) for every pointing of ``obs``."""
    alm_topo = beam_alm_topo(beam_maps, obs, sampling)
    return beam_response_from_alm(alm_topo, obs)


def beam_response_from_alm(alm_topo, obs):
    """As ``beam_response``, from full-sphere beam-frame alm (F, L, 2L-1)."""
    alms, norms, aboves = [], [], []
    frame = obs.frame
    for euler in obs.pointings:
        beam = pointed_beam(alm_topo, euler, obs)
        norm = beam.compute_norm()
        alms.append(frame.topo_to_sim(beam.compute_alm()))
        norms.append(norm)
        aboves.append(norm * (1.0 - beam.compute_fgnd()))
    return BeamResponse(jnp.stack(alms), jnp.stack(norms), jnp.stack(aboves))


def load_beam(path, freqs_mhz=None):
    """Read a beam file; returns ``(maps, freqs_mhz, sampling)``.

    The file is an ``.npz`` with ``bm`` (F, Npix) HEALPix RING or (F, Ntheta, Nphi) on an
    s2fft grid, and ``freqs`` in Hz or MHz (values above 1e4 are taken as Hz). The grid is
    ``sampling`` if stored, else MWSS for 2-D grids (EIGSEP's ``bowtie_v002`` export: MWSS at
    lmax 128, directivity normalized to 4 pi over the full sphere). ``freqs_mhz`` selects the
    nearest stored channels. The pattern must cover the full sphere (plan §8.2 Q2).
    """
    with np.load(path, allow_pickle=False) as f:
        bm = np.asarray(f["bm"], float)
        freqs = np.asarray(f["freqs"], float)
        sampling = str(f["sampling"]) if "sampling" in f.files else (
            "healpix" if bm.ndim == 2 else "mwss")
    if np.max(freqs) > 1e4:
        freqs = freqs / 1e6
    if freqs_mhz is not None:
        idx = np.array([np.argmin(np.abs(freqs - nu)) for nu in np.atleast_1d(freqs_mhz)])
        if np.any(np.abs(freqs[idx] - np.atleast_1d(freqs_mhz)) > 1e-6):
            raise ValueError(f"requested frequencies not in the file; it has {freqs.min()}-"
                             f"{freqs.max()} MHz")
        bm, freqs = bm[idx], freqs[idx]
    return bm, freqs, sampling
