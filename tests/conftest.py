import warnings

import numpy as np
import pytest

import eigsep_mapmaking  # noqa: F401  (enables jax x64 before any test runs)
from eigsep_mapmaking import instrument, simulate

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSIDE = 4
LMAX = 8
FREQS = np.array([50.0, 55.0])


@pytest.fixture(scope="session")
def obs():
    """A small observation: 2 freqs, 3 pointings, a uniform sidereal day (m-space exact)."""
    times, _ = instrument.uniform_times(2 * LMAX + 5)
    pointings = np.stack([instrument.pointing_euler(t, az) for t, az in
                          [(0, 0), (40, 0), (70, 120)]])
    return instrument.Observation(times_jd=times, freqs=FREQS, lmax=LMAX, pointings=pointings)


@pytest.fixture(scope="session")
def beam_maps():
    return simulate.synthetic_beam_maps(NSIDE, FREQS)


@pytest.fixture(scope="session")
def sky_maps():
    return simulate.synthetic_sky_maps(NSIDE, FREQS, lmax=LMAX)


@pytest.fixture(scope="session")
def response(beam_maps, obs):
    return instrument.beam_response(beam_maps, obs)
