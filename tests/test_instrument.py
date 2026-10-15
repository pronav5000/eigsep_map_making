import croissant
import jax.numpy as jnp
import numpy as np
import pytest

from eigsep_mapmaking import alm, instrument, sky
from conftest import FREQS, LMAX, NSIDE


@pytest.mark.parametrize("tilt,az", [(30, 90), (45, 0), (60, 200)])
def test_pointing_convention(tilt, az):
    """Tilt moves the boresight to altitude 90 - tilt at compass azimuth az."""
    R = instrument.pointing_matrix(tilt, az)
    b = R @ np.array([0, 0, 1.0])
    phi = np.arctan2(b[1], b[0])
    np.testing.assert_allclose(np.degrees(np.arccos(b[2])), tilt, atol=1e-10)
    np.testing.assert_allclose((90 - np.degrees(phi)) % 360, az % 360, atol=1e-10)


def test_rotate_to_grid_moves_peak_as_pointing_says():
    lmax = 24
    obs = instrument.Observation(times_jd=[2460000.0], freqs=[50.0], lmax=lmax)
    nside = 16
    theta, _ = __import__("healpy").pix2ang(nside, np.arange(12 * nside**2))
    a = instrument.beam_alm_topo(np.exp(-(theta / 0.3) ** 2)[None], obs)
    g = croissant.rotations.rotate_to_grid(a, jnp.asarray(instrument.pointing_euler(50, 90)),
                                           sampling="mwss")[0]
    th = croissant.utils.generate_theta(lmax, "mwss")
    ph = croissant.utils.generate_phi(lmax, "mwss")
    i, j = np.unravel_index(np.argmax(g), g.shape)
    dirn = lambda t, p: np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    want = instrument.pointing_matrix(50, 90) @ [0, 0, 1.0]
    assert np.degrees(np.arccos(dirn(th[i], ph[j]) @ want)) < 8  # within a grid cell


def test_norm_is_rotation_invariant_and_fgnd(response, beam_maps):
    npix = beam_maps.shape[-1]
    integral = beam_maps.sum(-1) * 4 * np.pi / npix
    np.testing.assert_allclose(response.norm, np.broadcast_to(response.norm[0], response.norm.shape),
                               rtol=1e-10)
    np.testing.assert_allclose(response.norm[0], integral, rtol=2e-2)  # HEALPix quadrature
    # zenith pointing: ground fraction is the below-horizon share of the beam map
    theta, _ = __import__("healpy").pix2ang(NSIDE, np.arange(npix))
    w = np.where(np.isclose(theta, np.pi / 2), 0.5, theta > np.pi / 2)
    np.testing.assert_allclose(response.fgnd[0], (beam_maps * w).sum(-1) / beam_maps.sum(-1),
                               atol=0.03)
    assert np.all(response.fgnd[2] > response.fgnd[0] + 0.1)  # 70 deg tilt
    assert np.all((response.fgnd > -0.02) & (response.fgnd < 0.7))


def test_response_is_linear(obs, beam_maps):
    b2 = beam_maps[::-1] * 0.7
    r = instrument.beam_response(2.0 * beam_maps - 3.0 * b2, obs)
    r1 = instrument.beam_response(beam_maps, obs)
    r2 = instrument.beam_response(b2, obs)
    want = r1.scale(2.0) + r2.scale(-3.0)
    for x, y in [(r.alm, want.alm), (r.norm, want.norm), (r.above, want.above)]:
        np.testing.assert_allclose(x, y, atol=1e-10 * float(jnp.max(jnp.abs(y))))


def test_symmetric_beam_tilted_to_horizon_has_half_on_ground():
    obs = instrument.Observation(times_jd=[2460000.0], freqs=[50.0], lmax=LMAX,
                                 pointings=instrument.pointing_euler(90, 30)[None])
    theta, _ = __import__("healpy").pix2ang(NSIDE, np.arange(12 * NSIDE**2))
    r = instrument.beam_response(np.exp(-(theta / 0.4) ** 2)[None], obs)
    np.testing.assert_allclose(r.fgnd, 0.5, atol=5e-3)


@pytest.mark.parametrize("integrate", [False, True])
def test_matches_croissant_simulator(obs, beam_maps, sky_maps, response, integrate):
    """V = [conv(alm_h, sky) + (norm - above) Tgnd] / norm equals croissant's Simulator.sim
    for every pointing (the Simulator is given the same pointed, horizon-weighted Beam)."""
    times, ends = instrument.uniform_times(obs.n_times, integrate=integrate)
    o = instrument.Observation(times_jd=times, times_end_jd=ends, freqs=FREQS, lmax=LMAX,
                               pointings=obs.pointings, Tgnd=[280.0, 250.0])
    sky_alm = sky.sky_alm_sim(sky_maps, o)
    csky = croissant.Sky(jnp.asarray(sky_maps), FREQS, sampling="healpix", lmax=LMAX)
    a_topo = instrument.beam_alm_topo(beam_maps, o)
    for p, euler in enumerate(o.pointings):
        beam = instrument.pointed_beam(a_topo, euler, o)
        simr = croissant.Simulator(beam, csky, jnp.asarray(times), FREQS, o.lon, o.lat,
                                   lmax=LMAX, world="earth", Tgnd=o.Tgnd_f,
                                   times_end_jd=None if ends is None else jnp.asarray(ends))
        want = simr.sim()  # (T, F)
        mine = croissant.simulator.convolve(response.alm[p], sky_alm, o.phases).real
        mine = (mine + (response.norm[p] - response.above[p]) * o.Tgnd_f) / response.norm[p]
        np.testing.assert_allclose(mine, want, rtol=1e-10)
        np.testing.assert_allclose(sky_alm, simr.precompute_sky_alm(), atol=1e-9)


def test_phases_and_uniform_day(obs):
    assert obs.is_uniform_full_day()
    m = np.arange(-LMAX, LMAX + 1)
    np.testing.assert_allclose(obs.phases, np.exp(-1j * m[None] * obs.rotation_angle[:, None]),
                               atol=1e-9)
    short = instrument.Observation(times_jd=obs.times_jd[:-1], freqs=FREQS, lmax=LMAX)
    assert not short.is_uniform_full_day()


def test_sky_prior():
    p = sky.SkyPrior.power_law(LMAX, FREQS, amp=10.0, alpha=2.0, beta=2.5, nu0=50.0,
                               monopole_sigma=100.0)
    assert p.S.shape == (2, alm.n_real(LMAX))
    np.testing.assert_allclose(p.cl[0, 2], 10.0 / 4)
    np.testing.assert_allclose(p.cl[1, 2] / p.cl[0, 2], (55 / 50) ** -5)
    np.testing.assert_allclose(p.cl[0, 0], 4 * np.pi * 100.0**2)
    # every packed entry of a given l carries C_l
    ell = alm.ell_of_packed(LMAX)
    np.testing.assert_allclose(p.S[0], p.cl[0][ell])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, alm.n_real(LMAX)))
    o = sky.SkyPrior.oracle(alm.unpack(x), floor=0.5)
    cl = alm.cl_from_packed(x)
    np.testing.assert_allclose(o.cl, np.maximum(cl, 0.5 * cl.max(-1, keepdims=True)))


def test_sky_roundtrip_file(tmp_path, sky_maps):
    sky.save_sky(tmp_path / "s.npz", sky_maps, FREQS)
    m, f, c = sky.load_sky(tmp_path / "s.npz")
    np.testing.assert_array_equal(m, sky_maps)
    assert c == "galactic"


@pytest.mark.network
def test_gsm_maps():
    pytest.importorskip("pygdsm")
    m = sky.gsm_maps([50.0, 60.0], NSIDE)
    assert m.shape == (2, 12 * NSIDE**2) and np.all(m > 0) and m[0].mean() > m[1].mean()
