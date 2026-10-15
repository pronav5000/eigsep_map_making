import jax.numpy as jnp
import numpy as np

from eigsep_mapmaking import alm, diagnostics, perturbations as pt, simulate
from eigsep_mapmaking.forward import ForwardModel
from conftest import NSIDE


def test_normalized_beam_error_ignores_scale(beam_maps):
    e = diagnostics.normalized_beam_error(beam_maps * np.array([[2.0], [0.3]]), beam_maps)
    np.testing.assert_allclose(e, 0.0, atol=1e-12)
    assert np.all(diagnostics.normalized_beam_error(beam_maps ** 1.1, beam_maps) > 1e-3)
    np.testing.assert_allclose(diagnostics.scale_fraction(0.1 * beam_maps, beam_maps), 1.0)


def test_mode_overlap(beam_maps):
    basis = pt.lowl_basis(beam_maps, lmax_pert=1, n_freq_funcs=1)
    assert abs(diagnostics.mode_overlap(basis, basis.perturbation([0.1, 0.2, -0.1])) - 1) < 1e-10


def test_adjoint_and_rank(response, obs):
    fm = ForwardModel.from_response(response, obs)
    assert diagnostics.adjoint_test(fm) < 1e-12
    bound = diagnostics.rank_bound(obs.n_pointings, obs.lmax)
    assert bound == 45  # 3 pointings, lmax 8
    r = diagnostics.mspace_rank(fm, rtol=1e-8)
    assert np.all(r <= bound) and np.all(r >= bound - 4)


def test_map_error_per_ell():
    x = np.random.default_rng(0).normal(size=(1, alm.n_real(4)))
    np.testing.assert_allclose(diagnostics.map_error_per_ell(x, x), 0.0)
    np.testing.assert_allclose(diagnostics.map_error_per_ell(1.5 * x, x), 0.5)
    assert abs(diagnostics.relative_error(1.1 * x, x) - 0.1) < 1e-12


def test_notebook_amp_phase_perturbation():
    """The notebooks' 'amp 1.1, phase 5 deg' error (all s2fft coefficients times 1.1 e^{i5deg},
    reality=True synthesis): the amplitude is invisible after normalization, and the phase
    acts as cos(5 deg) on m = 0 plus a turn of the m > 0 part, identical at every frequency
    (docs/math.md §5)."""
    import s2fft
    nside, lmax = 4, 8
    B = simulate.synthetic_beam_maps(nside, [50.0])[0]
    fwd = lambda f: s2fft.forward(jnp.asarray(f), L=lmax + 1, nside=nside, sampling="healpix",
                                  reality=True, method="jax")
    inv = lambda a: np.asarray(s2fft.inverse(a, L=lmax + 1, nside=nside, sampling="healpix",
                                             reality=True, method="jax")).real
    a = fwd(B)
    c = 1.1 * np.exp(1j * np.radians(5.0))
    pert = inv(a * c)  # the notebook's perturbed map
    m = np.arange(-lmax, lmax + 1)
    want = np.where(m > 0, c, 0) * a  # m > 0: times c
    want = want + np.where(m == 0, (c * a).real, 0)  # m = 0: imaginary part dropped
    np.testing.assert_allclose(pert, inv(jnp.asarray(want)), atol=1e-10 * float(B.max()))
    # the m = 0 part is scaled by 1.1 cos 5deg, the m > 0 part by 1.1 and turned by 5 deg in phase
    np.testing.assert_allclose(want[:, lmax], 1.1 * np.cos(np.radians(5)) * a[:, lmax])
