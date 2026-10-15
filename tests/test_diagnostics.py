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
