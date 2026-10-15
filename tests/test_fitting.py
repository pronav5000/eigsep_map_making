import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eigsep_mapmaking import alm, diagnostics, fitting, mapmaking, perturbations as pt, simulate
from eigsep_mapmaking.sky import SkyPrior
from conftest import NSIDE


@pytest.fixture(scope="module")
def setup(obs, sky_maps):
    band = simulate.synthetic_beam_maps(NSIDE, np.arange(8) * 5.0 + 50.0)
    B0 = band[:2]
    basis = pt.svd_basis(band, k=2, select=[0, 1])
    lin = pt.LinearBeamModel.build(B0, basis, obs)
    beta_true = np.array([0.08, -0.05])
    data = simulate.simulate(pt.inject_error(B0, basis, beta_true), sky_maps, obs,
                             noise_scale=30.0, seed=3)
    prior = SkyPrior.oracle(alm.unpack(data.x_true), floor=1e-3)
    return dict(B0=B0, basis=basis, lin=lin, beta_true=beta_true, data=data, prior=prior)


def test_mspace_marginal_equals_dense(setup):
    s = setup
    d = s["data"]
    f_m = lambda b: fitting.loglike_marginal_mspace(b, s["lin"], s["prior"], d.sigma, d.y)
    f_d = lambda b: fitting.loglike_marginal_dense(b, s["lin"], s["prior"], d.sigma, d.y)
    b0, b1, b2 = jnp.zeros(2), jnp.asarray(s["beta_true"]), jnp.array([-0.1, 0.1])
    for b in (b1, b2):
        dm, dd = f_m(b) - f_m(b0), f_d(b) - f_d(b0)
        np.testing.assert_allclose(dm, dd, rtol=1e-6, atol=1e-6 * abs(float(dd)) + 1e-6)


def test_fisher_is_expected_negative_hessian(setup):
    s = setup
    lin, prior, sigma = s["lin"], s["prior"], s["data"].sigma
    b = jnp.asarray(s["beta_true"])
    fm = lin.forward_model(b)
    C, mu0 = fitting._mspace_terms(fm, prior, sigma)
    L = jnp.linalg.cholesky(C)

    def draw(key):
        k1, k2 = jax.random.split(key)
        z = (jax.random.normal(k1, C.shape[:-1]) + 1j * jax.random.normal(k2, C.shape[:-1]))
        Y = jnp.einsum("fmpq,fmq->fmp", L, z / jnp.sqrt(2))
        Y0 = mu0 + jnp.einsum("fpq,fq->fp", L[:, 0].real, jax.random.normal(k1, mu0.shape))
        return Y.at[:, 0].set(Y0.astype(Y.dtype))

    H = jax.jit(jax.vmap(lambda k: jax.hessian(
        lambda bb: fitting.loglike_marginal_mspace(bb, lin, prior, sigma, Y=draw(k)))(b)))
    Hs = H(jax.random.split(jax.random.PRNGKey(0), 400))
    F = fitting.fisher(b, lin, prior, sigma)
    np.testing.assert_allclose(-Hs.mean(0), F, rtol=0.15, atol=0.05 * float(jnp.max(F)))
    w, v = fitting.whiten(F)
    assert w[0] >= w[-1] > 0
    np.testing.assert_allclose(v @ jnp.diag(w) @ v.T, F, rtol=1e-8)


def test_nuts_recovers_injected_beam_error(setup):
    s = setup
    d = s["data"]
    samples = fitting.run_nuts(s["lin"], s["prior"], d.sigma, d.y, s["basis"].sigma,
                               num_warmup=200, num_samples=300, key=1)
    mean, sd = samples.mean(0), samples.std(0)
    z = (mean - s["beta_true"]) / sd
    assert np.all(np.abs(z) < 4), (mean, sd)
    assert np.all(sd < 0.5 * np.abs(s["beta_true"]))  # informative
    # recovered beam is closer to the truth than the starting beam (normalized comparison)
    true = pt.inject_error(s["B0"], s["basis"], s["beta_true"])
    fit = pt.inject_error(s["B0"], s["basis"], np.asarray(mean))
    assert np.all(diagnostics.normalized_beam_error(fit, true) <
                  diagnostics.normalized_beam_error(s["B0"], true))


def test_rotation_about_pole_is_exactly_degenerate(setup):
    """Turning every beam about the celestial pole (b_lm -> b_lm e^{-i m a}) is absorbed by
    the isotropic sky prior: the marginal likelihood does not change (docs/math.md §5)."""
    s = setup
    d = s["data"]
    fm = s["lin"].forward_model(jnp.zeros(2))
    m = jnp.arange(-fm.lmax, fm.lmax + 1)
    rot = type(fm)(fm.beam * jnp.exp(-1j * m * 0.7), fm.offset, fm.phases, fm.phi, fm.lmax)
    Y = fm.to_mspace(d.y)
    ll = lambda f: fitting.loglike_mspace_Y(*fitting._mspace_terms(f, s["prior"], d.sigma), Y)
    np.testing.assert_allclose(ll(rot), ll(fm), rtol=1e-12)


def test_find_map_escapes_secondary_mode(setup):
    """With 3 pointings the posterior has a secondary mode near (-0.045, -0.037) that BFGS
    from beta = 0 falls into; the multi-start MAP finds the global one."""
    s = setup
    d = s["data"]
    b1, lp1 = fitting.find_map(s["lin"], s["prior"], d.sigma, d.y, s["basis"].sigma, n_starts=1)
    b, lp = fitting.find_map(s["lin"], s["prior"], d.sigma, d.y, s["basis"].sigma)
    assert lp >= lp1
    np.testing.assert_allclose(b, s["beta_true"], atol=0.03)
