"""Beam inference: likelihoods in the beam parameters beta, NUTS, Fisher information.

The recommended likelihood marginalizes the sky analytically (docs/math.md, section 6):

    p(y | beta) = N(y; g(beta), A(beta) S A(beta)^T + N),

which for a uniform full sidereal day and stationary noise splits by m into P x P Gaussians:
m = 0 real, Y_0 ~ N(g(beta), C_0); m > 0 complex circular, Y_m ~ CN(0, C_m), with
C_m = R_m diag(C_l) R_m^H + diag(sigma^2) / T. It needs no sky estimate. The fixed-sky
likelihood is kept for comparison: conditioning on a point estimate of the sky (Wiener map,
true sky, or the alternating ICM loop of the notebooks) biases beta (review 2.3.1).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .mapmaking import _check_mspace


def _mspace_terms(fm, prior, sigma):
    sig = _check_mspace(fm, sigma)  # (F, P)
    T = fm.shape_data[1]
    R = fm.mspace_response()
    R = R.at[:, 0].set(R[:, 0].real.astype(R.dtype))
    C = jnp.einsum("fmpl,fl,fmql->fmpq", R, prior.cl, jnp.conj(R))
    C = C + jax.vmap(jnp.diag)(sig**2 / T)[:, None].astype(C.dtype)
    mu0 = fm.offset.T  # (F, P): ground term, m = 0 only
    return C, mu0


def loglike_mspace_Y(C, mu0, Y):
    """log N for m-space data Y (F, M+, P) given covariances C (F, M+, P, P) and m = 0 mean."""
    P = Y.shape[-1]
    # m = 0: real Gaussian
    C0 = C[:, 0].real
    r0 = Y[:, 0].real - mu0
    L0 = jnp.linalg.cholesky(C0)
    z0 = jax.scipy.linalg.solve_triangular(L0, r0[..., None], lower=True)[..., 0]
    ll0 = -0.5 * jnp.sum(z0**2) - jnp.sum(jnp.log(jnp.diagonal(L0, axis1=-2, axis2=-1))) \
        - 0.5 * P * np.log(2 * np.pi) * C0.shape[0]
    # m > 0: circular complex Gaussian, -log det(pi C) - Y^H C^-1 Y
    Cm = C[:, 1:]
    Lm = jnp.linalg.cholesky(Cm)
    zm = jax.scipy.linalg.solve_triangular(Lm, Y[:, 1:, :, None], lower=True)[..., 0]
    llm = -jnp.sum(jnp.abs(zm) ** 2) \
        - 2 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(Lm, axis1=-2, axis2=-1)))) \
        - P * np.log(np.pi) * Cm.shape[0] * Cm.shape[1]
    return ll0 + llm


def loglike_marginal_mspace(beta, lin, prior, sigma, y=None, Y=None):
    """Sky-marginalized log-likelihood of beta, exact by m. Pass time-domain ``y`` (F, T, P)
    or its m-space projection ``Y`` (``ForwardModel.to_mspace``, which does not depend on
    beta). Differs from the time-domain marginal likelihood by a beta-independent constant."""
    fm = lin.forward_model(beta)
    if Y is None:
        Y = fm.to_mspace(y)
    C, mu0 = _mspace_terms(fm, prior, sigma)
    return loglike_mspace_Y(C, mu0, Y)


def loglike_marginal_dense(beta, lin, prior, sigma, y):
    """The same marginal likelihood in the time domain with dense matrices (small problems)."""
    from .mapmaking import forward_matrix

    fm = lin.forward_model(beta)
    A = forward_matrix(fm)
    N = jnp.broadcast_to(jnp.asarray(sigma), fm.shape_data).ravel() ** 2
    C = (A * prior.S.ravel()) @ A.T + jnp.diag(N)
    r = (y - fm.ground()).ravel()
    return jax.scipy.stats.multivariate_normal.logpdf(r, jnp.zeros_like(r), C)


def loglike_fixed_sky(beta, lin, x_sky, sigma, y):
    """Gaussian log-likelihood with the sky held fixed at ``x_sky`` (biased unless x_sky is
    the truth; kept for comparison with the notebooks)."""
    fm = lin.forward_model(beta)
    r = (y - fm.predict(x_sky)) / sigma
    return -0.5 * jnp.sum(r**2) - jnp.sum(jnp.log(jnp.broadcast_to(sigma, r.shape)))


def fisher(beta, lin, prior, sigma):
    """Fisher matrix of beta for the sky-marginalized m-space likelihood:
    m = 0: 1/2 tr(C^-1 dC_i C^-1 dC_j) + dmu_i^T C^-1 dmu_j;  m > 0: tr(C^-1 dC_i C^-1 dC_j)."""
    beta = jnp.asarray(beta, float)

    terms = lambda b: _mspace_terms(lin.forward_model(b), prior, sigma)
    C, _ = terms(beta)
    dC, dmu = jax.jacfwd(terms)(beta)  # derivatives on a trailing axis k
    Cinv = jnp.linalg.inv(C)
    X = jnp.einsum("fmpq,fmqrk->fmprk", Cinv, dC)  # C^-1 dC_k
    tr = jnp.einsum("fmpqi,fmqpj->fmij", X, X).real  # tr(C^-1 dC_i C^-1 dC_j)
    info = 0.5 * tr[:, 0].sum(0) + tr[:, 1:].sum((0, 1))
    return info + jnp.einsum("fpi,fpq,fqj->ij", dmu, Cinv[:, 0].real, dmu)


def whiten(F):
    """Eigen-decomposition of a Fisher matrix: ``(evals, evecs)``; gamma = evecs^T beta has
    independent Fisher errors 1/sqrt(evals). Small eigenvalues are poorly constrained
    directions (e.g. near-degeneracies with the sky)."""
    w, v = jnp.linalg.eigh(F)
    return w[::-1], v[:, ::-1]


def numpyro_model(lin, prior, sigma, Y, beta_sigma):
    """numpyro model: beta_k ~ N(0, beta_sigma_k), likelihood = sky-marginalized m-space."""
    import numpyro
    import numpyro.distributions as dist

    beta = numpyro.sample("beta", dist.Normal(0.0, jnp.asarray(beta_sigma)).to_event(1))
    numpyro.factor("loglike", loglike_marginal_mspace(beta, lin, prior, sigma, Y=Y))


def run_nuts(lin, prior, sigma, y, beta_sigma, key=0, num_warmup=300, num_samples=500,
             init=None, progress_bar=False):
    """Sample p(beta | y) with NUTS; returns the samples (num_samples, K).

    The chain starts at ``init`` (default beta = 0, the nominal beam). numpyro's default,
    uniform in (-2, 2), starts tens of prior sigmas out, where the likelihood can have
    secondary modes (the beam integral is a ratio of linear functions of beta); a chain
    started there was seen to stick in one.
    """
    from numpyro.infer import MCMC, NUTS, init_to_value

    Y = lin.forward_model(jnp.zeros(len(beta_sigma))).to_mspace(y)
    init = jnp.zeros(len(beta_sigma)) if init is None else jnp.asarray(init)
    kern = NUTS(numpyro_model, init_strategy=init_to_value(values={"beta": init}))
    mcmc = MCMC(kern, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(key), lin, prior, sigma, Y, jnp.asarray(beta_sigma))
    return mcmc.get_samples()["beta"]
