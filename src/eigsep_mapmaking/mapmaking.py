"""Linear-Gaussian sky inference: the Wiener filter and constrained realizations.

Model: d = y - g = A x + n with x ~ N(0, S) (``sky.SkyPrior``) and n ~ N(0, N), N diagonal.
The posterior mean (Wiener filter) is

    x_hat = S A^T (A S A^T + N)^{-1} d          (dual / data-space form)

``wiener_cg`` solves the dual form with conjugate gradients and works for any sampling and
any diagonal N. ``wiener_mspace`` is exact and fast when the samples evenly tile one
sidereal day and the noise is stationary per (frequency, pointing): the problem then splits
into one P x P system per (frequency, m) (docs/math.md, sections 3-4).
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import alm as alm_mod


def _sigma_full(sigma, shape):
    return jnp.broadcast_to(jnp.asarray(sigma), shape)


def wiener_cg(fm, prior, sigma, y, tol=1e-10, maxiter=10_000, x0=None):
    """Wiener filter by CG on (A S A^T + N) alpha = y - g; returns ``(x_hat, n_iter_info)``.

    ``sigma`` broadcasts to the data shape (F, T, P). Uses N^{-1} as preconditioner.
    """
    d = jnp.asarray(y) - fm.ground()
    N = _sigma_full(sigma, d.shape) ** 2
    S = prior.S

    def matvec(a):
        return fm.apply(S * fm.adjoint(a)) + N * a

    alpha, info = jax.scipy.sparse.linalg.cg(matvec, d, x0=x0, tol=tol, maxiter=maxiter,
                                             M=lambda a: a / N)
    x_hat = S * fm.adjoint(alpha)
    resid = jnp.linalg.norm(matvec(alpha) - d) / jnp.linalg.norm(d)
    return x_hat, resid


def forward_matrix(fm):
    """Dense A, shape (F*T*P, F*L**2). For small problems and tests only."""
    n = int(np.prod(fm.shape_sky))
    f = lambda v: fm.apply(v.reshape(fm.shape_sky)).ravel()
    return jax.jacfwd(f)(jnp.zeros(n))


def wiener_dense(fm, prior, sigma, y):
    """Wiener filter and posterior covariance by dense linear algebra (small problems only).
    Returns ``(x_hat (F, L**2), cov (F*L**2, F*L**2))``."""
    A = forward_matrix(fm)
    d = (jnp.asarray(y) - fm.ground()).ravel()
    N = _sigma_full(sigma, fm.shape_data).ravel() ** 2
    S = prior.S.ravel()
    C = (A * S) @ A.T + jnp.diag(N)
    K = (S[:, None] * A.T) @ jnp.linalg.inv(C)
    return (K @ d).reshape(fm.shape_sky), jnp.diag(S) - K @ (A * S)


def _check_mspace(fm, sigma):
    sigma = jnp.asarray(sigma)
    F, T, P = fm.shape_data
    if sigma.ndim == 3 and sigma.shape[1] != 1:
        raise ValueError("m-space needs stationary noise: sigma of shape (F, 1, P)")
    if T <= 2 * fm.lmax:
        raise ValueError("m-space needs more than 2 lmax samples over the day")
    return jnp.broadcast_to(sigma, (F, 1, P))[:, 0, :]  # (F, P)


def mspace_blocks(fm, prior, sigma):
    """Per-(f, m) quantities: R (F, M+, P, L), cl (F, L), noise variance in m-space (F, P),
    and the data covariance C (F, M+, P, P) = R diag(C_l) R^H + diag(sigma^2 / T).

    m = 0 is real: its R and C are the real parts (the imaginary parts are rounding)."""
    sig = _check_mspace(fm, sigma)
    T = fm.shape_data[1]
    R = fm.mspace_response()
    R = R.at[:, 0].set(R[:, 0].real.astype(R.dtype))
    cl = prior.cl
    nvar = sig**2 / T
    C = jnp.einsum("fmpl,fl,fmql->fmpq", R, cl, jnp.conj(R))
    C = C + jax.vmap(jnp.diag)(nvar)[:, None].astype(C.dtype)
    C = C.at[:, 0].set(C[:, 0].real.astype(C.dtype))
    return R, cl, nvar, C


def wiener_mspace(fm, prior, sigma, y):
    """Exact Wiener filter by m (uniform full sidereal day, stationary noise). Returns x_hat
    (F, L**2). Matches ``wiener_cg``/``wiener_dense`` up to the time-sampling accuracy."""
    R, cl, _, C = mspace_blocks(fm, prior, sigma)
    Y = fm.to_mspace(jnp.asarray(y) - fm.ground())  # (F, M+, P)
    Y = Y.at[:, 0].set(Y[:, 0].real.astype(Y.dtype))
    u = jnp.linalg.solve(C, Y[..., None])[..., 0]  # C^{-1} Y
    conj_t = cl[:, None, :] * jnp.einsum("fmpl,fmp->fml", jnp.conj(R), u)  # E[conj(t_lm)|Y]
    return _pack_mplus(jnp.conj(conj_t), fm.lmax)


def _pack_mplus(t, lmax):
    """t[f, m, l] (m >= 0, zero for l < m) -> packed (F, L**2)."""
    F, L = t.shape[0], lmax + 1
    a = jnp.zeros((F, L, 2 * L - 1), complex).at[..., lmax:].set(jnp.swapaxes(t, 1, 2))
    a = a.at[..., :, lmax].set(a[..., :, lmax].real.astype(complex))
    return alm_mod.pack(a)


def posterior_var_mspace(fm, prior, sigma):
    """Posterior variance of every packed sky entry (F, L**2), exact by m."""
    R, cl, _, C = mspace_blocks(fm, prior, sigma)
    # Var(t_lm | Y) = C_l - C_l^2 R^H C^{-1} R  (complex; Re and Im each get half for m > 0)
    CinvR = jnp.linalg.solve(C, R)  # (F, M+, P, L)
    red = jnp.einsum("fmpl,fmpl->fml", jnp.conj(R), CinvR).real
    var = cl[:, None, :] - cl[:, None, :] ** 2 * red  # (F, M+, L)
    # For m > 0 the posterior of t_lm is circular, so sqrt2 Re t and sqrt2 Im t each have
    # variance Var(t_lm); for m = 0 t_l0 is real.
    ell, m = alm_mod.ell_of_packed(fm.lmax), alm_mod.m_of_packed(fm.lmax)
    return var[:, m, ell]


def constrained_realization(fm, prior, sigma, y, key, method="mspace", **kw):
    """A draw from the sky posterior p(x | y): x_hat(y) + x_s - x_hat(A x_s + n_s) with
    x_s ~ N(0, S), n_s ~ N(0, N). ``method`` is "mspace" or "cg"."""
    k1, k2 = jax.random.split(key)
    x_s = prior.sample(k1)
    n_s = _sigma_full(sigma, fm.shape_data) * jax.random.normal(k2, fm.shape_data)
    if method == "mspace":
        solve = lambda d: wiener_mspace(fm, prior, sigma, d)
    elif method == "cg":
        solve = lambda d: wiener_cg(fm, prior, sigma, d, **kw)[0]
    else:
        raise ValueError(method)
    g = fm.ground()
    return solve(y) + x_s - solve(fm.apply(x_s) + n_s + g)


def clip_positive_for_display(x, nside):
    """HEALPix maps of packed skies with negative pixels set to zero. For plots only: the
    clipped map is not a linear estimate and must not feed the beam fit (review 2.1.6)."""
    from .sky import sky_maps_from_alm

    return np.maximum(sky_maps_from_alm(alm_mod.unpack(x), nside), 0.0)
