"""Diagnostics for maps, beams and the forward model."""

import jax
import jax.numpy as jnp
import numpy as np

from . import alm as alm_mod
from .perturbations import _inner, quad_weights


def normalized_beam_error(beam_est, beam_true, sampling="healpix"):
    """|| B_est / int B_est - B_true / int B_true || / || B_true / int B_true || per frequency.

    Only the normalized beam is observable (the data are invariant to B -> c B); comparing
    unnormalized beams, as the notebooks did, counts the invisible scale (review 2.2.4)."""
    w = quad_weights(np.shape(beam_true)[1:], sampling)
    ne = beam_est / _inner(beam_est, 1.0, w)[(...,) + (None,) * w.ndim]
    nt = beam_true / _inner(beam_true, 1.0, w)[(...,) + (None,) * w.ndim]
    return np.sqrt(_inner(ne - nt, ne - nt, w) / _inner(nt, nt, w))


def scale_fraction(delta, beam_maps, sampling="healpix"):
    """Fraction of the power of a beam change ``delta`` (F, *grid) along the beam itself,
    i.e. invisible to the data, per frequency."""
    w = quad_weights(np.shape(beam_maps)[1:], sampling)
    along = _inner(delta, beam_maps, w) ** 2 / _inner(beam_maps, beam_maps, w)
    return along / _inner(delta, delta, w)


def mode_overlap(basis, delta, sampling=None):
    """Fraction of the power of ``delta`` (F, *grid) inside the span of a ``BeamBasis``."""
    sampling = basis.sampling if sampling is None else sampling
    w = quad_weights(np.shape(delta)[1:], sampling)
    K = basis.k
    M = (basis.modes * np.sqrt(w)).reshape(K, -1)
    d = (np.asarray(delta) * np.sqrt(w)).ravel()
    coef, *_ = np.linalg.lstsq(M.T, d, rcond=None)
    return float(np.sum((M.T @ coef) ** 2) / np.sum(d**2))


def map_error_per_ell(x_est, x_true):
    """sqrt(C_l(x_est - x_true) / C_l(x_true)), (F, L)."""
    return jnp.sqrt(alm_mod.cl_from_packed(x_est - x_true) / alm_mod.cl_from_packed(x_true))


def relative_error(a, b):
    """||a - b|| / ||b||. (The notebooks printed the square root of this.)"""
    return float(jnp.linalg.norm(jnp.ravel(a - b)) / jnp.linalg.norm(jnp.ravel(b)))


def adjoint_test(fm, key=0):
    """|<A x, y> - <x, A^T y>| / |<A x, y>| for random x, y."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    x = jax.random.normal(k1, fm.shape_sky)
    y = jax.random.normal(k2, fm.shape_data)
    a, b = jnp.vdot(fm.apply(x), y), jnp.vdot(x, fm.adjoint(y))
    return float(jnp.abs(a - b) / jnp.abs(a))


def rank_bound(n_pointings, lmax):
    """Upper bound on the number of sky modes a full-day drift scan constrains per
    frequency: sum over m of min(P, lmax + 1 - |m|) (docs/math.md, section 3)."""
    return int(sum(min(n_pointings, lmax + 1 - abs(m)) for m in range(-lmax, lmax + 1)))


def mspace_rank(fm, rtol=1e-10):
    """Numerical rank of the m-space response, counted like ``rank_bound`` (m > 0 blocks
    count twice: real and imaginary parts), per frequency, (F,)."""
    R = np.asarray(fm.mspace_response())  # (F, M+, P, L)
    out = []
    for f in range(R.shape[0]):
        tot = 0
        for m in range(R.shape[1]):
            s = np.linalg.svd(R[f, m], compute_uv=False)
            r = int(np.sum(s > rtol * s.max())) if s.max() > 0 else 0
            tot += r if m == 0 else 2 * r
        out.append(tot)
    return np.array(out)
