"""Beam-error models: bases of beam perturbations and the forward model linear in them.

A perturbed beam is B(beta) = B0 + sum_k beta_k M_k, with modes M_k (F, *grid) on the grid
of the beam pattern (beam frame). The data cannot see B_f -> c_f B_f: the 1/integral
normalization cancels it (docs/math.md, section 5). Both bases therefore remove, at every
frequency, the component of each mode along B0_f ("the scale direction"); the notebooks' SVD
basis was dominated by it (review 2.2.3).

Everything ``instrument.beam_response`` computes is linear in the pattern, so the responses of
B0 and of each mode are precomputed once; ``LinearBeamModel.forward_model(beta)`` is then a
cheap linear combination with no spherical harmonic transform inside a likelihood.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from . import alm as alm_mod
from . import instrument
from .forward import ForwardModel


def quad_weights(grid_shape, sampling, nside=None):
    """Quadrature weights broadcastable to a grid: integral f dOmega = sum w f."""
    if sampling == "healpix":
        npix = grid_shape[-1]
        return np.full(npix, 4 * np.pi / npix)
    L = grid_shape[0] - 1 if sampling == "mwss" else grid_shape[0]
    w = np.asarray(s2fft.utils.quadrature.quad_weights(L=L, sampling=sampling))
    return w[:, None] * np.ones(grid_shape)


def _inner(a, b, w):
    """<a, b> = sum over the grid of w a b, per leading index."""
    return np.sum(a * b * w, axis=tuple(range(-w.ndim, 0)))


def remove_scale(modes, beam_maps, w):
    """Remove from every mode, per frequency, its component along the beam."""
    bb = _inner(beam_maps, beam_maps, w)  # (F,)
    c = _inner(modes, beam_maps[None], w) / bb[None]  # (K, F)
    return modes - c[(...,) + (None,) * w.ndim] * beam_maps[None]


@dataclass(frozen=True, eq=False)
class BeamBasis:
    """Modes (K, F, *grid) in the beam frame and the prior std of their coefficients (K,).

    Modes are orthonormal in <a, b> = sum_f int a_f b_f dOmega and then scaled to the RMS of
    the beam over frequencies and the sphere, so beta_k = 0.05 is a 5 % (RMS) beam error.
    """

    modes: np.ndarray
    sigma: np.ndarray
    sampling: str
    names: tuple = ()

    @property
    def k(self):
        return self.modes.shape[0]

    def perturbation(self, beta):
        return jnp.tensordot(jnp.asarray(beta), jnp.asarray(self.modes), axes=1)


def _finish(raw, beam_maps, sampling, k, rel_sigma, names, remove):
    w = quad_weights(beam_maps.shape[1:], sampling)
    if remove:
        raw = remove_scale(raw, beam_maps, w)
    K = raw.shape[0]
    sw = np.sqrt(w)
    # orthonormalize in the weighted inner product; keep the k strongest directions
    _, s, vt = np.linalg.svd((raw * sw).reshape(K, -1), full_matrices=False)
    keep = min(k, int(np.sum(s > s[0] * 1e-8)))
    ortho = vt[:keep].reshape((keep,) + beam_maps.shape) / sw
    rms = np.sqrt(_inner(beam_maps, beam_maps, w).sum())  # ||B|| over freq and sphere
    modes = ortho * rms
    return BeamBasis(modes, np.full(keep, rel_sigma), sampling, tuple(names[:keep]))


def svd_basis(beam_maps, k, sampling="healpix", select=None, remove=True, rel_sigma=0.05):
    """Modes from the SVD of the beam over (frequency, direction), B = sum_j s_j u_j(f) v_j(n).

    The SVD is taken over all channels of ``beam_maps`` (pass the full band: with few channels
    it has few modes); ``select`` (indices) picks the channels the modes are evaluated at, the
    ones being fitted. Raw modes are u_j(f) v_j(n), j < k (+1 with ``remove``); the scale
    direction is removed at each selected channel and the result re-orthonormalized.

    The notebooks used the raw u_j v_j, whose leading mode is essentially the beam itself.
    """
    B = np.asarray(beam_maps, float)
    F = B.shape[0]
    sel = np.arange(F) if select is None else np.atleast_1d(select)
    u, s, vt = np.linalg.svd(B.reshape(F, -1), full_matrices=False)
    n = min(k + 1 if remove else k, len(s))
    raw = np.einsum("fj,jn->jfn", u[sel, :n], vt[:n]).reshape((n, len(sel)) + B.shape[1:])
    return _finish(raw, B[sel], sampling, k, rel_sigma, [f"svd{j}" for j in range(n)], remove)


def real_harmonics(lmax, grid_shape, sampling, nside=None):
    """Orthonormal real spherical harmonics on a grid, (L**2, *grid), in packed order."""
    n = alm_mod.n_real(lmax)
    if sampling == "healpix":
        nside = int(np.sqrt(grid_shape[-1] / 12))
        L = max(lmax + 1, 2 * nside)
    else:
        L = grid_shape[0] - 1 if sampling == "mwss" else grid_shape[0]
    a = alm_mod.unpack(np.eye(n), lmax)  # (n, lmax+1, 2 lmax+1)
    pad = jnp.zeros((n, L, 2 * L - 1), complex)
    pad = pad.at[:, : lmax + 1, L - 1 - lmax : L + lmax].set(a)
    inv = lambda x: s2fft.inverse(x, L=L, nside=nside, sampling=sampling, reality=True,
                                  method="jax").real
    return np.asarray(jax.vmap(inv)(pad)).reshape((n,) + tuple(grid_shape))


def lowl_basis(beam_maps, lmax_pert, n_freq_funcs=2, sampling="healpix", rel_sigma=0.05):
    """Generic multiplicative modes B0_f(n) Y_i(n) P_j(x_f): real harmonics Y_i with
    1 <= l <= lmax_pert (l = 0 is the scale) times Legendre polynomials P_j, j < n_freq_funcs,
    in frequency mapped to [-1, 1]; scale direction removed, re-orthonormalized."""
    B = np.asarray(beam_maps, float)
    F = B.shape[0]
    Y = real_harmonics(lmax_pert, B.shape[1:], sampling)[1:]  # drop l = 0
    ell = alm_mod.ell_of_packed(lmax_pert)[1:]
    m = alm_mod.m_of_packed(lmax_pert)[1:]
    x = np.linspace(-1, 1, F) if F > 1 else np.zeros(1)
    P = np.stack([np.polynomial.legendre.Legendre.basis(j)(x) for j in range(n_freq_funcs)])
    raw, names = [], []
    for j in range(min(n_freq_funcs, F)):
        pj = P[j].reshape((F,) + (1,) * (B.ndim - 1))
        for i in range(len(Y)):
            raw.append(B * Y[i][None] * pj)
            names.append(f"l{ell[i]}m{m[i]}P{j}")
    raw = np.array(raw)
    return _finish(raw, B, sampling, len(raw), rel_sigma, names, True)


def inject_error(beam_maps, basis, beta):
    """The perturbed beam B0 + sum_k beta_k M_k (for simulations)."""
    return np.asarray(beam_maps) + np.asarray(basis.perturbation(beta))


class LinearBeamModel(eqx.Module):
    """Precomputed instrument response of B0 and of each mode; ``forward_model(beta)`` is the
    ``ForwardModel`` of B0 + sum beta_k M_k, differentiable in beta."""

    base_alm: jax.Array  # (P, F, L, M)
    base_norm: jax.Array  # (P, F)
    base_above: jax.Array
    mode_alm: jax.Array  # (K, P, F, L, M)
    mode_norm: jax.Array  # (K, P, F)
    mode_above: jax.Array
    Tgnd: jax.Array  # (F,)
    phases: jax.Array
    phi: jax.Array
    lmax: int = eqx.field(static=True)

    @classmethod
    def build(cls, beam_maps, basis, obs, sampling=None):
        sampling = basis.sampling if sampling is None else sampling
        base = instrument.beam_response(beam_maps, obs, sampling=sampling)
        modes = [instrument.beam_response(mk, obs, sampling=sampling) for mk in basis.modes]
        stack = lambda f: jnp.stack([f(r) for r in modes])
        return cls(base.alm, base.norm, base.above, stack(lambda r: r.alm),
                   stack(lambda r: r.norm), stack(lambda r: r.above), obs.Tgnd_f,
                   jnp.asarray(obs.phases), jnp.asarray(obs.rotation_angle), obs.lmax)

    def response(self, beta):
        beta = jnp.asarray(beta)
        alm = self.base_alm + jnp.tensordot(beta, self.mode_alm, axes=1)
        norm = self.base_norm + jnp.tensordot(beta, self.mode_norm, axes=1)
        above = self.base_above + jnp.tensordot(beta, self.mode_above, axes=1)
        return instrument.BeamResponse(alm, norm, above)

    def forward_model(self, beta):
        r = self.response(beta)
        return ForwardModel(r.alm / r.norm[..., None, None], r.fgnd * self.Tgnd[None, :],
                            self.phases, self.phi, self.lmax)
