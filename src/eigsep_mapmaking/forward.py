"""The linear drift-scan measurement model and its m-space (Fourier-in-time) form.

Data are y[f, t, p] (frequency, time, pointing):

    y[f,t,p] = sum_lm conj(s[f,l,m]) e^{-i m phi(t)} b[p,f,l,m] / N[p,f]  +  g[p,f] + noise

with s the sky alm and b the horizon-weighted beam alm, both in the simulation frame,
N the full-sphere beam integral, and g = (N - N_above) / N * Tgnd the ground term
(docs/math.md, section 2). The sky enters through the packed real vector x[f, :] (``alm``).

For samples that evenly tile one sidereal day (T > 2 lmax), the projection
Y[f,m,p] = (1/T) sum_t y[f,t,p] e^{+i m phi_t}, m = 0..lmax, decouples the problem by m:

    Y[f,m,p] = w_m sum_{l>=m} conj(s[f,l,m]) b[p,f,l,m] / N[p,f]  (+ g[p,f] at m = 0)

where w_m = 1 for snapshots and e^{-i m dphi/2} sinc(m dphi / 2) for integration windows of
rotation dphi (docs/math.md, section 3).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import alm as alm_mod


class ForwardModel(eqx.Module):
    """y = A x + g for packed skies x (F, L**2) and data y (F, T, P).

    ``apply`` is A (linear, no ground term), ``adjoint`` is A^T, ``ground`` is g broadcast to
    the data shape and ``predict`` is A x + g.
    """

    beam: jax.Array  # (P, F, L, 2L-1) horizon-weighted beam alm / norm
    offset: jax.Array  # (P, F) ground term g
    phases: jax.Array  # (T, 2L-1) exp(-i m phi(t)), window-averaged if integrating
    phi: jax.Array  # (T,) rotation angle at each sample start
    lmax: int = eqx.field(static=True)

    @classmethod
    def from_response(cls, response, obs):
        """From an ``instrument.BeamResponse`` and the ``Observation`` it was built for."""
        norm = response.norm
        beam = response.alm / norm[..., None, None]
        offset = response.fgnd * obs.Tgnd_f[None, :]
        return cls(beam, offset, jnp.asarray(obs.phases), jnp.asarray(obs.rotation_angle),
                   obs.lmax)

    @property
    def shape_data(self):
        return (self.beam.shape[1], self.phases.shape[0], self.beam.shape[0])

    @property
    def shape_sky(self):
        return (self.beam.shape[1], alm_mod.n_real(self.lmax))

    def apply(self, x):
        s = alm_mod.unpack(x, self.lmax)  # (F, L, M)
        c = jnp.einsum("flm,pflm->fpm", jnp.conj(s), self.beam)  # sum over l first
        return jnp.einsum("tm,fpm->ftp", self.phases, c).real

    def adjoint(self, y):
        return jax.linear_transpose(self.apply, jnp.zeros(self.shape_sky))(y)[0]

    def ground(self):
        F, T, P = self.shape_data
        return jnp.broadcast_to(self.offset.T[:, None, :], (F, T, P))

    def predict(self, x):
        return self.apply(x) + self.ground()

    # ------------------------------------------------------------------ m-space
    def window_factor(self):
        """w_m for m = 0..lmax: the mean over samples of phases[t, m] e^{+i m phi_t}.
        Exact (and constant in t) for evenly spaced samples with equal windows."""
        m = jnp.arange(self.lmax + 1)
        ph = self.phases[:, self.lmax :]
        return jnp.mean(ph * jnp.exp(1j * m[None] * self.phi[:, None]), axis=0)

    def to_mspace(self, y):
        """Y[f, m, p] = (1/T) sum_t y[f,t,p] e^{+i m phi_t}, m = 0..lmax (complex)."""
        m = jnp.arange(self.lmax + 1)
        e = jnp.exp(1j * m[None, :] * self.phi[:, None])  # (T, M+)
        return jnp.einsum("ftp,tm->fmp", y, e) / y.shape[1]

    def mspace_response(self):
        """R[f, m, p, l] = w_m b[p,f,l,m] / N[p,f], zero for l < m.

        Y[f,m,:] = R[f,m] @ conj(s[f,:,m]) (+ the ground term at m = 0)."""
        L = self.lmax + 1
        b = self.beam[..., self.lmax :]  # (P, F, L, M+) m >= 0
        R = jnp.transpose(b, (1, 3, 0, 2)) * self.window_factor()[None, :, None, None]
        ell, m = np.arange(L)[:, None], np.arange(L)[None, :]
        return R * jnp.asarray((ell >= m).T, float)[None, :, None, :]

    def mspace_ground(self):
        """The ground term in m-space: g at m = 0, zero elsewhere, (F, L, P)."""
        F, _, P = self.shape_data
        return jnp.zeros((F, self.lmax + 1, P), complex).at[:, 0, :].set(self.offset.T)


def mspace_alm(s, lmax=None):
    """conj(s[f, l, m]) for m >= 0 as (F, M+, L): the sky vector each m-block acts on."""
    s = jnp.asarray(s)
    lmax = alm_mod.lmax_from_shape(s.shape) if lmax is None else lmax
    return jnp.swapaxes(jnp.conj(s[..., lmax:]), -1, -2)
