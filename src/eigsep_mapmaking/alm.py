"""Spherical harmonic coefficients of real fields: s2fft layout and real packing.

s2fft (and croissant) store the alm of a band-limited field as a complex array of
shape ``(..., L, 2L-1)`` with ``alm[..., l, lmax + m]`` holding a_lm, L = lmax + 1.
For a real field a_{l,-m} = (-1)^m conj(a_lm), so only m >= 0 is independent.

The *packed* real vector of length L**2 is

    x = [ a_{l0}            for l = 0..lmax,
          sqrt2 Re a_lm     for m = 1..lmax, l = m..lmax,
          sqrt2 Im a_lm     for m = 1..lmax, l = m..lmax ].

It is an orthonormal real basis: x . y = sum_lm a_lm conj(b_lm) = int f g dOmega
for real fields f, g (see docs/math.md, section 1). An isotropic field with
power spectrum C_l therefore has Var(x_i) = C_l(i) for every entry, m = 0 and
m > 0 alike.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


def n_real(lmax):
    """Number of real parameters of a real field band-limited at lmax: L**2."""
    return (lmax + 1) ** 2


def lmax_from_shape(shape):
    """lmax of an s2fft-layout array ``(..., L, 2L-1)``."""
    L, M = shape[-2], shape[-1]
    if M != 2 * L - 1:
        raise ValueError(f"expected shape (..., L, 2L-1), got {tuple(shape)}")
    return L - 1


def lmax_from_n_real(n):
    """lmax from the packed length L**2."""
    L = int(round(np.sqrt(n)))
    if L * L != n:
        raise ValueError(f"packed length {n} is not a square")
    return L - 1


@lru_cache(maxsize=None)
def _indices(lmax):
    """Rows (l) and columns (lmax + m) of the m > 0 entries, in packing order."""
    rows, cols = [], []
    for m in range(1, lmax + 1):
        for ell in range(m, lmax + 1):
            rows.append(ell)
            cols.append(lmax + m)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def pack(alm):
    """s2fft-layout alm ``(..., L, 2L-1)`` of a real field -> packed ``(..., L**2)``.

    Only m >= 0 is read; the m < 0 half is assumed to be the Hermitian partner.
    """
    alm = jnp.asarray(alm)
    lmax = lmax_from_shape(alm.shape)
    rows, cols = _indices(lmax)
    pos = np.sqrt(2.0) * alm[..., rows, cols]
    return jnp.concatenate([alm[..., :, lmax].real, pos.real, pos.imag], axis=-1)


def unpack(x, lmax=None):
    """Packed ``(..., L**2)`` -> s2fft-layout alm ``(..., L, 2L-1)`` (complex)."""
    x = jnp.asarray(x)
    if lmax is None:
        lmax = lmax_from_n_real(x.shape[-1])
    elif x.shape[-1] != n_real(lmax):
        raise ValueError(f"packed length {x.shape[-1]} does not match lmax={lmax}")
    L = lmax + 1
    rows, cols = _indices(lmax)
    npos = len(rows)
    c = (x[..., L : L + npos] + 1j * x[..., L + npos :]) / np.sqrt(2.0)
    sign = (-1.0) ** (cols - lmax)
    alm = jnp.zeros(x.shape[:-1] + (L, 2 * L - 1), dtype=complex)
    # unique_indices lets jax.linear_transpose build the adjoint
    alm = alm.at[..., :, lmax].set(x[..., :L].astype(complex), unique_indices=True)
    alm = alm.at[..., rows, cols].set(c, unique_indices=True)
    alm = alm.at[..., rows, 2 * lmax - cols].set(sign * jnp.conj(c), unique_indices=True)
    return alm


@lru_cache(maxsize=None)
def _ell_m(lmax):
    rows, cols = _indices(lmax)
    ell = np.concatenate([np.arange(lmax + 1), rows, rows])
    m = np.concatenate([np.zeros(lmax + 1, dtype=int), cols - lmax, cols - lmax])
    return ell, m


def ell_of_packed(lmax):
    """l of every packed entry, shape ``(L**2,)``."""
    return _ell_m(lmax)[0].copy()


def m_of_packed(lmax):
    """m (>= 0) of every packed entry, shape ``(L**2,)``."""
    return _ell_m(lmax)[1].copy()


def cl_from_packed(x):
    """Angular power spectrum C_l = sum_m |a_lm|^2 / (2l+1) from packed vectors."""
    x = jnp.asarray(x)
    lmax = lmax_from_n_real(x.shape[-1])
    ell = ell_of_packed(lmax)
    onehot = jnp.asarray(ell[:, None] == np.arange(lmax + 1)[None, :], dtype=x.dtype)
    return (x**2) @ onehot / (2 * np.arange(lmax + 1) + 1)


def cl_from_alm(alm):
    """Angular power spectrum of s2fft-layout alm of a real field, ``(..., L)``."""
    return cl_from_packed(pack(alm))


def cl_to_packed(cl):
    """Broadcast C_l ``(..., L)`` to the packed layout ``(..., L**2)``."""
    cl = jnp.asarray(cl)
    return cl[..., ell_of_packed(cl.shape[-1] - 1)]


def reduce_lmax(alm, lmax):
    """Truncate s2fft-layout alm to a smaller band limit."""
    alm = jnp.asarray(alm)
    old = lmax_from_shape(alm.shape)
    if lmax > old:
        raise ValueError(f"cannot raise lmax from {old} to {lmax}")
    return alm[..., : lmax + 1, old - lmax : old + lmax + 1]
