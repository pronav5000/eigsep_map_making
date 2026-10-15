import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from eigsep_mapmaking import alm


def random_hermitian(lmax, rng, lead=()):
    x = rng.normal(size=lead + (alm.n_real(lmax),))
    return alm.unpack(x, lmax), x


@pytest.mark.parametrize("lmax", [0, 1, 5, 8])
def test_roundtrip(lmax):
    rng = np.random.default_rng(0)
    a, x = random_hermitian(lmax, rng, lead=(3, 2))
    np.testing.assert_allclose(alm.pack(a), x, atol=1e-14)
    np.testing.assert_allclose(alm.unpack(alm.pack(a)), a, atol=1e-14)
    # Hermitian symmetry of the unpacked array
    L = lmax + 1
    for m in range(1, L):
        np.testing.assert_allclose(a[..., :, lmax - m], (-1) ** m * np.conj(a[..., :, lmax + m]))


def test_inner_product_is_sum_over_lm():
    rng = np.random.default_rng(1)
    lmax = 6
    a, x = random_hermitian(lmax, rng)
    b, y = random_hermitian(lmax, rng)
    np.testing.assert_allclose(x @ y, jnp.sum(a * jnp.conj(b)).real, rtol=1e-12)
    assert abs(jnp.sum(a * jnp.conj(b)).imag) < 1e-12


def test_inner_product_is_integral():
    nside, lmax = 8, 16
    rng = np.random.default_rng(2)
    a, x = random_hermitian(lmax, rng)
    b, y = random_hermitian(lmax, rng)
    fa = np.asarray(s2fft.inverse(a, L=lmax + 1, nside=nside, sampling="healpix",
                                  reality=True, method="jax")).real
    fb = np.asarray(s2fft.inverse(b, L=lmax + 1, nside=nside, sampling="healpix",
                                  reality=True, method="jax")).real
    integral = np.sum(fa * fb) * 4 * np.pi / fa.size
    np.testing.assert_allclose(x @ y, integral, rtol=2e-2)  # HEALPix quadrature error


def test_index_helpers():
    lmax = 4
    ell, m = alm.ell_of_packed(lmax), alm.m_of_packed(lmax)
    assert len(ell) == alm.n_real(lmax) == 25
    assert np.all(ell >= m)
    # each (l, m>0) appears twice (Re, Im), each (l, 0) once
    for l_ in range(lmax + 1):
        assert np.sum(ell == l_) == 2 * l_ + 1


def test_isotropic_prior_variance_is_cl_for_every_entry():
    """With the sqrt2 packing, Var(x_i) = C_l for m = 0 and m > 0 (review finding 2.1.2)."""
    lmax = 8
    cl = 1.0 / (1 + np.arange(lmax + 1)) ** 2
    np.random.seed(3)
    hp_alms = np.array([hp.synalm(cl, lmax=lmax) for _ in range(4000)])
    xs = np.asarray(alm.pack(hp_to_s2fft(hp_alms, lmax)))
    m = alm.m_of_packed(lmax)
    ratio = xs.var(0) / alm.cl_to_packed(cl)
    assert abs(ratio[m == 0].mean() - 1) < 0.05
    assert abs(ratio[m > 0].mean() - 1) < 0.05


def hp_to_s2fft(hp_alm, lmax):
    """healpy alm (..., nalm) -> s2fft layout (..., L, 2L-1)."""
    out = np.zeros(hp_alm.shape[:-1] + (lmax + 1, 2 * lmax + 1), complex)
    for m in range(lmax + 1):
        for l_ in range(m, lmax + 1):
            v = hp_alm[..., hp.Alm.getidx(lmax, l_, m)]
            out[..., l_, lmax + m] = v
            out[..., l_, lmax - m] = (-1) ** m * np.conj(v)
    return out


def test_cl_matches_healpy():
    lmax = 7
    rng = np.random.default_rng(4)
    a, x = random_hermitian(lmax, rng)
    hp_alm = np.array([a[l_, lmax + m] for m in range(lmax + 1) for l_ in range(m, lmax + 1)])
    np.testing.assert_allclose(alm.cl_from_alm(a), hp.alm2cl(hp_alm), rtol=1e-12)
    np.testing.assert_allclose(alm.cl_from_packed(x), hp.alm2cl(hp_alm), rtol=1e-12)


def test_reduce_lmax():
    rng = np.random.default_rng(5)
    a, _ = random_hermitian(6, rng)
    b = alm.reduce_lmax(a, 3)
    assert b.shape == (4, 7)
    np.testing.assert_allclose(b[2, 3 + 1], a[2, 6 + 1])
    np.testing.assert_allclose(b[3, 0], a[3, 6 - 3])
