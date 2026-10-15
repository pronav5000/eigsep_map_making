"""Check that the drift-scan forward model is block diagonal in m, so that the
Wiener filter and the sky-marginalised likelihood can be computed per m with
(N_beam x N_beam) matrices. Compared against dense time-domain linear algebra."""
import warnings

warnings.filterwarnings("ignore")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import healpy as hp
import s2fft
import croissant.jax as crojax

nside, lmax = 4, 8
L = lmax + 1
nb, nt = 3, 2 * lmax + 1 + 4  # full sidereal day, uniform
import croissant; day = croissant.constants.sidereal_day["earth"]
fwd = lambda m: s2fft.forward_jax(jnp.asarray(m), L=L, spin=0, nside=nside, sampling="healpix", reality=True)
theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2))
rng = np.random.default_rng(1)

rows, cpos = [], []
for m in range(1, lmax + 1):
    for l in range(m, lmax + 1):
        rows.append(l); cpos.append(lmax + m)
rows, cpos = np.array(rows), np.array(cpos)
npos = len(rows)


def unpack(x):
    a = jnp.zeros((L, 2 * L - 1), complex).at[:, lmax].set(x[:L])
    c = (x[L:L + npos] + 1j * x[L + npos:]) / np.sqrt(2)
    a = a.at[rows, cpos].set(c)
    return a.at[rows, 2 * lmax - cpos].set((-1.0) ** (cpos - lmax) * jnp.conj(c))


cl = 1.0 / (1 + np.arange(L)) ** 1.5
ell_of_x = np.concatenate([np.arange(L), rows, rows])
S = cl[ell_of_x]


def make_beams(eps):
    out = []
    for k in range(nb):
        tilt = 0.3 * k
        bm = np.exp(-((np.arccos(np.clip(np.cos(theta) * np.cos(tilt) + np.sin(theta) * np.sin(tilt) * np.cos(phi), -1, 1)) / 0.7) ** 2))
        bm = bm * (1 + eps * np.cos(2 * phi + k)) * (np.cos(theta) > 0)
        out.append(fwd(bm))
    b = jnp.stack(out)
    return b / crojax.alm.total_power(b, lmax)[:, None, None]  # fold the 1/norm in


phases = crojax.simulator.rot_alm_z(lmax, nt, day / nt, world="earth")
sigma = 0.05


def A_dense(b):
    return jax.jacfwd(lambda x: jnp.einsum("lm,tm,blm->bt", unpack(x).conjugate(), phases, b).real.ravel())(jnp.zeros(L * L))


def logL_dense(b, y):
    A = A_dense(b)
    C = A @ jnp.diag(S) @ A.T + sigma**2 * jnp.eye(A.shape[0])
    return jax.scipy.stats.multivariate_normal.logpdf(y, jnp.zeros(len(y)), C)


def logL_mspace(b, y):
    Y = jnp.fft.fft(y.reshape(nb, nt), axis=1) / nt  # Y[:, k] for e^{-i 2 pi k t / nt}
    # croissant phases are exp(-i m phi): signal at FFT bin  m mod nt
    tot = 0.0
    noise = sigma**2 / nt
    for m in range(0, lmax + 1):
        bl = b[:, m:, lmax + m]  # (nb, L-m)  b_{lm}, l>=m
        Ym = Y[:, (-m) % nt]  # exp(-i m phi) <-> FFT bin -m with numpy sign
        Cm = (bl * cl[m:]) @ bl.conj().T + noise * jnp.eye(nb)
        if m == 0:
            Cm, Ym = Cm.real, Ym.real
            tot += jax.scipy.stats.multivariate_normal.logpdf(Ym, jnp.zeros(nb), Cm)
        else:
            sol = jnp.linalg.solve(Cm, Ym)
            tot += -jnp.linalg.slogdet(np.pi * Cm)[1] - (Ym.conj() @ sol).real
    return tot


b_true = make_beams(0.3)
x_true = jnp.asarray(rng.normal(size=L * L) * np.sqrt(S))
y = A_dense(b_true) @ x_true + sigma * rng.normal(size=nb * nt)

# 1. per-m data: FFT bin equals sum_l b_lm conj(t_lm)
t = unpack(x_true)
Y = jnp.fft.fft((A_dense(b_true) @ x_true).reshape(nb, nt), axis=1) / nt
m = 3
print("1. V_m from FFT vs sum_l b_lm conj(t_lm):", float(jnp.max(jnp.abs(Y[:, (-m) % nt] - jnp.sum(b_true[:, :, lmax + m] * t[:, lmax + m].conj(), axis=1)))))
# 2. marginal likelihood differences agree
for eps in (0.0, 0.15, 0.5):
    b = make_beams(eps)
    d1 = logL_dense(b, y) - logL_dense(b_true, y)
    d2 = logL_mspace(b, y) - logL_mspace(b_true, y)
    print(f"2. eps={eps}: dlogL dense {float(d1):.6f}   m-space {float(d2):.6f}")
# 3. per-m Wiener filter equals dense Wiener filter
A = A_dense(b_true)
x_wf_dense = S * (A.T @ jnp.linalg.solve(A @ jnp.diag(S) @ A.T + sigma**2 * jnp.eye(A.shape[0]), y))
Yd = jnp.fft.fft(y.reshape(nb, nt), axis=1) / nt
t_wf = jnp.zeros((L, 2 * L - 1), complex)
for m in range(0, lmax + 1):
    bl = b_true[:, m:, lmax + m]
    Ym = Yd[:, (-m) % nt]
    Cm = (bl * cl[m:]) @ bl.conj().T + sigma**2 / nt * jnp.eye(nb)
    if m == 0:
        Cm, Ym, bl = Cm.real, Ym.real, bl.real
    # V_m = sum_l b_lm conj(t_lm)  =>  conj(t_lm) = C_l b_lm^* ... : t_lm = C_l b_lm^T conj(C^-1 V)
    tm = cl[m:] * (bl.T @ jnp.conj(jnp.linalg.solve(Cm, Ym)))
    t_wf = t_wf.at[m:, lmax + m].set(tm)
    if m > 0:
        t_wf = t_wf.at[m:, lmax - m].set((-1.0) ** m * jnp.conj(tm))
print("3. max|WF dense - WF per-m| (alm):", float(jnp.max(jnp.abs(unpack(x_wf_dense) - t_wf))), " scale", float(jnp.max(jnp.abs(t_wf))))
