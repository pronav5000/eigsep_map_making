"""Numerical checks of claims made while reviewing the notebooks (see docs/plan.md, section 2).

Needs Python 3.12 with croissant-sim v4.2.1 (croissant.jax), s2fft, healpy, jax.
Uses a small nside and synthetic beams/skies, so no data files are needed.
"""
import warnings

warnings.filterwarnings("ignore")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import healpy as hp
import s2fft
import croissant.jax as crojax

nside = 8
lmax = 2 * nside
L = lmax + 1
npix = 12 * nside**2
rng = np.random.default_rng(0)

fwd = lambda m: s2fft.forward_jax(jnp.asarray(m), L=L, spin=0, nside=nside, sampling="healpix", reality=True)
inv = lambda a: s2fft.inverse_jax(a, L=L, spin=0, nside=nside, sampling="healpix", reality=True)


def bandlimited_random_map():
    alm = hp.synalm(1.0 / (1 + np.arange(L)) ** 2, lmax=lmax)
    return hp.alm2map(alm, nside, lmax=lmax)


theta, phi = hp.pix2ang(nside, np.arange(npix))
pix_area = 4 * np.pi / npix

# ---------------------------------------------------------------- A
print("A. forward-model convention")
B = np.exp(-((theta / 0.6) ** 2)) * (1 + 0.5 * np.cos(phi) + 0.3 * np.sin(2 * phi))  # azimuthal structure
B = hp.alm2map(hp.map2alm(B, lmax=lmax), nside, lmax=lmax)  # band-limit
T = bandlimited_random_map() + 3.0
b = fwd(B)
t = fwd(T)
ntimes = 7
dt = 86164.0905 / ntimes / 3.0
phases = crojax.simulator.rot_alm_z(lmax, ntimes, dt, world="earth")
v_cro = crojax.simulator.convolve(b[None], t[None], phases)[:, 0].real
v_fresh = jnp.einsum("lm,tm,lm->t", t.conjugate(), phases, b).real
v_fixed = jnp.einsum("lm,tm,lm->t", t, phases.conjugate(), b).real
print(f"   max|cro - fresh-notebook einsum| = {float(jnp.max(jnp.abs(v_cro - v_fresh))):.2e}")
print(f"   max|cro - fixed-notebook einsum| = {float(jnp.max(jnp.abs(v_cro - v_fixed))):.2e}  (rel {float(jnp.max(jnp.abs(v_cro - v_fixed)) / jnp.max(jnp.abs(v_cro))):.2e})")
B_mirror = hp.alm2map(hp.map2alm(np.exp(-((theta / 0.6) ** 2)) * (1 + 0.5 * np.cos(phi) - 0.3 * np.sin(2 * phi)), lmax=lmax), nside, lmax=lmax)
v_cro_mirror = crojax.simulator.convolve(fwd(B_mirror)[None], t[None], phases)[:, 0].real
print(f"   max|fixed-notebook - cro(beam mirrored phi->-phi)| = {float(jnp.max(jnp.abs(v_fixed - v_cro_mirror))):.2e}")

# ---------------------------------------------------------------- B
print("B. packed real-alm vector is an orthonormal real basis")
rows, cpos = [], []
for m in range(1, lmax + 1):
    for l in range(m, lmax + 1):
        rows.append(l); cpos.append(lmax + m)
rows, cpos = np.array(rows), np.array(cpos)


def pack(a):
    return jnp.concatenate([a[:, lmax].real, np.sqrt(2) * a[rows, cpos].real, np.sqrt(2) * a[rows, cpos].imag])


x1, x2 = pack(b), pack(t)
full = jnp.sum(b * t.conjugate()).real
print(f"   x1.x2 = {float(x1 @ x2):.6f},  sum_lm b conj(t) = {float(full):.6f},  int B T dOmega = {np.sum(B * T) * pix_area:.6f}")
# isotropic field: variance of each packed entry should equal C_l (not C_l/2)
nsim = 4000
cl = np.ones(L)
xs = []
for _ in range(nsim):
    m = hp.synfast(cl, nside, lmax=lmax)
    xs.append(np.asarray(pack(fwd(m))))
xs = np.array(xs)
print(f"   isotropic C_l=1 field: mean var of m=0 entries {xs[:, :L].var(0).mean():.3f}, of m>0 entries {xs[:, L:].var(0).mean():.3f}  (=> S_diag should be C_l for all entries)")

# ---------------------------------------------------------------- C
print("C. the injected 'amp 1.1, phase 5 deg' perturbation")
a_pert = b * (1.10 * np.exp(1j * np.radians(5.0)))
B_pert = np.asarray(inv(a_pert).real)
P_true = float(crojax.alm.total_power(b, lmax))
P_raw = float(crojax.alm.total_power(fwd(B_pert), lmax))
print(f"   total_power is linear: P(2B)/P(B) = {float(crojax.alm.total_power(fwd(2 * B), lmax)) / P_true:.3f}")
print(f"   P_pert_raw/P_true = {P_raw / P_true:.4f}  (1.1*cos5deg = {1.1 * np.cos(np.radians(5)):.4f})")
B_pert_scaled = B_pert * np.sqrt(P_true / P_raw)
print(f"   after notebook's sqrt rescale: P/P_true = {float(crojax.alm.total_power(fwd(B_pert_scaled), lmax)) / P_true:.4f}  (not 1 -> rescale should be a ratio, not sqrt)")
# what survives normalisation: compare B/int(B)
n_true = B / P_true
n_pert = B_pert / P_raw
d_raw = B_pert_scaled - B
d_norm = n_pert - n_true
print(f"   |B_pert - B|/|B|                 = {np.linalg.norm(d_raw) / np.linalg.norm(B):.4f}")
print(f"   |B_pert/P - B/P|/|B/P| (observable) = {np.linalg.norm(d_norm) / np.linalg.norm(n_true):.4f}")
coef = np.dot(d_raw, B) / np.dot(B, B)
print(f"   fraction of perturbation power along the beam itself (unobservable scale): {coef**2 * np.dot(B, B) / np.dot(d_raw, d_raw):.3f}")
# how does s2fft's reality=True inverse treat the m=0 imaginary part and the m<0 half?
a_test = jnp.zeros((L, 2 * L - 1), complex).at[2, lmax].set(1j)
print(f"   inverse of a purely imaginary a_20 (reality=True): max|map| = {float(jnp.max(jnp.abs(inv(a_test)))):.2e}")
a_test2 = jnp.zeros((L, 2 * L - 1), complex).at[3, lmax + 2].set(1.0)  # m=+2 only, no m=-2 partner
a_test3 = a_test2.at[3, lmax - 2].set(1.0)  # (+ (-1)^2 conj(1) = 1, hermitian partner)
print(f"   m<0 half ignored by reality=True inverse? max|inv(m=+2 only) - inv(hermitian pair)| = {float(jnp.max(jnp.abs(inv(a_test2) - inv(a_test3)))):.2e}")

# ---------------------------------------------------------------- D
print("D. horizon mask via [npix//2:] = 0 in RING ordering")
z = np.cos(theta)
print(f"   pixels with z>0 zeroed: {(z[npix // 2:] > 1e-12).sum()},  pixels with z<0 kept: {(z[:npix // 2] < -1e-12).sum()},  equator (z=0) kept/zeroed: {(np.abs(z[:npix//2])<1e-12).sum()}/{(np.abs(z[npix//2:])<1e-12).sum()}")

# ---------------------------------------------------------------- E
print("E. time step used in the radiometer noise")
dt_sim = 86164.0905 / 500
jd0 = 2461010.875
times_jd = jd0 + np.arange(500) * dt_sim  # seconds added to days, as in the notebook
dt_nb = (times_jd[1] - times_jd[0]) * 24 * 3600
print(f"   dt_sim = {dt_sim:.2f} s, notebook dt = {dt_nb:.4e} s, ratio = {dt_nb / dt_sim:.0f}; sigma too small by sqrt(ratio) = {np.sqrt(dt_nb / dt_sim):.0f}x")

# ---------------------------------------------------------------- F
print("F. information content of a single-frequency drift scan")
for nb in (1, 2, 5, 20):
    rng2 = np.random.default_rng(nb)
    beams = jnp.stack([fwd(hp.alm2map(hp.synalm(1.0 / (1 + np.arange(L)) ** 3, lmax=lmax), nside, lmax=lmax) * (z > 0)) for _ in range(nb)])
    nt = 2 * lmax + 1 + 10
    ph = crojax.simulator.rot_alm_z(lmax, nt, 86164.0905 / nt, world="earth")
    nparam = L * L

    def unpack(x):
        a = jnp.zeros((L, 2 * L - 1), complex).at[:, lmax].set(x[:L])
        c = (x[L:L + len(rows)] + 1j * x[L + len(rows):]) / np.sqrt(2)
        a = a.at[rows, cpos].set(c)
        return a.at[rows, 2 * lmax - cpos].set((-1.0) ** (cpos - lmax) * jnp.conj(c))

    A = jax.jacfwd(lambda x: jnp.einsum("lm,tm,blm->bt", unpack(x).conjugate(), ph, beams).real.ravel())(jnp.zeros(nparam))
    sv = np.linalg.svd(np.asarray(A), compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    bound = sum(min(nb, L - abs(m)) for m in range(-lmax, lmax + 1))
    print(f"   {nb:2d} beams, {nt} times: rank {rank} of {nparam} sky params  (bound sum_m min(n_beams, lmax+1-|m|) = {bound})")
