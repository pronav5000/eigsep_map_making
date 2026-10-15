# Mathematical foundation

This document sets out the model behind `eigsep_mapmaking`: what is measured, how the sky and
the beam enter, what a drift scan can and cannot constrain, and how the sky and the beam are
inferred. Every formula is implemented in the module named next to it, and the test given in
brackets checks it numerically. All tests run on synthetic beams and skies at small band limit
(`pytest tests`).

Notation: ν is frequency, t a time sample (T of them), p a pointing (P of them), ℓ, m
spherical-harmonic indices up to ℓ_max, L = ℓ_max + 1. A star is complex conjugation.

---

## 1. Fields on the sphere

### 1.1 Real fields and their harmonic coefficients

The sky temperature T(n̂) and the beam power pattern B(n̂) are real fields. Band-limited at
ℓ_max,

  f(n̂) = Σ_{ℓ≤ℓmax} Σ_{|m|≤ℓ} a_ℓm Y_ℓm(n̂),  with  a_{ℓ,−m} = (−1)^m a*_ℓm  (reality).

Orthonormality of the Y_ℓm gives the Parseval relation for two real fields:

  ∫ f g dΩ = Σ_ℓm a_ℓm b*_ℓm.                                                  (1.1)

In particular the integral of a field is its monopole: ∫ f dΩ = √(4π) a_00.

### 1.2 The s2fft layout

s2fft and croissant store a_ℓm in a complex array of shape (L, 2L−1), with a_ℓm at
`[ℓ, ℓmax + m]` and zeros where |m| > ℓ. Leading axes (frequency, pointing) come first.

### 1.3 Real packing (`alm.pack`, `alm.unpack`)

Only m ≥ 0 is independent. The packed real vector of length L² is

  x = ( a_ℓ0  for ℓ = 0..ℓmax ;  √2 Re a_ℓm ;  √2 Im a_ℓm  for m = 1..ℓmax, ℓ = m..ℓmax ).

Splitting the sum in (1.1) into m = 0 and pairs (m, −m) gives
Σ_ℓm a_ℓm b*_ℓm = Σ_ℓ a_ℓ0 b_ℓ0 + Σ_{m>0} 2 Re(a_ℓm b*_ℓm) = x · y, so the packed basis is
orthonormal [`test_alm::test_inner_product_is_sum_over_lm`, `…_is_integral`].

### 1.4 Isotropic Gaussian fields

An isotropic field has E[a_ℓm a*_ℓ'm'] = C_ℓ δ_ℓℓ' δ_mm'. For m > 0 the real and imaginary
parts are independent with variance C_ℓ/2 each (E[a_ℓm a_ℓm] = 0), so **every packed entry has
variance C_ℓ**, for m = 0 and m > 0 alike [`test_alm::test_isotropic_prior_variance_is_cl_for_every_entry`].
The covariance is diagonal in the packed basis: S = diag(C_ℓ(i)) (`sky.SkyPrior`). One notebook
halved the m > 0 entries; that was wrong.

The power spectrum of one field is Ĉ_ℓ = Σ_m |a_ℓm|² / (2ℓ+1) = Σ_{i: ℓ(i)=ℓ} x_i² / (2ℓ+1)
(`alm.cl_from_packed`) [`test_alm::test_cl_matches_healpy`].

---

## 2. The measurement equation

### 2.1 Frames

* **Beam frame:** topocentric for a zenith-pointing antenna, x East, y North, z up; θ is the
  zenith angle and φ runs from East towards North (croissant's convention). The beam file is a
  full-sphere pattern B_ν(n̂) in this frame.
* **Pointing p:** an active rotation R_p of the antenna in the topocentric frame,
  R_p = R_z(φ₀) R_y(τ) R_z(−φ₀) R_z(−ρ), with φ₀ = π/2 − A. It first turns the antenna about
  the zenith by ρ (clockwise seen from above, croissant's `beam_rot`), then tilts the boresight
  by τ towards compass azimuth A (`instrument.pointing_matrix`)
  [`test_instrument::test_pointing_convention`, `…_moves_peak_as_pointing_says`]. The notebooks'
  healpy `rot=(0, 0, el)` tilts along the North–South line. The default pointings are 20 tilts
  of 0–90° towards North.
* **Simulation frame:** croissant `main`'s frame, CIRS at the first sample t₀ on Earth (MEPA on
  the Moon). Its z axis is the rotation axis, so the sky turns about z. Galactic sky alm are
  rotated in with `Sky.compute_alm_eq(world, et=et_ref)`; topocentric beam alm with
  `Simulator.topo_to_sim` (`sky.sky_alm_sim`, `instrument.beam_response`).

### 2.2 Horizon, ground and normalization

The horizon is fixed to the ground: H(n̂) ∈ [0, 1] is the visible fraction of each direction,
built by croissant from a horizon colatitude θ_h (default π/2) with fractional weights on the
boundary cells. For pointing p, B_p(n̂) = B(R_p⁻¹ n̂) is the rotated pattern. croissant `main`
defines

  N_p = ∫ B_p dΩ  (whole sphere),   N_p^↑ = ∫ H B_p dΩ,   f_gnd,p = 1 − N_p^↑ / N_p,

and models the ground as a uniform emitter at T_gnd. The antenna temperature is

  V_p(t) = [ ∫ H(n̂) B_p(n̂) T_t(n̂) dΩ + (N_p − N_p^↑) T_gnd ] / N_p.                (2.1)

The first term is the sky seen above the horizon, the second the ground term
g_p = f_gnd,p T_gnd. N_p does not depend on the pointing [`test_instrument::test_norm_is_rotation_invariant_and_fgnd`].

### 2.3 Harmonic form and sky rotation

Let b_{p,ℓm} be the alm of H·B_p in the simulation frame and s_ℓm those of the sky at t₀. The
Earth turns by φ(t) = 2π (t − t₀)/T_sid about z (T_sid = 86400/1.00273781191135448 s, one turn
of the Earth rotation angle), which carries the beam to b_ℓm e^{−imφ}. With (1.1),

  V_p(t) = Σ_ℓm s*_ℓm e^{−imφ(t)} b_{p,ℓm} / N_p + g_p                              (2.2)

(`forward.ForwardModel`, croissant's `convolve`). The sum is real because b and s are
Hermitian. In the packed sky vector x this is y = A x + g with A linear (`apply`), adjoint Aᵀ
(`adjoint`, by `jax.linear_transpose`) [`test_forward::test_adjoint_dot_product`].
`ForwardModel.predict` equals croissant's `Simulator.sim` for every pointing to 1e-10, with and
without integration windows [`test_instrument::test_matches_croissant_simulator`,
`test_forward::test_predict_matches_croissant_convolve`].

The notebooks' "fixed" variant used Σ s_ℓm e^{+imφ} b_ℓm instead of Σ s*_ℓm e^{−imφ} b_ℓm. That is (2.2)
for the beam mirrored in azimuth (φ → −φ) (`scripts/review/check_notebook_claims.py`, check A).

### 2.4 Invariance under beam scaling

b, N and N^↑ are all linear in B, so (2.1) is unchanged by B_ν → c_ν B_ν at each frequency,
ground term included [`test_forward::test_beam_scale_invariance`]. This is the scale null
direction of §5.

### 2.5 Integration windows

A sample integrating over [t, t + Δt] sees the time average of e^{−imφ}. The sky turns at a
constant rate, so with Δφ = 2πΔt/T_sid,

  ⟨e^{−imφ}⟩ = e^{−im(φ_t + Δφ/2)} sinc(mΔφ/2),  sinc x = sin x / x                 (2.3)

(croissant `rot_alm_z(times_end=…)`). Still diagonal in m.

---

## 3. Fourier structure of a drift scan

### 3.1 Per-m decoupling

Let the T samples evenly tile one sidereal day, φ_t = 2πt/T, with T > 2ℓmax. Then
(1/T) Σ_t e^{i(m−m')φ_t} = δ_mm' for |m − m'| ≤ 2ℓmax < T. Projecting the data,

  Y_{p,m} ≡ (1/T) Σ_t y_p(t) e^{+imφ_t} = w_m Σ_{ℓ≥m} s*_ℓm b_{p,ℓm} / N_p  (+ g_p at m = 0),   (3.1)

with w_m = 1 for snapshots and w_m = e^{−imΔφ/2} sinc(mΔφ/2) for equal back-to-back windows
(`ForwardModel.to_mspace`, `mspace_response`, `window_factor`)
[`test_forward::test_mspace_matches_time_domain`, `…_with_integration_windows`; agreement is
1e-8, limited by Julian-day rounding of the sample times, ~40 µs].

Writing R_m[p, ℓ] = w_m b_{p,ℓm}/N_p, a P × (L−m) matrix, and t_m = (s_ℓm)_{ℓ≥m}:

  Y_m = R_m t*_m + n_m.                                                              (3.2)

For real data Y_{−m} = Y*_m, so m = 0..ℓmax carries all the information. m = 0 is real.

### 3.2 Noise in m-space

White noise of variance σ_p² per sample gives E|n_{p,m}|² = σ_p²/T. For 0 < m < T/2 the noise
is circular complex (E[n n] = 0), and n_0 is real. The m-blocks are independent. The noise must
be stationary per (frequency, pointing) for this to hold; otherwise use the time domain (CG).

### 3.3 Identifiability

Block m constrains at most min(P, L − |m|) complex (m > 0) or real (m = 0) combinations of the
sky, so per frequency

  rank ≤ Σ_{m=−ℓmax}^{ℓmax} min(P, ℓmax + 1 − |m|)                                   (3.3)

out of L² sky parameters (`diagnostics.rank_bound`, `mspace_rank`)
[`test_diagnostics::test_adjoint_and_rank`; review check F reproduces the bound exactly for
random beams]. For 20 pointings at ℓmax = 128 that is 4,760 of 16,641 (≤ 29 %). The bound does
not depend on T once T > 2ℓmax: more time samples lower the noise but add no modes. More
pointings add up to one mode per m each, and only while they are linearly independent at that
m. Frequencies are independent problems unless the prior couples them. Everything outside the
span of the R_m is set by the prior. That is why even a perfect beam gives a 30 % map error.

---

## 4. Linear-Gaussian inference for the sky

### 4.1 Prior

x ~ N(0, S) with S = diag(C_ℓ(i)) (§1.4). `SkyPrior.oracle` takes C_ℓ from the true sky, as
the notebooks did, floored at a fraction of its peak. `SkyPrior.power_law` is a non-oracle
choice, C_ℓ(ν) = A ℓ^{−α} (ν/ν₀)^{−2β}, with a separate monopole variance 4π σ_T̄².

### 4.2 Wiener filter

With d = y − g (the ground term is known) and n ~ N(0, N), N diagonal, the posterior is
Gaussian with

  x̂ = (S⁻¹ + AᵀN⁻¹A)⁻¹ AᵀN⁻¹ d = S Aᵀ (A S Aᵀ + N)⁻¹ d,                               (4.1)
  Σ_post = S − S Aᵀ (A S Aᵀ + N)⁻¹ A S.                                                (4.2)

`mapmaking.wiener_cg` solves the dual (data-space) form of (4.1) by conjugate gradients
preconditioned with N⁻¹; `wiener_dense` uses dense matrices (small problems).

### 4.3 Exact per-m Wiener filter

Under §3's conditions (4.1) splits into independent blocks. With C_m = R_m diag(C_ℓ) R_m^H + N_m,
N_m = diag(σ_p²)/T,

  E[t*_m | Y_m] = diag(C_ℓ) R_m^H C_m⁻¹ Y_m,                                           (4.3)
  Var(s_ℓm | Y) = C_ℓ − C_ℓ² [R_m^H C_m⁻¹ R_m]_ℓℓ,                                     (4.4)

a P × P solve per (ν, m) (`wiener_mspace`, `posterior_var_mspace`). For m > 0 the posterior is
circular, so √2 Re and √2 Im each carry the variance (4.4). For m = 0 everything is real.
[`test_mapmaking::test_cg_equals_dense` (1e-7), `test_mspace_equals_dense` (mean 1e-6,
variance 1e-5).]

### 4.4 Constrained realizations

x_cr = x̂(d) + x_s − x̂(A x_s + n_s) with x_s ~ N(0, S), n_s ~ N(0, N) is a draw from the
posterior (`constrained_realization`) [`test_mapmaking::test_constrained_realizations_have_posterior_covariance`,
400 draws against (4.1)–(4.2)].

The positivity clip and γ rescale of the notebooks make a non-linear, biased estimate. They
remain only as `clip_positive_for_display`, for plots.

### 4.5 Noise

Radiometer equation: σ = T_ant / √(Δν τ), with Δν in Hz and τ in s (`noise.radiometer_sigma`).
The default (plan D8) is stationary: one σ per (ν, p) from the time-averaged model T_ant
(`sigma_from_model`). σ(t) ∝ T_ant(t) is an option, valid with CG only. τ is the window length,
or the sample spacing for snapshots, computed in seconds from Julian days
(`integration_time_s`) [`test_noise_simulate::test_radiometer_and_units`]. The notebooks added
seconds to Julian days (σ ~294× too small) and used Δν in Hz × 10⁶ (σ 1000× too small); their
data also had no noise realization.

---

## 5. Beam errors

### 5.1 Parameterization

B_ν(β) = B0_ν + Σ_k β_k M_k,ν, with modes on the beam's grid in the beam frame
(`perturbations.BeamBasis`). Rotation, horizon weighting, transforms and integrals are linear,
so

  b_p(β) = b_p⁰ + Σ β_k b_p^k,   N_p(β) = N_p⁰ + Σ β_k N_p^k,   N_p^↑(β) = …,

and V(β) is (2.1) with these. The responses are precomputed once (`LinearBeamModel.build`),
so the likelihood does no transforms. The model stays exact: norm and ground fraction are
ratios of linear functions of β [`test_perturbations::test_linear_model_equals_recompute`, 1e-11].

### 5.2 The scale null direction

By §2.4, a mode M_ν ∝ B0_ν at every frequency changes nothing
[`test_perturbations::test_scale_direction_is_invisible`]. Both bases therefore project it out
per frequency, M_ν ← M_ν − (⟨M_ν, B0_ν⟩ / ⟨B0_ν, B0_ν⟩) B0_ν with ⟨a, b⟩ = ∫ a b dΩ, and then
re-orthonormalize in Σ_ν ∫ dΩ. Modes are scaled to the beam's RMS, so β_k = 0.05 is a 5 % RMS
error.

* `svd_basis`: B = Σ_j s_j u_j(ν) v_j(n̂) over the **whole band**, raw modes u_j(ν) v_j(n̂).
  Without the projection the leading mode is ≥ 95 % scale
  [`test_perturbations::test_svd_raw_basis_is_mostly_scale`]. That explains the notebooks'
  near-degenerate fits and the "oracle" p ≈ −0.047·(1, …, 1). With one channel the SVD has a
  single mode, the beam itself, so the SVD must be taken over the band.
* `lowl_basis`: B0_ν(n̂) Y_i(n̂) P_j(x_ν), real harmonics 1 ≤ ℓ ≤ ℓ_pert and Legendre
  polynomials in frequency.

The comparison metric is the normalized beam B/∫B (`diagnostics.normalized_beam_error`).

### 5.3 The notebooks' "amplitude 1.1, phase 5°" error

Multiplying every s2fft coefficient by c = 1.1 e^{i5°} and synthesizing with `reality=True`
(which reads m ≥ 0 only) gives a beam whose m = 0 part is scaled by 1.1 cos 5° and whose m > 0
part is scaled by 1.1 and multiplied by e^{i5°}
[`test_diagnostics::test_notebook_amp_phase_perturbation`]. The 1.1 is invisible (§2.4). The
phase is not a rotation, which would be e^{−imα}, depending on m. It is an artefact of the
storage layout, the same at every frequency.

### 5.4 Sky–beam degeneracies

Any change of R_m that a change of t_m can absorb without changing the prior probability is
invisible to the marginal likelihood. Exact example: turning every pointing about the celestial
pole, b_ℓm → b_ℓm e^{−imα}, is undone by s_ℓm → s_ℓm e^{−imα}, and the isotropic prior cannot
tell the two apart [`test_fitting::test_rotation_about_pole_is_exactly_degenerate`]. So is a
clock offset. More generally, a beam error is constrained only through how it changes the
column space of R_m relative to the prior. The Fisher matrix (§6.3) measures this.

---

## 6. Joint inference of beam and sky

### 6.1 Profile / fixed-sky fits (not recommended)

The notebooks maximized p(y | β, x̂) with x̂ a Wiener map (shrunk towards zero, clipped and
rescaled), or with x̂ the true sky, and alternated sky and beam updates (iterated conditional
modes). The Wiener map is shrunk in the unconstrained modes, and the beam fit then compensates
for the shrinkage. ICM converges to a joint mode with no uncertainty. `fitting.loglike_fixed_sky`
is kept for comparison.

### 6.2 Sky-marginalized likelihood

Integrating x out of the linear-Gaussian model,

  p(y | β) = N(y; g(β), A(β) S A(β)ᵀ + N).                                             (6.1)

Under §3's conditions it factorizes over (ν, m):

  log p = Σ_ν [ log N_ℝ(Y_0; g(β), C_0) + Σ_{m>0} ( −log det(π C_m) − Y_m^H C_m⁻¹ Y_m ) ] + const,   (6.2)

with C_m = R_m(β) diag(C_ℓ) R_m(β)^H + N_m (`fitting.loglike_marginal_mspace`, Cholesky per
block). The constant (the Jacobian of the projection and the pure-noise frequency bins) does
not depend on β [`test_fitting::test_mspace_marginal_equals_dense`: differences agree with
dense (6.1) to 1e-6]. It needs no sky estimate, and it is exact and differentiable.

### 6.3 Fisher information and whitening

For a real Gaussian with mean μ(β) and covariance C(β),
F_ij = ½ tr(C⁻¹∂_iC C⁻¹∂_jC) + ∂_iμᵀ C⁻¹ ∂_jμ; for a circular complex Gaussian with zero mean,
F_ij = tr(C⁻¹∂_iC C⁻¹∂_jC). Summing (6.2)'s blocks gives `fitting.fisher`
[`test_fitting::test_fisher_is_expected_negative_hessian`, against the mean −Hessian over 400
data draws]. `whiten` diagonalizes F. Its small eigenvalues are the poorly constrained
directions.

### 6.4 Sampling

NUTS on β, with prior β_k ~ N(0, σ_k) and the factor (6.2) (`numpyro_model`, `run_nuts`). The
posterior is usually far narrower than the prior, and (6.2) can have secondary modes, so the
chain starts at the MAP (`find_map`: BFGS on log p(β | y) with JAX gradients, from β = 0 and
from prior draws, keeping the best) and uses a dense mass matrix. Three failures led to this.
A chain started by numpyro's default (uniform in (−2, 2)) stuck at log L = −387, against −305
at the truth. A chain started at β = 0 stalled during warmup when the posterior was ~100×
narrower than the prior. With 3 pointings, BFGS from β = 0 found a secondary mode
[`test_fitting::test_find_map_escapes_secondary_mode`]. Given β samples, sky posterior draws
come from §4.4 for each β. A Gibbs sampler for partial-day data is a follow-up
[`test_fitting::test_nuts_recovers_injected_beam_error`].

---

## 7. What the simulations can and cannot validate

* **Self-consistency, not truth.** Mock data are made with the same forward model as the
  inference. Tests check the algebra (adjoints, decompositions, likelihoods, samplers), not the
  physics of the instrument.
* **Gaussian, isotropic prior.** The real sky is neither. With an oracle C_ℓ the synthetic
  recovery test still works, but beam posteriors on real data inherit the prior's
  misspecification. A non-oracle prior (`power_law`) and hyperparameter sampling are needed for
  honest results.
* **Discretization.** HEALPix transforms at `niter=0` integrate with errors of order 1 % at
  ℓmax = 2 nside. MWSS grids are exact for band-limited fields, which is why pointed beams use
  MWSS. A horizon-masked beam is not band-limited: truncating it at ℓmax rings, which can give
  small negative ground fractions for zenith pointings.
* **m-space assumptions.** Exactly one sidereal day, uniform sampling, T > 2ℓmax, equal
  windows, stationary noise per (ν, p). Anything else (gaps, flags, partial days, σ(t)) needs
  the time-domain CG path. The beam likelihood then needs a time-domain Gibbs sampler (not
  implemented).
* **Ground.** A uniform T_gnd, known and subtracted. The EIGSEP bowtie file is a full-sphere
  directivity with 51–57 % of its power below θ = 90° (no ground plane in the pattern), so
  f_gnd ≈ 0.5 for a zenith pointing. Whether that is the right physical model is an open
  question.
* **Not modelled:** polarization, aberration (croissant's note: up to 20″), ground scattering,
  beam chromaticity faster than the beam file's channels, and frequency coupling in the prior.
