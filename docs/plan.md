# Reorganization plan: notebooks → `eigsep_mapmaking` package

**Status: revised after the first review. Decisions D2, D4, D5, D6 and D7 are settled (§8).
The open questions are in §8.2. Nothing from §3 onwards is implemented yet.**
Progress is tracked in [`LOGBOOK.md`](../LOGBOOK.md).

Deliverable: branch `refactor/modules`, pushed to the fork `christianhbye/eigsep_map_making`,
with a PR into `pronav5000/eigsep_map_making:main`. The branch extracts the mapmaking, beam
perturbation and MCMC beam fitting into tested modules, plus a README and `docs/math.md`, a
document setting out the mathematical foundation.

---

## 1. What is in the repo today

| notebook | what it does | lineage |
|---|---|---|
| `croissant_jax.ipynb` | Smallest example: a Moon drift-scan simulation with `croissant.jax` (ULSA sky, one beam). Not mapmaking. | standalone |
| `wiener_eigsep_beams.ipynb` | Single-frequency Wiener-filter mapmaking with 2 beams (bowtie + vivaldi), Earth, scipy `LinearOperator` and a **dense** inverse of the data-space matrix. | first mapmaker |
| `wiener_eigsep_beams_fast.ipynb` | Same as above, but pure JAX (`einsum` forward, `jax.linear_transpose` adjoint, JAX CG). | fast version of ↑ |
| `mcmc_beam_and_sky_fresh.ipynb` | 20 pointings of the bowtie, a "perturbed" model beam, a Wiener sky, then NUTS on 5 SVD beam-mode coefficients with the Wiener sky held fixed. | first beam fit |
| `simultaneous_beam_and_sky_fixed.ipynb` | ↑ plus 9 "fixes", alternating sky/beam optimization (LBFGS), NUTS, Fisher whitening, many diagnostics. Its last cells never ran successfully. | debugging of ↑ |

All of the above repeat the same setup code with small variations: loading beams and skies, SHTs,
frame rotations, alm packing, and the forward model. Data files (`eigsep_bowtie_v000.npz`,
`eigsep_vivaldi.npz`, `gsm_model.npy`, `beam.npy`, `ulsa.npy`) are **not in the repo**.

The notebooks use `croissant.jax` from croissant v4.x. **Per D2 we move to croissant `main`**
(v5 line, JAX-native; §2.4 lists what changes).

## 2. Review findings

Each claim below was checked numerically on small synthetic problems:
[`scripts/review/check_notebook_claims.py`](../scripts/review/check_notebook_claims.py) (sections A–F) and
[`scripts/review/check_mspace_decomposition.py`](../scripts/review/check_mspace_decomposition.py).
Both were run against croissant v4.2.1, the version the notebooks used. The `convolve` and
`rot_alm_z` conventions they test are unchanged on croissant `main`.

### 2.1 Mapmaking: I agree it is mostly sound

Correct as written:
- The measurement equation V_b(t) = ∫B_b T(t) dΩ / ∫B_b dΩ, evaluated in harmonic space. In
  `wiener_eigsep_beams_fast` and `mcmc_beam_and_sky_fresh` the `einsum` matches croissant's
  `convolve` to 1e-15 (check A).
- The real-alm packing (m=0 real, then √2·Re, √2·Im for m>0). It is an orthonormal real basis:
  x·y = Σ_lm b_lm t*_lm (check B).
- The adjoint comes from `jax.linear_transpose`, and the dot-product test passes.
- The dual-form Wiener filter (A S Aᵀ + N)α = y, m = S Aᵀα, solved with CG.

Issues to fix while extracting:
1. **Forward model regression in `simultaneous_beam_and_sky_fixed`.** It uses
   `einsum(sky, conj(phases), beam)` where the other notebooks use `einsum(conj(sky), phases, beam)`.
   The two are not equivalent: the "fixed" version is exactly croissant's visibility for the beam
   **mirrored in azimuth** (φ → −φ), see check A. It goes unnoticed in that notebook because the
   mock data are made with the same operator.
2. **Prior variance halved for m>0 in `simultaneous_beam_and_sky_fixed`.** With the √2 packing,
   every packed entry of an isotropic field has variance C_ℓ (check B). The earlier
   `mcmc_beam_and_sky_fresh` had this right.
3. **Noise σ is wrong.** In the beam notebooks, seconds are added to Julian days, so `dt` comes
   out 86400× too large and σ ~294× too small (check E). In the Wiener notebooks, `df` in Hz×1e6
   makes σ 1000× too small. This is why a "1.5% model floor" dominated `Ndiag`.
4. **Simulated data are noiseless**, so posterior widths and coverage cannot be tested.
5. **The prior uses the true sky's C_ℓ, floored at 1e-4 of the peak** (an oracle prior). Keep
   it as an explicit option, and add a non-oracle one.
6. **The positivity clip followed by a γ rescale is not a linear estimator.** Use it for display
   only; it must not feed the beam fit.
7. **The information content is fundamentally limited, and "more times" will not help.** For a
   uniformly sampled sidereal day, the time FFT separates the data by m:
   V_{b,m} = Σ_ℓ b_{b,ℓm} t*_{ℓm}. The rank is therefore ≤ Σ_m min(N_beam, ℓmax+1−|m|): for 20
   pointings at ℓmax=128 that is **≤ 4,760 of 16,641 sky modes (≤ 29%) per frequency**,
   however many time samples there are beyond 2ℓmax+1 (check F confirms the bound exactly). The
   32–36% map error seen with a *perfect* beam is mostly this null space being filled in by the
   prior.
8. Small issues: the `rel err` print takes `sqrt` of a norm ratio. The horizon mask
   `[npix//2:] = 0` keeps half of the equatorial ring (check D), which croissant `main`'s
   fractional horizon weights handle properly.

The same m-decomposition gives an *exact* per-m Wiener filter using N_beam×N_beam solves (verified
against the dense solution to 1e-14). It serves as a fast, exact reference for the CG solver.

### 2.2 Beam perturbation: this is where most of the problems are

1. **The injected amplitude error is invisible.** V is invariant to B → cB because of the ∫B
   normalization (this is still true with croissant `main`'s ground term, see §2.4).
   `total_power` is linear in B, so the "power-conserving" rescale should be a ratio, not `sqrt`
   (check C).
2. **The "5° phase" is not a physical perturbation.** Multiplying every s2fft coefficient by
   e^{iφ₀} gives m=0 an imaginary part, and with `reality=True` the m<0 half is ignored on
   synthesis (check C). It is an artifact of how s2fft stores coefficients, identical at every
   frequency.
3. **The fitting basis is the SVD of the beam itself, so it is dominated by the unobservable
   scale direction.** The "oracle" `p ≈ −0.047·(1,…,1)` amounts to "subtract ~5% of the beam". This
   explains the notebook's near-degenerate modes (condition number ~5,400) and its "anti-aligned"
   fits.
4. **The beam-error metric compares unnormalized beams.** It should compare B/∫B.
5. All 20 pointings are tilts about one axis (`az = 0`). The healpy `rot=(0, 0, el)`
   convention needs to be made explicit.

### 2.3 Beam fitting / MCMC

1. **Fixed-sky fits are biased.** Both NUTS runs condition on a Wiener sky (shrunk, clipped,
   rescaled) or on the true sky (the "cheat"). The alternating LBFGS loop (iterated conditional
   modes) has the same problem and gives no uncertainty.
2. `mcmc_beam_and_sky_fresh` normalizes with the stale `norms_PERTURBED` and uses a Student-t
   likelihood to absorb the ℓ>10 truncation. Both are symptoms of 1.
3. **Replacement (D6): marginalize the sky analytically.** p(y | β) = N(y; 0, A(β) S A(β)ᵀ + N).
   Per m this is a sum of N_beam×N_beam complex Gaussian log-densities: cheap, exact and
   differentiable, so NUTS samples β with no sky estimate at all. Verified against the dense
   marginal likelihood to machine precision. The sky posterior then follows from the per-m Wiener
   filter and constrained realizations, conditioned on posterior beam samples.
4. Keep as diagnostics: the Fisher matrix in β, mode correlations and whitening, and the adjoint
   dot test.

### 2.4 What croissant `main` changes for us

Read from `src/croissant/{simulator,beam,sky,rotations}.py` at `01d26cb`:

| topic | croissant v4 (notebooks) | croissant `main` | consequence for us |
|---|---|---|---|
| Earth simulation frame | fk5 (J2000) | CIRS at `times_jd[0]` (`Simulator.et_ref`); sky via `Sky.compute_alm_eq(world, et)` | sky and beam alm must share `et_ref` |
| sidereal day (Earth) | 86164.2 s (rounded) | 86400/1.00273781191135448 s (Earth rotation angle) | none; we take it from `croissant.constants` |
| normalization | integral of the horizon-masked beam | **integral over the full sphere** (`Beam.compute_norm`), plus a ground term f_gnd·T_gnd | measurement equation gains a known offset (m=0 only); still invariant to B → cB |
| horizon | hard `npix//2` cut | fractional weights (`horizon`, `horizon_theta`, topocentric or beam frame) | horizon applied after the pointing rotation |
| tilted pointings | healpy `Rotator` on pixel maps | `Beam(beam_tilt=…)` is **not implemented**; use `rotations.rotate_to_grid(alm, euler)` → horizon weights → SHT → `Simulator.topo_to_sim` | our `instrument.py` builds per-pointing beams with these primitives |
| time integration | snapshots | optional windows (`times_end`): sinc taper per m | still diagonal in m; the m-space formulas carry over with the taper |
| `convolve`, `rot_alm_z` sign | conj(sky)·e^{-imφ}·beam | unchanged | the review findings stand |
| SHT | s2fft | s2fft fork (`slosar/s2fft@cefdf46`) + "kernel"/"dense" engines | install from git; Python 3.11–3.13 |

## 3. Proposed layout

```
pyproject.toml              # deps: croissant @ git main, numpyro, healpy; optional extra [gsm] = pygdsm
README.md                   # what this is, install, data, quickstart, layout
LOGBOOK.md                  # dated progress log + current state
docs/
  plan.md                   # this file
  math.md                   # mathematical foundation (deliverable)
src/eigsep_mapmaking/
  __init__.py
  alm.py                    # real packing <-> s2fft (L, 2L-1) layout, ell/m index helpers, C_l
  instrument.py             # observation config; per-pointing beam alm in the sim frame via croissant
                            #   (rotate_to_grid -> horizon -> SHT -> topo_to_sim), norms, f_gnd, phases
  sky.py                    # GSM via pygdsm (optional), Sky -> alm in the sim frame, sky priors S
  forward.py                # time-domain operator + adjoint; m-space transform and response
  noise.py                  # radiometer noise, noise realizations, stationary approximation
  simulate.py               # build a mock dataset from a config dataclass (+ seed)
  mapmaking.py              # Wiener filter: CG (dual form) and exact per-m; constrained realizations
  perturbations.py          # beam-error bases (SVD w/o scale direction; low-ell), injection
  fitting.py                # likelihoods (fixed-sky; sky-marginalized m-space), NUTS, Fisher
  diagnostics.py            # normalized beam error, per-ell map error, adjoint test, mode overlap
tests/                      # pytest, small nside, synthetic beams/skies -> no data files needed
scripts/
  review/                   # the checks behind section 2
  make_gsm.py               # writes data/gsm_<nside>.npz with pygdsm (needs network, see 8.2)
examples/
  simulate_map_and_fit.py   # end-to-end on synthetic data, or on data/ files when present
data/                       # gitignored; see README "Data"
notebooks/archive/          # the five original notebooks, unmodified, with outputs
```

## 4. Module APIs (sketch)

Shapes: `(B, F, L, 2L-1)` for beam alms (pointing, frequency), `(F, L, 2L-1)` for sky alms, and
`(F, T, B)` for data.

- `alm`: `pack(alm) -> x`, `unpack(x, lmax) -> alm` (vectorized, any leading axes),
  `n_real(lmax)`, `ell_of_packed(lmax)`, `m_of_packed(lmax)`, `cl_from_alm`.
- `instrument`: `Observation` (site, `times_jd`, optional `times_end_jd`, freqs, lmax, world,
  `Tgnd`, horizon spec, pointings as ZYZ Euler angles). `beam_alm_sim(beam_maps, obs) ->
  (alm (B,F,L,M), norm (B,F), fgnd (B,F))` and `phases(obs)`, wrapping `croissant.rotations`,
  `croissant.Beam` and `croissant.Simulator`. This module and `sky` are the only ones that import
  croissant.
- `sky`: `gsm_maps(freqs_mhz, nside, model="gsm16")` (pygdsm), `load_sky(path)`,
  `sky_alm_sim(maps, obs)` (via `croissant.Sky.compute_alm_eq(et=obs.et_ref)`), and `SkyPrior`
  (diagonal S in packed space: `oracle(true_alm)`, `power_law(A, α, β, ν₀)`).
- `forward`: `ForwardModel` (an equinox module holding beam alm, norm, f_gnd, `Tgnd`, phases)
  with `.apply(x) -> y`, `.adjoint(y) -> x` and `.ground() -> offset`. Plus `to_mspace(y) ->
  Y[f, m, b]` and `mspace_response(fm) -> R[f, m, b, ℓ]`. It is a pytree, so it passes through
  `jit` without recompiling.
- `noise`: `radiometer_sigma(T_ant, df_hz, dt_s)`, `realize(sigma, key)`.
- `mapmaking`: `wiener_cg(fm, prior, sigma, y, tol, maxiter)`, `wiener_mspace(fm, prior, sigma, y)`,
  `constrained_realization(..., key)`, `clip_positive_for_display(...)`.
- `perturbations`: `BeamBasis` (modes `(K, F, Npix)` in the topocentric beam frame plus a prior σ),
  `svd_basis(beam_maps, k, remove_scale=True)` and `lowl_basis(beam_maps, lmax_pert, n_freq_funcs)`.
  The beam is linear in β, so the per-pointing mode alms, norms and above-horizon integrals are
  precomputed once and the perturbed forward model is a cheap ratio of linear functions of β
  (no SHT inside the likelihood). Also `inject_error(...)` for simulations.
- `fitting`: `loglike_fixed_sky(β, ...)`, `loglike_marginal_mspace(β, ...)`, `numpyro_model(...)`,
  `run_nuts(...)`, `fisher(β, ...)`, `whiten(basis, fisher)`.
- `diagnostics`: `normalized_beam_error`, `map_error_per_ell`, `adjoint_test`, `mode_overlap`.

## 5. Tests (all run in seconds at nside 4–8, no data files)

- alm: pack/unpack round trip; orthonormality; prior variance = C_ℓ for every packed entry.
- instrument/forward: `ForwardModel.apply` matches `croissant.Simulator.sim` (including the
  ground term) for the same beam and sky; the adjoint dot test; invariance under B → cB;
  time-domain vs m-space agreement, with and without integration windows.
- mapmaking: CG Wiener = dense Wiener = per-m Wiener; constrained-realization mean/covariance on
  a tiny problem.
- perturbations: the SVD basis is orthogonal to the scale direction; the precomputed linear
  forward model equals a full recompute through croissant.
- fitting: m-space marginal likelihood = dense marginal likelihood; NUTS recovers an in-basis
  injected error within its posterior on a small noisy simulation; Fisher matches the Hessian.

## 6. `docs/math.md` outline

1. Sky and beam on the sphere; real-field spherical harmonics; the s2fft layout; real packing.
2. The measurement equation with horizon and ground term; frames (galactic → CIRS at t₀,
   topocentric → CIRS at t₀); sky rotation as phases e^{-imφ(t)}; integration windows.
3. Fourier structure of a drift scan: per-m decoupling, identifiability counting, what more
   pointings and frequencies buy.
4. Linear-Gaussian inference for the sky: prior S from C_ℓ; Wiener filter (primal/dual), CG;
   constrained realizations; noise from the radiometer equation.
5. Beam errors: parameterization, the normalization null direction, sky–beam degeneracies.
6. Joint inference: profile (fixed sky / ICM) vs. marginal vs. Gibbs; the m-space marginal
   likelihood; Fisher information and whitening; NUTS.
7. What the simulations can and cannot validate; known limitations.

## 7. Implementation sequence (one or more commits each; logbook updated each time)

1. Plan + logbook + review scripts ✅
2. Packaging skeleton: `pyproject.toml`, `src/` layout, `.gitignore` (incl. `data/`), move
   notebooks to `notebooks/archive/`
3. `alm.py` + tests
4. `instrument.py`, `sky.py` + tests (incl. agreement with `croissant.Simulator`)
5. `forward.py` (time domain + m-space) + tests
6. `noise.py`, `simulate.py` + tests
7. `mapmaking.py` + tests
8. `perturbations.py` + tests
9. `fitting.py`, `diagnostics.py` + tests (incl. synthetic recovery test)
10. `docs/math.md`
11. README, `scripts/make_gsm.py`, example script, final logbook update
12. Push the branch to the fork and open the PR into `pronav5000/eigsep_map_making:main`

## 8. Decisions

### 8.1 Settled

| # | decision | outcome |
|---|---|---|
| D1 | package name | `eigsep_mapmaking`, `src/` layout (default, no objection) |
| D2 | croissant | **croissant `main`** (v5 line), installed from git; Python 3.11–3.13 |
| D3 | old notebooks | move to `notebooks/archive/` unchanged, outputs kept (default) |
| D4 | data | the sky is generated with **pygdsm** at the beam's frequencies, so the GSM/beam channel offset question goes away. The beam is **your current version**, kept outside git in `data/` (see 8.2). Tests use synthetic data only. |
| D5 | conventions | corrected conventions; old notebook numbers will not reproduce; every change listed in the logbook |
| D6 | beam-fitting scope | extraction + bug fixes + scale-aware bases + sky-marginalized m-space likelihood + NUTS, validated by synthetic recovery. Gibbs sampler is a follow-up. |
| D7 | perturbation basis | single beam, so **SVD basis (scale direction removed) + generic low-ℓ basis** behind one `BeamBasis` interface |
| D8 | noise model | stationary radiometer σ per (pointing, freq) as default, σ(t) ∝ T_ant(t) as an option; noise realizations in sims; model floor optional (default) |
| D9 | positivity clip | display-only (default) |
| D10 | GitHub | push to the fork `christianhbye/eigsep_map_making`, PR from the fork into `pronav5000:main` |

### 8.2 Open

| # | question | proposal |
|---|---|---|
| Q1 | **Where to put the beam.** | `data/beam.npz`, gitignored, with keys `bm` `(nfreq, npix)` HEALPix RING *or* `(nfreq, ntheta, nphi)` on an s2fft grid, `freqs` (MHz), and `sampling`/`nside`. If your file uses other keys, tell me what they are and the loader will map them. For cloud sessions the file must be reachable from the container: attach it to a GitHub release on the fork (a fetch script will pull it), or share it on Google Drive. |
| Q2 | Does the beam file cover the full sphere (below the horizon too)? | croissant `main` normalizes by the full-sphere integral and adds f_gnd·T_gnd, so the below-horizon response now matters. |
| Q3 | GSM version | GSM2016 (`pygdsm.GlobalSkyModel16`) at nside 64 in K, as in croissant v4's `Sky.gsm`. Needs `zenodo.org` or `apps.datacentral.org.au` allowed in the cloud environment's network settings (currently blocked). |
| Q4 | Ground model in sims | croissant's default (uniform `Tgnd`=300 K, known in the inversion and subtracted). Alternative: `Tgnd` as a nuisance parameter. |
| Q5 | Default observation | keep the notebooks' setup as the default config (Earth, lat 39.2477, lon −113.4033, start 2025-12-01 09:00 UTC, 500 times over one sidereal day, 20 tilts 0–90° about one horizontal axis, freqs 52–59 MHz). Everything is configurable. |

## 9. Out of scope for this PR (follow-ups)

- A Gibbs sampler for time-domain data (partial days, flags, time-varying noise).
- The Moon case (`croissant_jax.ipynb`) and real EIGSEP data.
- Sampling sky-prior hyperparameters (C_ℓ amplitude, spectral index) jointly with the beam;
  `Tgnd` and the horizon profile as nuisance parameters.
- GPU / memory work for full-resolution, many-frequency runs.
