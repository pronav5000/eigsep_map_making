# Reorganization plan: notebooks → `eigsep_mapmaking` package

**Status: DRAFT for review. Nothing in §3 onwards is implemented yet.**
Decisions that need your input are collected in §8. Once those are settled, the
implementation proceeds in the commit sequence in §7. Progress is tracked in
[`LOGBOOK.md`](../LOGBOOK.md).

Deliverable for this session: a branch `refactor/modules` with a PR into `main`
that extracts the mapmaking, beam perturbation and MCMC beam fitting into tested
Python modules, plus a README and `docs/math.md`, a document setting out the
mathematical foundation.

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

Environment note: the notebooks use `croissant.jax`, which exists only in croissant
v4.x (git tags `v4.0.0`–`v4.2.1`; PyPI stops at 3.0.0 and v5 removed `croissant.jax`).
v4.2.1 requires Python < 3.13.

## 2. Review findings

Each claim below was checked numerically on small synthetic problems:
[`scripts/review/check_notebook_claims.py`](../scripts/review/check_notebook_claims.py) (sections A–F) and
[`scripts/review/check_mspace_decomposition.py`](../scripts/review/check_mspace_decomposition.py).

### 2.1 Mapmaking: I agree it is mostly sound

Correct as written:
- The measurement equation V_b(t) = ∫B_b T(t) dΩ / ∫B_b dΩ, evaluated in harmonic space. In
  `wiener_eigsep_beams_fast` and `mcmc_beam_and_sky_fresh` the `einsum` matches
  `croissant.jax.simulator.convolve` to 1e-15 (check A).
- The real-alm packing (m=0 real, then √2·Re, √2·Im for m>0). It is an orthonormal real basis:
  x·y = Σ_lm b_lm t*_lm (check B). The vectorized index version is clean.
- The adjoint comes from `jax.linear_transpose`, and the dot-product test passes.
- The dual-form Wiener filter (A S Aᵀ + N)α = y, m = S Aᵀα, solved with CG. This is the right
  choice when N_data < N_sky.

Issues to fix while extracting:
1. **Forward model regression in `simultaneous_beam_and_sky_fixed`.** It uses
   `einsum(sky, conj(phases), beam)` where the other notebooks use `einsum(conj(sky), phases, beam)`.
   The two are not equivalent: the "fixed" version is exactly croissant's visibility for the beam
   **mirrored in azimuth** (φ → −φ), see check A. It goes unnoticed in that notebook because the
   mock data are made with the same (wrong) operator.
2. **Prior variance halved for m>0 in `simultaneous_beam_and_sky_fixed`.** With the √2 packing,
   every packed entry of an isotropic field has variance C_ℓ (check B). The original
   `mcmc_beam_and_sky_fresh` had this right; the later "fix" (`/2.0`) is wrong.
3. **Noise σ is wrong.** In the beam notebooks, `times_jd = jd0 + arange(N)*dt_sim` adds
   seconds to days, so `dt` comes out 86400× too large and σ ~294× too small (check E). In the
   Wiener notebooks, `df = 1e6*(freq[1]-freq[0])` with `freq` already in Hz makes σ 1000× too
   small. Together these explain why the thermal noise was negligible and a "1.5% model
   uncertainty floor" ended up dominating `Ndiag`.
4. **Simulated data are noiseless.** `y_true` never gets a noise realization, so posterior widths
   and coverage cannot be tested.
5. **The prior uses the true sky's C_ℓ, floored at 1e-4 of the peak.** That is fine for a
   simulation, but it should be an explicit, documented choice (an "oracle prior") with a
   non-oracle option (e.g., a power-law C_ℓ or a C_ℓ from another sky model).
6. **The positivity clip followed by a least-squares γ rescale is not a linear estimator.** It is
   fine for display, but it is currently fed into the beam fit (see 2.3).
7. **The information content is fundamentally limited, and "more times" will not help.** For a
   uniformly sampled sidereal day, the time FFT separates the data by m:
   V_{b,m} = Σ_ℓ b_{b,ℓm} t*_{ℓm}. Each m therefore gives N_beam numbers to constrain the
   (ℓmax+1−|m|) sky coefficients at that m, so the rank is ≤ Σ_m min(N_beam, ℓmax+1−|m|). For 20
   pointings at ℓmax=128 that is **≤ 4,760 of 16,641 sky modes (≤ 29%) per frequency**, however
   many time samples there are beyond 2ℓmax+1 (check F confirms the bound exactly). The 32–36% map
   error seen with a *perfect* beam is mostly this null space being filled in by the prior; it is
   not a solver failure.
8. Small issues: the `rel err` print takes `sqrt` of a norm ratio; the horizon mask
   `[npix//2:] = 0` is right except that it keeps half of the equatorial ring (check D);
   `croissant.constants.sidereal_day["earth"] = 86164.2` s (rounded, which is fine as long as it
   is used consistently).

**Consequence of 7 for this PR:** the same m-decomposition gives an *exact* per-m Wiener filter
using N_beam×N_beam solves (verified against the dense solution to 1e-14). It is a fast, exact
reference to test the CG solver against, and it is the basis of the beam-fitting proposal below.
It needs uniform full-day time sampling and noise that is stationary in time. The CG path stays
for everything else (flags, partial days, time-varying noise).

### 2.2 Beam perturbation: this is where most of the problems are

1. **The injected error is mostly unobservable by construction.** The visibilities are
   normalized by ∫B, so V is invariant to B → cB. The 10% amplitude part of the
   perturbation is therefore exactly invisible. `total_power` is *linear* in B (Re a₀₀/Y₀₀), so the
   "power-conserving" rescale should be a ratio, not `sqrt`. With `sqrt`, P_pert/P_true = 1.045
   (check C, and it matches the notebook printout of 1.0468).
2. **The "5° phase" is not a physical perturbation.** Multiplying every s2fft coefficient by
   e^{iφ₀} gives m=0 an imaginary part, and with `reality=True` the m<0 half is simply ignored
   on synthesis (check C), so what the map ends up with depends on s2fft internals. It is
   neither a rotation nor a smooth physical distortion, and it is identical at every frequency.
3. **The fitting basis is the SVD of the beam itself, so its leading directions ≈ the beam ≈
   the unobservable scale direction.** The "oracle" coefficients `p ≈ −0.047·(1,1,1,1,1)` amount to
   "subtract ~5% of the beam", which is mostly the invisible normalization. On a test beam, 43%
   of the injected error power lies along B itself (check C). This explains the notebook's
   finding #5 (near-degenerate modes 0–2, condition number ~5,400) and finding #7 (a confident fit
   "anti-aligned" with the oracle). The oracle is not identifiable, so a correct fit cannot be
   expected to find it.
4. **The beam-error metric compares unnormalized beams.** The pre/post "5.63%" pixel RMSE mostly
   measures the invisible scale. The metric should compare B/∫B (on our test beam, 6.4% raw drops
   to 4.6% observable).
5. All 20 pointings are tilts about one axis (`az = 0`). The `rotate_bowtie` Euler convention
   (healpy `rot=(lon, lat, psi)`, el applied as ψ) should be confirmed and documented.

### 2.3 Beam fitting / MCMC

1. **Fixed-sky fits are biased.** Both NUTS runs condition on a *Wiener* sky (shrunk toward the
   prior mean, then clipped and rescaled), or on the true sky (the "cheat"). Beam parameters then
   absorb the Wiener shrinkage. The alternating LBFGS loop (iterated conditional modes) has the
   same problem, plus no uncertainty.
2. `mcmc_beam_and_sky_fresh` normalizes with the stale `norms_PERTURBED` (fixed in the later
   notebook) and uses a Student-t likelihood to "forgive" the ℓ>10 truncation. Both are
   symptoms of 1.
3. **Proposed replacement: marginalize the sky analytically.** The sky enters linearly with a
   Gaussian prior, so p(y | β) = N(y; 0, A(β) S A(β)ᵀ + N). Per m this becomes a sum of
   N_beam×N_beam (20×20) complex Gaussian log-densities, which are cheap, exact and
   differentiable, so NUTS can sample β directly with no sky estimate at all. I verified it
   against the dense time-domain marginal likelihood to machine precision
   (`check_mspace_decomposition.py`). The sky posterior then follows from the per-m Wiener filter
   and constrained realizations, conditioned on posterior beam samples.
4. Fit diagnostics worth keeping as library functions: the Fisher matrix in β, mode
   correlations and whitening (the notebook's fix #5 idea is sound), and the adjoint dot test.

## 3. Proposed layout

```
pyproject.toml              # package metadata, deps, pytest config
README.md                   # what this is, install, quickstart, layout
LOGBOOK.md                  # dated progress log + current state (for picking up later)
docs/
  plan.md                   # this file
  math.md                   # mathematical foundation (deliverable)
src/eigsep_mapmaking/
  __init__.py
  alm.py                    # s2fft (L, 2L-1) layout helpers, real packing, healpy<->s2fft, C_l
  frames.py                 # only place that touches croissant: rotations, phases, constants
  sky.py                    # load/regrid sky maps, gal->eq alm, sky prior S (oracle / power law)
  beam.py                   # load beams, pointings, horizon mask, topo->eq alm, normalization
  forward.py                # time-domain operator + adjoint; m-space (Fourier) operator
  noise.py                  # radiometer noise, noise realizations, stationary approximation
  simulate.py               # build a mock dataset from a config dataclass (+ seed)
  mapmaking.py              # Wiener filter: CG (dual form) and exact per-m; constrained realizations
  perturbations.py          # beam-error bases, null-mode (scale) projection, error injection
  fitting.py                # likelihoods (fixed-sky; sky-marginalized m-space), NUTS wrapper, Fisher
  diagnostics.py            # metrics (normalized beam error, per-l map error), adjoint test
tests/                      # pytest, small nside, synthetic beams/skies -> no data files needed
scripts/review/             # the checks behind §2 (kept for reproducibility)
examples/
  simulate_map_and_fit.py   # end-to-end on synthetic data; real-data paths via CLI args
notebooks/archive/          # the five original notebooks, unmodified, with outputs
```

## 4. Module APIs (sketch)

Shapes follow the notebooks: `(B, F, L, 2L-1)` for beam alms (pointing, frequency), `(F, L, 2L-1)`
for sky alms, and `(F, T, B)` for data.

- `alm`: `pack(alm) -> x`, `unpack(x, lmax) -> alm` (vectorized, any leading axes),
  `n_real(lmax)`, `ell_of_packed(lmax)`, `healpy_to_s2fft` / `s2fft_to_healpy`, `cl_from_alm`.
- `frames`: `sky_rotation_phases(lmax, times | (ntimes, dt), world)`,
  `rotate_alm(alm, from_frame, to_frame, lmax)`, `beam_normalization(beam_alm)` (= ∫B).
- `sky`: `load_sky_maps(path, freq_slice, nside)`, `sky_alm_equatorial(maps, lmax)`,
  `SkyPrior` (diagonal S in packed space; `from_true_sky` = oracle, `power_law(A, α, β, ν₀)`).
- `beam`: `load_beam(path)`, `pointing_rotations(beam_maps, pointings)`, `apply_horizon(maps)`,
  `beam_alm_equatorial(...)`, `normalize(beam_alm)`.
- `forward`: a `ForwardModel` dataclass (beam alms, phases, lmax) with `.apply(x)`, `.adjoint(y)`
  (pytree-friendly so it can be passed into `jit` without recompiling), and
  `to_mspace(y) -> Y[f, m, b]` / `mspace_response(beam_alm) -> b[f, m, b, ℓ]`.
- `noise`: `radiometer_sigma(T_ant, df_hz, dt_s)`, `realize(sigma, key)`.
- `mapmaking`: `wiener_cg(fm, S, N, y, tol, maxiter)`, `wiener_mspace(...)`,
  `constrained_realization(..., key)`, `clip_positive_for_display(...)`.
- `perturbations`: `BeamBasis` (modes `(K, F, L, 2L-1)` in the topocentric frame plus a prior σ),
  `svd_basis(beam_maps, k, project_out_scale=True)`, `lowl_basis(lmax_pert, freq_funcs)`,
  `apply(basis, β) -> beam`, `inject_error(beam, basis | other_beam, ...)`.
- `fitting`: `loglike_fixed_sky(β, ...)`, `loglike_marginal_mspace(β, ...)`, `numpyro_model(...)`,
  `run_nuts(...)`, `fisher(β, ...)`, `whiten(basis, fisher)`.
- `diagnostics`: `normalized_beam_error`, `map_error_per_ell`, `adjoint_test`, `mode_overlap`.

## 5. Tests (all run in seconds at nside 4–8, no data files)

- alm: pack/unpack round trip; orthonormality; prior variance = C_ℓ for every packed entry.
- forward: matches `croissant.jax.simulator.convolve`; adjoint dot test; invariance under B → cB;
  time-domain vs m-space agreement.
- mapmaking: CG Wiener = dense Wiener = per-m Wiener; constrained-realization mean/covariance on
  a tiny problem.
- perturbations: the projected basis is orthogonal to the scale direction; injection round trip.
- fitting: m-space marginal likelihood = dense marginal likelihood; NUTS recovers an in-basis
  injected error within its posterior on a small noisy simulation; Fisher matches the Hessian.

## 6. `docs/math.md` outline

1. Sky and beam on the sphere; real-field spherical harmonics; the s2fft layout; real packing.
2. The measurement equation; frames (galactic → equatorial, topocentric → equatorial); sky
   rotation as phases e^{-imφ(t)}; normalization by ∫B.
3. Fourier structure of a drift scan: per-m decoupling, identifiability counting, what more
   pointings/frequencies buy.
4. Linear-Gaussian inference for the sky: prior S from C_ℓ; Wiener filter (primal/dual), CG;
   constrained realizations; noise from the radiometer equation.
5. Beam errors: parameterization, the normalization null direction, sky–beam degeneracies.
6. Joint inference: profile (fixed sky / ICM) vs. marginal vs. Gibbs; the m-space marginal
   likelihood; Fisher information and whitening; NUTS.
7. What the simulations can and cannot validate; known limitations.

## 7. Implementation sequence (one or more commits each; logbook updated each time)

1. Plan + logbook + review scripts ← *this commit*
2. Packaging skeleton: `pyproject.toml`, `src/` layout, `.gitignore`, move notebooks to `notebooks/archive/`
3. `alm.py` + tests
4. `frames.py`, `sky.py`, `beam.py` + tests
5. `forward.py` (time domain + m-space) + tests
6. `noise.py`, `simulate.py` + tests
7. `mapmaking.py` + tests
8. `perturbations.py` + tests
9. `fitting.py`, `diagnostics.py` + tests (incl. synthetic recovery test)
10. `docs/math.md`
11. README, example script, final logbook update
12. Push branch, open PR into `main`

## 8. Decisions

| # | decision | options | recommendation |
|---|---|---|---|
| D1 | package name | `eigsep_mapmaking`, other | `eigsep_mapmaking` (`src/` layout) |
| D2 | croissant dependency | (a) pin croissant v4.2.1 from git, Python 3.10–3.12; (b) port to croissant v5 now; (c) vendor the ~30 lines we use | **(a)**, isolated in `frames.py` so moving to v5 later is a one-file change |
| D3 | old notebooks | (a) move to `notebooks/archive/` unchanged; (b) strip outputs; (c) delete | **(a)**: the outputs are the only record of past results |
| D4 | data files | where do they live / come from? | loaders take paths; `data/` gitignored; tests synthetic. *Please tell me the source/version of the beam and GSM files for the README*, and whether the GSM/beam channel offset (notebook fix #8: `gsm[i]` ↔ `beam[i+1]`) is right. |
| D5 | conventions | (a) modules use corrected conventions (croissant sign, C_ℓ prior, fixed dt/df), so old notebook numbers won't reproduce; (b) reproduce notebooks exactly | **(a)**, with each change listed in the logbook |
| D6 | beam-fitting scope in this PR | (a) extract + fix bugs only; (b) (a) + scale-aware bases + sky-marginalized m-space likelihood + NUTS, validated on synthetic recovery; (c) (b) + full Gibbs sampler (time domain) | **(b)**; (c) as a follow-up |
| D7 | perturbation basis | (a) beam-SVD with scale projected out; (b) generic low-ℓ topocentric alm × smooth frequency functions; (c) PCA of differences between beam simulations | ship **(a)+(b)** behind one `BeamBasis` interface; (c) is the physically best choice if you have several beam sims (is there a `bowtie_v001`…?) |
| D8 | noise model | radiometer σ per (pointing, freq) from the time-averaged T_ant (stationary → m-space exact) vs σ(t) ∝ T_ant(t) | both; stationary as default; noise realizations added to sims; "model floor" kept only as an option |
| D9 | positivity clip + γ rescale | keep in inference / display-only | display-only |
| D10 | GitHub access | pushes from this session are currently refused | install the Claude GitHub App on `pronav5000/eigsep_map_making` (or reconnect GitHub in claude.ai settings), otherwise I'll leave the branch committed locally |

## 9. Out of scope for this PR (follow-ups)

- A Gibbs sampler for time-domain data (partial days, flags, time-varying noise).
- Moving to croissant v5; the Moon (`croissant_jax.ipynb` case); real EIGSEP data.
- Beam-error bases from EM-simulation ensembles; sampling sky-prior hyperparameters (C_ℓ amplitude,
  spectral index) jointly with the beam.
- GPU / memory work for full-resolution, many-frequency runs.
