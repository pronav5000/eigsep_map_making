# Logbook

Running log of the reorganization from notebooks into the `eigsep_mapmaking` package.
The newest entry goes at the top of "Entries". "Current state" is rewritten each time so that
anyone picking this up can start there.

## Current state

- **Branch:** `refactor/modules` on the fork `christianhbye/eigsep_map_making` (pushed). The PR
  will go from the fork into `pronav5000/eigsep_map_making:main`.
- **Phase:** implementing plan §7. Done: steps 2–8. Next: step 9 (`fitting.py`, `diagnostics.py`).
  D1–D10 settled (plan §8.1); Q3–Q5 use the proposals as working defaults.
- **Beam (Q1/Q2 answered by the file):** `data/beam.npz` (gitignored), fetched from Christian's
  Google Drive link (see README "Data"). Bowtie `bm` (200, 130, 258) on MWSS at lmax 128,
  51–250 MHz (`freqs` in Hz), directivity (∫ = 4π), full sphere. It is close to up/down
  symmetric (free space?): 51–57 % of the power is below θ = 90°, so with croissant `main`'s
  ground model f_gnd ≈ 0.5. **Open: whether that is intended** (asked Christian).
- **Package:** `alm`, `instrument` (+ `load_beam`), `sky`, `forward` (`ForwardModel`: apply,
  adjoint, ground, m-space transform/response), `noise` (radiometer σ, stationary or σ(t), realizations), `simulate` (synthetic beam/sky, `simulate` → `Dataset`), `mapmaking` (Wiener CG / dense / per-m, posterior variance, constrained realizations, display clip), `perturbations` (`BeamBasis`: SVD and low-ℓ bases with the scale direction removed; `LinearBeamModel`).
- **Tests:** `pytest tests` — 43 pass in ~55 s (mostly JIT compilation); `pytest -m network`
  runs the pygdsm download test.
- **Notebooks:** moved unchanged (outputs kept) to `notebooks/archive/`.
- **Reproducible checks:** `scripts/review/*.py` back up the review findings in plan §2.
- **Network:** the cloud policy now allows zenodo.org (pygdsm), GitHub releases, Google Drive.

### Environments that work

The package targets croissant `main` (D2):

```bash
uv venv -p 3.13 .venv            # croissant main supports Python 3.11-3.13
uv pip install -p .venv/bin/python \
    "croissant-sim @ git+https://github.com/christianhbye/croissant.git@main" \
    healpy numpyro pygdsm pytest
```

The review scripts reproduce the *notebooks'* setup, so they need croissant v4.2.1, the last
version with `croissant.jax`. That version needs Python < 3.13:

```bash
uv venv -p 3.12 .venv-v4
uv pip install -p .venv-v4/bin/python "jax[cpu]" s2fft healpy numpyro jaxopt pytest \
    "git+https://github.com/christianhbye/croissant.git@v4.2.1"
.venv-v4/bin/python scripts/review/check_notebook_claims.py
.venv-v4/bin/python scripts/review/check_mspace_decomposition.py
```

Network: in the Claude cloud environment, `pygdsm` cannot download its sky-model data because
`zenodo.org` and `apps.datacentral.org.au` are blocked by the environment's network policy.

### Open questions (see plan §8.2)

Q1 where to put the beam file (and its format), Q2 whether the beam covers the full sphere,
Q3 the GSM version and network access, Q4 the ground model, Q5 the default observation setup.

## Entries

### 2026-10-15: step 8, `perturbations.py`

- `BeamBasis(modes (K, F, *grid), sigma, sampling, names)`: modes orthonormal in
  Σ_f ∫ a_f b_f dΩ (HEALPix or s2fft quadrature), scaled to the beam's RMS, so β_k = 0.05 is a 5 %
  RMS beam error (default prior σ).
- `svd_basis(beam_maps, k, select=...)`: SVD over the *full band* (with one channel, as in the
  notebooks, the SVD has a single mode, the beam itself), modes u_j(f) v_j(n) evaluated at the
  selected channels, scale direction removed per channel, re-orthonormalized. Test: without
  removal the leading mode is ≥ 95 % scale direction (97 % for the synthetic beam; review 2.2.3).
- `lowl_basis(beam_maps, lmax_pert, n_freq_funcs)`: multiplicative B0·Y_ℓm·P_j(ν), 1 ≤ ℓ ≤
  lmax_pert, Legendre in frequency, scale removed. `real_harmonics` on any grid.
- `LinearBeamModel.build(beam, basis, obs)`: responses of B0 and of each mode precomputed;
  `forward_model(β)` is a linear combination (norm and ground fraction stay exact ratios).
  Verified equal to a full recompute through croissant (1e-11), differentiable under `jit`, and a
  "mode" equal to the beam leaves the data unchanged.

### 2026-10-15: step 7, `mapmaking.py`

- `wiener_cg`: dual form (A S Aᵀ + N)α = y − g with N⁻¹ preconditioning; returns x̂ and the
  relative residual. Ground term subtracted first (it is known: f_gnd·Tgnd).
- `wiener_mspace`: exact per-(f, m) Wiener filter with P×P solves; m = 0 handled as real.
  `posterior_var_mspace`: exact posterior variance of every packed entry (the m > 0 posterior is
  circular, so √2 Re and √2 Im each carry Var(t_ℓm)). `mspace_blocks` exposes R, C_ℓ, N_m, C_m.
- `wiener_dense`/`forward_matrix` for small problems; `constrained_realization` (x̂(y) + x_s −
  x̂(A x_s + n_s)) with either solver; `clip_positive_for_display` (no γ rescale; review 2.1.6).
- Tests (noisy data, oracle prior with floor): CG = dense to 1e-7, per-m = dense to 1e-6 (mean)
  and 1e-5 (variance); 400 constrained realizations match the dense posterior mean and variance.

### 2026-10-15: step 6, `noise.py` and `simulate.py`

- `noise`: `radiometer_sigma`, `channel_width_hz` (MHz → Hz once), `integration_time_s` (window
  length, or sample spacing, in seconds from Julian days ×86400 — fixes review 2.1.3),
  `sigma_from_model` (stationary σ per (f, p) by default, σ(t) ∝ T_ant(t) optional, optional
  fractional floor, off by default), `realize`.
- `simulate.simulate` → `Dataset(obs, fm, x_true, y_clean, y, sigma, response)`: noisy mock data
  (review 2.1.4) with σ from the radiometer equation; `noise_scale` for tests.

### 2026-10-15: step 5, `forward.py`; the beam file

- `ForwardModel` (equinox module): `apply` contracts over ℓ first, then over m with the phases
  (P·F·L·M + T·F·P·M instead of T·P·F·L·M); `adjoint` via `jax.linear_transpose` (needed
  `unique_indices=True` in `alm.unpack`); `ground`; `predict`. Built from a `BeamResponse`.
- m-space: `to_mspace` (projection on e^{+imφ_t}), `window_factor` w_m, `mspace_response`
  R[f,m,p,ℓ], `mspace_ground`. Verified Y = R·conj(s) + ground to 1e-8 (limited by Julian-day
  rounding of the sample times, ~40 µs) for snapshots and for integration windows, where
  w_m = e^{−imΔφ/2} sinc(mΔφ/2).
- Tests: `predict` = croissant `convolve`/norm + ground; adjoint dot test (1e-12); invariance
  under B → c_f B including the ground term.
- `instrument.load_beam`: npz with `bm` + `freqs` (Hz or MHz), HEALPix or MWSS, channel
  selection. Christian's bowtie file loads (see Current state); its below-horizon half is an
  open physics question.

### 2026-10-15: step 4, `instrument.py` and `sky.py`

- `Observation` (frozen dataclass): times (+ optional windows), freqs, lmax, site, world, Tgnd,
  ground-fixed horizon, pointings as ZYZ Euler angles; `et_ref`, `phases`, and `frame` (a
  croissant `Simulator` with an empty beam and sky, used for `topo_to_sim` and phases).
  `Observation.default` is plan Q5 (notebook site/start, 500 times over a sidereal day, 20 tilts).
- Pointing convention made explicit: `pointing_matrix(tilt, az, rot)` turns the antenna about the
  zenith (clockwise, croissant's `beam_rot`), then tilts the boresight by `tilt` towards compass
  azimuth `az`. The notebooks' healpy `rot=(0,0,el)` tilts along the N–S line; the default here
  tilts towards North.
- `beam_response`: full-sphere beam alm → `rotate_to_grid` (MWSS) per pointing → croissant
  `Beam` with ground-fixed `horizon_theta` (fractional edge weights) → `compute_alm`,
  `compute_norm` (full sphere), `compute_fgnd` → `topo_to_sim`. Returns `BeamResponse(alm, norm,
  above)`, linear in the beam pattern.
- Verified: V = [conv(alm_h, sky) + (norm − above)·Tgnd]/norm equals `croissant.Simulator.sim`
  to 1e-10 for every pointing, with and without integration windows; sky alm equal
  `precompute_sky_alm`; norm is pointing-independent; zenith fgnd matches the pixel sum; a
  symmetric beam tilted to the horizon has fgnd = 0.5; the response is linear.
- `sky`: `gsm_maps` (pygdsm GSM2016, K RJ, ud_grade to nside), `save_sky`/`load_sky`,
  `sky_alm_sim`, `SkyPrior` (`from_cl`, `oracle` with floor, non-oracle `power_law`, `sample`).
- `simulate`: `synthetic_beam_maps` (chromatic, elliptical, with a back lobe) and
  `synthetic_sky_maps` (positive, spatially varying spectral index).

### 2026-10-15: step 3, `alm.py`

`pack`/`unpack` (any leading axes), `n_real`, `ell_of_packed`, `m_of_packed`, `cl_from_packed`,
`cl_from_alm`, `cl_to_packed`, `reduce_lmax`. Packing order is the notebooks' (m=0 real; then
√2 Re, √2 Im for m>0, ordered by m then l), so old packed vectors stay readable. The package
enables `jax_enable_x64` on import. Tests: round trip, Hermitian symmetry, x·y = Σ a b* and
= ∫fg (HEALPix quadrature, 2%), Var(x_i) = C_ℓ for m=0 and m>0 (4000 draws), C_ℓ = healpy's.

### 2026-10-15: step 2, packaging skeleton

Work continues in a session whose source is the fork; the three planning commits were applied
from a patch series and pushed. Added `pyproject.toml` (src layout, croissant from git `main`,
optional extras `gsm` and `dev`), `src/eigsep_mapmaking/__init__.py`, extended `.gitignore`
(`data/`, build files), and moved the five notebooks to `notebooks/archive/` with `git mv`
(contents and outputs byte-identical). Environment: Python 3.13, croissant `main` resolved to
the s2fft fork `slosar/s2fft@cefdf46`, JAX 0.11.2.

### 2026-10-15: first review of the plan; move to croissant `main`

Decisions from review: croissant `main` (D2); the sky comes from pygdsm and the beam is
Christian's current version (D4); corrected conventions are fine (D5); beam-fitting scope as
proposed (D6); SVD + low-ℓ basis, because there is only one beam (D7). The rest take the
defaults.

Read croissant `main` (`01d26cb`) and recorded what changes for us in plan §2.4:
- the Earth simulation frame is CIRS at `times_jd[0]`;
- normalization is the full-sphere integral, with a ground term f_gnd·T_gnd;
- horizon edges get fractional weights;
- tilted pointings use `rotate_to_grid` + `topo_to_sim`, since `beam_tilt` is not implemented;
- integration windows are a sinc taper per m;
- the `convolve`/`rot_alm_z` conventions are unchanged.

The package design now wraps croissant in `instrument.py`/`sky.py` instead of a `frames.py`.
`pygdsm` installs, but its data download is blocked by the cloud environment's network policy.

Plan approved. Work moves to a new Claude session whose source is the fork, because this
session could not push there (see "Current state").

### 2026-10-15: review of notebooks, plan drafted

Read all five notebooks end to end, checked the croissant v4.2.1 source for the conventions of
`convolve`, `rot_alm_z` and `total_power`, and verified the main review claims numerically on
small synthetic problems (nside 4–8):

- Mapmaking core is sound (forward model, real packing, adjoint, dual-form CG Wiener filter).
- Found in `simultaneous_beam_and_sky_fixed`: the forward model is the azimuthally mirrored beam
  (sign/conjugation regression), and the m>0 prior variance is wrongly halved.
- Noise σ is ~294× (beam notebooks: seconds added to JD) and 1000× (Wiener notebooks: df in
  Hz×1e6) too small, and the mock data carry no noise realization.
- Beam perturbation: the amplitude part is invisible after ∫B normalization, the "phase" part is
  an s2fft-layout artifact, and the SVD fitting basis is dominated by the unobservable scale
  direction, which explains the degeneracies and the "anti-aligned" fits seen in the notebooks.
- Fixed-sky beam fits (Wiener sky or true sky) are biased; the alternating LBFGS is ICM.
- New result: for full-day uniform sampling the problem is block diagonal in m. This gives an
  exact per-m Wiener filter and an exact, cheap, sky-marginalized beam likelihood (N_beam×N_beam
  per m). Verified against dense linear algebra to ~1e-14.
- Identifiability: ≤ Σ_m min(N_beam, ℓmax+1−|m|) sky modes per frequency (≤ 29% for 20
  pointings at ℓmax=128), independent of the number of time samples.

Wrote `docs/plan.md` (layout, APIs, tests, math-doc outline, commit sequence, decisions) and this
logbook. Waiting on review of the plan before implementing.
