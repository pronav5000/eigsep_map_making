# Logbook

Running log of the reorganization from notebooks into the `eigsep_mapmaking` package.
The newest entry goes at the top of "Entries". "Current state" is rewritten each time so that
anyone picking this up can start there.

## Current state

- **Branch:** `refactor/modules` on the fork `christianhbye/eigsep_map_making` (pushed). The PR
  will go from the fork into `pronav5000/eigsep_map_making:main`.
- **Phase:** implementing plan §7. Done: steps 2–3. Next: step 4 (`instrument.py`, `sky.py`).
  D1–D10 settled (plan §8.1); Q1–Q5 use the proposals as working defaults. Q1/Q2: Christian will
  share the beam file (>30 MB) via a GitHub release on the fork or a public link.
- **Package:** `eigsep_mapmaking.alm` (packing, index helpers, C_l). `pip install -e .` works.
- **Tests:** `pytest tests` — 10 pass in ~9 s.
- **Notebooks:** moved unchanged (outputs kept) to `notebooks/archive/`.
- **Reproducible checks:** `scripts/review/*.py` back up the review findings in plan §2.
- **Data:** none in the repo; `data/` is gitignored. The cloud network policy now allows
  zenodo.org (pygdsm), GitHub release downloads, Google Drive and Dropbox.

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
