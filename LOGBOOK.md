# Logbook

Running log of the reorganization from notebooks into the `eigsep_mapmaking` package.
The newest entry goes at the top of "Entries". "Current state" is rewritten each time so that
anyone picking this up can start there.

## Current state

- **Branch:** `refactor/modules` (off `main` @ `433bea7`). Push target is the fork
  `christianhbye/eigsep_map_making`; the PR goes from the fork into `pronav5000/eigsep_map_making:main`.
  Not pushed yet: the first Claude session had `pronav5000/eigsep_map_making` attached (no push
  access) and could not also attach the same-named fork, so work has to continue in a session
  whose source is the fork.
- **Phase:** planning. [`docs/plan.md`](docs/plan.md) is a draft awaiting review; the decisions
  are in §8 there. No package code has been written yet.
- **Code that works today:** only the notebooks in `notebooks/` (see plan §1 for what each one does).
- **Reproducible checks:** `scripts/review/*.py` back up the review findings in plan §2.
- **Data:** none of the data files the notebooks load are in the repo
  (`eigsep_bowtie_v000.npz`, `eigsep_vivaldi.npz`, `gsm_model.npy`, `beam.npy`, `ulsa.npy`).

### Environment that works

```bash
uv venv -p 3.12 .venv            # croissant v4.x requires Python < 3.13
uv pip install -p .venv/bin/python "jax[cpu]" s2fft healpy numpyro jaxopt pytest \
    "git+https://github.com/christianhbye/croissant.git@v4.2.1"
.venv/bin/python scripts/review/check_notebook_claims.py
.venv/bin/python scripts/review/check_mspace_decomposition.py
```

`croissant.jax` exists only in croissant v4.0.0–v4.2.1 (git tags). PyPI stops at 3.0.0, which
has no `croissant.jax`, and v5 removed it.

### Open questions (see plan §8)

D1–D10 in the plan, in particular: the croissant version (D2), the source of the data files and
the channel alignment between GSM and beams (D4), the scope of beam-fitting changes (D6), and
GitHub push access (D10).

## Entries

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
