# eigsep_map_making

Drift-scan mapmaking and beam inference for EIGSEP, built on
[croissant](https://github.com/christianhbye/croissant) (`main`).

The package `eigsep_mapmaking` simulates what a set of antenna pointings sees over a sidereal
day, recovers the sky with a Wiener filter, and infers beam errors with the sky marginalized
analytically. For a full sidereal day, the time Fourier transform splits the problem by m, which
makes the Wiener filter and the beam likelihood exact and cheap. The mathematics, with every
formula tied to the code and a test, is in [`docs/math.md`](docs/math.md). The
reorganization plan and review findings are in [`docs/plan.md`](docs/plan.md), and progress is
logged in [`LOGBOOK.md`](LOGBOOK.md).

## Install

Python 3.11–3.13 (croissant `main` installs an s2fft fork from git):

```bash
uv venv -p 3.13 .venv
uv pip install -p .venv/bin/python -e ".[gsm,dev]"
.venv/bin/python -m pytest            # ~1.5 min, synthetic data only
.venv/bin/python -m pytest -m network # also downloads GSM data via pygdsm
```

## Data

Nothing is in git. `data/` is ignored.

| file | how to get it |
|---|---|
| `data/beam.npz` | `python scripts/fetch_beam.py` (EIGSEP bowtie export, shared by Christian on Google Drive, sha256-checked). Keys: `bm` (200, 130, 258) on an MWSS grid at lmax 128, `freqs` 51–250 MHz in Hz; a full-sphere directivity. Any `.npz` with `bm` (HEALPix `(F, Npix)` or an s2fft grid `(F, Nθ, Nφ)`) and `freqs` (Hz or MHz) works with `instrument.load_beam`. |
| `data/gsm_<nside>.npz` | `python scripts/make_gsm.py --beam data/beam.npz --fmin 52 --fmax 59 --nside 64` (GSM2016 via pygdsm, K, galactic; needs network access to zenodo.org the first time). |

## Quick start

```python
import numpy as np
from eigsep_mapmaking import alm, fitting, instrument, mapmaking, perturbations, simulate, sky

freqs = np.array([55.0, 56.0])
obs = instrument.Observation.default(freqs, lmax=16, n_times=40, n_pointings=20)
band = simulate.synthetic_beam_maps(8, np.arange(50.0, 70.0))   # or instrument.load_beam(...)
beam = band[[5, 6]]
sky_maps = simulate.synthetic_sky_maps(8, freqs, lmax=16)       # or sky.load_sky(...)

data = simulate.simulate(beam, sky_maps, obs)                   # noisy mock data, (F, T, P)
prior = sky.SkyPrior.oracle(alm.unpack(data.x_true))            # or SkyPrior.power_law(...)
x_map = mapmaking.wiener_mspace(data.fm, prior, data.sigma, data.y)

basis = perturbations.svd_basis(band, k=3, select=[5, 6])       # scale direction removed
lin = perturbations.LinearBeamModel.build(beam, basis, obs)
beta = fitting.run_nuts(lin, prior, data.sigma, data.y, basis.sigma)
```

`examples/simulate_map_and_fit.py` does all of this end to end, injecting a beam error first.
Run it with `--data` to use the files above. On the EIGSEP bowtie and GSM2016 (ℓmax 32,
20 pointings, 2 channels, ~2 min on a CPU), NUTS recovers an injected 3-mode beam error to its
Fisher precision, and the normalized beam error falls from 9.4 % to 0.13 %.

## Layout

```
src/eigsep_mapmaking/
  alm.py            real packing <-> s2fft (L, 2L-1) layout, index helpers, C_l
  instrument.py     Observation, pointings, beam response via croissant (rotate_to_grid -> Beam
                    -> topo_to_sim), beam file loader
  sky.py            GSM (pygdsm), sky alm in the simulation frame, SkyPrior
  forward.py        ForwardModel: y = A x + ground; adjoint; m-space transform and response
  noise.py          radiometer noise
  simulate.py       synthetic beams/skies, mock datasets
  mapmaking.py      Wiener filter (CG, dense, exact per-m), posterior variance, constrained
                    realizations, display-only positivity clip
  perturbations.py  beam-error bases (SVD, low-l; scale removed), LinearBeamModel
  fitting.py        sky-marginalized m-space likelihood, Fisher, MAP, NUTS
  diagnostics.py    normalized beam error, per-l map error, adjoint test, rank bound, ...
tests/              pytest, small nside, synthetic data
docs/               math.md (foundation), plan.md (plan and review)
scripts/            fetch_beam.py, make_gsm.py, review/ (checks behind the review, croissant v4)
examples/           simulate_map_and_fit.py
notebooks/archive/  the original notebooks, unmodified
```

## Changes from the notebooks

The notebooks' numbers will not reproduce. These conventions were corrected (details in
`docs/plan.md` §2 and `docs/math.md`):

- **Forward model:** one notebook used Σ s e^{+imφ} b, which is the beam mirrored in azimuth. The
  package uses croissant's Σ s* e^{−imφ} b throughout.
- **Prior:** every packed entry has variance C_ℓ. One notebook halved the m > 0 entries.
- **Noise:** σ = T/√(Δν τ) with Δν in Hz and τ in seconds. The notebooks' σ was ~294× and
  1000× too small. Mock data now include a noise realization.
- **croissant `main`:** CIRS simulation frame, full-sphere normalization plus a ground term
  f_gnd·T_gnd, fractional horizon weights, optional integration windows.
- **Beam errors:** perturbations live in bases with the unobservable scale direction removed.
  The "amp + phase" error, an s2fft storage artefact, is gone. Beams are compared after
  normalization (B/∫B).
- **Beam fitting:** the sky is marginalized analytically instead of fixed to a Wiener or true
  map. The positivity clip is display-only.
