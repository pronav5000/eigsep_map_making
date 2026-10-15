"""End to end: simulate a drift scan with a beam error, make maps, and fit the beam.

    python examples/simulate_map_and_fit.py                 # synthetic beam and sky (no files)
    python examples/simulate_map_and_fit.py --data          # data/beam.npz + data/gsm_64.npz

Steps: build the observation (plan Q5 defaults, scaled down by the options); build an SVD beam
basis over the band (scale direction removed); inject an in-basis error; simulate noisy data
with the true beam; Wiener-filter maps with the true and the nominal beam (per-m, exact); sample
the beam coefficients with NUTS from the sky-marginalized likelihood; report.
Results are written to ``--out`` (npz).
"""

import argparse
import time

import jax.numpy as jnp
import numpy as np

from eigsep_mapmaking import (alm, diagnostics, fitting, instrument, mapmaking,
                              perturbations as pt, simulate, sky)
from eigsep_mapmaking.forward import ForwardModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", action="store_true", help="use data/beam.npz and data/gsm_<nside>.npz")
    ap.add_argument("--lmax", type=int, default=16)
    ap.add_argument("--nside", type=int, default=8, help="synthetic maps / GSM file nside")
    ap.add_argument("--freqs", type=float, nargs="+", default=[55.0, 56.0])
    ap.add_argument("--pointings", type=int, default=20)
    ap.add_argument("--ntimes", type=int, default=None, help="default 2 lmax + 8")
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--beta-sigmas", type=float, nargs="*", default=[1.5, -1.0, 0.5],
                    help="injected error in units of the prior sigma")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="example_results.npz")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    freqs = np.array(a.freqs)
    if a.data:
        band, band_freqs, sampling = instrument.load_beam("data/beam.npz")
        sky_maps, sky_freqs, coord = sky.load_sky(f"data/gsm_{a.nside}.npz")
        sky_maps = sky_maps[[int(np.argmin(np.abs(sky_freqs - f))) for f in freqs]]
    else:
        band_freqs = np.arange(50.0, 70.0, 1.0)
        sampling = "healpix"
        band = simulate.synthetic_beam_maps(a.nside, band_freqs)
        sky_maps = simulate.synthetic_sky_maps(a.nside, freqs, lmax=min(a.lmax, 3 * a.nside - 1))
    sel = [int(np.argmin(np.abs(band_freqs - f))) for f in freqs]
    B0 = band[sel]

    ntimes = a.ntimes or 2 * a.lmax + 8
    obs = instrument.Observation.default(freqs, a.lmax, n_times=ntimes, n_pointings=a.pointings)
    log(f"observation: {obs.n_freqs} freqs, {obs.n_pointings} pointings, {obs.n_times} times, "
        f"lmax {obs.lmax}; rank bound {diagnostics.rank_bound(obs.n_pointings, obs.lmax)} of "
        f"{alm.n_real(obs.lmax)} sky modes per frequency")

    basis = pt.svd_basis(band, k=a.modes, sampling=sampling, select=sel)
    beta_true = np.array(a.beta_sigmas[: basis.k]) * basis.sigma
    B_true = pt.inject_error(B0, basis, beta_true)
    log(f"basis: {basis.k} SVD modes; injected beta = {beta_true}; normalized beam error "
        f"{diagnostics.normalized_beam_error(B0, B_true, sampling)}")

    lin = pt.LinearBeamModel.build(B0, basis, obs, sampling=sampling)
    log("precomputed responses of the nominal beam and the modes")
    data = simulate.simulate(B_true, sky_maps, obs, beam_sampling=sampling,
                             noise_scale=a.noise_scale, seed=a.seed)
    log(f"simulated data; f_gnd (pointing 0, first freq) = {float(data.response.fgnd[0, 0]):.3f}, "
        f"sigma ~ {float(jnp.mean(data.sigma)):.3g} K")

    prior = sky.SkyPrior.oracle(alm.unpack(data.x_true), floor=1e-4)
    x_true_beam = mapmaking.wiener_mspace(data.fm, prior, data.sigma, data.y)
    x_nominal = mapmaking.wiener_mspace(lin.forward_model(jnp.zeros(basis.k)), prior,
                                        data.sigma, data.y)
    for name, x in [("true beam", x_true_beam), ("nominal beam", x_nominal)]:
        log(f"Wiener map with {name}: relative error {diagnostics.relative_error(x, data.x_true):.3f}")

    samples = fitting.run_nuts(lin, prior, data.sigma, data.y, basis.sigma, key=a.seed,
                               num_warmup=a.samples // 2, num_samples=a.samples)
    mean, sd = np.asarray(samples.mean(0)), np.asarray(samples.std(0))
    log("NUTS beta: " + ", ".join(f"{m:+.4f} ± {s:.4f} (true {t:+.4f})"
                                   for m, s, t in zip(mean, sd, beta_true)))
    F = fitting.fisher(jnp.asarray(beta_true), lin, prior, data.sigma)
    log(f"Fisher errors at the truth: {np.sqrt(np.diag(np.linalg.inv(F)))}")
    B_fit = pt.inject_error(B0, basis, mean)
    x_fit = mapmaking.wiener_mspace(lin.forward_model(jnp.asarray(mean)), prior, data.sigma,
                                    data.y)
    log(f"normalized beam error: nominal {diagnostics.normalized_beam_error(B0, B_true, sampling)}"
        f" -> fitted {diagnostics.normalized_beam_error(B_fit, B_true, sampling)}")
    log(f"Wiener map with fitted beam: relative error "
        f"{diagnostics.relative_error(x_fit, data.x_true):.3f}")
    np.savez(a.out, beta_true=beta_true, samples=np.asarray(samples), fisher=np.asarray(F),
             x_true=np.asarray(data.x_true), x_wiener_true_beam=np.asarray(x_true_beam),
             x_wiener_nominal=np.asarray(x_nominal), x_wiener_fit=np.asarray(x_fit),
             freqs=freqs, lmax=a.lmax)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
