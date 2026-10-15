"""Write GSM sky maps for the beam's frequencies to data/gsm_<nside>.npz.

Needs pygdsm (pip install -e .[gsm]) and network access to its data hosts the first time
(zenodo.org / apps.datacentral.org.au).

    python scripts/make_gsm.py --freqs 52 53 54 55 56 57 58 59 --nside 64
    python scripts/make_gsm.py --beam data/beam.npz --fmin 52 --fmax 59 --nside 64
"""

import argparse
from pathlib import Path

import numpy as np

from eigsep_mapmaking import instrument, sky


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--freqs", type=float, nargs="*", help="MHz")
    ap.add_argument("--beam", help="take the frequencies from this beam file")
    ap.add_argument("--fmin", type=float, default=-np.inf)
    ap.add_argument("--fmax", type=float, default=np.inf)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--model", default="gsm16", choices=["gsm16", "gsm08"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.beam:
        _, freqs, _ = instrument.load_beam(a.beam)
    elif a.freqs:
        freqs = np.array(a.freqs)
    else:
        ap.error("give --freqs or --beam")
    freqs = freqs[(freqs >= a.fmin) & (freqs <= a.fmax)]
    maps = sky.gsm_maps(freqs, a.nside, model=a.model)
    out = Path(a.out or f"data/gsm_{a.nside}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    sky.save_sky(out, maps, freqs)
    print(f"wrote {out}: {len(freqs)} channels {freqs.min()}-{freqs.max()} MHz, nside {a.nside}")


if __name__ == "__main__":
    main()
