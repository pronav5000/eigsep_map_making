"""Download the EIGSEP beam file to data/beam.npz.

The file (>30 MB) is kept out of git. Pass a URL to override the default share link.

    python scripts/fetch_beam.py [URL]
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# Christian's Google Drive share of the bowtie beam (bowtie_v002 export; see LOGBOOK.md)
DRIVE_ID = "1cJv8riQWa6X3fhHA2zlnpMd6vV5cvEES"
DEFAULT_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
SHA256 = "ffe974f43b7bde0d591fd291e59696442f1fa367ce07ddf42edc6f42a9986cb3"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out = Path("data/beam.npz")
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if url == DEFAULT_URL and digest != SHA256:
        tmp.unlink()
        sys.exit(f"sha256 mismatch: got {digest}, expected {SHA256}")
    tmp.rename(out)
    print(f"wrote {out} (sha256 {digest})")


if __name__ == "__main__":
    main()
