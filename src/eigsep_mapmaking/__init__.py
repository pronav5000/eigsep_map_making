"""Drift-scan mapmaking and beam inference for EIGSEP."""

import jax

# Harmonic transforms and the Wiener filter need double precision; croissant
# warns without it. Enabled here so every entry point gets it.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
