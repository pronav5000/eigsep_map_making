import warnings

import eigsep_mapmaking  # noqa: F401  (enables jax x64 before any test runs)

warnings.filterwarnings("ignore", category=DeprecationWarning)
