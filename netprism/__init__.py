"""NetPrism — see every layer of a deep network in 3D.

The public entry point is the ``netprism`` CLI::

    python -m netprism view ./some_models --open

Parsing lives in :mod:`netprism.core`, the HTTP/UI server in
:mod:`netprism.server` and the data contract in :mod:`netprism.schema`.
"""

from __future__ import annotations

from . import schema
from .schema import GENERATOR_NAME, GENERATOR_VERSION, SCHEMA_VERSION

__all__ = [
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "schema",
    "__version__",
]

__version__ = GENERATOR_VERSION
