"""Carga las clases de fkernel directamente desde sus archivos originales."""

from __future__ import annotations

import sys
from pathlib import Path

ORIGINAL_DIR = Path(__file__).resolve().parents[2] / "references" / "original"
if str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

# Estas importaciones apuntan a las copias verificadas byte por byte. No son adaptaciones.
from KANN import KANNC  # noqa: E402,F401
from KSVM import KSVC  # noqa: E402,F401

__all__ = ["KANNC", "KSVC"]
