#!/usr/bin/env python3
"""Genera diagnósticos interpretables a partir de los seis modelos finales."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irradiance_kernel.diagnostics import build_diagnostics


if __name__ == "__main__":
    manifest = build_diagnostics()
    print(f"Modelos enriquecidos: {len(manifest['models'])}")
    print(ROOT / "artifacts" / "spatial_validation.csv")
    print(ROOT / "artifacts" / "leaderboard_by_discretizer.csv")
