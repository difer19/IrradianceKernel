#!/usr/bin/env python3
"""Punto de entrada para ejecutar la matriz experimental."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from irradiance_kernel.evaluation import generate_configurations, run_all_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="Procesos paralelos")
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Límite de diagnóstico por dataset; omitir para las 648 configuraciones",
    )
    args = parser.parse_args()
    print(f"Configuraciones estructurales por dataset: {len(generate_configurations())}")
    manifest = run_all_experiments(n_jobs=args.jobs, max_configs=args.max_configs)
    print(f"Modelos guardados: {len(manifest['models'])}")
    print(ROOT / "artifacts" / "manifest.json")


if __name__ == "__main__":
    main()
