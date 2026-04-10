"""Shared CLI entry for ``qlattice_standalone.py`` and ``tabpfn_standalone.py``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main_for(pipeline_module: str) -> None:
    """Run ``main()`` from ``src.<pipeline_module>`` (e.g. ``qlattice_pipeline``)."""
    train_eval = Path(__file__).resolve().parent.parent
    if str(train_eval) not in sys.path:
        sys.path.insert(0, str(train_eval))
    mod = importlib.import_module(f"src.{pipeline_module}")
    mod.main()
