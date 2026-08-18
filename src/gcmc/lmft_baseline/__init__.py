"""
LMFT Baseline reference package.
Provides exact analytical restructuring potential/field convolutions and
Stillinger-Lovett thermodynamic corrections for liquid-state cDFT without external dependencies.
"""

from .baseline_solver import (
    CdftPicardSolver,
    compute_restructuring_field_1d,
    compute_restructuring_potential_1d,
    stillinger_lovett_corrections,
)

__all__ = [
    "compute_restructuring_potential_1d",
    "compute_restructuring_field_1d",
    "stillinger_lovett_corrections",
    "CdftPicardSolver",
]
