"""Order-preserving structure representation used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(slots=True)
class Structure:
    """A small, immutable-in-practice structure container.

    Atom order is never changed.  ``siesta_index`` is therefore always the
    original one-based input position.
    """

    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    pbc: tuple[bool, bool, bool] = (False, False, False)
    species_ids: list[int | None] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        self.symbols = [str(s) for s in self.symbols]
        self.positions = np.asarray(self.positions, dtype=float).reshape((-1, 3))
        self.cell = np.asarray(self.cell, dtype=float).reshape((3, 3))
        self.pbc = tuple(bool(value) for value in self.pbc)  # type: ignore[assignment]
        if len(self.symbols) != len(self.positions):
            raise ValueError("symbols and positions have different lengths")
        if self.species_ids is None:
            self.species_ids = [None] * len(self.symbols)
        if len(self.species_ids) != len(self.symbols):
            raise ValueError("species_ids and symbols have different lengths")

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def fractional_positions(self) -> np.ndarray:
        if abs(float(np.linalg.det(self.cell))) < 1e-12:
            raise ValueError("fractional coordinates require a nonsingular cell")
        return np.linalg.solve(self.cell.T, self.positions.T).T

    @property
    def mapping(self) -> list[dict[str, object]]:
        """Return the required original/ASE/SIESTA atom-index mapping."""

        return [
            {
                "original_index": i,
                "ase_index": i,
                "siesta_index": i + 1,
                "element": symbol,
                "species_id": self.species_ids[i],
                "position": self.positions[i].copy(),
            }
            for i, symbol in enumerate(self.symbols)
        ]

    def with_pbc(self, pbc: Sequence[bool]) -> "Structure":
        return Structure(
            self.symbols.copy(),
            self.positions.copy(),
            self.cell.copy(),
            tuple(pbc),
            list(self.species_ids or []),
            self.source,
        )


def periodic_axes_to_pbc(value: str | Iterable[str] | None) -> tuple[bool, bool, bool]:
    if value is None:
        return False, False, False
    letters = set(value if not isinstance(value, str) else value.lower())
    unknown = letters.difference({"x", "y", "z"})
    if unknown:
        raise ValueError(f"invalid periodic axes: {''.join(sorted(unknown))}")
    return "x" in letters, "y" in letters, "z" in letters
