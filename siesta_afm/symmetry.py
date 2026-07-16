"""Optional crystallographic-symmetry helpers."""

from __future__ import annotations

import numpy as np
from ase.data import atomic_numbers
from ase.geometry import find_mic

from .structure import Structure


def structure_symmetry_permutations(
    structure: Structure, *, symprec: float = 1e-3
) -> list[tuple[int, ...]]:
    """Return unique atom permutations induced by the structure's symmetries.

    Each returned tuple maps a source atom index to its transformed target atom
    index: ``permutation[i] == j`` means atom ``i`` is moved onto atom ``j`` by
    that symmetry operation.
    """

    if not all(structure.pbc) or abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError(
            "symmetry-based deduplication requires a fully periodic, "
            "nonsingular cell"
        )
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError(
            "--symmetry-dedup requires the optional spglib dependency; "
            "install it with pip install -e '.[symmetry]' or pip install spglib"
        ) from exc

    lattice = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.fractional_positions, dtype=float)
    numbers = np.asarray(
        [atomic_numbers[symbol] for symbol in structure.symbols], dtype=int
    )
    symmetry = spglib.get_symmetry(
        (lattice, positions, numbers), symprec=float(symprec)
    )

    identity = tuple(range(len(structure)))
    permutations = [identity]
    seen = {identity}
    if symmetry is None:
        return permutations

    rotations = symmetry.get("rotations", ())
    translations = symmetry.get("translations", ())
    tolerance = float(symprec)
    for rotation, translation in zip(rotations, translations):
        transformed = positions @ np.asarray(rotation, dtype=float).T
        transformed += np.asarray(translation, dtype=float)
        permutation: list[int] = []
        used_targets: set[int] = set()
        for source_index, transformed_position in enumerate(transformed):
            deltas = (positions - transformed_position) @ lattice
            _, distances = find_mic(deltas, cell=lattice, pbc=True)
            candidates = [
                target_index
                for target_index in np.argsort(distances)
                if numbers[target_index] == numbers[source_index]
                and target_index not in used_targets
            ]
            if not candidates or distances[candidates[0]] > tolerance:
                break
            target_index = int(candidates[0])
            permutation.append(target_index)
            used_targets.add(target_index)
        if len(permutation) != len(structure):
            continue
        item = tuple(permutation)
        if item not in seen:
            seen.add(item)
            permutations.append(item)
    return permutations
