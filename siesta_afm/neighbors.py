"""Periodic magnetic-neighbor distances and graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import networkx as nx
import numpy as np

from .structure import Structure


@dataclass(slots=True, frozen=True)
class PairDistance:
    i: int
    j: int
    distance: float
    vector: np.ndarray


def minimum_image_vector(structure: Structure, i: int, j: int) -> np.ndarray:
    """Return the shortest cell-image vector from atom ``i`` to ``j``."""

    direct = structure.positions[j] - structure.positions[i]
    if not any(structure.pbc):
        return direct
    if abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError("periodic distances require a nonsingular cell")
    fractional = np.linalg.solve(structure.cell.T, direct)
    # Search around the nearest fractional image.  This also handles input
    # coordinates outside the primary unit cell, while the adjacent-image
    # search remains robust for ordinary non-orthogonal cells.
    choices = []
    for value, periodic in zip(fractional, structure.pbc):
        if periodic:
            center = int(np.rint(-value))
            choices.append((center - 1, center, center + 1))
        else:
            choices.append((0,))
    candidates = [
        direct + np.asarray(shift, dtype=float) @ structure.cell
        for shift in product(*choices)
    ]
    return min(candidates, key=lambda vector: float(np.dot(vector, vector)))


def periodic_self_image_distance(structure: Structure) -> float | None:
    """Return the shortest nonzero periodic cell translation.

    This is the nearest possible distance between an atom and one of its own
    periodic images.  ``None`` denotes a nonperiodic structure.
    """

    periodic_axes = [axis for axis, periodic in enumerate(structure.pbc) if periodic]
    if not periodic_axes:
        return None
    if abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError("periodic distances require a nonsingular cell")
    choices = [(-1, 0, 1) if axis in periodic_axes else (0,) for axis in range(3)]
    vectors = [
        np.asarray(shift, dtype=float) @ structure.cell
        for shift in product(*choices)
        if any(shift)
    ]
    return min(float(np.linalg.norm(vector)) for vector in vectors)


def magnetic_pair_distances(
    structure: Structure, indices: Sequence[int]
) -> list[PairDistance]:
    pairs: list[PairDistance] = []
    for left, i in enumerate(indices):
        for j in indices[left + 1 :]:
            vector = minimum_image_vector(structure, i, j)
            distance = float(np.linalg.norm(vector))
            if distance > 1e-10:
                pairs.append(PairDistance(i, j, distance, vector))
    return sorted(pairs, key=lambda pair: (pair.distance, pair.i, pair.j))


def cross_pair_distances(
    structure: Structure,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
) -> list[PairDistance]:
    """Return minimum-image distances between two atom-index groups."""

    pairs: list[PairDistance] = []
    seen: set[tuple[int, int]] = set()
    for i in left_indices:
        for j in right_indices:
            if i == j or (i, j) in seen:
                continue
            seen.add((i, j))
            vector = minimum_image_vector(structure, i, j)
            distance = float(np.linalg.norm(vector))
            if distance > 1e-10:
                pairs.append(PairDistance(i, j, distance, vector))
    return sorted(pairs, key=lambda pair: (pair.distance, pair.i, pair.j))


def distance_shells(
    pairs: Sequence[PairDistance], tolerance: float = 0.05
) -> list[tuple[float, list[PairDistance]]]:
    """Cluster pair distances into coordination shells."""

    shells: list[list[PairDistance]] = []
    for pair in sorted(pairs, key=lambda item: item.distance):
        if (
            not shells
            or pair.distance - float(np.mean([item.distance for item in shells[-1]]))
            > tolerance
        ):
            shells.append([pair])
        else:
            shells[-1].append(pair)
    return [
        (float(np.mean([item.distance for item in shell])), shell) for shell in shells
    ]


def automatic_cutoff(
    pairs: Sequence[PairDistance], *, shell: int = 1, tolerance: float = 0.05
) -> float:
    shells = distance_shells(pairs, tolerance=tolerance)
    if shell < 1 or shell > len(shells):
        raise ValueError(
            f"neighbor shell {shell} unavailable; found {len(shells)} distance shells"
        )
    target_pairs = shells[shell - 1][1]
    upper = max(pair.distance for pair in target_pairs)
    if shell < len(shells):
        next_lower = min(pair.distance for pair in shells[shell][1])
        return (upper + next_lower) / 2.0
    return upper * 1.05 + 1e-6


def resolve_cutoff(
    pairs: Sequence[PairDistance], cutoff: str | float | None, neighbor_shell: int = 1
) -> float:
    if cutoff is None or str(cutoff).lower() == "auto":
        return automatic_cutoff(pairs, shell=neighbor_shell)
    value = float(cutoff)
    if value <= 0:
        raise ValueError("neighbor cutoff must be positive")
    return value


def build_neighbor_graph(
    structure: Structure,
    indices: Sequence[int],
    cutoff: str | float | None = "auto",
    *,
    neighbor_shell: int = 1,
) -> tuple[nx.Graph, float, list[PairDistance]]:
    pairs = magnetic_pair_distances(structure, indices)
    if len(indices) > 1 and not pairs:
        raise ValueError("no finite magnetic-atom pair distances")
    if pairs:
        resolved = resolve_cutoff(pairs, cutoff, neighbor_shell)
    elif cutoff is None or str(cutoff).lower() == "auto":
        resolved = 0.0
    else:
        resolved = resolve_cutoff((), cutoff, neighbor_shell)
    graph = nx.Graph()
    graph.add_nodes_from(indices)
    for pair in pairs:
        if pair.distance <= resolved + 1e-9:
            graph.add_edge(pair.i, pair.j, distance=pair.distance)
    return graph, resolved, pairs


def shell_summary(
    pairs: Sequence[PairDistance], tolerance: float = 0.05, limit: int = 6
) -> list[dict[str, float | int]]:
    return [
        {"distance": distance, "pairs": len(items)}
        for distance, items in distance_shells(pairs, tolerance)[:limit]
    ]
