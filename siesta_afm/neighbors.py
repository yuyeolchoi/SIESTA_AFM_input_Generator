"""Periodic magnetic-neighbor distances and graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import cos, radians
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


def classify_coordination_geometry(
    vectors: Sequence[Sequence[float] | np.ndarray],
    *,
    trans_angle: float = 170.0,
) -> str:
    """Classify a ligand environment from its coordination vectors.

    The deliberately conservative classifier only assigns a familiar geometry
    when both the coordination number and number of near-linear ligand pairs
    match a supported pattern.  It otherwise returns ``CN=<n>``.
    """

    array = np.asarray(vectors, dtype=float)
    if array.size == 0:
        return "CN=0"
    array = np.reshape(array, (-1, 3))
    coordination = len(array)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        return f"CN={coordination}"
    unit = array / norms[:, None]
    trans_limit = cos(radians(trans_angle))
    trans_pairs = sum(
        float(np.dot(unit[left], unit[right])) <= trans_limit + 1e-12
        for left in range(coordination)
        for right in range(left + 1, coordination)
    )
    labels = {
        (2, 1): "linear",
        (3, 0): "trigonal",
        (4, 0): "Td",
        (4, 2): "square-planar",
        (5, 1): "trigonal-bipyramidal",
        (5, 0): "square-pyramidal",
        (6, 3): "Oh",
    }
    return labels.get((coordination, trans_pairs), f"CN={coordination}")


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
    *,
    all_images: bool = False,
    cutoff: float | None = None,
) -> list[PairDistance]:
    """Return distances between two atom-index groups.

    The default preserves the usual one-minimum-image-per-pair behavior.
    ``all_images=True`` returns every periodic image considered, which is
    required for coordination numbers because distinct images of one basis
    atom can be separate ligands.  With a cutoff, the image search range is
    expanded far enough to include every image inside that radius.
    """

    if cutoff is not None and cutoff <= 0:
        raise ValueError("image-distance cutoff must be positive")

    pairs: list[PairDistance] = []
    seen: set[tuple[int, int]] = set()
    for i in left_indices:
        for j in right_indices:
            if i == j or (i, j) in seen:
                continue
            seen.add((i, j))
            vectors = (
                _periodic_image_vectors(structure, i, j, cutoff=cutoff)
                if all_images
                else [minimum_image_vector(structure, i, j)]
            )
            for vector in vectors:
                distance = float(np.linalg.norm(vector))
                if distance > 1e-10 and (cutoff is None or distance <= cutoff + 1e-9):
                    pairs.append(PairDistance(i, j, distance, vector))
    return sorted(pairs, key=lambda pair: (pair.distance, pair.i, pair.j))


def _periodic_image_vectors(
    structure: Structure,
    i: int,
    j: int,
    *,
    cutoff: float | None,
) -> list[np.ndarray]:
    direct = structure.positions[j] - structure.positions[i]
    if not any(structure.pbc):
        return [direct]
    if abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError("periodic distances require a nonsingular cell")
    fractional = np.linalg.solve(structure.cell.T, direct)
    if cutoff is None:
        radii = (1, 1, 1)
    else:
        # For v = (fractional + shift) @ cell and |v| <= cutoff,
        # each fractional component is bounded by the corresponding reciprocal
        # basis-vector norm.  The extra half covers rounding to the nearest
        # image center.
        inverse = np.linalg.inv(structure.cell)
        bounds = cutoff * np.linalg.norm(inverse, axis=0)
        radii = tuple(max(1, int(np.ceil(bound + 0.5))) for bound in bounds)
    choices: list[Sequence[int]] = []
    for axis, (value, periodic) in enumerate(zip(fractional, structure.pbc)):
        if periodic:
            center = int(np.rint(-value))
            radius = radii[axis]
            choices.append(range(center - radius, center + radius + 1))
        else:
            choices.append((0,))
    return [
        direct + np.asarray(shift, dtype=float) @ structure.cell
        for shift in product(*choices)
    ]


def count_anion_neighbors(
    structure: Structure,
    index: int,
    anion_indices: Sequence[int],
    cutoff: float,
) -> int:
    """Count all anion periodic images within ``cutoff`` of one atom."""

    return len(
        cross_pair_distances(
            structure, [index], anion_indices, all_images=True, cutoff=cutoff
        )
    )


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


def resolve_first_shell(
    pairs: Sequence[PairDistance],
    *,
    min_gap_ratio: float = 0.10,
    max_neighbors: int = 12,
) -> tuple[float, list[PairDistance]]:
    """Resolve a local coordination shell from the first significant gap.

    Relaxed bonds in one physical shell can vary by more than the fixed
    tolerance used by :func:`distance_shells`.  This coordination-specific
    resolver therefore retains consecutive ligands until their next distance
    jump is at least ``min_gap_ratio`` of the current distance.  Only the first
    ``max_neighbors`` possible boundaries are considered so a large gap near
    the edge of the finite periodic-image search cannot swallow several shells.
    If no significant gap exists, the legacy fixed-tolerance first shell is
    retained as a conservative fallback.
    """

    if min_gap_ratio <= 0:
        raise ValueError("first-shell minimum gap ratio must be positive")
    if max_neighbors < 1:
        raise ValueError("first-shell maximum neighbor count must be positive")
    ordered = sorted(pairs, key=lambda item: (item.distance, item.i, item.j))
    if not ordered:
        raise ValueError("cannot resolve a first shell without distances")

    boundary_limit = min(len(ordered) - 1, max_neighbors)
    for boundary in range(1, boundary_limit + 1):
        upper = ordered[boundary - 1].distance
        lower = ordered[boundary].distance
        if (lower - upper) / max(upper, 1e-12) >= min_gap_ratio:
            return (upper + lower) / 2.0, ordered[:boundary]

    first_shell = distance_shells(ordered)[0][1]
    upper = max(pair.distance for pair in first_shell)
    if len(first_shell) < len(ordered):
        lower = ordered[len(first_shell)].distance
        cutoff = (upper + lower) / 2.0
    else:
        cutoff = upper * 1.05 + 1e-6
    return cutoff, list(first_shell)


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
