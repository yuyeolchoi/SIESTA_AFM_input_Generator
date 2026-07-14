"""AFM sign-assignment algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from .neighbors import (
    PairDistance,
    build_neighbor_graph,
    magnetic_pair_distances,
    resolve_cutoff,
)
from .structure import Structure


NON_BIPARTITE_MESSAGE = """The magnetic-neighbor graph is not bipartite.
A unique two-sublattice AFM assignment cannot be generated from nearest-neighbor connectivity.

Suggested alternatives:
- --method layer
- --method propagation-vector
- --method manual-groups
- increase/decrease --neighbor-cutoff"""

FRUSTRATED_WARNING = """The generated spin assignment is a heuristic initial state for a frustrated magnetic network.
It is not guaranteed to represent the experimental magnetic ground state."""


class NonBipartiteError(ValueError):
    pass


@dataclass(slots=True)
class SpinAssignment:
    signs: dict[int, int]
    method: str
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def moments(self, magnitudes: Mapping[int, float]) -> dict[int, float]:
        return {
            index: self.signs[index] * abs(magnitudes[index]) for index in self.signs
        }

    @property
    def n_up(self) -> int:
        return sum(sign > 0 for sign in self.signs.values())

    @property
    def n_down(self) -> int:
        return sum(sign < 0 for sign in self.signs.values())


def alternating_index(indices: Sequence[int]) -> SpinAssignment:
    return SpinAssignment(
        {index: 1 if order % 2 == 0 else -1 for order, index in enumerate(indices)},
        "alternating-index",
    )


def detect_layers(
    structure: Structure,
    indices: Sequence[int],
    axis: str = "z",
    tolerance: float = 0.25,
    *,
    fractional: bool = False,
) -> list[list[int]]:
    if axis not in "xyz":
        raise ValueError("axis must be x, y, or z")
    if tolerance < 0:
        raise ValueError("layer tolerance must be nonnegative")
    coordinates = structure.fractional_positions if fractional else structure.positions
    values = [
        (float(coordinates[index, "xyz".index(axis)]), index) for index in indices
    ]
    layers: list[list[tuple[float, int]]] = []
    for value, index in sorted(values):
        if (
            not layers
            or abs(value - float(np.mean([item[0] for item in layers[-1]]))) > tolerance
        ):
            layers.append([(value, index)])
        else:
            layers[-1].append((value, index))
    return [[index for _, index in layer] for layer in layers]


def layer_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    axis: str = "z",
    tolerance: float = 0.25,
    fractional: bool = False,
) -> SpinAssignment:
    layers = detect_layers(structure, indices, axis, tolerance, fractional=fractional)
    signs = {
        index: 1 if layer_number % 2 == 0 else -1
        for layer_number, layer in enumerate(layers)
        for index in layer
    }
    return SpinAssignment(
        signs,
        "layer",
        {"axis": axis, "layer_tolerance": tolerance, "layers": layers},
    )


def checkerboard_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    plane: str = "xy",
    cutoff: str | float | None = "auto",
    normal_tolerance: float = 0.25,
) -> SpinAssignment:
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be xy, xz, or yz")
    if normal_tolerance < 0:
        raise ValueError("checkerboard normal tolerance must be nonnegative")
    normal = ({"x", "y", "z"} - set(plane)).pop()
    normal_axis = "xyz".index(normal)
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    projected: list[PairDistance] = []
    for pair in magnetic_pair_distances(structure, indices):
        # Treat atoms within a thin normal-coordinate layer as belonging to
        # the same checkerboard plane.  The projected distance then defines
        # the in-plane first-neighbor shell independently of interlayer gaps.
        if abs(pair.vector[normal_axis]) > normal_tolerance:
            continue
        vector = pair.vector.copy()
        vector[normal_axis] = 0.0
        distance = float(np.linalg.norm(vector[plane_axes]))
        if distance > 1e-10:
            projected.append(PairDistance(pair.i, pair.j, distance, vector))
    if len(indices) > 1 and not projected:
        raise ValueError(f"no in-plane magnetic pairs found for plane {plane}")
    resolved = resolve_cutoff(projected, cutoff) if projected else 0.0
    graph = nx.Graph()
    graph.add_nodes_from(indices)
    graph.add_edges_from(
        (pair.i, pair.j) for pair in projected if pair.distance <= resolved + 1e-9
    )
    if not nx.is_bipartite(graph):
        raise NonBipartiteError(NON_BIPARTITE_MESSAGE)
    signs = _bipartite_signs(graph)
    return SpinAssignment(
        signs,
        "checkerboard",
        {
            "plane": plane,
            "cutoff": resolved,
            "normal_tolerance": normal_tolerance,
        },
    )


def _bipartite_signs(graph: nx.Graph) -> dict[int, int]:
    signs: dict[int, int] = {}
    for component_nodes in sorted(nx.connected_components(graph), key=lambda c: min(c)):
        subgraph = graph.subgraph(component_nodes)
        colors = nx.algorithms.bipartite.color(subgraph)
        # Make the lowest input index positive for deterministic output.
        flip = -1 if colors[min(component_nodes)] else 1
        signs.update(
            {node: flip * (1 if color == 0 else -1) for node, color in colors.items()}
        )
    return signs


def _frustrated_signs(graph: nx.Graph, seed: int = 0) -> dict[int, int]:
    """Deterministic-restart local search for the unweighted Max-Cut problem."""

    nodes = sorted(graph.nodes)
    rng = np.random.default_rng(seed)
    best: dict[int, int] | None = None
    best_cut = -1
    for restart in range(max(8, min(64, len(nodes) * 2))):
        if restart == 0:
            signs = {
                node: 1 if order % 2 == 0 else -1 for order, node in enumerate(nodes)
            }
        else:
            signs = {node: int(rng.choice((-1, 1))) for node in nodes}
        improved = True
        while improved:
            improved = False
            for node in nodes:
                old_opposite = sum(signs[node] != signs[other] for other in graph[node])
                new_opposite = graph.degree[node] - old_opposite
                if new_opposite > old_opposite:
                    signs[node] *= -1
                    improved = True
        cut = sum(signs[left] != signs[right] for left, right in graph.edges)
        if cut > best_cut:
            best_cut, best = cut, signs.copy()
    return best or {node: 1 for node in nodes}


def neighbor_bipartite_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
    allow_frustrated: bool = False,
    seed: int = 0,
) -> SpinAssignment:
    graph, resolved, _ = build_neighbor_graph(
        structure, indices, cutoff, neighbor_shell=neighbor_shell
    )
    metadata = {
        "cutoff": resolved,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }
    if nx.is_bipartite(graph):
        return SpinAssignment(_bipartite_signs(graph), "neighbor-bipartite", metadata)
    if not allow_frustrated:
        raise NonBipartiteError(NON_BIPARTITE_MESSAGE)
    return SpinAssignment(
        _frustrated_signs(graph, seed),
        "neighbor-bipartite",
        {**metadata, "heuristic": "max-cut local search", "frustrated": True},
        [FRUSTRATED_WARNING],
    )


def propagation_vector_ordering(
    structure: Structure,
    indices: Sequence[int],
    q_vector: Sequence[float],
    *,
    phase: float = 0.0,
    fractional_coordinates: bool = True,
) -> SpinAssignment:
    q = np.asarray(q_vector, dtype=float)
    if q.shape != (3,):
        raise ValueError("q-vector must have three components")
    if fractional_coordinates and abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError(
            "fractional propagation-vector coordinates require a nonsingular cell; "
            "use --cartesian-coordinates for nonperiodic structures"
        )
    coordinates = (
        structure.fractional_positions
        if fractional_coordinates
        else structure.positions
    )
    signs: dict[int, int] = {}
    for index in indices:
        value = float(np.cos(2.0 * pi * np.dot(q, coordinates[index]) + phase))
        signs[index] = 1 if value >= 0 else -1
    return SpinAssignment(
        signs,
        "propagation-vector",
        {"q_vector": q.tolist(), "phase": phase, "fractional": fractional_coordinates},
    )


def manual_groups_ordering(
    indices: Sequence[int], up_atoms: Sequence[int], down_atoms: Sequence[int]
) -> SpinAssignment:
    """Assign manually supplied one-based atom groups."""

    up = set(up_atoms)
    down = set(down_atoms)
    if up & down:
        raise ValueError(f"manual up/down groups overlap at atom {min(up & down)}")
    expected = {index + 1 for index in indices}
    supplied = up | down
    outside = supplied - expected
    missing = expected - supplied
    if outside:
        raise ValueError(f"manual group contains nonmagnetic atom {min(outside)}")
    if missing:
        raise ValueError(f"manual groups omit magnetic atom {min(missing)}")
    return SpinAssignment(
        {one_based - 1: 1 for one_based in up}
        | {one_based - 1: -1 for one_based in down},
        "manual-groups",
    )


def read_group_file(path: str | Path) -> tuple[list[int], list[int]]:
    """Read the small YAML group schema, using PyYAML when available."""

    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        return [int(v) for v in data.get("up", [])], [
            int(v) for v in data.get("down", [])
        ]
    except ImportError:
        groups: dict[str, list[int]] = {"up": [], "down": []}
        current: str | None = None
        for raw in text.splitlines():
            stripped = raw.split("#", 1)[0].strip()
            if stripped.rstrip(":") in groups and stripped.endswith(":"):
                current = stripped[:-1]
            elif current and stripped.startswith("-"):
                groups[current].append(int(stripped[1:].strip()))
        return groups["up"], groups["down"]
