"""AFM sign-assignment algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import isfinite, pi
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from .neighbors import (
    PairDistance,
    build_neighbor_graph,
    classify_coordination_geometry,
    cross_pair_distances,
    magnetic_pair_distances,
    periodic_self_image_distance,
    resolve_cutoff,
    resolve_first_shell,
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

SMALL_CELL_WARNING = (
    "cell too small for AFM ordering; use a supercell because a magnetic atom's "
    "periodic self-image lies within the neighbor cutoff"
)

RANDOM_WARNING = (
    "random spin assignment is an exploratory initial state, not a physical "
    "magnetic-ordering model"
)

GRAPH_COLORING_WARNING = (
    "proper graph coloring only avoids equal colors on adjacent atoms; it does "
    "not minimize magnetic energy. For a collinear approximation to a frustrated "
    "network, --allow-frustrated (max-cut) is usually the appropriate method"
)

COMMON_ANIONS = ("O", "S", "Se", "Te", "N", "F", "Cl")


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


@dataclass(slots=True)
class CoordinationAnalysis:
    """Per-site ligand data shared by ordering and GUI presentation."""

    anion_species: list[str]
    cutoff: float
    site_cutoffs: dict[int, float]
    coordination_numbers: dict[int, int]
    ligand_vectors: dict[int, tuple[np.ndarray, ...]]


def alternating_index(indices: Sequence[int]) -> SpinAssignment:
    return SpinAssignment(
        {index: 1 if order % 2 == 0 else -1 for order, index in enumerate(indices)},
        "alternating-index",
    )


def random_ordering(indices: Sequence[int], *, seed: int = 0) -> SpinAssignment:
    rng = np.random.default_rng(seed)
    signs = {index: int(rng.choice((-1, 1))) for index in indices}
    return SpinAssignment(signs, "random", {"seed": seed}, [RANDOM_WARNING])


def graph_component_sizes(graph: nx.Graph) -> list[int]:
    """Return deterministic connected-component sizes in lowest-index order."""

    return [
        len(component)
        for component in sorted(nx.connected_components(graph), key=lambda nodes: min(nodes))
    ]


def disconnected_component_warning(component_sizes: Sequence[int]) -> str | None:
    """Explain the arbitrary relative signs of disconnected graph components."""

    if len(component_sizes) < 2:
        return None
    sizes = ", ".join(str(size) for size in component_sizes)
    return (
        f"magnetic-neighbor graph has {len(component_sizes)} disconnected components "
        f"(sizes: {sizes}). Relative spin signs between components are set by a "
        "deterministic convention and have no physical meaning; increase "
        "--neighbor-cutoff to include interlayer (superexchange) neighbors, or "
        "consider layer/propagation-vector ordering"
    )


def by_species_ordering(
    structure: Structure,
    indices: Sequence[int],
    magnetic_species: Sequence[str],
    up_species: Sequence[str],
    down_species: Sequence[str],
) -> SpinAssignment:
    """Assign opposite signs to two explicitly named element sublattices."""

    magnetic = {item.strip().lower() for item in magnetic_species if item.strip()}
    up = {item.strip().lower() for item in up_species if item.strip()}
    down = {item.strip().lower() for item in down_species if item.strip()}
    if not up or not down:
        raise ValueError(
            "by-species requires both --up-species and --down-species; use "
            "by-coordination when one element occupies both Td/Oh inverse-spinel sites"
        )
    overlap = up & down
    if overlap:
        raise ValueError(
            f"up/down species overlap: {', '.join(sorted(overlap))}"
        )
    supplied = up | down
    missing = magnetic - supplied
    extra = supplied - magnetic
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing magnetic species: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"nonmagnetic/extra species: {', '.join(sorted(extra))}")
        raise ValueError(
            "by-species groups must exactly cover --magnetic-species ("
            + "; ".join(details)
            + "); use by-coordination when one element occupies both Td/Oh "
            "inverse-spinel sites"
        )
    signs = {
        index: 1 if structure.symbols[index].lower() in up else -1
        for index in indices
    }
    return SpinAssignment(
        signs,
        "by-species",
        {"up_species": sorted(up), "down_species": sorted(down)},
    )


def _resolve_anion_species(
    structure: Structure, anion_species: Sequence[str] | None
) -> list[str]:
    if anion_species:
        requested = [item.strip() for item in anion_species if item.strip()]
        if not requested:
            raise ValueError("--anion-species must name at least one element")
        return requested
    present = {symbol.lower() for symbol in structure.symbols}
    detected = [symbol for symbol in COMMON_ANIONS if symbol.lower() in present]
    if not detected:
        raise ValueError(
            "could not auto-detect an anion species; specify --anion-species"
        )
    if len(detected) > 1:
        raise ValueError(
            "multiple possible anion species were found "
            f"({', '.join(detected)}); specify --anion-species"
        )
    return detected


def analyze_coordination_sites(
    structure: Structure,
    indices: Sequence[int],
    *,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
) -> CoordinationAnalysis:
    """Resolve first-shell anions and retain the vectors used to count them."""

    anions = _resolve_anion_species(structure, anion_species)
    wanted = {item.lower() for item in anions}
    anion_indices = [
        index for index, symbol in enumerate(structure.symbols) if symbol.lower() in wanted
    ]
    if not anion_indices:
        raise ValueError(f"no atoms matched anion species: {', '.join(anions)}")
    overlap = sorted(set(indices) & set(anion_indices))
    if overlap:
        raise ValueError(
            f"anion species overlaps magnetic atom {overlap[0] + 1}; choose distinct "
            "magnetic and anion species"
        )

    site_cutoffs: dict[int, float] = {}
    automatic = anion_cutoff is None or str(anion_cutoff).lower() == "auto"
    if automatic:
        pairs = cross_pair_distances(
            structure, indices, anion_indices, all_images=True
        )
        if not pairs:
            raise ValueError("no finite magnetic-anion distances were found")
        # Distorted spinels commonly have different Td-O and Oh-O bond
        # lengths. Resolve the first shell per magnetic site.
        for index in indices:
            site_pairs = [pair for pair in pairs if pair.i == index]
            if not site_pairs:
                raise ValueError(
                    f"no finite anion distances were found for atom {index + 1}"
                )
            site_cutoffs[index], _ = resolve_first_shell(site_pairs)
        resolved = max(site_cutoffs.values())
    else:
        resolved = float(anion_cutoff)
        if resolved <= 0:
            raise ValueError("anion cutoff must be positive")
        site_cutoffs = {index: resolved for index in indices}
    vectors = {
        index: tuple(
            pair.vector
            for pair in cross_pair_distances(
                structure,
                [index],
                anion_indices,
                all_images=True,
                cutoff=site_cutoffs[index],
            )
        )
        for index in indices
    }
    return CoordinationAnalysis(
        anion_species=anions,
        cutoff=resolved,
        site_cutoffs=site_cutoffs,
        coordination_numbers={index: len(vectors[index]) for index in indices},
        ligand_vectors=vectors,
    )


def coordination_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
    up_coordination: Sequence[int] = (6,),
    down_coordination: Sequence[int] = (4,),
    coordination_tolerance: int = 0,
) -> SpinAssignment:
    """Classify magnetic sites by their first-shell anion coordination."""

    if coordination_tolerance < 0:
        raise ValueError("coordination tolerance must be a nonnegative integer")
    up_values = {int(value) for value in up_coordination}
    down_values = {int(value) for value in down_coordination}
    if not up_values or not down_values or min(up_values | down_values) < 0:
        raise ValueError("up/down coordination lists must contain nonnegative integers")
    analysis = analyze_coordination_sites(
        structure,
        indices,
        anion_species=anion_species,
        anion_cutoff=anion_cutoff,
    )
    coordinations = analysis.coordination_numbers
    geometries = {
        index: classify_coordination_geometry(analysis.ligand_vectors[index])
        for index in indices
    }

    signs: dict[int, int] = {}
    sublattices: dict[int, str] = {}
    unclassified: list[str] = []
    for index in indices:
        coordination = coordinations[index]
        matches_up = any(
            abs(coordination - target) <= coordination_tolerance
            for target in up_values
        )
        matches_down = any(
            abs(coordination - target) <= coordination_tolerance
            for target in down_values
        )
        if matches_up == matches_down:
            reason = "matches both sublattices" if matches_up else "is unclassified"
            unclassified.append(
                f"atom {index + 1} ({structure.symbols[index]}, CN={coordination}) {reason}"
            )
            continue
        signs[index] = 1 if matches_up else -1
        sublattices[index] = "up" if matches_up else "down"
    if unclassified:
        raise ValueError(
            "; ".join(unclassified)
            + ". If an unclassified site is a surface-truncated atom in a slab, "
            "exclude it with --exclude-atoms or include its CN in "
            "--up-coordination/--down-coordination."
        )

    warnings: list[str] = []
    up_elements = {
        structure.symbols[index].lower() for index, sign in signs.items() if sign > 0
    }
    down_elements = {
        structure.symbols[index].lower() for index, sign in signs.items() if sign < 0
    }
    shared = sorted(up_elements & down_elements)
    if shared:
        warnings.append(
            "the same magnetic element appears in both coordination sublattices "
            f"({', '.join(shared)}), as can occur in an inverse spinel"
        )
    return SpinAssignment(
        signs,
        "by-coordination",
        {
            "anion_species": analysis.anion_species,
            "anion_cutoff": analysis.cutoff,
            "anion_cutoffs": analysis.site_cutoffs,
            "coordination_numbers": coordinations,
            "coordination_geometry": geometries,
            "sublattice_classification": sublattices,
            "up_coordination": sorted(up_values),
            "down_coordination": sorted(down_values),
            "coordination_tolerance": coordination_tolerance,
        },
        warnings,
    )


def detect_layers(
    structure: Structure,
    indices: Sequence[int],
    axis: str = "z",
    tolerance: float = 0.25,
    *,
    fractional: bool = False,
) -> list[list[int]]:
    layers, _ = _detect_layers_with_metadata(
        structure, indices, axis, tolerance, fractional=fractional
    )
    return layers


def _detect_layers_with_metadata(
    structure: Structure,
    indices: Sequence[int],
    axis: str = "z",
    tolerance: float = 0.25,
    *,
    fractional: bool = False,
) -> tuple[list[list[int]], bool]:
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
    wrapped_layer_merged = False
    axis_number = "xyz".index(axis)
    if len(layers) > 1 and structure.pbc[axis_number]:
        period = 1.0 if fractional else float(np.linalg.norm(structure.cell[axis_number]))
        first_mean = float(np.mean([item[0] for item in layers[0]]))
        last_mean = float(np.mean([item[0] for item in layers[-1]]))
        if period - (last_mean - first_mean) <= tolerance + 1e-12:
            # Keep the wrapped layer first so its sign and atom ordering remain
            # deterministic, then continue through the interior layers.
            layers = [layers[-1] + layers[0], *layers[1:-1]]
            wrapped_layer_merged = True
    return [[index for _, index in layer] for layer in layers], wrapped_layer_merged


def layer_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    axis: str = "z",
    tolerance: float = 0.25,
    fractional: bool = False,
) -> SpinAssignment:
    layers, wrapped_layer_merged = _detect_layers_with_metadata(
        structure, indices, axis, tolerance, fractional=fractional
    )
    signs = {
        index: 1 if layer_number % 2 == 0 else -1
        for layer_number, layer in enumerate(layers)
        for index in layer
    }
    warnings: list[str] = []
    axis_number = "xyz".index(axis)
    if structure.pbc[axis_number] and len(layers) % 2:
        warnings.append(
            f"periodic {axis}-axis contains {len(layers)} magnetic layers; the "
            "odd layer count breaks alternating AFM order across the PBC boundary"
        )
    elif not structure.pbc[axis_number] and len(layers) % 2:
        warnings.append(
            "uncompensated AFM slab: net initial moment is nonzero "
            "(odd magnetic layer count)"
        )
    warnings.extend(
        _combined_layer_species_warnings(structure, signs, f"along {axis}")
    )
    return SpinAssignment(
        signs,
        "layer",
        {
            "axis": axis,
            "layer_tolerance": tolerance,
            "fractional_layers": fractional,
            "layers": layers,
            "wrapped_layer_merged": wrapped_layer_merged,
        },
        warnings,
    )


def direction_layer_ordering(
    structure: Structure,
    indices: Sequence[int],
    direction: Sequence[float],
    *,
    tolerance: float = 0.25,
) -> SpinAssignment:
    """Alternate layers along an arbitrary Cartesian projection direction."""

    layers, vector = detect_direction_layers(
        structure, indices, direction, tolerance=tolerance
    )
    signs = {
        index: 1 if layer_number % 2 == 0 else -1
        for layer_number, layer in enumerate(layers)
        for index in layer
    }
    fully_periodic_direction = all(
        structure.pbc[axis]
        for axis, component in enumerate(vector)
        if abs(component) > 1e-12
    )
    has_periodic_component = any(
        structure.pbc[axis] and abs(component) > 1e-12
        for axis, component in enumerate(vector)
    )
    warnings: list[str] = []
    if has_periodic_component:
        warnings.append(
            "layers crossing a periodic cell boundary are not merged for an "
            "arbitrary layer direction"
        )
    if fully_periodic_direction and len(layers) % 2:
        warnings.append(
            f"periodic layer direction contains {len(layers)} magnetic layers; the "
            "odd layer count may break alternating AFM order across a PBC boundary"
        )
    elif not has_periodic_component and len(layers) % 2:
        warnings.append(
            "uncompensated AFM slab: net initial moment is nonzero "
            "(odd magnetic layer count)"
        )
    direction_text = " ".join(f"{value:.6g}" for value in vector)
    warnings.extend(
        _combined_layer_species_warnings(
            structure, signs, f"along direction ({direction_text})"
        )
    )
    return SpinAssignment(
        signs,
        "layer",
        {
            "layer_direction": vector.tolist(),
            "layer_tolerance": tolerance,
            "layers": layers,
        },
        warnings,
    )


def _combined_layer_species_warnings(
    structure: Structure,
    signs: Mapping[int, int],
    direction: str,
) -> list[str]:
    """Warn when combined multi-element layering collapses one species' sign."""

    signs_by_species: dict[str, set[int]] = {}
    display_names: dict[str, str] = {}
    for index, sign in signs.items():
        symbol = structure.symbols[index]
        normalized = symbol.lower()
        display_names.setdefault(normalized, symbol)
        signs_by_species.setdefault(normalized, set()).add(sign)
    if len(signs_by_species) < 2:
        return []

    warnings: list[str] = []
    for normalized, species_signs in signs_by_species.items():
        if len(species_signs) != 1:
            continue
        sign = next(iter(species_signs))
        orientation = "spin-up" if sign > 0 else "spin-down"
        parity = "even" if sign > 0 else "odd"
        warnings.append(
            f"species {display_names[normalized]} is entirely {orientation} under "
            f"combined layering {direction}; its sublattice coincides only with "
            f"{parity}-parity layers of the combined stack. Consider --method "
            "by-coordination for multi-species ferrimagnets, or layer each "
            "species separately."
        )
    return warnings


def detect_direction_layers(
    structure: Structure,
    indices: Sequence[int],
    direction: Sequence[float],
    *,
    tolerance: float = 0.25,
) -> tuple[list[list[int]], np.ndarray]:
    """Cluster atoms by Cartesian projection along an arbitrary direction."""

    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError("layer direction must contain three values and be nonzero")
    if tolerance < 0:
        raise ValueError("layer tolerance must be nonnegative")
    unit = vector / np.linalg.norm(vector)
    values = sorted(
        (float(np.dot(structure.positions[index], unit)), index) for index in indices
    )
    grouped: list[list[tuple[float, int]]] = []
    for value, index in values:
        if (
            not grouped
            or abs(value - float(np.mean([item[0] for item in grouped[-1]])))
            > tolerance
        ):
            grouped.append([(value, index)])
        else:
            grouped[-1].append((value, index))
    layers = [[index for _, index in layer] for layer in grouped]
    return layers, vector


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
    component_sizes = graph_component_sizes(graph)
    warnings: list[str] = []
    component_warning = disconnected_component_warning(component_sizes)
    if component_warning:
        warnings.append(component_warning)
    return SpinAssignment(
        signs,
        "checkerboard",
        {
            "plane": plane,
            "cutoff": resolved,
            "normal_tolerance": normal_tolerance,
            "component_sizes": component_sizes,
        },
        warnings,
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
    warnings: list[str] = []
    component_sizes = graph_component_sizes(graph)
    metadata["component_sizes"] = component_sizes
    component_warning = disconnected_component_warning(component_sizes)
    if component_warning:
        warnings.append(component_warning)
    self_image = periodic_self_image_distance(structure)
    if self_image is not None and (
        (resolved > 0 and self_image <= resolved + 1e-9)
        or (len(indices) == 1 and (cutoff is None or str(cutoff).lower() == "auto"))
    ):
        warnings.append(SMALL_CELL_WARNING)
        metadata["periodic_self_image_distance"] = self_image
    if nx.is_bipartite(graph):
        return SpinAssignment(
            _bipartite_signs(graph), "neighbor-bipartite", metadata, warnings
        )
    if not allow_frustrated:
        raise NonBipartiteError(NON_BIPARTITE_MESSAGE)
    return SpinAssignment(
        _frustrated_signs(graph, seed),
        "neighbor-bipartite",
        {**metadata, "heuristic": "max-cut local search", "frustrated": True},
        [*warnings, FRUSTRATED_WARNING],
    )


def _color_spin_values(
    n_colors: int, color_spins: str | Sequence[int] | None
) -> tuple[int, ...]:
    if color_spins is None:
        defaults = {
            1: (1,),
            2: (1, -1),
            3: (1, -1, 0),
            4: (1, -1, 1, -1),
        }
        if n_colors not in defaults:
            raise ValueError(
                f"no default spin map for {n_colors} colors; specify --color-spins"
            )
        return defaults[n_colors]
    raw = (
        [part for part in color_spins.replace(",", " ").split() if part]
        if isinstance(color_spins, str)
        else list(color_spins)
    )
    try:
        values = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("--color-spins must contain comma-separated -1, 0, or +1") from exc
    if len(values) != n_colors:
        raise ValueError(
            f"--color-spins length {len(values)} does not match {n_colors} graph colors"
        )
    if any(value not in {-1, 0, 1} for value in values):
        raise ValueError("--color-spins values must be -1, 0, or +1")
    return values


def _select_color_spin_map(
    colors: Mapping[int, int],
    base_values: Sequence[int],
    *,
    balance_colors: bool,
    magnitudes: Mapping[int, float] | None,
    seed: int,
) -> dict[int, int]:
    candidates = list(dict.fromkeys(permutations(base_values)))
    if balance_colors:
        if magnitudes is not None:
            missing = sorted(set(colors).difference(magnitudes))
            if missing:
                one_based = ", ".join(str(index + 1) for index in missing)
                raise ValueError(
                    "moment magnitudes are missing for magnetic atom(s): " + one_based
                )
        color_weights = {
            color: sum(
                1.0 if magnitudes is None else abs(float(magnitudes[index]))
                for index, node_color in colors.items()
                if node_color == color
            )
            for color in range(len(base_values))
        }
        scores = [
            abs(
                sum(
                    color_weights[color] * value
                    for color, value in enumerate(candidate)
                )
            )
            for candidate in candidates
        ]
        best = min(scores)
        candidates = [
            candidate
            for candidate, score in zip(candidates, scores)
            if abs(score - best) <= 1e-12
        ]
    selected = candidates[seed % len(candidates)]
    return {color: value for color, value in enumerate(selected)}


def graph_coloring_ordering(
    structure: Structure,
    indices: Sequence[int],
    *,
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
    max_colors: int = 4,
    color_spins: str | Sequence[int] | None = None,
    balance_colors: bool = False,
    magnitudes: Mapping[int, float] | None = None,
    seed: int = 0,
) -> SpinAssignment:
    """Generate a collinear candidate from a proper graph coloring."""

    if max_colors < 1:
        raise ValueError("--max-colors must be a positive integer")
    graph, resolved, _ = build_neighbor_graph(
        structure, indices, cutoff, neighbor_shell=neighbor_shell
    )
    bipartite = nx.is_bipartite(graph)
    if bipartite:
        reference_signs = _bipartite_signs(graph)
        if any(sign < 0 for sign in reference_signs.values()):
            colors = {
                index: 0 if reference_signs[index] > 0 else 1 for index in indices
            }
            n_colors = 2
        else:
            colors = {index: 0 for index in indices}
            n_colors = 1
        strategy = "bipartite fallback"
    else:
        raw_colors = nx.greedy_color(graph, strategy="DSATUR")
        remap = {
            original: normalized
            for normalized, original in enumerate(sorted(set(raw_colors.values())))
        }
        colors = {index: remap[raw_colors[index]] for index in indices}
        n_colors = len(remap)
        strategy = "DSATUR"
    if n_colors > max_colors:
        raise ValueError(
            f"graph coloring requires {n_colors} colors, exceeding --max-colors "
            f"{max_colors}"
        )
    base_values = _color_spin_values(n_colors, color_spins)
    if bipartite and color_spins is None:
        canonical_values = (1,) if n_colors == 1 else (1, -1)
        color_spin_map = {
            color: value for color, value in enumerate(canonical_values)
        }
        signs = reference_signs
    else:
        color_spin_map = _select_color_spin_map(
            colors,
            base_values,
            balance_colors=balance_colors,
            magnitudes=magnitudes,
            seed=seed,
        )
        signs = {index: color_spin_map[color] for index, color in colors.items()}
    component_sizes = graph_component_sizes(graph)
    warnings = [GRAPH_COLORING_WARNING]
    component_warning = disconnected_component_warning(component_sizes)
    if component_warning:
        warnings.append(component_warning)
    self_image = periodic_self_image_distance(structure)
    metadata: dict[str, object] = {
        "cutoff": resolved,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "component_sizes": component_sizes,
        "colors": colors,
        "n_colors": n_colors,
        "color_spin_map": color_spin_map,
        "strategy": strategy,
        "balance_colors": balance_colors,
        "balance_metric": "moment" if magnitudes is not None else "unit-spin",
        "seed": seed,
    }
    if self_image is not None and (
        (resolved > 0 and self_image <= resolved + 1e-9)
        or (len(indices) == 1 and (cutoff is None or str(cutoff).lower() == "auto"))
    ):
        warnings.append(SMALL_CELL_WARNING)
        metadata["periodic_self_image_distance"] = self_image
    return SpinAssignment(signs, "graph-coloring", metadata, warnings)


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
    node_indices: list[int] = []
    for index in indices:
        value = float(np.cos(2.0 * pi * np.dot(q, coordinates[index]) + phase))
        if abs(value) < 1e-6:
            node_indices.append(index)
        signs[index] = 1 if value >= 0 else -1
    warnings: list[str] = []
    if node_indices:
        displayed = ", ".join(str(index + 1) for index in node_indices[:12])
        if len(node_indices) > 12:
            displayed += f", ... ({len(node_indices)} total)"
        coordinate_hint = (
            "q-vector components are fractional coordinates of the input cell and "
            "must be scaled when using a supercell"
            if fractional_coordinates
            else "q-vector components are Cartesian because Cartesian-coordinate "
            "mode is active"
        )
        warnings.append(
            f"propagation-vector cosine is near a node for {len(node_indices)} atom(s) "
            f"(1-based indices: {displayed}); {coordinate_hint}"
        )
    n_up = sum(sign > 0 for sign in signs.values())
    n_down = len(signs) - n_up
    if abs(n_up - n_down) > max(2, 0.1 * len(signs)):
        warnings.append(
            f"propagation-vector assignment is strongly imbalanced "
            f"({n_up} up, {n_down} down); check q-vector scaling and phase"
        )
    return SpinAssignment(
        signs,
        "propagation-vector",
        {"q_vector": q.tolist(), "phase": phase, "fractional": fractional_coordinates},
        warnings,
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


def manual_spins_ordering(
    structure: Structure,
    indices: Sequence[int],
    spin_values: Mapping[int, float],
    *,
    fill_unspecified_zero: bool = False,
) -> SpinAssignment:
    """Assign signed moments supplied for one-based atom indices.

    Unlike the established site-moment path, these values are direct signed
    moments rather than magnitudes to be combined with method-derived signs.
    """

    expected = {index + 1 for index in indices}
    supplied: dict[int, float] = {}
    for raw_index, raw_value in spin_values.items():
        one_based = int(raw_index)
        if one_based != raw_index or not 1 <= one_based <= len(structure):
            raise ValueError(f"manual spin atom index out of range: {raw_index}")
        value = float(raw_value)
        if not isfinite(value):
            raise ValueError(
                f"manual spin for atom {one_based} must be a finite number"
            )
        if one_based not in expected:
            raise ValueError(
                f"manual spin atom {one_based} is not selected by "
                f"--magnetic-species (actual element: "
                f"{structure.symbols[one_based - 1]})"
            )
        supplied[one_based] = value

    missing = expected - set(supplied)
    if missing and not fill_unspecified_zero:
        one_based = min(missing)
        raise ValueError(
            f"manual spins omit magnetic atom {one_based} "
            f"({structure.symbols[one_based - 1]}); specify it or pass "
            "--fill-unspecified-zero"
        )
    direct_spins = {
        one_based - 1: supplied.get(one_based, 0.0)
        for one_based in sorted(expected)
    }
    signs = {
        index: 1 if value > 0 else -1 if value < 0 else 0
        for index, value in direct_spins.items()
    }
    return SpinAssignment(
        signs,
        "manual-spins",
        {"spin_values": direct_spins, "fill_unspecified_zero": fill_unspecified_zero},
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
