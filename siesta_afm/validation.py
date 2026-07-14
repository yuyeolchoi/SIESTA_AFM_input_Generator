"""Validation and analysis of generated initial-spin patterns."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence

import networkx as nx

from .io import parse_dm_init_spin
from .neighbors import build_neighbor_graph, shell_summary
from .ordering import (
    detect_direction_layers,
    detect_layers,
    disconnected_component_warning,
    graph_component_sizes,
)
from .structure import Structure


@dataclass(slots=True)
class ValidationReport:
    n_up: int
    n_down: int
    n_zero: int
    net_initial_spin: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    antiparallel_fraction: float | None = None
    parallel_fraction: float | None = None
    graph_nodes: int | None = None
    graph_edges: int | None = None
    connected_components: int | None = None
    component_sizes: list[int] | None = None
    layer_spin_distribution: list[dict[str, int]] | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, **asdict(self)}


def validate_spins(
    rows: Sequence[tuple[int, float]],
    *,
    structure: Structure | None = None,
    magnetic_species: Sequence[str] | None = None,
    cutoff: str | float | None = "auto",
    axis: str = "z",
    layer_tolerance: float = 0.25,
) -> ValidationReport:
    indices = [index for index, _ in rows]
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    errors = [f"duplicate DM.InitSpin atom index: {index}" for index in duplicates]
    spins_one_based: dict[int, float] = {}
    for index, spin in rows:
        spins_one_based.setdefault(index, spin)
    if any(index <= 0 for index in indices):
        errors.append("DM.InitSpin atom indices must be one-based positive integers")

    if structure is not None:
        outside = sorted(index for index in indices if index > len(structure))
        errors.extend(
            f"DM.InitSpin atom index out of range: {index}" for index in outside
        )
        if magnetic_species:
            wanted = {symbol.lower() for symbol in magnetic_species}
            nonmagnetic = sorted(
                index
                for index in indices
                if 1 <= index <= len(structure)
                and structure.symbols[index - 1].lower() not in wanted
                and spins_one_based.get(index, 0.0) != 0.0
            )
            errors.extend(
                f"nonmagnetic species atom has nonzero spin: {index} "
                f"({structure.symbols[index - 1]})"
                for index in nonmagnetic
            )

    values = list(spins_one_based.values())
    report = ValidationReport(
        n_up=sum(value > 0 for value in values),
        n_down=sum(value < 0 for value in values),
        n_zero=sum(value == 0 for value in values),
        net_initial_spin=float(sum(values)),
        errors=errors,
    )
    valid_zero_based = {
        index - 1: value
        for index, value in spins_one_based.items()
        if index > 0 and (structure is None or index <= len(structure)) and value != 0
    }
    if structure is not None and len(valid_zero_based) > 1:
        graph, _, _ = build_neighbor_graph(structure, sorted(valid_zero_based), cutoff)
        report.graph_nodes = graph.number_of_nodes()
        report.graph_edges = graph.number_of_edges()
        report.connected_components = nx.number_connected_components(graph)
        report.component_sizes = graph_component_sizes(graph)
        component_warning = disconnected_component_warning(report.component_sizes)
        if component_warning:
            report.warnings.append(component_warning)
        if graph.number_of_edges():
            opposite = sum(
                valid_zero_based[left] * valid_zero_based[right] < 0
                for left, right in graph.edges
            )
            report.antiparallel_fraction = opposite / graph.number_of_edges()
            report.parallel_fraction = 1.0 - report.antiparallel_fraction
        layers = detect_layers(
            structure, sorted(valid_zero_based), axis, layer_tolerance
        )
        report.layer_spin_distribution = [
            {
                "layer": number + 1,
                "up": sum(valid_zero_based[index] > 0 for index in layer),
                "down": sum(valid_zero_based[index] < 0 for index in layer),
            }
            for number, layer in enumerate(layers)
        ]
    return report


def validate_spin_file(
    path: str,
    *,
    structure: Structure | None = None,
    magnetic_species: Sequence[str] | None = None,
    cutoff: str | float | None = "auto",
) -> ValidationReport:
    parse_warnings: list[str] = []
    report = validate_spins(
        parse_dm_init_spin(path, warnings=parse_warnings),
        structure=structure,
        magnetic_species=magnetic_species,
        cutoff=cutoff,
    )
    report.warnings.extend(parse_warnings)
    return report


def analyze_structure(
    structure: Structure,
    magnetic_indices: Sequence[int],
    *,
    magnetic_species: Sequence[str],
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
    axis: str = "z",
    layer_tolerance: float = 0.25,
    fractional_layers: bool = False,
    layer_direction: Sequence[float] | None = None,
) -> dict[str, object]:
    graph, resolved, pairs = build_neighbor_graph(
        structure, magnetic_indices, cutoff, neighbor_shell=neighbor_shell
    )
    component_sizes = graph_component_sizes(graph)
    warnings: list[str] = []
    component_warning = disconnected_component_warning(component_sizes)
    if component_warning:
        warnings.append(component_warning)
    if layer_direction is None:
        layers = detect_layers(
            structure,
            magnetic_indices,
            axis,
            layer_tolerance,
            fractional=fractional_layers,
        )
        layer_label = axis
    else:
        layers, direction = detect_direction_layers(
            structure,
            magnetic_indices,
            layer_direction,
            tolerance=layer_tolerance,
        )
        layer_label = "direction " + " ".join(f"{value:g}" for value in direction)
    return {
        "number_of_atoms": len(structure),
        "magnetic_species": list(magnetic_species),
        "number_of_magnetic_atoms": len(magnetic_indices),
        "distance_shells": shell_summary(pairs),
        "suggested_cutoff": resolved,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
        "component_sizes": component_sizes,
        "bipartite": nx.is_bipartite(graph),
        "axis": layer_label,
        "detected_layers": len(layers),
        "atoms_per_layer": [len(layer) for layer in layers],
        "warnings": warnings,
    }


def format_analysis(report: dict[str, object]) -> str:
    lines = [
        f"Number of atoms: {report['number_of_atoms']}",
        f"Magnetic species: {' '.join(report['magnetic_species'])}",
        f"Number of magnetic atoms: {report['number_of_magnetic_atoms']}",
        "Nearest magnetic-atom distances:",
    ]
    for shell in report["distance_shells"]:
        lines.append(f"  {shell['distance']:.3f} Angstrom: {shell['pairs']} pairs")
    lines.extend(
        [
            f"Suggested first-shell cutoff: {report['suggested_cutoff']:.3f} Angstrom",
            f"Graph nodes: {report['graph_nodes']}",
            f"Graph edges: {report['graph_edges']}",
            f"Connected components: {report['connected_components']}",
            "Component sizes: "
            + ", ".join(str(value) for value in report["component_sizes"]),
            f"Bipartite: {report['bipartite']}",
            f"Detected layers along {report['axis']}: {report['detected_layers']}",
            "Atoms per layer: " + ", ".join(str(v) for v in report["atoms_per_layer"]),
        ]
    )
    lines.extend(f"WARNING: {warning}" for warning in report.get("warnings", []))
    return "\n".join(lines)


def format_validation(report: ValidationReport) -> str:
    lines = [
        f"Valid: {report.valid}",
        f"Spin-up atoms: {report.n_up}",
        f"Spin-down atoms: {report.n_down}",
        f"Zero-spin atoms: {report.n_zero}",
        f"Net initial spin: {report.net_initial_spin:.6g}",
    ]
    if report.antiparallel_fraction is not None:
        lines.extend(
            [
                f"Nearest-neighbor antiparallel fraction: {report.antiparallel_fraction:.3f}",
                f"Nearest-neighbor parallel fraction: {report.parallel_fraction:.3f}",
                f"AFM score: {report.antiparallel_fraction:.3f}",
            ]
        )
    lines.extend(f"WARNING: {warning}" for warning in report.warnings)
    lines.extend(f"ERROR: {error}" for error in report.errors)
    return "\n".join(lines)
