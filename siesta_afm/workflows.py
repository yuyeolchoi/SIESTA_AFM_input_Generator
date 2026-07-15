"""Reusable high-level spin-generation workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fdf_writer import render_dm_init_spin

from .magnetic_sites import (
    built_in_element_moments,
    parse_atom_indices,
    resolve_moments,
    resolve_moments_with_sources,
    select_magnetic_sites,
)
from .ordering import (
    NonBipartiteError,
    SpinAssignment,
    alternating_index,
    by_species_ordering,
    checkerboard_ordering,
    coordination_ordering,
    direction_layer_ordering,
    graph_coloring_ordering,
    layer_ordering,
    manual_groups_ordering,
    neighbor_bipartite_ordering,
    propagation_vector_ordering,
    random_ordering,
    read_group_file,
)
from .neighbors import build_neighbor_graph
from .structure import Structure


ENUMERATION_METHODS = (
    "alternating-index",
    "random",
    "layer",
    "checkerboard",
    "neighbor-bipartite",
    "graph-coloring",
    "propagation-vector",
    "manual-groups",
    "by-species",
    "by-coordination",
    "frustrated",
)


@dataclass(slots=True)
class EnumerationResult:
    """Files and diagnostics produced by one candidate-enumeration run."""

    manifest: list[dict[str, object]]
    failures: list[str]
    notices: list[str]
    written_files: list[Path]
    manifest_path: Path


def generate_assignment(
    structure: Structure,
    magnetic_species: Sequence[str],
    method: str,
    moment: Sequence[str] | str | None,
    *,
    exclude_atoms: str | Sequence[str] | None = None,
    adsorbate_indices: str | Sequence[str] | None = None,
    site_moment_file: str | None = None,
    axis: str = "z",
    layer_direction: Sequence[float] | None = None,
    layer_tolerance: float = 0.25,
    fractional_layers: bool = False,
    plane: str = "xy",
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
    allow_frustrated: bool = False,
    q_vector: Sequence[float] | None = None,
    afm_type: str | None = None,
    phase: float = 0.0,
    fractional_coordinates: bool = True,
    up_atoms: str | Sequence[str] | None = None,
    down_atoms: str | Sequence[str] | None = None,
    group_file: str | None = None,
    up_species: Sequence[str] | None = None,
    down_species: Sequence[str] | None = None,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
    up_coordination: Sequence[int] = (6,),
    down_coordination: Sequence[int] = (4,),
    coordination_tolerance: int = 0,
    max_colors: int = 4,
    color_spins: str | Sequence[int] | None = None,
    balance_colors: bool = False,
    seed: int = 0,
) -> tuple[list[int], SpinAssignment, dict[int, float]]:
    indices = select_magnetic_sites(
        structure,
        magnetic_species,
        exclude_atoms=exclude_atoms,
        adsorbate_indices=adsorbate_indices,
    )
    default_moments: dict[str, float] | None = None
    if moment is None:
        default_moments = built_in_element_moments(structure, indices)
        moment = [
            f"{element}={value:.1f}" for element, value in default_moments.items()
        ]
    resolved_magnitudes: dict[int, float] | None = None
    resolved_moment_sources: dict[int, str] | None = None
    if method == "alternating-index":
        assignment = alternating_index(indices)
    elif method == "random":
        assignment = random_ordering(indices, seed=seed)
    elif method == "by-species":
        assignment = by_species_ordering(
            structure,
            indices,
            magnetic_species,
            up_species or (),
            down_species or (),
        )
    elif method == "by-coordination":
        assignment = coordination_ordering(
            structure,
            indices,
            anion_species=anion_species,
            anion_cutoff=anion_cutoff,
            up_coordination=up_coordination,
            down_coordination=down_coordination,
            coordination_tolerance=coordination_tolerance,
        )
    elif method == "layer":
        if layer_direction is not None:
            if fractional_layers:
                raise ValueError(
                    "--fractional-layers cannot be combined with --layer-direction"
                )
            assignment = direction_layer_ordering(
                structure, indices, layer_direction, tolerance=layer_tolerance
            )
        else:
            assignment = layer_ordering(
                structure,
                indices,
                axis=axis,
                tolerance=layer_tolerance,
                fractional=fractional_layers,
            )
    elif method == "checkerboard":
        assignment = checkerboard_ordering(
            structure,
            indices,
            plane=plane,
            cutoff=cutoff,
            normal_tolerance=layer_tolerance,
        )
    elif method == "graph-coloring":
        if default_moments is not None:
            resolved_magnitudes, resolved_moment_sources = (
                resolve_moments_with_sources(
                    structure,
                    indices,
                    moment,
                    site_moment_file=site_moment_file,
                )
            )
        else:
            resolved_magnitudes = resolve_moments(
                structure,
                indices,
                moment,
                site_moment_file=site_moment_file,
            )
        assignment = graph_coloring_ordering(
            structure,
            indices,
            cutoff=cutoff,
            neighbor_shell=neighbor_shell,
            max_colors=max_colors,
            color_spins=color_spins,
            balance_colors=balance_colors,
            magnitudes=resolved_magnitudes,
            seed=seed,
        )
    elif method in {"neighbor-bipartite", "frustrated"}:
        assignment = neighbor_bipartite_ordering(
            structure,
            indices,
            cutoff=cutoff,
            neighbor_shell=neighbor_shell,
            allow_frustrated=allow_frustrated or method == "frustrated",
            seed=seed,
        )
        if method == "frustrated":
            assignment.method = "frustrated"
    elif method == "propagation-vector":
        if q_vector is not None and afm_type is not None:
            raise ValueError("--q-vector and --afm-type are mutually exclusive")
        presets = {
            "A": (0.0, 0.0, 0.5),
            "C": (0.5, 0.5, 0.0),
            "G": (0.5, 0.5, 0.5),
        }
        if afm_type is not None:
            if not fractional_coordinates:
                raise ValueError(
                    "--afm-type presets use fractional coordinates and cannot be "
                    "combined with --cartesian-coordinates"
                )
            afm_type = afm_type.upper()
            if afm_type not in presets:
                raise ValueError("AFM type must be A, C, or G")
            q_vector = presets[afm_type]
        if q_vector is None:
            raise ValueError(
                "--q-vector or --afm-type is required for propagation-vector"
            )
        assignment = propagation_vector_ordering(
            structure,
            indices,
            q_vector,
            phase=phase,
            fractional_coordinates=fractional_coordinates,
        )
        if afm_type is not None:
            assignment.metadata["afm_type"] = afm_type
    elif method == "manual-groups":
        if group_file:
            up, down = read_group_file(group_file)
        else:
            up = sorted(parse_atom_indices(up_atoms))
            down = sorted(parse_atom_indices(down_atoms))
        assignment = manual_groups_ordering(indices, up, down)
    else:
        raise ValueError(f"unsupported AFM method: {method}")
    coordinations = (
        assignment.metadata.get("coordination_numbers")
        if method == "by-coordination"
        else None
    )
    if resolved_magnitudes is None:
        if method == "by-coordination" and isinstance(coordinations, dict):
            resolved_magnitudes, resolved_moment_sources = resolve_moments_with_sources(
                structure,
                indices,
                moment,
                site_moment_file=site_moment_file,
                coordinations=coordinations,
            )
            cn_by_element: dict[str, set[int]] = {}
            display_names: dict[str, str] = {}
            for index in indices:
                symbol = structure.symbols[index]
                normalized = symbol.lower()
                display_names.setdefault(normalized, symbol)
                cn_by_element.setdefault(normalized, set()).add(coordinations[index])
            for element, coordination_values in cn_by_element.items():
                if len(coordination_values) < 2 or not any(
                    resolved_moment_sources[index] in {"element", "global"}
                    for index in indices
                    if structure.symbols[index].lower() == element
                ):
                    continue
                display = display_names[element]
                ordered = sorted(coordination_values)
                sites = " and ".join(f"CN={value}" for value in ordered)
                specifications = " and ".join(
                    f"{display}@{value}=..." for value in ordered
                )
                qualifier = "both " if len(ordered) == 2 else ""
                sublattices = (
                    "the two sublattices" if len(ordered) == 2 else "the sites"
                )
                assignment.warnings.append(
                    f"element {display} occupies {qualifier}{sites} sites but its initial "
                    "moment was taken from a single value; use "
                    f"{specifications} to set {sublattices} independently"
                )
        elif default_moments is not None:
            resolved_magnitudes, resolved_moment_sources = resolve_moments_with_sources(
                structure,
                indices,
                moment,
                site_moment_file=site_moment_file,
                coordinations=(
                    coordinations if isinstance(coordinations, dict) else None
                ),
            )
        else:
            resolved_magnitudes = resolve_moments(
                structure,
                indices,
                moment,
                site_moment_file=site_moment_file,
                coordinations=(
                    coordinations if isinstance(coordinations, dict) else None
                ),
            )
    if default_moments is not None:
        applied_defaults = {
            element: value
            for element, value in default_moments.items()
            if any(
                structure.symbols[index].lower() == element.lower()
                and resolved_moment_sources is not None
                and resolved_moment_sources[index] != "site"
                for index in indices
            )
        }
        applied = ", ".join(
            f"{element}={value:.1f}" for element, value in applied_defaults.items()
        ) or "none (all selected sites use site-moment overrides)"
        assignment.warnings.insert(
            0,
            "using built-in default initial moments (generic high-spin guesses): "
            f"{applied}. These ignore oxidation and spin state -- e.g. low-spin Co3+ "
            "is ~0 -- and are only a starting guess; pass --moment to set them "
            "explicitly.",
        )
    return indices, assignment, assignment.moments(resolved_magnitudes)


def _canonical_pattern(
    signs: dict[int, int], keep_global_spin_inversion: bool
) -> tuple[int, ...]:
    pattern = tuple(signs[index] for index in sorted(signs))
    if keep_global_spin_inversion:
        return pattern
    inverse = tuple(-value for value in pattern)
    return min(pattern, inverse)


def enumerate_candidates(
    structure: Structure,
    magnetic_species: Sequence[str],
    methods: Sequence[str],
    moment: Sequence[str] | str | None,
    n_configs: int,
    output_dir: str | Path,
    *,
    keep_global_spin_inversion: bool = False,
    site_comments: bool = True,
    seed_offset: int = 0,
    **workflow_kwargs: Any,
) -> EnumerationResult:
    """Generate distinct candidate spin blocks and their manifest.

    This is the reusable, UI-independent implementation behind the CLI and GUI.
    Diagnostics are returned to the caller instead of being printed.
    """

    if n_configs <= 0:
        raise ValueError("--n-configs must be positive")
    normalized_methods = [str(item).strip() for item in methods if str(item).strip()]
    if not normalized_methods:
        raise ValueError("--methods must contain at least one method")
    allowed = set(ENUMERATION_METHODS)
    unknown = [method for method in normalized_methods if method not in allowed]
    if unknown:
        raise ValueError(f"unsupported enumeration method: {unknown[0]}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, ...]] = set()
    manifest: list[dict[str, object]] = []
    failures: list[str] = []
    notices: list[str] = []
    written_files: list[Path] = []
    max_attempts = max(n_configs * 20, len(normalized_methods))
    for attempt in range(max_attempts):
        if len(manifest) >= n_configs:
            break
        method = normalized_methods[attempt % len(normalized_methods)]
        try:
            indices, assignment, spins = generate_assignment(
                structure,
                magnetic_species,
                method,
                moment,
                seed=seed_offset + attempt,
                **workflow_kwargs,
            )
        except (ValueError, NonBipartiteError) as exc:
            message = f"{method}: {exc}"
            if message not in failures:
                failures.append(message)
            continue
        key = _canonical_pattern(assignment.signs, keep_global_spin_inversion)
        if key in seen:
            continue
        try:
            graph, _, _ = build_neighbor_graph(
                structure,
                indices,
                workflow_kwargs.get("cutoff", "auto"),
                neighbor_shell=workflow_kwargs.get("neighbor_shell", 1),
            )
        except ValueError as exc:
            message = f"{method}: {exc}"
            if message not in failures:
                failures.append(message)
            continue
        score = (
            sum(
                assignment.signs[left] * assignment.signs[right] < 0
                for left, right in graph.edges
            )
            / graph.number_of_edges()
            if graph.number_of_edges()
            else 0.0
        )
        seen.add(key)
        config_id = f"{len(manifest) + 1:03d}"
        file_name = f"afm_{config_id}.fdf"
        text = render_dm_init_spin(
            spins,
            method=assignment.method,
            magnetic_species=magnetic_species,
            metadata=assignment.metadata,
            structure=structure,
            site_comments=site_comments,
        )
        spin_path = destination / file_name
        spin_path.write_text(text, encoding="utf-8")
        written_files.append(spin_path)
        manifest.append(
            {
                "config_id": config_id,
                "method": assignment.method,
                "n_up": assignment.n_up,
                "n_down": assignment.n_down,
                "net_spin": sum(spins.values()),
                "afm_score": score,
                "file": file_name,
            }
        )
        for warning in assignment.warnings:
            if warning not in notices:
                notices.append(warning)
    if not manifest:
        detail = "\n".join(failures) if failures else "no distinct patterns"
        raise ValueError(f"no AFM configurations could be generated:\n{detail}")

    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    if len(manifest) < n_configs:
        notices.append(
            f"requested {n_configs}, but only {len(manifest)} distinct patterns "
            "were found."
        )
    return EnumerationResult(
        manifest=manifest,
        failures=failures,
        notices=notices,
        written_files=written_files,
        manifest_path=manifest_path,
    )
