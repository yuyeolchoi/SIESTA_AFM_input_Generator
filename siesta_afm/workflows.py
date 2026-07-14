"""Reusable high-level spin-generation workflow."""

from __future__ import annotations

from typing import Sequence

from .magnetic_sites import parse_atom_indices, resolve_moments, select_magnetic_sites
from .ordering import (
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
from .structure import Structure


def generate_assignment(
    structure: Structure,
    magnetic_species: Sequence[str],
    method: str,
    moment: Sequence[str] | str,
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
    resolved_magnitudes: dict[int, float] | None = None
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
        resolved_magnitudes = resolve_moments(
            structure,
            indices,
            moment,
            site_moment_file=site_moment_file,
            coordinations=coordinations if isinstance(coordinations, dict) else None,
        )
    return indices, assignment, assignment.moments(resolved_magnitudes)
