"""Reusable high-level spin-generation workflow."""

from __future__ import annotations

from typing import Sequence

from .magnetic_sites import parse_atom_indices, resolve_moments, select_magnetic_sites
from .ordering import (
    SpinAssignment,
    alternating_index,
    checkerboard_ordering,
    layer_ordering,
    manual_groups_ordering,
    neighbor_bipartite_ordering,
    propagation_vector_ordering,
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
    layer_tolerance: float = 0.25,
    fractional_layers: bool = False,
    plane: str = "xy",
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
    allow_frustrated: bool = False,
    q_vector: Sequence[float] | None = None,
    phase: float = 0.0,
    fractional_coordinates: bool = True,
    up_atoms: str | Sequence[str] | None = None,
    down_atoms: str | Sequence[str] | None = None,
    group_file: str | None = None,
    seed: int = 0,
) -> tuple[list[int], SpinAssignment, dict[int, float]]:
    indices = select_magnetic_sites(
        structure,
        magnetic_species,
        exclude_atoms=exclude_atoms,
        adsorbate_indices=adsorbate_indices,
    )
    magnitudes = resolve_moments(
        structure, indices, moment, site_moment_file=site_moment_file
    )
    if method == "alternating-index":
        assignment = alternating_index(indices)
    elif method == "layer":
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
        if q_vector is None:
            raise ValueError("--q-vector is required for propagation-vector")
        assignment = propagation_vector_ordering(
            structure,
            indices,
            q_vector,
            phase=phase,
            fractional_coordinates=fractional_coordinates,
        )
    elif method == "manual-groups":
        if group_file:
            up, down = read_group_file(group_file)
        else:
            up = sorted(parse_atom_indices(up_atoms))
            down = sorted(parse_atom_indices(down_atoms))
        assignment = manual_groups_ordering(indices, up, down)
    else:
        raise ValueError(f"unsupported AFM method: {method}")
    return indices, assignment, assignment.moments(magnitudes)
