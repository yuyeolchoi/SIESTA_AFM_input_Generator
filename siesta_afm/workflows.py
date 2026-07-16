"""Reusable high-level spin-generation workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from .fdf_writer import render_dm_init_spin

from .magnetic_sites import (
    built_in_element_moments,
    parse_atom_indices,
    parse_moment_arguments,
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
    manual_spins_ordering,
    neighbor_bipartite_ordering,
    propagation_vector_ordering,
    random_ordering,
    read_group_file,
)
from .neighbors import build_neighbor_graph
from .structure import Structure
from .symmetry import structure_symmetry_permutations


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

ENUMERATION_CONFIG_LIMIT = 200


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
    layer_per_species: bool = False,
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
    spin_values: dict[int, float] | None = None,
    fill_unspecified_zero: bool = False,
    spin_mode: str = "collinear",
    seed: int = 0,
) -> tuple[list[int], SpinAssignment, dict[int, float]]:
    if layer_per_species and method != "layer":
        raise ValueError("--layer-per-species requires --method layer")
    if spin_mode not in {"collinear", "non-collinear"}:
        raise ValueError("spin mode must be 'collinear' or 'non-collinear'")
    if spin_mode == "non-collinear" and method != "graph-coloring":
        raise ValueError(
            "non-collinear generation is only supported with --method "
            "graph-coloring; other assignment methods only produce collinear "
            "±1 signs"
        )
    if spin_mode == "non-collinear" and color_spins is not None:
        raise ValueError(
            "--spin-mode non-collinear cannot be combined with --color-spins"
        )
    indices = select_magnetic_sites(
        structure,
        magnetic_species,
        exclude_atoms=exclude_atoms,
        adsorbate_indices=adsorbate_indices,
    )
    default_moments: dict[str, float] | None = None
    if method == "manual-spins":
        if spin_values is None:
            raise ValueError(
                "manual-spins requires --spin-values or --spin-values-file"
            )
        if moment is not None:
            raise ValueError(
                "--moment/--moment-config cannot be combined with --method "
                "manual-spins; the supplied spins are already signed values"
            )
        if site_moment_file is not None:
            raise ValueError(
                "--site-moment-file cannot be combined with --method manual-spins; "
                "use --spin-values-file for signed values"
            )
    else:
        if spin_values is not None:
            raise ValueError("--spin-values/--spin-values-file require --method manual-spins")
        if fill_unspecified_zero:
            raise ValueError("--fill-unspecified-zero requires --method manual-spins")
    if moment is None and method != "manual-spins":
        default_moments = built_in_element_moments(structure, indices)
        moment = [
            f"{element}={value:.1f}" for element, value in default_moments.items()
        ]
    resolved_magnitudes: dict[int, float] | None = None
    resolved_moment_sources: dict[int, str] | None = None
    if method == "manual-spins":
        assignment = manual_spins_ordering(
            structure,
            indices,
            spin_values or {},
            fill_unspecified_zero=fill_unspecified_zero,
        )
    elif method == "alternating-index":
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
                structure,
                indices,
                layer_direction,
                tolerance=layer_tolerance,
                per_species=layer_per_species,
            )
        else:
            assignment = layer_ordering(
                structure,
                indices,
                axis=axis,
                tolerance=layer_tolerance,
                fractional=fractional_layers,
                per_species=layer_per_species,
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
    if method == "manual-spins":
        direct_spins = assignment.metadata.get("spin_values")
        if not isinstance(direct_spins, dict):
            raise RuntimeError("manual-spins assignment is missing direct spin values")
        return indices, assignment, {
            int(index): float(value) for index, value in direct_spins.items()
        }
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
    spins = assignment.moments(resolved_magnitudes)
    if spin_mode == "non-collinear":
        assignment.metadata["spin_mode"] = spin_mode
        spins = {
            index: abs(float(resolved_magnitudes[index]))
            for index in indices
        }
    return indices, assignment, spins


def _canonical_pattern(
    signs: dict[int, int],
    keep_global_spin_inversion: bool,
    symmetry_permutations: Sequence[tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    pattern = tuple(signs[index] for index in sorted(signs))
    if not symmetry_permutations:
        if keep_global_spin_inversion:
            return pattern
        inverse = tuple(-value for value in pattern)
        return min(pattern, inverse)

    equivalent_patterns: list[tuple[int, ...]] = []
    for permutation in symmetry_permutations:
        if len(permutation) != len(pattern):
            continue
        transformed = [0] * len(pattern)
        for source_index, target_index in enumerate(permutation):
            transformed[target_index] = pattern[source_index]
        equivalent_patterns.append(tuple(transformed))
    if not equivalent_patterns:
        equivalent_patterns.append(pattern)
    if keep_global_spin_inversion:
        return min(equivalent_patterns)
    inverted_patterns = [
        tuple(-value for value in item) for item in equivalent_patterns
    ]
    return min(*equivalent_patterns, *inverted_patterns)


def _project_symmetry_permutations(
    permutations: Sequence[tuple[int, ...]], indices: Sequence[int]
) -> list[tuple[int, ...]]:
    """Restrict full-structure source-to-target permutations to selected sites."""

    ordered_indices = tuple(sorted(indices))
    local_index = {
        atom_index: index for index, atom_index in enumerate(ordered_indices)
    }
    identity = tuple(range(len(ordered_indices)))
    projected = [identity]
    seen = {identity}
    for permutation in permutations:
        try:
            item = tuple(
                local_index[permutation[atom_index]] for atom_index in ordered_indices
            )
        except (IndexError, KeyError):
            # Site exclusions can remove a symmetry mate.  Retain only the
            # subgroup that maps the selected magnetic set onto itself.
            continue
        if len(set(item)) != len(ordered_indices) or item in seen:
            continue
        seen.add(item)
        projected.append(item)
    return projected


def apply_coordination_labels(
    structure: Structure,
    indices: Sequence[int],
    assignment: SpinAssignment,
    coordination_labels: Sequence[tuple[str, int, str]],
) -> None:
    """Apply user-edited coordination labels to assignment metadata in place."""

    if assignment.method != "by-coordination" or not coordination_labels:
        return
    coordinations = assignment.metadata.get("coordination_numbers")
    if not isinstance(coordinations, dict):
        return
    labels = {
        (element.lower(), coordination): label
        for element, coordination, label in coordination_labels
        if label
    }
    geometry = dict(assignment.metadata.get("coordination_geometry", {}))
    for index in indices:
        key = (structure.symbols[index].lower(), int(coordinations[index]))
        if key in labels:
            geometry[index] = labels[key]
    assignment.metadata["coordination_geometry"] = geometry


def enumerate_candidates(
    structure: Structure,
    magnetic_species: Sequence[str],
    methods: Sequence[str],
    moment: Sequence[str] | str | None,
    n_configs: int,
    output_dir: str | Path,
    *,
    moment_sweep: Sequence[str] | str | None = None,
    keep_global_spin_inversion: bool = False,
    symmetry_dedup: bool = False,
    symprec: float = 1e-3,
    site_comments: bool = True,
    coordination_labels: Sequence[tuple[str, int, str]] = (),
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

    if moment_sweep is not None:
        return _enumerate_candidates_with_moment_sweep(
            structure,
            magnetic_species,
            normalized_methods,
            moment,
            moment_sweep,
            n_configs,
            output_dir,
            keep_global_spin_inversion=keep_global_spin_inversion,
            symmetry_dedup=symmetry_dedup,
            symprec=symprec,
            site_comments=site_comments,
            coordination_labels=coordination_labels,
            seed_offset=seed_offset,
            **workflow_kwargs,
        )

    magnetic_symmetry_permutations: list[tuple[int, ...]] | None = None
    if symmetry_dedup:
        symmetry_indices = select_magnetic_sites(
            structure,
            magnetic_species,
            exclude_atoms=workflow_kwargs.get("exclude_atoms"),
            adsorbate_indices=workflow_kwargs.get("adsorbate_indices"),
        )
        full_permutations = structure_symmetry_permutations(
            structure, symprec=symprec
        )
        magnetic_symmetry_permutations = _project_symmetry_permutations(
            full_permutations, symmetry_indices
        )

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
            method_workflow_kwargs = dict(workflow_kwargs)
            if method != "layer":
                method_workflow_kwargs["layer_per_species"] = False
            indices, assignment, spins = generate_assignment(
                structure,
                magnetic_species,
                method,
                moment,
                seed=seed_offset + attempt,
                **method_workflow_kwargs,
            )
        except (ValueError, NonBipartiteError) as exc:
            message = f"{method}: {exc}"
            if message not in failures:
                failures.append(message)
            continue
        apply_coordination_labels(
            structure, indices, assignment, coordination_labels
        )
        key = _canonical_pattern(
            assignment.signs,
            keep_global_spin_inversion,
            magnetic_symmetry_permutations,
        )
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


def _moment_sweep_items(values: Sequence[str] | str) -> list[str]:
    source = [values] if isinstance(values, str) else list(values)
    return [
        item
        for value in source
        for item in str(value).split()
        if item
    ]


def _parse_moment_sweep(
    moment: Sequence[str] | str | None,
    moment_sweep: Sequence[str] | str,
) -> list[tuple[str, str | None, list[float]]]:
    """Return ``(display target, element, values)`` sweep definitions."""

    targets: list[tuple[str, str | None, list[float]]] = []
    seen_targets: set[str] = set()
    for item in _moment_sweep_items(moment_sweep):
        has_target = "=" in item
        if has_target:
            raw_target, raw_values = item.split("=", 1)
            raw_target = raw_target.strip()
        else:
            raw_target, raw_values = "", item
        value_texts = raw_values.split(",")
        if (
            (has_target and not raw_target)
            or not raw_values
            or any(not value.strip() for value in value_texts)
        ):
            raise ValueError(f"invalid --moment-sweep specification: {item}")

        values: list[float] = []
        target_name: str | None = None
        target_key: str | None = None
        element: str | None = None
        for value_text in value_texts:
            specification = (
                f"{raw_target}={value_text.strip()}"
                if raw_target
                else value_text.strip()
            )
            try:
                global_value, by_element, by_coordination = parse_moment_arguments(
                    specification
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid --moment-sweep specification: {item}: {exc}"
                ) from exc
            if global_value is not None:
                current_name = "global"
                current_key = "global"
                current_element = None
                parsed_value = global_value
            elif by_element:
                current_element, parsed_value = next(iter(by_element.items()))
                current_name = raw_target
                current_key = current_element
            else:
                (current_element, coordination), parsed_value = next(
                    iter(by_coordination.items())
                )
                element_name = raw_target.split("@", 1)[0].strip()
                current_name = f"{element_name}@{coordination}"
                current_key = f"{current_element}@{coordination}"
            if target_key is None:
                target_name = current_name
                target_key = current_key
                element = current_element
            values.append(parsed_value)

        assert target_name is not None and target_key is not None
        if target_key in seen_targets:
            raise ValueError(f"duplicate --moment-sweep target: {target_name}")
        seen_targets.add(target_key)
        targets.append((target_name, element, values))

    if not targets:
        raise ValueError("--moment-sweep requires at least one specification")

    if moment is not None:
        global_moment, by_element, by_coordination = parse_moment_arguments(moment)
        moment_elements = set(by_element) | {
            element for element, _ in by_coordination
        }
        for target_name, element, _ in targets:
            overlaps = (
                target_name == "global" and global_moment is not None
            ) or (element is not None and element in moment_elements)
            if overlaps:
                raise ValueError(
                    f"--moment and --moment-sweep both specify {target_name}; "
                    "an element may only be supplied by one of them"
                )
    return targets


def _combined_moment_values(
    moment: Sequence[str] | str | None,
    sweep_values: Sequence[str],
) -> list[str]:
    if moment is None:
        return list(sweep_values)
    if isinstance(moment, str):
        baseline = [item for item in moment.replace(",", " ").split() if item]
    else:
        baseline = list(moment)
    return [*baseline, *sweep_values]


def _enumerate_candidates_with_moment_sweep(
    structure: Structure,
    magnetic_species: Sequence[str],
    normalized_methods: Sequence[str],
    moment: Sequence[str] | str | None,
    moment_sweep: Sequence[str] | str,
    n_configs: int,
    output_dir: str | Path,
    *,
    keep_global_spin_inversion: bool,
    symmetry_dedup: bool,
    symprec: float,
    site_comments: bool,
    coordination_labels: Sequence[tuple[str, int, str]],
    seed_offset: int,
    **workflow_kwargs: Any,
) -> EnumerationResult:
    targets = _parse_moment_sweep(moment, moment_sweep)
    combination_count = 1
    for _, _, values in targets:
        combination_count *= len(values)
    requested_count = combination_count * n_configs
    if requested_count > ENUMERATION_CONFIG_LIMIT:
        raise ValueError(
            f"moment sweep combinations ({combination_count}) × --n-configs "
            f"({n_configs}) = {requested_count} exceeds the limit of "
            f"{ENUMERATION_CONFIG_LIMIT} configurations"
        )

    magnetic_symmetry_permutations: list[tuple[int, ...]] | None = None
    if symmetry_dedup:
        symmetry_indices = select_magnetic_sites(
            structure,
            magnetic_species,
            exclude_atoms=workflow_kwargs.get("exclude_atoms"),
            adsorbate_indices=workflow_kwargs.get("adsorbate_indices"),
        )
        full_permutations = structure_symmetry_permutations(
            structure, symprec=symprec
        )
        magnetic_symmetry_permutations = _project_symmetry_permutations(
            full_permutations, symmetry_indices
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    failures: list[str] = []
    notices: list[str] = []
    written_files: list[Path] = []
    value_sets = [values for _, _, values in targets]
    max_attempts = max(n_configs * 20, len(normalized_methods))
    for combination_index, combination in enumerate(product(*value_sets), start=1):
        sweep_columns = {
            f"moment_{target_name}": value
            for (target_name, _, _), value in zip(targets, combination)
        }
        sweep_specs = [
            str(value) if target_name == "global" else f"{target_name}={value}"
            for (target_name, _, _), value in zip(targets, combination)
        ]
        combination_moment = _combined_moment_values(moment, sweep_specs)
        combination_label = ", ".join(
            f"{name}={value}" for name, value in sweep_columns.items()
        )
        seen: set[tuple[int, ...]] = set()
        combination_rows = 0
        for attempt in range(max_attempts):
            if combination_rows >= n_configs:
                break
            method = normalized_methods[attempt % len(normalized_methods)]
            try:
                method_workflow_kwargs = dict(workflow_kwargs)
                if method != "layer":
                    method_workflow_kwargs["layer_per_species"] = False
                indices, assignment, spins = generate_assignment(
                    structure,
                    magnetic_species,
                    method,
                    combination_moment,
                    seed=seed_offset + attempt,
                    **method_workflow_kwargs,
                )
            except (ValueError, NonBipartiteError) as exc:
                message = (
                    f"m{combination_index} ({combination_label}): "
                    f"{method}: {exc}"
                )
                if message not in failures:
                    failures.append(message)
                continue
            apply_coordination_labels(
                structure, indices, assignment, coordination_labels
            )
            key = _canonical_pattern(
                assignment.signs,
                keep_global_spin_inversion,
                magnetic_symmetry_permutations,
            )
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
                message = (
                    f"m{combination_index} ({combination_label}): "
                    f"{method}: {exc}"
                )
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
            combination_rows += 1
            config_id = f"{len(manifest) + 1:03d}"
            file_name = f"afm_{config_id}_m{combination_index}.fdf"
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
                    **sweep_columns,
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
        if combination_rows < n_configs:
            notices.append(
                f"moment sweep m{combination_index} ({combination_label}): requested "
                f"{n_configs}, but only {combination_rows} distinct patterns were found."
            )

    if not manifest:
        detail = "\n".join(failures) if failures else "no distinct patterns"
        raise ValueError(f"no AFM configurations could be generated:\n{detail}")

    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return EnumerationResult(
        manifest=manifest,
        failures=failures,
        notices=notices,
        written_files=written_files,
        manifest_path=manifest_path,
    )
