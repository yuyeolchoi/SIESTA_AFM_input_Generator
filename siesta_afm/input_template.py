"""Render a complete, order-preserving SIESTA starting input."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from ase.data import atomic_numbers

from .fdf_writer import render_dm_init_spin
from .structure import Structure


TEMPLATE_WARNING = (
    "generated SIESTA input is a starting template only; converge and validate "
    "the basis, pseudopotentials, k-grid, MeshCutoff, SCF settings, Hubbard U, "
    "and magnetic state before using results in a publication"
)

# Materials Project calibrated oxide values (eV).  They are useful starting
# points, not transferable constants for every chemistry or code setup.
DEFAULT_HUBBARD_U: dict[str, float] = {
    "V": 3.25,
    "Cr": 3.70,
    "Mn": 3.90,
    "Fe": 5.30,
    "Co": 3.32,
    "Ni": 6.20,
    "Mo": 4.38,
    "W": 6.20,
}


@dataclass(frozen=True, slots=True)
class InputTemplateResult:
    """Rendered input plus decisions that callers should surface to users."""

    text: str
    warnings: tuple[str, ...]
    kgrid: tuple[int, int, int]
    species_ids: tuple[int, ...]


def parse_hubbard_u(
    values: Sequence[str] | Mapping[str, float] | None,
) -> dict[str, float]:
    """Parse ``Element=U`` and ``Element@CN=U`` Hubbard-U overrides."""

    if values is None:
        return {}
    items = values.items() if isinstance(values, Mapping) else _split_assignments(values)
    parsed: dict[str, float] = {}
    for raw_key, raw_value in items:
        key = _canonical_hubbard_key(str(raw_key))
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"Hubbard U must be nonnegative: {key}={value:g}")
        parsed[key] = value
    return parsed


def _split_assignments(values: Sequence[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for raw in values:
        for token in str(raw).replace(",", " ").split():
            if "=" not in token:
                raise ValueError(
                    f"invalid Hubbard U specification {token!r}; use "
                    "Element=value or Element@CN=value"
                )
            key, value = token.split("=", 1)
            items.append((key, value))
    return items


def _canonical_element(value: str) -> str:
    element = value.strip().capitalize()
    if element not in atomic_numbers:
        raise ValueError(f"unknown element in Hubbard U specification: {value}")
    return element


def _canonical_hubbard_key(value: str) -> str:
    key = value.strip()
    if "@" not in key:
        return _canonical_element(key)
    element_text, coordination_text = key.split("@", 1)
    element = _canonical_element(element_text)
    try:
        coordination = int(coordination_text)
    except ValueError as exc:
        raise ValueError(
            "coordination in Hubbard U specification must be an integer: "
            f"{value}"
        ) from exc
    if coordination < 0:
        raise ValueError(
            "coordination in Hubbard U specification must be nonnegative: "
            f"{value}"
        )
    return f"{element}@{coordination}"


def _hubbard_key_parts(key: str) -> tuple[str, int | None]:
    if "@" not in key:
        return key, None
    element, coordination = key.split("@", 1)
    return element, int(coordination)


def automatic_kgrid(
    structure: Structure, cutoff: float = 30.0
) -> tuple[int, int, int]:
    """Choose ``ceil(cutoff / |a_i|)`` on periodic axes and Gamma otherwise."""

    if cutoff <= 0:
        raise ValueError("k-grid cutoff must be positive")
    result: list[int] = []
    for axis in range(3):
        length = float(np.linalg.norm(structure.cell[axis]))
        if structure.pbc[axis]:
            if length <= 1e-12:
                raise ValueError(
                    f"periodic axis {'xyz'[axis]} requires a nonzero lattice vector"
                )
            result.append(max(1, ceil(cutoff / length)))
        else:
            result.append(1)
    return result[0], result[1], result[2]


def render_complete_input(
    structure: Structure,
    spins: Mapping[int, float],
    *,
    method: str,
    magnetic_species: Sequence[str],
    metadata: Mapping[str, object] | None = None,
    angles: Mapping[int, tuple[float, float]] | None = None,
    basis_size: str = "DZP",
    kgrid_cutoff: float = 30.0,
    kgrid: Sequence[int] | None = None,
    hubbard_u: Sequence[str] | Mapping[str, float] | None = None,
    lda_u: bool = True,
    dftu_keyword: str = "ldau",
    system_name: str | None = None,
    write_zero_spins: bool = False,
    site_comments: bool = True,
    split_species_by_coordination: bool = False,
) -> InputTemplateResult:
    """Render a self-contained SIESTA FDF input without reordering atoms."""

    basis_size = basis_size.upper()
    if basis_size not in {"SZ", "SZP", "DZ", "DZP", "TZP"}:
        raise ValueError("basis size must be one of SZ, SZP, DZ, DZP, or TZP")
    keyword = dftu_keyword.lower()
    if keyword not in {"ldau", "dftu"}:
        raise ValueError("DFT+U keyword must be 'ldau' or 'dftu'")

    metadata = dict(metadata or {})
    overrides = parse_hubbard_u(hubbard_u)
    has_coordination_u = any("@" in key for key in overrides)
    if has_coordination_u and not split_species_by_coordination:
        raise ValueError(
            "@CN Hubbard U requires --split-species-by-coordination"
        )
    if split_species_by_coordination and method != "by-coordination":
        raise ValueError(
            "--split-species-by-coordination requires --method by-coordination"
        )

    species_comments: dict[int, str] = {}
    species_coordinations: dict[int, int] | None = None
    split_labels_by_element: dict[str, list[str]] = {}
    if split_species_by_coordination:
        (
            species_rows,
            atom_species_ids,
            species_comments,
            species_coordinations,
            split_labels_by_element,
        ) = _coordination_split_species(
            structure, magnetic_species, metadata
        )
    else:
        species_rows, atom_species_ids = _species_rows(structure)
    selected_grid = _validated_kgrid(kgrid) if kgrid is not None else automatic_kgrid(
        structure, kgrid_cutoff
    )
    name = system_name or _default_system_name(structure)
    label = _system_label(name)

    required_elements = ", ".join(dict.fromkeys(structure.symbols))
    required_pseudos = ", ".join(
        f"{species_label}.psml/{species_label}.psf"
        for _, _, species_label in species_rows
    )
    header_lines = [
        "# Complete SIESTA starting input generated by siesta-afm",
        f"# WARNING: {TEMPLATE_WARNING}.",
        "# Every automatically selected numerical value below is annotated;",
        "# replace it only after system-specific convergence tests.",
        f"# Requires pseudopotentials for: {required_elements} in the run directory.",
        f"# Expected filenames by species label: {required_pseudos}.",
        "# Prefer tested PSML for SIESTA 5; do not keep a same-label PSF beside it",
        "# because SIESTA gives PSF files precedence during discovery.",
        "# Pseudopotential guidance: https://siesta-project.org/siesta/Documentation/Pseudopotentials/",
    ]
    for element, labels in split_labels_by_element.items():
        pseudo_names = _joined_names([f"{item}.psf" for item in labels])
        quantity = "both" if len(labels) == 2 else "all"
        header_lines.append(
            f"# Split-species note: {pseudo_names} must {quantity} be provided "
            f"(copies of the {element} pseudopotential; use matching .psml names "
            "when applicable)."
        )
    sections: list[str] = [
        "\n".join(header_lines),
        _section("system", [f"SystemName {name}", f"SystemLabel {label}"]),
        _structure_section(
            structure,
            species_rows,
            atom_species_ids,
            species_comments=species_comments,
        ),
        _basis_section(basis_size),
        _section(
            "exchange-correlation",
            [
                "XC.functional GGA  # PBE starting functional; keep pseudo and XC consistent",
                "XC.authors PBE",
            ],
        ),
        _scf_section(structure, selected_grid, kgrid_cutoff, kgrid is not None),
        _section(
            "spin initialization",
            render_dm_init_spin(
                spins,
                method=method,
                magnetic_species=magnetic_species,
                metadata=metadata,
                angles=angles,
                structure=structure,
                write_zero_spins=write_zero_spins,
                site_comments=site_comments,
            ).rstrip().splitlines(),
        ),
    ]

    warnings: list[str] = [TEMPLATE_WARNING]
    dftu_lines, dftu_warnings = _dftu_section(
        structure,
        magnetic_species,
        metadata,
        species_rows,
        overrides,
        enabled=lda_u,
        keyword=keyword,
        species_coordinations=species_coordinations,
        split_labels_by_element=split_labels_by_element,
    )
    sections.append(_section("DFT+U", dftu_lines))
    warnings.extend(dftu_warnings)
    sections.append(
        _section(
            "output",
            [
                "WriteMullikenPop 1  # Needed to inspect atom-resolved spin populations",
                "WriteForces T",
                "WriteCoorStep T",
            ],
        )
    )
    return InputTemplateResult(
        text="\n\n".join(sections).rstrip() + "\n",
        warnings=tuple(warnings),
        kgrid=selected_grid,
        species_ids=tuple(atom_species_ids),
    )


def _section(name: str, lines: Sequence[str]) -> str:
    return "\n".join([f"# ---- {name} ----", *lines])


def _species_rows(
    structure: Structure,
    atom_species_ids: Sequence[int] | None = None,
) -> tuple[list[tuple[int, int, str]], list[int]]:
    atom_ids: list[int] = []
    if atom_species_ids is not None:
        if len(atom_species_ids) != len(structure) or any(
            not isinstance(value, int) or value <= 0 for value in atom_species_ids
        ):
            raise ValueError(
                "atom species IDs must contain one positive integer per atom"
            )
        atom_ids = [int(value) for value in atom_species_ids]
    else:
        original_ids = list(structure.species_ids or [])
        preserve = len(original_ids) == len(structure) and all(
            isinstance(value, int) and value > 0 for value in original_ids
        )
        if preserve:
            atom_ids = [int(value) for value in original_ids]
        else:
            by_element: dict[str, int] = {}
            for symbol in structure.symbols:
                by_element.setdefault(symbol, len(by_element) + 1)
                atom_ids.append(by_element[symbol])

    symbol_by_id: dict[int, str] = {}
    order: list[int] = []
    for symbol, species_id in zip(structure.symbols, atom_ids, strict=True):
        if species_id in symbol_by_id and symbol_by_id[species_id] != symbol:
            raise ValueError(
                f"species id {species_id} maps to both {symbol_by_id[species_id]} and {symbol}"
            )
        if species_id not in symbol_by_id:
            symbol_by_id[species_id] = symbol
            order.append(species_id)

    element_multiplicity: dict[str, int] = {}
    for species_id in order:
        symbol = symbol_by_id[species_id]
        element_multiplicity[symbol] = element_multiplicity.get(symbol, 0) + 1
    rows = [
        (
            species_id,
            atomic_numbers[symbol_by_id[species_id]],
            (
                symbol_by_id[species_id]
                if element_multiplicity[symbol_by_id[species_id]] == 1
                else f"{symbol_by_id[species_id]}_{species_id}"
            ),
        )
        for species_id in order
    ]
    return rows, atom_ids


def _coordination_split_species(
    structure: Structure,
    magnetic_species: Sequence[str],
    metadata: Mapping[str, object],
) -> tuple[
    list[tuple[int, int, str]],
    list[int],
    dict[int, str],
    dict[int, int],
    dict[str, list[str]],
]:
    raw_coordinations = metadata.get("coordination_numbers")
    if not isinstance(raw_coordinations, Mapping):
        raise ValueError(
            "--split-species-by-coordination requires coordination_numbers metadata"
        )
    coordinations = {
        int(index): int(value) for index, value in raw_coordinations.items()
    }
    selected = {value.strip().lower() for value in magnetic_species if value.strip()}
    cn_by_element: dict[str, set[int]] = {}
    display_elements: dict[str, str] = {}
    for index, coordination in coordinations.items():
        if index < 0 or index >= len(structure):
            raise ValueError(
                f"coordination metadata atom index is out of range: {index}"
            )
        symbol = structure.symbols[index]
        normalized = symbol.lower()
        if normalized in selected:
            display_elements.setdefault(normalized, symbol)
            cn_by_element.setdefault(normalized, set()).add(coordination)
    split_elements = [
        element for element, values in cn_by_element.items() if len(values) > 1
    ]

    _, original_atom_ids = _species_rows(structure)
    group_ids: dict[tuple[object, ...], int] = {}
    group_by_id: dict[int, tuple[object, ...]] = {}
    atom_ids: list[int] = []
    for index, (symbol, original_id) in enumerate(
        zip(structure.symbols, original_atom_ids, strict=True)
    ):
        normalized = symbol.lower()
        if normalized in split_elements and index in coordinations:
            group: tuple[object, ...] = (
                "coordination",
                normalized,
                coordinations[index],
            )
        else:
            group = ("original", original_id)
        if group not in group_ids:
            species_id = len(group_ids) + 1
            group_ids[group] = species_id
            group_by_id[species_id] = group
        atom_ids.append(group_ids[group])

    rows, atom_ids = _species_rows(structure, atom_ids)
    species_coordinations: dict[int, int] = {}
    for species_id in group_by_id:
        values = {
            coordinations[index]
            for index, atom_species_id in enumerate(atom_ids)
            if atom_species_id == species_id and index in coordinations
        }
        if len(values) == 1:
            species_coordinations[species_id] = next(iter(values))

    raw_geometries = metadata.get("coordination_geometry")
    geometries = raw_geometries if isinstance(raw_geometries, Mapping) else {}
    species_comments: dict[int, str] = {}
    for species_id, group in group_by_id.items():
        if group[0] != "coordination":
            continue
        normalized = str(group[1])
        coordination = int(group[2])
        geometry_values = list(
            dict.fromkeys(
                str(geometries[index])
                for index, atom_species_id in enumerate(atom_ids)
                if atom_species_id == species_id and index in geometries
            )
        )
        geometry = f" ({geometry_values[0]})" if len(geometry_values) == 1 else ""
        species_comments[species_id] = (
            f"{display_elements[normalized]} CN={coordination}{geometry}"
        )

    label_by_id = {species_id: label for species_id, _, label in rows}
    split_labels_by_element: dict[str, list[str]] = {}
    for normalized in split_elements:
        element = display_elements[normalized]
        labels = [
            label_by_id[species_id]
            for species_id, group in group_by_id.items()
            if (
                (group[0] == "coordination" and group[1] == normalized)
                or (
                    group[0] == "original"
                    and any(
                        atom_species_id == species_id
                        and structure.symbols[index].lower() == normalized
                        for index, atom_species_id in enumerate(atom_ids)
                    )
                )
            )
        ]
        split_labels_by_element[element] = labels
    return (
        rows,
        atom_ids,
        species_comments,
        species_coordinations,
        split_labels_by_element,
    )


def _joined_names(values: Sequence[str]) -> str:
    if len(values) < 2:
        return "".join(values)
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _structure_section(
    structure: Structure,
    species_rows: Sequence[tuple[int, int, str]],
    atom_species_ids: Sequence[int],
    *,
    species_comments: Mapping[int, str] | None = None,
) -> str:
    lines = [
        f"NumberOfAtoms {len(structure)}",
        f"NumberOfSpecies {len(species_rows)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    comments = species_comments or {}
    for species_id, number, label in species_rows:
        line = f"{species_id:4d} {number:4d} {label}"
        if species_id in comments:
            line += f"  # {comments[species_id]}"
        lines.append(line)
    lines.append("%endblock ChemicalSpeciesLabel")
    if np.any(np.abs(structure.cell) > 1e-12):
        lines.extend(["", "LatticeConstant 1.0 Ang", "%block LatticeVectors"])
        lines.extend(" ".join(f"{value:16.10f}" for value in vector) for vector in structure.cell)
        lines.append("%endblock LatticeVectors")
    else:
        lines.extend(
            [
                "",
                "# No lattice was present in the source; LatticeVectors is omitted",
                "# for this nonperiodic structure.",
            ]
        )
    lines.extend(["", "AtomicCoordinatesFormat Ang", "%block AtomicCoordinatesAndAtomicSpecies"])
    for index, (symbol, position, species_id) in enumerate(
        zip(structure.symbols, structure.positions, atom_species_ids, strict=True),
        start=1,
    ):
        lines.append(
            " ".join(f"{value:16.10f}" for value in position)
            + f" {species_id:4d}  # {symbol} {index}"
        )
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    return _section("species and structure", lines)


def _basis_section(basis_size: str) -> str:
    return _section(
        "basis and pseudopotentials",
        [
            f"PAO.BasisSize {basis_size}  # Starting basis; converge size for energies and moments",
            "PAO.EnergyShift 100 meV  # Starting confinement; converge it with the basis",
            "# Use tested, XC-consistent pseudopotentials; curated Pseudo-Dojo PSML",
            "# files are recommended by current SIESTA documentation.",
        ],
    )


def _scf_section(
    structure: Structure,
    grid: tuple[int, int, int],
    cutoff: float,
    explicit_grid: bool,
) -> str:
    if explicit_grid:
        grid_reason = "explicit --kgrid override"
    else:
        grid_reason = (
            f"ceil({cutoff:g} Ang / lattice-vector length) on periodic axes; "
            "1 on nonperiodic axes"
        )
    lines = [
        "MeshCutoff 400 Ry  # Starting point (typical 150-600 Ry); converge for this pseudo/basis",
        "ElectronicTemperature 300 K  # Starting smearing; validate for the electronic state",
        "MaxSCFIterations 300  # Extra headroom for slow magnetic-oxide SCF cycles",
        "SCF.Mixer.Method Pulay  # Modern SIESTA mixer; convergence remains system-dependent",
        "SCF.Mixer.Weight 0.10  # Conservative starting weight; tune if SCF is unstable/slow",
        "SCF.Mixer.History 5  # Starting history length; test together with mixer weight",
        "SCF.DM.Tolerance 1.d-4  # Starting tolerance; tighten for final reported quantities",
        "SolutionMethod diagon  # Conservative starting solver; validate scaling for large cells",
        "",
        f"# Selected k-grid {grid[0]} {grid[1]} {grid[2]}: {grid_reason}.",
        "%block kgrid_Monkhorst_Pack",
        f"{grid[0]:4d} 0 0 0.0",
        f"0 {grid[1]:4d} 0 0.0",
        f"0 0 {grid[2]:4d} 0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "# Converge the k-grid against energies, forces, and local magnetic moments.",
    ]
    if not any(structure.pbc):
        lines.append("# All axes are nonperiodic, so Gamma-only sampling is used.")
    return _section("SCF and calculation", lines)


def _validated_kgrid(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("k-grid must contain exactly three integers")
    grid = tuple(int(value) for value in values)
    if any(value <= 0 for value in grid):
        raise ValueError("k-grid values must be positive")
    return grid[0], grid[1], grid[2]


def _dftu_section(
    structure: Structure,
    magnetic_species: Sequence[str],
    metadata: Mapping[str, object],
    species_rows: Sequence[tuple[int, int, str]],
    overrides: Mapping[str, float],
    *,
    enabled: bool,
    keyword: str,
    species_coordinations: Mapping[int, int] | None = None,
    split_labels_by_element: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[str], list[str]]:
    if not enabled:
        return ["# DFT+U omitted by request (--no-lda-u)."], []
    if species_coordinations is not None:
        return _coordination_dftu_section(
            structure,
            magnetic_species,
            species_rows,
            overrides,
            species_coordinations,
            split_labels_by_element or {},
            keyword=keyword,
        )

    present = {symbol.lower(): symbol for symbol in structure.symbols}
    selected: list[str] = []
    for raw in magnetic_species:
        normalized = raw.lower()
        if normalized not in present:
            raise ValueError(f"magnetic species {raw!r} is not present in the structure")
        symbol = present[normalized]
        if symbol not in selected:
            selected.append(symbol)
    unexpected = [element for element in overrides if element.lower() not in {v.lower() for v in selected}]
    if unexpected:
        raise ValueError(
            "Hubbard U override is not a selected magnetic species: "
            + ", ".join(unexpected)
        )

    values: list[tuple[str, int, float]] = []
    unsupported: list[str] = []
    for symbol in selected:
        canonical = _canonical_element(symbol)
        value = overrides.get(canonical, DEFAULT_HUBBARD_U.get(canonical))
        shell = _d_shell(canonical)
        if value is None or shell is None:
            unsupported.append(canonical)
        else:
            values.append((canonical, shell, value))

    lines = [
        "# Dudarev form: U_eff = U - J; this template writes U=U_eff and J=0.",
        "# Dudarev et al., Phys. Rev. B 57, 1505 (1998): https://doi.org/10.1103/PhysRevB.57.1505",
        "# Default U values are Materials Project oxide calibrations:",
        "# https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga%2Bu-calculations/hubbard-u-values",
        "# U can change the converged spin/ground state; validate it for the chemistry,",
        "# pseudopotential, projector definition, and target observable.",
    ]
    lines.extend(f"# no default U for {symbol}; pass --hubbard-u {symbol}=... to opt in" for symbol in unsupported)
    if values:
        block = "LDAU.proj" if keyword == "ldau" else "DFTU.Proj"
        labels_by_element: dict[str, list[str]] = {}
        for _, number, label in species_rows:
            element = next(
                symbol
                for symbol in dict.fromkeys(structure.symbols)
                if atomic_numbers[symbol] == number
            )
            labels_by_element.setdefault(element.lower(), []).append(label)
        if keyword == "dftu":
            lines.append("DFTU.ProjectorGenerationMethod 2")
        lines.append(f"%block {block}")
        for symbol, shell, value in values:
            for label in labels_by_element[symbol.lower()]:
                lines.extend(
                    [
                        f"{label} 1  # species label, one correlated shell",
                        f" n={shell} 2  # valence d shell: n={shell}, l=2",
                        f" {value:.2f} 0.00  # U_eff (eV), J=0",
                        " 0.00 0.00  # automatic rc and omega",
                    ]
                )
        lines.append(f"%endblock {block}")
    else:
        lines.append("# No DFT+U projector was generated for the selected species.")

    warnings: list[str] = []
    coordinations = metadata.get("coordination_numbers")
    if isinstance(coordinations, Mapping):
        for symbol, _, _ in values:
            site_values = {
                int(coordinations[index])
                for index, atom_symbol in enumerate(structure.symbols)
                if atom_symbol.lower() == symbol.lower() and index in coordinations
            }
            if len(site_values) > 1:
                warnings.append(
                    f"element {symbol} occupies two coordination sublattices but "
                    "LDA+U is per species; a single U is applied to both. Split "
                    "the species manually or pass "
                    "--split-species-by-coordination if they need different U."
                )
    return lines, warnings


def _coordination_dftu_section(
    structure: Structure,
    magnetic_species: Sequence[str],
    species_rows: Sequence[tuple[int, int, str]],
    overrides: Mapping[str, float],
    species_coordinations: Mapping[int, int],
    split_labels_by_element: Mapping[str, Sequence[str]],
    *,
    keyword: str,
) -> tuple[list[str], list[str]]:
    present = {symbol.lower(): symbol for symbol in structure.symbols}
    selected: list[str] = []
    for raw in magnetic_species:
        normalized = raw.lower()
        if normalized not in present:
            raise ValueError(f"magnetic species {raw!r} is not present in the structure")
        symbol = present[normalized]
        if symbol not in selected:
            selected.append(symbol)
    selected_lower = {value.lower() for value in selected}
    unexpected = [
        key
        for key in overrides
        if _hubbard_key_parts(key)[0].lower() not in selected_lower
    ]
    if unexpected:
        raise ValueError(
            "Hubbard U override is not a selected magnetic species: "
            + ", ".join(unexpected)
        )

    labels_by_element: dict[str, list[tuple[int, str]]] = {}
    for species_id, number, label in species_rows:
        element = next(
            symbol
            for symbol in dict.fromkeys(structure.symbols)
            if atomic_numbers[symbol] == number
        )
        labels_by_element.setdefault(element.lower(), []).append((species_id, label))
    available_coordination_keys = {
        (element, species_coordinations[species_id])
        for element, labels in labels_by_element.items()
        for species_id, _ in labels
        if species_id in species_coordinations
    }
    unmatched = [
        key
        for key in overrides
        if (
            (parts := _hubbard_key_parts(key))[1] is not None
            and (parts[0].lower(), parts[1]) not in available_coordination_keys
        )
    ]
    if unmatched:
        raise ValueError(
            "Hubbard U override has no matching coordination species: "
            + ", ".join(unmatched)
        )

    projector_values: list[tuple[str, int, float]] = []
    unsupported: list[str] = []
    for symbol in selected:
        canonical = _canonical_element(symbol)
        shell = _d_shell(canonical)
        generated = False
        for species_id, label in labels_by_element.get(symbol.lower(), []):
            coordination = species_coordinations.get(species_id)
            specific_key = (
                f"{canonical}@{coordination}" if coordination is not None else None
            )
            value = (
                overrides[specific_key]
                if specific_key is not None and specific_key in overrides
                else overrides.get(canonical, DEFAULT_HUBBARD_U.get(canonical))
            )
            if value is not None and shell is not None:
                projector_values.append((label, shell, value))
                generated = True
        if not generated:
            unsupported.append(canonical)

    lines = [
        "# Dudarev form: U_eff = U - J; this template writes U=U_eff and J=0.",
        "# Dudarev et al., Phys. Rev. B 57, 1505 (1998): https://doi.org/10.1103/PhysRevB.57.1505",
        "# Default U values are Materials Project oxide calibrations:",
        "# https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga%2Bu-calculations/hubbard-u-values",
        "# U can change the converged spin/ground state; validate it for the chemistry,",
        "# pseudopotential, projector definition, and target observable.",
    ]
    lines.extend(
        f"# no default U for {symbol}; pass --hubbard-u {symbol}=... to opt in"
        for symbol in unsupported
    )
    if projector_values:
        block = "LDAU.proj" if keyword == "ldau" else "DFTU.Proj"
        if keyword == "dftu":
            lines.append("DFTU.ProjectorGenerationMethod 2")
        lines.append(f"%block {block}")
        for label, shell, value in projector_values:
            lines.extend(
                [
                    f"{label} 1  # species label, one correlated shell",
                    f" n={shell} 2  # valence d shell: n={shell}, l=2",
                    f" {value:.2f} 0.00  # U_eff (eV), J=0",
                    " 0.00 0.00  # automatic rc and omega",
                ]
            )
        lines.append(f"%endblock {block}")
    else:
        lines.append("# No DFT+U projector was generated for the selected species.")

    warnings: list[str] = []
    for element, labels in split_labels_by_element.items():
        canonical = _canonical_element(element)
        if canonical not in overrides:
            continue
        has_specific = any(
            _hubbard_key_parts(key)[0] == canonical and "@" in key
            for key in overrides
        )
        if not has_specific:
            warnings.append(
                f"plain Hubbard U {canonical}={overrides[canonical]:g} applies the "
                f"same U to split species {_joined_names(list(labels))}."
            )
    return lines, warnings


def _d_shell(element: str) -> int | None:
    number = atomic_numbers[element]
    if 21 <= number <= 30:
        return 3
    if 39 <= number <= 48:
        return 4
    if 72 <= number <= 80:
        return 5
    return None


def _default_system_name(structure: Structure) -> str:
    if structure.source:
        return Path(structure.source).stem or "siesta_afm"
    return "siesta_afm"


def _system_label(name: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_.-")
    return label or "siesta_afm"
