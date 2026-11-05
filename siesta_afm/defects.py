"""Vacancy-site enumeration and parent-index bookkeeping for defect models.

This module deliberately stops at *structural* descriptors.  Symmetry
multiplicity, coordination, and layer position say which vacancy sites are
distinct and which magnetic atoms they touch; they do not say which vacancy
forms.  Defect formation energies, charge states, and the final local moments
must come from the SIESTA calculations prepared from these candidates.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .magnetic_sites import select_magnetic_sites
from .neighbors import (
    build_neighbor_graph,
    classify_coordination_geometry,
    count_anion_neighbors,
    cross_pair_distances,
    resolve_first_shell,
)
from .ordering import analyze_coordination_sites, detect_layers
from .structure import Structure
from .symmetry import structure_symmetry_permutations, symmetry_orbits


VACANCY_SCOPE_WARNING = (
    "vacancy candidates are ordered by structural descriptors only (surface "
    "position, coordination number, symmetry multiplicity); they are not defect "
    "formation energies and do not predict which vacancy forms"
)

SLAB_SYMMETRY_NOTICE = (
    "symmetry orbits were detected by treating the slab cell as fully periodic; "
    "this is valid only when the nonperiodic direction contains vacuum padding"
)


@dataclass(frozen=True, slots=True)
class VacancySite:
    """One symmetry-distinct vacancy candidate.

    All atom indices are zero-based; :meth:`to_dict` converts them to the
    one-based SIESTA convention used in CLI output.
    """

    orbit_id: int
    representative: int
    members: tuple[int, ...]
    element: str
    multiplicity: int
    coordination_number: int
    coordination_geometry: str
    neighbor_elements: tuple[tuple[str, int], ...]
    first_shell_atoms: tuple[int, ...]
    affected_magnetic_atoms: tuple[int, ...]
    layer: int | None
    layer_count: int | None
    surface: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "orbit_id": self.orbit_id,
            "representative": self.representative + 1,
            "members": [index + 1 for index in self.members],
            "element": self.element,
            "multiplicity": self.multiplicity,
            "coordination_number": self.coordination_number,
            "coordination_geometry": self.coordination_geometry,
            "neighbor_elements": {
                element: count for element, count in self.neighbor_elements
            },
            "first_shell_atoms": [index + 1 for index in self.first_shell_atoms],
            "affected_magnetic_atoms": [
                index + 1 for index in self.affected_magnetic_atoms
            ],
            "layer": self.layer + 1 if self.layer is not None else None,
            "layer_count": self.layer_count,
            "surface": self.surface,
        }


@dataclass(slots=True)
class VacancyEnumeration:
    """Candidate vacancy sites for one removed species."""

    species: str
    magnetic_species: tuple[str, ...]
    symmetry_dedup: bool
    n_candidates: int
    sites: list[VacancySite]
    notices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "species": self.species,
            "magnetic_species": list(self.magnetic_species),
            "symmetry_dedup": self.symmetry_dedup,
            "n_candidates": self.n_candidates,
            "n_distinct_sites": len(self.sites),
            "sites": [site.to_dict() for site in self.sites],
            "notices": list(self.notices),
        }


def _symmetry_permutations(
    structure: Structure, symprec: float, notices: list[str]
) -> list[tuple[int, ...]]:
    if abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError(
            "symmetry-based vacancy deduplication requires a nonsingular cell; "
            "omit --symmetry-dedup for cell-free structures"
        )
    if all(structure.pbc):
        return structure_symmetry_permutations(structure, symprec=symprec)
    notices.append(SLAB_SYMMETRY_NOTICE)
    return structure_symmetry_permutations(
        structure.with_pbc((True, True, True)), symprec=symprec
    )


def _first_shell(
    structure: Structure, index: int, pool: Sequence[int]
) -> tuple[list[int], list[np.ndarray]]:
    """Return first-shell neighbor basis indices and their vectors."""

    if not pool:
        return [], []
    pairs = cross_pair_distances(structure, [index], pool, all_images=True)
    if not pairs:
        return [], []
    _, shell = resolve_first_shell(pairs)
    return [pair.j for pair in shell], [pair.vector for pair in shell]


def enumerate_vacancy_sites(
    structure: Structure,
    species: str,
    *,
    magnetic_species: Sequence[str] | None = None,
    symmetry_dedup: bool = False,
    symprec: float = 1e-3,
    axis: str = "z",
    layer_tolerance: float = 0.25,
) -> VacancyEnumeration:
    """List distinct single-vacancy candidates for one element.

    The first shell of a candidate is resolved among atoms of *other* elements
    (anions around a cation vacancy, cations around an anion vacancy).  For an
    elemental structure every other atom is eligible.  With
    ``symmetry_dedup`` the candidates are grouped into crystallographic orbits
    through spglib; otherwise every atom is its own candidate.  Sites are
    ordered surface-first, then by increasing coordination number, then by
    representative index.
    """

    wanted = species.strip().lower()
    if not wanted:
        raise ValueError("a vacancy species must be specified")
    candidates = [
        index for index, symbol in enumerate(structure.symbols) if symbol.lower() == wanted
    ]
    if not candidates:
        raise ValueError(f"no atoms of species {species} were found in the structure")
    if len(candidates) == len(structure):
        pool_all = list(range(len(structure)))
    else:
        pool_all = [
            index for index, symbol in enumerate(structure.symbols) if symbol.lower() != wanted
        ]
    magnetic = tuple(item.strip() for item in (magnetic_species or ()) if item.strip())
    magnetic_lower = {item.lower() for item in magnetic}

    notices: list[str] = [VACANCY_SCOPE_WARNING]
    if symmetry_dedup:
        permutations = _symmetry_permutations(structure, symprec, notices)
        orbits = symmetry_orbits(permutations, candidates)
    else:
        orbits = [[index] for index in candidates]

    layers = detect_layers(structure, list(range(len(structure))), axis, layer_tolerance)
    layer_of: dict[int, int] = {
        index: number for number, layer in enumerate(layers) for index in layer
    }
    axis_number = "xyz".index(axis)
    axis_periodic = structure.pbc[axis_number]
    layer_count = len(layers)

    sites: list[VacancySite] = []
    for orbit_id, members in enumerate(orbits, start=1):
        representative = members[0]
        shell_atoms, vectors = _first_shell(structure, representative, pool_all)
        unique_shell = sorted(set(shell_atoms))
        coordination = len(shell_atoms)
        geometry = classify_coordination_geometry(vectors) if vectors else "CN=0"
        neighbor_counts = Counter(structure.symbols[j] for j in shell_atoms)
        affected = tuple(
            index
            for index in unique_shell
            if structure.symbols[index].lower() in magnetic_lower
        )
        if symmetry_dedup and len(members) > 1:
            other_counts = {
                len(_first_shell(structure, member, pool_all)[0]) for member in members[1:]
            }
            if other_counts and other_counts != {coordination}:
                notices.append(
                    f"orbit {orbit_id} ({species}) mixes coordination numbers "
                    f"{sorted(other_counts | {coordination})}; tighten --symprec or "
                    "inspect the structure"
                )
        layer = layer_of.get(representative)
        surface = (
            not axis_periodic
            and layer is not None
            and layer_count > 1
            and layer in {0, layer_count - 1}
        )
        sites.append(
            VacancySite(
                orbit_id=orbit_id,
                representative=representative,
                members=tuple(members),
                element=structure.symbols[representative],
                multiplicity=len(members),
                coordination_number=coordination,
                coordination_geometry=geometry,
                neighbor_elements=tuple(sorted(neighbor_counts.items())),
                first_shell_atoms=tuple(unique_shell),
                affected_magnetic_atoms=affected,
                layer=layer,
                layer_count=layer_count,
                surface=surface,
            )
        )
    sites.sort(key=lambda site: (not site.surface, site.coordination_number, site.representative))
    renumbered = {site.orbit_id: number for number, site in enumerate(sites, start=1)}
    sites = [replace(site, orbit_id=renumbered[site.orbit_id]) for site in sites]
    notices = [
        (
            f"orbit {renumbered[int(notice.split()[1])]}" + notice[len(notice.split()[0]) + len(notice.split()[1]) + 1 :]
            if notice.startswith("orbit ")
            else notice
        )
        for notice in notices
    ]
    if magnetic and not any(site.affected_magnetic_atoms for site in sites):
        notices.append(
            "no candidate has a magnetic atom in its first shell; check "
            "--magnetic-species or the structure's bond lengths"
        )
    return VacancyEnumeration(
        species=structure.symbols[candidates[0]],
        magnetic_species=magnetic,
        symmetry_dedup=symmetry_dedup,
        n_candidates=len(candidates),
        sites=sites,
        notices=notices,
    )


def format_vacancy_table(enumeration: VacancyEnumeration) -> str:
    lines = [
        f"Vacancy species: {enumeration.species} "
        f"({enumeration.n_candidates} atoms, {len(enumeration.sites)} distinct "
        f"{'orbits' if enumeration.symmetry_dedup else 'sites'})",
    ]
    if enumeration.magnetic_species:
        lines.append(f"Magnetic species: {' '.join(enumeration.magnetic_species)}")
    header = (
        f"{'orbit':>5} {'atom':>5} {'mult':>4} {'CN':>3} {'geometry':<22} "
        f"{'layer':>7} {'surface':>7}  neighbors / affected magnetic atoms"
    )
    lines.append(header)
    for site in enumeration.sites:
        neighbors = " ".join(f"{el}{n}" for el, n in site.neighbor_elements) or "-"
        affected = (
            ",".join(str(index + 1) for index in site.affected_magnetic_atoms) or "-"
        )
        layer = (
            f"{site.layer + 1}/{site.layer_count}" if site.layer is not None else "-"
        )
        lines.append(
            f"{site.orbit_id:>5} {site.representative + 1:>5} {site.multiplicity:>4} "
            f"{site.coordination_number:>3} {site.coordination_geometry:<22} "
            f"{layer:>7} {'yes' if site.surface else 'no':>7}  {neighbors} / {affected}"
        )
    lines.append(
        "Use `siesta-afm make-defect STRUCTURE --remove-atoms <atom>` with a "
        "representative atom index to build the vacancy model."
    )
    return "\n".join(lines)


def apply_vacancies(structure: Structure, remove: Sequence[int]) -> Structure:
    """Remove atoms (zero-based indices) and record the parent mapping.

    Remaining atoms keep their relative order and species IDs.  If the input
    already carries ``parent_indices`` the mapping is chained so the result
    still refers to the original pristine structure.
    """

    removal = sorted({int(index) for index in remove})
    if not removal:
        raise ValueError("at least one atom index must be removed")
    if removal[0] < 0 or removal[-1] >= len(structure):
        bad = removal[0] if removal[0] < 0 else removal[-1]
        raise ValueError(f"atom index to remove is out of range: {bad + 1}")
    if len(removal) >= len(structure):
        raise ValueError("refusing to remove every atom from the structure")
    removed = set(removal)
    keep = [index for index in range(len(structure)) if index not in removed]
    parents = structure.parent_indices
    species_ids = list(structure.species_ids or [None] * len(structure))
    return Structure(
        [structure.symbols[index] for index in keep],
        structure.positions[keep].copy(),
        structure.cell.copy(),
        structure.pbc,
        [species_ids[index] for index in keep],
        structure.source,
        [parents[index] if parents is not None else index for index in keep],
    )


def index_map(structure: Structure) -> list[tuple[int, int, str]]:
    """Return one-based ``(parent_index, child_index, element)`` rows."""

    parents = structure.parent_indices
    if parents is None:
        parents = list(range(len(structure)))
    return [
        (parent + 1, child + 1, structure.symbols[child])
        for child, parent in enumerate(parents)
    ]


def write_index_map_csv(path: str | Path, structure: Structure) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent_index", "child_index", "element"])
        writer.writerows(index_map(structure))
    return destination


def remap_spin_rows(
    rows: Sequence[tuple],
    structure: Structure,
    *,
    parent_length: int | None = None,
) -> tuple[list[tuple], list[int]]:
    """Renumber one-based ``DM.InitSpin`` rows from parent to child indices.

    Rows whose parent atom was removed are dropped and returned separately.
    Rows beyond ``parent_length`` (when known) raise an error.  Any trailing
    fields (for example non-collinear angles) are preserved unchanged.
    """

    parents = structure.parent_indices
    if parents is None:
        raise ValueError("structure has no parent mapping; nothing to remap")
    child_of = {parent + 1: child + 1 for child, parent in enumerate(parents)}
    remapped: list[tuple] = []
    dropped: list[int] = []
    for row in rows:
        parent_index = int(row[0])
        if parent_index <= 0 or (parent_length is not None and parent_index > parent_length):
            raise ValueError(f"DM.InitSpin atom index out of parent range: {parent_index}")
        child = child_of.get(parent_index)
        if child is None:
            dropped.append(parent_index)
            continue
        remapped.append((child, *row[1:]))
    return remapped, dropped


@dataclass(frozen=True, slots=True)
class MagneticSiteChange:
    """Pristine-versus-defect local environment of one magnetic atom."""

    parent_index: int
    child_index: int | None
    element: str
    anion_coordination_before: int | None
    anion_coordination_after: int | None
    magnetic_neighbors_before: int | None
    magnetic_neighbors_after: int | None

    @property
    def removed(self) -> bool:
        return self.child_index is None

    @property
    def changed(self) -> bool:
        return (
            self.removed
            or self.anion_coordination_before != self.anion_coordination_after
            or self.magnetic_neighbors_before != self.magnetic_neighbors_after
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["parent_index"] = self.parent_index + 1
        data["child_index"] = self.child_index + 1 if self.child_index is not None else None
        data["removed"] = self.removed
        data["changed"] = self.changed
        return data


@dataclass(slots=True)
class MagneticEnvironmentReport:
    magnetic_species: tuple[str, ...]
    anion_species: tuple[str, ...]
    neighbor_cutoff: float | None
    changes: list[MagneticSiteChange]
    notices: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[MagneticSiteChange]:
        return [change for change in self.changes if change.changed]

    def to_dict(self) -> dict[str, object]:
        return {
            "magnetic_species": list(self.magnetic_species),
            "anion_species": list(self.anion_species),
            "neighbor_cutoff": self.neighbor_cutoff,
            "sites": [change.to_dict() for change in self.changes],
            "changed_parent_indices": [c.parent_index + 1 for c in self.changed],
            "notices": list(self.notices),
            "suggestions": list(self.suggestions),
        }


def compare_magnetic_environments(
    parent: Structure,
    child: Structure,
    magnetic_species: Sequence[str],
    *,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
    cutoff: str | float | None = "auto",
    neighbor_shell: int = 1,
) -> MagneticEnvironmentReport:
    """Report how each magnetic atom's environment changed in a defect model.

    Anion coordination is counted in the child with each atom's *pristine*
    first-shell cutoff, so the number answers "how many first-shell anions
    remain".  Magnetic-neighbor counts use the pristine neighbor cutoff for
    both structures.  Removed magnetic atoms appear with ``child_index=None``.
    """

    if child.parent_indices is None:
        raise ValueError("child structure has no parent mapping; use apply_vacancies")
    if max(child.parent_indices) >= len(parent):
        raise ValueError("child parent mapping refers past the end of the parent structure")
    child_of = {parent_index: index for index, parent_index in enumerate(child.parent_indices)}
    parent_indices = select_magnetic_sites(parent, magnetic_species)
    child_indices = [child_of[index] for index in parent_indices if index in child_of]
    notices: list[str] = []

    cn_before: dict[int, int] = {}
    cn_after: dict[int, int] = {}
    anions: tuple[str, ...] = ()
    try:
        analysis = analyze_coordination_sites(
            parent, parent_indices, anion_species=anion_species, anion_cutoff=anion_cutoff
        )
    except ValueError as exc:
        notices.append(f"anion coordination unavailable: {exc}")
    else:
        anions = tuple(analysis.anion_species)
        wanted = {item.lower() for item in anions}
        child_anions = [
            index for index, symbol in enumerate(child.symbols) if symbol.lower() in wanted
        ]
        cn_before = dict(analysis.coordination_numbers)
        for parent_index in parent_indices:
            child_index = child_of.get(parent_index)
            if child_index is None:
                continue
            cn_after[parent_index] = (
                count_anion_neighbors(
                    child, child_index, child_anions, analysis.site_cutoffs[parent_index]
                )
                if child_anions
                else 0
            )

    degree_before: dict[int, int] = {}
    degree_after: dict[int, int] = {}
    resolved_cutoff: float | None = None
    if len(parent_indices) > 1:
        try:
            graph, resolved_cutoff, _ = build_neighbor_graph(
                parent, parent_indices, cutoff, neighbor_shell=neighbor_shell
            )
        except ValueError as exc:
            notices.append(f"magnetic neighbor graph unavailable: {exc}")
        else:
            degree_before = {index: graph.degree(index) for index in parent_indices}
            if len(child_indices) > 1:
                child_graph, _, _ = build_neighbor_graph(
                    child, child_indices, resolved_cutoff, neighbor_shell=neighbor_shell
                )
                degree_after = {
                    child.parent_indices[index]: child_graph.degree(index)
                    for index in child_indices
                }
            else:
                degree_after = {child.parent_indices[index]: 0 for index in child_indices}

    changes = [
        MagneticSiteChange(
            parent_index=index,
            child_index=child_of.get(index),
            element=parent.symbols[index],
            anion_coordination_before=cn_before.get(index),
            anion_coordination_after=cn_after.get(index),
            magnetic_neighbors_before=degree_before.get(index),
            magnetic_neighbors_after=degree_after.get(index),
        )
        for index in parent_indices
    ]

    suggestions: list[str] = []
    cn_changed = [
        change
        for change in changes
        if not change.removed
        and change.anion_coordination_before is not None
        and change.anion_coordination_before != change.anion_coordination_after
    ]
    if cn_changed:
        new_specs = sorted(
            {
                f"{change.element}@{change.anion_coordination_after}=..."
                for change in cn_changed
            }
        )
        suggestions.append(
            "anion coordination changed for defect atoms "
            + ", ".join(
                f"{c.child_index + 1} ({c.element}, CN {c.anion_coordination_before}->"
                f"{c.anion_coordination_after})"
                for c in cn_changed
                if c.child_index is not None
            )
            + f"; on the defect structure set their moments with --moment {' '.join(new_specs)} "
            "or a --site-moment-file (precedence: site CSV > Element@CN > Element)"
        )
        suggestions.append(
            "by-coordination/coordination-bipartite must list the new CN in "
            "--up-coordination/--down-coordination or --active-coordination, or use "
            "--coordination-tolerance; otherwise the defect-adjacent atoms are unclassified"
        )
    removed = [change for change in changes if change.removed]
    if removed:
        suggestions.append(
            "removed magnetic atoms "
            + ", ".join(f"{c.parent_index + 1} ({c.element})" for c in removed)
            + " leave a hole in the magnetic graph; re-run analyze/generate on the "
            "defect structure and check connectivity and bipartiteness"
        )
    suggestions.append(
        "these are structural changes only; the charge left by a vacancy "
        "(e.g. two electrons per O vacancy) and any resulting oxidation- or "
        "spin-state change must be chosen explicitly and compared by total energy"
    )
    return MagneticEnvironmentReport(
        magnetic_species=tuple(magnetic_species),
        anion_species=anions,
        neighbor_cutoff=resolved_cutoff,
        changes=changes,
        notices=notices,
        suggestions=suggestions,
    )


def format_environment_report(report: MagneticEnvironmentReport) -> str:
    def cell(value: int | None) -> str:
        return "-" if value is None else str(value)

    lines = [
        "Magnetic environment changes (pristine -> defect):",
        f"{'parent':>6} {'child':>6} {'el':<3} {'anion CN':>9} {'mag. nbrs':>10}  note",
    ]
    for change in report.changes:
        note = "removed" if change.removed else ("changed" if change.changed else "")
        lines.append(
            f"{change.parent_index + 1:>6} "
            f"{cell(change.child_index + 1 if change.child_index is not None else None):>6} "
            f"{change.element:<3} "
            f"{cell(change.anion_coordination_before)}->{cell(change.anion_coordination_after):<5} "
            f"{cell(change.magnetic_neighbors_before)}->{cell(change.magnetic_neighbors_after):<6}  "
            f"{note}"
        )
    if not report.changed:
        lines.append("No magnetic atom changed its first-shell environment.")
    for notice in report.notices:
        lines.append(f"NOTE: {notice}")
    for suggestion in report.suggestions:
        lines.append(f"SUGGESTION: {suggestion}")
    return "\n".join(lines)
