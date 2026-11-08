"""Enumerate initial-spin candidates for a vacancy from formal electron counting.

Removing an O²⁻ leaves two electrons behind; removing a cation leaves holes.
Where those carriers localize, and whether the receiving ion ends up high- or
low-spin, is exactly what the DFT calculation must decide.  This module
therefore does not pick an answer: it writes one ``DM.InitSpin`` candidate per
plausible bookkeeping choice (all carriers on one acceptor, one per acceptor,
or spread evenly), per requested spin state, and per sign choice for moments
that did not exist in the pristine state, in the same directory layout that
``enumerate`` produces so ``prepare-array`` and ``prepare-cp2k-array`` can run
them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

from .defects import (
    _first_shell,
    apply_vacancies,
    index_map,
    remap_spin_rows,
    write_index_map_csv,
)
from .fdf_writer import render_dm_init_spin
from .io import write_structure
from .magnetic_sites import (
    DEFAULT_ANION_OXIDATION_STATES,
    d_electron_count,
    select_magnetic_sites,
    spin_only_moment,
)
from .neighbors import build_neighbor_graph
from .ordering import analyze_coordination_sites
from .structure import Structure


SCHEMES = ("localized", "split", "delocalized")
METHOD_NAME = "defect-electron-count"

ELECTRON_COUNT_WARNING = (
    "defect spin candidates come from formal electron counting and spin-only "
    "moments; carrier localization, spin state, and sign must be decided by "
    "comparing converged total energies, not by this enumeration"
)


@dataclass(frozen=True, slots=True)
class DefectSpinCandidate:
    config_id: str
    scheme: str
    spin_state: str
    sign_variant: str
    acceptors: tuple[int, ...]
    carriers: dict[int, float]
    oxidation_states: dict[int, float]
    moments: dict[int, float]
    multiplicity: int

    def to_dict(self, child: Structure) -> dict[str, object]:
        parents = child.parent_indices or list(range(len(child)))
        return {
            "config_id": self.config_id,
            "scheme": self.scheme,
            "spin_state": self.spin_state,
            "sign_variant": self.sign_variant,
            "acceptors_parent": [index + 1 for index in self.acceptors],
            "acceptors_child": [parents.index(index) + 1 for index in self.acceptors],
            "carriers_per_acceptor": {
                str(index + 1): value for index, value in self.carriers.items()
            },
            "new_oxidation_states": {
                str(index + 1): value for index, value in self.oxidation_states.items()
            },
            "moments_child": {str(index + 1): value for index, value in self.moments.items()},
            "multiplicity": self.multiplicity,
        }


@dataclass(slots=True)
class DefectSpinEnumeration:
    parent: Structure
    child: Structure
    removed: tuple[int, ...]
    magnetic_species: tuple[str, ...]
    carriers: float
    acceptors: tuple[int, ...]
    pristine_oxidation_states: dict[int, float]
    candidates: list[DefectSpinCandidate]
    notices: list[str] = field(default_factory=list)


def _resolve_state(
    element: str,
    coordination: int | None,
    by_element: dict[str, float],
    by_coordination: dict[tuple[str, int], float],
    *,
    anion_defaults: bool,
) -> float | None:
    key = element.lower()
    if coordination is not None and (key, coordination) in by_coordination:
        return by_coordination[(key, coordination)]
    if key in by_element:
        return by_element[key]
    if any(k == key for k, _ in by_coordination):
        return None
    if anion_defaults and element.capitalize() in DEFAULT_ANION_OXIDATION_STATES:
        return float(DEFAULT_ANION_OXIDATION_STATES[element.capitalize()])
    return None


def _acceptors_for(
    parent: Structure, removed: Sequence[int], magnetic: set[int]
) -> tuple[list[int], str]:
    """Magnetic atoms that receive a vacancy's carriers (first or second shell)."""

    removed_set = set(removed)
    found: set[int] = set()
    shell_kind = "first"
    for index in removed:
        element = parent.symbols[index].lower()
        pool = [i for i, s in enumerate(parent.symbols) if s.lower() != element and i not in removed_set]
        if not pool:
            pool = [i for i in range(len(parent)) if i != index and i not in removed_set]
        shell, _ = _first_shell(parent, index, pool)
        direct = {j for j in shell if j in magnetic}
        if direct:
            found |= direct
            continue
        shell_kind = "second"
        for ligand in set(shell):
            ligand_element = parent.symbols[ligand].lower()
            ligand_pool = [
                i for i, s in enumerate(parent.symbols)
                if s.lower() != ligand_element and i not in removed_set
            ]
            second, _ = _first_shell(parent, ligand, ligand_pool)
            found |= {j for j in second if j in magnetic}
    return sorted(found), shell_kind


def enumerate_defect_spin_candidates(
    parent: Structure,
    remove: Sequence[int],
    *,
    magnetic_species: Sequence[str],
    parent_spin_rows: Sequence[tuple[int, float]],
    oxidation_by_element: dict[str, float],
    oxidation_by_coordination: dict[tuple[str, int], float] | None = None,
    charge: int = 0,
    schemes: Sequence[str] = SCHEMES,
    spin_states: Sequence[str] = ("high",),
    new_moment_signs: str = "both",
    flip_changed: bool = False,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
    cutoff: str | float | None = "auto",
) -> DefectSpinEnumeration:
    """Build vacancy spin candidates from the pristine pattern and electron counting."""

    schemes = tuple(dict.fromkeys(str(item).strip().lower() for item in schemes))
    unknown = [item for item in schemes if item not in SCHEMES]
    if unknown:
        raise ValueError(f"unknown carrier scheme {unknown[0]!r}; choose from {', '.join(SCHEMES)}")
    spin_states = tuple(dict.fromkeys(str(item).strip().lower() for item in spin_states))
    for state in spin_states:
        if state not in {"high", "low"}:
            raise ValueError("spin states must be 'high' and/or 'low'")
    if new_moment_signs not in {"both", "up", "down"}:
        raise ValueError("new moment signs must be both, up, or down")
    by_coordination = dict(oxidation_by_coordination or {})
    removed = sorted({int(index) for index in remove})
    if not removed:
        raise ValueError("at least one atom must be removed")
    child = apply_vacancies(parent, removed)
    notices: list[str] = [ELECTRON_COUNT_WARNING]

    magnetic_indices = select_magnetic_sites(parent, magnetic_species)
    magnetic = set(magnetic_indices)
    coordinations: dict[int, int] = {}
    try:
        analysis = analyze_coordination_sites(
            parent, magnetic_indices, anion_species=anion_species, anion_cutoff=anion_cutoff
        )
        coordinations = dict(analysis.coordination_numbers)
    except ValueError as exc:
        if by_coordination:
            raise ValueError(
                f"Element@CN oxidation states need anion coordination, which failed: {exc}"
            ) from exc
        notices.append(f"anion coordination unavailable: {exc}")

    pristine: dict[int, float] = {}
    for index in magnetic_indices:
        state = _resolve_state(
            parent.symbols[index], coordinations.get(index), oxidation_by_element,
            by_coordination, anion_defaults=False,
        )
        if state is None:
            cn = coordinations.get(index)
            hint = f"{parent.symbols[index]}@{cn}=..." if cn is not None else f"{parent.symbols[index]}=..."
            raise ValueError(
                f"no oxidation state for magnetic atom {index + 1} ({parent.symbols[index]}"
                + (f", CN={cn}" if cn is not None else "")
                + f"); pass --oxidation-states {hint}"
            )
        pristine[index] = state

    carriers = 0.0
    for index in removed:
        state = pristine.get(index)
        if state is None:
            state = _resolve_state(
                parent.symbols[index], coordinations.get(index), oxidation_by_element,
                by_coordination, anion_defaults=True,
            )
        if state is None:
            raise ValueError(
                f"no oxidation state for removed atom {index + 1} ({parent.symbols[index]}); "
                f"pass --oxidation-states {parent.symbols[index]}=..."
            )
        carriers += -state
    carriers -= float(charge)
    if abs(carriers - round(carriers)) > 1e-9:
        raise ValueError(f"non-integer carrier count {carriers:g}; check --oxidation-states and --charge")
    carriers = float(round(carriers))

    parent_signed = {index - 1: float(value) for index, value in parent_spin_rows}
    remapped, dropped = remap_spin_rows(parent_spin_rows, child, parent_length=len(parent))
    if dropped:
        notices.append(
            "dropped pristine DM.InitSpin rows of removed atoms: " + ", ".join(map(str, dropped))
        )
    child_of = {p: c for c, p in enumerate(child.parent_indices or [])}
    base_child_spins: dict[int, float] = {}
    for child_index in range(len(child)):
        parent_index = child.parent_indices[child_index]
        if parent_index in magnetic:
            base_child_spins[child_index] = parent_signed.get(parent_index, 0.0)
    acceptors, shell_kind = _acceptors_for(parent, removed, magnetic)
    if not acceptors:
        raise ValueError("no magnetic atom neighbors the removed atom(s); nothing can accept the carriers")
    if shell_kind == "second":
        notices.append(
            "the removed atom has no magnetic first-shell neighbor; carriers were offered "
            "to magnetic atoms sharing one of its first-shell ligands"
        )

    def acceptor_key(index: int) -> tuple[str, int | None, int, float]:
        sign = parent_signed.get(index, 0.0)
        return (
            parent.symbols[index],
            coordinations.get(index),
            (sign > 0) - (sign < 0),
            pristine[index],
        )

    distributions: list[tuple[str, dict[int, float], int]] = []
    if carriers == 0:
        notices.append("no carriers are left by the defect (check --charge); only the inherited pattern is written")
        distributions.append(("inherited", {}, 1))
    else:
        count = int(abs(carriers))
        direction = 1.0 if carriers > 0 else -1.0
        if "localized" in schemes:
            groups: dict[tuple, list[int]] = {}
            for index in acceptors:
                groups.setdefault(acceptor_key(index), []).append(index)
            for members in groups.values():
                distributions.append(("localized", {members[0]: direction * count}, len(members)))
        if "split" in schemes:
            if count < 2:
                notices.append("split scheme skipped: fewer than two carriers")
            elif len(acceptors) < count:
                notices.append(
                    f"split scheme skipped: {count} carriers but only {len(acceptors)} acceptors"
                )
            else:
                seen_keys: dict[tuple, tuple[int, ...]] = {}
                counts: dict[tuple, int] = {}
                for combo in combinations(acceptors, count):
                    key = tuple(sorted(acceptor_key(index) for index in combo))
                    seen_keys.setdefault(key, combo)
                    counts[key] = counts.get(key, 0) + 1
                for key, combo in seen_keys.items():
                    distributions.append(
                        ("split", {index: direction for index in combo}, counts[key])
                    )
        if "delocalized" in schemes:
            share = direction * count / len(acceptors)
            distributions.append(("delocalized", {index: share for index in acceptors}, 1))

    candidates: list[DefectSpinCandidate] = []
    seen_moments: set[tuple[tuple[int, float], ...]] = set()
    for scheme, carriers_map, multiplicity in distributions:
        new_states: dict[int, float] = {}
        invalid = False
        for index, amount in carriers_map.items():
            state = pristine[index] - amount  # electrons lower the oxidation state
            try:
                d_electron_count(parent.symbols[index], state)
            except ValueError as exc:
                notices.append(f"{scheme}: skipped acceptor {index + 1}: {exc}")
                invalid = True
                break
            new_states[index] = state
        if invalid:
            continue
        for spin_state in spin_states:
            changed_child: dict[int, float] = {}
            for index, state in new_states.items():
                # Low spin is an octahedral (strong-field) outcome; keep other
                # coordinations high-spin even when "low" is requested.
                effective = (
                    "high"
                    if spin_state == "low" and coordinations.get(index, 6) != 6
                    else spin_state
                )
                if effective != spin_state:
                    note = (
                        f"low-spin requested but acceptor {index + 1} "
                        f"({parent.symbols[index]}, CN={coordinations.get(index)}) is not "
                        "octahedral; it stays high-spin"
                    )
                    if note not in notices:
                        notices.append(note)
                magnitude = spin_only_moment(parent.symbols[index], state, effective)
                changed_child[child_of[index]] = magnitude
            variants: list[tuple[str, dict[int, float]]] = []
            base = dict(base_child_spins)
            new_moment_atoms = [
                c for c, mag in changed_child.items()
                if base.get(c, 0.0) == 0.0 and mag != 0.0
            ]
            sign_choices = (
                [("new-up", 1.0), ("new-down", -1.0)] if new_moment_signs == "both"
                else [("new-up", 1.0)] if new_moment_signs == "up" else [("new-down", -1.0)]
            ) if new_moment_atoms else [("inherit", 1.0)]
            for label, new_sign in sign_choices:
                moments = dict(base)
                for c, magnitude in changed_child.items():
                    old = base.get(c, 0.0)
                    sign = (1.0 if old > 0 else -1.0) if old != 0.0 else new_sign
                    moments[c] = sign * magnitude
                variants.append((label, moments))
                if flip_changed and changed_child:
                    flipped = dict(moments)
                    for c in changed_child:
                        flipped[c] = -flipped[c]
                    variants.append((label + "+flip", flipped))
            for label, moments in variants:
                key = tuple(sorted((c, round(v, 9)) for c, v in moments.items()))
                if key in seen_moments:
                    continue
                seen_moments.add(key)
                candidates.append(
                    DefectSpinCandidate(
                        config_id=f"{len(candidates) + 1:03d}",
                        scheme=scheme,
                        spin_state=spin_state if new_states else "inherited",
                        sign_variant=label,
                        acceptors=tuple(sorted(carriers_map)),
                        carriers=dict(carriers_map),
                        oxidation_states=new_states,
                        moments=moments,
                        multiplicity=multiplicity,
                    )
                )
    if not candidates:
        raise ValueError("no defect spin candidate could be built:\n" + "\n".join(notices))
    vanished = [
        c for cand in candidates for c, v in cand.moments.items()
        if v == 0.0 and base_child_spins.get(c, 0.0) != 0.0
    ]
    if vanished:
        notices.append(
            "some candidates remove a pristine moment entirely (e.g. Cu2+ -> Cu+ d10); "
            "re-check the magnetic graph connectivity on the defect structure"
        )
    return DefectSpinEnumeration(
        parent=parent,
        child=child,
        removed=tuple(removed),
        magnetic_species=tuple(magnetic_species),
        carriers=carriers,
        acceptors=tuple(acceptors),
        pristine_oxidation_states=pristine,
        candidates=candidates,
        notices=notices,
    )


def _afm_score(child: Structure, moments: dict[int, float], cutoff: str | float | None) -> float:
    active = sorted(index for index, value in moments.items() if value != 0.0)
    if len(active) < 2:
        return 0.0
    try:
        graph, _, _ = build_neighbor_graph(child, active, cutoff)
    except ValueError:
        return 0.0
    if not graph.number_of_edges():
        return 0.0
    return sum(moments[a] * moments[b] < 0 for a, b in graph.edges) / graph.number_of_edges()


def write_defect_spin_candidates(
    enumeration: DefectSpinEnumeration,
    output_dir: str | Path,
    *,
    structure_name: str = "defect.fdf",
    site_comments: bool = True,
    cutoff: str | float | None = "auto",
) -> dict[str, Path]:
    """Write the defect structure, index map, candidate spin blocks, and manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    child = enumeration.child
    removed_text = ", ".join(
        f"{index + 1} ({enumeration.parent.symbols[index]})" for index in enumeration.removed
    )
    structure_path = write_structure(
        child, destination / structure_name,
        source_note=f"vacancy model; removed parent atoms {removed_text}",
    )
    map_path = write_index_map_csv(destination / "index_map.csv", child)
    manifest_rows: list[dict[str, object]] = []
    for candidate in enumeration.candidates:
        file_name = f"afm_{candidate.config_id}.fdf"
        metadata = {
            "scheme": candidate.scheme,
            "spin_state": candidate.spin_state,
            "sign_variant": candidate.sign_variant,
            "acceptors": [child.parent_indices.index(i) + 1 for i in candidate.acceptors],
            "new_oxidation_states": [
                f"{enumeration.parent.symbols[i]}{s:+g}" for i, s in candidate.oxidation_states.items()
            ],
            "removed_parent_atoms": [index + 1 for index in enumeration.removed],
            "carriers": enumeration.carriers,
        }
        text = render_dm_init_spin(
            candidate.moments,
            method=METHOD_NAME,
            magnetic_species=enumeration.magnetic_species,
            metadata=metadata,
            structure=child,
            site_comments=site_comments,
        )
        (destination / file_name).write_text(text, encoding="utf-8")
        values = list(candidate.moments.values())
        manifest_rows.append(
            {
                "config_id": candidate.config_id,
                "method": METHOD_NAME,
                "n_up": sum(v > 0 for v in values),
                "n_down": sum(v < 0 for v in values),
                "net_spin": sum(values),
                "afm_score": _afm_score(child, candidate.moments, cutoff),
                "file": file_name,
                "scheme": candidate.scheme,
                "spin_state": candidate.spin_state,
                "sign_variant": candidate.sign_variant,
                "acceptors": " ".join(str(a) for a in metadata["acceptors"]),
                "new_oxidation_states": " ".join(metadata["new_oxidation_states"]),
                "multiplicity": candidate.multiplicity,
            }
        )
    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "structure": structure_path.name,
        "index_map": map_path.name,
        "removed_parent_atoms": [index + 1 for index in enumeration.removed],
        "carriers": enumeration.carriers,
        "acceptors_parent": [index + 1 for index in enumeration.acceptors],
        "pristine_oxidation_states": {
            str(index + 1): state for index, state in enumeration.pristine_oxidation_states.items()
        },
        "candidates": [candidate.to_dict(child) for candidate in enumeration.candidates],
        "notices": list(enumeration.notices),
    }
    report_path = destination / "candidates.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "structure": structure_path,
        "index_map": map_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def format_defect_spin_summary(enumeration: DefectSpinEnumeration) -> str:
    parent = enumeration.parent
    lines = [
        f"Removed parent atoms: "
        + ", ".join(f"{i + 1} ({parent.symbols[i]})" for i in enumeration.removed),
        f"Carriers left in the lattice: {enumeration.carriers:+g} electrons",
        "Acceptors (parent index, element, pristine state): "
        + ", ".join(
            f"{i + 1} {parent.symbols[i]}{enumeration.pristine_oxidation_states[i]:+g}"
            for i in enumeration.acceptors
        ),
        f"{'id':>4} {'scheme':<12} {'spin':<9} {'signs':<14} {'mult':>4}  acceptors -> new states / moments",
    ]
    parents = enumeration.child.parent_indices or []
    for candidate in enumeration.candidates:
        detail = "; ".join(
            f"{i + 1}:{parent.symbols[i]}{s:+g}->{candidate.moments[parents.index(i)]:+g}"
            for i, s in candidate.oxidation_states.items()
        ) or "pristine pattern"
        lines.append(
            f"{candidate.config_id:>4} {candidate.scheme:<12} {candidate.spin_state:<9} "
            f"{candidate.sign_variant:<14} {candidate.multiplicity:>4}  {detail}"
        )
    return "\n".join(lines)
