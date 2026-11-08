"""Render a CP2K Quickstep starting input from an atom-resolved spin assignment.

CP2K sets the initial magnetization per ``&KIND`` rather than per atom, so an
antiferromagnetic initial state needs one kind per (element, initial spin)
group, for example ``Cu_up`` / ``Cu_dn``.  The ``&COORD`` block names the kind
of every atom in the original input order, so atom indices are preserved
exactly as in the SIESTA path.

The numerical defaults (DZVP-MOLOPT-SR-GTH, CUTOFF 600 Ry, REL_CUTOFF 60,
PBE + D3(BJ), OT with CG) are starting points, not converged settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .input_template import (
    _canonical_element,
    _d_shell,
    _default_system_name,
    _hubbard_key_parts,
    _system_label,
    parse_hubbard_u,
)
from .structure import Structure


CP2K_TEMPLATE_WARNING = (
    "generated CP2K input is a starting template only; converge CUTOFF/REL_CUTOFF, "
    "the basis set, EPS_SCF, any Hubbard U, and the magnetic state before using "
    "results in a publication"
)

CP2K_U_WARNING = (
    "Hubbard U values do not transfer between codes: a SIESTA U (PAO projectors) is "
    "not the same quantity as CP2K U_MINUS_J (Mulliken/Lowdin projection); choose "
    "and validate U for CP2K separately"
)

# Standard GTH-PBE pseudopotential valence electron counts (GTH_POTENTIALS,
# the ``GTH-PBE-qN`` entries normally paired with MOLOPT basis sets).  Several
# elements also ship alternative q values; override them with ``potentials``.
GTH_PBE_VALENCE: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Ne": 8, "Na": 9, "Mg": 10, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
    "Ar": 8, "K": 9, "Ca": 10, "Sc": 11, "Ti": 12, "V": 13, "Cr": 14, "Mn": 15,
    "Fe": 16, "Co": 17, "Ni": 18, "Cu": 11, "Zn": 12, "Ga": 13, "Ge": 4,
    "As": 5, "Se": 6, "Br": 7, "Kr": 8, "Rb": 9, "Sr": 10, "Y": 11, "Zr": 12,
    "Nb": 13, "Mo": 14, "Tc": 15, "Ru": 16, "Rh": 17, "Pd": 18, "Ag": 11,
    "Cd": 12, "In": 13, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8, "Cs": 9,
    "Ba": 10, "La": 11, "Hf": 12, "Ta": 13, "W": 14, "Re": 15, "Os": 16,
    "Ir": 17, "Pt": 18, "Au": 11, "Hg": 12, "Tl": 13, "Pb": 4, "Bi": 5,
}

# Ground-state valence occupations of the open (polarizable) shells as
# (l, electrons).  CP2K's atomic guess can only polarize partly filled shells,
# so the largest |MAGNETIZATION| it accepts is the number of unpaired electrons
# these shells can host.  Example: Cu 4s1 3d10 allows |M| <= 1.
POLARIZABLE_SHELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "H": ((0, 1),), "Li": ((0, 1),), "Na": ((0, 1),), "K": ((0, 1),),
    "B": ((1, 1),), "C": ((1, 2),), "N": ((1, 3),), "O": ((1, 4),), "F": ((1, 5),),
    "Al": ((1, 1),), "Si": ((1, 2),), "P": ((1, 3),), "S": ((1, 4),), "Cl": ((1, 5),),
    "Sc": ((2, 1),), "Ti": ((2, 2),), "V": ((2, 3),), "Cr": ((0, 1), (2, 5)),
    "Mn": ((2, 5),), "Fe": ((2, 6),), "Co": ((2, 7),), "Ni": ((2, 8),),
    "Cu": ((0, 1),), "Zn": (),
    "Y": ((2, 1),), "Zr": ((2, 2),), "Nb": ((0, 1), (2, 4)), "Mo": ((0, 1), (2, 5)),
    "Tc": ((2, 5),), "Ru": ((0, 1), (2, 7)), "Rh": ((0, 1), (2, 8)), "Pd": (),
    "Ag": ((0, 1),), "Cd": (),
    "Hf": ((2, 2),), "Ta": ((2, 3),), "W": ((2, 4),), "Re": ((2, 5),),
    "Os": ((2, 6),), "Ir": ((2, 7),), "Pt": ((0, 1), (2, 9)), "Au": ((0, 1),), "Hg": (),
}


@dataclass(frozen=True, slots=True)
class CP2KKind:
    """One ``&KIND`` entry: a group of atoms sharing element and initial spin."""

    label: str
    element: str
    magnetization: float
    atom_indices: tuple[int, ...]
    coordination: int | None
    potential: str
    hubbard_u: float | None


@dataclass(frozen=True, slots=True)
class CP2KInputResult:
    text: str
    warnings: tuple[str, ...]
    kinds: tuple[CP2KKind, ...]
    atom_kinds: tuple[str, ...]
    multiplicity: int
    charge: int


def max_atomic_magnetization(element: str) -> float | None:
    """Largest |MAGNETIZATION| the GTH atomic guess can realize, or None if unknown."""

    shells = POLARIZABLE_SHELLS.get(_canonical_element(element))
    if shells is None:
        return None
    return float(sum(min(occupied, 2 * (2 * l + 1) - occupied) for l, occupied in shells))


def _kind_groups(
    structure: Structure,
    spins: Mapping[int, float],
    coordinations: Mapping[int, int] | None,
) -> list[tuple[tuple[str, float, int | None], list[int]]]:
    groups: dict[tuple[str, float, int | None], list[int]] = {}
    for index, symbol in enumerate(structure.symbols):
        value = round(float(spins.get(index, 0.0)), 6)
        if value == 0:
            value = 0.0
        coordination = (
            int(coordinations[index])
            if coordinations is not None and index in coordinations
            else None
        )
        groups.setdefault((symbol, value, coordination), []).append(index)
    return list(groups.items())


def _kind_labels(
    groups: Sequence[tuple[tuple[str, float, int | None], list[int]]],
) -> list[str]:
    by_element: dict[str, list[int]] = {}
    for number, ((element, _, _), _) in enumerate(groups):
        by_element.setdefault(element, []).append(number)
    labels = [""] * len(groups)
    for element, numbers in by_element.items():
        if len(numbers) == 1:
            labels[numbers[0]] = element
            continue
        spins_present = {groups[n][0][1] for n in numbers}
        cns_present = {groups[n][0][2] for n in numbers}
        for number in numbers:
            _, value, coordination = groups[number][0]
            parts = [element]
            if len(cns_present) > 1 and coordination is not None:
                parts.append(f"cn{coordination}")
            if len(spins_present) > 1:
                parts.append("up" if value > 0 else "dn" if value < 0 else "0")
            labels[number] = "_".join(parts)
        # Disambiguate equal labels (same sign and CN, different magnitude).
        seen: dict[str, list[int]] = {}
        for number in numbers:
            seen.setdefault(labels[number], []).append(number)
        for label, same in seen.items():
            if len(same) > 1:
                ordered = sorted(same, key=lambda n: -abs(groups[n][0][1]))
                for rank, number in enumerate(ordered, start=1):
                    labels[number] = f"{label}{rank}"
    return labels


def _format_vector(vector: Sequence[float]) -> str:
    return " ".join(f"{float(value):16.10f}" for value in vector)


def render_cp2k_input(
    structure: Structure,
    spins: Mapping[int, float],
    *,
    method: str,
    magnetic_species: Sequence[str],
    metadata: Mapping[str, object] | None = None,
    angles: Mapping[int, tuple[float, float]] | None = None,
    project_name: str | None = None,
    run_type: str = "ENERGY",
    basis_set: str = "DZVP-MOLOPT-SR-GTH",
    basis_set_file: str = "BASIS_MOLOPT",
    potential_file: str = "GTH_POTENTIALS",
    potentials: Mapping[str, str] | Sequence[str] | None = None,
    cutoff: float = 600.0,
    rel_cutoff: float = 60.0,
    eps_scf: float = 1.0e-6,
    max_scf: int = 60,
    outer_max_scf: int = 20,
    ot_minimizer: str = "CG",
    dispersion: bool = True,
    hubbard_u: Sequence[str] | Mapping[str, float] | None = None,
    plus_u_method: str = "MULLIKEN",
    charge: int = 0,
    magnetization_check: bool = True,
    site_comments: bool = True,
) -> CP2KInputResult:
    """Render a self-contained CP2K input without reordering atoms."""

    if angles and any(index in angles for index in spins):
        raise ValueError(
            "non-collinear spins cannot be written for CP2K; this writer supports "
            "collinear UKS initial states only"
        )
    run_type = run_type.upper()
    if run_type not in {"ENERGY", "ENERGY_FORCE", "GEO_OPT", "CELL_OPT", "MD"}:
        raise ValueError("run type must be ENERGY, ENERGY_FORCE, GEO_OPT, CELL_OPT, or MD")
    ot_minimizer = ot_minimizer.upper()
    if ot_minimizer not in {"CG", "DIIS", "SD", "BROYDEN"}:
        raise ValueError("OT minimizer must be CG, DIIS, SD, or BROYDEN")
    plus_u_method = plus_u_method.upper()
    if plus_u_method not in {"MULLIKEN", "MULLIKEN_CHARGES", "LOWDIN"}:
        raise ValueError("PLUS_U_METHOD must be MULLIKEN, MULLIKEN_CHARGES, or LOWDIN")
    if cutoff <= 0 or rel_cutoff <= 0:
        raise ValueError("CUTOFF and REL_CUTOFF must be positive")
    if abs(float(np.linalg.det(structure.cell))) < 1e-12:
        raise ValueError(
            "CP2K requires a simulation cell; the structure has no nonsingular cell "
            "(add a box around a molecule before writing a CP2K input)"
        )
    charge = int(charge)

    metadata = dict(metadata or {})
    raw_coordinations = metadata.get("coordination_numbers")
    coordinations: dict[int, int] | None = (
        {int(k): int(v) for k, v in raw_coordinations.items()}
        if isinstance(raw_coordinations, Mapping)
        else None
    )
    overrides = parse_hubbard_u(hubbard_u)
    if any("@" in key for key in overrides) and coordinations is None:
        raise ValueError(
            "Element@CN Hubbard U requires coordination metadata; use a "
            "coordination-aware method (by-coordination/coordination-bipartite)"
        )
    magnetic_lower = {item.strip().lower() for item in magnetic_species if item.strip()}
    for key in overrides:
        element, _ = _hubbard_key_parts(key)
        if element.lower() not in magnetic_lower:
            raise ValueError(
                f"Hubbard U override is not a selected magnetic species: {element}"
            )
        if _d_shell(element) is None:
            raise ValueError(
                f"DFT+U for {element} is not supported: only d-block elements (L=2) "
                "are written by this template"
            )

    potential_overrides: dict[str, str] = {}
    if potentials:
        items = (
            potentials.items()
            if isinstance(potentials, Mapping)
            else (
                token.split("=", 1)
                for raw in potentials
                for token in str(raw).replace(",", " ").split()
            )
        )
        for element, name in items:
            if not str(name).strip():
                raise ValueError(f"empty potential name for {element}")
            potential_overrides[_canonical_element(str(element))] = str(name).strip()

    warnings: list[str] = [CP2K_TEMPLATE_WARNING]
    groups = _kind_groups(structure, spins, coordinations)
    labels = _kind_labels(groups)
    kinds: list[CP2KKind] = []
    atom_kinds = [""] * len(structure)
    valence_total = 0
    for ((element, value, coordination), indices), label in zip(groups, labels, strict=True):
        canonical = _canonical_element(element)
        potential = potential_overrides.get(canonical)
        if potential is None:
            if canonical not in GTH_PBE_VALENCE:
                raise ValueError(
                    f"no default GTH-PBE potential is known for {element}; pass "
                    f"--potential {element}=GTH-PBE-qN"
                )
            potential = f"GTH-PBE-q{GTH_PBE_VALENCE[canonical]}"
        q_text = potential.rsplit("q", 1)[-1]
        if q_text.isdigit():
            valence_total += int(q_text) * len(indices)
        else:
            warnings.append(
                f"could not read the valence electron count from potential {potential}; "
                "MULTIPLICITY parity was not checked for that kind"
            )
        if value != 0 and magnetization_check and canonical not in potential_overrides:
            limit = max_atomic_magnetization(canonical)
            if limit is not None and abs(value) > limit + 1e-9:
                raise ValueError(
                    f"MAGNETIZATION {value:g} on kind {label} exceeds what the "
                    f"{potential} atomic guess can polarize ({element} open shells "
                    f"allow |M| <= {limit:g}); CP2K aborts in this case. Lower the "
                    "initial moment, or pass --no-magnetization-check to write it anyway"
                )
        u_value: float | None = None
        if overrides:
            specific = f"{canonical}@{coordination}" if coordination is not None else None
            if specific is not None and specific in overrides:
                u_value = overrides[specific]
            elif canonical in overrides:
                u_value = overrides[canonical]
        kind = CP2KKind(
            label=label,
            element=element,
            magnetization=value,
            atom_indices=tuple(indices),
            coordination=coordination,
            potential=potential,
            hubbard_u=u_value,
        )
        kinds.append(kind)
        for index in indices:
            atom_kinds[index] = label

    if overrides:
        warnings.append(CP2K_U_WARNING)
    for key in overrides:
        element, cn = _hubbard_key_parts(key)
        applied = [
            k for k in kinds
            if _canonical_element(k.element) == element and (cn is None or k.coordination == cn)
        ]
        if not applied:
            raise ValueError(f"Hubbard U {key} matched no kind in the structure")

    net = float(sum(float(spins.get(index, 0.0)) for index in range(len(structure))))
    unpaired = int(round(abs(net)))
    if abs(abs(net) - unpaired) > 1e-6:
        warnings.append(
            f"net initial magnetization {net:g} is not an integer; MULTIPLICITY uses "
            f"{unpaired} unpaired electrons"
        )
    electrons = valence_total - charge
    if (electrons - unpaired) % 2 != 0:
        warnings.append(
            f"{electrons} valence electrons cannot give {unpaired} unpaired electrons; "
            f"MULTIPLICITY raised to {unpaired + 2} (check the intended spin state "
            "and --charge)"
        )
        unpaired += 1
    multiplicity = unpaired + 1

    pbc_text = "".join(axis for axis, periodic in zip("XYZ", structure.pbc) if periodic)
    if not all(structure.pbc):
        warnings.append(
            f"the structure is periodic along {pbc_text or 'no axis'} only, but the "
            "cell is written with PERIODIC XYZ (vacuum padding in the cell handles "
            "slabs/molecules with the periodic GPW Poisson solver); use a 2D/0D "
            "Poisson solver if you change PERIODIC"
        )

    name = _system_label(project_name or _default_system_name(structure))
    parents = structure.parent_indices
    magnetic_display = " ".join(dict.fromkeys(magnetic_species))

    lines: list[str] = [
        "# CP2K Quickstep starting input generated by siesta-afm",
        f"# WARNING: {CP2K_TEMPLATE_WARNING}.",
        f"# Spin method: {method}; magnetic species: {magnetic_display}",
        "# Initial spins are per &KIND (MAGNETIZATION); atoms sharing an element and",
        "# initial spin form one kind. Atom order follows the input exactly.",
        "# CUTOFF 600 / REL_CUTOFF 60 and DZVP-MOLOPT-SR-GTH are unconverged starting values.",
        "&GLOBAL",
        f"  PROJECT {name}",
        f"  RUN_TYPE {run_type}",
        "  PRINT_LEVEL LOW",
        "&END GLOBAL",
        "&FORCE_EVAL",
        "  METHOD Quickstep",
        "  &DFT",
        f"    BASIS_SET_FILE_NAME {basis_set_file}",
        f"    POTENTIAL_FILE_NAME {potential_file}",
        "    UKS",
        f"    MULTIPLICITY {multiplicity}",
        f"    CHARGE {charge}",
    ]
    if any(kind.hubbard_u is not None for kind in kinds):
        lines.append(
            f"    PLUS_U_METHOD {plus_u_method}  # projection for U_MINUS_J; "
            "Mulliken is the CP2K default, Lowdin is a common alternative"
        )
    lines.extend(
        [
            "    &MGRID",
            f"      CUTOFF {cutoff:g}  # Ry; converge for the system",
            f"      REL_CUTOFF {rel_cutoff:g}",
            "      NGRIDS 5",
            "    &END MGRID",
            "    &QS",
            "      METHOD GPW",
            "      EPS_DEFAULT 1.0E-12",
            "      EXTRAPOLATION ASPC",
            "    &END QS",
            "    &SCF",
            "      SCF_GUESS ATOMIC",
            f"      EPS_SCF {eps_scf:.1E}",
            f"      MAX_SCF {max_scf}",
            "      &OT",
            "        PRECONDITIONER FULL_SINGLE_INVERSE",
            f"        MINIMIZER {ot_minimizer}  # CG is more robust than DIIS for +U AFM oxides",
            "        ENERGY_GAP 0.01",
            "      &END OT",
            "      &OUTER_SCF",
            f"        MAX_SCF {outer_max_scf}",
            f"        EPS_SCF {eps_scf:.1E}",
            "      &END OUTER_SCF",
            "    &END SCF",
            "    &XC",
            "      &XC_FUNCTIONAL PBE",
            "      &END XC_FUNCTIONAL",
        ]
    )
    if dispersion:
        lines.extend(
            [
                "      &VDW_POTENTIAL",
                "        POTENTIAL_TYPE PAIR_POTENTIAL",
                "        &PAIR_POTENTIAL",
                "          TYPE DFTD3(BJ)",
                "          PARAMETER_FILE_NAME dftd3.dat",
                "          REFERENCE_FUNCTIONAL PBE",
                "        &END PAIR_POTENTIAL",
                "      &END VDW_POTENTIAL",
            ]
        )
    lines.extend(
        [
            "    &END XC",
            "    &PRINT",
            "      &MULLIKEN ON  # atom-resolved spin moments to check the AFM state",
            "      &END MULLIKEN",
            "    &END PRINT",
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
            f"      A {_format_vector(structure.cell[0])}",
            f"      B {_format_vector(structure.cell[1])}",
            f"      C {_format_vector(structure.cell[2])}",
            f"      PERIODIC XYZ  # siesta-afm periodic axes: {pbc_text or 'none'}",
            "    &END CELL",
            "    &COORD",
        ]
    )
    width = max(len(label) for label in labels)
    for index, (label, position) in enumerate(zip(atom_kinds, structure.positions, strict=True)):
        line = f"      {label:<{width}} {_format_vector(position)}"
        if site_comments:
            comment = f"# {index + 1} {structure.symbols[index]}"
            if parents is not None and parents[index] != index:
                comment += f" (parent {parents[index] + 1})"
            if coordinations is not None and index in coordinations:
                comment += f" CN={coordinations[index]}"
            line += f"  {comment}"
        lines.append(line)
    lines.append("    &END COORD")
    for kind in kinds:
        lines.append(f"    &KIND {kind.label}")
        lines.append(f"      ELEMENT {kind.element}")
        lines.append(f"      BASIS_SET {basis_set}")
        lines.append(f"      POTENTIAL {kind.potential}")
        detail = f"{len(kind.atom_indices)} atoms"
        if kind.coordination is not None:
            detail += f", CN={kind.coordination}"
        if kind.magnetization != 0:
            lines.append(f"      MAGNETIZATION {kind.magnetization:g}  # {detail}")
        else:
            lines.append(f"      # MAGNETIZATION 0 ({detail})")
        if kind.hubbard_u is not None:
            lines.extend(
                [
                    "      &DFT_PLUS_U",
                    "        L 2  # d shell",
                    f"        U_MINUS_J [eV] {kind.hubbard_u:.2f}  # CP2K-specific value; not a SIESTA U",
                    "      &END DFT_PLUS_U",
                ]
            )
        lines.append("    &END KIND")
    lines.extend(["  &END SUBSYS", "&END FORCE_EVAL"])

    return CP2KInputResult(
        text="\n".join(lines) + "\n",
        warnings=tuple(warnings),
        kinds=tuple(kinds),
        atom_kinds=tuple(atom_kinds),
        multiplicity=multiplicity,
        charge=charge,
    )


def parse_cp2k_coord(text: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Read ``(kind, position)`` rows from a CP2K ``&COORD`` block (for tests/validation)."""

    rows: list[tuple[str, tuple[float, float, float]]] = []
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("&COORD"):
            inside = True
            continue
        if upper.startswith("&END") and inside:
            break
        if inside:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"malformed CP2K coordinate row: {raw!r}")
            rows.append((fields[0], (float(fields[1]), float(fields[2]), float(fields[3]))))
    if not rows:
        raise ValueError("no &COORD block was found")
    return rows
