"""Structure and SIESTA FDF readers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.units import Bohr

from .structure import Structure, periodic_axes_to_pbc


_COMMENT_RE = re.compile(r"\s*[#!].*$")
_INCLUDE_RE = re.compile(r"^\s*%include\s+(?P<name>.+?)\s*$", re.IGNORECASE)


def _fdf_label_pattern(name: str) -> str:
    """Return a regex honoring FDF's case/separator-insensitive labels."""

    normalized = re.sub(r"[._-]", "", name)
    return r"[._-]*".join(re.escape(character) for character in normalized)


def _strip_comment(line: str) -> str:
    return _COMMENT_RE.sub("", line).strip()


def read_fdf_with_includes(path: str | Path, _stack: tuple[Path, ...] = ()) -> str:
    """Read an FDF file and recursively expand ``%include`` directives."""

    file_path = Path(path).expanduser().resolve()
    if file_path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, file_path))
        raise ValueError(f"recursive FDF include detected: {chain}")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    expanded: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8-sig").splitlines():
        match = _INCLUDE_RE.match(_strip_comment(raw_line))
        if not match:
            expanded.append(raw_line)
            continue
        name = match.group("name").strip().strip("\"'")
        include_path = (file_path.parent / name).resolve()
        expanded.append(read_fdf_with_includes(include_path, (*_stack, file_path)))
    return "\n".join(expanded) + "\n"


def _block(text: str, name: str) -> list[str] | None:
    label = _fdf_label_pattern(name)
    pattern = re.compile(
        rf"(?ims)^\s*%block\s+{label}\s*$"
        rf"(?P<body>.*?)"
        rf"^\s*%endblock(?:\s+{label})?\s*$"
    )
    match = pattern.search(text)
    return match.group("body").splitlines() if match else None


def _keyword(text: str, name: str, default: str | None = None) -> str | None:
    pattern = re.compile(rf"(?im)^\s*{_fdf_label_pattern(name)}\s+(?P<value>[^#!\n]+)")
    match = pattern.search(text)
    return match.group("value").strip() if match else default


def parse_fdf_structure(path: str | Path) -> Structure:
    text = read_fdf_with_includes(path)
    species_block = _block(text, "ChemicalSpeciesLabel")
    coords_block = _block(text, "AtomicCoordinatesAndAtomicSpecies")
    origin_block = _block(text, "AtomicCoordinatesOrigin")
    lattice_block = _block(text, "LatticeVectors")
    if species_block is None or coords_block is None:
        raise ValueError(
            "FDF structure requires ChemicalSpeciesLabel and "
            "AtomicCoordinatesAndAtomicSpecies blocks"
        )

    from ase.data import chemical_symbols

    species: dict[int, str] = {}
    for raw in species_block:
        fields = _strip_comment(raw).split()
        if len(fields) >= 3:
            atomic_number = abs(int(fields[1]))
            if 0 < atomic_number < len(chemical_symbols):
                element = chemical_symbols[atomic_number]
            else:
                match = re.match(r"[A-Z][a-z]?", fields[2])
                if not match:
                    raise ValueError(
                        f"cannot determine element from species line: {raw}"
                    )
                element = match.group(0)
            species[int(fields[0])] = element
    if not species:
        raise ValueError("ChemicalSpeciesLabel block is empty")

    lattice_constant = _parse_lattice_constant(
        _keyword(text, "LatticeConstant", "1.0 Ang") or "1.0 Ang"
    )
    cell = np.zeros((3, 3), dtype=float)
    if lattice_block is not None:
        rows = [
            [float(value) for value in _strip_comment(raw).split()[:3]]
            for raw in lattice_block
            if len(_strip_comment(raw).split()) >= 3
        ]
        if len(rows) != 3:
            raise ValueError("LatticeVectors must contain exactly three vectors")
        cell = np.asarray(rows, dtype=float) * lattice_constant

    coordinate_format = (
        (_keyword(text, "AtomicCoordinatesFormat", "Ang") or "Ang")
        .lower()
        .replace("_", "")
    )
    positions: list[list[float]] = []
    symbols: list[str] = []
    species_ids: list[int] = []
    for raw in coords_block:
        fields = _strip_comment(raw).split()
        if len(fields) < 4:
            continue
        xyz = [float(value) for value in fields[:3]]
        species_id = int(fields[3])
        if species_id not in species:
            raise ValueError(f"unknown FDF species id {species_id}")
        positions.append(xyz)
        symbols.append(species[species_id])
        species_ids.append(species_id)

    coordinates = np.asarray(positions, dtype=float)
    if coordinate_format in {"fractional", "scaledbylatticevectors"}:
        if not np.any(cell):
            raise ValueError("fractional FDF coordinates require LatticeVectors")
        coordinates = coordinates @ cell
    elif coordinate_format in {"scaledcartesian", "scaledcartesianangstrom"}:
        coordinates *= lattice_constant
    elif "bohr" in coordinate_format:
        coordinates *= Bohr
    elif coordinate_format not in {"ang", "angstrom", "notscaledcartesianang"}:
        raise ValueError(f"unsupported AtomicCoordinatesFormat: {coordinate_format}")

    if origin_block is not None:
        origin_rows = [
            [float(value) for value in _strip_comment(raw).split()[:3]]
            for raw in origin_block
            if len(_strip_comment(raw).split()) >= 3
        ]
        if len(origin_rows) != 1:
            raise ValueError(
                "AtomicCoordinatesOrigin must contain exactly one vector"
            )
        coordinates += np.asarray(origin_rows[0], dtype=float) * lattice_constant

    pbc = (True, True, True) if np.any(cell) else (False, False, False)
    return Structure(symbols, coordinates, cell, pbc, species_ids, str(Path(path)))


def _parse_lattice_constant(value: str) -> float:
    fields = value.split()
    magnitude = float(fields[0])
    unit = fields[1].lower() if len(fields) > 1 else "ang"
    if unit.startswith("bohr"):
        magnitude *= Bohr
    return magnitude


def parse_xv_structure(path: str | Path) -> Structure:
    """Parse the structural part of a SIESTA XV file (coordinates are in Bohr)."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 4:
        raise ValueError("truncated XV file")
    cell = (
        np.asarray(
            [[float(value) for value in lines[i].split()[:3]] for i in range(3)],
            dtype=float,
        )
        * Bohr
    )
    count = int(lines[3].split()[0])
    if len(lines) < count + 4:
        raise ValueError("truncated XV atom list")
    from ase.data import chemical_symbols

    symbols: list[str] = []
    positions: list[list[float]] = []
    species_ids: list[int] = []
    for raw in lines[4 : 4 + count]:
        fields = raw.split()
        if len(fields) < 5:
            raise ValueError(f"invalid XV atom line: {raw}")
        species_id, atomic_number = int(fields[0]), int(fields[1])
        species_ids.append(species_id)
        symbols.append(chemical_symbols[atomic_number])
        positions.append([float(value) * Bohr for value in fields[2:5]])
    return Structure(
        symbols, positions, cell, (True, True, True), species_ids, str(path)
    )


def _parse_xv_species_ids(path: str | Path) -> list[int]:
    """Read the SIESTA species ids that ASE's XV reader does not retain."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 4:
        raise ValueError("truncated XV file")
    count = int(lines[3].split()[0])
    if len(lines) < count + 4:
        raise ValueError("truncated XV atom list")

    species_ids: list[int] = []
    for raw in lines[4 : 4 + count]:
        fields = raw.split()
        if len(fields) < 5:
            raise ValueError(f"invalid XV atom line: {raw}")
        species_ids.append(int(fields[0]))
    return species_ids


def from_ase(atoms: Atoms, source: str | None = None) -> Structure:
    return Structure(
        list(atoms.get_chemical_symbols()),
        atoms.get_positions(),
        np.asarray(atoms.cell),
        tuple(bool(value) for value in atoms.pbc),
        [None] * len(atoms),
        source,
    )


def read_structure(
    path: str | Path,
    *,
    slab: bool = False,
    periodic_axes: str | None = None,
) -> Structure:
    """Read a supported structure without reordering its atoms."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".fdf":
        structure = parse_fdf_structure(file_path)
    elif suffix == ".xv":
        try:
            structure = from_ase(ase_read(file_path), str(file_path))
            species_ids = _parse_xv_species_ids(file_path)
            if len(species_ids) != len(structure):
                raise ValueError("XV atom count differs between ASE and file data")
            structure.species_ids = species_ids
        except Exception:
            structure = parse_xv_structure(file_path)
    else:
        # ASE recognizes CIF, XYZ, POSCAR, and CONTCAR.  Explicitly pass the
        # VASP format for extensionless POSCAR/CONTCAR files.
        fmt = "vasp" if file_path.name.upper() in {"POSCAR", "CONTCAR"} else None
        structure = from_ase(ase_read(file_path, format=fmt), str(file_path))

    if periodic_axes is not None:
        structure = structure.with_pbc(periodic_axes_to_pbc(periodic_axes))
    elif slab:
        structure = structure.with_pbc((True, True, False))
    return structure


def parse_dm_init_spin(
    text_or_path: str | Path,
    *,
    warnings: list[str] | None = None,
    include_angles: bool = False,
) -> list[tuple[int, float]] | list[tuple[int, float, float, float]]:
    """Return DM.InitSpin rows while preserving duplicates for validation.

    SIESTA's maximum-polarization shorthands ``+`` and ``-`` are represented
    as ``+1.0`` and ``-1.0``.  By default, non-collinear theta/phi fields are
    accepted but ignored and reported through ``warnings``.  With
    ``include_angles=True``, every row is normalized to
    ``(atom_index, moment, theta, phi)``; two-field rows use an absolute
    magnitude plus a synthesized 0/180-degree theta.
    """

    candidate = (
        Path(text_or_path) if not isinstance(text_or_path, Path) else text_or_path
    )
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    text = candidate.read_text(encoding="utf-8-sig") if is_file else str(text_or_path)
    block = _block(text, "DM.InitSpin")
    if block is None:
        raise ValueError("DM.InitSpin block not found")
    rows: list[tuple[int, float]] = []
    angle_rows: list[tuple[int, float, float, float]] = []
    for raw in block:
        fields = _strip_comment(raw).split()
        if len(fields) >= 2:
            atom_index = int(fields[0])
            if fields[1] == "+":
                moment = 1.0
            elif fields[1] == "-":
                moment = -1.0
            else:
                moment = float(fields[1].replace("d", "e").replace("D", "E"))
            if include_angles:
                if len(fields) >= 3:
                    theta = float(fields[2].replace("d", "e").replace("D", "E"))
                    phi = (
                        float(fields[3].replace("d", "e").replace("D", "E"))
                        if len(fields) >= 4
                        else 0.0
                    )
                    angle_rows.append((atom_index, moment, theta, phi))
                else:
                    angle_rows.append(
                        (
                            atom_index,
                            abs(moment),
                            0.0 if moment >= 0 else 180.0,
                            0.0,
                        )
                    )
            else:
                rows.append((atom_index, moment))
            if not include_angles and len(fields) >= 3 and warnings is not None:
                warnings.append(
                    f"DM.InitSpin atom {atom_index} has non-collinear theta/phi "
                    "fields; angles were ignored by the collinear parser"
                )
    return angle_rows if include_angles else rows


def read_plain_fdf(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig")
