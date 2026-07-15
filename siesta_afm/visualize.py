"""Spin-pattern plotting and structure export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from .neighbors import detect_bonds
from .structure import Structure


def _spin_category(spins: Mapping[int, float], index: int) -> str:
    if index not in spins or np.isclose(spins[index], 0.0):
        return "zero"
    if spins[index] > 0:
        return "up"
    if spins[index] < 0:
        return "down"
    return "zero"


def element_spin_counts(
    structure: Structure, spins: Mapping[int, float]
) -> dict[str, tuple[int, int, int]]:
    """Return per-element ``(up, down, zero/unassigned)`` spin counts."""

    counts = {element: [0, 0, 0] for element in dict.fromkeys(structure.symbols)}
    category_index = {"up": 0, "down": 1, "zero": 2}
    for index, element in enumerate(structure.symbols):
        counts[element][category_index[_spin_category(spins, index)]] += 1
    return {element: tuple(values) for element, values in counts.items()}


def classify_spin_indices(
    structure: Structure,
    spins: Mapping[int, float],
    visible_spin_elements: set[str] | None = None,
    coordination_numbers: Mapping[int, int] | None = None,
    visible_coordination_numbers: set[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return nonmagnetic/zero, spin-up, and spin-down atom indices."""

    nonmagnetic: list[int] = []
    up: list[int] = []
    down: list[int] = []
    for index, element in enumerate(structure.symbols):
        category = _spin_category(spins, index)
        if visible_spin_elements is not None and element not in visible_spin_elements:
            category = "zero"
        if (
            visible_coordination_numbers is not None
            and coordination_numbers is not None
            and index in coordination_numbers
            and coordination_numbers[index] not in visible_coordination_numbers
        ):
            category = "zero"
        if category == "up":
            up.append(index)
        elif category == "down":
            down.append(index)
        else:
            nonmagnetic.append(index)
    return nonmagnetic, up, down


def plot_spin_pattern(
    structure: Structure,
    spins: Mapping[int, float],
    output: str | Path,
    *,
    show_indices: bool = False,
    color_by_layer: bool = False,
    color_mode: str = "sign",
    up_color: str = "tab:red",
    down_color: str = "tab:blue",
    nonmagnetic_color: str = "0.65",
    visible_spin_elements: set[str] | None = None,
    coordination_numbers: Mapping[int, int] | None = None,
    visible_coordination_numbers: set[int] | None = None,
    show_bonds: bool = False,
    bond_radius_scale: float = 1.0,
) -> Path:
    """Write a PNG/SVG plot or an XYZ/CIF structure with magmom metadata."""

    if color_mode not in {"sign", "value"}:
        raise ValueError("color_mode must be 'sign' or 'value'")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".xyz", ".cif"}:
        atoms = Atoms(
            structure.symbols,
            positions=structure.positions,
            cell=structure.cell,
            pbc=structure.pbc,
        )
        atoms.set_initial_magnetic_moments(
            [float(spins.get(index, 0.0)) for index in range(len(structure))]
        )
        # Extended XYZ retains the per-site initial magnetic moments.
        if suffix == ".xyz":
            ase_write(destination, atoms, format="extxyz")
        else:
            ase_write(destination, atoms)
        return destination
    if suffix not in {".png", ".svg"}:
        raise ValueError("plot output must be PNG, SVG, XYZ, or CIF")

    fig = create_spin_figure(
        structure,
        spins,
        show_indices=show_indices,
        color_by_layer=color_by_layer,
        color_mode=color_mode,
        up_color=up_color,
        down_color=down_color,
        nonmagnetic_color=nonmagnetic_color,
        visible_spin_elements=visible_spin_elements,
        coordination_numbers=coordination_numbers,
        visible_coordination_numbers=visible_coordination_numbers,
        show_bonds=show_bonds,
        bond_radius_scale=bond_radius_scale,
    )
    try:
        fig.savefig(destination, dpi=180)
    finally:
        fig.clear()
    return destination


def create_spin_figure(
    structure: Structure,
    spins: Mapping[int, float],
    *,
    show_indices: bool = False,
    color_by_layer: bool = False,
    color_mode: str = "sign",
    up_color: str = "tab:red",
    down_color: str = "tab:blue",
    nonmagnetic_color: str = "0.65",
    visible_spin_elements: set[str] | None = None,
    coordination_numbers: Mapping[int, int] | None = None,
    visible_coordination_numbers: set[int] | None = None,
    show_bonds: bool = False,
    bond_radius_scale: float = 1.0,
) -> Any:
    """Build and return a backend-neutral matplotlib Figure for a spin pattern."""

    if color_mode not in {"sign", "value"}:
        raise ValueError("color_mode must be 'sign' or 'value'")

    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "PNG/SVG plotting requires the optional plot dependency; "
            "install with 'pip install siesta-afm[plot]'"
        ) from exc

    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if show_bonds:
        for left, right in detect_bonds(
            structure, radius_scale=bond_radius_scale
        ):
            positions = structure.positions[[left, right]]
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color="0.5",
                linewidth=0.8,
                zorder=1,
            )
    nonmagnetic, up_indices, down_indices = classify_spin_indices(
        structure,
        spins,
        visible_spin_elements,
        coordination_numbers,
        visible_coordination_numbers,
    )
    if nonmagnetic:
        pos = structure.positions[nonmagnetic]
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            s=18,
            c=nonmagnetic_color,
            alpha=0.65,
            label="nonmagnetic",
            zorder=2,
        )
    if color_mode == "sign":
        for sign, selected, color, label in (
            (1, up_indices, up_color, "spin up"),
            (-1, down_indices, down_color, "spin down"),
        ):
            if not selected:
                continue
            pos = structure.positions[selected]
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                s=42,
                c=color,
                label=label,
                zorder=2,
            )
            dz = np.full(len(selected), 0.45 * sign)
            ax.quiver(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                np.zeros(len(selected)),
                np.zeros(len(selected)),
                dz,
                color=color,
                arrow_length_ratio=0.35,
                linewidth=1.2,
                zorder=3,
            )
    else:
        selected = sorted([*up_indices, *down_indices])
        values = np.asarray([float(spins[index]) for index in selected])
        vmax = float(np.max(np.abs(values))) if values.size else 0.0
        if np.isclose(vmax, 0.0):
            vmax = 1.0
        norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
        cmap = matplotlib.colormaps["coolwarm"]
        if selected:
            pos = structure.positions[selected]
            colors = cmap(norm(values))
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                s=42,
                c=colors,
                label="magnetic",
                zorder=2,
            )
            for position, value, color in zip(pos, values, colors, strict=True):
                ax.quiver(
                    *position,
                    0.0,
                    0.0,
                    0.45 * np.sign(value),
                    color=color,
                    arrow_length_ratio=0.35,
                    linewidth=1.2,
                    zorder=3,
                )
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.1, shrink=0.72)
        colorbar.set_label("initial spin (μB)")
    if show_indices:
        for index, position in enumerate(structure.positions):
            ax.text(*position, f" {index + 1}", fontsize=7)
    if color_by_layer:
        # Layer guide planes are intentionally subtle so spin sign stays legible.
        for z in sorted(
            {round(float(value), 4) for value in structure.positions[:, 2]}
        ):
            ax.text(
                float(np.min(structure.positions[:, 0])),
                float(np.min(structure.positions[:, 1])),
                z,
                f"z={z:g}",
                color="0.4",
                fontsize=6,
            )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("SIESTA initial spin pattern")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
