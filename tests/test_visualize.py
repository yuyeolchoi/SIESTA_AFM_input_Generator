from pathlib import Path

import numpy as np
import pytest

from siesta_afm.structure import Structure
from siesta_afm.visualize import (
    classify_spin_indices,
    create_spin_figure,
    plot_spin_pattern,
)


def test_zero_spin_is_classified_as_nonmagnetic() -> None:
    structure = Structure(
        ["Cu", "O", "Cu"], np.zeros((3, 3)), np.eye(3), (False, False, False)
    )
    nonmagnetic, up, down = classify_spin_indices(structure, {0: 0.0, 1: 0.5, 2: -0.5})
    assert nonmagnetic == [0]
    assert up == [1]
    assert down == [2]


def test_spin_element_filter_keeps_hidden_elements_as_nonmagnetic() -> None:
    structure = Structure(
        ["Ni", "Co", "O"], np.zeros((3, 3)), np.eye(3), (False, False, False)
    )
    spins = {0: 1.0, 1: -4.0, 2: 0.5}

    assert classify_spin_indices(structure, spins, {"Ni"}) == ([1, 2], [0], [])
    assert classify_spin_indices(structure, spins, None) == ([], [0, 2], [1])


def test_plot_with_explicit_zero_spin_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Cu", "O", "Cu"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    output = plot_spin_pattern(
        structure, {0: 0.0, 1: 0.5, 2: -0.5}, tmp_path / "spins.png"
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_value_color_mode_writes_png_for_different_moments(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Cu", "Cu", "Cu", "O"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    output = plot_spin_pattern(
        structure,
        {0: 0.7, 1: 0.5, 2: -0.5, 3: 0.0},
        tmp_path / "spin_values.png",
        color_mode="value",
    )
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    ("name", "spins"),
    [("all_zero", {0: 0.0, 1: 0.0}), ("empty", {})],
)
def test_value_color_mode_accepts_no_nonzero_spins(
    tmp_path: Path, name: str, spins: dict[int, float]
) -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Cu", "O"],
        [[0, 0, 0], [1, 0, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    output = plot_spin_pattern(
        structure,
        spins,
        tmp_path / f"{name}_spin_values.png",
        color_mode="value",
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_invalid_color_mode_is_rejected(tmp_path: Path) -> None:
    structure = Structure(["Cu"], [[0, 0, 0]], np.eye(3), (False, False, False))
    with pytest.raises(ValueError, match="color_mode"):
        plot_spin_pattern(
            structure, {0: 0.5}, tmp_path / "invalid.png", color_mode="element"
        )


def test_create_spin_figure_returns_embeddable_figure_with_value_colorbar() -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Cu", "Cu"], [[0, 0, 0], [1, 0, 0]], np.eye(3), (False, False, False)
    )
    figure = create_spin_figure(structure, {0: 0.7, 1: -0.5}, color_mode="value")
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "initial spin (μB)"
    figure.clear()


def test_value_color_scale_uses_only_visible_spin_elements() -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Ni", "Co"], [[0, 0, 0], [1, 0, 0]], np.eye(3), (False, False, False)
    )
    figure = create_spin_figure(
        structure,
        {0: 1.0, 1: -10.0},
        color_mode="value",
        visible_spin_elements={"Ni"},
    )
    assert figure.axes[1].get_ylim() == pytest.approx((-1.0, 1.0))
    figure.clear()


def test_create_spin_figure_draws_detected_bonds() -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["H", "H"], [[0, 0, 0], [0.7, 0, 0]], np.eye(3) * 5, (False,) * 3
    )
    figure = create_spin_figure(structure, {}, show_bonds=True)
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].lines[0].get_color() == "0.5"
    figure.clear()
