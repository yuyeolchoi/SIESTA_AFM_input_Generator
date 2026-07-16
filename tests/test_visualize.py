from pathlib import Path

import numpy as np
import pytest

from siesta_afm.structure import Structure
from siesta_afm.visualize import (
    _spin_vectors,
    classify_spin_indices,
    create_spin_figure,
    element_spin_counts,
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


def test_coordination_filter_is_optional_and_combines_with_element_filter() -> None:
    structure = Structure(
        ["Co", "Co", "Ni", "Ni", "O"],
        np.zeros((5, 3)),
        np.eye(3),
        (False, False, False),
    )
    spins = {0: 1.0, 1: -2.0, 2: 3.0, 3: -4.0, 4: 5.0}
    coordinations = {0: 4, 1: 6, 2: 6, 3: 4}

    assert classify_spin_indices(
        structure,
        spins,
        coordination_numbers=coordinations,
        visible_coordination_numbers={6},
    ) == ([0, 3], [2, 4], [1])
    assert classify_spin_indices(
        structure,
        spins,
        coordination_numbers=None,
        visible_coordination_numbers={6},
    ) == ([], [0, 2, 4], [1, 3])
    assert classify_spin_indices(
        structure,
        spins,
        visible_spin_elements={"Co"},
        coordination_numbers=coordinations,
        visible_coordination_numbers={6},
    ) == ([0, 2, 3, 4], [], [1])


def test_element_spin_counts_includes_zero_and_unassigned_elements() -> None:
    structure = Structure(
        ["Ni", "Ni", "Co", "Co", "O", "O", "O"],
        np.zeros((7, 3)),
        np.eye(3),
        (False, False, False),
    )
    spins = {0: 2.0, 1: 1.0, 2: -3.0, 3: 0.0}

    counts = element_spin_counts(structure, spins)

    assert counts == {
        "Ni": (2, 0, 0),
        "Co": (0, 1, 1),
        "O": (0, 0, 3),
    }
    nonmagnetic, up, down = classify_spin_indices(structure, spins)
    for element, expected in counts.items():
        indices = {
            index for index, symbol in enumerate(structure.symbols) if symbol == element
        }
        assert expected == (
            len(indices.intersection(up)),
            len(indices.intersection(down)),
            len(indices.intersection(nonmagnetic)),
        )


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


def test_value_color_scale_uses_only_visible_coordination_numbers() -> None:
    pytest.importorskip("matplotlib")
    structure = Structure(
        ["Co", "Co"], [[0, 0, 0], [1, 0, 0]], np.eye(3), (False, False, False)
    )
    figure = create_spin_figure(
        structure,
        {0: 1.0, 1: -10.0},
        color_mode="value",
        coordination_numbers={0: 4, 1: 6},
        visible_coordination_numbers={4},
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


def test_spin_vectors_keep_collinear_directions_exact() -> None:
    dx, dy, dz = _spin_vectors({0: 1.0, 1: -1.0}, [0, 1], None)

    assert dx.tolist() == [0.0, 0.0]
    assert dy.tolist() == [0.0, 0.0]
    assert dz.tolist() == pytest.approx([0.45, -0.45])


def test_spin_vectors_follow_noncollinear_theta_phi_angles() -> None:
    dx, dy, dz = _spin_vectors(
        {0: 1.0, 1: 1.0, 2: 1.0},
        [0, 1, 2],
        {0: (90.0, 0.0), 1: (0.0, 37.0), 2: (180.0, 91.0)},
    )

    assert dx == pytest.approx([0.45, 0.0, 0.0])
    assert dy == pytest.approx([0.0, 0.0, 0.0])
    assert dz == pytest.approx([0.0, 0.45, -0.45])
