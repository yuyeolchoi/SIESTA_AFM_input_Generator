import itertools
import sys

import numpy as np
import pytest

from siesta_afm.structure import Structure
from siesta_afm.symmetry import structure_symmetry_permutations


def _symmetric_three_copper_sites() -> Structure:
    cell = np.diag([2.0, 2.0, 2.0])
    fractional = np.asarray(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    return Structure(
        ["Cu", "Cu", "Cu"],
        fractional @ cell,
        cell=cell,
        pbc=(True, True, True),
    )


def test_structure_symmetry_permutations_returns_known_triplet_group() -> None:
    pytest.importorskip("spglib")

    found = structure_symmetry_permutations(_symmetric_three_copper_sites())

    assert found[0] == (0, 1, 2)
    assert set(found) == set(itertools.permutations(range(3)))


@pytest.mark.parametrize(
    "structure",
    [
        Structure(
            ["Cu"],
            [[0.0, 0.0, 0.0]],
            cell=np.eye(3),
            pbc=(True, True, False),
        ),
        Structure(
            ["Cu"],
            [[0.0, 0.0, 0.0]],
            cell=np.zeros((3, 3)),
            pbc=(True, True, True),
        ),
    ],
)
def test_structure_symmetry_permutations_requires_periodic_nonsingular_cell(
    structure: Structure,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires a fully periodic, nonsingular cell",
    ):
        structure_symmetry_permutations(structure)


def test_structure_symmetry_permutations_reports_missing_spglib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "spglib", None)

    with pytest.raises(
        RuntimeError,
        match=r"--symmetry-dedup requires the optional spglib dependency",
    ):
        structure_symmetry_permutations(_symmetric_three_copper_sites())
