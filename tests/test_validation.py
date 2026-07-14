import numpy as np

from siesta_afm.structure import Structure
from siesta_afm.validation import validate_spins


def sample_structure() -> Structure:
    return Structure(
        ["Cu", "O", "Cu"],
        [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]],
        np.eye(3) * 10,
        (False, False, False),
    )


def test_duplicate_atom_index_is_reported() -> None:
    report = validate_spins([(1, 0.5), (1, -0.5)])
    assert not report.valid
    assert any("duplicate" in error for error in report.errors)


def test_out_of_range_index_is_reported() -> None:
    report = validate_spins([(4, 0.5)], structure=sample_structure())
    assert not report.valid
    assert any("out of range" in error for error in report.errors)


def test_nonmagnetic_species_is_reported() -> None:
    report = validate_spins(
        [(1, 0.5), (2, -0.5)],
        structure=sample_structure(),
        magnetic_species=["Cu"],
    )
    assert not report.valid
    assert any("nonmagnetic species" in error for error in report.errors)


def test_afm_score_is_computed() -> None:
    report = validate_spins(
        [(1, 0.5), (3, -0.5)], structure=sample_structure(), cutoff=1.1
    )
    assert report.valid
    assert report.antiparallel_fraction == 1.0


def test_validation_reports_disconnected_component_reliability_warning() -> None:
    structure = Structure(
        ["Cu"] * 4,
        [[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]],
        np.eye(3) * 10,
        (False, False, False),
    )
    report = validate_spins(
        [(1, 0.5), (2, -0.5), (3, 0.5), (4, -0.5)],
        structure=structure,
        cutoff=1.01,
    )
    assert report.component_sizes == [2, 2]
    assert "no physical meaning" in "\n".join(report.warnings)
