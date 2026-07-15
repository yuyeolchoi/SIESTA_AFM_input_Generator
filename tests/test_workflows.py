from pathlib import Path

import pytest

from siesta_afm.structure import Structure
from siesta_afm.workflows import enumerate_candidates


def _two_copper_sites() -> Structure:
    return Structure(["Cu", "Cu"], [[0, 0, 0], [1, 0, 0]])


def test_enumerate_candidates_removes_global_inversion_duplicates(
    tmp_path: Path,
) -> None:
    result = enumerate_candidates(
        _two_copper_sites(),
        ["Cu"],
        ["random"],
        "1",
        4,
        tmp_path / "collapsed",
        cutoff=1.1,
    )
    assert len(result.manifest) == 2
    assert len(result.written_files) == 2
    assert result.manifest_path.is_file()
    assert result.notices[-1] == (
        "requested 4, but only 2 distinct patterns were found."
    )

    kept = enumerate_candidates(
        _two_copper_sites(),
        ["Cu"],
        ["random"],
        "1",
        4,
        tmp_path / "kept",
        keep_global_spin_inversion=True,
        cutoff=1.1,
    )
    assert len(kept.manifest) == 4


def test_enumerate_candidates_accumulates_method_failures(tmp_path: Path) -> None:
    result = enumerate_candidates(
        _two_copper_sites(),
        ["Cu"],
        ["propagation-vector", "alternating-index"],
        "1",
        1,
        tmp_path,
        cutoff=1.1,
    )
    assert len(result.manifest) == 1
    assert result.failures == [
        "propagation-vector: --q-vector or --afm-type is required for "
        "propagation-vector"
    ]


def test_enumerate_candidates_raises_when_no_candidate_is_possible(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"(?s)no AFM configurations could be generated.*propagation-vector",
    ):
        enumerate_candidates(
            _two_copper_sites(),
            ["Cu"],
            ["propagation-vector"],
            "1",
            1,
            tmp_path,
            cutoff=1.1,
        )
    assert not list(tmp_path.glob("afm_*.fdf"))
    assert not (tmp_path / "manifest.csv").exists()
