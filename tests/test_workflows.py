import csv
import itertools
from pathlib import Path

import numpy as np
import pytest

from siesta_afm.io import parse_dm_init_spin
from siesta_afm.structure import Structure
from siesta_afm.workflows import enumerate_candidates


def _two_copper_sites() -> Structure:
    return Structure(["Cu", "Cu"], [[0, 0, 0], [1, 0, 0]])


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


def test_enumerate_candidates_symmetry_dedup_is_opt_in(tmp_path: Path) -> None:
    pytest.importorskip("spglib")
    structure = _symmetric_three_copper_sites()
    common = {
        "structure": structure,
        "magnetic_species": ["Cu"],
        "methods": ["random"],
        "moment": "1",
        "n_configs": 8,
        "cutoff": 1.5,
    }

    legacy = enumerate_candidates(
        output_dir=tmp_path / "legacy",
        symmetry_dedup=False,
        **common,
    )
    symmetry_aware = enumerate_candidates(
        output_dir=tmp_path / "symmetry",
        symmetry_dedup=True,
        **common,
    )

    assert len(legacy.manifest) == 4
    assert len(symmetry_aware.manifest) == 2


def test_enumerate_candidates_computes_symmetry_only_once_before_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[float] = []

    def fake_permutations(
        structure: Structure, *, symprec: float
    ) -> list[tuple[int, ...]]:
        calls.append(symprec)
        return list(itertools.permutations(range(len(structure))))

    monkeypatch.setattr(
        "siesta_afm.workflows.structure_symmetry_permutations",
        fake_permutations,
    )
    result = enumerate_candidates(
        _symmetric_three_copper_sites(),
        ["Cu"],
        ["random"],
        "1",
        8,
        tmp_path,
        symmetry_dedup=True,
        symprec=2e-4,
        cutoff=1.5,
    )

    assert len(result.manifest) == 2
    assert calls == [2e-4]


def test_enumerate_candidates_moment_sweep_writes_each_pattern_and_value(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweep"
    result = enumerate_candidates(
        Structure(["Cu", "Ni"], [[0, 0, 0], [1, 0, 0]]),
        ["Cu", "Ni"],
        ["random"],
        "Ni=0.5",
        2,
        output,
        moment_sweep="Cu=1.0,2.0",
        cutoff=1.1,
    )

    assert [path.name for path in result.written_files] == [
        "afm_001_m1.fdf",
        "afm_002_m1.fdf",
        "afm_003_m2.fdf",
        "afm_004_m2.fdf",
    ]
    assert [row["moment_Cu"] for row in result.manifest] == [1.0, 1.0, 2.0, 2.0]
    for row in result.manifest:
        spins = dict(parse_dm_init_spin(output / str(row["file"])))
        assert abs(spins[1]) == row["moment_Cu"]
        assert abs(spins[2]) == 0.5
    with result.manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["moment_Cu"] for row in rows] == ["1.0", "1.0", "2.0", "2.0"]


def test_enumerate_candidates_default_moment_sweep_is_legacy_output(
    tmp_path: Path,
) -> None:
    implicit = enumerate_candidates(
        _two_copper_sites(),
        ["Cu"],
        ["random"],
        "1",
        2,
        tmp_path / "implicit",
        cutoff=1.1,
    )
    explicit = enumerate_candidates(
        _two_copper_sites(),
        ["Cu"],
        ["random"],
        "1",
        2,
        tmp_path / "explicit",
        moment_sweep=None,
        cutoff=1.1,
    )

    assert implicit.manifest == explicit.manifest
    assert list(implicit.manifest[0]) == [
        "config_id",
        "method",
        "n_up",
        "n_down",
        "net_spin",
        "afm_score",
        "file",
    ]
    assert [path.name for path in implicit.written_files] == [
        "afm_001.fdf",
        "afm_002.fdf",
    ]
    assert [path.read_text(encoding="utf-8") for path in implicit.written_files] == [
        path.read_text(encoding="utf-8") for path in explicit.written_files
    ]


def test_enumerate_candidates_rejects_moment_sweep_element_overlap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "overlap"
    with pytest.raises(
        ValueError,
        match=r"--moment and --moment-sweep.*Cu.*only be supplied by one",
    ):
        enumerate_candidates(
            _two_copper_sites(),
            ["Cu"],
            ["random"],
            "Cu@4=1.0",
            2,
            output,
            moment_sweep="Cu@6=1.0,2.0",
            cutoff=1.1,
        )
    assert not output.exists()


def test_enumerate_candidates_rejects_moment_sweep_over_limit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "too-many"
    with pytest.raises(
        ValueError,
        match=r"combinations \(2\).*--n-configs \(101\) = 202.*limit of 200",
    ):
        enumerate_candidates(
            _two_copper_sites(),
            ["Cu"],
            ["random"],
            "0.5",
            101,
            output,
            moment_sweep="Cu=1.0,2.0",
            cutoff=1.1,
        )
    assert not output.exists()
