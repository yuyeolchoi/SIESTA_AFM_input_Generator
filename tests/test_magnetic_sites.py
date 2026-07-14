from pathlib import Path

from siesta_afm.magnetic_sites import load_moment_config, parse_atom_indices


def test_atom_index_ranges_are_inclusive() -> None:
    assert parse_atom_indices("2,5,8-10") == {2, 5, 8, 9, 10}


def test_moment_yaml_config(tmp_path: Path) -> None:
    path = tmp_path / "moments.yaml"
    path.write_text("moments:\n  Cu: 0.5\n  Ni: 1.0\n", encoding="utf-8")
    assert load_moment_config(path) == ["Cu=0.5", "Ni=1.0"]
