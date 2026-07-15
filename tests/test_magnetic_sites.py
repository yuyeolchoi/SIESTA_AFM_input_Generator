from pathlib import Path

from siesta_afm.io import read_structure
from siesta_afm.magnetic_sites import (
    load_moment_config,
    parse_atom_indices,
    resolve_moments,
    resolve_moments_with_sources,
)
from siesta_afm.structure import Structure
from siesta_afm.workflows import generate_assignment


ROOT = Path(__file__).parents[1]


def test_atom_index_ranges_are_inclusive() -> None:
    assert parse_atom_indices("2,5,8-10") == {2, 5, 8, 9, 10}


def test_moment_yaml_config(tmp_path: Path) -> None:
    path = tmp_path / "moments.yaml"
    path.write_text("moments:\n  Cu: 0.5\n  Ni: 1.0\n", encoding="utf-8")
    assert load_moment_config(path) == ["Cu=0.5", "Ni=1.0"]


def test_moment_resolution_sources_follow_existing_precedence(tmp_path: Path) -> None:
    structure = Structure(
        ["Co", "Co", "Co", "Ni"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    )
    site_file = tmp_path / "sites.csv"
    site_file.write_text("atom_index,moment\n1,0.4\n", encoding="utf-8")
    arguments = ["0.1", "Co=0.2", "Co@4=0.3"]
    coordinations = {0: 4, 1: 6, 2: 4, 3: 6}
    moments, sources = resolve_moments_with_sources(
        structure,
        range(4),
        arguments,
        site_moment_file=site_file,
        coordinations=coordinations,
    )
    assert moments == {0: 0.4, 1: 0.2, 2: 0.3, 3: 0.1}
    assert sources == {0: "site", 1: "element", 2: "coordination", 3: "global"}
    assert resolve_moments(
        structure,
        range(4),
        arguments,
        site_moment_file=site_file,
        coordinations=coordinations,
    ) == moments


def _inverse_spinel_assignment(moment: str, species: tuple[str, ...] = ("Ni", "Co")):
    structure = read_structure(
        ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
        slab=True,
    )
    return generate_assignment(
        structure,
        species,
        "by-coordination",
        moment,
        anion_species=["O"],
    )[1]


def test_coordination_moment_warns_when_one_element_spans_multiple_sites() -> None:
    assignment = _inverse_spinel_assignment("Ni=2.0 Co=2.0")
    warning = "\n".join(assignment.warnings)
    assert "element Co occupies both CN=4 and CN=6 sites" in warning
    assert "Co@4=... and Co@6=..." in warning

    global_warning = "\n".join(_inverse_spinel_assignment("0.5").warnings)
    assert "element Co occupies both CN=4 and CN=6 sites" in global_warning


def test_coordination_moment_explicit_sites_do_not_warn() -> None:
    assignment = _inverse_spinel_assignment("Ni@6=2.0 Co@4=2.0 Co@6=0.0")
    assert "moment was taken from a single value" not in "\n".join(
        assignment.warnings
    )


def test_coordination_moment_single_sublattice_does_not_warn() -> None:
    assignment = _inverse_spinel_assignment("Ni=2.0", species=("Ni",))
    assert "moment was taken from a single value" not in "\n".join(
        assignment.warnings
    )
