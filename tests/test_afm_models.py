import json
from pathlib import Path

from siesta_afm.afm_models import (
    AFM_MODELS,
    get_afm_model,
    recommend_afm_models,
)
from siesta_afm.cli import main
from siesta_afm.io import parse_dm_init_spin, read_structure
from siesta_afm.ordering import (
    analyze_coordination_sites,
    coordination_bipartite_ordering,
)
from siesta_afm.structure import Structure


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_unique_ids_primary_references_and_actionable_strategies() -> None:
    identifiers = [model.model_id for model in AFM_MODELS]
    assert len(identifiers) == len(set(identifiers))
    assert {
        "rocksalt-afm-ii",
        "cuo-af1",
        "perovskite-a",
        "perovskite-c",
        "perovskite-g",
        "co3o4-a-site-afm",
        "spinel-neel-ferrimagnetic",
        "spinel-yafet-kittel",
        "corundum-cr2o3",
    } == set(identifiers)
    for model in AFM_MODELS:
        assert "--method" in model.cli_strategy or model.model_id == "spinel-yafet-kittel"
        assert model.references
        assert all(reference.doi and reference.url.startswith("https://") for reference in model.references)


def test_co3o4_recommendation_uses_measured_coordination_signature() -> None:
    structure = read_structure(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif")

    report = recommend_afm_models(structure, ["Co"])

    assert report.formula == "Co3O4"
    assert report.coordination_counts == {"Co": {4: 8, 6: 16}}
    assert [item.model_id for item in report.recommendations] == [
        "co3o4-a-site-afm"
    ]
    assert "coordination-bipartite" in get_afm_model(
        "co3o4-a-site-afm"
    ).cli_strategy


def test_coordination_bipartite_orders_co3o4_a_sites_and_zeros_b_sites() -> None:
    structure = read_structure(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif")
    indices = [
        index for index, symbol in enumerate(structure.symbols) if symbol == "Co"
    ]
    coordination = analyze_coordination_sites(
        structure, indices, anion_species=["O"]
    ).coordination_numbers

    result = coordination_bipartite_ordering(
        structure, indices, anion_species=["O"]
    )

    active_signs = [
        result.signs[index] for index in indices if coordination[index] == 4
    ]
    assert active_signs.count(1) == active_signs.count(-1) == 4
    assert all(
        result.signs[index] == 0 for index in indices if coordination[index] == 6
    )
    assert result.metadata["component_sizes"] == [8]
    assert "explicit zero rows" in "\n".join(result.warnings)


def test_perovskite_formula_returns_a_c_and_g_screening_candidates() -> None:
    structure = Structure(
        ["La", "Fe", "O", "O", "O"],
        [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    report = recommend_afm_models(structure, ["Fe"])

    assert [item.model_id for item in report.recommendations] == [
        "perovskite-a",
        "perovskite-c",
        "perovskite-g",
    ]
    assert all(
        "not a formula-derived ground state" in item.observations[0]
        for item in report.recommendations
    )


def test_afm_models_cli_writes_filtered_json_report(tmp_path: Path) -> None:
    output = tmp_path / "models.json"

    code = main(
        [
            "afm-models",
            str(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"),
            "--magnetic-species",
            "Co",
            "--model",
            "co3o4-a-site-afm",
            "--json",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["formula"] == "Co3O4"
    assert payload["coordination_counts"] == {"Co": {"4": 8, "6": 16}}
    assert [item["model_id"] for item in payload["recommendations"]] == [
        "co3o4-a-site-afm"
    ]
    assert payload["model_details"][0]["references"][0]["doi"] == (
        "10.1016/0022-3697(64)90156-8"
    )


def test_co3o4_cli_generation_preserves_explicit_zero_sites(tmp_path: Path) -> None:
    output = tmp_path / "co3o4_spin.fdf"

    code = main(
        [
            "generate",
            str(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"),
            "--magnetic-species",
            "Co",
            "--method",
            "coordination-bipartite",
            "--anion-species",
            "O",
            "--moment",
            "Co@4=3.02",
            "Co@6=0",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    rows = dict(parse_dm_init_spin(output))
    assert len(rows) == 24
    assert sum(value > 0 for value in rows.values()) == 4
    assert sum(value < 0 for value in rows.values()) == 4
    assert all(rows[index] == 0 for index in range(1, 17))
    assert not set(range(17, 49)).intersection(rows)


def test_co3o4_candidate_enumeration_scores_only_active_sublattice(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidates"

    code = main(
        [
            "enumerate",
            str(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"),
            "--magnetic-species",
            "Co",
            "--methods",
            "coordination-bipartite",
            "--anion-species",
            "O",
            "--moment",
            "Co@4=3.02",
            "Co@6=0",
            "--n-configs",
            "1",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    manifest = (output / "manifest.csv").read_text(encoding="utf-8")
    assert "coordination-bipartite,4,4,0.0,1.0" in manifest
    rows = dict(parse_dm_init_spin(output / "afm_001.fdf"))
    assert len(rows) == 24
    assert all(rows[index] == 0 for index in range(1, 17))
    assert not set(range(17, 49)).intersection(rows)
