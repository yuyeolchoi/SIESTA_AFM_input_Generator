"""Literature-backed magnetic-ordering models and conservative matching.

The catalog intentionally recommends *candidate initial states*, not magnetic
ground states. Composition and coordination are insufficient to infer exchange
constants, oxidation/spin states, magnetic propagation vectors, or canting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import gcd
from typing import Sequence

from .magnetic_sites import select_magnetic_sites
from .ordering import analyze_coordination_sites
from .structure import Structure


TRANSITION_METALS = frozenset({"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"})


@dataclass(frozen=True, slots=True)
class LiteratureReference:
    """Primary source attached to a curated ordering model."""

    citation: str
    doi: str
    url: str


@dataclass(frozen=True, slots=True)
class AFMModel:
    """One experimentally or theoretically established ordering archetype."""

    model_id: str
    name: str
    family: str
    examples: tuple[str, ...]
    ordering: str
    implementation_status: str
    cli_strategy: str
    caveats: tuple[str, ...]
    references: tuple[LiteratureReference, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelRecommendation:
    """A structural match to a catalog model, never a ground-state claim."""

    model_id: str
    match_basis: str
    observations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AFMRecommendationReport:
    """Composition/coordination screening result for one input structure."""

    formula: str
    reduced_composition: dict[str, int]
    magnetic_species: tuple[str, ...]
    coordination_counts: dict[str, dict[int, int]]
    recommendations: tuple[ModelRecommendation, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


AFM_MODELS: tuple[AFMModel, ...] = (
    AFMModel(
        model_id="rocksalt-afm-ii",
        name="Rocksalt monoxide AFM-II",
        family="rocksalt transition-metal monoxide",
        examples=("MnO", "FeO", "CoO", "NiO"),
        ordering=(
            "Ferromagnetic (111) cation planes alternate in sign along [111]."
        ),
        implementation_status="direct candidate after magnetic-cell validation",
        cli_strategy=(
            "Use a cell commensurate with AFM-II, then --method layer "
            "--layer-direction 1 1 1; verify equal up/down populations and PBC."
        ),
        caveats=(
            "A crystallographic primitive or conventional cell may be too small; build "
            "and validate a magnetic supercell first.",
            "The [111] direction assumes the input Cartesian axes follow the cubic "
            "rocksalt setting.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "W. L. Roth, Magnetic Structures of MnO, FeO, CoO, and NiO, "
                    "Phys. Rev. 110, 1333 (1958)"
                ),
                doi="10.1103/PhysRev.110.1333",
                url="https://doi.org/10.1103/PhysRev.110.1333",
            ),
        ),
    ),
    AFMModel(
        model_id="cuo-af1",
        name="CuO low-temperature AF1",
        family="monoclinic tenorite",
        examples=("CuO",),
        ordering=(
            "Commensurate collinear AF1 with q=(0.5, 0, -0.5); moments lie along b."
        ),
        implementation_status="validated reference indices required",
        cli_strategy=(
            "Use --method manual-spins with atom-indexed signed moments copied from a "
            "validated AF1 magnetic cell. Do not infer the basis phases from q alone."
        ),
        caveats=(
            "The q vector does not by itself determine signs for every atom in an "
            "arbitrary origin, basis, or supercell.",
            "Near the AF1-AF2 transition CuO has an incommensurate non-collinear phase "
            "that a collinear DM.InitSpin model does not reproduce.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "Z. Wang et al., Magnetoelectric effect and phase transitions in "
                    "CuO in external magnetic fields, Nat. Commun. 7, 10295 (2016)"
                ),
                doi="10.1038/ncomms10295",
                url="https://doi.org/10.1038/ncomms10295",
            ),
            LiteratureReference(
                citation=(
                    "A. Maity et al., Magnons in the Multiferroic Phase of Cupric "
                    "Oxide, Phys. Status Solidi B 257, 1900704 (2020)"
                ),
                doi="10.1002/pssb.201900704",
                url="https://doi.org/10.1002/pssb.201900704",
            ),
        ),
    ),
    AFMModel(
        model_id="perovskite-a",
        name="Perovskite A-type AFM",
        family="ABO3 perovskite",
        examples=("LaMnO3",),
        ordering=(
            "Ferromagnetic planes stack antiferromagnetically along one crystal axis."
        ),
        implementation_status="direct candidate after axis/cell validation",
        cli_strategy=(
            "Use --method propagation-vector --afm-type A, or explicit layer ordering "
            "when the antiferromagnetic stacking axis is known."
        ),
        caveats=(
            "The A preset assumes the desired stacking axis is fractional c; transform "
            "the cell or provide an explicit direction otherwise.",
            "Octahedral tilts and orbital order can change which crystallographic axis "
            "carries the AFM stacking.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "F. Moussa et al., Spin waves in the antiferromagnet perovskite "
                    "LaMnO3, Phys. Rev. B 54, 15149 (1996)"
                ),
                doi="10.1103/PhysRevB.54.15149",
                url="https://doi.org/10.1103/PhysRevB.54.15149",
            ),
        ),
    ),
    AFMModel(
        model_id="perovskite-c",
        name="Perovskite C-type AFM",
        family="ABO3 perovskite",
        examples=("LaVO3",),
        ordering=(
            "Antiferromagnetic order in two directions and ferromagnetic chains in the "
            "third direction."
        ),
        implementation_status="direct candidate after axis/cell validation",
        cli_strategy="Use --method propagation-vector --afm-type C.",
        caveats=(
            "The C preset uses q=(0.5,0.5,0) in the input fractional basis; map the "
            "experimental axes into that basis before generation.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "L. D. Tung et al., Spin, orbital ordering, and magnetic dynamics "
                    "of LaVO3, Phys. Rev. B 78, 054416 (2008)"
                ),
                doi="10.1103/PhysRevB.78.054416",
                url="https://doi.org/10.1103/PhysRevB.78.054416",
            ),
        ),
    ),
    AFMModel(
        model_id="perovskite-g",
        name="Perovskite G-type AFM",
        family="ABO3 perovskite",
        examples=("LaFeO3",),
        ordering="Every nearest magnetic neighbor is antiparallel in three dimensions.",
        implementation_status="direct candidate after axis/cell validation",
        cli_strategy=(
            "Use --method propagation-vector --afm-type G or "
            "--method neighbor-bipartite after verifying the magnetic graph."
        ),
        caveats=(
            "Weak canting from octahedral rotations or Dzyaloshinskii-Moriya coupling "
            "is outside a purely collinear G-type initial state.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "T. Peterlin-Neumaier and E. Steichele, Antiferromagnetic structure "
                    "of LaFeO3 from high resolution TOF neutron diffraction, J. Magn. "
                    "Magn. Mater. 59, 351 (1986)"
                ),
                doi="10.1016/0304-8853(86)90431-2",
                url="https://doi.org/10.1016/0304-8853(86)90431-2",
            ),
        ),
    ),
    AFMModel(
        model_id="co3o4-a-site-afm",
        name="Normal-spinel Co3O4 A-site AFM",
        family="normal spinel",
        examples=("Co3O4",),
        ordering=(
            "Tetrahedral A-site Co2+ ions form two antiparallel sublattices; "
            "octahedral B-site Co3+ ions have zero ordered moment."
        ),
        implementation_status="direct structure-validated candidate",
        cli_strategy=(
            "Use --method coordination-bipartite --anion-species O "
            "--active-coordination 4 --inactive-coordination 6 "
            "--moment Co@4=3.02 Co@6=0; inactive sites are written explicitly."
        ),
        caveats=(
            "Confirm that CN=4 and CN=6 identify Td and Oh sites in the actual bulk or "
            "slab; surface truncation can change coordination.",
            "This model is specific to normal Co3O4 and must not be transferred to an "
            "inverse or mixed-valence spinel solely from the AB2O4 formula.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "W. L. Roth, The magnetic structure of Co3O4, J. Phys. Chem. "
                    "Solids 25, 1 (1964)"
                ),
                doi="10.1016/0022-3697(64)90156-8",
                url="https://doi.org/10.1016/0022-3697(64)90156-8",
            ),
        ),
    ),
    AFMModel(
        model_id="spinel-neel-ferrimagnetic",
        name="Spinel Neel A-vs-B ferrimagnetic candidate",
        family="normal or inverse spinel",
        examples=("Fe3O4", "NiCo2O4", "CoFe2O4"),
        ordering=(
            "Tetrahedral A and octahedral B sublattices are antiparallel, generally "
            "with unequal moments and a nonzero net magnetization."
        ),
        implementation_status="direct candidate, chemistry must be supplied",
        cli_strategy=(
            "Use --method by-coordination with explicit Element@CN moments and verified "
            "cation distribution/oxidation states."
        ),
        caveats=(
            "This is ferrimagnetic rather than compensated AFM.",
            "Composition alone cannot determine normal/inverse cation occupancy, mixed "
            "valence, or site-resolved moments.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "Y. Yafet and C. Kittel, Antiferromagnetic Arrangements in Ferrites, "
                    "Phys. Rev. 87, 290 (1952)"
                ),
                doi="10.1103/PhysRev.87.290",
                url="https://doi.org/10.1103/PhysRev.87.290",
            ),
        ),
    ),
    AFMModel(
        model_id="spinel-yafet-kittel",
        name="Yafet-Kittel canted spinel candidate",
        family="frustrated spinel ferrite",
        examples=("selected ferrites with strong B-B exchange",),
        ordering=(
            "The B sublattice splits into canted components rather than one collinear "
            "A-vs-B pair."
        ),
        implementation_status="exploratory non-collinear only",
        cli_strategy=(
            "Use graph-coloring/non-collinear output only to enumerate starting states; "
            "construct literature angles manually for a quantitative Yafet-Kittel model."
        ),
        caveats=(
            "Generic graph coloring does not optimize exchange energy or reproduce a "
            "published canting angle.",
            "SIESTA spin-orbit and non-collinear settings require separate validation."
        ),
        references=(
            LiteratureReference(
                citation=(
                    "Y. Yafet and C. Kittel, Antiferromagnetic Arrangements in Ferrites, "
                    "Phys. Rev. 87, 290 (1952)"
                ),
                doi="10.1103/PhysRev.87.290",
                url="https://doi.org/10.1103/PhysRev.87.290",
            ),
        ),
    ),
    AFMModel(
        model_id="corundum-cr2o3",
        name="Corundum Cr2O3 collinear AFM",
        family="corundum M2O3",
        examples=("Cr2O3",),
        ordering=(
            "Two oppositely magnetized Cr sublattices form the collinear bulk AFM state."
        ),
        implementation_status="reference groups required",
        cli_strategy=(
            "Map the crystallographic Cr sublattices for the chosen hexagonal/rhombohedral "
            "setting, then use --method manual-groups."
        ),
        caveats=(
            "Do not infer the Cr groups from atom index parity; cell setting and origin "
            "change the basis order.",
            "Magnetoelectric domains and surface magnetism need additional treatment.",
        ),
        references=(
            LiteratureReference(
                citation=(
                    "D. E. Cox, W. J. Takei, and G. Shirane, Magnetic and neutron "
                    "diffraction study of the Cr2O3-Fe2O3 system, J. Phys. Chem. Solids "
                    "24, 405 (1963)"
                ),
                doi="10.1016/0022-3697(63)90199-9",
                url="https://doi.org/10.1016/0022-3697(63)90199-9",
            ),
        ),
    ),
)

_MODEL_BY_ID = {model.model_id: model for model in AFM_MODELS}


def get_afm_model(model_id: str) -> AFMModel:
    """Return one catalog model or raise a user-facing error."""

    try:
        return _MODEL_BY_ID[model_id]
    except KeyError as exc:
        choices = ", ".join(model.model_id for model in AFM_MODELS)
        raise ValueError(f"unknown AFM model {model_id!r}; choose one of: {choices}") from exc


def _reduced_composition(symbols: Sequence[str]) -> dict[str, int]:
    counts = Counter(symbols)
    divisor = 0
    for value in counts.values():
        divisor = gcd(divisor, value)
    return {symbol: counts[symbol] // divisor for symbol in counts}


def _formula(composition: dict[str, int]) -> str:
    return "".join(
        symbol + (str(count) if count != 1 else "")
        for symbol, count in composition.items()
    )


def _same_formula(composition: dict[str, int], expected: dict[str, int]) -> bool:
    return {key.lower(): value for key, value in composition.items()} == {
        key.lower(): value for key, value in expected.items()
    }


def _infer_magnetic_species(structure: Structure) -> tuple[str, ...]:
    seen: set[str] = set()
    inferred: list[str] = []
    for symbol in structure.symbols:
        if symbol in TRANSITION_METALS and symbol not in seen:
            seen.add(symbol)
            inferred.append(symbol)
    return tuple(inferred)


def _coordination_counts(
    structure: Structure, magnetic_species: Sequence[str]
) -> tuple[dict[str, dict[int, int]], str | None]:
    if "O" not in structure.symbols or not magnetic_species:
        return {}, None
    indices = select_magnetic_sites(structure, magnetic_species)
    try:
        analysis = analyze_coordination_sites(
            structure, indices, anion_species=("O",)
        )
    except ValueError as exc:
        return {}, str(exc)
    result: dict[str, dict[int, int]] = {}
    for symbol in magnetic_species:
        counts = Counter(
            analysis.coordination_numbers[index]
            for index in indices
            if structure.symbols[index].lower() == symbol.lower()
        )
        if counts:
            result[symbol] = dict(sorted(counts.items()))
    return result, None


def recommend_afm_models(
    structure: Structure,
    magnetic_species: Sequence[str] | None = None,
) -> AFMRecommendationReport:
    """Return conservative model candidates from formula and O coordination.

    Matching deliberately stops before spin assignment. The caller must inspect
    the cited model, cell setting, propagation vector, and atom-resolved sites.
    """

    composition = _reduced_composition(structure.symbols)
    chosen_species = (
        tuple(magnetic_species)
        if magnetic_species is not None
        else _infer_magnetic_species(structure)
    )
    if magnetic_species is not None:
        # Validate explicit user input even when coordination analysis is impossible.
        select_magnetic_sites(structure, chosen_species)
    coordination, coordination_error = _coordination_counts(
        structure, chosen_species
    )
    recommendations: list[ModelRecommendation] = []

    def add(model_id: str, basis: str, *observations: str) -> None:
        recommendations.append(
            ModelRecommendation(model_id, basis, tuple(observations))
        )

    if _same_formula(composition, {"Cu": 1, "O": 1}):
        add(
            "cuo-af1",
            "exact reduced composition",
            "CuO composition matches tenorite chemistry but not necessarily its polymorph.",
            "Use a published/validated AF1 atom-index map; q alone is insufficient.",
        )
    elif any(
        _same_formula(composition, {metal: 1, "O": 1})
        for metal in ("Mn", "Fe", "Co", "Ni")
    ):
        add(
            "rocksalt-afm-ii",
            "exact monoxide composition",
            "Confirm rocksalt symmetry and octahedral metal coordination.",
            "Confirm that the magnetic cell is commensurate with alternating (111) planes.",
        )

    if _same_formula(composition, {"Co": 3, "O": 4}):
        add(
            "co3o4-a-site-afm",
            "exact reduced composition",
            "Normal bulk Co3O4 should show eight CN=4 and sixteen CN=6 Co sites per conventional cell.",
            "Reject this model if the structure is inverse, disordered, or surface-truncated.",
        )
    elif composition.get("O") == 4 and sum(
        count for symbol, count in composition.items() if symbol != "O"
    ) == 3:
        add(
            "spinel-neel-ferrimagnetic",
            "AB2O4 composition candidate",
            "Verify Td/Oh occupancy and oxidation states before assigning moments.",
        )
        add(
            "spinel-yafet-kittel",
            "AB2O4 frustrated-family candidate",
            "Treat as an alternative only when literature or exchange arguments support canting.",
        )

    nonoxygen = {
        symbol: count for symbol, count in composition.items() if symbol != "O"
    }
    if (
        composition.get("O") == 3
        and len(nonoxygen) == 2
        and all(value == 1 for value in nonoxygen.values())
    ):
        for model_id, example in (
            ("perovskite-a", "A-type"),
            ("perovskite-c", "C-type"),
            ("perovskite-g", "G-type"),
        ):
            add(
                model_id,
                "ABO3 composition candidate",
                f"{example} is a screening candidate, not a formula-derived ground state.",
                "Confirm perovskite connectivity, magnetic B site, axes, and magnetic cell.",
            )

    if _same_formula(composition, {"Cr": 2, "O": 3}):
        add(
            "corundum-cr2o3",
            "exact reduced composition",
            "Confirm corundum symmetry and map the Cr sublattices in the chosen setting.",
        )

    limitations = [
        "Recommendations are candidate initial states, not predictions of the magnetic ground state.",
        "Formula/CN matching cannot determine oxidation state, spin state, exchange constants, canting, or magnetic domains.",
        "Compare multiple symmetry-allowed initial states by converged total energy and final local moments.",
    ]
    if coordination_error:
        limitations.append(f"O-coordination diagnostic unavailable: {coordination_error}")
    if not recommendations:
        limitations.append(
            "No curated family matched; use literature-defined manual groups/spins or enumerate carefully validated candidates."
        )
    return AFMRecommendationReport(
        formula=_formula(composition),
        reduced_composition=composition,
        magnetic_species=chosen_species,
        coordination_counts=coordination,
        recommendations=tuple(recommendations),
        limitations=tuple(limitations),
    )
