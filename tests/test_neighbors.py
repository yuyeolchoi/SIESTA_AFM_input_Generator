import networkx as nx
import numpy as np

from siesta_afm.neighbors import (
    PairDistance,
    automatic_cutoff,
    build_neighbor_graph,
    minimum_image_vector,
)
from siesta_afm.structure import Structure


def structure(points, cell=None, pbc=(False, False, False)) -> Structure:
    return Structure(
        ["Cu"] * len(points),
        np.asarray(points, dtype=float),
        np.asarray(cell if cell is not None else np.eye(3) * 20.0),
        pbc,
    )


def test_one_dimensional_chain_is_bipartite() -> None:
    atoms = structure([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    graph, _, _ = build_neighbor_graph(atoms, range(4), 1.01)
    assert nx.is_bipartite(graph)
    assert graph.number_of_edges() == 3


def test_square_lattice_is_bipartite() -> None:
    atoms = structure([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    graph, _, _ = build_neighbor_graph(atoms, range(4), 1.01)
    assert nx.is_bipartite(graph)
    assert graph.number_of_edges() == 4


def test_triangle_is_non_bipartite() -> None:
    root3 = np.sqrt(3.0)
    atoms = structure([[0, 0, 0], [1, 0, 0], [0.5, root3 / 2, 0]])
    graph, _, _ = build_neighbor_graph(atoms, range(3), 1.01)
    assert not nx.is_bipartite(graph)


def test_disconnected_graph_retains_isolated_nodes() -> None:
    atoms = structure([[0, 0, 0], [1, 0, 0], [10, 0, 0]])
    graph, _, _ = build_neighbor_graph(atoms, range(3), 1.01)
    assert set(graph.nodes) == {0, 1, 2}
    assert nx.number_connected_components(graph) == 2


def test_slab_does_not_wrap_z() -> None:
    atoms = structure(
        [[0, 0, 0.1], [0, 0, 9.9]], cell=np.eye(3) * 10, pbc=(True, True, False)
    )
    graph, _, pairs = build_neighbor_graph(atoms, [0, 1], 1.0)
    assert graph.number_of_edges() == 0
    assert pairs[0].distance == 9.8


def test_periodic_distance_handles_coordinates_outside_primary_cell() -> None:
    atoms = structure(
        [[0, 0, 0], [20.2, 0, 0]], cell=np.eye(3) * 10, pbc=(True, False, False)
    )
    _, _, pairs = build_neighbor_graph(atoms, [0, 1], 0.5)
    assert np.isclose(pairs[0].distance, 0.2)


def test_minimum_image_vector_in_nonorthogonal_cell() -> None:
    cell = np.asarray([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    atoms = structure([[0, 0, 0], [2.7, 1.8, 0]], cell=cell, pbc=(True, True, False))
    assert np.allclose(minimum_image_vector(atoms, 0, 1), [-0.3, -0.2, 0.0])


def test_automatic_cutoff_selects_requested_shell() -> None:
    pairs = [
        PairDistance(0, 1, 1.00, np.zeros(3)),
        PairDistance(0, 2, 1.02, np.zeros(3)),
        PairDistance(0, 3, 2.00, np.zeros(3)),
        PairDistance(0, 4, 2.02, np.zeros(3)),
    ]
    assert np.isclose(automatic_cutoff(pairs, shell=1), 1.51)
    assert automatic_cutoff(pairs, shell=2) > 2.02
