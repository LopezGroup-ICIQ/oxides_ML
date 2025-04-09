"""
This module contains functions and classes for creating, manipulating and analyzying graphs
from ASE Atoms objects to PyG Data format.
"""

from itertools import product

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import numpy as np
import torch 
from scipy.spatial import Voronoi
from ase import Atoms
from networkx import Graph, set_node_attributes, connected_components, get_node_attributes
 
from oxides_ml.constants import CORDERO

def get_voronoi_neighbourlist(atoms: Atoms, 
                              tol: float, 
                              scaling_factor: float, 
                              adsorbate_indices: list[int], 
                              mic=True) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The surface atoms are not part of the adsorbate (i.e., adsorbate atoms are known by index).
    
    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-material system.
        tol (float): Tolerance added to the covalent radius sum.
        scaling_factor (float): Extra scaling applied to surface atoms bonded to adsorbate atoms.
        adsorbate_indices (list[int]): List of atom indices belonging to the adsorbate.
        mic (bool): Whether to apply minimum image convention.

    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.
    """

    if len(atoms) == 0:
        return np.array([])

    num_adsorbate_atoms = len(adsorbate_indices)

    # First necessary condition for two atoms to be linked: Sharing a Voronoi facet
    coords_arr = np.repeat(np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)

    increment = 0
    if len(adsorbate_indices) == 0 or len(adsorbate_indices) == len(atoms):
        return np.sort(np.array(pairs_corr), axis=1)
    
    while True:
        pairs = []
        for pair in pairs_corr:
            i, j = pair
            atom1, atom2 = atoms[i].symbol, atoms[j].symbol
            threshold = CORDERO[atom1] + CORDERO[atom2] + tol

            is_adsorbate_1 = i in adsorbate_indices
            is_adsorbate_2 = j in adsorbate_indices

            if is_adsorbate_1 and not is_adsorbate_2:
                threshold += max(scaling_factor + increment - 1.0, 0) * CORDERO[atom2]
            if not is_adsorbate_1 and is_adsorbate_2:
                threshold += max(scaling_factor + increment - 1.0, 0) * CORDERO[atom1]

            distance = atoms.get_distance(i, j, mic=mic)
            if distance <= threshold:
                pairs.append(pair)

        if num_adsorbate_atoms == 0:
            pairs = pairs_corr
            break
        else:
            c1 = any((i in adsorbate_indices and j not in adsorbate_indices) for i, j in pairs)
            c2 = any((i not in adsorbate_indices and j in adsorbate_indices) for i, j in pairs)
            if (c1 or c2) or all(i in adsorbate_indices for i in range(len(atoms))):
                break
            else:
                increment += 0.2

    return np.sort(np.array(pairs), axis=1)

def atoms_to_nx(atoms: Atoms, 
                voronoi_tolerance: float, 
                scaling_factor: float,
                second_order: bool, 
                adsorbate_indices: list[int], 
                mode: str) -> Graph:
    """
    Convert ASE Atoms object to NetworkX graph, representing the adsorbate-surface system.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-material system.
        voronoi_tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the surface atoms.
        adsorbate_indices (list[int]): list of atom indices belonging to the adsorbate.
        mode (str): whether the graph is created for the TS or the reactant/product.

    Returns:
        Graph: NetworkX graph representing the adsorbate-material system.
    """
    adsorbate_idxs = set(adsorbate_indices)
    neighbour_list = get_voronoi_neighbourlist(atoms, voronoi_tolerance, scaling_factor, adsorbate_indices)

    # Get surface atoms that are neighbours of the adsorbate
    surface_neighbours_idxs = {
        j if i in adsorbate_idxs else i
        for i, j in neighbour_list
        if (i in adsorbate_idxs and j not in adsorbate_idxs) or 
           (j in adsorbate_idxs and i not in adsorbate_idxs)
    }

    if second_order:
        # Add surface atoms connected to surface neighbours (but not in adsorbate)
        second_order_neighbors = {
            j if i in surface_neighbours_idxs else i
            for i, j in neighbour_list
            if (i in surface_neighbours_idxs and j not in adsorbate_idxs) or 
               (j in surface_neighbours_idxs and i not in adsorbate_idxs)
        }
        surface_neighbours_idxs |= second_order_neighbors

    # Construct the graph from adsorbate + relevant surface atoms
    graph = Graph()

    # graph_nodes = list(adsorbate_idxs | surface_neighbours_idxs)

    if len(adsorbate_idxs) == 0:
        graph_nodes = list(range(len(atoms)))  # Use all atoms for bare slab
    else:
        graph_nodes = list(adsorbate_idxs | surface_neighbours_idxs)
    
    graph.add_nodes_from(graph_nodes)
    set_node_attributes(graph, {i: atoms[i].symbol for i in graph_nodes}, "elem")

    ensemble_neighbour_list = [pair for pair in neighbour_list if pair[0] in graph.nodes and pair[1] in graph.nodes]
    graph.add_edges_from(ensemble_neighbour_list, ts_edge=0)
    
    return graph, list(surface_neighbours_idxs), None


def atoms_to_pyg(atoms: Atoms,
                calc_type: str,
                voronoi_tol: float,
                scaling_factor: float,
                second_order: bool,
                one_hot_encoder: OneHotEncoder, 
                adsorbate_indices: list[int]) -> Data:
    """
    Convert ASE Atoms object to PyG Data object, representing the adsorbate-surface system.   

    Args:
        atoms (Atoms): ASE Atoms object.
        calc_type (str): type of calculation performed on the system.
                         "int": intermediate, "ts": transition state.                         
        voronoi_tol (float): Tolerance applied during the graph conversion.
        scaling_factor (float): Scaling factor applied to atomic radius of materials.
        one_hot_encoder (OneHotEncoder): One-hot encoder.
        adsorbate_elems (list[str]): list of elements present in the adsorbate.
    Returns:
        graph (torch_geometric.data.Data): graph representation of the transition state.

    Notes:
        The graph is constructed as follows:
        - Nodes: one-hot encoded elements of the adsorbate and the surface atoms in contact with it.
        - Edges: connectivity list of the adsorbate-surface system.
        - Edge features: 1 if the edge corresponds to the broken bond in the TS, 0 otherwise.
    """
    if calc_type not in ["int", "ts"]:
        raise ValueError("calc_type must be either 'int' or 'ts'.")
    
    nx, surface_neighbors, bb_idxs = atoms_to_nx(atoms, voronoi_tol, scaling_factor, second_order, adsorbate_indices, calc_type)

    elem_list = list(get_node_attributes(nx, "elem").values())
    if len(elem_list) == 0:
        raise ValueError(f"No atoms found in graph — check adsorbate_indices and system type.")
    
    elem_array = np.array(elem_list).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    x = torch.from_numpy(elem_enc).float()

    nodes_list = list(nx.nodes)
    edge_tails_heads = [(nodes_list.index(edge[0]), nodes_list.index(edge[1])) for edge in nx.edges]
    edge_tails = [x for x, _ in edge_tails_heads] + [y for _, y in edge_tails_heads]
    edge_heads = [y for _, y in edge_tails_heads] + [x for x, _ in edge_tails_heads]    
    edge_index = torch.tensor([edge_tails, edge_heads], dtype=torch.long)

    # edge attributes
    edge_attr = torch.zeros(edge_index.shape[1], 1)
    if calc_type == 'ts':
        for i in range(edge_index.shape[1]):
            edge_tuple = (nodes_list[edge_index[0, i].item()], nodes_list[edge_index[1, i].item()])
            if nx.edges[edge_tuple]['ts_edge'] == 1:
                edge_attr[i, 0] = 1  # As the nxgraph is undirected, the edge attribute is repeated twice
    return Data(x, edge_index, edge_attr, elem=elem_list), surface_neighbors, bb_idxs
