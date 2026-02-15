from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import dgl
import numpy as np
import torch
from dgl import DGLGraph

from models.CHGnet.graph.graph import Graph as BaseGraph
from models.CHGnet.graph.graph import Node as BaseNode
from models.alignn.pyg2dgl import compute_bond_cosines

# Temporarily disable cython fast-path to keep graph construction behavior
# fully on the Python BaseGraph path during debugging/refactor.
# try:
#     from models.CHGnet.graph.cygraph import make_graph as fast_make_graph
# except (ImportError, AttributeError):
#     fast_make_graph = None
fast_make_graph = None

TORCH_DTYPE = torch.float32


@dataclass
class GraphComponents:
    atom_graph_local: torch.Tensor
    directed2undirected_local: torch.Tensor
    undirected2directed_local: torch.Tensor
    image_local: torch.Tensor
    line_graph_local: torch.Tensor


def _assign_node_features(
    g_local: DGLGraph, batch, index: int, device, encoding: str
) -> None:
    atom_indices = batch.crystal_atom_idx[index]
    if encoding == "prop":
        src_tensor = batch.atom_fea
        if src_tensor.device != atom_indices.device:
            atom_indices = atom_indices.to(src_tensor.device)
        atom_features = src_tensor.index_select(0, atom_indices).to(device)
        g_local.ndata["atom_features"] = atom_features
    else:
        src_tensor = batch.atom_num
        if src_tensor.device != atom_indices.device:
            atom_indices = atom_indices.to(src_tensor.device)
        atom_numbers = src_tensor.index_select(0, atom_indices).to(device)
        g_local.ndata["atom_numbers"] = atom_numbers


def _empty_graph_pair(device) -> Tuple[DGLGraph, DGLGraph]:
    empty = dgl.graph(
        (torch.tensor([], dtype=torch.long, device=device),) * 2,
        num_nodes=0,
        device=device,
    )
    empty_lg = dgl.graph(
        (torch.tensor([], dtype=torch.long, device=device),) * 2,
        num_nodes=0,
        device=device,
    )
    empty_lg.edata["h"] = torch.zeros(0, dtype=TORCH_DTYPE, device=device)
    return empty, empty_lg


def _build_graph_local_from_structure(structure, cutoff: float) -> tuple[BaseGraph, int]:
    center_index, neighbor_index, image, distance = structure.get_neighbor_list(
        r=cutoff, sites=structure.sites, numerical_tol=1e-8
    )
    center_index = np.ascontiguousarray(center_index, dtype=np.int64)
    neighbor_index = np.ascontiguousarray(neighbor_index, dtype=np.int64)
    image = np.ascontiguousarray(image, dtype=np.int64)
    distance = np.ascontiguousarray(distance, dtype=np.float64)
    n_atoms = len(structure)

    if fast_make_graph is not None and center_index.size > 0:
        nodes, dir_edges_list, undir_edges_list, undirected_edges = fast_make_graph(
            center_index,
            len(center_index),
            neighbor_index,
            image,
            distance,
            n_atoms,
        )
        graph_local = BaseGraph(nodes=nodes)
        graph_local.directed_edges_list = dir_edges_list
        graph_local.undirected_edges_list = undir_edges_list
        graph_local.undirected_edges = undirected_edges
    else:
        graph_local = BaseGraph([BaseNode(index=i) for i in range(n_atoms)])
        for ii, jj, img, dist in zip(
            center_index, neighbor_index, image, distance, strict=True
        ):
            graph_local.add_edge(
                center_index=int(ii),
                neighbor_index=int(jj),
                image=img,
                distance=float(dist),
            )
    return graph_local, n_atoms


def _graph_local_to_tensors(
    graph_local: BaseGraph, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_graph_list, directed2undirected_list = graph_local.adjacency_list()
    atom_graph_np = np.asarray(atom_graph_list, dtype=np.int64).reshape(-1, 2)
    directed2undirected_np = np.asarray(directed2undirected_list, dtype=np.int64)
    undirected2directed_np = np.asarray(graph_local.undirected2directed(), dtype=np.int64)
    image_local_np = np.asarray(
        [edge.info["image"] for edge in graph_local.directed_edges_list],
        dtype=np.int64,
    ).reshape(-1, 3)

    atom_graph_local = torch.tensor(atom_graph_np, dtype=torch.long, device=device)
    directed2undirected_local = torch.tensor(
        directed2undirected_np, dtype=torch.long, device=device
    )
    undirected2directed_local = torch.tensor(
        undirected2directed_np, dtype=torch.long, device=device
    )
    image_local = torch.tensor(image_local_np, dtype=dtype, device=device)
    return (
        atom_graph_local,
        directed2undirected_local,
        undirected2directed_local,
        image_local,
    )


def _build_displacement_and_distances(
    structure, src: torch.Tensor, dst: torch.Tensor, image_local: torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() == 0:
        return (
            torch.zeros((0, 3), dtype=TORCH_DTYPE, device=device),
            torch.zeros(0, dtype=TORCH_DTYPE, device=device),
        )

    coords = torch.as_tensor(structure.cart_coords, dtype=TORCH_DTYPE, device=device)
    lattice = torch.as_tensor(structure.lattice.matrix, dtype=TORCH_DTYPE, device=device)
    # PBC-aware bond vector: r_ij = (x_j + image_ij @ lattice) - x_i
    displacement = coords[dst] + image_local @ lattice - coords[src]
    distances = torch.linalg.norm(displacement, dim=-1)
    return displacement, distances


def _build_line_graph_with_angles(
    g_local: DGLGraph, device
) -> DGLGraph:
    lg_local = g_local.line_graph(shared=True)
    if lg_local.num_edges() > 0:
        lg_local.apply_edges(compute_bond_cosines)
    else:
        lg_local.edata["h"] = torch.zeros(0, dtype=TORCH_DTYPE, device=device)
    return lg_local


def build_graph_from_structure(
    structure,
    batch,
    index,
    device,
    cutoff: float,
    encoding: str,
):
    graph_local, n_atoms = _build_graph_local_from_structure(structure, cutoff)
    atom_graph_local, _, _, image_local = _graph_local_to_tensors(
        graph_local, device=device, dtype=TORCH_DTYPE
    )
    src = atom_graph_local[:, 0]
    dst = atom_graph_local[:, 1]

    g_local = dgl.graph((src, dst), num_nodes=int(n_atoms), device=device)
    _assign_node_features(g_local, batch, index, device, encoding)

    displacement, distances = _build_displacement_and_distances(
        structure=structure,
        src=src,
        dst=dst,
        image_local=image_local,
        device=device,
    )

    g_local.edata["r"] = displacement
    g_local.edata["distances"] = distances

    lg_local = _build_line_graph_with_angles(g_local=g_local, device=device)
    return g_local, lg_local


def build_dgl_graphs_from_batch(batch, device, cutoff: float, encoding: str):
    structures = getattr(batch, "structures", None)
    if structures is None:
        raise TypeError(
            "ALIGNN expects BatchData to include `structures` for graph conversion."
        )

    graphs = []
    line_graphs = []
    for idx, structure in enumerate(structures):
        g_local, lg_local = build_graph_from_structure(
            structure=structure,
            batch=batch,
            index=idx,
            device=device,
            cutoff=cutoff,
            encoding=encoding,
        )
        graphs.append(g_local)
        line_graphs.append(lg_local)

    if not graphs:
        return _empty_graph_pair(device)
    return dgl.batch(graphs).to(device), dgl.batch(line_graphs).to(device)


def build_graph_components(
    structure,
    device,
    atom_graph_cutoff: float,
    bond_graph_cutoff: float,
    dtype=torch.float32,
):
    """Build CHG-style graph tensors from a single structure.

    Returns GraphComponents in local atom indexing.
    """
    graph_local, _ = _build_graph_local_from_structure(structure, atom_graph_cutoff)
    (
        atom_graph_local,
        directed2undirected_local,
        undirected2directed_local,
        image_local,
    ) = _graph_local_to_tensors(graph_local, device=device, dtype=dtype)

    line_graph_list, _ = graph_local.line_graph_adjacency_list(cutoff=bond_graph_cutoff)
    line_graph_np = np.asarray(line_graph_list, dtype=np.int64).reshape(-1, 5)
    line_graph_local = torch.tensor(line_graph_np, dtype=torch.long, device=device)

    return GraphComponents(
        atom_graph_local=atom_graph_local,
        directed2undirected_local=directed2undirected_local,
        undirected2directed_local=undirected2directed_local,
        image_local=image_local,
        line_graph_local=line_graph_local,
    )
