
import random

from copy import deepcopy

import torch
from torch_geometric.data import Data


def train_test_split(seed_nodes: list, prop_val: float, prop_test: float, max_nodes: int = None, seed: int = 0):
    copy_seed_nodes = deepcopy(seed_nodes)
    copy_seed_nodes = sorted(copy_seed_nodes)
    random.Random(seed).shuffle(copy_seed_nodes)
    if max_nodes:
        copy_seed_nodes = copy_seed_nodes[: max_nodes]
    end_train = int(len(copy_seed_nodes) * (1 - prop_val - prop_test))
    end_val = int(len(copy_seed_nodes) * (1 - prop_test))
    return copy_seed_nodes[: end_train], copy_seed_nodes[end_train: end_val], copy_seed_nodes[end_val:]


def reorder_batch(batch, graph_indexes: torch.tensor):

    nodes_ordered_by_sampling_hop = []
    for i, graph_index in enumerate(graph_indexes.numpy()):
        if len(nodes_ordered_by_sampling_hop) - 1 < graph_index:
            nodes_ordered_by_sampling_hop.append(i)

    num_hops = int(batch.num_nodes_sampled.shape[0] / batch.num_graphs)

    edges_ordered_by_sampling_hop = []

    for n_hop in range(num_hops):

        for n_seed in range(batch.num_graphs):

            start_graph = nodes_ordered_by_sampling_hop[n_seed]
            num_nodes_already_sampled = int(batch.num_nodes_sampled[num_hops * n_seed: num_hops * n_seed + n_hop].sum())
            num_new_nodes_sampled = int(batch.num_nodes_sampled[num_hops * n_seed + n_hop])
            nodes_ordered_by_sampling_hop.extend(
                [start_graph + 1 + num_nodes_already_sampled + i for i in range(num_new_nodes_sampled)])

            edges_ordered_by_sampling_hop.extend([2 * (start_graph - n_seed) + 2 * num_nodes_already_sampled + i
                                                  for i in range(2 * num_new_nodes_sampled)])

    permutation = {j: i for i, j in enumerate(nodes_ordered_by_sampling_hop)}

    new_edge_index = torch.tensor([permutation[e.item()] for e in batch.edge_index.flatten()])
    new_edge_index = new_edge_index.view(batch.edge_index.shape)
    new_edge_index[0, :] = new_edge_index[0, edges_ordered_by_sampling_hop]
    new_edge_index[1, :] = new_edge_index[1, edges_ordered_by_sampling_hop]

    reordered_batch = Data(x=batch.x[nodes_ordered_by_sampling_hop], edge_index=new_edge_index)

    if hasattr(batch, "seed"):
        reordered_batch.seed = batch.seed
    if hasattr(batch, "label"):
        reordered_batch.label = batch.label

    num_hops = int(batch.num_nodes_sampled.shape[0] / batch.num_graphs)
    num_nodes_sampled = [0] * num_hops
    for i, nns in enumerate(batch.num_nodes_sampled.numpy()):
        num_nodes_sampled[i % num_hops] += nns
    reordered_batch.num_nodes_sampled = torch.tensor(num_nodes_sampled, dtype=torch.long)

    return reordered_batch
