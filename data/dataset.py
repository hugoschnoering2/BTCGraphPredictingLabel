
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

from data.postgresql import PostgresqlDataService, Condition
from data.sampler import load_sample

import torch
from torch_geometric.data import Data


class NodeFeaturesExtractor:

    def __init__(self, db: dict, max_num_nodes: int):

        self.db = db
        self.max_num_nodes = max_num_nodes

    def extract(self, nodes: list, label_values: list = None):

        if label_values:
            label_indexes = {label: i for i, label in enumerate(label_values)}

        with tqdm(total=len(nodes), leave=False, disable=True) as pbar:

            ds = PostgresqlDataService(**self.db)

            features = []

            i = 0
            while i < len(nodes):
                nodes_step = []
                while (i < len(nodes)
                       and (len(nodes_step) <= self.max_num_nodes)):
                    nodes_step.append(nodes[i])
                    i += 1
                rows = ds.fetch(table="node_features", conditions=[Condition("alias", "IN", nodes_step)])
                pbar.update(len(rows))
                features.extend([dict(row) for row in rows])

        columns = list(features[0].keys())
        columns = [col for col in columns if col != "label" and col != "alias"]

        X = {row["alias"]: [row[col] for col in columns] for row in features}
        if label_values:
            y = {row["alias"]: label_indexes.get(row["label"]) for row in features}
        else:
            y = {row["alias"]: row["label"] for row in features}

        X = pd.DataFrame(X).T
        X.columns = columns
        X.index.name = "alias"

        return X, y


class DatasetBuilder:

    def build(self, nodes_visited: dict, edges_used: dict, num_nodes_sampled: dict,
              X: dict, y: dict, min_num_samples_per_label: int = None,
              max_num_samples_per_label: int = None):

        dataset = []

        label_to_samples = dict()

        nodes_visited_list = list(nodes_visited.keys())
        random.shuffle(nodes_visited_list)

        for k, seed in enumerate(nodes_visited_list):

            if y.get(seed) is None:
                continue

            nv = nodes_visited[seed]
            nv_dict = {node: i for i, node in enumerate(nv)}
            eu = edges_used[seed]
            nns = num_nodes_sampled[seed]

            x = np.array([X[node] for node in nv]).astype(float)
            x = np.nan_to_num(x, nan=0.)

            edge_index = np.array([(nv_dict[a], nv_dict[b]) for a, b in eu]).T
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            graph = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)
            graph.num_nodes_sampled = torch.tensor(nns, dtype=torch.long)
            graph.label = y[seed]
            graph.seed = seed

            label = y[seed]
            if label not in label_to_samples:
                label_to_samples[label] = []
            if (max_num_samples_per_label is not None) and (len(label_to_samples[label]) > max_num_samples_per_label):
                continue
            label_to_samples[label].append(len(dataset))

            dataset.append(graph)

        if min_num_samples_per_label:
            for label, samples in label_to_samples.items():
                num_labels = len(samples)
                if num_labels < min_num_samples_per_label:
                    new_samples = np.random.choice(samples, size=min_num_samples_per_label-num_labels,
                                                   replace=True)
                    for new_sample in new_samples:
                        dataset.append(dataset[new_sample])

        return dataset


def create_dataset(folder: str, config, preprocessing=None, fit_preprocessing: bool = False,
                   min_num_samples_per_label: int = None, max_num_samples_per_label: int = None):
    extractor = NodeFeaturesExtractor(db=config["db"], max_num_nodes=10000)
    nodes_visited, edges_used, num_nodes_sampled = load_sample(folder)
    X, y = extractor.extract(nodes=list(set([e for nodes in nodes_visited.values() for e in nodes])),
                             label_values=config["categories"])
    if fit_preprocessing:
        preprocessing.fit(X)
    X = preprocessing.transform(X)
    X = {alias: list(row) for alias, row in X.iterrows()}
    dataset = DatasetBuilder().build(nodes_visited=nodes_visited, edges_used=edges_used,
                                     num_nodes_sampled=num_nodes_sampled, X=X, y=y,
                                     min_num_samples_per_label=min_num_samples_per_label,
                                     max_num_samples_per_label=max_num_samples_per_label)
    return dataset
