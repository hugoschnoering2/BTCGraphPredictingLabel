
import os
import pickle

import numpy as np
import yaml

from tqdm import tqdm
from collections import ChainMap

from mpire import WorkerPool

from data.utils import train_test_split
from data.postgresql import PostgresqlDataService as DataService, Condition


SIZE_TRANSACTION_EDGES = 785954737  # number of edges in the database


def construct_basic_query(nodes: list):
    query = (f"SELECT a, b FROM transaction_edges WHERE a IN ({','.join([str(n) for n in nodes])}) "
             f"OR b IN ({','.join([str(n) for n in nodes])})")
    return query


def construct_big_degree_query(node: int, frac_table_in_pct, sampling_method: str = "SYSTEM"):
    query = (f"SELECT a, b FROM transaction_edges "
             f"TABLESAMPLE {sampling_method} ({frac_table_in_pct}) "
             f"WHERE (a = {node} or b = {node})")
    return query


def construct_num_edges_query(nodes: list):
    query = (f"SELECT alias, (degree_in + degree_out) AS num_edges FROM node_features "
             f"WHERE alias IN ({','.join([str(n) for n in nodes])})")
    return query


def random_neighbor_sampling(seed_nodes: list, num_neighbors: list, connector=None,
                             max_edges_per_request: int = 100000, factor_sampling: int = 20) -> tuple[dict, dict, dict]:

    max_len = len(num_neighbors)
    nodes_visited = {seed_node: [seed_node] for seed_node in seed_nodes}
    nodes_to_be_visited = {seed_node: [seed_node] for seed_node in seed_nodes}
    edges_used = {seed_node: [] for seed_node in seed_nodes}
    num_nodes_sampled = {seed_node: [] for seed_node in seed_nodes}

    for k in range(max_len):

        all_nodes_to_be_visited = list(set([node for nodes in nodes_to_be_visited.values() for node in nodes]))
        new_nodes_found = {node: [] for node in all_nodes_to_be_visited}

        if len(all_nodes_to_be_visited) > 0:

            query_num_edges = construct_num_edges_query(nodes=all_nodes_to_be_visited)
            rows = DataService.execute_query_w_connector(connector=connector, query=query_num_edges, fetch="all")
            big_degree_nodes = [(row["alias"], row["num_edges"]) for row in rows if row["num_edges"]
                                and row["num_edges"] > max_edges_per_request]
            small_degree_nodes = [(row["alias"], row["num_edges"]) for row in rows if row["num_edges"]
                                  and row["num_edges"] <= max_edges_per_request]

            for node, num_edges in big_degree_nodes:
                frac_table_in_pct = num_neighbors[k] * factor_sampling / num_edges
                frac_table_in_pct = max(frac_table_in_pct, 0.0001) * 100
                query_edges = construct_big_degree_query(node=node, frac_table_in_pct=frac_table_in_pct)
                rows = DataService.execute_query_w_connector(connector=connector, query=query_edges, fetch="all")
                new_nodes_found[node].extend([row["b"] if row["a"] == node else row["a"] for row in rows])

            i = 0
            while i < len(small_degree_nodes):
                num_edges = 0
                nodes_step = []
                while (i < len(small_degree_nodes)
                       and (len(nodes_step) == 0
                            or num_edges + small_degree_nodes[i][1] <= max_edges_per_request)):
                    nodes_step.append(small_degree_nodes[i][0])
                    num_edges += small_degree_nodes[i][1]
                    i += 1
                query = construct_basic_query(nodes=nodes_step)
                rows = DataService.execute_query_w_connector(connector=connector, query=query, fetch="all")
                for row in rows:
                    if row["a"] in nodes_step:
                        new_nodes_found[row["a"]].append(row["b"])
                    if row["b"] in nodes_step:
                        new_nodes_found[row["b"]].append(row["a"])

        next_nodes_to_be_visited = {seed_node: [] for seed_node in seed_nodes}
        new_nodes_found = {node: list(set(nodes)) for node, nodes in new_nodes_found.items()}

        for seed_node in seed_nodes:

            for node in nodes_to_be_visited[seed_node]:

                destinations = [node for node in new_nodes_found[node] if node not in nodes_visited[seed_node]
                                and node not in next_nodes_to_be_visited[seed_node]]
                if len(destinations) > num_neighbors[k]:
                    destinations = list(np.random.choice(destinations, size=num_neighbors[k], replace=False))
                next_nodes_to_be_visited[seed_node].extend(destinations)
                edges_used[seed_node].extend([(node, dest) for dest in destinations])
                edges_used[seed_node].extend([(dest, node) for dest in destinations])

        nodes_to_be_visited = next_nodes_to_be_visited

        for seed_node in seed_nodes:
            nodes_visited[seed_node].extend(nodes_to_be_visited[seed_node])
            num_nodes_sampled[seed_node].append(len(nodes_to_be_visited[seed_node]))

    return nodes_visited, edges_used, num_nodes_sampled


class RandomNeighborSampler:

    def __init__(self, db: dict, num_neighbors: list, n_jobs=-1):

        self.db = db
        self.num_neighbors = num_neighbors
        self.n_jobs = n_jobs

    def sample(self, seed_nodes: list) -> tuple[dict, dict, dict]:

        with tqdm(total=len(seed_nodes), leave=False) as pbar:

            def init_db_conn(worker_state):
                worker_state["pool"] = DataService(**self.db).pool(min_connection=5, max_connection=20)

            def close_db_conn(worker_state):
                worker_state["pool"].closeall()

            def get_neighborhood(worker_state, node):
                connector = worker_state["pool"].getconn()
                neighborhood = random_neighbor_sampling(seed_nodes=[node], num_neighbors=self.num_neighbors,
                                                        connector=connector)
                worker_state["pool"].putconn(connector)
                pbar.update(1)
                return neighborhood

            with WorkerPool(n_jobs=self.n_jobs, start_method="threading", use_worker_state=True) as pool:
                neighborhoods = pool.map(get_neighborhood, seed_nodes, progress_bar=False,
                                         worker_init=init_db_conn, worker_exit=close_db_conn)

        nodes_visited = dict(ChainMap(*[neighborhood[0] for neighborhood in neighborhoods]))
        edges_used = dict(ChainMap(*[neighborhood[1] for neighborhood in neighborhoods]))
        num_nodes_sampled = dict(ChainMap(*[neighborhood[2] for neighborhood in neighborhoods]))

        return nodes_visited, edges_used, num_nodes_sampled


def create_buffer(config):

    if os.path.exists(config["buffer"]["folder"]):
        previous_config = os.path.join(config["buffer"]["folder"], "config.yaml")
        previous_config = yaml.load(open(previous_config, "r"), Loader=yaml.FullLoader)
        try:
            assert config["categories"] == previous_config["categories"]
            assert config["learning"]["train_test_split"] == previous_config["learning"]["train_test_split"]
            assert config["sampling"] == previous_config["sampling"]
            assert config["buffer"] == previous_config["buffer"]
            return
        except AssertionError:
            raise Exception("A buffer already exists and does not match the current parameters.")
        except Exception as e:
            raise e

    os.mkdir(config["buffer"]["folder"])
    labelled_nodes = DataService(**config["db"]).fetch(
        table="node_features", columns=["alias", "label"],
        conditions=[Condition("label", "IN", config["categories"])], limit=None)
    labelled_nodes = {int(row["alias"]): row["label"] for row in labelled_nodes}
    train_nodes, val_nodes, test_nodes = train_test_split(list(labelled_nodes.keys()),
                                                          **config["learning"]["train_test_split"])

    sampler = RandomNeighborSampler(db=config["db"], **config["sampling"])

    for i in range(config["buffer"]["size"]):

        print(f"Batch {i}")

        nodes_visited, edges_used, num_nodes_sampled = sampler.sample(seed_nodes=train_nodes)
        path = os.path.join(config["buffer"]["folder"], f"train_{i}")
        os.mkdir(path)
        with open(os.path.join(path, f"nodes_visited.pkl"), "wb") as f:
            pickle.dump(nodes_visited, f)
        with open(os.path.join(path, f"edges_used.pkl"), "wb") as f:
            pickle.dump(edges_used, f)
        with open(os.path.join(path, f"num_nodes_sampled.pkl"), "wb") as f:
            pickle.dump(num_nodes_sampled, f)

        nodes_visited, edges_used, num_nodes_sampled = sampler.sample(seed_nodes=val_nodes)
        path = os.path.join(config["buffer"]["folder"], f"val_{i}")
        os.mkdir(path)
        with open(os.path.join(path, f"nodes_visited.pkl"), "wb") as f:
            pickle.dump(nodes_visited, f)
        with open(os.path.join(path, f"edges_used.pkl"), "wb") as f:
            pickle.dump(edges_used, f)
        with open(os.path.join(path, f"num_nodes_sampled.pkl"), "wb") as f:
            pickle.dump(num_nodes_sampled, f)

        nodes_visited, edges_used, num_nodes_sampled = sampler.sample(seed_nodes=test_nodes)
        path = os.path.join(config["buffer"]["folder"], f"test_{i}")
        os.mkdir(path)
        with open(os.path.join(path, f"nodes_visited.pkl"), "wb") as f:
            pickle.dump(nodes_visited, f)
        with open(os.path.join(path, f"edges_used.pkl"), "wb") as f:
            pickle.dump(edges_used, f)
        with open(os.path.join(path, f"num_nodes_sampled.pkl"), "wb") as f:
            pickle.dump(num_nodes_sampled, f)

    with open(os.path.join(config["buffer"]["folder"], "config.yaml"), "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def load_sample(folder: str):
    with open(os.path.join(folder, "nodes_visited.pkl"), "rb") as f:
        nodes_visited = pickle.load(f)
    with open(os.path.join(folder, "edges_used.pkl"), "rb") as f:
        edges_used = pickle.load(f)
    with open(os.path.join(folder, "num_nodes_sampled.pkl"), "rb") as f:
        num_nodes_sampled = pickle.load(f)
    return nodes_visited, edges_used, num_nodes_sampled
