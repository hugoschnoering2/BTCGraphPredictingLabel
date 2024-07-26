
import json

import numpy as np
import pandas as pd

from copy import deepcopy

from joblib import delayed, Parallel


with open("data/block-dates.json", "r") as f:
    block_dates = json.load(f)


with open("data/market-price.json", "r") as f:
    market_price = json.load(f)
market_price = pd.DataFrame(market_price["market-price"]).astype(float)
market_price.columns = ["timestamp", "price"]
market_price["timestamp"] /= 1000.
market_price = market_price.set_index("timestamp")
market_price = market_price.replace(to_replace=0., value=np.nan).dropna()
market_price = market_price["price"]
min_index_price = market_price.index.min()


dict_pre_processing_degree = {"transform": "log",
                              "normalization": True, "norm_quantile_min": 0., "norm_quantile_max": 0.95,
                              "clip_final_min": 0, "clip_final_max": 1,
                              "fill_na": 0.
                              }

dict_pre_processing_sent = {"transform": "log",
                            "normalization": True, "norm_quantile_min": 0.05, "norm_quantile_max": 0.95,
                            "clip_final_min": 0, "clip_final_max": 1,
                            "fill_na": 0.
                            }


class Preprocessing:

    def __init__(self, dict_pre_processing: dict = None):

        if dict_pre_processing:
            self.dict_pre_processing = dict_pre_processing
        else:
            self.dict_pre_processing = {

                # already present
                "degree": dict_pre_processing_degree,
                "degree_in": dict_pre_processing_degree,
                "degree_out": dict_pre_processing_degree,
                "total_transactions_in": dict_pre_processing_degree,
                "total_transactions_out": dict_pre_processing_degree,
                "min_sent": dict_pre_processing_sent,
                "total_sent": dict_pre_processing_sent,
                "max_sent": dict_pre_processing_sent,
                "min_received": dict_pre_processing_sent,
                "total_received": dict_pre_processing_sent,
                "max_received": dict_pre_processing_sent,
                "cluster_size": dict_pre_processing_degree,
                "cluster_num_edges": dict_pre_processing_degree,
                "cluster_num_nodes_in_cc": dict_pre_processing_degree,

                # to be computed
                "avg_sent": dict_pre_processing_sent,
                "avg_received": dict_pre_processing_sent,

                "age": dict_pre_processing_degree,

                "degree_age_ratio": dict_pre_processing_degree,
                "degree_in_age_ratio": dict_pre_processing_degree,
                "degree_out_age_ratio": dict_pre_processing_degree,

                "transactions_in_age_ratio": dict_pre_processing_degree,
                "transactions_out_age_ratio": dict_pre_processing_degree,

                "cluster_size_age_ratio": dict_pre_processing_degree,
                "cluster_num_edges_age_ratio": dict_pre_processing_degree,

                "proportion_nodes_in_cc": {"fill_na": 1.},
                "time_before_first_transaction": dict_pre_processing_degree,
                "degree_out_in_ratio": dict_pre_processing_degree,

                "min_sent_usd": dict_pre_processing_sent,
                "total_sent_usd": dict_pre_processing_sent,
                "avg_sent_usd": dict_pre_processing_sent,
                "max_sent_usd": dict_pre_processing_sent,
                "min_received_usd": dict_pre_processing_sent,
                "total_received_usd": dict_pre_processing_sent,
                "max_received_usd": dict_pre_processing_sent,
                "avg_received_usd": dict_pre_processing_sent,

            }

        self.dict_pre_processing = {k: deepcopy(v) for k, v in self.dict_pre_processing.items()}
        self.columns = None

    def fit(self, X: pd.DataFrame):
        new_features = self._compute_new_features(X=X)
        new_features = pd.concat([X.copy(), new_features], axis=1)
        self._compute_norm_bounds(X=new_features)
        self.columns = [c for c in new_features.columns if c in self.dict_pre_processing]

    def _compute_new_features(self, X: pd.DataFrame):

        new_features = []

        avg_sent = X["total_sent"] / X["total_transactions_out"].fillna(1.)
        avg_sent.name = "avg_sent"
        new_features.append(avg_sent)

        avg_received = X["total_received"] / X["total_transactions_in"].fillna(1.)
        avg_received.name = "avg_received"
        new_features.append(avg_received)

        ages = self._compute_age(X)
        ages.name = "age"
        new_features.append(ages)

        degree_age_ratio = X["degree"] / np.clip(ages.fillna(1.), 1., np.inf)
        degree_age_ratio.name = "degree_age_ratio"
        new_features.append(degree_age_ratio)

        degree_in_age_ratio = X["degree_in"] / np.clip(ages.fillna(1.), 1., np.inf)
        degree_in_age_ratio.name = "degree_in_age_ratio"
        new_features.append(degree_in_age_ratio)

        degree_out_age_ratio = X["degree_out"] / np.clip(ages.fillna(1.), 1., np.inf)
        degree_out_age_ratio.name = "degree_out_age_ratio"
        new_features.append(degree_out_age_ratio)

        transactions_in_age_ratio = X["total_transactions_in"] / np.clip(ages.fillna(1.), 1., np.inf)
        transactions_in_age_ratio.name = "transactions_in_age_ratio"
        new_features.append(transactions_in_age_ratio)

        transactions_out_age_ratio = X["total_transactions_out"] / np.clip(ages.fillna(1.), 1., np.inf)
        transactions_out_age_ratio.name = "transactions_out_age_ratio"
        new_features.append(transactions_out_age_ratio)

        cluster_size_age_ratio = X["cluster_size"] / np.clip(ages.fillna(1.), 1., np.inf)
        cluster_size_age_ratio.name = "cluster_size_age_ratio"
        new_features.append(cluster_size_age_ratio)

        cluster_num_edges_age_ratio = X["cluster_num_edges"] / np.clip(ages.fillna(1.), 1., np.inf)
        cluster_num_edges_age_ratio.name = "cluster_num_edges_age_ratio"
        new_features.append(cluster_num_edges_age_ratio)

        proportion_nodes_in_cc = X["cluster_num_nodes_in_cc"] / X["cluster_size"]
        proportion_nodes_in_cc.name = "proportion_nodes_in_cc"
        new_features.append(proportion_nodes_in_cc)

        time_before_first_transaction = X["first_transaction_out"] - X["first_transaction_in"]
        time_before_first_transaction.name = "time_before_first_transaction"
        new_features.append(time_before_first_transaction)

        degree_out_in_ratio = X["degree_out"] / X["degree_in"]
        degree_out_in_ratio.name = "degree_out_in_ratio"
        new_features.append(degree_out_in_ratio)

        median_prices = self._compute_median_price(X)

        min_sent_usd = X["min_sent"] * median_prices / (10 ** 8)
        min_sent_usd.name = "min_sent_usd"
        new_features.append(min_sent_usd)

        total_sent_usd = X["total_sent"] * median_prices / (10 ** 8)
        total_sent_usd.name = "total_sent_usd"
        new_features.append(total_sent_usd)

        avg_sent_usd = X["total_sent"] / X["total_transactions_out"].fillna(1.) * median_prices / (10 ** 8)
        avg_sent_usd.name = "avg_sent_usd"
        new_features.append(avg_sent_usd)

        max_sent_usd = X["max_sent"] * median_prices / (10 ** 8)
        max_sent_usd.name = "max_sent_usd"
        new_features.append(max_sent_usd)

        min_received_usd = X["min_received"] * median_prices / (10 ** 8)
        min_received_usd.name = "min_received_usd"
        new_features.append(min_received_usd)

        total_received_usd = X["total_received"] * median_prices / (10 ** 8)
        total_received_usd.name = "total_received_usd"
        new_features.append(total_received_usd)

        avg_received_usd = X["total_received"] / X["total_transactions_in"].fillna(1.) * median_prices / (10 ** 8)
        avg_received_usd.name = "avg_received_usd"
        new_features.append(avg_received_usd)

        max_received_usd = X["max_received"] * median_prices / (10 ** 8)
        max_received_usd.name = "max_received_usd"
        new_features.append(max_received_usd)

        new_features = pd.concat(new_features, axis=1)

        return new_features

    def _compute_median_price(self, X: pd.DataFrame):

        def compute_median_price(row):
            alias, first_block, last_block_in, last_block_out = row
            if np.isnan(first_block) or (np.isnan(last_block_in) and np.isnan(last_block_out)):
                return alias, np.nan
            elif np.isnan(last_block_in):
                last_block = last_block_out
            elif np.isnan(last_block_out):
                last_block = last_block_in
            else:
                last_block = max(last_block_in, last_block_out)
            first_block = int(1000 * (first_block // 1000))
            first_block_ts = block_dates.get(str(first_block))
            last_block = int(1000 * (last_block // 1000 + 1))
            last_block_ts = block_dates.get(str(last_block))
            if (first_block_ts is None or last_block_ts is None) or first_block_ts < min_index_price:
                return alias, np.nan
            prices_when_active = market_price.loc[
                (market_price.index >= first_block_ts) & (market_price.index <= last_block_ts)]
            return alias, prices_when_active.median()

        rows = X.reset_index()[["alias", "first_transaction_in", "last_transaction_in", "last_transaction_out"]].values

        median_prices = Parallel(n_jobs=-1)(delayed(compute_median_price)(row) for row in rows)
        median_prices = pd.Series([e[1] for e in median_prices], index=[e[0] for e in median_prices])

        return median_prices

    def _compute_age(self, X: pd.DataFrame):
        last_active = X[["last_transaction_out", "last_transaction_in"]].max(axis=1)
        first_active = X[["first_transaction_out", "first_transaction_in"]].min(axis=1)
        return last_active - first_active

    def _compute_norm_bounds(self, X: pd.DataFrame):

        for col, d in self.dict_pre_processing.items():
            x = X[col]
            tf = d.get("transform")
            if tf == "log":
                x = x[x > 0]
                x = np.log(x)
            if d.get("normalization", False):
                v_min = np.quantile(x, d.get("norm_quantile_min", 0.))
                v_max = np.quantile(x, d.get("norm_quantile_max", 1.))
                d["norm_value_min"] = v_min
                d["norm_value_max"] = v_max

    def transform(self, X: pd.DataFrame):
        preprocessed_features = []
        new_features = self._compute_new_features(X=X)
        new_features = pd.concat([X.copy(), new_features], axis=1)
        for col, d in self.dict_pre_processing.items():
            x = new_features[col]
            tf = d.get("transform")
            if tf == "log":
                x = x[x > 0]
                x = np.log(x)
            if d.get("normalization", False):
                x = (x - d["norm_value_min"]) / (d["norm_value_max"] - d["norm_value_min"])
            x = np.clip(x, d.get("clip_final_min", -np.inf), d.get("clip_final_max", np.inf))
            x = x.reindex(X.index)
            x = x.fillna(d["fill_na"])
            x.name = col
            preprocessed_features.append(x)
        return pd.concat(preprocessed_features, axis=1)[self.columns]
