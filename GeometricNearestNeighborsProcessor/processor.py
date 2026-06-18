from __future__ import annotations

from typing import Callable, Iterable, Dict
import warnings

import numpy as np
import pandas as pd
from OutlierIdentifiers import (
    hampel_identifier_parameters,
    quartile_identifier_parameters,
    splus_quartile_identifier_parameters,
)
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class GeometricNearestNeighborsProcessor:
    """Chainable processor for geometric nearest-neighbor workflows."""

    def __init__(self, data: pd.DataFrame | np.ndarray | Iterable[Iterable[float|int]] | Dict[str, Iterable[float|int]]) -> None:
        self.value = None
        self.data = None
        self.number_of_nns = None
        self.distance_function = None
        self.nearest_neighbor_distances = None
        self.radius_function = None
        self.radius = None
        self.lower_threshold = None
        self.upper_threshold = None
        self.kd_tree_object = None
        self.point_ids = None
        self.set_data(data)

    @staticmethod
    def _is_str_list(obj):
        return isinstance(obj, list) and all([isinstance(x, str) for x in obj])

    @staticmethod
    def _normalize_distance_function(distance_function: str | None) -> str:
        if distance_function is None:
            return "euclidean"
        s = str(distance_function).strip().lower()
        aliases = {
            "euclidean": "euclidean",
            "euclideandistance": "euclidean",
            "cosine": "cosine",
            "cosinedistance": "cosine",
            "minkowski": "minkowski",
            "minkowskidistance": "minkowski",
            "cityblock": "cityblock",
            "cityblockdistance": "cityblock",
            "taxicab": "manhattan",
            "taxicabdistance": "manhattan",
            "manhattan": "manhattan",
            "manhattandistance": "manhattan",
            "haversine": "haversine",
            "haversinedistance": "haversine",
        }
        if s not in aliases:
            raise ValueError(f"Unknown distance function: '{distance_function}'. The following distance functions are supported: '{"','".join(aliases.keys())}'.")
        return aliases[s]

    @staticmethod
    def _normalize_method(method: str | None) -> str:
        if method is None:
            return "kdtree"
        s = str(method).strip().lower()
        aliases = {
            "kdtree": "kdtree",
            "kd_tree": "kdtree",
            "scan": "scan",
            "brute": "scan",
        }
        if s not in aliases:
            raise ValueError(f"Unknown method: {method}.")
        return aliases[s]

    @staticmethod
    def _to_frame(data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError("Data must be a 2D array-like object.")
            columns = [f"x{i + 1}" for i in range(arr.shape[1])]
            frame = pd.DataFrame(arr, columns=columns)
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError("Data must have at least one row and one column.")
        if frame.index.isnull().any():
            frame = frame.reset_index(drop=True)
        return frame

    @staticmethod
    def _resolve_aggregation_function(aggregation_function):
        if callable(aggregation_function):
            return aggregation_function
        if aggregation_function is None:
            return "mean"
        agg = str(aggregation_function).strip().lower()
        mapping = {
            "mean": "mean",
            "avg": "mean",
            "median": "median",
            "max": "max",
            "min": "min",
        }
        if agg not in mapping:
            raise ValueError(f"Unknown aggregation function: {aggregation_function}")
        return mapping[agg]

    @staticmethod
    def _resolve_outlier_identifier(outlier_identifier) -> Callable[[Iterable[float]], tuple[float, float]]:
        if callable(outlier_identifier):
            return outlier_identifier
        if outlier_identifier is None:
            return hampel_identifier_parameters
        ident = str(outlier_identifier).strip().lower()
        mapping = {
            "hampelidentifierparameters": hampel_identifier_parameters,
            "hampel_identifier_parameters": hampel_identifier_parameters,
            "quartileidentifierparameters": quartile_identifier_parameters,
            "quartile_identifier_parameters": quartile_identifier_parameters,
            "splusquartileidentifierparameters": splus_quartile_identifier_parameters,
            "splus_quartile_identifier_parameters": splus_quartile_identifier_parameters,
        }
        if ident not in mapping:
            raise ValueError(f"Unknown outlier identifier: {outlier_identifier}.")
        return mapping[ident]

    def set_value(self, value):
        self.value = value
        return self

    def take_value(self):
        return self.value

    def echo_value(self):
        print(self.value)
        return self

    def echo_function_value(self, label: str = '', function = None):
        func = function
        if func is None:
            func = lambda x: x
        if callable(func):
            if label:
                print(label)
            print(func(self.value))
            return self
        raise ValueError("The argument function is expected to be a callable or None.")


    def set_data(self, data):
        if isinstance(data, dict):
            self.set_data(list(data.values()))
            self.set_point_ids(list(data.keys()))
            return self
        frame = self._to_frame(data)
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=True)
        self.data = frame
        width = max(1, len(str(len(frame))))
        self.point_ids = [f"p{i:0{width}d}" for i in range(0, len(frame))]
        return self

    def take_data(self, copy: bool = True) -> pd.DataFrame:
        if copy:
            return self.data.copy()
        else:
            return self.data

    def set_point_identifiers(self, identifiers):
        return self.set_point_ids(identifiers)

    def take_point_identifiers(self):
        return self.take_point_ids()

    def set_point_ids(self, ids):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Cannot find data.")
        if self._is_str_list(ids) and len(ids) == self.data.shape[0]:
            self.point_ids = ids
            if len(list(dict.fromkeys(ids))) < self.data.shape[0]:
                warnings.warn("Not all elements are unique.", UserWarning)
        else:
            raise ValueError(f"The first argument is expected to be a string list of length {self.data.shape[0]}.")
        return self

    def take_point_ids(self):
        return self.point_ids

    def set_number_of_nns(self, number_of_nns):
        self.number_of_nns = int(number_of_nns) if number_of_nns is not None else None
        return self

    def take_number_of_nns(self):
        return self.number_of_nns

    def set_distance_function(self, distance_function):
        self.distance_function = self._normalize_distance_function(distance_function) if distance_function is not None else None
        return self

    def take_distance_function(self):
        return self.distance_function

    def set_nearest_neighbor_distances(self, nearest_neighbor_distances):
        self.nearest_neighbor_distances = nearest_neighbor_distances
        return self

    def take_nearest_neighbor_distances(self):
        return self.nearest_neighbor_distances

    def set_radius_function(self, radius_function):
        self.radius_function = radius_function
        return self

    def take_radius_function(self):
        return self.radius_function

    def set_radius(self, radius):
        self.radius = radius
        return self

    def take_radius(self):
        return self.radius

    def set_lower_threshold(self, lower_threshold):
        self.lower_threshold = lower_threshold
        return self

    def take_lower_threshold(self):
        return self.lower_threshold

    def set_upper_threshold(self, upper_threshold):
        self.upper_threshold = upper_threshold
        return self

    def take_upper_threshold(self):
        return self.upper_threshold

    def set_kd_tree_object(self, kd_tree_object):
        self.kd_tree_object = kd_tree_object
        return self

    def take_kd_tree_object(self):
        return self.kd_tree_object

    def create_kd_tree_object(self, distance_function: str = "euclidean", algorithm: str = "kd_tree"):
        metric = self._normalize_distance_function(distance_function)
        tree = NearestNeighbors(metric=metric, algorithm=algorithm)
        tree.fit(self.data.to_numpy(dtype=float))
        self.kd_tree_object = tree
        self.distance_function = metric
        return self

    def make_nearest_function(self, distance_function: str = "euclidean", method: str | None = "kdtree"):
        methodLocal = self._normalize_method(method)
        if method == "kdtree":
            algorithm = "kd_tree"
        else:
            algorithm = "brute"
        return self.create_kd_tree_object(distance_function=distance_function, algorithm=algorithm)

    def compute_matrix_distances(self, point, distance_function: str | None = "euclidean"):
        metric = self._normalize_distance_function(distance_function or self.distance_function)
        p = np.asarray(point, dtype=float).reshape(1, -1)
        data = self.data.to_numpy(dtype=float)
        if p.shape[1] != data.shape[1]:
            raise ValueError("Point dimension must match data dimension.")
        if metric == "euclidean":
            dists = np.linalg.norm(data - p, axis=1)
        elif metric == "cosine":
            data_norm = np.linalg.norm(data, axis=1)
            p_norm = np.linalg.norm(p[0])
            denom = np.where(data_norm == 0, 1.0, data_norm) * (p_norm if p_norm != 0 else 1.0)
            sim = np.sum(data * p[0], axis=1) / denom
            dists = 1.0 - sim
        else:
            raise ValueError(f"Unknown distance function: {metric}")
        self.value = pd.Series(dists, index=self.point_ids, name="Distance")
        return self

    def compute_nearest_neighbor_distances(
        self,
        n_top_nns: int = 6,
        radius: float = np.inf,
        distance_function: str = "euclidean",
    ):
        data = self.data.to_numpy(dtype=float)
        if n_top_nns < 1 or n_top_nns > len(data):
            raise ValueError("n_top_nns must be between 1 and number of data rows.")
        metric = self._normalize_distance_function(distance_function)
        nn = NearestNeighbors(n_neighbors=min(n_top_nns + 1, len(data)), metric=metric)
        nn.fit(data)
        distances, indices = nn.kneighbors(data)
        rows = []
        for i in range(len(data)):
            sid = self.point_ids[i]
            for dist, idx in zip(distances[i][1:], indices[i][1:]):
                if np.isfinite(radius) and dist > radius:
                    continue
                rows.append(
                    {
                        "SearchID": sid,
                        "SearchIndex": i + 1,
                        "ID": self.point_ids[int(idx)],
                        "Distance": float(dist),
                    }
                )
        df = pd.DataFrame(rows, columns=["SearchID", "SearchIndex", "ID", "Distance"])
        self.nearest_neighbor_distances = df
        self.number_of_nns = int(n_top_nns)
        self.distance_function = metric
        return self

    def compute_thresholds(
        self,
        number_of_nearest_neighbors: int | None = None,
        aggregation_function="mean",
        outlier_identifier="hampel_identifier_parameters",
    ):
        if self.nearest_neighbor_distances is None:
            n = number_of_nearest_neighbors or 6
            self.compute_nearest_neighbor_distances(
                n_top_nns=n,
                distance_function=self.distance_function or "euclidean",
            )
        elif number_of_nearest_neighbors is not None and number_of_nearest_neighbors != self.number_of_nns:
            self.compute_nearest_neighbor_distances(
                n_top_nns=number_of_nearest_neighbors,
                distance_function=self.distance_function or "euclidean",
            )

        agg = self._resolve_aggregation_function(aggregation_function)
        ident = self._resolve_outlier_identifier(outlier_identifier)
        grouped = self.nearest_neighbor_distances.groupby("SearchIndex")["Distance"]
        stats = grouped.agg(Radius=agg, Mean="mean", SD="std").reset_index()
        thresholds = ident(stats["Radius"].to_numpy())
        if not (isinstance(thresholds, tuple | list | np.ndarray) and len(thresholds) == 2):
            raise ValueError("Outlier identifier must return (lower_threshold, upper_threshold).")

        self.value = stats
        self.radius_function = agg
        if callable(agg):
            self.radius = float(agg(stats["Radius"].to_numpy()))
        else:
            self.radius = float(getattr(np, agg)(stats["Radius"].to_numpy()))
        self.lower_threshold = float(thresholds[0])
        self.upper_threshold = float(thresholds[1])
        return self

    def find_nearest(
        self,
        point,
        n: int = 12,
        radius: float = np.inf,
        method: str | None = "kdtree",
        distance_function: str | None = "euclidean",
    ):
        methodLocal = self._normalize_method(method)
        metric = self._normalize_distance_function(distance_function or self.distance_function)
        p = np.asarray(point, dtype=float).reshape(1, -1)

        if methodLocal == "scan":
            dists = self.compute_matrix_distances(point=p[0], distance_function=metric).take_value()
            order = np.argsort(dists.to_numpy())[: min(n, len(dists))]
            df = pd.DataFrame({"Index": order, "Distance": dists.to_numpy()[order]})
            if np.isfinite(radius):
                df = df[df["Distance"] <= radius]
        elif methodLocal == "kdtree":
            if self.kd_tree_object is None:
                self.create_kd_tree_object(distance_function=metric)
            model = self.kd_tree_object
            if np.isfinite(radius):
                idxs, dists = model.radius_neighbors(p, radius=radius, sort_results=True)
                idx = idxs[0]
                dist = dists[0]
                if len(idx) > n:
                    idx = idx[:n]
                    dist = dist[:n]
            else:
                dist, idx = model.kneighbors(p, n_neighbors=min(n, len(self.data)))
                idx = idx[0]
                dist = dist[0]
            df = pd.DataFrame({"Index": idx, "ID": [self.point_ids[i] for i in idx], "Distance": dist})
        else:
            raise ValueError('method must be one of None, "scan", or "kdtree".')

        self.value = df.reset_index(drop=True)
        return self

    def classify(self, points=None, method: str | None = None):
        if self.number_of_nns is None or self.upper_threshold is None or self.radius_function is None:
            raise ValueError("Run compute_thresholds before classify.")

        if points is None:
            points_frame = self.data.copy()
        else:
            points_frame = self._to_frame(points)
        if points_frame.shape[1] != self.data.shape[1]:
            raise ValueError("Points dimension must match data dimension.")

        rows = []
        for i, row in points_frame.iterrows():
            recs = self.find_nearest(
                point=row.to_numpy(dtype=float),
                n=self.number_of_nns,
                method=method,
                distance_function=self.distance_function or "euclidean",
            ).take_value()
            if callable(self.radius_function):
                mrad = float(self.radius_function(recs["Distance"].to_numpy()))
            else:
                mrad = float(getattr(np, str(self.radius_function))(recs["Distance"].to_numpy()))
            rows.append({"Index": int(i), "Radius": mrad, "Label": bool(mrad <= self.upper_threshold)})

        self.value = pd.DataFrame(rows, columns=["Index", "Radius", "Label"])
        return self

    def find_anomalies(self, points=None, method: str | None = None):
        if points is None:
            base_points = self.data.copy()
        else:
            base_points = self._to_frame(points)
        classified = self.classify(base_points, method=method).take_value()
        anomaly_rows = classified[~classified["Label"]].copy()
        if anomaly_rows.empty:
            self.value = base_points.iloc[0:0].copy()
            self.value["Radius"] = pd.Series(dtype=float)
            return self

        idx = (anomaly_rows["Index"].to_numpy(dtype=int)).tolist()
        points_df = base_points.iloc[idx].reset_index(drop=True)
        points_df["Radius"] = anomaly_rows["Radius"].to_numpy()
        self.value = points_df
        return self

    def compute_proximity_matrix(self, n: int | None = None):
        if self.nearest_neighbor_distances is None:
            raise ValueError("Run compute_nearest_neighbor_distances first.")
        df = self.nearest_neighbor_distances.copy()
        if n is None:
            counts = df.groupby("SearchIndex").size()
            if counts.empty:
                raise ValueError("No nearest-neighbor distances are available.")
            n = int(np.floor(counts.min()))
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1.")

        ids = list(self.point_ids)
        diag_df = pd.DataFrame({"SearchID": ids, "ID": ids, "Distance": np.zeros(len(ids))})
        df = pd.concat([df[["SearchID", "ID", "Distance"]], diag_df], ignore_index=True)

        id_to_idx = {sid: i for i, sid in enumerate(ids)}
        r = df["SearchID"].map(id_to_idx).to_numpy()
        c = df["ID"].map(id_to_idx).to_numpy()
        x = df["Distance"].to_numpy(dtype=float)

        positive = x[x > 0]
        if len(positive) > 0:
            breaks = np.quantile(positive, q=np.linspace(0.0, 1.0, num=n + 1))
            bins = np.digitize(x, bins=breaks, right=False)
            m = int(np.max(bins))
            if m > 0:
                m2 = float(m * m)
                x = (m2 + bins - m * bins) / m2
        mat = sparse.csr_matrix((x, (r, c)), shape=(len(ids), len(ids)))
        mat.setdiag(1.0)
        self.value = mat
        return self

    def plot_points(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self, data=None, x=None, y=None, title=None, template=None, **kwargs):
        import plotly.express as px

        frame = self._to_frame(data) if data is not None else (self.value if isinstance(self.value, pd.DataFrame) else self.data)
        if frame.shape[1] == 2:
            raise ValueError("Plotting requires exactly two columns of object's data.")
        x_col = x or frame.columns[0]
        y_col = y or frame.columns[1]
        fig = px.scatter(frame, x=x_col, y=y_col, title=title, template=template, **kwargs)

        self.value = fig
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            res = "GeometricNearestNeighborsProcessor object with %d points of dimension %d." % self.data.shape
        else:
            res = "GeometricNearestNeighborsProcessor object without points."

        return res

    def __repr__(self):
        """Representation of GeometricNearestNeighborsProcessor object."""
        if isinstance(self.take_data(), pd.DataFrame):
            return f"<GeometricNearestNeighborsProcessor object with data shape dimensions {self.take_data().shape}\n" + f"\tand distance function \"{self.take_distance_function()}\">"
        else:
            return "GeometricNearestNeighborsProcessor object without points."