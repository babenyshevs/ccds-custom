from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import OPTICS, AffinityPropagation, KMeans
from tqdm import tqdm

from src.features.math import get_derivative


class ClusteringMetricsCalculator:
    def __init__(self):
        """
        Initializes the ClusteringMetricsCalculator instance.

        Example:
            mc = ClusteringMetricsCalculator()
            mc.calculate_metrics(clustering_instance1)
            mc.calculate_metrics(clustering_instance2)
            all_metrics_df = mc.get_metrics()
        """
        self.metrics = {}

    def calculate_metrics(self, clustering_instance) -> None:
        """
        Calculates clustering evaluation metrics for a given clustering instance and stores them in the metrics dictionary.

        Args:
            clustering_instance: A fitted clustering instance.
        """
        try:
            labels = clustering_instance.labels
        except AttributeError as e:
            print(f"{e}. Perhaps, you didn't fit the model yet.")
            return None

        name = type(clustering_instance).__name__
        X = clustering_instance.X

        silhouette_coef = np.round(metrics.silhouette_score(X, labels, metric="euclidean"), 2)
        calinski_harabasz_index = np.round(metrics.calinski_harabasz_score(X, labels), 2)
        davies_bouldin_index = np.round(metrics.davies_bouldin_score(X, labels), 2)

        if name == "KMeansClustering":
            inertia = clustering_instance.kmeans.inertia_
        else:
            inertia = None

        self.metrics[name] = {
            "Silhouette Coefficient": silhouette_coef,
            "Calinski-Harabasz Index": calinski_harabasz_index,
            "Davies-Bouldin Index": davies_bouldin_index,
            "Distortion": inertia,
        }

    def get_metrics(self) -> pd.DataFrame:
        """
        Returns the stored metrics as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing metrics of different clustering algorithms.
        """
        return pd.DataFrame(self.metrics)


class AffinityPropagationClustering:
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the clustering instance with an optional random state seed.

        Args:
            seed (Optional[int]): Random state seed for reproducibility.

        Example:
            apc = AffinityPropagationClustering(seed=SEED)
            params = {
                'damping': 0.9,
                'max_iter': 250,
                'convergence_iter': 20,
                'preference': -50,
                'affinity': 'euclidean'
            }
            apc.fit(X, params=params)
            metrics = apc.calculate_metrics()
            apc.plot_clusters()
            details = apc.get_cluster_details()
        """
        self.seed = seed

    def fit(self, X: np.ndarray, params: Optional[Dict[str, float]] = None) -> None:
        """
        Fits the AffinityPropagation model to the data.

        Args:
            X (np.ndarray): Data to fit the model.
            params (Optional[Dict[str, float]]): Parameters for AffinityPropagation.
                - damping (float, optional): Damping factor (between 0.5 and 1). Default is 0.5.
                - max_iter (float, optional): Maximum number of iterations. Default is 200.
                - convergence_iter (float, optional): Iterations with no change in the number of estimated clusters
                    that stop the convergence. Default is 15.
                - preference (float, optional): Preferences for points, if None, median of pairwise distances is used.
                - affinity (str, optional): Metric used to compute the affinity matrix. Default is 'euclidean'.
        """
        self.X = X
        if params is None:
            params = {}

        self.af = AffinityPropagation(
            damping=params.get("damping", 0.5),
            max_iter=params.get("max_iter", 200),
            convergence_iter=params.get("convergence_iter", 15),
            preference=params.get("preference", None),
            affinity=params.get("affinity", "euclidean"),
            random_state=self.seed,
        ).fit(self.X)

        self.cluster_centers_indices = self.af.cluster_centers_indices_
        self.labels = self.af.labels_
        self.n_clusters = len(self.cluster_centers_indices)

    def plot_clusters(self) -> None:
        """
        Plots the clusters.
        """
        try:
            _ = self.n_clusters
        except AttributeError as e:
            print(f"{e}. Perhaps, you didn't fit the model yet.")
            return None

        plt.close("all")
        plt.figure(1)
        plt.clf()
        colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, self.n_clusters)))

        for k, col in zip(range(self.n_clusters), colors):
            class_members = self.labels == k
            cluster_center = self.X[self.cluster_centers_indices[k]]
            plt.scatter(
                self.X[class_members, 0],
                self.X[class_members, 1],
                color=col["color"],
                marker=".",
            )
            plt.scatter(cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o")
            for x in self.X[class_members]:
                plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"])

        plt.title(f"Number of clusters: {self.n_clusters}")
        plt.show()

    def get_cluster_details(self) -> Dict[str, np.ndarray]:
        """
        Returns the details of the clusters.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing cluster details.
        """
        return {
            "n_clusters": self.n_clusters,
            "cluster_centers_indices": self.cluster_centers_indices,
            "labels": self.labels,
        }


class KMeansClustering:
    def __init__(self, X: np.ndarray, seed: int = None):
        """
        Initializes the KMeansClustering instance.

        Args:
            X (np.ndarray): Data to fit the model.
            seed (int): Random state seed for reproducibility. Default is None.

        Example:
            km = KMeansClustering(X, seed=SEED)
            km.fit(n_clusters=10, X)
            km.plot_clusters()
            km.grid_search(cluster_list=[5, 10, 20, 40, 80, 160, 320])
            km.plot_elbow_graph("Distortion")
            km.plot_elbow_graph("Silhouette")
        """
        self.X = X
        self.seed = seed

    def fit(self, n_clusters: int) -> None:
        """
        Fits the KMeans model to the data.

        Args:
            n_clusters (int): The number of clusters to form.
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.seed, n_init=10, init="k-means++"
        ).fit(self.X)
        self.labels = self.kmeans.labels_
        self.cluster_centers = self.kmeans.cluster_centers_

    def plot_clusters(self) -> None:
        """
        Plots the clusters.
        """
        try:
            _ = self.n_clusters
        except AttributeError as e:
            print(f"{e}. Perhaps, you didn't fit the model yet.")
            return None

        plt.close("all")
        plt.figure(1)
        plt.clf()
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))

        for k, col in zip(range(self.n_clusters), colors):
            class_members = self.labels == k
            plt.plot(
                self.X[class_members, 0],
                self.X[class_members, 1],
                ".",
                markerfacecolor=col,
            )
            plt.plot(
                self.cluster_centers[k, 0],
                self.cluster_centers[k, 1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Number of clusters: {self.n_clusters}")
        plt.show()

    def get_cluster_details(self) -> dict:
        """
        Returns the details of the clusters.

        Returns:
            dict: Dictionary containing cluster details.
        """
        return {
            "n_clusters": self.n_clusters,
            "cluster_centers": self.cluster_centers,
            "labels": self.labels,
        }

    def grid_search(
        self,
        min_clusters: int = 1,
        max_clusters: int = 10,
        step: int = 1,
        cluster_list: list = None,
    ) -> None:
        """
        Calculates the distortions, silhouette score and their derivatives
        for a range of cluster numbers or a specified list of cluster numbers.

        Args:
            min_clusters (int): Minimum number of clusters to consider.
            max_clusters (int): Maximum number of clusters to consider.
            step (int): Step size for iterating over the number of clusters.
            cluster_list (list): List of cluster numbers to try out.
            If provided, min_clusters, max_clusters, and step will be ignored.
        """
        if cluster_list:
            num_clusters = cluster_list
        else:
            num_clusters = range(min_clusters, max_clusters + 1, step)

        distortions = []
        silhouette_scores = []

        self.opt_num_clusters = {}

        for i in tqdm(num_clusters):
            model = KMeans(n_clusters=i, random_state=self.seed, n_init=10, init="k-means++")
            labels = model.fit_predict(self.X)
            distortion = model.inertia_
            sil_score = metrics.silhouette_score(self.X, labels)

            distortions.append(distortion)
            silhouette_scores.append(sil_score)

        distortion_derivatives = get_derivative(num_clusters, distortions)
        silhouette_derivatives = get_derivative(num_clusters, silhouette_scores)

        self.distortions_df = pd.DataFrame(
            {
                "Number of Clusters": num_clusters,
                "Distortion": distortions,
                "Distortion Derivative": distortion_derivatives,
                "Silhouette": silhouette_scores,
                "Silhouette Derivative": silhouette_derivatives,
            }
        )

        self.opt_num_clusters["Distortion"] = num_clusters[
            np.where(np.diff(distortion_derivatives) >= 0)[0][0]
        ]
        self.opt_num_clusters["Silhouette"] = num_clusters[
            np.where(np.diff(silhouette_derivatives) >= 0)[0][0]
        ]

    def plot_elbow_graph(
        self, variable: Literal["Distortion", "Silhouette"] = "Distortion"
    ) -> None:
        """
        Plots the elbow graph with optimal number of clusters.

        Args:
            variable (str): Variable to depict.
        """
        try:
            _ = self.distortions_df
        except AttributeError as e:
            print(
                f"{e}. Perhaps, you didn't do the grid search yet, call grid_search() method first."
            )
            return None

        # Main variable
        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel(variable, color=color)
        ax1.plot(
            self.distortions_df["Number of Clusters"],
            self.distortions_df[f"{variable}"],
            marker="o",
            color=color,
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Derivative", color=color)
        ax2.plot(
            self.distortions_df["Number of Clusters"],
            self.distortions_df[f"{variable} Derivative"],
            marker="o",
            color=color,
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Optimal N of clusters
        if self.opt_num_clusters:
            opt_point = self.distortions_df[
                self.distortions_df["Number of Clusters"] == self.opt_num_clusters[variable]
            ]
            ax1.plot(
                opt_point["Number of Clusters"],
                opt_point[variable],
                marker="o",
                markersize=15,
                markeredgecolor="black",
                markerfacecolor="none",
                linestyle="None",
            )

            ax2.plot(
                opt_point["Number of Clusters"],
                opt_point[f"{variable} Derivative"],
                marker="o",
                markersize=15,
                markeredgecolor="black",
                markerfacecolor="none",
                linestyle="None",
            )

        fig.tight_layout()
        plt.title(
            f"Number of clusters vs {variable} (optimal N = {self.opt_num_clusters[variable]})"
        )
        plt.show()


class OPTICSClustering:
    def __init__(self, X: np.ndarray):
        """
        Initialize OPTICSClustering object.

        Parameters:
        - X (np.ndarray): Input data array of shape (n_samples, n_features).

        # Example of usage:
            # Initialize and fit OPTICSClustering
            optics = OPTICSClustering(X)
            optics.fit()

            # Plot clusters
            optics.plot_clusters(X)
        """
        self.X = X
        self.optics_model = None

    def fit(
        self,
        min_samples: int = 5,
        max_eps: float = np.inf,
        metric: str = "minkowski",
        p: int = 2,
        cluster_method: str = "xi",
    ):
        """
        Fit the OPTICS clustering model to the data.

        Parameters:
        - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        - max_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - metric (str): The metric to use when calculating distance between instances in a feature array.
        - p (int): The power of the Minkowski metric to be used to calculate distance between points.
        - cluster_method (str): The method used to extract clusters. Possible values: 'xi' or 'dbscan'.

        Returns:
        - None
        """
        self.optics_model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            p=p,
            cluster_method=cluster_method,
        )
        self.optics_model.fit(self.X)
        self.labels = self.optics_model.labels_

    def predict(self) -> np.ndarray:
        """
        Predict cluster labels for the fitted data.

        Returns:
        - np.ndarray: Array of cluster labels.
        """
        if self.optics_model is None:
            raise Exception("Model not fitted yet. Call fit() first.")
        return self.optics_model.labels_

    def plot_clusters(self):
        """
        Plot clusters based on fitted data.

        Parameters:
        - data (np.ndarray): Data array of shape (n_samples, n_features).

        Returns:
        - None
        """
        if self.optics_model is None:
            raise Exception("Model not fitted yet. Call fit() first.")
        unique_labels = np.unique(self.optics_model.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = "black"  # Outliers
            cluster_member_mask = self.optics_model.labels_ == label
            cluster_points = self.X[cluster_member_mask]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f"Cluster {label}"
            )

        plt.title("OPTICS Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
