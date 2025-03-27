from typing import List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_performance(
    df_trainings: pd.DataFrame,
    df_matches: pd.DataFrame,
    features: List[str],
    n_clusters: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies K-Means clustering to both training and match data to classify performance levels.

    :param df_trainings: DataFrame containing the training data.
    :param df_matches: DataFrame containing the match data.
    :param features: List of feature column names to be used for clustering.
    :param n_clusters: Number of clusters to create (default: 3).
    :return: A tuple of DataFrames (df_trainings_copy, df_matches_copy) with assigned cluster labels.
    """

    df_training_copy = df_trainings.copy()
    df_matches_copy = df_matches.copy()

    scaler = StandardScaler()
    df_training_scaled = scaler.fit_transform(df_training_copy[features])

    # Apply K-Means clustering on training data
    kmeans_trainings = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    )
    df_training_copy["cluster"] = kmeans_trainings.fit_predict(
        df_training_scaled
    )

    # Sort clusters and assign them to training sessions
    cluster_distances_trainings = df_training_copy.groupby("cluster")[
        "distance"
    ].mean()
    sorted_clusters_trainings = cluster_distances_trainings.sort_values(
        ascending=False
    ).index
    cluster_labels_trainings = {
        sorted_clusters_trainings[0]: "Better performances",
        sorted_clusters_trainings[1]: "Usual performances",
        sorted_clusters_trainings[2]: "Lower performances",
    }
    df_training_copy["cluster_label"] = df_training_copy["cluster"].map(
        cluster_labels_trainings
    )

    df_matches_scaled = scaler.transform(df_matches_copy[features])

    kmeans_matches = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_matches_copy["cluster"] = kmeans_matches.fit_predict(df_matches_scaled)

    cluster_distances_matches = df_matches_copy.groupby("cluster")[
        "distance"
    ].mean()
    sorted_clusters_matches = cluster_distances_matches.sort_values(
        ascending=False
    ).index
    cluster_labels_matches = {
        sorted_clusters_matches[0]: "Better performances",
        sorted_clusters_matches[1]: "Usual performances",
        sorted_clusters_matches[2]: "Lower performances",
    }
    df_matches_copy["cluster_label"] = df_matches_copy["cluster"].map(
        cluster_labels_matches
    )

    return df_training_copy, df_matches_copy
