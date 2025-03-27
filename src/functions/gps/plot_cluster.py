import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cluster(df: pd.DataFrame, x: str, y: str) -> None:
    """
    Visualizes the clusters of matches based on two given features (x and y axes).

    This function creates a scatter plot of matches, colored by their assigned cluster label.
    The scatter plot helps to visualize the clustering of matches based on their distance and peak speed.

    :param df: A pandas DataFrame containing the match data, including the 'cluster_label' column.
    :param x: The feature to plot on the x-axis (e.g., 'distance').
    :param y: The feature to plot on the y-axis (e.g., 'peak_speed').
    :return: None. The function displays a seaborn scatter plot.
    """

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df[x], y=df[y], hue=df["cluster_label"], palette="Set1", s=100
    )
    plt.title("Clustering of Matches based on Distance and Peak Speed")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title="Cluster")
    plt.show()
