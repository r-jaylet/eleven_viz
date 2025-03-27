import pandas as pd
import plotly.express as px

from .get_duration_matchs import get_duration_matchs


def plot_distance_distribution_by_duration(df_filtered: pd.DataFrame) -> None:
    """
    Plots a histogram of the distance distribution based on the match duration.

    Args:
    - df_filtered (pd.DataFrame): DataFrame containing match data, must include 'distance' and a match duration feature.

    Returns:
    - None: Displays a histogram plot.
    """
    df_short, df_medium, df_long, _ = get_duration_matchs(df_filtered.copy())

    df_short["duration_group"] = "<30min"
    df_medium["duration_group"] = "30-60min"
    df_long["duration_group"] = ">60min"
    df_combined = pd.concat([df_short, df_medium, df_long], ignore_index=True)

    color_map = {
        "<30min": "rgba(255, 0, 0, 0.8)",  # Red for <30min
        "30-60min": "rgba(0, 255, 0, 0.6)",  # Green for 30-60min
        ">60min": "rgba(0, 0, 255, 0.2)",  # Blue with opacity 0.4 for >60min
    }

    fig = px.histogram(
        df_combined,
        x="distance",
        color="duration_group",
        barmode="overlay",
        title="Distance Distribution by Match Duration",
        color_discrete_map=color_map,
    )

    fig.show()
