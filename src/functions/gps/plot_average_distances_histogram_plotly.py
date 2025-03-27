import pandas as pd
import plotly.graph_objects as go

from .average_distances_by_recovery import average_distances_by_recovery
from .list_matches_with_recovery import list_matches_with_recovery


def plot_average_distances_histogram_plotly(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the average distances traveled and the number of matches by recovery days.

    This function calculates the average distance for each recovery day and the corresponding number of matches.
    It then plots a grouped bar chart with dual y-axes: one for the average distance and another for the number of matches.

    :param df: A pandas DataFrame containing the match data with 'date', 'opposition_code', 'distance', and 'md_plus_code'.
    :return: None. The function displays a Plotly figure.
    """

    matchs_liste = list_matches_with_recovery(df)
    average_by_recovery = average_distances_by_recovery(matchs_liste)

    count_by_recovery = {}
    for match in matchs_liste:
        _, _, _, md_plus_code = match
        if md_plus_code is not None:
            count_by_recovery[md_plus_code] = (
                count_by_recovery.get(md_plus_code, 0) + 1
            )

    df_plot = pd.DataFrame(
        {
            "Recovery Days": list(average_by_recovery.keys()),
            "Average Distance": list(average_by_recovery.values()),
            "Number of Matches": [
                count_by_recovery[k] for k in average_by_recovery.keys()
            ],
        }
    )

    max_matches = max(df_plot["Number of Matches"])
    y2_max = max_matches * 1.7

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_plot["Recovery Days"],
            y=df_plot["Average Distance"],
            name="Average Distance (km)",
            text=df_plot["Average Distance"].round(1),
            textposition="outside",
            marker_color="royalblue",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_plot["Recovery Days"],
            y=df_plot["Number of Matches"],
            name="Number of Matches",
            text=df_plot["Number of Matches"],
            textposition="outside",
            marker_color="orange",
            opacity=0.7,
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Average Distance Traveled and Number of Matches by Recovery Days",
        xaxis_title="Number of Recovery Days",
        yaxis=dict(
            title="Average Distance Traveled (km)", side="left", showgrid=False
        ),
        yaxis2=dict(
            title="Number of Matches",
            side="right",
            overlaying="y",
            showgrid=False,
            range=[0, y2_max],
        ),
        barmode="group",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
        ),
    )

    fig.show()
