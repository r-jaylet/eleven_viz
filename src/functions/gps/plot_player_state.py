import pandas as pd
import plotly.express as px


def plot_player_state(
    df: pd.DataFrame,
    season: str = None,
    start_date: str = None,
    end_date: str = None,
) -> None:
    """
    Displays a stacked bar chart representing the player's state
    over a given period or for a specific season.

    Args:
    - df (pd.DataFrame): DataFrame containing the player data.
    - season (str, optional): Season name to filter by (e.g., '2022-2023').
    - start_date (str, optional): Start date in 'YYYY-MM-DD' format.
    - end_date (str, optional): End date in 'YYYY-MM-DD' format.

    Returns:
    - None: Displays the stacked bar chart.
    """

    # Filter by season if specified
    if season:
        df_filtered = df[df["season"] == season]

    elif start_date and end_date:
        df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    else:
        print("Please specify a valid season or date range.")
        return

    if df_filtered.empty:
        print("No data available for the specified period.")
        return

    fig = px.bar(
        df_filtered,
        x="date",
        color="cluster_label",
        title=f"Player State Distribution {f'- {season}' if season else ''}",
        labels={"cluster_label": "Player State", "date": "Date"},
        category_orders={
            "cluster_label": [
                "Better performances",
                "Usual performances",
                "Lower performances",
            ]
        },
        barmode="stack",
    )

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Number of Events", showlegend=True
    )

    fig.show()
