from typing import Optional

import pandas as pd
import plotly.graph_objects as go


def plot_global_recovery_score(
    df: pd.DataFrame,
    start_date: str = "01/01/2023",
    end_date: str = "31/12/2030",
    show_plot: bool = True,
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Generates a time series plot showing the evolution of the global recovery score (emboss_baseline_score)
    with a configurable moving average window and filtering based on the provided date range. The function also returns
    the filtered DataFrame with the rolling mean. The plot is displayed if `show_plot` is True.

    Args:
    - df (pd.DataFrame): DataFrame containing recovery data with columns 'sessionDate', 'metric', and 'value'.
    - start_date (str, optional): Start date for filtering the data, in the format 'dd/mm/yyyy'. Default is "01/01/2023".
    - end_date (str, optional): End date for filtering the data, in the format 'dd/mm/yyyy'. Default is "31/12/2030".
    - show_plot (bool, optional): Whether to display the plot. Default is True.
    - window_size (int, optional): Size of the rolling window for the moving average. Default is 7 days.

    Returns:
    - pd.DataFrame: Filtered DataFrame with the rolling mean added.
    """
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    # Filter the data to keep only rows where 'metric' is 'emboss_baseline_score'
    df_total = df[df["metric"] == "emboss_baseline_score"].copy()

    df_total = df_total[
        (
            df_total["sessionDate"]
            >= pd.to_datetime(start_date, format="%d/%m/%Y")
        )
        & (
            df_total["sessionDate"]
            <= pd.to_datetime(end_date, format="%d/%m/%Y")
        )
    ]
    df_total = df_total.sort_values("sessionDate")

    # Calculate the rolling mean with the given window size
    df_total["rolling_mean"] = (
        df_total["value"].rolling(window=window_size, min_periods=1).mean()
    )

    # plot section
    if show_plot:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_total["sessionDate"],
                y=df_total["value"],
                mode="lines",
                name="Raw Score",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_total["sessionDate"],
                y=df_total["rolling_mean"],
                mode="lines",
                name=f"{window_size}-day MA",
                line=dict(dash="dot"),
            )
        )

        fig.update_layout(
            title="Evolution of Global Recovery Score",
            xaxis_title="Date",
            yaxis_title="Score",
            template="plotly_white",
        )

        fig.show()

    return df_total
