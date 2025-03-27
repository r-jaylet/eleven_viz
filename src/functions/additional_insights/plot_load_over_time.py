import pandas as pd
import plotly.graph_objects as go


def plot_load_over_time(
    df_load_by_date: pd.DataFrame,
    start_date: str = None,  # Date range start
    end_date: str = None,  # Date range end
    rolling_window: int = 1,  # Rolling window (1 for no rolling)
) -> None:
    """
    Displays the load evolution over time with Plotly, with optional date range and
    rolling average.

    Args:
        df_load_by_date (pd.DataFrame): DataFrame containing two columns:
            - 'date_str' (str): Date in 'dd/mm/yyyy' format.
            - 'load' (float): Average load of the player on this date.
        start_date (str, optional): Start date of the range in 'dd/mm/yyyy'. No limit by default.
        end_date (str, optional): End date of the range in 'dd/mm/yyyy'. No limit by default.
        rolling_window (int, optional): Number of days for rolling average. Default is 1 (no rolling).

    Returns:
        None: The function displays a plot but does not return anything.
    """

    df_load_by_date = df_load_by_date.copy()
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )

    # Filter data based on the specified date range
    if start_date:
        df_load_by_date = df_load_by_date[
            df_load_by_date["date"]
            >= pd.to_datetime(start_date, format="%d/%m/%Y")
        ]
    if end_date:
        df_load_by_date = df_load_by_date[
            df_load_by_date["date"]
            <= pd.to_datetime(end_date, format="%d/%m/%Y")
        ]

    # Apply rolling average if necessary
    if rolling_window > 1:
        df_load_by_date["load"] = (
            df_load_by_date["load"]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
        )

    # plot section
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_load_by_date["date"],
            y=df_load_by_date["load"],
            mode="lines+markers",
            name="Average Load",
            line=dict(color="blue"),
            marker=dict(color="blue", size=6),
        )
    )

    fig.update_layout(
        title="Load Evolution Over Time",
        xaxis_title="Date",
        yaxis_title="Average Load",
        xaxis=dict(tickformat="%d/%m/%Y"),
        template="plotly_white",
        showlegend=True,
        autosize=True,
    )

    fig.show()
