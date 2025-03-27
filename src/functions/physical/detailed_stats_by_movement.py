import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def detailed_stats_by_movement(
    df_filtered: pd.DataFrame,
    start_date: str = "01/01/2023",
    end_date: str = "31/12/2050",
) -> None:
    """
    Plots detailed performance statistics over time based on different movement types, expressions, and quality categories.
    Each plot represents a line chart with the benchmark percentage (benchmarkPct) over time.

    Args:
    - df_filtered (pd.DataFrame): DataFrame containing performance data with 'testDate', 'movement', 'expression', and 'quality' columns.
    - start_date (str, optional): Start date in the format 'DD/MM/YYYY' to filter the data. Defaults to '01/01/2023'.
    - end_date (str, optional): End date in the format 'DD/MM/YYYY' to filter the data. Defaults to '31/12/2050'.

    Returns:
    - None: Displays line charts for each movement type with benchmarkPct over time for each expression and quality category.
    """

    stat = "benchmarkPct"

    df_filtered["testDate"] = pd.to_datetime(
        df_filtered["testDate"], dayfirst=True
    )

    # Filter based on the provided date range
    if start_date:
        start_date = pd.to_datetime(start_date, dayfirst=True)
        df_filtered = df_filtered[df_filtered["testDate"] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date, dayfirst=True)
        df_filtered = df_filtered[df_filtered["testDate"] <= end_date]

    df_filtered = df_filtered.sort_values(by="testDate")

    # plot section
    color_palette = px.colors.qualitative.Set1

    # For each movement type
    for movement in df_filtered["movement"].unique():
        df_movement = df_filtered[df_filtered["movement"] == movement]

        fig = go.Figure()

        color_idx = 0

        for expression in sorted(df_movement["expression"].unique()):
            df_expr = df_movement[df_movement["expression"] == expression]

            for quality in df_expr["quality"].unique():
                df_curve = df_expr[df_expr["quality"] == quality]

                fig.add_trace(
                    go.Scatter(
                        x=df_curve["testDate"],
                        y=df_curve[stat],
                        mode="lines",
                        name=f"{expression} - Quality: {quality}",
                        line=dict(
                            color=color_palette[color_idx % len(color_palette)]
                        ),
                    )
                )
                color_idx += 1

        fig.update_layout(
            title=f"{stat} over time - Movement: {movement}",
            xaxis_title="Date",
            yaxis_title=stat,
            legend_title="Expression - Quality",
        )

        fig.show()
