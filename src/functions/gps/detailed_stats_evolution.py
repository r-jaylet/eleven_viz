import pandas as pd
import plotly.graph_objects as go


def detailed_stats_evolution(df_filtered: pd.DataFrame, stat: str) -> None:
    """
    Generates and displays a detailed statistical evolution graph for a given performance metric.

    :param df_filtered: DataFrame containing the filtered data with 'season' and 'date' columns.
    :param stat: The name of the column representing the performance metric to visualize.
    """
    # Compute mean and standard deviation for each season
    season_stats = (
        df_filtered.groupby("season")[stat].agg(["mean", "std"]).reset_index()
    )

    # Create a graph for each season
    for season, df_season in df_filtered.groupby("season"):
        mean_value = season_stats.loc[
            season_stats["season"] == season, "mean"
        ].values[0]
        std_value = season_stats.loc[
            season_stats["season"] == season, "std"
        ].values[0]

        lower_limit = mean_value - 0.5 * std_value
        upper_limit = mean_value + 0.5 * std_value

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_season["date"], y=df_season[stat], mode="lines", name=stat
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df_season["date"].min(), df_season["date"].max()],
                y=[mean_value, mean_value],
                mode="lines",
                name="Mean",
                line=dict(color="black", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df_season["date"].min(), df_season["date"].max()],
                y=[lower_limit, lower_limit],
                mode="lines",
                name="Mean - 0.5*Std",
                line=dict(color="blue", dash="dash", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df_season["date"].min(), df_season["date"].max()],
                y=[upper_limit, upper_limit],
                mode="lines",
                name="Mean + 0.5*Std",
                line=dict(color="blue", dash="dash", width=1),
            )
        )

        # Add background color zones
        for i in range(len(df_season)):
            if df_season[stat].iloc[i] < lower_limit:
                color = "rgba(255, 0, 0, 0.2)"  # Red (low performance)
            elif df_season[stat].iloc[i] < upper_limit:
                color = "rgba(255, 255, 0, 0.2)"  # Yellow (medium performance)
            else:
                color = "rgba(0, 128, 0, 0.2)"  # Green (high performance)

            x0 = (
                df_season["date"].iloc[i]
                if i == 0
                else df_season["date"].iloc[i]
                - (df_season["date"].iloc[i] - df_season["date"].iloc[i - 1])
                / 2
            )
            x1 = (
                df_season["date"].iloc[i]
                if i == len(df_season) - 1
                else df_season["date"].iloc[i]
                + (df_season["date"].iloc[i + 1] - df_season["date"].iloc[i])
                / 2
            )

            fig.add_shape(
                type="rect",
                x0=x0,
                y0=0,
                x1=x1,
                y1=df_season[stat].max(),
                line=dict(width=0),
                fillcolor=color,
            )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="rgba(255, 0, 0, 0.2)", size=10),
                name="Low (Red)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="rgba(255, 255, 0, 0.2)", size=10),
                name="Medium (Yellow)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="rgba(0, 128, 0, 0.2)", size=10),
                name="High (Green)",
            )
        )

        fig.update_layout(
            title=f"{stat} Over Time with Mean and Standard Deviation - {season}"
        )
        fig.show()
