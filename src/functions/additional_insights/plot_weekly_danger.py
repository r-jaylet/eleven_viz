import pandas as pd
import plotly.graph_objects as go


def plot_weekly_danger(
    merged_df: pd.DataFrame, danger_rate: float = 10, show_plot: bool = True
) -> None:
    """
    Given merged_df containing a 'date' column (datetime) and a 'danger_score' column,
    and a danger_rate (percentage), this function computes a danger threshold such that
    the top danger_rate% of days (by danger_score) are considered dangerous.

    Then, for each week, if at least 3 days are dangerous, the week is marked as dangerous.

    The function then plots a bar chart (using Plotly) with one bar per week:
      - The bar is colored red if the week is dangerous (>= 3 dangerous days),
      - The bar is colored yellow if there are 1 or 2 dangerous days,
      - The bar is colored green if there are no dangerous days.

    All bars have the same height (fixed at 1) for visualization purposes.

    Args:
      merged_df (pd.DataFrame): DataFrame containing at least the columns 'date' (datetime) and 'danger_score'.
      danger_rate (float): Percentage for determining the top x% threshold (default 10).
      show_plot (bool): Whether to display the plot (default True).

    Returns:
      None: Displays the Plotly bar chart.
    """
    # Calculate the danger threshold (top danger_score %)
    threshold = merged_df["danger_score"].quantile(1 - danger_rate / 100)

    # Group by week and count dangerous days
    merged_df["week"] = (
        merged_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    )
    weekly = (
        merged_df.groupby("week")
        .agg(dangerous_days=("danger_score", lambda x: (x >= threshold).sum()))
        .reset_index()
    )

    def get_color(dangerous_days):
        if dangerous_days >= 3:
            return "red"
        elif dangerous_days > 0:
            return "yellow"
        else:
            return "green"

    weekly["color"] = weekly["dangerous_days"].apply(get_color)

    # plot section

    fig = go.Figure(
        data=[
            go.Bar(
                x=weekly["week"],
                y=[1] * len(weekly),
                marker_color=weekly["color"],
                text=weekly["dangerous_days"],
                textposition="auto",
                showlegend=False,
            )
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Normal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="yellow", size=10),
            name="Overload",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="red", size=10),
            name="High Overload",
        )
    )

    fig.update_layout(
        title="Weekly Training Overload Distribution ",
        xaxis_title="Week",
        yaxis=dict(tickvals=[], showticklabels=False),
        template="plotly_white",
        legend_title="Danger Levels ",
    )

    if show_plot:
        fig.show()
