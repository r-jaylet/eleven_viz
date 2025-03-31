from typing import Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.graph_objs import Figure

# Consistent color palette matching other viz files
COLORS = {
    "primary": "#1A237E",  # Dark blue
    "secondary": "#004D40",  # Dark green
    "accent1": "#311B92",  # Deep purple
    "accent2": "#01579B",  # Dark cyan
    "accent3": "#33691E",  # Dark lime
    "warning": "#FFC107",  # Amber for warnings
    "danger": "#C62828",  # Dark red for danger
    "success": "#2E7D32",  # Dark green for success
    "text": "#212121",  # Almost black text
}
QUALITATIVE_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent1"],
    COLORS["accent2"],
    COLORS["accent3"],
]
TEMPLATE = "plotly_white"
COMMON_MARGINS = dict(l=50, r=50, t=80, b=50)


def compute_load(
    df_gps: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    top_x_percent: float = 5.0,
    window_size: int = 7,
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate player load from GPS data and display analysis plots using Plotly.
    Returns the threshold for top x% highest loads and a DataFrame with daily average load.
    """
    df_gps = df_gps.dropna(
        subset=["distance", "distance_over_21", "accel_decel_over_2_5"]
    ).copy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-(x - x.mean()) / x.std()))

    df_gps["load"] = (
        (sigmoid(df_gps["distance"]) ** alpha)
        * (sigmoid(df_gps["distance_over_21"]) ** gamma)
        * (sigmoid(df_gps["accel_decel_over_2_5"]) ** beta)
    ) ** (1 / (alpha + beta + gamma))

    df_gps["date"] = pd.to_datetime(df_gps["date"], dayfirst=True)
    df_load_by_date = (
        df_gps.groupby(df_gps["date"].dt.strftime("%d/%m/%Y"))["load"]
        .mean()
        .reset_index()
    )
    df_load_by_date.rename(columns={"date": "date_str"}, inplace=True)
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )
    df_load_by_date = df_load_by_date.sort_values("date")
    df_load_by_date["window_mean"] = (
        df_load_by_date["load"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    threshold_value = np.percentile(df_gps["load"], 100 - top_x_percent)

    return df_gps, threshold_value, df_load_by_date


def plot_load(df_gps: pd.DataFrame):
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Load vs Distance",
            "Load vs Distance Over 21 km/h",
            "Load vs Accel/Decel > 2.5 m/s²",
            "Empirical CDF of Load",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_gps["distance"],
            y=df_gps["load"],
            mode="markers",
            marker=dict(
                color=COLORS["primary"],
                size=8,
                opacity=0.7,
                line=dict(width=1, color="#FFFFFF"),
            ),
            name="Distance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_gps["distance_over_21"],
            y=df_gps["load"],
            mode="markers",
            marker=dict(
                color=COLORS["secondary"],
                size=8,
                opacity=0.7,
                line=dict(width=1, color="#FFFFFF"),
            ),
            name="Distance > 21 km/h",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df_gps["accel_decel_over_2_5"],
            y=df_gps["load"],
            mode="markers",
            marker=dict(
                color=COLORS["accent1"],
                size=8,
                opacity=0.7,
                line=dict(width=1, color="#FFFFFF"),
            ),
            name="Accel/Decel > 2.5",
        ),
        row=2,
        col=1,
    )

    sorted_load = np.sort(df_gps["load"])
    cdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
    fig.add_trace(
        go.Scatter(
            x=sorted_load,
            y=cdf,
            mode="lines+markers",
            marker=dict(color=COLORS["accent2"], size=6, opacity=0.7),
            line=dict(color=COLORS["accent2"], width=2),
            name="CDF",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title={
            "text": "Load Analysis",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        height=650,
        width=900,
        template=TEMPLATE,
        margin=COMMON_MARGINS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )

    fig.update_xaxes(title_text="Distance (meters)", row=1, col=1)
    fig.update_xaxes(title_text="Distance > 21 km/h (meters)", row=1, col=2)
    fig.update_xaxes(title_text="Accel/Decel > 2.5 m/s²", row=2, col=1)
    fig.update_xaxes(title_text="Load Value", row=2, col=2)

    fig.update_yaxes(title_text="Load", row=1, col=1)
    fig.update_yaxes(title_text="Load", row=1, col=2)
    fig.update_yaxes(title_text="Load", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)

    return fig


def plot_weekly_danger(
    merged_df: pd.DataFrame, danger_rate: float = 10
) -> Figure:
    """
    Visualize weekly training overload distribution with color-coded danger levels.

    Args:
        merged_df: DataFrame with columns 'date' (datetime) and 'danger_score'
        danger_rate: Percentage to determine dangerous thresholds (default 10%)

    Returns:
        Plotly figure object with weekly danger visualization
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
            return COLORS["danger"]  # Dark red for high risk
        elif dangerous_days > 0:
            return COLORS["warning"]  # Amber for warning
        else:
            return COLORS["success"]  # Dark green for normal

    weekly["color"] = weekly["dangerous_days"].apply(get_color)

    # Create interactive visualization
    fig = go.Figure()

    # Main bar chart
    fig.add_trace(
        go.Bar(
            x=weekly["week"],
            y=[1] * len(weekly),
            marker_color=weekly["color"],
            text=weekly["dangerous_days"],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12, family="Arial"),
            showlegend=False,
            hovertemplate="Week: %{x}<br>Dangerous days: %{text}<extra></extra>",
            marker_line=dict(width=1, color="#FFFFFF"),
        )
    )

    # Add legend items
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=COLORS["success"], size=12),
            name="Normal (0 days)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=COLORS["warning"], size=12),
            name="Warning (1-2 days)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=COLORS["danger"], size=12),
            name="High Risk (3+ days)",
        )
    )

    fig.update_layout(
        title={
            "text": "Weekly Training Overload Distribution",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        xaxis_title={"text": "Week", "font": {"size": 14}},
        yaxis=dict(tickvals=[], showticklabels=False),
        template=TEMPLATE,
        legend_title={"text": "Risk Levels", "font": {"size": 14}},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        height=500,
        margin=COMMON_MARGINS,
    )

    return fig


def plot_load_over_time(
    df_load_by_date: pd.DataFrame,
    rolling_window: int = 1,
) -> Figure:
    """
    Display load evolution over time with interactive visualization.

    Args:
        df_load_by_date: DataFrame with columns 'date_str' (dd/mm/yyyy) and 'load'
        rolling_window: Number of days for rolling average (default 1 = no rolling)

    Returns:
        Plotly figure with load evolution chart
    """
    df_viz = df_load_by_date.copy()
    df_viz["date"] = pd.to_datetime(df_viz["date_str"], format="%d/%m/%Y")

    # Apply rolling average if specified
    if rolling_window > 1:
        df_viz["load"] = (
            df_viz["load"].rolling(window=rolling_window, min_periods=1).mean()
        )

    # Create interactive visualization
    fig = go.Figure()

    # Add main data trace
    fig.add_trace(
        go.Scatter(
            x=df_viz["date"],
            y=df_viz["load"],
            mode="lines+markers",
            name="Average Load",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(
                color=COLORS["primary"],
                size=8,
                line=dict(color="#FFFFFF", width=1),
            ),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Load: %{y:.3f}<extra></extra>",
        )
    )

    # Add mean line
    mean_load = df_viz["load"].mean()
    fig.add_shape(
        type="line",
        x0=df_viz["date"].min(),
        y0=mean_load,
        x1=df_viz["date"].max(),
        y1=mean_load,
        line=dict(color=COLORS["accent1"], width=2, dash="dash"),
    )

    # Add annotation for mean
    fig.add_annotation(
        x=df_viz["date"].max(),
        y=mean_load,
        text=f"Avg: {mean_load:.3f}",
        showarrow=False,
        font=dict(size=12, color=COLORS["accent1"]),
        xanchor="right",
        yanchor="bottom",
        xshift=10,
    )

    fig.update_layout(
        title={
            "text": "Load Evolution Over Time",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Average Load", "font": {"size": 14}},
        xaxis=dict(
            tickformat="%d/%m/%Y",
            tickangle=-45,
            gridcolor="rgba(238, 238, 238, 0.5)",
        ),
        yaxis=dict(gridcolor="rgba(238, 238, 238, 0.5)"),
        template=TEMPLATE,
        showlegend=True,
        autosize=True,
        hovermode="x unified",
        height=550,
        margin=COMMON_MARGINS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
    )

    return fig


def compute_load_vs_recovery(
    df_recovery: pd.DataFrame,
    df_gps: pd.DataFrame,
    alpha: float = 2,
    beta: float = 1,
    gamma: float = 3,
    top_x_percent: float = 5,
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Plot relationship between recovery score and load with danger score calculation.

    Args:
        df_recovery: DataFrame with recovery data (columns: 'sessionDate', 'metric', 'value')
        df_gps: DataFrame with GPS data (columns: 'date', 'distance', etc.)
        alpha: Exponent for distance in load calculation
        beta: Exponent for accelerations/decelerations
        gamma: Exponent for distance over 21 km/h
        top_x_percent: Percentage for determining threshold
        window_size: Window size for moving average

    Returns:
        DataFrame
    """
    df_recovery_filtered = df_recovery[
        df_recovery["metric"] == "emboss_baseline_score"
    ].copy()
    df_recovery_filtered = df_recovery_filtered.sort_values("sessionDate")
    df_recovery_filtered["rolling_mean"] = (
        df_recovery_filtered["value"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # Calculate load data
    df_gps, threshold, df_load_by_date = compute_load(
        df_gps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_x_percent=top_x_percent,
        window_size=window_size,
    )

    # Merge recovery and load data
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )

    merged_df = pd.merge(
        df_recovery_filtered,
        df_load_by_date,
        left_on="sessionDate",
        right_on="date",
        how="inner",
    )

    # Calculate danger score
    max_load = merged_df["window_mean"].max()
    max_recovery_score = merged_df["rolling_mean"].max()
    merged_df["danger_score"] = (merged_df["window_mean"] / max_load) * (
        1 - merged_df["rolling_mean"] / max_recovery_score
    )
    return merged_df


def plot_load_vs_recovery(merged_df: pd.DataFrame) -> Figure:
    # Create interactive visualization with Plotly
    fig = go.Figure()

    # Add load trace
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["window_mean"],
            mode="lines",
            name="Load",
            line=dict(color=COLORS["primary"], width=3),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Load: %{y:.3f}<extra></extra>",
        )
    )

    # Add recovery score trace
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["rolling_mean"],
            mode="lines",
            name="Recovery Score",
            line=dict(color=COLORS["secondary"], width=3),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Recovery: %{y:.3f}<extra></extra>",
        )
    )

    # Add danger score trace
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["danger_score"],
            mode="lines",
            name="Risk Score",
            line=dict(color=COLORS["danger"], width=3),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Risk: %{y:.3f}<extra></extra>",
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title={
            "text": "Load, Recovery and Risk Score Over Time",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Score", "font": {"size": 14}},
        xaxis=dict(
            tickformat="%d/%m/%Y",
            tickangle=-45,
            gridcolor="rgba(238, 238, 238, 0.5)",
        ),
        yaxis=dict(gridcolor="rgba(238, 238, 238, 0.5)"),
        template=TEMPLATE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
        ),
        hovermode="x unified",
        height=600,
        margin=COMMON_MARGINS,
    )

    return fig
