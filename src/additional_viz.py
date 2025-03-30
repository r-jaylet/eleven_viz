from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.graph_objs import Figure


def compute_load_and_plot(
    df_gps: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    top_x_percent: float = 5.0,
    show_plot: bool = True,
    window_size: int = 7,
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate player load from GPS data using sigmoid transformation and display analysis plots.
    Returns the threshold for top x% highest loads and a DataFrame with daily average load.

    Args:
        df_gps: DataFrame with columns 'date', 'distance', 'distance_over_21', 'accel_decel_over_2_5'
        alpha: Exponent for distance in load calculation (default 1.0)
        beta: Exponent for accelerations/decelerations (default 1.0)
        gamma: Exponent for distance over 21 km/h (default 1.0)
        top_x_percent: Percentage for determining threshold of highest loads (default 5.0)
        show_plot: Whether to display analysis plots (default True)
        window_size: Window size for moving average calculation (default 7 days)

    Returns:
        Threshold for top x% highest loads
        DataFrame with daily average load (date format 'dd/mm/yyyy') and moving average
    """
    # Remove rows with missing values in relevant variables
    df_gps_cleaned = df_gps.dropna(
        subset=["distance", "distance_over_21", "accel_decel_over_2_5"]
    ).copy()

    # Apply sigmoid transformation
    def sigmoid(x: pd.Series) -> pd.Series:
        return 1 / (1 + np.exp(-(x - x.mean()) / x.std()))

    df_gps_cleaned["distance_sigmoid"] = sigmoid(df_gps_cleaned["distance"])
    df_gps_cleaned["distance_over_21_sigmoid"] = sigmoid(
        df_gps_cleaned["distance_over_21"]
    )
    df_gps_cleaned["accel_decel_over_2_5_sigmoid"] = sigmoid(
        df_gps_cleaned["accel_decel_over_2_5"]
    )

    # Calculate load
    df_gps_cleaned["load"] = (
        (df_gps_cleaned["distance_sigmoid"] ** alpha)
        * (df_gps_cleaned["distance_over_21_sigmoid"] ** gamma)
        * (df_gps_cleaned["accel_decel_over_2_5_sigmoid"] ** beta)
    ) ** (1 / (alpha + beta + gamma))

    # Display analysis plots if requested
    if show_plot:
        fig = plt.figure(figsize=(24, 6))

        # Create subplots with proper spacing
        gs = fig.add_gridspec(1, 4, wspace=0.25)
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

        # Load vs Distance
        axs[0].scatter(
            df_gps_cleaned["distance"],
            df_gps_cleaned["load"],
            color="#4285F4",
            alpha=0.7,
            s=40,
        )
        axs[0].set_title("Load vs Distance", fontsize=14, fontweight="bold")
        axs[0].set_xlabel("Distance", fontsize=12)
        axs[0].set_ylabel("Load", fontsize=12)
        axs[0].grid(alpha=0.3)

        # Load vs Distance over 21 km/h
        axs[1].scatter(
            df_gps_cleaned["distance_over_21"],
            df_gps_cleaned["load"],
            color="#0F9D58",
            alpha=0.7,
            s=40,
        )
        axs[1].set_title(
            "Load vs Distance Over 21 km/h", fontsize=14, fontweight="bold"
        )
        axs[1].set_xlabel("Distance Over 21 km/h", fontsize=12)
        axs[1].set_ylabel("Load", fontsize=12)
        axs[1].grid(alpha=0.3)

        # Load vs Accel/Decel > 2.5 m/s²
        axs[2].scatter(
            df_gps_cleaned["accel_decel_over_2_5"],
            df_gps_cleaned["load"],
            color="#DB4437",
            alpha=0.7,
            s=40,
        )
        axs[2].set_title(
            "Load vs Accel/Decel > 2.5 m/s²", fontsize=14, fontweight="bold"
        )
        axs[2].set_xlabel("Accel/Decel > 2.5 m/s²", fontsize=12)
        axs[2].set_ylabel("Load", fontsize=12)
        axs[2].grid(alpha=0.3)

        # Empirical cumulative distribution function (CDF) of load
        sorted_load = np.sort(df_gps_cleaned["load"])
        cdf = np.arange(1, len(sorted_load) + 1) / len(sorted_load)
        axs[3].plot(
            sorted_load,
            cdf,
            marker="o",
            linestyle="-",
            color="#9C27B0",
            markersize=4,
        )
        axs[3].set_title(
            "Empirical CDF of Load", fontsize=14, fontweight="bold"
        )
        axs[3].set_xlabel("Load", fontsize=12)
        axs[3].set_ylabel("Cumulative Probability", fontsize=12)
        axs[3].grid(alpha=0.3)

        plt.tight_layout()

    # Create DataFrame with load by date
    df_gps_cleaned["date"] = pd.to_datetime(
        df_gps_cleaned["date"], dayfirst=True
    )
    df_gps_cleaned["date_str"] = df_gps_cleaned["date"].dt.strftime("%d/%m/%Y")

    # Calculate average load by day
    df_load_by_date = df_gps_cleaned.groupby("date_str", as_index=False)[
        "load"
    ].mean()

    # Convert 'date_str' to datetime and sort
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )
    df_load_by_date = df_load_by_date.sort_values(by="date")

    # Add moving average of load
    df_load_by_date["window_mean"] = (
        df_load_by_date["load"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # Determine threshold for top x% of highest loads
    threshold_value = np.percentile(
        df_gps_cleaned["load"], 100 - top_x_percent
    )

    return threshold_value, df_load_by_date


def plot_weekly_danger(
    merged_df: pd.DataFrame, danger_rate: float = 10, show_plot: bool = True
) -> Figure:
    """
    Visualize weekly training overload distribution with color-coded danger levels.

    Args:
        merged_df: DataFrame with columns 'date' (datetime) and 'danger_score'
        danger_rate: Percentage to determine dangerous thresholds (default 10%)
        show_plot: Whether to display the plot (default True)

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
            return "#E53935"  # Bright red
        elif dangerous_days > 0:
            return "#FFD600"  # Bright yellow
        else:
            return "#43A047"  # Bright green

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
            showlegend=False,
            hovertemplate="Week: %{x}<br>Dangerous days: %{text}<extra></extra>",
        )
    )

    # Add legend items
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#43A047", size=12),
            name="Normal (0 days)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#FFD600", size=12),
            name="Warning (1-2 days)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#E53935", size=12),
            name="High Risk (3+ days)",
        )
    )

    fig.update_layout(
        title={
            "text": "Weekly Training Overload Distribution",
            "font": {"size": 20, "color": "#212121"},
        },
        xaxis_title={"text": "Week", "font": {"size": 16}},
        yaxis=dict(tickvals=[], showticklabels=False),
        template="plotly_white",
        legend_title={"text": "Risk Levels", "font": {"size": 16}},
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        height=500,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def plot_load_over_time(
    df_load_by_date: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rolling_window: int = 1,
) -> Figure:
    """
    Display load evolution over time with interactive visualization.

    Args:
        df_load_by_date: DataFrame with columns 'date_str' (dd/mm/yyyy) and 'load'
        start_date: Optional start date in dd/mm/yyyy format
        end_date: Optional end date in dd/mm/yyyy format
        rolling_window: Number of days for rolling average (default 1 = no rolling)

    Returns:
        Plotly figure with load evolution chart
    """
    df_viz = df_load_by_date.copy()
    df_viz["date"] = pd.to_datetime(df_viz["date_str"], format="%d/%m/%Y")

    # Filter data by date range if specified
    if start_date:
        df_viz = df_viz[
            df_viz["date"] >= pd.to_datetime(start_date, format="%d/%m/%Y")
        ]
    if end_date:
        df_viz = df_viz[
            df_viz["date"] <= pd.to_datetime(end_date, format="%d/%m/%Y")
        ]

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
            line=dict(color="#2196F3", width=3),
            marker=dict(
                color="#2196F3", size=8, line=dict(color="#FFFFFF", width=1)
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
        line=dict(color="#FF5722", width=2, dash="dash"),
    )

    # Add annotation for mean
    fig.add_annotation(
        x=df_viz["date"].max(),
        y=mean_load,
        text=f"Avg: {mean_load:.3f}",
        showarrow=False,
        font=dict(size=12, color="#FF5722"),
        xanchor="right",
        yanchor="bottom",
        xshift=10,
    )

    fig.update_layout(
        title={
            "text": "Load Evolution Over Time",
            "font": {"size": 20, "color": "#212121"},
        },
        xaxis_title={"text": "Date", "font": {"size": 16}},
        yaxis_title={"text": "Average Load", "font": {"size": 16}},
        xaxis=dict(
            tickformat="%d/%m/%Y",
            tickangle=-45,
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(gridcolor="#EEEEEE"),
        template="plotly_white",
        showlegend=True,
        autosize=True,
        hovermode="x unified",
        height=550,
        margin=dict(l=40, r=40, t=80, b=60),
    )

    return fig


def plot_load_vs_recovery(
    df_recovery: pd.DataFrame,
    df_gps: pd.DataFrame,
    start_date: str = "01/01/2023",
    end_date: str = "31/12/2030",
    alpha: float = 2,
    beta: float = 1,
    gamma: float = 3,
    top_x_percent: float = 5,
    show_plot: bool = True,
    window_size: int = 7,
) -> Figure:
    """
    Plot relationship between recovery score and load with danger score calculation.

    Args:
        df_recovery: DataFrame with recovery data (columns: 'sessionDate', 'metric', 'value')
        df_gps: DataFrame with GPS data (columns: 'date', 'distance', etc.)
        start_date: Start date for filtering data ("dd/mm/yyyy")
        end_date: End date for filtering data ("dd/mm/yyyy")
        alpha: Exponent for distance in load calculation
        beta: Exponent for accelerations/decelerations
        gamma: Exponent for distance over 21 km/h
        top_x_percent: Percentage for determining threshold
        show_plot: Whether to display the plot
        window_size: Window size for moving average

    Returns:
        Plotly figure with load vs recovery visualization
    """
    # Import and use relevant functions from other modules
    from recovery_viz import plot_global_recovery_score

    # Get recovery score data
    df_recovery_filtered = plot_global_recovery_score(
        df_recovery,
        start_date=start_date,
        end_date=end_date,
        show_plot=False,
        window_size=window_size,
    )

    # Calculate load data
    threshold, df_load_by_date = compute_load_and_plot(
        df_gps,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_x_percent=top_x_percent,
        show_plot=False,
        window_size=window_size,
    )

    # Merge recovery and load data
    df_load_by_date["date"] = pd.to_datetime(
        df_load_by_date["date_str"], format="%d/%m/%Y"
    )
    df_recovery_filtered["sessionDate"] = pd.to_datetime(
        df_recovery_filtered["sessionDate"], format="%d/%m/%Y"
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

    # Create interactive visualization with Plotly
    fig = go.Figure()

    # Add load trace
    fig.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["window_mean"],
            mode="lines",
            name="Load",
            line=dict(color="#2196F3", width=3),
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
            line=dict(color="#FFA000", width=3),
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
            line=dict(color="#E53935", width=3),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Risk: %{y:.3f}<extra></extra>",
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title={
            "text": "Load, Recovery and Risk Score Over Time",
            "font": {"size": 20, "color": "#212121"},
        },
        xaxis_title={"text": "Date", "font": {"size": 16}},
        yaxis_title={"text": "Score", "font": {"size": 16}},
        xaxis=dict(
            tickformat="%d/%m/%Y",
            tickangle=-45,
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(gridcolor="#EEEEEE"),
        template="plotly_white",
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
        margin=dict(l=40, r=40, t=80, b=60),
    )

    return fig
