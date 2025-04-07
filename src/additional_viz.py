from typing import Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


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

def plot_player_load_vs_expression(
    gps_df_active: pd.DataFrame,
    capability_df: pd.DataFrame,
) -> Figure:
    """
    Compare normalized GPS composite load with physical expression benchmark over time.
    """
    # Normalize relevant GPS metrics
    load_metrics = ["distance", "accel_decel_over_2_5", "distance_over_24", "peak_speed"]
    gps_df = gps_df_active.copy()

    for metric in load_metrics:
        col_norm = f"{metric}_norm"
        min_val = gps_df[metric].min()
        max_val = gps_df[metric].max()
        gps_df[col_norm] = (gps_df[metric] - min_val) / (max_val - min_val) if max_val != min_val else 0.0

    # Compute composite load score per session
    gps_df["composite_load"] = gps_df[[f"{m}_norm" for m in load_metrics]].mean(axis=1)

    # Daily average composite load
    gps_daily = gps_df.groupby("date")["composite_load"].mean().reset_index()

    # Daily average benchmark percent
    capability_daily = capability_df.groupby("testDate")["benchmarkPct"].mean().reset_index()

   # Apply 7-day rolling average to smooth the load curve
    gps_daily["composite_load_smooth"] = (
        gps_daily["composite_load"].rolling(window=7, min_periods=1).mean() 
    ) 

    # Create dual-axis figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=gps_daily["date"],
            y=gps_daily["composite_load"],
            mode="lines+markers",
            name="Composite Load (Normalized)",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(color=COLORS["primary"], size=6),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Load: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=gps_daily["date"],
            y=gps_daily["composite_load_smooth"],
            mode="lines",
            name="Smoothed Load (7-day Avg)",
            line=dict(color=COLORS["warning"], width=3, dash="dash"),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Smoothed Load: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=capability_daily["testDate"],
            y=capability_daily["benchmarkPct"],
            mode="lines+markers",
            name="BenchmarkPct (Expression)",
            line=dict(color=COLORS["accent2"], width=3),
            marker=dict(color=COLORS["accent2"], size=6),
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Benchmark: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title={
            "text": "Player Load vs Expression Development",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        xaxis_title="Date",
        yaxis_title="Composite Load (Normalized)",
        template=TEMPLATE,
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

    fig.update_yaxes(
        title_text="Composite Load (Normalized)",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(220,220,220,0.5)",
    )
    fig.update_yaxes(
        title_text="Benchmark Percentage",
        secondary_y=True,
        showgrid=False,
    )

    return fig


def plot_pre_post_match_recovery_dumbbell(
    gps_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    recovery_metric: str = "soreness_baseline_composite",
) -> Figure:
    """
    Compare pre- vs. post-match recovery scores for each match using a color-coded dumbbell chart (single player).

    Args:
        gps_df: GPS data from `load_gps()` with 'date' and 'is_match_day'.
        recovery_df: Recovery data from `load_recovery_status()` with 'sessionDate', 'metric', 'value'.
        recovery_metric: The specific recovery metric to compare.
    """
    match_days = gps_df[gps_df["is_match_day"] == True]["date"].dropna().unique()
    match_days = pd.to_datetime(match_days)

    rec = recovery_df[
        (recovery_df["metric"] == recovery_metric)
        & recovery_df["sessionDate"].notna()
        & recovery_df["value"].notna()
    ].copy()

    data = []
    for match_date in match_days:
        pre_day = match_date - pd.Timedelta(days=1)
        post_day = match_date + pd.Timedelta(days=1)

        # Allow tolerance window ±1 day
        pre_rec = rec[(rec["sessionDate"] >= pre_day - pd.Timedelta(days=1)) &
                      (rec["sessionDate"] <= pre_day + pd.Timedelta(days=1))]
        post_rec = rec[(rec["sessionDate"] >= post_day - pd.Timedelta(days=1)) &
                       (rec["sessionDate"] <= post_day + pd.Timedelta(days=1))]

        pre_value = pre_rec["value"].mean()
        post_value = post_rec["value"].mean()

        if pd.notna(pre_value) and pd.notna(post_value):
            delta = pre_value - post_value
            if delta > 0.3:
                color = COLORS["danger"]
                category = "🟥 High Impact"
            elif delta > 0.1:
                color = COLORS["warning"]
                category = "🟨 Moderate Impact"
            else:
                color = COLORS["success"]
                category = "🟩 Low Impact"

            data.append({
                "match_date": match_date.strftime("%d/%m/%Y"),
                "pre": pre_value,
                "post": post_value,
                "color": color,
                "category": category,
            })

    if not data:
        return go.Figure().update_layout(
            title="No recovery data available for pre- and post-match comparisons.",
            template=TEMPLATE,
        )

    df = pd.DataFrame(data)
    fig = go.Figure()

    # Dumbbell lines (color-coded)
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["pre"], row["post"]],
                y=[row["match_date"], row["match_date"]],
                mode="lines",
                line=dict(color=row["color"], width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Pre-match markers
    fig.add_trace(
        go.Scatter(
            x=df["pre"],
            y=df["match_date"],
            mode="markers+text",
            name="Pre-Match",
            marker=dict(color=COLORS["primary"], size=10),
            text=["Pre" for _ in df["pre"]],
            textposition="middle right",
            hovertemplate="Match: %{y}<br>Pre: %{x:.2f}<extra></extra>",
        )
    )

    # Post-match markers
    fig.add_trace(
        go.Scatter(
            x=df["post"],
            y=df["match_date"],
            mode="markers+text",
            name="Post-Match",
            marker=dict(color=COLORS["warning"], size=10),
            text=["Post" for _ in df["post"]],
            textposition="middle left",
            hovertemplate="Match: %{y}<br>Post: %{x:.2f}<extra></extra>",
        )
    )

    # Final layout
    fig.update_layout(
        title={
            "text": f"Pre vs Post Match Recovery – {recovery_metric.replace('_', ' ').title()}",
            "x": 0.5,
            "font": {"size": 18, "color": COLORS["text"]},
        },
        xaxis_title="Recovery Score",
        yaxis_title="Match Date",
        template=TEMPLATE,
        height=650,
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


import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Figure

def plot_recovery_vs_load_peaks(
    gps_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    recovery_metric: str = "soreness_baseline_composite",
    rolling_window: int = 7
) -> Figure:
    """
    Compare weekly rolling average of GPS composite load with recovery scores over time.

    Args:
        gps_df: Preprocessed GPS data including 'date' and load metrics.
        recovery_df: Preprocessed recovery data including 'sessionDate', 'metric', and 'value'.
        recovery_metric: Recovery metric to track (e.g. soreness, sleep).
        rolling_window: Days to calculate the rolling average for GPS load.

    Returns:
        Plotly Figure showing load (area) vs. recovery (line).
    """
    # Parse date columns
    gps_df["date"] = pd.to_datetime(gps_df["date"], errors="coerce")
    recovery_df["sessionDate"] = pd.to_datetime(recovery_df["sessionDate"], errors="coerce")

    # Calculate composite load (normalized)
    metrics = ["distance", "accel_decel_over_2_5", "distance_over_24", "peak_speed"]
    gps = gps_df.copy()
    for m in metrics:
        min_val, max_val = gps[m].min(), gps[m].max()
        gps[f"{m}_norm"] = (gps[m] - min_val) / (max_val - min_val) if max_val != min_val else 0
    gps["composite_load"] = gps[[f"{m}_norm" for m in metrics]].mean(axis=1)

    # Aggregate and compute rolling average
    gps_daily = gps.groupby("date")["composite_load"].mean().reset_index()
    gps_daily["rolling_load"] = gps_daily["composite_load"].rolling(window=rolling_window, min_periods=1).mean()

    # Filter and average recovery data
    rec = recovery_df[recovery_df["metric"] == recovery_metric].copy()
    rec_daily = rec.groupby("sessionDate")["value"].mean().reset_index()

    # Merge on date
    df = pd.merge(gps_daily, rec_daily, left_on="date", right_on="sessionDate", how="outer").sort_values("date")

    # Create the figure
    fig = go.Figure()

    # Area for training load
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rolling_load"],
            name="Rolling Load",
            fill="tozeroy",
            mode="lines",
            line=dict(color=COLORS["primary"]),
            yaxis="y1",
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Load: %{y:.2f}<extra></extra>",
        )
    )

    # Line for recovery score
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            name="Recovery Score",
            mode="lines+markers",
            line=dict(color=COLORS["warning"], width=2),
            yaxis="y2",
            hovertemplate="Date: %{x|%d/%m/%Y}<br>Recovery: %{y:.2f}<extra></extra>",
        )
    )

    # Layout
    fig.update_layout(
        title={
            "text": f"Recovery Score vs Training Load Peaks – {recovery_metric.replace('_', ' ').title()}",
            "x": 0.5,
            "font": {"size": 18, "color": COLORS["text"]},
        },
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Composite Load (Rolling Avg)",
            showgrid=False,
            titlefont=dict(color=COLORS["primary"]),
            tickfont=dict(color=COLORS["primary"]),
        ),
        yaxis2=dict(
            title="Recovery Score",
            overlaying="y",
            side="right",
            showgrid=False,
            titlefont=dict(color=COLORS["warning"]),
            tickfont=dict(color=COLORS["warning"]),
        ),
        template=TEMPLATE,
        height=600,
        margin=COMMON_MARGINS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def plot_readiness_snapshot_before_matches(
    gps_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    capability_df: pd.DataFrame,
    recovery_metric: str = "soreness_baseline_composite",
    readiness_thresholds: tuple = (-0.1, 0.1)
) -> go.Figure:
    """
    Display player readiness before matches using GPS load, recovery scores, and latest or fallback benchmarkPct.

    Args:
        gps_df: DataFrame with GPS tracking data ('date', 'is_match_day', and metrics).
        recovery_df: DataFrame with recovery scores ('sessionDate', 'metric', 'value').
        capability_df: DataFrame with physical benchmarks ('testDate', 'benchmarkPct').
        recovery_metric: The recovery metric to use (default: soreness_baseline_composite).
        readiness_thresholds: Tuple (low, high) for readiness color thresholds.

    Returns:
        Plotly horizontal bar chart showing color-coded readiness.
    """
    gps_df["date"] = pd.to_datetime(gps_df["date"])
    recovery_df["sessionDate"] = pd.to_datetime(recovery_df["sessionDate"])
    capability_df["testDate"] = pd.to_datetime(capability_df["testDate"])

    match_days = pd.to_datetime(gps_df[gps_df["is_match_day"] == True]["date"].dropna().unique())
    fallback_benchmark = (
        capability_df.sort_values("testDate")["benchmarkPct"]
        .dropna().iloc[0]
        if not capability_df.empty else None
    )

    readiness_records = []
    for match_date in match_days:
        load = gps_df[gps_df["date"] == match_date]["distance"].mean()

        rec = recovery_df[
            (recovery_df["metric"] == recovery_metric) &
            (recovery_df["sessionDate"] >= match_date - pd.Timedelta(days=2)) &
            (recovery_df["sessionDate"] <= match_date - pd.Timedelta(days=1))
        ]
        recovery_score = rec["value"].mean() if not rec.empty else None

        cap = capability_df[capability_df["testDate"] <= match_date]
        benchmark_score = (
            cap.sort_values("testDate")["benchmarkPct"].iloc[-1]
            if not cap.empty else fallback_benchmark
        )

        if pd.isna(recovery_score) and pd.isna(benchmark_score):
            continue

        scores = [s for s in [recovery_score, benchmark_score] if pd.notna(s)]
        composite = sum(scores) / len(scores) if scores else None

        if composite is None:
            color = COLORS["accent2"]
        elif composite < readiness_thresholds[0]:
            color = COLORS["danger"]
        elif composite > readiness_thresholds[1]:
            color = COLORS["success"]
        else:
            color = COLORS["warning"]

        readiness_records.append({
            "match_date": match_date,
            "load": load,
            "recovery": recovery_score,
            "benchmarkPct": benchmark_score,
            "readiness": composite,
            "color": color,
        })

    df_ready = pd.DataFrame(readiness_records)

    if df_ready.empty:
        return go.Figure().update_layout(
            title="No match readiness data available.",
            template=TEMPLATE
        )

    fig = go.Figure()
    for _, row in df_ready.iterrows():
        fig.add_trace(go.Bar(
            x=[row["readiness"]],
            y=[row["match_date"]],
            orientation="h",
            marker=dict(color=row["color"]),
            hovertemplate=(
                f"Match: {row['match_date'].strftime('%d/%m/%Y')}<br>"
                f"Load: {row['load']:.0f}<br>"
                f"Recovery: {row['recovery']:.2f}<br>"
                f"Benchmark: {row['benchmarkPct']:.2f}<br>"
                f"Readiness Score: {row['readiness']:.2f}<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title={
            "text": "📅 Readiness Snapshot Before Each Match",
            "x": 0.5,
            "font": {"size": 18, "color": COLORS["text"]},
        },
        xaxis=dict(
            title="Readiness Score (Recovery + Benchmark)",
            gridcolor="rgba(200,200,200,0.2)",
            zeroline=True
        ),
        yaxis=dict(
            title="Match Date",
            autorange="reversed",
            gridcolor="rgba(200,200,200,0.2)"
        ),
        height=500 + len(df_ready) * 12,
        template=TEMPLATE,
        margin=COMMON_MARGINS
    )

    return fig
