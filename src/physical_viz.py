from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# Consistent color palette matching gps_viz.py
COLORS = {
    "primary": "#1A237E",  # Dark blue
    "secondary": "#004D40",  # Dark green
    "accent1": "#311B92",  # Deep purple
    "accent2": "#01579B",  # Dark cyan
    "accent3": "#33691E",  # Dark lime
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


def create_expression_count_chart(df: pd.DataFrame) -> Figure:
    """Create bar chart showing count of tests by expression type."""
    expression_colors = {
        "isometric": COLORS["primary"],
        "dynamic": COLORS["secondary"],
    }

    fig = px.bar(
        df["expression"].value_counts().reset_index(),
        x="expression",
        y="count",
        color="expression",
        labels={"count": "Number of Tests", "expression": "Expression Type"},
        color_discrete_map=expression_colors,
        template=TEMPLATE,
    )

    fig.update_layout(
        xaxis_title={"text": "Expression Type", "font": {"size": 14}},
        yaxis_title={"text": "Number of Tests", "font": {"size": 14}},
        showlegend=False,
        hovermode="closest",
        margin=COMMON_MARGINS,
    )
    return fig


def create_expression_performance_boxplot(df: pd.DataFrame) -> Figure:
    """Create boxplot showing performance distribution by expression type."""
    expression_colors = {
        "isometric": COLORS["primary"],
        "dynamic": COLORS["secondary"],
    }

    fig = px.box(
        df.dropna(subset=["benchmarkPct"]),
        x="expression",
        y="benchmarkPct",
        color="expression",
        labels={
            "benchmarkPct": "Benchmark Percentile",
            "expression": "Expression Type",
        },
        color_discrete_map=expression_colors,
        template=TEMPLATE,
    )

    fig.update_layout(
        xaxis_title={"text": "Expression Type", "font": {"size": 14}},
        yaxis_title={"text": "Benchmark Percentile", "font": {"size": 14}},
        showlegend=False,
        hovermode="closest",
        margin=COMMON_MARGINS,
    )
    return fig


def create_expression_timeline(df: pd.DataFrame) -> Figure:
    """Create timeline chart showing performance by expression type over time."""
    filtered_df = df.dropna(subset=["benchmarkPct"])
    expression_colors = {
        "isometric": COLORS["primary"],
        "dynamic": COLORS["secondary"],
    }

    fig = px.scatter(
        filtered_df,
        x="testDate",
        y="benchmarkPct",
        color="expression",
        size=[10] * len(filtered_df),
        hover_data=["movement", "quality"],
        labels={
            "testDate": "Test Date",
            "benchmarkPct": "Benchmark Percentile",
        },
        color_discrete_map=expression_colors,
        template=TEMPLATE,
    )

    for expr in df["expression"].unique():
        sub_df = df[(df["expression"] == expr) & df["benchmarkPct"].notna()]
        if len(sub_df) > 1:
            fig.add_trace(
                go.Scatter(
                    x=sub_df["testDate"],
                    y=sub_df["benchmarkPct"],
                    mode="lines",
                    name=f"{expr} trend",
                    line=dict(
                        dash="dash",
                        width=1.5,
                        color=expression_colors.get(expr, COLORS["accent1"]),
                    ),
                    opacity=0.7,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis_title={"text": "Test Date", "font": {"size": 14}},
        yaxis_title={"text": "Benchmark Percentile", "font": {"size": 14}},
        hovermode="closest",
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


def create_movement_pie_chart(df: pd.DataFrame) -> Figure:
    """Create pie chart showing distribution of movement types."""
    fig = px.pie(
        df,
        names="movement",
        color_discrete_sequence=QUALITATIVE_PALETTE,
        template=TEMPLATE,
        hole=0.5,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+percent+name",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
        textfont=dict(size=12, color="#FFFFFF"),
    )

    fig.update_layout(
        margin=COMMON_MARGINS,
    )
    return fig

def create_movement_over_time_chart(df: pd.DataFrame) -> Figure:
    """Create stacked bar chart showing movement type distribution over months."""
    # Convert to month and year (e.g., '2024-03')
    df["month"] = df["testDate"].dt.to_period("M").astype(str)

    # Group by month and movement type
    monthly_counts = df.groupby(["month", "movement"]).size().reset_index(name="count")

    # Sort months properly
    monthly_counts["month"] = pd.to_datetime(monthly_counts["month"])
    monthly_counts = monthly_counts.sort_values("month")
    monthly_counts["month"] = monthly_counts["month"].dt.strftime("%b %Y")  # e.g., 'Mar 2024'

    fig = px.bar(
        monthly_counts,
        x="month",
        y="count",
        color="movement",
        template=TEMPLATE,
        color_discrete_sequence=QUALITATIVE_PALETTE,
        labels={
            "month": "Month",
            "count": "Number of Tests",
            "movement": "Movement Type",
        },
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title={"text": "Test Month", "font": {"size": 14}},
        yaxis_title={"text": "Number of Tests", "font": {"size": 14}},
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

def create_movement_quality_heatmap(df: pd.DataFrame) -> Figure:
    """Create heatmap showing relationship between movement types and quality."""
    pivot = pd.crosstab(df["movement"], df["quality"])

    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Quality", y="Movement", color="Count"),
        color_continuous_scale=["#E8EAF6", COLORS["primary"]],
        template=TEMPLATE,
    )

    fig.update_layout(
        height=max(400, len(df["movement"].unique()) * 40),
        margin=COMMON_MARGINS,
        coloraxis_colorbar=dict(
            title={"text": "Count", "font": {"size": 12}},
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300,
            titleside="right",
        ),
    )
    return fig


def create_movement_performance_chart(df: pd.DataFrame) -> Figure:
    """Create bar chart showing average performance by movement type with standard deviation."""
    movement_perf = (
        df.groupby("movement")["benchmarkPct"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    movement_perf = movement_perf.sort_values("mean", ascending=False)
    movement_perf = movement_perf[movement_perf["count"] > 0].copy()

    movement_perf["count_with_data"] = (
        df.groupby("movement")["benchmarkPct"]
        .count()
        .reindex(movement_perf["movement"])
        .values
    )

    movement_perf["upper"] = movement_perf["mean"] + movement_perf["std"]
    movement_perf["lower"] = movement_perf["mean"] - movement_perf["std"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=movement_perf["movement"],
            y=movement_perf["mean"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=movement_perf["upper"] - movement_perf["mean"],
                arrayminus=movement_perf["mean"] - movement_perf["lower"],
                color=COLORS["accent1"],
            ),
            name="Average Performance",
            hovertemplate="<b>%{x}</b><br>Avg Performance: %{y:.2f}<br>Tests with data: %{text}",
            text=movement_perf["count_with_data"],
            marker_color=COLORS["primary"],
        )
    )

    fig.update_layout(
        xaxis_title={"text": "Movement Type", "font": {"size": 14}},
        yaxis_title={
            "text": "Average Benchmark Percentile",
            "font": {"size": 14},
        },
        hovermode="closest",
        template=TEMPLATE,
        margin=COMMON_MARGINS,
    )
    return fig


def create_movement_trend_chart(df: pd.DataFrame) -> Figure:
    """Create line chart showing movement performance trends over the last 5 months."""
    # Ensure testDate is datetime
    df["testDate"] = pd.to_datetime(df["testDate"])

    # Filter to last 5 full calendar months
    latest_date = df["testDate"].max()
    five_months_ago = latest_date - pd.DateOffset(months=5)
    df_recent = df[df["testDate"] >= five_months_ago].copy()

    # Extract month in Year-Month format
    df_recent["month"] = df_recent["testDate"].dt.to_period("M").astype(str)

    # Group by month and movement
    trend_data = (
        df_recent.groupby(["month", "movement"])["benchmarkPct"]
        .mean()
        .reset_index()
    )

    # Sort months for proper display
    trend_data["month"] = pd.to_datetime(trend_data["month"])
    trend_data = trend_data.sort_values("month")
    trend_data["month"] = trend_data["month"].dt.strftime("%b %Y")

    # Create line plot
    fig = px.line(
        trend_data,
        x="month",
        y="benchmarkPct",
        color="movement",
        markers=True,
        template=TEMPLATE,
        color_discrete_sequence=QUALITATIVE_PALETTE,
        labels={
            "month": "Month",
            "benchmarkPct": "Avg Benchmark Percentile",
            "movement": "Movement Type",
        },
    )

    fig.update_layout(
        title="Movement Performance Trends (Last 5 Months)",
        xaxis_title={"text": "Month", "font": {"size": 14}},
        yaxis_title={"text": "Benchmark Percentile", "font": {"size": 14}},
        yaxis=dict(range=[0, 1]),
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


def create_performance_trend_chart(df: pd.DataFrame) -> Figure:
    """Create improved line chart showing smoothed performance trend over time."""
    filtered_df = df.sort_values("testDate").dropna(subset=["benchmarkPct"])

    # Daily mean
    daily_avg = (
        filtered_df.groupby("testDate")["benchmarkPct"]
        .mean()
        .reset_index()
    )
    
    # Smoothed trend (rolling mean)
    daily_avg["Smoothed"] = daily_avg["benchmarkPct"].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()

    # Raw daily average line
    fig.add_trace(go.Scatter(
        x=daily_avg["testDate"],
        y=daily_avg["benchmarkPct"],
        mode="lines+markers",
        name="Daily Avg",
        line=dict(color="gray", width=1),
        marker=dict(size=4)
    ))

    # Smoothed trend line
    fig.add_trace(go.Scatter(
        x=daily_avg["testDate"],
        y=daily_avg["Smoothed"],
        mode="lines",
        name="Smoothed Trend",
        line=dict(color=COLORS["primary"], width=3)
    ))

    fig.update_layout(
        title="Performance Trend Over Time",
        xaxis_title={"text": "Test Date", "font": {"size": 14}},
        yaxis_title={"text": "Benchmark Percentile", "font": {"size": 14}},
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template=TEMPLATE,
        margin=COMMON_MARGINS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        )
    )

    return fig


def create_monthly_performance_chart(df: pd.DataFrame) -> Optional[Figure]:
    """Create bar chart showing monthly average performance."""
    df = df.copy()
    df["month"] = df["testDate"].dt.strftime("%Y-%m")

    monthly_avg = df.groupby("month")["benchmarkPct"].mean().reset_index()
    monthly_count = df.groupby("month")["benchmarkPct"].count().reset_index()

    monthly_data = pd.merge(
        monthly_avg, monthly_count, on="month", suffixes=("_avg", "_count")
    )
    monthly_data = monthly_data[monthly_data["benchmarkPct_count"] > 0]

    if monthly_data.empty:
        return None

    fig = px.bar(
        monthly_data,
        x="month",
        y="benchmarkPct_avg",
        labels={
            "month": "Month",
            "benchmarkPct_avg": "Average Benchmark Percentile",
        },
        text=monthly_data["benchmarkPct_count"].apply(lambda x: f"{x} tests"),
        template=TEMPLATE,
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Performance: %{y:.1f}<br>Tests: %{text}",
        marker_line=dict(width=1, color="#FFFFFF"),
    )

    fig.update_layout(
        xaxis_title={"text": "Month", "font": {"size": 14}},
        yaxis_title={
            "text": "Average Benchmark Percentile",
            "font": {"size": 14},
        },
        margin=COMMON_MARGINS,
        xaxis=dict(tickangle=-45),
    )
    return fig


def detailed_stats_by_movement(
    df: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_date: str = "2050-12-31",
) -> Figure:
    """
    Create detailed performance chart for each movement type over time.

    Args:
        df: DataFrame containing performance data
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)

    Returns:
        Plotly figure with performance trends by movement type
    """
    df_filtered = df.copy()
    df_filtered["testDate"] = pd.to_datetime(df_filtered["testDate"])

    # Filter by date range
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered["testDate"] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered["testDate"] <= end_date]

    df_filtered = df_filtered.sort_values(by="testDate")

    # Create figure for first movement type
    if not df_filtered["movement"].empty:
        movement = df_filtered["movement"].unique()[0]
        df_movement = df_filtered[df_filtered["movement"] == movement]

        fig = go.Figure()
        color_idx = 0

        for expression in sorted(df_movement["expression"].unique()):
            df_expr = df_movement[df_movement["expression"] == expression]

            for quality in df_expr["quality"].unique():
                df_curve = df_expr[df_expr["quality"] == quality]

                if not df_curve.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_curve["testDate"],
                            y=df_curve["benchmarkPct"],
                            mode="lines+markers",
                            name=f"{expression} - {quality}",
                            line=dict(
                                color=QUALITATIVE_PALETTE[
                                    color_idx % len(QUALITATIVE_PALETTE)
                                ],
                                width=2,
                            ),
                            marker=dict(
                                size=8, line=dict(width=1, color="#FFFFFF")
                            ),
                        )
                    )
                    color_idx += 1

        fig.update_layout(
            xaxis_title={"text": "Date", "font": {"size": 14}},
            yaxis_title={"text": "Benchmark Percentile", "font": {"size": 14}},
            legend_title={
                "text": "Expression - Quality",
                "font": {"size": 14},
            },
            template=TEMPLATE,
            hovermode="closest",
            margin=COMMON_MARGINS,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
        )

        return fig

    # Return empty figure if no data
    return go.Figure()


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key performance indicators from performance data.

    Args:
        df: DataFrame containing performance metrics

    Returns:
        Dictionary with KPI calculations
    """
    kpis = {
        "total_entries": df.shape[0],
        "average_benchmark": df["benchmarkPct"].mean(),
        "benchmark_by_movement": df.groupby("movement")["benchmarkPct"]
        .mean()
        .to_dict(),
        "benchmark_by_quality": df.groupby("quality")["benchmarkPct"]
        .mean()
        .to_dict(),
        "benchmark_by_expression": df.groupby("expression")["benchmarkPct"]
        .mean()
        .to_dict(),
    }

    return kpis


def get_data_for_date(
    df: pd.DataFrame, date: Union[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Filter DataFrame to get data for a specific date.

    Args:
        df: DataFrame containing performance data
        date: Target date (string or timestamp)

    Returns:
        Filtered DataFrame containing only data for the specified date
    """
    df_copy = df.copy()
    df_copy["testDate"] = pd.to_datetime(df_copy["testDate"])
    date = pd.to_datetime(date)

    df_filtered = df_copy[df_copy["testDate"].dt.date == date.date()]

    if "movement" in df_copy.columns:
        df_filtered = df_filtered.sort_values(by="movement")

    return df_filtered
