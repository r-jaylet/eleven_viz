from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

# Common visualization settings
CHART_TEMPLATE = "plotly_white"
COLOR_MAP = {"isometric": "#1f77b4", "dynamic": "#ff7f0e"}
HOVER_MODE = "closest"
COMMON_MARGINS = dict(l=50, r=20, t=50, b=50)


def create_expression_count_chart(df: pd.DataFrame) -> Figure:
    """Create bar chart showing count of tests by expression type."""
    fig = px.bar(
        df["expression"].value_counts().reset_index(),
        x="expression",
        y="count",
        color="expression",
        title="Number of Tests by Expression Type",
        labels={"count": "Number of Tests", "expression": "Expression Type"},
        color_discrete_map=COLOR_MAP,
        template=CHART_TEMPLATE,
    )
    fig.update_layout(
        showlegend=False, hovermode=HOVER_MODE, margin=COMMON_MARGINS
    )
    return fig


def create_expression_performance_boxplot(df: pd.DataFrame) -> Figure:
    """Create boxplot showing performance distribution by expression type."""
    fig = px.box(
        df.dropna(subset=["benchmarkPct"]),
        x="expression",
        y="benchmarkPct",
        color="expression",
        title="Performance by Expression Type",
        labels={
            "benchmarkPct": "Benchmark Percentile",
            "expression": "Expression Type",
        },
        color_discrete_map=COLOR_MAP,
        template=CHART_TEMPLATE,
    )
    fig.update_layout(
        showlegend=False, hovermode=HOVER_MODE, margin=COMMON_MARGINS
    )
    return fig


def create_expression_timeline(df: pd.DataFrame) -> Figure:
    """Create timeline chart showing performance by expression type over time."""
    filtered_df = df.dropna(subset=["benchmarkPct"])

    fig = px.scatter(
        filtered_df,
        x="testDate",
        y="benchmarkPct",
        color="expression",
        size=[10] * len(filtered_df),
        hover_data=["movement", "quality"],
        title="Performance Timeline by Expression Type",
        labels={
            "testDate": "Test Date",
            "benchmarkPct": "Benchmark Percentile",
        },
        color_discrete_map=COLOR_MAP,
        template=CHART_TEMPLATE,
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
                    line=dict(dash="dash", width=1),
                    opacity=0.6,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(hovermode=HOVER_MODE, margin=COMMON_MARGINS)
    return fig


def create_movement_pie_chart(df: pd.DataFrame) -> Figure:
    """Create pie chart showing distribution of movement types."""
    fig = px.pie(
        df,
        names="movement",
        title="Distribution of Movement Types",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=CHART_TEMPLATE,
        hole=0.4,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+percent+name",
    )
    fig.update_layout(margin=COMMON_MARGINS)
    return fig


def create_movement_quality_heatmap(df: pd.DataFrame) -> Figure:
    """Create heatmap showing relationship between movement types and quality."""
    pivot = pd.crosstab(df["movement"], df["quality"])

    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title="Movement vs Quality Heatmap",
        labels=dict(x="Quality", y="Movement", color="Count"),
        color_continuous_scale="Viridis",
        template=CHART_TEMPLATE,
    )

    fig.update_layout(
        height=max(400, len(df["movement"].unique()) * 40),
        margin=COMMON_MARGINS,
        coloraxis_colorbar=dict(
            title="Count",
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300,
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
            ),
            name="Average Performance",
            hovertemplate="<b>%{x}</b><br>Avg Performance: %{y:.2f}<br>Tests with data: %{text}",
            text=movement_perf["count_with_data"],
            marker_color="#1f77b4",
        )
    )

    fig.update_layout(
        title="Average Performance by Movement Type",
        xaxis_title="Movement Type",
        yaxis_title="Average Benchmark Percentile",
        hovermode=HOVER_MODE,
        template=CHART_TEMPLATE,
        margin=COMMON_MARGINS,
    )
    return fig


def create_performance_trend_chart(df: pd.DataFrame) -> Figure:
    """Create line chart showing overall performance trend over time with trend line."""
    filtered_df = df.sort_values("testDate").dropna(subset=["benchmarkPct"])

    fig = px.line(
        filtered_df,
        x="testDate",
        y="benchmarkPct",
        markers=True,
        labels={
            "testDate": "Test Date",
            "benchmarkPct": "Benchmark Percentile",
        },
        title="Overall Performance Trend Over Time",
        template=CHART_TEMPLATE,
    )

    x = filtered_df["testDate"].map(lambda x: x.toordinal()).values
    y = filtered_df["benchmarkPct"].values

    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        x_range = np.linspace(min(x), max(x), 100)
        fig.add_trace(
            go.Scatter(
                x=[datetime.fromordinal(int(i)) for i in x_range],
                y=p(x_range),
                mode="lines",
                name="Trend",
                line=dict(color="#ff7f0e", dash="dash", width=2),
                hoverinfo="skip",
            )
        )

    fig.update_traces(
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Performance: %{y:.1f}"
    )
    fig.update_layout(hovermode=HOVER_MODE, margin=COMMON_MARGINS)
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
        title="Monthly Average Performance",
        labels={
            "month": "Month",
            "benchmarkPct_avg": "Average Benchmark Percentile",
        },
        text=monthly_data["benchmarkPct_count"].apply(lambda x: f"{x} tests"),
        template=CHART_TEMPLATE,
        color_discrete_sequence=["#1f77b4"],
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Performance: %{y:.1f}<br>Tests: %{text}",
    )
    fig.update_layout(margin=COMMON_MARGINS, xaxis=dict(tickangle=-45))
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
        color_palette = px.colors.qualitative.Bold
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
                                color=color_palette[
                                    color_idx % len(color_palette)
                                ],
                                width=2,
                            ),
                            marker=dict(size=8),
                        )
                    )
                    color_idx += 1

        fig.update_layout(
            title=f"Performance Over Time - {movement}",
            xaxis_title="Date",
            yaxis_title="Benchmark Percentile",
            legend_title="Expression - Quality",
            template=CHART_TEMPLATE,
            hovermode=HOVER_MODE,
            margin=COMMON_MARGINS,
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
