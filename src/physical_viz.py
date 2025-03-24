from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_expression_count_chart(df):
    fig = px.bar(
        df["expression"].value_counts().reset_index(),
        x="expression",
        y="count",
        color="expression",
        title="Number of Tests by Expression Type",
        labels={"count": "Number of Tests", "expression": "Expression Type"},
        color_discrete_map={"isometric": "#636EFA", "dynamic": "#EF553B"},
    )
    fig.update_layout(showlegend=False)
    return fig


def create_expression_performance_boxplot(df):
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
        color_discrete_map={"isometric": "#636EFA", "dynamic": "#EF553B"},
    )
    fig.update_layout(showlegend=False)
    return fig


def create_expression_timeline(df):
    fig = px.scatter(
        df.dropna(subset=["benchmarkPct"]),
        x="testDate",
        y="benchmarkPct",
        color="expression",
        size=[10] * len(df.dropna(subset=["benchmarkPct"])),
        hover_data=["movement", "quality"],
        title="Performance Timeline by Expression Type",
        labels={
            "testDate": "Test Date",
            "benchmarkPct": "Benchmark Percentile",
        },
        color_discrete_map={"isometric": "#636EFA", "dynamic": "#EF553B"},
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
                )
            )
    return fig


def create_movement_pie_chart(df):
    fig = px.pie(
        df,
        names="movement",
        title="Distribution of Movement Types",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def create_movement_quality_heatmap(df):
    pivot = pd.crosstab(df["movement"], df["quality"])
    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title="Movement vs Quality Heatmap",
        labels=dict(x="Quality", y="Movement", color="Count"),
        color_continuous_scale="YlGnBu",
    )
    fig.update_layout(height=400)
    return fig


def create_movement_performance_chart(df):
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
            hovertemplate="Movement: %{x}<br>Avg Performance: %{y:.2f}<br>Tests with data: %{text}",
            text=movement_perf["count_with_data"],
            marker_color="royalblue",
        )
    )

    fig.update_layout(
        title="Average Performance by Movement Type (with Std Dev)",
        xaxis_title="Movement Type",
        yaxis_title="Average Benchmark Percentile",
        hovermode="x",
    )
    return fig


def create_performance_trend_chart(df):
    fig = px.line(
        df.sort_values("testDate").dropna(subset=["benchmarkPct"]),
        x="testDate",
        y="benchmarkPct",
        markers=True,
        labels={
            "testDate": "Test Date",
            "benchmarkPct": "Benchmark Percentile",
        },
        title="Overall Performance Trend Over Time",
    )

    df_filtered = df.dropna(subset=["benchmarkPct"])
    x = df_filtered["testDate"].map(lambda x: x.toordinal()).values
    y = df_filtered["benchmarkPct"].values

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
                line=dict(color="red", dash="dash"),
            )
        )

    fig.update_traces(
        hovertemplate="Date: %{x|%d %b %Y}<br>Performance: %{y:.3f}"
    )
    return fig


def create_monthly_performance_chart(df):
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
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="Month: %{x}<br>Avg Performance: %{y:.3f}<br>Tests: %{text}",
    )
    return fig
