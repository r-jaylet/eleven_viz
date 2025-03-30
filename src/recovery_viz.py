from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def create_completeness_heatmap(completeness_df: pd.DataFrame) -> go.Figure:
    pivot_completeness = completeness_df.pivot_table(
        index="sessionDate",
        columns="category",
        values="value",
        aggfunc="first",
    ).fillna(0)

    fig = px.imshow(
        pivot_completeness,
        labels=dict(x="Category", y="Date", color="Completeness"),
        x=pivot_completeness.columns,
        y=pivot_completeness.index.strftime("%d %b"),
        color_continuous_scale="viridis",
        title="Assessment Completeness by Category and Date",
    )

    fig.update_layout(
        height=400,
        xaxis_title="Assessment Category",
        yaxis_title="Session Date",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_category_completeness_bar(
    completeness_df: pd.DataFrame,
) -> go.Figure:
    category_completeness = (
        completeness_df.groupby("category")["value"].mean().reset_index()
    )

    fig = px.bar(
        category_completeness,
        x="category",
        y="value",
        title="Average Completeness by Category",
        labels={"value": "Average Completeness", "category": "Category"},
        color="category",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_daily_completeness_line(completeness_df: pd.DataFrame) -> go.Figure:
    daily_completeness = (
        completeness_df.groupby("sessionDate")["value"].mean().reset_index()
    )

    fig = px.line(
        daily_completeness,
        x="sessionDate",
        y="value",
        title="Daily Average Completeness",
        labels={"value": "Average Completeness", "sessionDate": "Date"},
        markers=True,
        line_shape="spline",
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_completeness_radar(latest_completeness: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=latest_completeness["value"],
            theta=latest_completeness["category"],
            fill="toself",
            name="Completeness",
            line_color="#1f77b4",
            fillcolor="rgba(31, 119, 180, 0.5)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks="outside",
            ),
            angularaxis=dict(
                showticklabels=True,
                ticks="outside",
            ),
        ),
        title="Latest Completeness by Category",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=70, r=70, t=80, b=50),
    )

    return fig


def create_category_completeness_time(
    completeness_data: pd.DataFrame, selected_category: str
) -> go.Figure:
    fig = px.line(
        completeness_data,
        x="sessionDate",
        y="value",
        title=f"{selected_category} Completeness Over Time",
        labels={"value": "Completeness", "sessionDate": "Date"},
        markers=True,
        line_shape="spline",
    )

    fig.update_layout(
        yaxis_range=[0, 1.1],
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_category_composite_time(
    composite_data: pd.DataFrame, selected_category: str
) -> go.Figure:
    fig = px.line(
        composite_data,
        x="sessionDate",
        y="value",
        title=f"{selected_category} Composite Score Over Time",
        labels={"value": "Composite Score", "sessionDate": "Date"},
        markers=True,
        line_shape="spline",
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_category_comparison(pivot_latest: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "completeness" in pivot_latest.columns:
        fig.add_trace(
            go.Bar(
                x=pivot_latest["category"],
                y=pivot_latest["completeness"],
                name="Completeness",
                marker_color="#1f77b4",
            )
        )

    if (
        "composite" in pivot_latest.columns
        and not pivot_latest["composite"].isna().all()
    ):
        if pivot_latest["composite"].max() > 0:
            normalized_composite = (
                pivot_latest["composite"] / pivot_latest["composite"].max()
            )

            fig.add_trace(
                go.Bar(
                    x=pivot_latest["category"],
                    y=normalized_composite,
                    name="Normalized Composite Score",
                    marker_color="#ff7f0e",
                )
            )

    fig.update_layout(
        title="Latest Category Metrics Comparison",
        xaxis_title="Category",
        yaxis_title="Value",
        barmode="group",
        legend_title="Metric Type",
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_daily_tracking(daily_stats: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        daily_stats,
        x="sessionDate",
        y="mean",
        size="count",
        color="mean",
        title="Daily Recovery Status Overview",
        labels={
            "sessionDate": "Date",
            "mean": "Average Value",
            "count": "Number of Metrics",
        },
        color_continuous_scale="viridis",
        size_max=20,
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats["sessionDate"],
            y=daily_stats["mean"],
            mode="lines",
            line=dict(color="rgba(128, 128, 128, 0.5)", width=1.5),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=400,
        xaxis_title="Session Date",
        yaxis_title="Average Recovery Value",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_date_metrics_bar(
    date_values: pd.DataFrame, selected_date: Union[str, pd.Timestamp]
) -> go.Figure:
    fig = px.bar(
        date_values,
        x="metric",
        y="value",
        color="category",
        title=f"All Recovery Metrics for {pd.to_datetime(selected_date).strftime('%d %b %Y')}",
        labels={"value": "Value", "metric": "Metric", "category": "Category"},
        hover_data=["metric_type"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        xaxis_title="",
        barmode="group",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_completeness_patterns(completeness_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        completeness_df,
        x="sessionDate",
        y="value",
        color="category",
        title="Completeness Patterns by Category",
        labels={
            "value": "Completeness",
            "sessionDate": "Date",
            "category": "Category",
        },
        markers=True,
        line_shape="spline",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Completeness Value",
        legend_title="Category",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="Correlation Between Recovery Metrics",
    )

    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_composite_line(composite_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        composite_df,
        x="sessionDate",
        y="value",
        color="category",
        title="Composite Scores by Category",
        labels={
            "value": "Composite Score",
            "sessionDate": "Date",
            "category": "Category",
        },
        markers=True,
        line_shape="spline",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Composite Score",
        legend_title="Category",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def plot_global_recovery_score(
    df: pd.DataFrame,
    start_date: str = "01/01/2023",
    end_date: str = "31/12/2030",
    window_size: int = 7,
) -> go.Figure:
    """
    Generates a time series plot showing the evolution of the global recovery score
    with a configurable moving average window.
    """
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    df_total = df[df["metric"] == "emboss_baseline_score"].copy()
    df_total = df_total[
        (
            df_total["sessionDate"]
            >= pd.to_datetime(start_date, format="%d/%m/%Y")
        )
        & (
            df_total["sessionDate"]
            <= pd.to_datetime(end_date, format="%d/%m/%Y")
        )
    ]
    df_total = df_total.sort_values("sessionDate")

    df_total["rolling_mean"] = (
        df_total["value"].rolling(window=window_size, min_periods=1).mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_total["sessionDate"],
            y=df_total["value"],
            mode="lines+markers",
            name="Raw Score",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_total["sessionDate"],
            y=df_total["rolling_mean"],
            mode="lines",
            name=f"{window_size}-day Moving Average",
            line=dict(color="#ff7f0e", width=2.5),
        )
    )

    fig.update_layout(
        title="Evolution of Global Recovery Score",
        xaxis_title="Date",
        yaxis_title="Score",
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig


def plot_recovery_metrics_by_category(
    df: pd.DataFrame, start_date: str, end_date: str
) -> go.Figure:
    """
    Generates time series for each category with completeness and composite metrics
    """
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    df = df[
        (df["sessionDate"] >= pd.to_datetime(start_date, format="%d/%m/%Y"))
        & (df["sessionDate"] <= pd.to_datetime(end_date, format="%d/%m/%Y"))
    ]

    categories = [
        "bio",
        "msk_joint_range",
        "msk_load_tolerance",
        "subjective",
        "soreness",
        "sleep",
    ]

    fig = sp.make_subplots(
        rows=3,
        cols=2,
        subplot_titles=categories,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, category in enumerate(categories):
        df_cat = df[df["category"] == category]

        df_completeness = df_cat[
            df_cat["metric"].str.endswith("_completeness")
        ]
        df_composite = df_cat[df_cat["metric"].str.endswith("_composite")]

        row, col = (i // 2) + 1, (i % 2) + 1

        fig.add_trace(
            go.Scatter(
                x=df_completeness["sessionDate"],
                y=df_completeness["value"],
                mode="lines",
                name=f"{category} - Completeness",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=df_composite["sessionDate"],
                y=df_composite["value"],
                mode="lines",
                name=f"{category} - Composite",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Completeness",
            line=dict(color="#1f77b4", width=2),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Composite",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Evolution of Recovery Metrics by Category",
        height=800,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50),
    )

    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 1.1], row=i, col=j)

    return fig
