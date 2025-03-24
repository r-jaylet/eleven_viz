import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_completeness_heatmap(completeness_df):
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
        color_continuous_scale="YlGnBu",
        title="Assessment Completeness by Category and Date",
    )

    fig.update_layout(
        height=400,
        xaxis_title="Assessment Category",
        yaxis_title="Session Date",
    )

    return fig


def create_category_completeness_bar(completeness_df):
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
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    fig.update_layout(showlegend=False)
    return fig


def create_daily_completeness_line(completeness_df):
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
    )

    return fig


def create_completeness_radar(latest_completeness):
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=latest_completeness["value"],
            theta=latest_completeness["category"],
            fill="toself",
            name="Completeness",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Latest Completeness by Category",
        showlegend=False,
    )

    return fig


def create_category_completeness_time(completeness_data, selected_category):
    fig = px.line(
        completeness_data,
        x="sessionDate",
        y="value",
        title=f"{selected_category} Completeness Over Time",
        labels={"value": "Completeness", "sessionDate": "Date"},
        markers=True,
        line_shape="linear",
    )

    fig.update_layout(yaxis_range=[0, 1.1])
    return fig


def create_category_composite_time(composite_data, selected_category):
    fig = px.line(
        composite_data,
        x="sessionDate",
        y="value",
        title=f"{selected_category} Composite Score Over Time",
        labels={"value": "Composite Score", "sessionDate": "Date"},
        markers=True,
        line_shape="linear",
    )

    return fig


def create_category_comparison(pivot_latest):
    fig = go.Figure()

    if "completeness" in pivot_latest.columns:
        fig.add_trace(
            go.Bar(
                x=pivot_latest["category"],
                y=pivot_latest["completeness"],
                name="Completeness",
                marker_color="royalblue",
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
                    marker_color="firebrick",
                )
            )

    fig.update_layout(
        title="Latest Category Metrics Comparison",
        xaxis_title="Category",
        yaxis_title="Value",
        barmode="group",
        legend_title="Metric Type",
    )

    return fig


def create_daily_tracking(daily_stats):
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
        color_continuous_scale="RdYlGn",
        size_max=20,
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats["sessionDate"],
            y=daily_stats["mean"],
            mode="lines",
            line=dict(color="grey", width=1),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=400,
        xaxis_title="Session Date",
        yaxis_title="Average Recovery Value",
    )

    return fig


def create_date_metrics_bar(date_values, selected_date):
    fig = px.bar(
        date_values,
        x="metric",
        y="value",
        color="category",
        title=f"All Recovery Metrics for {pd.to_datetime(selected_date).strftime('%d %b %Y')}",
        labels={"value": "Value", "metric": "Metric", "category": "Category"},
        hover_data=["metric_type"],
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        xaxis_title="",
        barmode="group",
    )

    return fig


def create_completeness_patterns(completeness_df):
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
    )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Completeness Value",
        legend_title="Category",
    )

    return fig


def create_correlation_heatmap(corr_matrix):
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="Correlation Between Recovery Metrics",
    )

    fig.update_layout(height=600, xaxis_tickangle=-45)
    return fig


def create_composite_line(composite_df):
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
    )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Composite Score",
        legend_title="Category",
    )

    return fig
