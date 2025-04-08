from typing import Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

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
        color_continuous_scale=["#E8EAF6", COLORS["primary"]],
    )

    fig.update_layout(
        height=400,
        xaxis_title={"text": "Assessment Category", "font": {"size": 14}},
        yaxis_title={"text": "Session Date", "font": {"size": 14}},
        template=TEMPLATE,
        margin=COMMON_MARGINS,
        coloraxis_colorbar=dict(
            title={"text": "Completeness", "font": {"size": 12}},
            thicknessmode="pixels",
            thickness=20,
        ),
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
        labels={"value": "Average Completeness", "category": "Category"},
        color="category",
        color_discrete_sequence=QUALITATIVE_PALETTE,
        template=TEMPLATE,
    )

    fig.update_layout(
        xaxis_title={"text": "Category", "font": {"size": 14}},
        yaxis_title={"text": "Average Completeness", "font": {"size": 14}},
        showlegend=False,
        height=400,
        margin=COMMON_MARGINS,
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
        labels={"value": "Average Completeness", "sessionDate": "Date"},
        markers=True,
        template=TEMPLATE,
    )

    fig.update_traces(
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(
            color=COLORS["primary"],
            size=8,
            line=dict(color="#FFFFFF", width=1),
        ),
    )

    fig.update_layout(
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Average Completeness", "font": {"size": 14}},
        height=400,
        margin=COMMON_MARGINS,
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
            line_color=COLORS["primary"],
            fillcolor=f"rgba{tuple(int(COLORS['primary'][i:i+2], 16) for i in (1, 3, 5)) + (0.5,)}",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks="outside",
                gridcolor="rgba(240, 240, 240, 0.3)",
            ),
            angularaxis=dict(
                showticklabels=True,
                ticks="outside",
                gridcolor="rgba(240, 240, 240, 0.3)",
                linecolor="rgba(240, 240, 240, 0.3)",
            ),
            bgcolor="rgba(248, 248, 248, 0.5)",
        ),
        showlegend=False,
        template=TEMPLATE,
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
        template=TEMPLATE,
    )

    fig.update_traces(
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(
            color=COLORS["primary"],
            size=8,
            line=dict(color="#FFFFFF", width=1),
        ),
    )

    fig.update_layout(
        title={"font": {"size": 18, "color": COLORS["text"]}, "x": 0.5},
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Completeness", "font": {"size": 14}},
        yaxis_range=[0, 1.1],
        height=400,
        margin=COMMON_MARGINS,
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
        template=TEMPLATE,
    )

    fig.update_traces(
        line=dict(color=COLORS["secondary"], width=2.5),
        marker=dict(
            color=COLORS["secondary"],
            size=8,
            line=dict(color="#FFFFFF", width=1),
        ),
    )

    fig.update_layout(
        title={"font": {"size": 18, "color": COLORS["text"]}, "x": 0.5},
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Composite Score", "font": {"size": 14}},
        height=400,
        margin=COMMON_MARGINS,
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
                marker_color=COLORS["primary"],
                marker_line=dict(color="#FFFFFF", width=1),
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
                    marker_color=COLORS["secondary"],
                    marker_line=dict(color="#FFFFFF", width=1),
                )
            )

    fig.update_layout(
        title={
            "text": "Latest Category Metrics Comparison",
            "font": {"size": 18, "color": COLORS["text"]},
            "x": 0.5,
        },
        xaxis_title={"text": "Category", "font": {"size": 14}},
        yaxis_title={"text": "Value", "font": {"size": 14}},
        barmode="group",
        legend_title={"text": "Metric Type", "font": {"size": 14}},
        template=TEMPLATE,
        height=450,
        margin=COMMON_MARGINS,
    )

    return fig


def create_daily_tracking(daily_stats: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        daily_stats,
        x="sessionDate",
        y="mean",
        size="count",
        color="mean",
        labels={
            "sessionDate": "Date",
            "mean": "Average Value",
            "count": "Number of Metrics",
        },
        color_continuous_scale=["#E8EAF6", COLORS["primary"]],
        size_max=20,
        template=TEMPLATE,
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats["sessionDate"],
            y=daily_stats["mean"],
            mode="lines",
            line=dict(color=COLORS["accent1"], width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title={"text": "Session Date", "font": {"size": 14}},
        yaxis_title={"text": "Average Recovery Value", "font": {"size": 14}},
        height=400,
        margin=COMMON_MARGINS,
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
        color_discrete_sequence=QUALITATIVE_PALETTE,
        template=TEMPLATE,
    )

    fig.update_layout(
        xaxis_title={"text": "", "font": {"size": 14}},
        yaxis_title={"text": "Value", "font": {"size": 14}},
        legend_title={"text": "Category", "font": {"size": 14}},
        height=500,
        xaxis_tickangle=-45,
        barmode="group",
        margin=COMMON_MARGINS,
    )

    fig.update_traces(
        marker_line=dict(color="#FFFFFF", width=1),
    )

    return fig


def create_completeness_patterns(completeness_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        completeness_df,
        x="sessionDate",
        y="value",
        color="category",
        labels={
            "value": "Completeness",
            "sessionDate": "Date",
            "category": "Category",
        },
        markers=True,
        color_discrete_sequence=QUALITATIVE_PALETTE,
        template=TEMPLATE,
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="#FFFFFF")),
        line=dict(width=2),
    )

    fig.update_layout(
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Completeness Value", "font": {"size": 14}},
        legend_title={"text": "Category", "font": {"size": 14}},
        height=400,
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


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale=[
            "#E74C3C",
            "#FFFFFF",
            COLORS["primary"],
        ],  # Red to white to blue
        range_color=[-1, 1],
        template=TEMPLATE,
    )

    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        margin=COMMON_MARGINS,
    )

    return fig


def create_composite_line(composite_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        composite_df,
        x="sessionDate",
        y="value",
        color="category",
        labels={
            "value": "Composite Score",
            "sessionDate": "Date",
            "category": "Category",
        },
        markers=True,
        color_discrete_sequence=QUALITATIVE_PALETTE,
        template=TEMPLATE,
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="#FFFFFF")),
        line=dict(width=2),
    )

    fig.update_layout(
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Composite Score", "font": {"size": 14}},
        legend_title={"text": "Category", "font": {"size": 14}},
        height=400,
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


def plot_global_recovery_score(
    df: pd.DataFrame,
    window_size: int = 7,
) -> go.Figure:
    """
    Generates a time series plot showing the evolution of the global recovery score
    with a configurable moving average window.
    """
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    df_total = df[df["metric"] == "emboss_baseline_score"].copy()
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
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(
                size=7,
                color=COLORS["primary"],
                line=dict(width=1, color="#FFFFFF"),
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_total["sessionDate"],
            y=df_total["rolling_mean"],
            mode="lines",
            name=f"{window_size}-day Moving Average",
            line=dict(color="#FFA500", width=4.5),
        )
    )

    fig.update_layout(
        xaxis_title={"text": "Date", "font": {"size": 14}},
        yaxis_title={"text": "Score", "font": {"size": 14}},
        template=TEMPLATE,
        height=450,
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
                line=dict(color=COLORS["primary"], width=2),
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
                line=dict(color=COLORS["secondary"], width=2, dash="dot"),
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
            line=dict(color=COLORS["primary"], width=2),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Composite",
            line=dict(color=COLORS["secondary"], width=2, dash="dot"),
            showlegend=True,
        )
    )

    fig.update_layout(
        height=800,
        template=TEMPLATE,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        margin=COMMON_MARGINS,
    )

    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 1.1], row=i, col=j)
            fig.update_xaxes(
                showgrid=True,
                gridcolor="rgba(240, 240, 240, 0.5)",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(240, 240, 240, 0.5)",
                row=i,
                col=j,
            )

    return fig


def plot_weekly_recovery_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Plots a weekly heatmap of recovery scores for a single player.
    Rows = Weeks (e.g. 2024-W14), Columns = Days of the week.
    Color = Mean recovery score.

    Parameters:
    - df: DataFrame containing 'sessionDate', 'metric', and 'value' columns.
    Returns:
    - Plotly Figure object.
    """
    df = df[df["metric"] == "emboss_baseline_score"]
    df["sessionDate"] = pd.to_datetime(df["sessionDate"], format="%d/%m/%Y")

    # Add weekday and week labels
    df["weekday"] = df["sessionDate"].dt.day_name()
    df["weekday_num"] = df["sessionDate"].dt.weekday
    df["week"] = df["sessionDate"].dt.isocalendar().week
    df["year"] = df["sessionDate"].dt.isocalendar().year
    df["week_label"] = df["year"].astype(str) + "-W" + df["week"].astype(str)

    # Pivot for heatmap
    pivot_df = df.pivot_table(
        index="week_label", columns="weekday", values="value", aggfunc="mean"
    )

    # Reorder weekdays
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    pivot_df = pivot_df.reindex(columns=weekday_order)

    # Build heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale="RdYlGn",
            colorbar=dict(title="Recovery Score"),
        )
    )

    fig.update_layout(
        title="Weekly Recovery Heatmap (Single Player)",
        xaxis_title="Day of Week",
        yaxis_title="Week",
        template="plotly_dark",
        height=600,
    )

    return fig
