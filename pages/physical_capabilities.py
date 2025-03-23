from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import load_physical_capabilities


def show():
    try:

        df = load_physical_capabilities()

        # Dashboard title
        st.title("Athletic Performance Analytics Dashboard")
        st.markdown("### Testing Performance Data Analysis")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Expression Analysis",
                "Movement Type Distribution",
                "Performance Trends",
                "Quality Comparison",
                "Testing Calendar",
            ]
        )

        # Tab 1: Expression Analysis - Distribution and Performance
        with tab1:
            st.subheader("Expression Type Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Count of tests by expression type
                fig_expr_count = px.bar(
                    df["expression"].value_counts().reset_index(),
                    x="expression",
                    y="count",
                    color="expression",
                    title="Number of Tests by Expression Type",
                    labels={
                        "count": "Number of Tests",
                        "expression": "Expression Type",
                    },
                    color_discrete_map={
                        "isometric": "#636EFA",
                        "dynamic": "#EF553B",
                    },
                )
                fig_expr_count.update_layout(showlegend=False)
                st.plotly_chart(fig_expr_count, use_container_width=True)

            with col2:
                # Performance by expression type (box plot)
                fig_expr_perf = px.box(
                    df.dropna(subset=["benchmarkPct"]),
                    x="expression",
                    y="benchmarkPct",
                    color="expression",
                    title="Performance by Expression Type",
                    labels={
                        "benchmarkPct": "Benchmark Percentile",
                        "expression": "Expression Type",
                    },
                    color_discrete_map={
                        "isometric": "#636EFA",
                        "dynamic": "#EF553B",
                    },
                )
                fig_expr_perf.update_layout(showlegend=False)
                st.plotly_chart(fig_expr_perf, use_container_width=True)

            # Scatter plot showing all tests with benchmark values by expression
            st.subheader("Performance Timeline by Expression Type")
            fig_expr_timeline = px.scatter(
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
                color_discrete_map={
                    "isometric": "#636EFA",
                    "dynamic": "#EF553B",
                },
            )
            # Add trend lines
            for expr in df["expression"].unique():
                sub_df = df[
                    (df["expression"] == expr) & df["benchmarkPct"].notna()
                ]
                if len(sub_df) > 1:  # Need at least 2 points for a line
                    fig_expr_timeline.add_trace(
                        go.Scatter(
                            x=sub_df["testDate"],
                            y=sub_df["benchmarkPct"],
                            mode="lines",
                            name=f"{expr} trend",
                            line=dict(dash="dash", width=1),
                            opacity=0.6,
                        )
                    )
            st.plotly_chart(fig_expr_timeline, use_container_width=True)

        # Tab 2: Movement Type Distribution and Analysis
        with tab2:
            st.subheader("Movement Type Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart of movement types
                fig_movement_pie = px.pie(
                    df,
                    names="movement",
                    title="Distribution of Movement Types",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig_movement_pie.update_traces(
                    textposition="inside", textinfo="percent+label"
                )
                st.plotly_chart(fig_movement_pie, use_container_width=True)

            with col2:
                # Heatmap of movement vs quality
                pivot = pd.crosstab(df["movement"], df["quality"])
                fig_heatmap = px.imshow(
                    pivot,
                    text_auto=True,
                    aspect="auto",
                    title="Movement vs Quality Heatmap",
                    labels=dict(x="Quality", y="Movement", color="Count"),
                    color_continuous_scale="YlGnBu",
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

            # Movement performance comparison
            st.subheader("Performance by Movement Type")
            movement_perf = (
                df.groupby("movement")["benchmarkPct"]
                .agg(["mean", "count", "std"])
                .reset_index()
            )
            movement_perf = movement_perf.sort_values("mean", ascending=False)

            # Only include movements with benchmark data
            movement_perf = movement_perf[movement_perf["count"] > 0].copy()
            movement_perf["count_with_data"] = (
                df.groupby("movement")["benchmarkPct"]
                .count()
                .reindex(movement_perf["movement"])
                .values
            )

            # Calculate error bars
            movement_perf["upper"] = (
                movement_perf["mean"] + movement_perf["std"]
            )
            movement_perf["lower"] = (
                movement_perf["mean"] - movement_perf["std"]
            )

            fig_movement_perf = go.Figure()
            fig_movement_perf.add_trace(
                go.Bar(
                    x=movement_perf["movement"],
                    y=movement_perf["mean"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=movement_perf["upper"] - movement_perf["mean"],
                        arrayminus=movement_perf["mean"]
                        - movement_perf["lower"],
                    ),
                    name="Average Performance",
                    hovertemplate="Movement: %{x}<br>Avg Performance: %{y:.2f}<br>Tests with data: %{text}",
                    text=movement_perf["count_with_data"],
                    marker_color="royalblue",
                )
            )

            fig_movement_perf.update_layout(
                title="Average Performance by Movement Type (with Std Dev)",
                xaxis_title="Movement Type",
                yaxis_title="Average Benchmark Percentile",
                hovermode="x",
            )
            st.plotly_chart(fig_movement_perf, use_container_width=True)

        # Tab 3: Performance Trends Over Time
        with tab3:
            st.subheader("Performance Trends Over Time")

            # Time series of all benchmark percentiles with hover info
            fig_time_series = px.line(
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

            # Add a trend line
            x = (
                df.dropna(subset=["benchmarkPct"])["testDate"]
                .map(lambda x: x.toordinal())
                .values
            )
            y = df.dropna(subset=["benchmarkPct"])["benchmarkPct"].values

            if len(x) >= 2:  # Need at least 2 points for regression
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)

                # Create trend line
                x_range = np.linspace(min(x), max(x), 100)
                fig_time_series.add_trace(
                    go.Scatter(
                        x=[datetime.fromordinal(int(i)) for i in x_range],
                        y=p(x_range),
                        mode="lines",
                        name="Trend",
                        line=dict(color="red", dash="dash"),
                    )
                )

            # Add hover data
            fig_time_series.update_traces(
                hovertemplate="Date: %{x|%d %b %Y}<br>Performance: %{y:.3f}"
            )

            st.plotly_chart(fig_time_series, use_container_width=True)

            # Monthly average performance
            df["month"] = df["testDate"].dt.strftime("%Y-%m")

            monthly_avg = (
                df.groupby("month")["benchmarkPct"].mean().reset_index()
            )
            monthly_count = (
                df.groupby("month")["benchmarkPct"].count().reset_index()
            )
            monthly_data = pd.merge(
                monthly_avg,
                monthly_count,
                on="month",
                suffixes=("_avg", "_count"),
            )
            monthly_data = monthly_data[monthly_data["benchmarkPct_count"] > 0]

            if not monthly_data.empty:
                fig_monthly = px.bar(
                    monthly_data,
                    x="month",
                    y="benchmarkPct_avg",
                    title="Monthly Average Performance",
                    labels={
                        "month": "Month",
                        "benchmarkPct_avg": "Average Benchmark Percentile",
                    },
                    text=monthly_data["benchmarkPct_count"].apply(
                        lambda x: f"{x} tests"
                    ),
                )

                fig_monthly.update_traces(
                    textposition="outside",
                    hovertemplate="Month: %{x}<br>Avg Performance: %{y:.3f}<br>Tests: %{text}",
                )

                st.plotly_chart(fig_monthly, use_container_width=True)

        # Tab 4: Quality Comparison
        with tab4:
            st.subheader("Movement Quality Analysis")

            # Get unique qualities with benchmark data
            qualities_with_data = df.dropna(subset=["benchmarkPct"])[
                "quality"
            ].unique()

    except FileNotFoundError:
        st.error(
            "Physical capabilities data file not found. Please ensure 'physical_capabilities.csv' exists in the directory."
        )
