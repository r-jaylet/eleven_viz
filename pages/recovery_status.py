from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import load_recovery_status


def show():
    try:

        df = load_recovery_status()
        st.title("Athlete Recovery Status Dashboard")
        st.markdown("### Recovery Metrics and Baseline Analysis")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Completeness Analysis",
                "Category Overview",
                "Daily Tracking",
                "Data Patterns",
            ]
        )

        # Tab 1: Completeness Analysis
        with tab1:
            st.subheader("Recovery Assessments Completeness Analysis")

            # Filter for completeness metrics
            completeness_df = df[df["metric_type"] == "completeness"].copy()

            # Pivot table for heatmap
            if not completeness_df.empty:
                pivot_completeness = completeness_df.pivot_table(
                    index="sessionDate",
                    columns="category",
                    values="value",
                    aggfunc="first",
                ).fillna(0)

                # Create a completeness heatmap by date and category
                fig_completeness = px.imshow(
                    pivot_completeness,
                    labels=dict(x="Category", y="Date", color="Completeness"),
                    x=pivot_completeness.columns,
                    y=pivot_completeness.index.strftime("%d %b"),
                    color_continuous_scale="YlGnBu",
                    title="Assessment Completeness by Category and Date",
                )

                fig_completeness.update_layout(
                    height=400,
                    xaxis_title="Assessment Category",
                    yaxis_title="Session Date",
                )

                st.plotly_chart(fig_completeness, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Category completeness bar chart
                category_completeness = (
                    completeness_df.groupby("category")["value"]
                    .mean()
                    .reset_index()
                )

                fig_cat_completeness = px.bar(
                    category_completeness,
                    x="category",
                    y="value",
                    title="Average Completeness by Category",
                    labels={
                        "value": "Average Completeness",
                        "category": "Category",
                    },
                    color="category",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )

                fig_cat_completeness.update_layout(showlegend=False)
                st.plotly_chart(fig_cat_completeness, use_container_width=True)

            with col2:
                # Daily completeness line chart
                daily_completeness = (
                    completeness_df.groupby("sessionDate")["value"]
                    .mean()
                    .reset_index()
                )

                fig_daily_completeness = px.line(
                    daily_completeness,
                    x="sessionDate",
                    y="value",
                    title="Daily Average Completeness",
                    labels={
                        "value": "Average Completeness",
                        "sessionDate": "Date",
                    },
                    markers=True,
                )

                st.plotly_chart(
                    fig_daily_completeness, use_container_width=True
                )

            # Radar chart of completeness by category
            latest_date = completeness_df["sessionDate"].max()
            latest_completeness = completeness_df[
                completeness_df["sessionDate"] == latest_date
            ]

            if not latest_completeness.empty:
                st.subheader(
                    f"Latest Completeness Overview ({latest_date.strftime('%d %b %Y')})"
                )

                fig_radar = go.Figure()

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=latest_completeness["value"],
                        theta=latest_completeness["category"],
                        fill="toself",
                        name="Completeness",
                    )
                )

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Latest Completeness by Category",
                    showlegend=False,
                )

                st.plotly_chart(fig_radar, use_container_width=True)

        # Tab 2: Category Overview
        with tab2:
            st.subheader("Recovery Categories Overview")

            # Filter for composite metrics (these would contain actual values)
            composite_df = df[df["metric_type"] == "composite"].copy()

            # Add EMBOSS score if available
            emboss_df = df[df["metric"] == "emboss_baseline_score"]

            # Create a category selector
            all_categories = df["category"].unique()
            selected_category = st.selectbox(
                "Select Category to Analyze:", all_categories
            )

            # Filter data for the selected category
            category_data = df[df["category"] == selected_category]

            # Show completeness vs composite for selected category
            completeness_data = category_data[
                category_data["metric_type"] == "completeness"
            ].copy()
            composite_data = category_data[
                category_data["metric_type"] == "composite"
            ].copy()

            col1, col2 = st.columns(2)

            with col1:
                if not completeness_data.empty:
                    fig_cat_completeness_time = px.line(
                        completeness_data,
                        x="sessionDate",
                        y="value",
                        title=f"{selected_category} Completeness Over Time",
                        labels={
                            "value": "Completeness",
                            "sessionDate": "Date",
                        },
                        markers=True,
                        line_shape="linear",
                    )

                    fig_cat_completeness_time.update_layout(
                        yaxis_range=[0, 1.1]
                    )
                    st.plotly_chart(
                        fig_cat_completeness_time, use_container_width=True
                    )

            with col2:
                if (
                    not composite_data.empty
                    and not pd.isna(composite_data["value"]).all()
                ):
                    fig_cat_composite_time = px.line(
                        composite_data,
                        x="sessionDate",
                        y="value",
                        title=f"{selected_category} Composite Score Over Time",
                        labels={
                            "value": "Composite Score",
                            "sessionDate": "Date",
                        },
                        markers=True,
                        line_shape="linear",
                    )

                    st.plotly_chart(
                        fig_cat_composite_time, use_container_width=True
                    )
                else:
                    st.info(
                        f"No composite score data available for {selected_category}"
                    )

            # Category breakdown
            st.subheader(f"All Categories Overview")

            # Get the latest data for each category and metric type
            latest_data = (
                df.sort_values("sessionDate")
                .groupby(["category", "metric_type"])
                .last()
                .reset_index()
            )

            if not latest_data.empty:
                # Pivot to get completeness and composite side by side
                pivot_latest = latest_data.pivot_table(
                    index="category", columns="metric_type", values="value"
                ).reset_index()

                # Only keep categories with some data
                pivot_latest = pivot_latest.dropna(
                    how="all", subset=["completeness", "composite"]
                ).fillna(0)

                if (
                    not pivot_latest.empty
                    and "completeness" in pivot_latest.columns
                ):
                    # Create a comparison bar chart
                    fig_category_comparison = go.Figure()

                    if "completeness" in pivot_latest.columns:
                        fig_category_comparison.add_trace(
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
                        # Normalize composite scores for comparison
                        if pivot_latest["composite"].max() > 0:
                            normalized_composite = (
                                pivot_latest["composite"]
                                / pivot_latest["composite"].max()
                            )

                            fig_category_comparison.add_trace(
                                go.Bar(
                                    x=pivot_latest["category"],
                                    y=normalized_composite,
                                    name="Normalized Composite Score",
                                    marker_color="firebrick",
                                )
                            )

                    fig_category_comparison.update_layout(
                        title="Latest Category Metrics Comparison",
                        xaxis_title="Category",
                        yaxis_title="Value",
                        barmode="group",
                        legend_title="Metric Type",
                    )

                    st.plotly_chart(
                        fig_category_comparison, use_container_width=True
                    )

        # Tab 3: Daily Tracking
        with tab3:
            st.subheader("Daily Recovery Status Tracking")

            # Create a calendar view
            all_dates = df["sessionDate"].dt.date.unique()

            # Get min and max dates
            min_date = df["sessionDate"].min()
            max_date = df["sessionDate"].max()

            # Group data by date
            daily_stats = (
                df.groupby("sessionDate")["value"]
                .agg(["mean", "count"])
                .reset_index()
            )
            daily_stats["mean"] = daily_stats["mean"].fillna(0)

            # Create a line chart with markers for daily tracking
            fig_daily_tracking = px.scatter(
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

            # Add connecting lines
            fig_daily_tracking.add_trace(
                go.Scatter(
                    x=daily_stats["sessionDate"],
                    y=daily_stats["mean"],
                    mode="lines",
                    line=dict(color="grey", width=1),
                    showlegend=False,
                )
            )

            fig_daily_tracking.update_layout(
                height=400,
                xaxis_title="Session Date",
                yaxis_title="Average Recovery Value",
            )

            st.plotly_chart(fig_daily_tracking, use_container_width=True)

            # Daily details
            selected_date = st.selectbox(
                "Select a date to view detailed metrics:",
                options=sorted(all_dates, reverse=True),
                format_func=lambda x: x.strftime("%d %b %Y"),
            )

            # Filter data for selected date
            date_data = df[df["sessionDate"].dt.date == selected_date]

            if not date_data.empty:
                st.subheader(
                    f"Recovery Metrics for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"
                )

                # Create a bar chart for all metrics on that day
                date_values = date_data.dropna(subset=["value"]).copy()

                if not date_values.empty:
                    fig_date_metrics = px.bar(
                        date_values,
                        x="metric",
                        y="value",
                        color="category",
                        title=f"All Recovery Metrics for {pd.to_datetime(selected_date).strftime('%d %b %Y')}",
                        labels={
                            "value": "Value",
                            "metric": "Metric",
                            "category": "Category",
                        },
                        hover_data=["metric_type"],
                    )

                    fig_date_metrics.update_layout(
                        height=500,
                        xaxis_tickangle=-45,
                        xaxis_title="",
                        barmode="group",
                    )

                    st.plotly_chart(fig_date_metrics, use_container_width=True)

                # Display raw data in a table
                st.subheader("Raw Data")

                display_df = date_data[
                    ["metric", "category", "value", "metric_type"]
                ].copy()
                display_df = display_df.sort_values(
                    ["category", "metric_type"]
                )
                display_df.columns = [
                    "Metric",
                    "Category",
                    "Value",
                    "Metric Type",
                ]

                st.dataframe(display_df, use_container_width=True)

        # Tab 4: Data Patterns
        with tab4:
            st.subheader("Recovery Data Patterns Analysis")

            # Completeness over time by category
            completeness_df = df[df["metric_type"] == "completeness"].copy()

            if not completeness_df.empty:
                fig_pattern = px.line(
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

                fig_pattern.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Completeness Value",
                    legend_title="Category",
                )

                st.plotly_chart(fig_pattern, use_container_width=True)

            # Correlation heatmap between metrics
            # First, pivot the data to have metrics as columns
            if not df.empty:
                pivot_corr = df.pivot_table(
                    index="sessionDate",
                    columns="metric",
                    values="value",
                    aggfunc="first",
                )

                # Calculate correlation
                corr_matrix = pivot_corr.corr(min_periods=1)

                # Drop columns and rows with all NaN
                corr_matrix = corr_matrix.dropna(how="all").dropna(
                    how="all", axis=1
                )

                if (
                    not corr_matrix.empty
                    and corr_matrix.shape[0] > 1
                    and corr_matrix.shape[1] > 1
                ):
                    # Create correlation heatmap
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(
                            x="Metric", y="Metric", color="Correlation"
                        ),
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        color_continuous_scale="RdBu_r",
                        range_color=[-1, 1],
                        title="Correlation Between Recovery Metrics",
                    )

                    fig_corr.update_layout(height=600, xaxis_tickangle=-45)

                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info(
                        "Not enough data to calculate meaningful correlations between metrics."
                    )

            # Add a composite metrics analysis section if there's data
            composite_df = df[df["metric_type"] == "composite"].copy()

            if (
                not composite_df.empty
                and not pd.isna(composite_df["value"]).all()
            ):
                st.subheader("Composite Metrics Analysis")

                # Create composite metrics visualization
                fig_composite = px.line(
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

                fig_composite.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Composite Score",
                    legend_title="Category",
                )

                st.plotly_chart(fig_composite, use_container_width=True)

        # Summary metrics at the bottom
        st.markdown("---")

        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_assessments = df.dropna(subset=["value"])[
                "sessionDate"
            ].nunique()
            st.metric("Total Assessment Days", total_assessments)

        with col2:
            avg_completeness = df[df["metric_type"] == "completeness"][
                "value"
            ].mean()
            st.metric("Avg Completeness", f"{avg_completeness:.2f}")

        with col3:
            categories_tracked = df["category"].nunique()
            st.metric("Categories Tracked", categories_tracked)

        with col4:
            latest_date = df["sessionDate"].max()
            days_since_last = (datetime.now() - latest_date).days
            st.metric("Days Since Last Assessment", days_since_last)

    except FileNotFoundError:
        st.error(
            "Recovery status data file not found. Please ensure 'recovery_status.csv' exists in the directory."
        )
