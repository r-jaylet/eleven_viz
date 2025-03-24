import pandas as pd
import streamlit as st

from src.data_preprocessing import load_recovery_status
from src.recovery_viz import (
    create_category_comparison,
    create_category_completeness_bar,
    create_category_completeness_time,
    create_category_composite_time,
    create_completeness_heatmap,
    create_completeness_patterns,
    create_completeness_radar,
    create_daily_completeness_line,
    create_daily_tracking,
    create_date_metrics_bar,
)


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
                fig_completeness = create_completeness_heatmap(completeness_df)
                st.plotly_chart(fig_completeness, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig_cat_completeness = create_category_completeness_bar(
                    completeness_df
                )
                st.plotly_chart(fig_cat_completeness, use_container_width=True)

            with col2:
                fig_daily_completeness = create_daily_completeness_line(
                    completeness_df
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
                fig_radar = create_completeness_radar(latest_completeness)
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
                    fig_cat_completeness_time = (
                        create_category_completeness_time(
                            completeness_data, selected_category
                        )
                    )
                    st.plotly_chart(
                        fig_cat_completeness_time, use_container_width=True
                    )

            with col2:
                if (
                    not composite_data.empty
                    and not pd.isna(composite_data["value"]).all()
                ):
                    fig_cat_composite_time = create_category_composite_time(
                        composite_data, selected_category
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
                    fig_category_comparison = create_category_comparison(
                        pivot_latest
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
            fig_daily_tracking = create_daily_tracking(daily_stats)
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
                    fig_date_metrics = create_date_metrics_bar(
                        date_values, selected_date
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
                fig_pattern = create_completeness_patterns(completeness_df)
                st.plotly_chart(fig_pattern, use_container_width=True)

            # Correlation heatmap between metrics
            if not df.empty:
                pivot_corr = df.pivot_table(
                    index="sessionDate",
                    columns="metric",
                    values="value",
                    aggfunc="first",
                )

    except FileNotFoundError:
        st.error(
            "Recovery status data file not found. Please ensure 'recovery_status.csv' exists in the directory."
        )
