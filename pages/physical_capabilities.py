from datetime import datetime

import streamlit as st

from src.data_preprocessing import load_physical_capabilities
from src.physical_viz import (
    create_expression_count_chart,
    create_expression_performance_boxplot,
    create_expression_timeline,
    create_monthly_performance_chart,
    create_movement_performance_chart,
    create_movement_pie_chart,
    create_movement_quality_heatmap,
    create_performance_trend_chart,
)


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
                fig_expr_count = create_expression_count_chart(df)
                st.plotly_chart(fig_expr_count, use_container_width=True)

            with col2:
                # Performance by expression type (box plot)
                fig_expr_perf = create_expression_performance_boxplot(df)
                st.plotly_chart(fig_expr_perf, use_container_width=True)

            # Scatter plot showing all tests with benchmark values by expression
            st.subheader("Performance Timeline by Expression Type")
            fig_expr_timeline = create_expression_timeline(df)
            st.plotly_chart(fig_expr_timeline, use_container_width=True)

        # Tab 2: Movement Type Distribution and Analysis
        with tab2:
            st.subheader("Movement Type Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Pie chart of movement types
                fig_movement_pie = create_movement_pie_chart(df)
                st.plotly_chart(fig_movement_pie, use_container_width=True)

            with col2:
                # Heatmap of movement vs quality
                fig_heatmap = create_movement_quality_heatmap(df)
                st.plotly_chart(fig_heatmap, use_container_width=True)

            # Movement performance comparison
            st.subheader("Performance by Movement Type")
            fig_movement_perf = create_movement_performance_chart(df)
            st.plotly_chart(fig_movement_perf, use_container_width=True)

        # Tab 3: Performance Trends Over Time
        with tab3:
            st.subheader("Performance Trends Over Time")

            # Time series of all benchmark percentiles with hover info
            fig_time_series = create_performance_trend_chart(df)
            st.plotly_chart(fig_time_series, use_container_width=True)

            # Monthly average performance
            fig_monthly = create_monthly_performance_chart(df)
            if fig_monthly:
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
