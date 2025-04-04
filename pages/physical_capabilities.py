import pandas as pd
import streamlit as st

from src.data_preprocessing import load_physical_capabilities
from src.physical_viz import (
    calculate_kpis,
    create_expression_count_chart,
    create_expression_performance_boxplot,
    create_movement_performance_chart,
    create_movement_pie_chart,
    create_performance_trend_chart,
    detailed_stats_by_movement,
    get_data_for_date,
    create_movement_over_time_chart, 
    create_movement_trend_chart
)


def show():
    st.title("Physical Capabilities")

    try:
        # Load data
        df = load_physical_capabilities(
            "data/players_data/marc_cucurella/CFC Physical Capability Data.csv",
            season=st.session_state.selected_season,
        )
        df_filtered = df.dropna(subset=["benchmarkPct"])

        # Key metrics
        kpis = calculate_kpis(df_filtered)

        # Display metrics in a cleaner layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tests Completed", kpis["total_entries"])
        with col2:
            st.metric("Average Benchmark", f"{kpis['average_benchmark']:.1f}%")
        with col3:
            best_movement = max(
                kpis["benchmark_by_movement"].items(), key=lambda x: x[1]
            )[0]
            st.metric("Best Movement", best_movement)

        # Simplified tabs
        tab1, tab2 = st.tabs(["Performance Overview", "Movement Analysis"])

        # PERFORMANCE OVERVIEW TAB
        with tab1:
            # Expression analysis in single row
            st.subheader("Expression Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Count of expressions
                fig_expr_count = create_expression_count_chart(df_filtered)
                st.plotly_chart(fig_expr_count, use_container_width=True)

            with col2:
                # Performance by expression type
                fig_expr_perf = create_expression_performance_boxplot(
                    df_filtered
                )
                st.plotly_chart(fig_expr_perf, use_container_width=True)

            # Clear performance trend chart
            st.subheader("Performance Trend")
            fig_trend = create_performance_trend_chart(df_filtered)
            st.plotly_chart(fig_trend, use_container_width=True)

        # MOVEMENT ANALYSIS TAB
        with tab2:
            st.subheader("Movement Breakdown")

            # Movement distribution and selection
            col1, col2 = st.columns(2)

            with col1:
                # Movement distribution pie chart
                fig_pie = create_movement_over_time_chart(df_filtered)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Movement selector
                movements = sorted(df_filtered["movement"].unique())
                selected_movement = st.selectbox("Select movement", movements)

                if selected_movement:
                    movement_data = df_filtered[
                        df_filtered["movement"] == selected_movement
                    ]
                    avg_benchmark = movement_data["benchmarkPct"].mean()

                    # Large, clear metric display
                    st.markdown(f"### {selected_movement}")
                    st.metric("Average Benchmark", f"{avg_benchmark:.1f}%")

            # Performance by movement - simplified chart
            st.subheader("Movement Performance")
            fig_movement_perf = create_movement_performance_chart(df_filtered)
            st.plotly_chart(fig_movement_perf, use_container_width=True)

            # Performance by movement - over time
            st.subheader("Movement Performance Trends over the last 5 months")
            fig_movement_perf_trend = create_movement_trend_chart(df_filtered)
            st.plotly_chart(fig_movement_perf_trend, use_container_width=True)


            # Specific date data
            st.subheader("Recent Test Results")
            latest_date = df_filtered["testDate"].max()
            selected_date = st.date_input("Select date", value=latest_date)

            date_data = get_data_for_date(df_filtered, date=selected_date)
            if not date_data.empty:
                st.dataframe(
                    date_data[
                        ["movement", "expression", "quality", "benchmarkPct"]
                    ],
                    use_container_width=True,
                )
            else:
                st.info(f"No data available for {selected_date}")

    except Exception as e:
        st.error(f"Error: {e}")
