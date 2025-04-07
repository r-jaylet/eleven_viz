import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    load_physical_capabilities, 
    load_gps
)
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

from src.additional_viz import (
    plot_player_load_vs_expression
)


def show():
    st.title("Physical Capabilities")

    try:
        # Load data
        df_gps, df_active = load_gps(
        "data/players_data/marc_cucurella/CFC GPS Data.csv",
        season=st.session_state.selected_season,
        )

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
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Overview", "Movement Analysis", "Training Load vs Expression Development", "Recent Results"])

        # PERFORMANCE OVERVIEW TAB
        with tab1:
            # Expression analysis in single row
            st.subheader("Expression Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Count of expressions
                st.markdown("""This chart shows the number of recorded tests for each expression type. """)
                fig_expr_count = create_expression_count_chart(df_filtered)
                st.plotly_chart(fig_expr_count, use_container_width=True)

            with col2:
                # Performance by expression type
                st.markdown("""This chart compares the distribution of performance (benchmark percentiles) between the two expression types.""")
                fig_expr_perf = create_expression_performance_boxplot(
                    df_filtered
                )
                st.plotly_chart(fig_expr_perf, use_container_width=True)

            # Clear performance trend chart
            st.subheader("Performance Trend")
            st.markdown("""This graph visualizes the evolution of benchmark performance over time. The grey line represents the daily average benchmark percentile on each test date, while the blue line applies a 3-day rolling average.""")
            fig_trend = create_performance_trend_chart(df_filtered)
            st.plotly_chart(fig_trend, use_container_width=True)

        # MOVEMENT ANALYSIS TAB
        with tab2:
            st.subheader("Movement Breakdown")
            st.markdown(""" 
            This graph provides a monthly breakdown of physical performance tests by movement type, such as agility, jump, sprint, and upper body movements.
            The dropdown allows you to select a specific movement type, displaying the average benchmark percentile for that type on the right.
                """)

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
            st.markdown(""" 
            This chart shows the average benchmark performance by movement type (agility, jump, sprint, upper body), along with error bars indicating variability in the results.
            The height of each bar represents the mean benchmark percentile across all tests of that movement.
            Error bars show one standard deviation above and below the mean, capturing performance consistency.
            The number of tests with available data is shown as a label above each bar.    
            """)
            fig_movement_perf = create_movement_performance_chart(df_filtered)
            st.plotly_chart(fig_movement_perf, use_container_width=True)

            # Performance by movement - over time
            st.subheader("Movement Performance Trends over the last 5 months")
            st.markdown(""" 
            This graph displays movement performance trends over the last five months, showing how the average benchmark percentile has evolved for each movement type: agility, jump, sprint, and upper body.
            """)
            fig_movement_perf_trend = create_movement_trend_chart(df_filtered)
            st.plotly_chart(fig_movement_perf_trend, use_container_width=True)
        
        with tab3: 
            # Load vs expression visualization 
            st.subheader("Plaver Load vs Expression Development")
            st.markdown("""
            This graph visualizes the relationship between a player’s physical training load and their expression development (performance benchmark) over time. It combines three curves:
            - Composite Load (Normalized): A daily average derived from normalized GPS metrics (distance, accelerations, high-speed distance, peak speed).
            - Smoothed Load (7-day Avg): A rolling average of the composite load to highlight weekly load trends.
            - BenchmarkPct (Expression): The player's benchmark performance percentage from capability assessments (e.g. sprint, jump, strength tests).
            This chart helps coaches assess whether increased training loads correlate with improvements in physical capabilities.  """)
            fig_load_vs_expression = plot_player_load_vs_expression(df_gps, df_filtered)
            st.plotly_chart(fig_load_vs_expression, use_container_width=True)

        with tab4:
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
