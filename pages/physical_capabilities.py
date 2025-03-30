import pandas as pd
import streamlit as st

from src.data_preprocessing import load_physical_capabilities
from src.physical_viz import (
    calculate_kpis,
    create_expression_count_chart,
    create_expression_performance_boxplot,
    create_expression_timeline,
    create_monthly_performance_chart,
    create_movement_performance_chart,
    create_movement_pie_chart,
    create_performance_trend_chart,
    detailed_stats_by_movement,
    get_data_for_date,
)


def print_data_for_date(df, date):
    date_data = get_data_for_date(df, date)
    if not date_data.empty:
        st.subheader(f"Performance Data for {date}")
        st.dataframe(
            date_data[["movement", "expression", "quality", "benchmarkPct"]],
            use_container_width=True,
        )
    else:
        st.info(f"No data available for {date}")


def show():
    st.title("Physical Capabilities Analysis")
    st.markdown("### Athlete Performance Metrics")

    try:
        df = load_physical_capabilities(
            "data/players_data/marc_cucurella/CFC Physical Capability Data.csv",
        )
        df_filtered = df.dropna(subset=["benchmarkPct"])

        kpis = calculate_kpis(df_filtered)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", kpis["total_entries"])
        with col2:
            st.metric("Avg Benchmark", f"{kpis['average_benchmark']:.1f}%")
        with col3:
            st.metric(
                "Best Movement",
                max(kpis["benchmark_by_movement"].items(), key=lambda x: x[1])[
                    0
                ],
            )

        st.subheader("Performance Trends")
        tabs = st.tabs(["Timeline", "Monthly", "By Movement", "By Expression"])

        with tabs[0]:
            fig_trend = create_performance_trend_chart(df_filtered)
            st.plotly_chart(fig_trend, use_container_width=True)

        with tabs[1]:
            fig_monthly = create_monthly_performance_chart(df_filtered)
            if fig_monthly:
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("Not enough data for monthly visualization")

        with tabs[2]:
            fig_movement_perf = create_movement_performance_chart(df_filtered)
            st.plotly_chart(fig_movement_perf, use_container_width=True)

        with tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                fig_expr_count = create_expression_count_chart(df_filtered)
                st.plotly_chart(fig_expr_count, use_container_width=True)
            with col2:
                fig_expr_perf = create_expression_performance_boxplot(
                    df_filtered
                )
                st.plotly_chart(fig_expr_perf, use_container_width=True)

            fig_expr_timeline = create_expression_timeline(df_filtered)
            st.plotly_chart(fig_expr_timeline, use_container_width=True)

        st.subheader("Movement Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = create_movement_pie_chart(df_filtered)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("##### Movement Breakdown")
            movements = sorted(df_filtered["movement"].unique())
            selected_movement = st.selectbox("Select movement", movements)
            if selected_movement:
                movement_data = df_filtered[
                    df_filtered["movement"] == selected_movement
                ]
                st.markdown(
                    f"Average benchmark: **{movement_data['benchmarkPct'].mean():.1f}%**"
                )

        st.subheader("Detailed Movement Statistics")
        fig_movement = detailed_stats_by_movement(df_filtered)
        st.plotly_chart(fig_movement, use_container_width=True)

        st.subheader("Data for Specific Date")
        selected_date = st.date_input(
            "Select date", value=pd.to_datetime("2024-01-01")
        )
        print_data_for_date(df_filtered, date=selected_date)

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
