import pandas as pd
import streamlit as st

from src.data_preprocessing import load_gps
from src.gps_viz import (
    plot_accel_decel,
    plot_daily_distance,
    plot_distance_by_matchday,
    plot_heart_rate_zones,
    plot_high_speed_distance,
    plot_peak_speed,
    plot_speed_percentage,
    plot_speed_zones,
    plot_weekly_distance,
)


def show():

    try:
        df, df_active = load_gps()

        # Dashboard title
        st.title("Player GPS Performance Dashboard")
        st.markdown("### Training and Match Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Distance Analysis",
                "Speed Metrics",
                "Heart Rate Analysis",
                "Week Overview",
            ]
        )

        # Tab 1: Distance Analysis
        with tab1:
            st.subheader("Distance Metrics Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Total distance over time
                fig_dist = plot_daily_distance(df_active)
                st.plotly_chart(fig_dist, use_container_width=True)

            with col2:
                # Distance by match day code
                fig_md_dist = plot_distance_by_matchday(df_active)
                st.plotly_chart(fig_md_dist, use_container_width=True)

            # High-speed distance analysis
            st.subheader("High-Speed Distance Analysis")
            fig_high_speed = plot_high_speed_distance(df_active)
            st.plotly_chart(fig_high_speed, use_container_width=True)

            # Speed distance percentage by day
            col1, col2 = st.columns(2)

            with col1:
                # Calculate percentages
                fig_pct = plot_speed_percentage(df_active)
                st.plotly_chart(fig_pct, use_container_width=True)

            with col2:
                # Acceleration/deceleration analysis
                fig_accel = plot_accel_decel(df_active)
                st.plotly_chart(fig_accel, use_container_width=True)

        # Tab 2: Speed Metrics
        with tab2:
            st.subheader("Speed Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Peak speed by day
                fig_peak = plot_peak_speed(df_active)
                st.plotly_chart(fig_peak, use_container_width=True)

            with col2:
                # Speed zones distribution
                fig_zones = plot_speed_zones(df_active)
                st.plotly_chart(fig_zones, use_container_width=True)

        # Tab 3: Heart Rate Analysis
        with tab3:
            st.subheader("Heart Rate Zone Analysis")
            fig_hr = plot_heart_rate_zones(df_active)

            if fig_hr is not None:
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.warning("No heart rate data available for visualization.")

        # Tab 4: Week Overview
        with tab4:
            st.subheader("Weekly Load Analysis")
            fig_weekly_dist = plot_weekly_distance(df)
            st.plotly_chart(fig_weekly_dist, use_container_width=True)

    except FileNotFoundError:
        st.error("GPS data file not found.")
