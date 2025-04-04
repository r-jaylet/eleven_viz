import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


from src.data_preprocessing import load_gps
from src.gps_viz import (
    cluster_performance,
    general_kpis,
    plot_average_distances_histogram_plotly,
    plot_cluster,
    plot_distance_distribution_by_duration,
    plot_player_state,
    plot_radar_chart,
    stats_vs_match_time,
    plot_avg_hr_zones, 
    plot_peak_speed_per_match, 
    plot_accel_decel_intensity_per_match
)


def show():
    st.title("GPS Data Exploration")

    try:
        # Load data
        df, df_active = load_gps(
            "data/players_data/marc_cucurella/CFC GPS Data.csv",
            season=st.session_state.selected_season,
        )
        df_filtered = df[df["distance"] > 0]
        df_matches = df_filtered[df_filtered["opposition_code"].notna()]
        df_trainings = df_filtered[df_filtered["opposition_code"].isna()]

        # Main tabs
        tabs = st.tabs(["Overview", "Match Analysis", "Training Analysis"])

        # OVERVIEW TAB
        with tabs[0]:
            st.header("Performance Summary")

            # Key metrics in 2 columns
            col1, col2 = st.columns(2)

            with col1:
                match_kpis = general_kpis(df_matches)
                st.subheader("Match Statistics")
                st.metric(
                    "Total Matches", f"{int(match_kpis['number_of_sessions'])}"
                )
                st.metric(
                    "Total Distance",
                    f"{match_kpis['total_distance_km']:.1f} km",
                )
                st.metric(
                    "Peak Speed",
                    f"{match_kpis['average_peak_speed']:.1f} km/h",
                )

            with col2:
                training_kpis = general_kpis(df_trainings)
                st.subheader("Training Statistics")
                st.metric(
                    "Total Sessions",
                    f"{int(training_kpis['number_of_sessions'])}",
                )
                st.metric(
                    "Total Distance",
                    f"{training_kpis['total_distance_km']:.1f} km",
                )
                st.metric(
                    "Peak Speed",
                    f"{training_kpis['average_peak_speed']:.1f} km/h",
                )

            # Simple distribution chart
            st.subheader("Distance by Duration")
            st.plotly_chart(
                plot_distance_distribution_by_duration(df_filtered),
                use_container_width=True,
            )

        # MATCH ANALYSIS TAB
        with tabs[1]:
            st.header("Match Performance")

            # Performance by match time - simplified to one metric at a time
            metric_option = st.selectbox(
                "Select metric",
                [
                    "Distance Coverage",
                    "Acceleration/Deceleration",
                    "Heart Rate Zones",
                ],
            )

            """
            fig_distance, fig_accel, fig_hr = stats_vs_match_time(df_matches)

            if metric_option == "Distance Coverage":
                st.plotly_chart(fig_distance, use_container_width=True)
            elif metric_option == "Acceleration/Deceleration":
                st.plotly_chart(fig_accel, use_container_width=True)
            else:
                st.plotly_chart(fig_hr, use_container_width=True)
            """
            # Individual match analysis with radar chart
            st.subheader("Match Details")
            available_dates = (
                df_matches["date"].dt.strftime("%d/%m/%Y").unique()
            )

            if len(available_dates) > 0:
                selected_date = st.selectbox(
                    "Select Match Date", available_dates
                )
                radar_chart = plot_radar_chart(df_filtered, selected_date)

                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True)
                else:
                    st.info("No data available for the selected date.")
            else:
                st.info("No matches available.")

            # Bar chart : average time spent in each heart rate zone for the selected date : 
            st.subheader("Match Details : Heart Rate Zones")
            available_dates = (
                df_matches["date"].dt.strftime("%d/%m/%Y").unique()
            )

            if len(available_dates) > 0:
                selected_date_str = st.selectbox(
                    "Select Match Date", available_dates, key="match_date_selectbox"
                )

                # Convert to datetime and filter again!
                selected_date = pd.to_datetime(selected_date_str, format="%d/%m/%Y")
                df_hr_filtered = df_matches[df_matches["date"] == selected_date].copy()

                heart_rate_bar_chart = plot_avg_hr_zones(df_hr_filtered)

                if heart_rate_bar_chart:
                    st.plotly_chart(heart_rate_bar_chart, use_container_width=True)
                else:
                    st.info("No data available for the selected date.")
            else:
                st.info("No matches available.")

            # Bar chart: Peak speed per match
            st.subheader("Match Details : Peak Speed")

            peak_speed_chart = plot_peak_speed_per_match(df_matches)

            if peak_speed_chart:
                st.plotly_chart(peak_speed_chart, use_container_width=True)
            else:
                st.info("No peak speed data available for match dates.")
            
            # Bar chart: Acceleration/Deceleration intensity
            st.subheader("Match Overview : Acceleration & Deceleration Intensity")

            accel_decel_chart = plot_accel_decel_intensity_per_match(df_matches)

            if accel_decel_chart:
                st.plotly_chart(accel_decel_chart, use_container_width=True)
            else:
                st.info("No acceleration/deceleration data available for match dates.")

        # TRAINING ANALYSIS TAB
        with tabs[2]:
            st.header("Training & Performance Clusters")

            # Training distribution
            st.subheader("Training Distance")
            st.plotly_chart(
                plot_distance_distribution_by_duration(df_trainings),
                use_container_width=True,
            )

            # Simplified clustering
            st.subheader("Performance Clusters")

            # Select only essential features for clustering
            cluster_features = [
                "distance",
                "distance_over_24",
                "accel_decel_over_3_5",
                "peak_speed",
                "hr_zone_4_hms",
            ]

            # Apply clustering with fewer features
            df_training_with_clusters, df_matches_with_clusters = (
                cluster_performance(df_trainings, df_matches, cluster_features)
            )

            # Combine for timeline analysis
            df_matches_with_clusters["type"] = "Match"
            df_training_with_clusters["type"] = "Training"
            df_combined = pd.concat(
                [
                    df_matches_with_clusters[
                        ["date", "season", "cluster_label", "type"]
                    ],
                    df_training_with_clusters[
                        ["date", "season", "cluster_label", "type"]
                    ],
                ]
            ).sort_values(by="date")

            # Simple feature selection
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox(
                    "X-axis feature", cluster_features, index=0
                )
            with col2:
                y_feature = st.selectbox(
                    "Y-axis feature", cluster_features, index=3
                )

            # Plot clusters
            st.plotly_chart(
                plot_cluster(df_training_with_clusters, x_feature, y_feature),
                use_container_width=True,
            )

            # Performance timeline
            st.subheader("Performance Timeline")
            seasons = df_combined["season"].unique()
            if len(seasons) > 0:
                selected_season = st.selectbox("Select Season", seasons)
                st.plotly_chart(
                    plot_player_state(df_combined, season=selected_season),
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
