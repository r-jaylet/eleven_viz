import pandas as pd
import streamlit as st

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
)


def show():
    st.title("GPS Data Exploration")

    try:
        # Load and filter data
        df = load_gps("data/players_data/marc_cucurella/CFC GPS Data.csv")
        df_filtered = df[df["distance"] > 0]
        df_matches = df_filtered[df_filtered["opposition_code"].notna()]
        df_trainings = df_filtered[df_filtered["opposition_code"].isna()]

        # Date selector
        min_date = df_filtered["date"].min().date()
        max_date = df_filtered["date"].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", min_date, min_value=min_date, max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date", max_date, min_value=min_date, max_value=max_date
            )

        # Filter data by selected date range
        date_mask = (df_filtered["date"].dt.date >= start_date) & (
            df_filtered["date"].dt.date <= end_date
        )
        df_filtered_date = df_filtered[date_mask]
        df_matches_date = df_matches[date_mask]
        df_trainings_date = df_trainings[date_mask]

        # Main tabs
        overview_tab, match_tab, training_tab, cluster_tab = st.tabs(
            ["Overview", "Matches", "Training", "Performance Clusters"]
        )

        # Overview tab
        with overview_tab:
            st.header("Overall Performance Metrics")

            # Display KPIs
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Match Statistics")
                match_kpis = general_kpis(df_matches_date)
                st.metric(
                    "Total Sessions",
                    f"{int(match_kpis['number_of_sessions'])}",
                )
                st.metric(
                    "Total Distance",
                    f"{match_kpis['total_distance_km']:.1f} km",
                )
                st.metric(
                    "Avg Peak Speed",
                    f"{match_kpis['average_peak_speed']:.1f} km/h",
                )

            with col2:
                st.subheader("Training Statistics")
                training_kpis = general_kpis(df_trainings_date)
                st.metric(
                    "Total Sessions",
                    f"{int(training_kpis['number_of_sessions'])}",
                )
                st.metric(
                    "Total Distance",
                    f"{training_kpis['total_distance_km']:.1f} km",
                )
                st.metric(
                    "Avg Peak Speed",
                    f"{training_kpis['average_peak_speed']:.1f} km/h",
                )

            st.subheader("Distance Distribution by Duration")
            st.plotly_chart(
                plot_distance_distribution_by_duration(df_filtered_date),
                use_container_width=True,
            )

            st.subheader("Recovery Analysis")
            st.plotly_chart(
                plot_average_distances_histogram_plotly(df_filtered_date),
                use_container_width=True,
            )

        # Match tab
        with match_tab:
            st.header("Match Performance Analysis")

            # Performance by match time
            st.subheader("Performance Metrics by Match Duration")

            # Get all three figures from stats_vs_match_time
            fig_distance, fig_accel, fig_hr = stats_vs_match_time(
                df_matches_date
            )

            metric_option = st.selectbox(
                "Select metric to display",
                [
                    "Distance Coverage",
                    "Acceleration/Deceleration",
                    "Heart Rate Zones",
                ],
            )

            if metric_option == "Distance Coverage":
                st.plotly_chart(fig_distance, use_container_width=True)
            elif metric_option == "Acceleration/Deceleration":
                st.plotly_chart(fig_accel, use_container_width=True)
            else:
                st.plotly_chart(fig_hr, use_container_width=True)

            # Specific match analysis
            st.subheader("Individual Match Analysis")

            available_dates = (
                df_matches_date["date"].dt.strftime("%d/%m/%Y").unique()
            )
            if len(available_dates) > 0:
                selected_date = st.selectbox(
                    "Select Match Date", available_dates
                )
                radar_chart = plot_radar_chart(df_filtered_date, selected_date)
                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True)
                else:
                    st.warning("No data available for the selected date.")
            else:
                st.warning("No matches available in the selected date range.")

        # Training tab
        with training_tab:
            st.header("Training Performance Analysis")

            # Add training specific visualizations here
            st.subheader("Training Distance Distribution")
            st.plotly_chart(
                plot_distance_distribution_by_duration(df_trainings_date),
                use_container_width=True,
            )

        # Cluster tab
        with cluster_tab:
            st.header("Performance Clustering Analysis")

            # Clustering features
            st.subheader("Performance Clusters")

            cluster_features = [
                "distance",
                "distance_over_21",
                "distance_over_24",
                "distance_over_27",
                "accel_decel_over_2_5",
                "accel_decel_over_3_5",
                "accel_decel_over_4_5",
                "peak_speed",
                "hr_zone_1_hms",
                "hr_zone_2_hms",
                "hr_zone_3_hms",
                "hr_zone_4_hms",
                "hr_zone_5_hms",
            ]

            # Apply clustering
            df_training_with_clusters, df_matches_with_clusters = (
                cluster_performance(
                    df_trainings_date,
                    df_matches_date,
                    cluster_features,
                )
            )

            # Prepare combined DataFrame for player state visualization
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

            # Select features for scatter plot
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox(
                    "Select X-axis feature", cluster_features, index=0
                )
            with col2:
                y_feature = st.selectbox(
                    "Select Y-axis feature", cluster_features, index=7
                )  # peak_speed as default

            st.plotly_chart(
                plot_cluster(df_training_with_clusters, x_feature, y_feature),
                use_container_width=True,
            )

            # Player state over time
            st.subheader("Performance States Over Time")

            # Season selector
            seasons = df_combined["season"].unique()
            selected_season = st.selectbox("Select Season", seasons)

            st.plotly_chart(
                plot_player_state(df_combined, season=selected_season),
                use_container_width=True,
            )

            # Time period selector
            st.subheader("Custom Time Period Analysis")
            col1, col2 = st.columns(2)
            with col1:
                custom_start = st.date_input(
                    "Custom Start Date",
                    value=df_combined["date"].min().date(),
                    min_value=df_combined["date"].min().date(),
                    max_value=df_combined["date"].max().date(),
                )
            with col2:
                custom_end = st.date_input(
                    "Custom End Date",
                    value=df_combined["date"].max().date(),
                    min_value=df_combined["date"].min().date(),
                    max_value=df_combined["date"].max().date(),
                )

            # Convert to string format required by the function
            custom_start_str = custom_start.strftime("%d/%m/%Y")
            custom_end_str = custom_end.strftime("%d/%m/%Y")

            custom_view = plot_player_state(
                df_combined,
                start_date=custom_start_str,
                end_date=custom_end_str,
            )

            if custom_view:
                st.plotly_chart(custom_view, use_container_width=True)
            else:
                st.warning("No data available for the selected time period.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
