import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.additional_viz import compute_load, plot_load, plot_load_over_time
from src.data_preprocessing import load_gps
from src.gps_viz import (
    cluster_performance,
    general_kpis,
    plot_accel_decel_intensity_per_match,
    plot_avg_hr_zones,
    plot_cluster,
    plot_distance_distribution_by_duration,
    plot_peak_speed_per_match,
    plot_player_state,
    plot_radar_chart,
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
        tabs = st.tabs(
            [
                "Key Performance Indicators",
                "Match Analysis",
                "Load (Training + Games)",
                "Performance Clustering",
            ]
        )

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

        # MATCH ANALYSIS TAB
        with tabs[1]:

            st.header("Match Analysis : per game")

            # Individual match analysis with radar chart
            st.subheader("Match Details")
            st.markdown(
                "This radar chart displays a comprehensive performance profile for a single match, highlighting a player's physical and cardiovascular output across multiple GPS and heart rate metrics."
            )
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
                    "Select Match Date",
                    available_dates,
                    key="match_date_selectbox",
                )

                # Convert to datetime and filter again!
                selected_date = pd.to_datetime(
                    selected_date_str, format="%d/%m/%Y"
                )
                df_hr_filtered = df_matches[
                    df_matches["date"] == selected_date
                ].copy()

                heart_rate_bar_chart = plot_avg_hr_zones(df_hr_filtered)

                if heart_rate_bar_chart:
                    st.plotly_chart(
                        heart_rate_bar_chart, use_container_width=True
                    )
                else:
                    st.info("No data available for the selected date.")
            else:
                st.info("No matches available.")

            st.header("Match Analysis : overview of all games")

            # Bar chart: Peak speed per match
            st.subheader("Match Overview : Peak Speed")

            peak_speed_chart = plot_peak_speed_per_match(df_matches)

            if peak_speed_chart:
                st.plotly_chart(peak_speed_chart, use_container_width=True)
            else:
                st.info("No peak speed data available for match dates.")

            # Bar chart: Acceleration/Deceleration intensity
            st.subheader(
                "Match Overview : Acceleration & Deceleration Intensity"
            )
            st.markdown(
                "This bar chart shows acceleration and deceleration intensity for each match across the season, categorized by effort thresholds."
            )
            accel_decel_chart = plot_accel_decel_intensity_per_match(
                df_matches
            )

            if accel_decel_chart:
                st.plotly_chart(accel_decel_chart, use_container_width=True)
            else:
                st.info(
                    "No acceleration/deceleration data available for match dates."
                )

            # Simple distribution chart
            st.subheader("Match Overview : Distance by Duration")
            st.markdown(
                """ 
            This graph shows a distribution of total distance covered during matches. 
            It visualizes how far players tend to run depending on how long they play. 
            Each bar represents the count of matches falling into a specific distance range, and the chart overlays mean (dashed) and median (solid) distance markers for each duration group.
            """
            )
            st.plotly_chart(
                plot_distance_distribution_by_duration(df_filtered),
                use_container_width=True,
            )

        # LOAD TAB
        with tabs[2]:
            st.header("Training Load")

            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("Distance Weight", 1.0, 3.0, 2.0, 0.5)
            with col2:
                beta = st.slider("Accel/Decel Weight", 1.0, 3.0, 1.0, 0.5)
            with col3:
                gamma = st.slider("High Speed Weight", 1.0, 3.0, 3.0, 0.5)

            rolling_window = st.slider("Average Window (days)", 3, 14, 7)

            # Calculate load
            df, threshold, df_load_by_date = compute_load(
                df,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_x_percent=5,
            )

            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Load", f"{df_load_by_date['load'].mean():.2f}"
                )
            with col2:
                st.metric("Max Load", f"{df_load_by_date['load'].max():.2f}")
            with col3:
                st.metric("High Load Threshold", f"{threshold:.2f}")

            # Load visualization
            st.subheader("Load Distribution")
            st.markdown(
                "This graph breaks down how training load is related to key physical metrics: total distance, high-speed distance (>21 km/h), and accel/decel efforts (>2.5 m/s²). "
            )
            fig_load = plot_load(df)
            st.plotly_chart(fig_load, use_container_width=True)

            st.subheader("Load Over Time")
            st.markdown(
                """
            This chart tracks the player’s daily training load over time. Training load is calculated using a composite formula that combines distance, high-speed distance, and accel/decel counts, each passed through a sigmoid transformation and weighted by coefficients (α, β, γ). 
            This approach gives a normalized intensity score that reflects both volume and intensity of physical effort across sessions."""
            )
            fig_load_time = plot_load_over_time(
                df_load_by_date,
                rolling_window=rolling_window,
            )
            st.plotly_chart(fig_load_time, use_container_width=True)

        # TRAINING ANALYSIS TAB
        with tabs[3]:
            st.header("Performance Clustering")

            # Simplified clustering
            st.subheader("Performance Clusters")

            st.markdown(
                """ 
            This visualization displays training session performance clusters based on selected physical metrics, using K-Means clustering. 
            Each point represents a training session, categorized into Lower, Usual, or Better performance levels, based on patterns in the data. 
            The clustering is computed using five features: distance, distance_over_24, accel_decel_over_3_5, peak_speed, and hr_zone_4_hms, which are scaled and grouped into three clusters.

            Users can select any two features to explore performance groupings—common and effective combinations include distance vs peak_speed or distance vs accel_decel_over_3_5, as these best reflect both volume and intensity. 
            The plot helps identify outliers, track training quality over time, and guide workload adjustments based on the underlying performance profile.
            """
            )

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
            st.markdown(
                """
            This timeline displays performance clusters over time, categorizing each session into Better, Usual, or Lower performances based on k-means clustering of key physical metrics (described in previous graph's decription). 
            Each point represents a training or match session, with shapes distinguishing the session type and colors indicating the performance level.
            This visualization helps identify periods of high or low performance, track consistency, and detect performance trends across the season.
            """
            )
            seasons = df_combined["season"].unique()
            if len(seasons) > 0:
                selected_season = st.selectbox("Select Season", seasons)
                st.plotly_chart(
                    plot_player_state(df_combined, season=selected_season),
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
