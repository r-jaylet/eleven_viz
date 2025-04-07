import streamlit as st

from src.data_preprocessing import(
    load_gps, 
    load_recovery_status, 
    load_physical_capabilities
)

from src.recovery_viz import (
    create_category_completeness_bar,
    create_completeness_radar,
    create_composite_line,
    create_correlation_heatmap,
    plot_global_recovery_score,
    plot_recovery_metrics_by_category,
    plot_weekly_recovery_heatmap
)

from src.additional_viz import (
    compute_load_vs_recovery,
    plot_load_vs_recovery,
    plot_weekly_danger,
    plot_player_load_vs_expression, 
    plot_pre_post_match_recovery_dumbbell, 
    plot_recovery_vs_load_peaks, 
    plot_readiness_snapshot_before_matches
)


def show():
    st.title("Recovery Status")

    try:
        # Load data
        df_gps, df_active = load_gps(
            "data/players_data/marc_cucurella/CFC GPS Data.csv",
            season=st.session_state.selected_season,
        )

        df_physical = load_physical_capabilities(
            "data/players_data/marc_cucurella/CFC Physical Capability Data.csv",
        )

        df = load_recovery_status(
            "data/players_data/marc_cucurella/CFC Recovery status Data.csv",
            season=st.session_state.selected_season,
        )

        # Simple date filter in sidebar
        min_date = df["sessionDate"].min().date()
        max_date = df["sessionDate"].max().date()

        start_date = st.sidebar.date_input("From", min_date)
        end_date = st.sidebar.date_input("To", max_date)

        # Filter data
        date_filtered_df = df[
            (df["sessionDate"].dt.date >= start_date)
            & (df["sessionDate"].dt.date <= end_date)
        ]

        # Convert dates for visualization functions
        start_date_str = start_date.strftime("%d/%m/%Y")
        end_date_str = end_date.strftime("%d/%m/%Y")

        # Main tabs
        tabs = st.tabs(["Recovery Analysis & Load Dynamics", "Match & Recovery (pre-match & post-match)", "Category Breakdown", "Weekly Patterns"])

        # Recovery Analysis & Load Dynamics
        with tabs[0]:
    
            # Global recovery score
            st.subheader("Recovery Summary")
            st.markdown(""" 
            This graph shows the evolution of the global recovery score over time, combining daily recovery values and a 7-day moving average. 
            The blue line with markers represents the raw emboss_baseline_score collected each day, while the bold orange line smooths these fluctuations.
            """)
            fig = plot_global_recovery_score(df)
            st.plotly_chart(fig, use_container_width=True)

            
            # Recovery metrics by category - simplified view
            st.subheader("Recovery by Category")
            st.markdown(""" 
            This multi-panel figure shows time series plots of recovery data by category, highlighting both data completeness (solid lines) and composite scores (dotted lines) over time. 
                        Each subplot corresponds to a specific recovery category like sleep, soreness, or subjective
            """)
            fig = plot_recovery_metrics_by_category(
                df, start_date_str, end_date_str
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recovery vs load
            st.subheader("Load vs Recovery")

            st.markdown("""
            This graph shows the relationship between player training load, recovery score, and a computed risk index (in red) over time.
            Load is calculated using a weighted formula that includes distance, high-speed running, and acceleration/deceleration events from GPS data.
            Recovery is derived from scores like emboss_baseline_score, averaged using a rolling window.
            Risk captures when a player might be under too much physical stress without enough recovery time, it is calculated as: (load / max_load) * (1 - recovery / max_recovery)
            """
            )

            window_size = st.slider("Recovery Window", 5, 10, 7)

            merged_df = compute_load_vs_recovery(
                df_recovery=df,
                df_gps=df_gps,
                window_size=window_size,
            )

            fig_load_vs_recovery = plot_load_vs_recovery(merged_df)
            st.plotly_chart(fig_load_vs_recovery, use_container_width=True)

            # Recovery score vs training load peaks
            st.subheader("Recovery Score vs Training Load Peaks")

            st.markdown(
                """
                This visualization compares the rolling training load (area in dark blue) with the player's daily recovery scores (yellow line and dots). It helps identify whether spikes in load are followed by dips in recovery quality.
                **Rolling Load** is calculated as a composite of GPS metrics (distance, acceleration/deceleration, peak speed, etc.), averaged over a selectable rolling window (e.g., 7 days).
                **Recovery Score** comes from selected wellness metrics like soreness_baseline_composite, sleep_baseline_composite, etc.
                You can customize the recovery metric and the load smoothing window to explore different trends.

                """
            )

            # Recovery metric selector
            recovery_metric = st.selectbox(
                "Select Recovery Metric",
                options=[
                    "soreness_baseline_composite",
                    "sleep_baseline_composite",
                    "subjective_baseline_composite"
                ],
                index=0,
            )

            # Rolling window selector
            rolling_window = st.slider("Rolling Load Window (days)", 3, 14, 7)

            # Generate and display the figure
            fig_recovery_vs_load = plot_recovery_vs_load_peaks(
                gps_df=df_gps,
                recovery_df=df,
                recovery_metric=recovery_metric,
                rolling_window=rolling_window,
            )

            st.plotly_chart(fig_recovery_vs_load, use_container_width=True)


        # Match & Recovery 
        with tabs[1]:
            # Pre to post match recovery status : 
            st.subheader("Comparison of pre- vs post-match recovery scores for each match using a dumbbell chart")

            st.markdown(
                """
                This graph compares recovery scores before and after each match using a color-coded dumbbell chart. Each line connects the pre- and post-match values, with color indicating the impact level:
                🟥 High (>0.3 drop), 🟨 Moderate (0.1-0.3), 🟩 Low (<0.1).
                It pulls recovery scores 1 day before and after matches (±1 day tolerance), computes the difference, and visually tracks how much recovery declines post-game.
                A large drop from pre to post likely reflects intense physical effort and accumulated muscle stress.
                """
            )

            fig_pre_post_match_recovery = plot_pre_post_match_recovery_dumbbell(df_gps, df, recovery_metric="soreness_baseline_composite")
            st.plotly_chart(fig_pre_post_match_recovery, use_container_width=True)

            # markdown : 
            st.markdown(
                """
                **Impact Legend (based on drop in recovery score after match):**
                - 🟥 **High Impact**: Drop > 0.3 &nbsp; → &nbsp; **Red line** — intense effort or low post-match recovery  
                - 🟨 **Moderate Impact**: Drop between 0.1 and 0.3 &nbsp; → &nbsp; **Amber line** — manageable fatigue  
                - 🟩 **Low Impact**: Drop < 0.1 &nbsp; → &nbsp; **Green line** — good resilience or light match load
                """,
                unsafe_allow_html=True
            )
            
            # readiness before each match : 
            st.subheader("Readiness Snapshot Before Each Match")
            st.markdown("This graph shows a readiness score for each match, calculated as the average of the player’s benchmark performance score (latest benchmarkPct) and recovery score (from 1–2 days before the match). " \
            "Bars are colored based on thresholds: 🟥 low readiness, 🟨 moderate, and 🟩 high readiness. " \
            "It depicts how well-prepared the player was heading into each match.")

            fig_readiness_snapshot = plot_readiness_snapshot_before_matches(
                gps_df=df_gps,
                recovery_df=df,
                capability_df=df_physical,
                recovery_metric="soreness_baseline_composite",
                readiness_thresholds=(-0.1, 0.1),  
            )

            st.plotly_chart(fig_readiness_snapshot, use_container_width=True)

        # Category Breakdown
        with tabs[2]:
            st.header("Category Analysis")

            # Filter data by type
            completeness_df = date_filtered_df[
                date_filtered_df["metric_type"] == "completeness"
            ]
            composite_df = date_filtered_df[
                date_filtered_df["metric_type"] == "composite"
            ]

            # Two-column layout for category analysis
            col1, col2 = st.columns(2)

            with col1:
                # Category completeness bar chart
                st.subheader("Completeness by Category")
                st.markdown("""
                This bar chart shows the average data completeness across recovery categories such as sleep, soreness, subjective, and various musculoskeletal metrics. 
                Each bar represents the proportion of days where data was available for that category, giving a snapshot of data coverage over time. 
                It's useful for identifying which wellness domains are consistently tracked and which may need improved data collection practices—e.g. “subjective” has the highest completeness, while bio and msk_load_tolerance are less consistently recorded.""")
                if not completeness_df.empty:
                    fig = create_category_completeness_bar(completeness_df)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Category selector with composite scores
                st.subheader("Category Scores")
                st.markdown("""This graph displays composite recovery scores across multiple wellness categories (e.g. soreness, sleep, subjective, bio) over time. """)
                if not composite_df.empty:
                    fig = create_composite_line(composite_df)
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation analysis at the bottom
            if len(date_filtered_df) > 5:
                st.subheader("Metric Correlations")

                # Create pivot table for correlation
                pivot_df = date_filtered_df.pivot_table(
                    index="sessionDate", columns="metric", values="value"
                ).fillna(0)

                # Only include columns with sufficient variation
                valid_cols = [
                    col for col in pivot_df.columns if pivot_df[col].std() > 0
                ]

                if len(valid_cols) > 1:
                    corr_matrix = pivot_df[valid_cols].corr()
                    fig = create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader("Weekly recovery heatmap")
            st.markdown("""
            This heatmap visualizes how recovery scores vary by day and week. 
            Each row represents a week (e.g., 2024-W14) and each column a day of the week, with color indicating the average emboss_baseline_score for that day. 
            Green tones reflect higher recovery, while red indicates poor recovery.""")
            weekly_heatmap = plot_weekly_recovery_heatmap(df)
            st.plotly_chart(weekly_heatmap, use_container_width=True)
           
            # Risk analysis
            st.subheader("Weekly Risk Assessment")
            st.markdown("""
            This chart shows a weekly summary of training overload risk, based on how many days per week the player's load was among the top 10% (most demanding). 
            The function computes a danger threshold from GPS-derived load scores, then counts how many days per week exceed that threshold. This helps visualize how frequently the player hits risky load zones.
            """
            )
            
            fig_weekly_danger = plot_weekly_danger(merged_df)
            st.plotly_chart(fig_weekly_danger, use_container_width=True)

            # Simple risk legend
            st.markdown(
                """
            Risk Levels:
            - Green: Normal load
            - Yellow: Warning - monitor closely
            - Red: High risk - reduce load
            """
            )

    except FileNotFoundError:
        st.error("Recovery data file not found")
    except Exception as e:
        st.error(f"Error: {str(e)}")
